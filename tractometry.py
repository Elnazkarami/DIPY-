"""Streamline-wise FA measurements from preprocessed diffusion MRI and a .trk tractogram.

The script fits a diffusion tensor (DIPY), computes fractional anisotropy (FA),
samples FA at every point of every streamline, and writes per-streamline
measurements, a summary with processing metadata, and the FA map.

It does NOT generate tractography and does NOT perform any preprocessing
(motion, eddy-current or susceptibility distortion correction). See README.md.

Tested with DIPY 1.12.1 (requires DIPY >= 1.10 for keyword-only
``gradient_table(bvals, bvecs=...)``).
"""

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import dipy
import nibabel as nib
import numpy as np
import scipy
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.streamline import load_tractogram
from dipy.io.utils import is_header_compatible
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu
from dipy.tracking.streamline import length as streamline_length
from nibabel.affines import apply_affine
from scipy.ndimage import map_coordinates

MIN_DWI_DIRECTIONS = 6  # minimum number of diffusion-weighted volumes for a tensor fit


# ---------------------------------------------------------------------------
# Gradients and mask
# ---------------------------------------------------------------------------

def load_gradients(bval_path, bvec_path, n_volumes, b0_threshold):
    """Read b-values/b-vectors explicitly and check them against the DWI."""
    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    if bvals.shape[0] != n_volumes:
        raise ValueError(
            f"{bvals.shape[0]} b-values but the diffusion image has {n_volumes} volumes."
        )
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold)
    n_dwi = int((~gtab.b0s_mask).sum())
    if n_dwi < MIN_DWI_DIRECTIONS:
        raise ValueError(
            f"Only {n_dwi} diffusion-weighted volumes (b > {b0_threshold}); "
            f"at least {MIN_DWI_DIRECTIONS} are needed to fit a tensor."
        )
    return gtab


def b0_volume_indices(gtab):
    """Indices of the b0 volumes (b <= b0_threshold), used for brain masking."""
    idx = np.flatnonzero(gtab.b0s_mask)
    if idx.size == 0:
        raise ValueError(
            "No b0 volumes found (b <= b0_threshold). Supply a brain mask with --mask "
            "or adjust --b0-threshold."
        )
    return idx


def load_and_check_mask(mask_path, dwi_img):
    """Load a user-supplied brain mask and check it matches the DWI grid."""
    mask_img = nib.load(str(mask_path))
    if mask_img.shape[:3] != dwi_img.shape[:3] or mask_img.ndim != 3:
        raise ValueError(
            f"Mask shape {mask_img.shape} does not match DWI spatial shape {dwi_img.shape[:3]}."
        )
    if not np.allclose(mask_img.affine, dwi_img.affine, atol=1e-3):
        raise ValueError("Mask affine does not match the DWI affine.")
    values = np.asarray(mask_img.dataobj)
    unique = np.unique(values)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"Mask must be binary (0/1); found values {unique[:10]}.")
    brain_mask = values.astype(bool)
    if not brain_mask.any():
        raise ValueError("Mask is empty.")
    return brain_mask


# ---------------------------------------------------------------------------
# Sampling along streamlines
# ---------------------------------------------------------------------------

def sample_volume_along_streamlines(volume, streamlines, affine, valid_mask=None,
                                    interpolation="nearest"):
    """Sample a 3D volume at every point of every streamline.

    Parameters
    ----------
    volume : (X, Y, Z) array
    streamlines : sequence of (N, 3) arrays in world (RAS+ mm) coordinates
    affine : (4, 4) voxel-to-world affine of ``volume``
    valid_mask : (X, Y, Z) bool array, optional
        Points whose nearest voxel is outside this mask are counted as invalid.
    interpolation : "nearest" or "linear"
        "linear" is trilinear; any non-finite neighbour makes the sample invalid.

    Returns
    -------
    list of dict, one per streamline, with keys
        values         : (N,) float array, NaN where the sample is invalid
        voxels         : (M, 3) int array of nearest-voxel indices of valid samples
        n_points, n_valid, n_out_of_bounds, n_outside_mask, n_nonfinite
    Invalid samples are reported, never replaced by zero.
    """
    shape = np.array(volume.shape[:3])
    inv_affine = np.linalg.inv(affine)
    results = []
    for sl in streamlines:
        pts = np.asarray(sl, dtype=float).reshape(-1, 3)
        n = pts.shape[0]
        values = np.full(n, np.nan)
        rec = dict(values=values, voxels=np.empty((0, 3), dtype=int), n_points=n,
                   n_valid=0, n_out_of_bounds=0, n_outside_mask=0, n_nonfinite=0)
        if n == 0:
            results.append(rec)
            continue

        vox = apply_affine(inv_affine, pts)  # integer coordinates = voxel centres
        nearest = np.rint(vox).astype(int)
        in_bounds = np.all((nearest >= 0) & (nearest < shape), axis=1)
        rec["n_out_of_bounds"] = int((~in_bounds).sum())

        usable = in_bounds.copy()
        if valid_mask is not None:
            ib = np.flatnonzero(in_bounds)
            inside = valid_mask[tuple(nearest[ib].T)]
            usable[ib[~inside]] = False
            rec["n_outside_mask"] = int((~inside).sum())

        idx = np.flatnonzero(usable)
        if idx.size:
            if interpolation == "nearest":
                sampled = volume[tuple(nearest[idx].T)].astype(float)
            elif interpolation == "linear":
                # NaN neighbours propagate, so a sample touching invalid data is invalid.
                sampled = map_coordinates(volume.astype(float), vox[idx].T, order=1,
                                          mode="nearest", prefilter=False)
            else:
                raise ValueError(f"Unknown interpolation {interpolation!r}")
            finite = np.isfinite(sampled)
            rec["n_nonfinite"] = int((~finite).sum())
            values[idx[finite]] = sampled[finite]
            rec["voxels"] = nearest[idx[finite]]
        rec["n_valid"] = int(np.isfinite(values).sum())
        results.append(rec)
    return results


def streamline_lengths_mm(streamlines):
    """Arc length of each streamline (sum of segment lengths) in world units."""
    if len(streamlines) == 0:
        return np.empty(0)
    return np.atleast_1d(np.asarray(streamline_length(streamlines), dtype=float))


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=None, sd=None, median=None, min=None, max=None)
    return dict(n=int(x.size), mean=float(x.mean()),
                sd=float(x.std(ddof=1)) if x.size > 1 else None,
                median=float(np.median(x)), min=float(x.min()), max=float(x.max()))


def bundle_summary(samples, lengths, volume):
    """Bundle-level summaries. The two FA summaries answer different questions:

    * mean_of_streamline_means: each streamline counts equally; regions crossed by
      many streamlines dominate.
    * mean_over_unique_voxels: each voxel touched by >= 1 valid sample counts once;
      describes the tissue occupied by the bundle, regardless of streamline density.
    """
    mean_fa = np.array([np.nanmean(s["values"]) if s["n_valid"] else np.nan for s in samples])
    voxels = [s["voxels"] for s in samples if len(s["voxels"])]
    if voxels:
        unique_vox = np.unique(np.vstack(voxels), axis=0)
        voxel_fa = volume[tuple(unique_vox.T)]
    else:
        unique_vox, voxel_fa = np.empty((0, 3)), np.empty(0)
    n_points = sum(s["n_points"] for s in samples)
    return dict(
        n_streamlines=len(samples),
        n_empty_streamlines=sum(s["n_points"] == 0 for s in samples),
        n_streamlines_without_valid_fa=int(np.sum(~np.isfinite(mean_fa))),
        samples=dict(
            n_points=int(n_points),
            n_valid=int(sum(s["n_valid"] for s in samples)),
            n_out_of_bounds=int(sum(s["n_out_of_bounds"] for s in samples)),
            n_outside_mask=int(sum(s["n_outside_mask"] for s in samples)),
            n_nonfinite_fa=int(sum(s["n_nonfinite"] for s in samples)),
        ),
        length_mm=_stats(lengths),
        mean_fa_per_streamline=_stats(mean_fa),
        bundle_fa=dict(
            mean_of_streamline_means=_stats(mean_fa)["mean"],
            mean_over_unique_voxels=_stats(voxel_fa)["mean"],
            n_unique_voxels=int(len(unique_vox)),
        ),
    ), mean_fa


# ---------------------------------------------------------------------------
# Spatial checks
# ---------------------------------------------------------------------------

def spatial_checks(dwi_img, tractogram_path, sft):
    """Checks recorded in summary.json. None of these proves anatomical alignment;
    inspect the overlay (--overlay) to judge that."""
    spatial_unit = dwi_img.header.get_xyzt_units()[0]
    checks = dict(
        streamline_space="RAS+ world coordinates (DIPY Space.RASMM, origin at voxel centre)",
        dwi_spatial_unit=spatial_unit,
        dwi_voxel_size=[float(v) for v in dwi_img.header.get_zooms()[:3]],
        length_unit="mm" if spatial_unit in ("mm", "unknown") else spatial_unit,
        all_streamlines_within_image_bbox=bool(sft.is_bbox_in_vox_valid()),
    )
    if Path(tractogram_path).suffix.lower() == ".trk":
        checks["trk_header_matches_dwi"] = bool(is_header_compatible(str(tractogram_path), dwi_img))
    return checks


def save_overlay(path, fa, streamlines, affine, max_streamlines=2000):
    """Mid-slice FA images with streamline points overlaid, for a visual alignment check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    pick = rng.permutation(len(streamlines))[:max_streamlines]
    pts = [np.asarray(streamlines[i]) for i in pick if len(streamlines[i])]
    vox = apply_affine(np.linalg.inv(affine), np.vstack(pts)) if pts else np.empty((0, 3))
    centre = np.array(fa.shape) // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, axis, (a, b), name in zip(axes, range(3), [(1, 2), (0, 2), (0, 1)],
                                      ["sagittal", "coronal", "axial"]):
        ax.imshow(np.take(fa, centre[axis], axis=axis).T, origin="lower", cmap="gray",
                  vmin=0, vmax=1)
        near = np.abs(vox[:, axis] - centre[axis]) < 2
        ax.scatter(vox[near, a], vox[near, b], s=0.3, c="tab:red", alpha=0.5)
        ax.set_title(f"{name} slice {centre[axis]} (voxel coords)")
        ax.axis("off")
    fig.suptitle("FA with streamline points within 2 voxels of the slice")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dwi", required=True, help="Preprocessed 4D diffusion image (.nii/.nii.gz)")
    p.add_argument("--bval", required=True, help="b-values file (FSL format)")
    p.add_argument("--bvec", required=True, help="b-vectors file (FSL format)")
    p.add_argument("--tractogram", required=True, help="Streamlines (.trk) aligned with --dwi")
    p.add_argument("--out-dir", required=True, help="Output directory (created if missing)")
    p.add_argument("--mask", help="Binary brain mask on the DWI grid. If omitted, a mask is "
                                  "computed with median_otsu on the b0 volumes.")
    p.add_argument("--b0-threshold", type=float, default=50,
                   help="b-values <= this are treated as b0 (default: 50)")
    p.add_argument("--median-radius", type=int, default=4, help="median_otsu radius (default: 4)")
    p.add_argument("--numpass", type=int, default=4, help="median_otsu passes (default: 4)")
    p.add_argument("--interpolation", choices=["nearest", "linear"], default="nearest",
                   help="FA sampling at streamline points (default: nearest)")
    p.add_argument("--skip-header-check", action="store_true",
                   help="Proceed even if the .trk header does not match the DWI grid")
    p.add_argument("--overlay", action="store_true",
                   help="Save overlay.png (FA + streamlines) for a visual alignment check")
    return p.parse_args(argv)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    # Diffusion data and gradients
    dwi_img = nib.load(args.dwi)
    if dwi_img.ndim != 4:
        raise ValueError(f"Expected a 4D diffusion image, got shape {dwi_img.shape}.")
    data = dwi_img.get_fdata(dtype=np.float32)
    affine = dwi_img.affine
    gtab = load_gradients(args.bval, args.bvec, data.shape[3], args.b0_threshold)

    # Brain mask
    if args.mask:
        brain_mask = load_and_check_mask(args.mask, dwi_img)
        masked_data = data * brain_mask[..., None]
        mask_info = dict(source="supplied", path=str(args.mask))
    else:
        b0_idx = b0_volume_indices(gtab)
        masked_data, brain_mask = median_otsu(data, vol_idx=b0_idx,
                                              median_radius=args.median_radius,
                                              numpass=args.numpass)
        mask_info = dict(source="median_otsu", b0_volume_indices=b0_idx.tolist(),
                         median_radius=args.median_radius, numpass=args.numpass)
    mask_info["n_voxels"] = int(brain_mask.sum())

    # Tensor fit and FA (zero outside the mask by construction)
    ten_model = TensorModel(gtab)
    ten_fit = ten_model.fit(masked_data, mask=brain_mask)
    fa = ten_fit.fa
    n_nonfinite_fa = int((~np.isfinite(fa[brain_mask])).sum())
    if n_nonfinite_fa:
        warnings.append(f"{n_nonfinite_fa} in-mask voxels have non-finite FA; "
                        "they are excluded from sampling.")

    # Sampling volume: invalid voxels are NaN so they cannot lower any average.
    fa_for_sampling = np.where(brain_mask & np.isfinite(fa), fa, np.nan)

    # FA map with the source image's spatial information (non-finite stored as 0)
    fa_hdr = dwi_img.header.copy()
    fa_hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(np.nan_to_num(fa, nan=0.0).astype(np.float32), affine, fa_hdr),
             out_dir / "fa.nii.gz")
    nib.save(nib.Nifti1Image(brain_mask.astype(np.uint8), affine, dwi_img.header.copy()),
             out_dir / "brain_mask.nii.gz")

    # Tractogram
    if Path(args.tractogram).suffix.lower() == ".trk" and not is_header_compatible(
            args.tractogram, dwi_img):
        msg = ("The .trk header (affine/dimensions/voxel size) does not match the DWI. "
               "The tractogram may have been produced on a different grid.")
        if not args.skip_header_check:
            raise ValueError(msg + " Use --skip-header-check to proceed anyway.")
        warnings.append(msg)
    sft = load_tractogram(args.tractogram, dwi_img, bbox_valid_check=False,
                          trk_header_check=False)
    sft.to_rasmm()
    sft.to_center()
    streamlines = sft.streamlines
    checks = spatial_checks(dwi_img, args.tractogram, sft)
    if not checks["all_streamlines_within_image_bbox"]:
        warnings.append("Some streamline points lie outside the image; they are counted "
                        "as out_of_bounds and excluded from FA averages.")

    # Measurements
    lengths = streamline_lengths_mm(streamlines)
    samples = sample_volume_along_streamlines(fa_for_sampling, streamlines, affine,
                                              valid_mask=brain_mask,
                                              interpolation=args.interpolation)
    summary, mean_fa_per_streamline = bundle_summary(samples, lengths, fa_for_sampling)
    if summary["samples"]["n_points"]:
        frac = summary["samples"]["n_outside_mask"] / summary["samples"]["n_points"]
        checks["fraction_points_outside_brain_mask"] = frac
        if frac > 0.05:
            warnings.append(f"{frac:.1%} of streamline points fall outside the brain mask; "
                            "check anatomical alignment with --overlay.")

    with open(out_dir / "streamline_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["streamline_id", "n_points", "length_mm", "mean_fa", "n_valid_samples",
                    "n_out_of_bounds", "n_outside_mask", "n_nonfinite_fa"])
        for i, (s, L, m) in enumerate(zip(samples, lengths, mean_fa_per_streamline)):
            w.writerow([i, s["n_points"], f"{L:.4f}", "" if np.isnan(m) else f"{m:.6f}",
                        s["n_valid"], s["n_out_of_bounds"], s["n_outside_mask"],
                        s["n_nonfinite"]])

    if args.overlay:
        save_overlay(out_dir / "overlay.png", np.nan_to_num(fa), streamlines, affine)

    summary.update(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        inputs=dict(dwi=str(args.dwi), bval=str(args.bval), bvec=str(args.bvec),
                    tractogram=str(args.tractogram), dwi_shape=list(dwi_img.shape)),
        settings=dict(b0_threshold=args.b0_threshold, n_b0=int(gtab.b0s_mask.sum()),
                      n_dwi=int((~gtab.b0s_mask).sum()),
                      bvalues=sorted(set(np.round(gtab.bvals).astype(int).tolist())),
                      brain_mask=mask_info, tensor_fit_method=ten_model.fit_method.__name__,
                      interpolation=args.interpolation,
                      n_nonfinite_fa_in_mask=n_nonfinite_fa,
                      invalid_sample_handling="excluded and counted; never set to zero"),
        spatial_checks=checks,
        warnings=warnings,
        software=dict(python=platform.python_version(), dipy=dipy.__version__,
                      nibabel=nib.__version__, numpy=np.__version__, scipy=scipy.__version__),
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for w_ in warnings:
        print("WARNING:", w_, file=sys.stderr)
    print(f"Wrote outputs to {out_dir}")
    return summary


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
