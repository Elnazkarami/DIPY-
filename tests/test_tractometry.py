"""Software verification on synthetic data.

These tests check the code's arithmetic and error handling with inputs whose
correct answers are known. They are not a validation on real diffusion MRI.
"""

import csv
import json

import nibabel as nib
import numpy as np
import pytest
from dipy.data import get_sphere
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.sims.voxel import single_tensor

import tractometry as tm

# 2 mm isotropic voxels with a non-zero origin, so voxel and world coordinates differ.
AFFINE = np.array([[2.0, 0, 0, -20], [0, 2.0, 0, -30], [0, 0, 2.0, -10], [0, 0, 0, 1]])
SHAPE = (20, 20, 20)


def world(voxel_coords):
    return nib.affines.apply_affine(AFFINE, np.asarray(voxel_coords, dtype=float))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
def test_constant_volume_returns_constant(interpolation):
    vol = np.full(SHAPE, 0.42)
    sl = world([[5, 5, 5], [6.3, 5.5, 5], [8, 7.7, 6.1]])
    (rec,) = tm.sample_volume_along_streamlines(vol, [sl], AFFINE, interpolation=interpolation)
    np.testing.assert_allclose(rec["values"], 0.42)
    assert rec["n_valid"] == 3 and rec["n_out_of_bounds"] == 0


def test_nearest_sampling_uses_world_to_voxel_transform():
    vol = np.zeros(SHAPE)
    vol[3, 4, 5] = 0.9
    (rec,) = tm.sample_volume_along_streamlines(vol, [world([[3, 4, 5]])], AFFINE)
    assert rec["values"][0] == pytest.approx(0.9)


def test_out_of_bounds_points_are_reported_not_zeroed():
    vol = np.full(SHAPE, 0.5)
    sl = world([[5, 5, 5], [10, 5, 5], [25, 5, 5], [-3, 5, 5]])  # last two outside
    (rec,) = tm.sample_volume_along_streamlines(vol, [sl], AFFINE)
    assert rec["n_out_of_bounds"] == 2
    assert rec["n_valid"] == 2
    assert np.isnan(rec["values"][2:]).all()
    assert np.nanmean(rec["values"]) == pytest.approx(0.5)  # not pulled toward 0


def test_streamline_entirely_outside_has_no_valid_samples():
    vol = np.full(SHAPE, 0.5)
    (rec,) = tm.sample_volume_along_streamlines(vol, [world([[50, 50, 50], [60, 50, 50]])],
                                                AFFINE)
    assert rec["n_valid"] == 0 and rec["n_out_of_bounds"] == 2


def test_mask_and_nonfinite_samples_are_counted():
    vol = np.full(SHAPE, 0.5)
    vol[7, 5, 5] = np.nan
    mask = np.ones(SHAPE, bool)
    mask[9, 5, 5] = False
    sl = world([[5, 5, 5], [7, 5, 5], [9, 5, 5]])
    (rec,) = tm.sample_volume_along_streamlines(vol, [sl], AFFINE, valid_mask=mask)
    assert rec["n_nonfinite"] == 1 and rec["n_outside_mask"] == 1 and rec["n_valid"] == 1


def test_empty_streamline():
    (rec,) = tm.sample_volume_along_streamlines(np.ones(SHAPE), [np.empty((0, 3))], AFFINE)
    assert rec["n_points"] == 0 and rec["n_valid"] == 0


# ---------------------------------------------------------------------------
# Lengths and summaries
# ---------------------------------------------------------------------------

def test_streamline_lengths_in_mm():
    straight = np.array([[0, 0, 0], [10, 0, 0], [30, 0, 0]], float)  # 30 mm
    diagonal = np.array([[1, 1, 1], [4, 5, 1]], float)  # 3-4-5 triangle: 5 mm
    np.testing.assert_allclose(tm.streamline_lengths_mm([straight, diagonal]), [30, 5])


def test_bundle_summaries_differ_as_documented():
    vol = np.zeros(SHAPE)
    vol[5, 5, 5], vol[6, 5, 5] = 0.2, 0.8
    # Three streamlines in the 0.2 voxel, one in the 0.8 voxel.
    sls = [world([[5, 5, 5]])] * 3 + [world([[6, 5, 5]])]
    samples = tm.sample_volume_along_streamlines(vol, sls, AFFINE)
    summary, _ = tm.bundle_summary(samples, tm.streamline_lengths_mm(sls), vol)
    assert summary["bundle_fa"]["mean_of_streamline_means"] == pytest.approx(0.35)
    assert summary["bundle_fa"]["mean_over_unique_voxels"] == pytest.approx(0.5)
    assert summary["bundle_fa"]["n_unique_voxels"] == 2


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def _gradients(n_dwi=30, n_b0=2):
    bvecs = np.vstack([np.zeros((n_b0, 3)), get_sphere(name="repulsion100").vertices[:n_dwi]])
    bvals = np.r_[np.zeros(n_b0), np.full(n_dwi, 1000.0)]
    return bvals, bvecs


def _write_gradients(tmp_path, bvals, bvecs):
    np.savetxt(tmp_path / "dwi.bval", bvals[None], fmt="%g")
    np.savetxt(tmp_path / "dwi.bvec", bvecs.T, fmt="%.6f")
    return tmp_path / "dwi.bval", tmp_path / "dwi.bvec"


def test_gradient_count_mismatch_raises(tmp_path):
    bval, bvec = _write_gradients(tmp_path, *_gradients())
    with pytest.raises(ValueError, match="b-values"):
        tm.load_gradients(bval, bvec, n_volumes=10, b0_threshold=50)


def test_b0_indices_selected_from_gradients(tmp_path):
    bvals, bvecs = _gradients()
    bvals, bvecs = np.r_[1000, bvals], np.vstack([bvecs[2], bvecs])  # b0s at 1, 2
    bval, bvec = _write_gradients(tmp_path, bvals, bvecs)
    gtab = tm.load_gradients(bval, bvec, len(bvals), 50)
    assert tm.b0_volume_indices(gtab).tolist() == [1, 2]


def test_mask_on_wrong_grid_raises(tmp_path):
    dwi = nib.Nifti1Image(np.zeros(SHAPE + (3,), np.float32), AFFINE)
    nib.save(nib.Nifti1Image(np.ones(SHAPE, np.uint8), np.eye(4)), tmp_path / "m.nii.gz")
    with pytest.raises(ValueError, match="affine"):
        tm.load_and_check_mask(tmp_path / "m.nii.gz", dwi)


def test_non_binary_mask_raises(tmp_path):
    dwi = nib.Nifti1Image(np.zeros(SHAPE + (3,), np.float32), AFFINE)
    nib.save(nib.Nifti1Image(np.full(SHAPE, 2, np.uint8), AFFINE), tmp_path / "m.nii.gz")
    with pytest.raises(ValueError, match="binary"):
        tm.load_and_check_mask(tmp_path / "m.nii.gz", dwi)


# ---------------------------------------------------------------------------
# End-to-end on a synthetic tensor phantom
# ---------------------------------------------------------------------------

EVALS = np.array([1.7e-3, 0.3e-3, 0.3e-3])


def expected_fa(ev):
    return np.sqrt(0.5 * ((ev[0] - ev[1]) ** 2 + (ev[1] - ev[2]) ** 2 + (ev[2] - ev[0]) ** 2)
                   / (ev ** 2).sum())


@pytest.fixture
def phantom(tmp_path):
    """A 20^3 volume whose central 12^3 block holds one noise-free anisotropic tensor."""
    bvals, bvecs = _gradients()
    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvals, bvecs=bvecs)
    signal = single_tensor(gtab, S0=1000, evals=EVALS, evecs=np.eye(3))
    data = np.zeros(SHAPE + (len(bvals),), np.float32)
    data[4:16, 4:16, 4:16] = signal
    dwi = nib.Nifti1Image(data, AFFINE)
    dwi.header.set_xyzt_units("mm")
    nib.save(dwi, tmp_path / "dwi.nii.gz")
    bval, bvec = _write_gradients(tmp_path, bvals, bvecs)

    sls = [world([[6, 10, 10], [10, 10, 10], [13, 10, 10]]),  # 14 mm, inside
           world([[10, 6, 8], [10, 12, 8]]),                   # 12 mm, inside
           world([[10, 10, 10], [10, 10, 25]])]                # leaves the image
    sft = StatefulTractogram(sls, dwi, Space.RASMM)
    save_tractogram(sft, str(tmp_path / "bundle.trk"), bbox_valid_check=False)
    return dict(dwi=tmp_path / "dwi.nii.gz", bval=bval, bvec=bvec,
                trk=tmp_path / "bundle.trk", out=tmp_path / "out")


@pytest.mark.parametrize("use_mask", [False, True])
def test_end_to_end(phantom, tmp_path, use_mask):
    argv = ["--dwi", str(phantom["dwi"]), "--bval", str(phantom["bval"]),
            "--bvec", str(phantom["bvec"]), "--tractogram", str(phantom["trk"]),
            "--out-dir", str(phantom["out"]), "--median-radius", "1", "--numpass", "1",
            "--overlay"]
    if use_mask:
        mask = np.zeros(SHAPE, np.uint8)
        mask[4:16, 4:16, 4:16] = 1
        nib.save(nib.Nifti1Image(mask, AFFINE), tmp_path / "mask.nii.gz")
        argv += ["--mask", str(tmp_path / "mask.nii.gz")]
    tm.main(argv)

    out = phantom["out"]
    for name in ["streamline_metrics.csv", "summary.json", "fa.nii.gz", "brain_mask.nii.gz",
                 "overlay.png"]:
        assert (out / name).exists(), name

    fa_img = nib.load(out / "fa.nii.gz")
    np.testing.assert_allclose(fa_img.affine, AFFINE)
    assert fa_img.shape == SHAPE
    assert fa_img.get_fdata()[10, 10, 10] == pytest.approx(expected_fa(EVALS), abs=1e-3)

    rows = list(csv.DictReader(open(out / "streamline_metrics.csv")))
    assert len(rows) == 3
    assert float(rows[0]["length_mm"]) == pytest.approx(14)
    assert float(rows[1]["length_mm"]) == pytest.approx(12)
    for r in rows[:2]:
        assert float(r["mean_fa"]) == pytest.approx(expected_fa(EVALS), abs=1e-3)
    # Two-point streamline: the second point is outside the image.
    assert rows[2]["n_out_of_bounds"] == "1" and rows[2]["n_valid_samples"] == "1"

    summary = json.load(open(out / "summary.json"))
    assert summary["n_streamlines"] == 3
    assert summary["spatial_checks"]["all_streamlines_within_image_bbox"] is False
    assert summary["spatial_checks"]["trk_header_matches_dwi"] is True
    assert summary["settings"]["brain_mask"]["source"] == ("supplied" if use_mask
                                                           else "median_otsu")
    assert summary["software"]["dipy"]
