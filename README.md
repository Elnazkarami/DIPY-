# Streamline-wise FA with DIPY

A small, inspectable script that measures fractional anisotropy (FA) along an
existing tractogram.

## Scope

`tractometry.py`:

- **Takes** a preprocessed 4D diffusion image (NIfTI), its b-values and
  b-vectors (FSL format), and an existing `.trk` tractogram.
- **Fits** a diffusion tensor (DIPY `TensorModel`, weighted least squares) inside
  a brain mask and calculates FA.
- **Samples** FA at every point of every streamline and calculates each
  streamline's length.

It does **not**:

- generate tractography — the streamlines must already exist;
- perform any preprocessing — no motion, eddy-current or susceptibility
  distortion correction, denoising or registration. Run those beforehand
  (e.g. FSL `topup`/`eddy`) and pass the corrected data and the **rotated**
  b-vectors that go with it;
- produce an along-tract profile (see [Terminology](#terminology)).

### Where the tractogram comes from

The input must be a **streamline** file (`.trk`) whose coordinates refer to the
same anatomical space as the diffusion image. Typical sources are DIPY's own
tracking (saved with `save_tractogram`), or MRtrix3 `tckgen` / DSI Studio output
converted to `.trk` with the diffusion image as reference.

FSL `probtrackx2` writes, by default, voxel-wise visitation/density maps
(e.g. `fdt_paths.nii.gz`). Those are images, not streamlines, and **cannot be
loaded as a `.trk` file**. They are not a valid input to this script.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python tractometry.py --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec \
    --tractogram bundle.trk --out-dir output/ --overlay
```

| Option | Meaning |
| --- | --- |
| `--mask` | Binary brain mask on the DWI grid. Checked for shape, affine and 0/1 values. If omitted, `median_otsu` is run on the b0 volumes (b ≤ `--b0-threshold`), selected from the b-values. |
| `--b0-threshold` | b-values at or below this count as b0 (default 50). |
| `--interpolation` | `nearest` (default): FA of the voxel containing the point. `linear`: trilinear; a sample touching an invalid voxel is invalid. |
| `--overlay` | Save `overlay.png` (FA mid-slices with nearby streamline points). |
| `--skip-header-check` | Continue even if the `.trk` header grid does not match the DWI (recorded as a warning). |

### Outputs

| File | Contents |
| --- | --- |
| `streamline_metrics.csv` | `streamline_id`, `n_points`, `length_mm`, `mean_fa`, `n_valid_samples`, and counts of invalid samples (`n_out_of_bounds`, `n_outside_mask`, `n_nonfinite_fa`). `mean_fa` is blank when a streamline has no valid samples. |
| `summary.json` | Streamline count, length and FA summary statistics, both bundle-level FA summaries, invalid-sample counts, spatial checks, warnings, settings, input paths and software versions. |
| `fa.nii.gz` | FA map with the source image's affine and header. FA is 0 outside the mask; non-finite in-mask values are stored as 0 here but are counted and excluded from sampling. |
| `brain_mask.nii.gz` | The mask actually used for fitting and sampling. |
| `overlay.png` | Only with `--overlay`. |

## Terminology

**`mean_fa`** in `streamline_metrics.csv` is **one mean FA value per
streamline**: the average of the valid FA samples at that streamline's points.
It is not an along-tract profile.

A true along-tract profile would require orienting all streamlines
consistently, resampling each to a fixed number of points, and aggregating FA
at corresponding positions (e.g. DIPY's `afq_profile`). That is a possible
later extension.

**Bundle-level summaries** (`summary.json → bundle_fa`) answer different
questions:

- `mean_of_streamline_means` — average of the per-streamline means. Each
  streamline counts once, so voxels crossed by many streamlines dominate.
- `mean_over_unique_voxels` — average FA over the distinct voxels containing at
  least one valid sample. Each voxel counts once, regardless of streamline
  density; this describes the tissue the bundle occupies.

Streamline points are not equally spaced in every tractogram, so neither is
the same as an arc-length-weighted average.

## Spatial checks and invalid samples

Before sampling, the script:

1. Loads the streamlines into **RAS+ world coordinates in mm** (DIPY
   `Space.RASMM`, voxel-centre origin) and maps them into voxel coordinates
   with the inverse of the DWI affine. Lengths are the sum of segment lengths
   in these world coordinates, so they are in mm provided the DWI header's
   spatial unit is mm (recorded as `dwi_spatial_unit`).
2. Compares the `.trk` header (affine, dimensions, voxel size) with the DWI and
   **stops** if they differ, unless `--skip-header-check` is given.
3. Records whether any streamline point lies outside the image
   (`all_streamlines_within_image_bbox`) and the fraction of points outside the
   brain mask; more than 5% triggers a warning.

Invalid samples are **excluded and counted, never set to zero**, so they cannot
quietly lower an average:

| Case | Handling |
| --- | --- |
| Point outside the image | `n_out_of_bounds` |
| Point outside the brain mask | `n_outside_mask` |
| Non-finite FA at the point | `n_nonfinite_fa` |
| Empty streamline (0 points) | `n_points = 0`, length 0, `mean_fa` blank |

Lengths are geometric and include any part of a streamline that lies outside
the image.

**These checks do not establish anatomical alignment.** A matching header or
a shared reference image only shows the files describe the same voxel grid;
a tractogram generated from differently registered data can pass every check.
Inspect `overlay.png` (or load `fa.nii.gz` and the `.trk` together in a viewer)
to confirm the streamlines follow the expected white matter.

## Verification and validation status

**Software verification (synthetic data) — passing.** `tests/test_tractometry.py`
checks, on synthetic inputs with known answers, that:

- sampling a constant FA volume returns that constant (nearest and trilinear);
- world-to-voxel mapping uses the affine correctly (non-identity affine);
- streamlines with known physical coordinates have the expected lengths
  (30 mm straight line; 3-4-5 triangle);
- out-of-bounds, outside-mask, non-finite and empty inputs are counted and do
  not lower the mean;
- the two bundle-level summaries differ as documented;
- mismatched gradients, a mask on the wrong grid and a non-binary mask raise
  clear errors;
- the full command-line workflow on a noise-free single-tensor phantom recovers
  the analytical FA (0.799) and writes all outputs, with both a supplied mask
  and an automatic `median_otsu` mask.

Run them with:

```bash
python -m pytest
```

Last run: 15 passed (Python 3.12.6, DIPY 1.12.1, macOS), 2026-10-13.

**Real diffusion data — not yet recorded.** This revision has not been run in
this repository on real diffusion MRI. When it is, record the dataset (or its
public identifier), acquisition (b-values, number of directions), preprocessing,
tractogram source, and attach `summary.json` and `overlay.png`.

## Data availability

No imaging data are included in this repository. The research dataset used
during development is not distributed here. All test inputs are generated
synthetically by the test suite. For a public example, DIPY's
`dipy.data.get_fnames()` provides downloadable diffusion datasets.

## Contribution

Original script and repository: Elnaz Karami. This revision restructures the
script into a command-line tool with input checks, explicit masking and gradient
handling, file outputs, documented invalid-sample handling, and a synthetic
verification test suite.

## Dependencies

See `requirements.txt` (tested versions). Requires DIPY ≥ 1.10.
