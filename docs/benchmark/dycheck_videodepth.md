# DyCheck VideoDepth

This integration expects the extracted **MegaSaM-distributed DyCheck evaluation
archive**, not an arbitrary export of the original DyCheck dataset. Each scene
must contain `dense/rgb/1x/0_*.png` and `depth/2x/0_*.npy`. Camera 0 is selected;
other cameras are ignored. The default scene list contains 12 scenes.

## Download

`scripts/download/dycheck.sh` fetches the 12 scene zips (pinned Google Drive file IDs) from the folder linked in
MegaSaM's README (https://github.com/mega-sam/mega-sam, "Running MegaSaM on DyCheck") with `gdown`, and unzips them
into `$DATA_ROOT_PATH/dycheck_raw/<scene>/`. On the cluster the raw data lives in
`benchmarks/VideoDepth_source/dycheck_raw`. `python -m spatialarena.download` runs the download and the
materialization below as one automated entry.

## Prepare the data

From the repository root:

```bash
python scripts/preprocess/materialize_dycheck.py \
  --source_root /path/to/extracted/dycheck \
  --benchmark_root /path/to/benchmark
```

`--scenes apple backpack` prepares a subset. Optional `--mask_subdir validity`
reads camera-0 `.npy` or `.png` masks relative to each source scene; mask filenames
must match RGB/depth stems and their dimensions must match depth. Masks are not
inferred from RGB or segmentation, and no synthetic mask is created by default.

The output is `<benchmark_root>/VideoDepth/dycheck/<scene>/images/`, `depths/`,
optional `masks/`, and `frame_manifest.json`, with dataset-level `metadata.json`.
Files are relative symbolic links: keep the extracted source data available.
The manifest records canonical paths and source-root-relative paths, never
personal absolute paths. It preserves all camera-0 frames in numeric order.
RGB and GT may have different resolutions; neither is resized or overwritten.
Depth arrays retain archive values (scale 1), without COLMAP pose-scale conversion.

Preparation checks pairing, duplicate stems, depth shape, aspect ratio and mask
shape before writing. Identical reruns are supported; a changed source or scene
selection must use a separate output root rather than replacing live artifacts.
This script does not download data or produce camera/SLAM ground truth.

## Camera priors (`rgb_cameras`)

VideoDepth `rgb_cameras` runs only on SpatialArena since 2026-11-14 (`TRACK_DATASETS` in
`spatialarena/task_track_matrix.py`): the released GT has no DyCheck cameras. If DyCheck is added back, the track
feeds each method the per-frame DyCheck cameras: `scripts/preprocess/
materialize_benchmark_camera_priors.py` follows each linked image back to `<scene>/dense/dycheck/camera/<frame>.json`
(Nerfies convention: `orientation` is world-to-camera, `position` the camera centre, OpenCV axes; intrinsics at
`image_size`, rescaled to the linked image) and writes a PINHOLE COLMAP model. The engine runs it on first use.

## Evaluate through YAML

DyCheck runs with the rest of VideoDepth in `configs/benchmark/full_videodepth.yaml`, for every registered VideoDepth
method; nothing about it needs a separate config. Model dependencies and checkpoints must already be installed. To
run DyCheck alone, restrict a runtime copy's `datasets` to `{task: VideoDepth, dataset: dycheck}` (and its `scenes`
if only a subset was prepared):

```bash
python -m spatialarena.run_config configs/benchmark/full_videodepth.yaml coverage
# Review the generated case lists, resource requests and scripts, then:
python -m spatialarena.run_config configs/benchmark/full_videodepth.yaml submit
```

Each scene gives up to 100 frames (the suite's cap), sampled with seed 42 (`BENCHMARK_MAX_INPUT_FRAMES_SEED`'s
default) and sorted chronologically. The canonical case interval is 1. This cap is a local benchmark setting, not the
dataset's sequence length. Each model is aligned in its declared output space (`depth_huber_irls_gt` for depth
methods such as DA3-L, `inverse_huber_irls_gt` for inverse-depth methods; see docs/benchmark/depth_methods.md).

Scoring restores predictions to native GT resolution with bilinear interpolation,
uses finite positive GT below **10 archive depth units** and any supplied validity
mask, and fits one affine transformation per selected sequence using Huber IRLS, in
the prediction's native output space (depth for DA3-L; inverse depth for
inverse-depth methods). The 10-unit cap is the existing FFB local indoor protocol, **not an
official DyCheck evaluation rule**. Predictions are not used to select valid GT
pixels. AbsRel, RMSE and delta1 are computed on that domain. The DIODE gradient
filter does not apply to DyCheck.

Before treating a run as complete, check expected scenes, prediction counts,
scored frames, skipped reasons and final reports; Slurm completion is insufficient.

Some upstream model registrations use site-specific absolute checkpoint defaults.
Coverage may reject these before applying per-model environment overrides. The
example uses DA3-L to avoid that existing validation limitation; no model registry
changes are part of this dataset integration.
