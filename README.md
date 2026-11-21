# SpatialArena

SpatialArena is the benchmark for geometric foundation models: evaluation, model adapters, benchmark
dataset preparation, reports and the results dashboard. It was split out of FeedForward-Benchmark, which keeps the
training-data side (dataset downloads, WebDataset conversion, dataloaders, MindCloud processing).

SpatialArena is an artifact-only benchmark suite. External methods are
responsible for reading canonical benchmark images and writing standardized
prediction artifacts. The evaluators then read only those artifacts plus
canonical ground truth; they do not import or instantiate DA3, VGGT, AMB3R, or
other external models.

The current full-standard report scope intentionally excludes ScanNet. See
`docs/evaluation.md` for the evaluation datasets, methods, tracks, protocol, metric
definitions, case lists and output layout.

## Install

```sh
cd SpatialArena
pip install -e .            # or: pip install -r requirements.txt (the pinned versions)
pip install -e .[test]      # plus pytest and jinja2, for `pytest test/`
```

Install PyTorch (`torch`, `torchvision`) for your CUDA version, or the CPU build, from https://pytorch.org before
this, so pip does not pick a default build.

```python
import spatialarena
```

### Data root

Every default path is under one folder, `SPATIALARENA_ROOT` (default `~/spatialarena`, `spatialarena/paths.py`):

```text
$SPATIALARENA_ROOT/benchmarks/                benchmark datasets (<Task>/<dataset>/): the evaluator's benchmark_root
$SPATIALARENA_ROOT/benchmarks/_evaluation/    evaluation outputs (reports/, artifacts/, viewer_assets/)
$SPATIALARENA_ROOT/benchmarks/_model_codes/   the methods' source checkouts
$SPATIALARENA_ROOT/benchmarks/_checkpoints/   model checkpoints
```

```sh
export SPATIALARENA_ROOT=/data/spatialarena   # e.g. where the Hugging Face release was downloaded to benchmarks/
```

Run configs write paths as `${SPATIALARENA_ROOT}/...`, and generated Slurm scripts record the root they were generated
with. `BENCHMARK_AUX_ROOT`, `BENCHMARK_EVAL_ROOT`, `METHOD_CODE_ROOT` and `CHECKPOINT_ROOT` still override one folder
each.

## License

SpatialArena is source-available for noncommercial use under the
PolyForm Noncommercial License 1.0.0.

Copyright (c) 2026 Yu Chen.

Commercial use requires separate written authorization from Yu Chen. Third-party
datasets, benchmarks, and model artifacts referenced by this repository remain
governed by their own licenses and access terms.

## Canonical Dataset Inputs

Datasets are organized under a benchmark root:

```text
BENCHMARK_ROOT/
  TASK/
    DATASET/
      metadata.json
      SCENE/
        images/
        depths/
        masks/
        sparse/0/
        materialized_full.json
```

Only the folders required by a task need to exist:

| Task | Current datasets | Required canonical inputs |
| --- | --- | --- |
| MonoDepth | Marigold subsets (NYUv2, KITTI, ETH3D, DIODE, ScanNet), AerialMetric, HAMMER, iBims-1, Spring, Synth4K | `images/`, `depths/`; optional `masks/`; `materialized_full.json` for materialized subsets. |
| VideoDepth | Sintel, Bonn, KITTI | `images/`, `depths/`. Bonn is paired by filename or nearest timestamp. |
| SfM | ETH3D, Tanks-and-Temples, IMC, DTU, ScanNet++, HiRoom, TexturePoorSfM | `images/` and GT COLMAP models. ETH3D can use calibration folders, and DTU/Tanks-and-Temples use local COLMAP-backed 3DGS copies. (`CameraPose` is accepted as an alias of `SfM`.) |
| SLAM | 7Scenes (18 test sequences), TUM RGB-D (dynamic objects), Oxford Spires, VBR, KITTI Odometry (00-10) | One continuous recording per case with its own GT COLMAP model; whole sequences in range tracks (short: stride 10, long: stride 3; large: Oxford Spires, VBR and KITTI Odometry), each with the methods that fit its longest sequence in one H200 pass (`full-slam` suite, see `docs/benchmark/slam_protocol.md`). 7Scenes is evaluated only here, not in SfM. |
| MVRecon | 7Scenes, ETH3D, Tanks-and-Temples, DTU, ScanNet++, HiRoom, NRGBD | `images/`, `depths/`, GT COLMAP models, and dataset-specific geometry ground truth described in `docs/evaluation.md`. |

Current benchmark data inventory is generated in `docs/evaluation.md` (Scene Inventory);
training datasets are documented in FeedForward-Benchmark's `docs/training_data.md`.

## External Method Outputs

Model outputs should be written under:

```text
EXPERIMENT_ROOT/EXPERIMENT_NAME/TASK/DATASET/METHOD/SCENE/
  depths/
    IMAGE_STEM.npy
    IMAGE_STEM.pfm
  sparse/0/
    cameras.bin or cameras.txt
    images.bin or images.txt
    points3D.bin or points3D.txt
  points3D_dense.ply
  metrics.json
```

Depth tasks consume `depths/`. SfM/Pose consumes `sparse/0`. Mesh evaluators
consume dataset-specific mesh or point artifacts. Methods may write additional
logs, but final evaluators only require the standardized artifacts above.

## Running Benchmark Adapters

Method scripts under `scripts/inference/` are adapters for producing test
artifacts. They are not used by the evaluators themselves. The supported
methods, their aliases, tracks and input-resolution profiles are listed in
`docs/evaluation.md` ("Methods and Tracks"); print them with:

```sh
python scripts/test/benchmark_methods.py --list
```

The recommended entry point is the Python API (`spatialarena.BenchmarkEvaluator`):
it plans the (method, task, track, dataset) runs, generates the Slurm scripts and
shows progress. See `docs/benchmark/evaluator_quickstart.md`.

The generated scripts run the case-list engine in
`spatialarena/templates/_engine/`, which can also be submitted directly. Each run
is an inference job (GPU) followed by an evaluation job (CPU) on the same method, suite and track:

```sh
ENGINE=spatialarena/templates/_engine
JOB=$(sbatch --parsable --export=NIL $ENGINE/run_method_benchmark_infer.sh VGGT full-depth rgb)
sbatch --export=NIL --dependency=afterok:$JOB $ENGINE/run_method_benchmark_eval.sh VGGT full-depth rgb

# SLAM runs one range track at a time (short = stride 10, long = stride 3):
sbatch --export=NIL,SLAM_RANGE=long $ENGINE/run_method_benchmark_infer.sh PI3_X full-slam rgb
```

To submit many methods at once, `scripts/test/submit_method_case_jobs.sh <suite> <METHOD>...` submits the same job
pair per run through `scripts/test/engine_pair.sh`; `python -m spatialarena.run_config <config> submit` does it for a
whole run config.

The suite (2nd argument) can be `sampled`, `all-task`, `multiscene`, `monodepth`,
`sparse`, `full-depth`, `full-pose`, or `full-slam`; each selects its default case
list under `scripts/test/` (override with `CASE_FILE`). The full-standard suites use
`full_standard_depth_cases.txt`, `full_standard_sfm_cases.txt` and
`full_standard_slam_cases_{short,long}.txt`. Predictions are written under
`$BENCHMARK_EVAL_ROOT/artifacts/`. Each full suite keeps **one report folder per method**, shared by every
run (`spatialarena/report_layout.py`; layout in `docs/evaluation.md`, "Outputs"):

```text
$BENCHMARK_EVAL_ROOT/reports/benchmark_full_reports/full-depth/<METHOD_ID>/
$BENCHMARK_EVAL_ROOT/reports/benchmark_full_pose_reports/full-pose/<METHOD_ID>/
$BENCHMARK_EVAL_ROOT/reports/benchmark_full_slam_reports/full-slam-<range>/<METHOD_ID>/
```

Reruns, retries and fixes write into that same folder: finished cases are reused and rerun cases overwrite their
results in place, so all metrics of a benchmark stay in one tree. The engine refuses (exit 8) any other report
folder inside `reports/`, except one-off test folders `reports/_<name>/` and the sampled smoke suites
(`reports/benchmark_sampled_reports/`). `RUN_ID` (4th argument) only names temporary and checkpoint folders.
Use `--export=NIL` so stale caller environment variables do not leak into the jobs.

Adapter code and checkpoint paths default to the cluster layout and can be
overridden per method with `<METHOD>_CODE_PATH` / `<METHOD>_CKPT_PATH` (the exact
names are in `spatialarena/methods.py`, `MethodSpec.code_env` / `ckpt_env`).
Some methods need one-time setup, for example:

```sh
bash scripts/download/vggt_omega.sh                         # VGGT-Omega Google Drive checkpoints
sbatch --export=NIL scripts/slurm/build_come_extensions.sh  # CoME CUDA extensions
```

## Evaluating Existing Prediction Artifacts

Use these commands when method artifacts already exist and only metrics need to
be recomputed.

MonoDepth:

```sh
python -m spatialarena.eval.eval_mono_depth \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --experiment_root ~/tmp \
  --experiment_name benchmark_full_depth_VGGT_70074 \
  --methods VGGT \
  --datasets marigold \
  --scenes nyuv2 kitti eth3d diode \
  --output_path ~/tmp/eval_mono_depth.json \
  --strict
```

VideoDepth:

```sh
python -m spatialarena.eval.eval_video_depth \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --experiment_root ~/tmp \
  --experiment_name benchmark_full_depth_DA3_70078 \
  --methods DA3 \
  --datasets sintel bonn kitti \
  --output_path ~/tmp/eval_video_depth.json \
  --strict
```


SfM/Pose:

```sh
python -m spatialarena.eval.sparse_reconstruction_evaluator \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --experiment_root ~/tmp \
  --experiment_name benchmark_full_pose_VGGT_70750 \
  --methods VGGT \
  --tasks SfM \
  --datasets eth3d tanks_and_temples imc dtu scannetpp hiroom texturepoorsfm \
  --output_path ~/tmp/eval_sparse_pose.json \
  --strict
```

Prediction artifacts can be checked before evaluation:

```sh
python scripts/preprocess/verify_prediction_outputs.py \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --experiment_root ~/tmp \
  --experiment_name benchmark_full_depth_DA3_70078 \
  --methods DA3 \
  --tasks MonoDepth \
  --datasets marigold \
  --scenes nyuv2 \
  --strict \
  --output_path ~/tmp/verify_prediction_outputs.json
```

## Metrics And Reports

Depth evaluators write `eval_depth.json`. Sparse pose evaluation writes
`eval_sparse.json`. Each JSON contains per-scene metrics and resolved protocol
fields such as depth alignment, GT scale, clipping range, and frame counts.

Generate the no-ScanNet full-standard summary from method report roots:

```sh
sbatch --export=NIL scripts/slurm/aggregate_full_standard_reports.sh \
  "$BENCHMARK_EVAL_ROOT"/reports/benchmark_full_reports/full-depth/<METHOD_ID> \
  "$BENCHMARK_EVAL_ROOT"/reports/benchmark_full_pose_reports/full-pose/<METHOD_ID>
```

It writes `full_standard_benchmark_summary.{md,tsv}` to
`$BENCHMARK_EVAL_ROOT/reports/full_standard_summary/` (override with `SUMMARY_DIR`); generated results are not
kept under `docs/`. The benchmark dashboard (`app/webpage`) collects results directly from the
reports tree.

Depth metrics include AbsRel, SqRel, RMSE, RMSE-log, log10, Delta@1.03,
Delta@1.25, Delta@1.25^2, Delta@1.25^3, iRMSE, and SILog. SfM/Pose metrics
include ATE, relative rotation error, direct relative translation-direction
error, standard translation-axis error, RRA@5/15/30, RTA@5/15/30, mAA@30,
and AUC@1/3/5/10/20/30. ATE uses a pose-aware Sim(3) alignment (LMedS RANSAC over
the GTSAM pose fit + robust refit, `spatialarena/eval/pose_alignment.py`); see `docs/evaluation.md`.

## Mesh And ScanNet++ Evaluators

These evaluators are separate from the adapter artifact-only depth and pose
workflow. They evaluate mesh or ScanNet++ depth/geometry artifacts with
dataset-specific protocols.

### DTU Mesh

Script:

```text
spatialarena/eval/eval_dtu_mesh.py
```

Inputs:

- `--dataset_path`: parent directory containing the DTU dataset folder `dtu/`.
  The evaluator reads DTU cameras, object masks, plane masks, and STL ground
  truth through the DTU dataset adapter.
- `--input_dir`: parent directory containing method reconstructions.
- `--suffix`: experiment suffix. For each scan, the evaluator expects:

```text
INPUT_DIR/SCAN_NAME_SUFFIX/evaluation/meshes/mesh_final.ply
```

Command:

```sh
python -m spatialarena.eval.eval_dtu_mesh \
  --dataset_path ~/datasets \
  --input_dir ~/datasets/dtu/dtu_exp \
  --suffix 2dgs
```

Outputs are written next to each input mesh under `evaluation/meshes/`:

- `results.json` with `mean_d2s`, `mean_s2d`, and `overall`.
- `vis_<scan>_d2s.ply` and `vis_<scan>_s2d.ply` error visualizations.
- `recon_aligned.ply` when coordinate-frame alignment is enabled by code.

### Tanks-And-Temples Mesh

Script:

```text
spatialarena/eval/eval_tnt_mesh.py
```

Inputs:

- `--dataset_path`: parent directory containing
  `tanks_and_temples_gof/SCENE/` with the Tanks-and-Temples GT point cloud,
  crop JSON, alignment transform, and COLMAP SfM trajectory log.
- `--input_dir`: parent directory containing method reconstructions.
- `--suffix`: experiment suffix. For each scene, the evaluator expects:

```text
INPUT_DIR/SCENE_SUFFIX/evaluation/meshes/mesh_final.ply
```

Command:

```sh
python -m spatialarena.eval.eval_tnt_mesh \
  --dataset_path ~/datasets \
  --input_dir ~/datasets/tanks_and_temples_exp \
  --suffix 2dgs
```

Outputs are written under each scene's `evaluation/meshes/` folder:

- `metrics.json` with distance threshold, precision, recall, and F-score.
- Histogram and plot artifacts produced by the Tanks-and-Temples toolbox.

### ScanNet++ Depth And Geometry

Script:

```text
spatialarena/eval/eval_scannetpp_depth.py
```

The current main benchmark summaries exclude ScanNet/ScanNet++, but this
evaluator is available for ScanNet++ depth and geometry experiments.

Depth-sidecar mode evaluates predicted per-frame depth maps, optionally builds a
point cloud or TSDF mesh, and can also compare metric depths against GT depth
PNGs:

```sh
python -m spatialarena.eval.eval_scannetpp_depth \
  --mode depth \
  --scannetpp_root /path/to/scannetpp/raw \
  --colmap_root /path/to/scannetpp_colmap \
  --scene_ids SCENE_ID_0,SCENE_ID_1 \
  --image_folder images_4 \
  --model_folder 0 \
  --depth_folder depths_4_3dgs \
  --scale_json_root /path/to/scale_json_root \
  --scale_source estimated \
  --output_dir /path/to/scannetpp_eval
```

PLY mode evaluates an existing point cloud or mesh against ScanNet++ GT:

```sh
python -m spatialarena.eval.eval_scannetpp_depth \
  --mode ply \
  --scannetpp_root /path/to/scannetpp/raw \
  --scene_ids SCENE_ID_0 \
  --input_ply_root /path/to/predicted_ply_root \
  --input_ply_name point_cloud.ply \
  --output_dir /path/to/scannetpp_eval
```

Outputs:

- `OUTPUT_DIR/metrics.json` with aggregate geometry and optional depth metrics.
- `OUTPUT_DIR/scenes/SCENE_ID/metrics.json` with per-scene results.
- Optional distance artifacts, visualizations, and TSDF meshes when requested.

## Dataset And Case Utilities

List available canonical scenes:

```sh
python spatialarena/evaluate.py \
  --list_scenes \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks
```

Generate full-standard case files:

```sh
python scripts/test/generate_full_standard_cases.py \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --output scripts/test/full_standard_depth_cases_no_scannet.txt \
  --tasks MonoDepth VideoDepth \
  --exclude_scannet

python scripts/test/generate_full_standard_cases.py \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --output scripts/test/full_standard_sfm_cases.txt \
  --tasks SfM

# SLAM: one list per range track (short = stride 10, long = stride 3, large = every frame)
python scripts/test/generate_full_standard_cases.py --slam_range short \
  --output scripts/test/full_standard_slam_cases_short.txt
python scripts/test/generate_full_standard_cases.py --slam_range long \
  --output scripts/test/full_standard_slam_cases_long.txt
python scripts/test/generate_full_standard_cases.py --slam_range large \
  --output scripts/test/full_standard_slam_cases_large.txt
```

Verify downloaded benchmark datasets:

```sh
python scripts/preprocess/verify_benchmark_datasets.py \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks
```

Regenerate dataset documentation:

```sh
python scripts/preprocess/export_benchmark_data_doc.py \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --output docs/evaluation.md
```

## Current Known Deviations

- ScanNet is intentionally excluded from current full-standard summaries.
- RealEstate10K is intentionally excluded from the current full-pose scope.
- Dense depth predictions are resized with bilinear interpolation when needed;
  this is the canonical policy for dense continuous depth maps.
