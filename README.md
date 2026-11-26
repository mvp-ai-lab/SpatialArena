<p align="center">
  <img src="app/webpage/assets/logo/spatialarena.svg" alt="SpatialArena logo" width="96">
</p>

<h1 align="center">SpatialArena</h1>

<p align="center">
  <b>Where Do Geometric Foundation Models Stand?</b>
</p>

<p align="center">
  <a href="https://aibluefisher.github.io/">Yu Chen</a><sup>†</sup> ·
  <a href="https://happy-hsy.github.io/">Haosen Yang</a> ·
  <a href="https://openreview.net/profile?id=~Su_Ee_Tan1">Su Ee Tan</a> ·
  <a href="https://zhangzhensong.github.io/">Zhensong Zhang</a> ·
  <a href="https://scholar.google.com/citations?user=9a1PjCIAAAAJ&hl=en">Jifei Song</a> ·
  <a href="https://jiankangdeng.github.io/">Jiankang Deng</a><sup>✉</sup>
  <br>
  MVP Lab &amp; Imperial College London
  <br>
  <sub><sup>†</sup> Project leader &nbsp; <sup>✉</sup> Corresponding author</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.32816"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2611.32816-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/mvp-lab/SpatialArena-Dataset"><img alt="Hugging Face dataset" src="https://img.shields.io/badge/Hugging%20Face-Dataset-ffd21e?logo=huggingface&logoColor=black"></a>
  <a href="https://mvp-ai-lab.github.io/SpatialArena/"><img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-online-4361ee"></a>
  <br>
  <a href="https://github.com/mvp-ai-lab/SpatialArena/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/mvp-ai-lab/SpatialArena/actions/workflows/tests.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/mvp-ai-lab/SpatialArena/actions/workflows/tests.yml"><img alt="Coverage" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmvp-ai-lab%2FSpatialArena%2Fbadges%2Fcoverage.json"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white">
  <a href="LICENSE"><img alt="License: PolyForm Noncommercial 1.0.0" src="https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-lightgrey"></a>
</p>

---

SpatialArena is an open arena for **geometric foundation models**: monocular depth networks, feed-forward 3D/4D
reconstruction models, and vision–language models that predict geometry. Every model is scored on the same data, with
the same metrics, under the same rules, with robust and deterministic alignment.

This repository is the benchmark toolbox: the evaluators, the model adapters, benchmark dataset preparation, reports
and the [results dashboard](https://mvp-ai-lab.github.io/SpatialArena/).

<table>
  <tr>
    <td align="center"><b>5</b><br><sub>tasks, from monocular depth to SLAM</sub></td>
    <td align="center"><b>23 + 1</b><br><sub>public datasets + our own captures</sub></td>
    <td align="center"><b>418,929</b><br><sub>images</sub></td>
    <td align="center"><b>52</b><br><sub>models evaluated</sub></td>
  </tr>
</table>

SpatialArena is **artifact-only**: a method reads the canonical benchmark images and writes standardized prediction
artifacts (depth maps, COLMAP cameras, point clouds). The evaluators read only those artifacts plus the canonical
ground truth; they never import or instantiate DA3, VGGT, AMB3R or any other external model.

## Contents

- [Tasks and datasets](#tasks-and-datasets)
- [Installation](#installation)
- [Quick start: evaluate your model](#quick-start-evaluate-your-model)
- [Running the full benchmark](#running-the-full-benchmark)
- [Data and artifact formats](#data-and-artifact-formats)
- [Scoring existing predictions](#scoring-existing-predictions)
- [Metrics and reports](#metrics-and-reports)
- [Dataset and case utilities](#dataset-and-case-utilities)
- [Citation](#citation)
- [License](#license)

## Tasks and datasets

| Task | Datasets | What a method predicts |
| --- | --- | --- |
| **MonoDepth** | SpatialArena (ours), Marigold subsets (NYUv2, KITTI, ETH3D, DIODE, ScanNet), AerialMetric, HAMMER, iBims-1, Spring, Synth4K | per-image depth |
| **VideoDepth** | SpatialArena (ours), Sintel, Bonn, KITTI | temporally consistent depth for a clip |
| **SfM** | SpatialArena (ours), ETH3D, Tanks-and-Temples, IMC, DTU, ScanNet++, HiRoom, TexturePoorSfM | camera poses (COLMAP) |
| **SLAM** | SpatialArena (ours), 7Scenes, TUM RGB-D (dynamic), Oxford Spires, VBR, KITTI Odometry | the trajectory of one continuous recording |
| **MVRecon** | SpatialArena (ours), 7Scenes, ETH3D, Tanks-and-Temples, DTU, ScanNet++, HiRoom, NRGBD | multi-view depth and geometry |

Models run on **tracks** that set which inputs they get: `rgb` (images only), and variants that also give depth
(`rgb_depth`, masked or sparse depth) or the ground-truth cameras (`rgb_cameras`) as a prior. The datasets, scene
counts, protocol, metric definitions and case lists are in [`docs/evaluation.md`](docs/evaluation.md); the SLAM
protocol is in [`docs/benchmark/slam_protocol.md`](docs/benchmark/slam_protocol.md).

> [!NOTE]
> The current full-standard report intentionally excludes ScanNet, and RealEstate10K is outside the current
> full-pose scope.

## Installation

```bash
git clone https://github.com/mvp-ai-lab/SpatialArena.git
cd SpatialArena

# Install PyTorch for your CUDA version (or the CPU build) from https://pytorch.org first,
# so pip does not pick a default build. Then:
pip install -e .            # or: pip install -r requirements.txt (the pinned versions)
pip install -e ".[test]"    # plus pytest and jinja2, for `pytest test/`
```

Every default path lives under one data root, `SPATIALARENA_ROOT` (default `~/spatialarena`):

```bash
export SPATIALARENA_ROOT=/data/spatialarena
```

```text
$SPATIALARENA_ROOT/benchmarks/                benchmark datasets (<Task>/<dataset>/)
$SPATIALARENA_ROOT/benchmarks/_evaluation/    evaluation outputs (reports/, artifacts/, viewer_assets/)
$SPATIALARENA_ROOT/benchmarks/_model_codes/   the methods' source checkouts
$SPATIALARENA_ROOT/benchmarks/_checkpoints/   model checkpoints
```

`BENCHMARK_AUX_ROOT`, `BENCHMARK_EVAL_ROOT`, `METHOD_CODE_ROOT` and `CHECKPOINT_ROOT` override one folder each.

**Download the benchmark data** from the [Hugging Face release](https://huggingface.co/datasets/mvp-lab/SpatialArena-Dataset).
Each scene is one `.tar`, so extract the release once after downloading:

```bash
hf download mvp-lab/SpatialArena-Dataset --repo-type dataset \
    --local-dir $SPATIALARENA_ROOT/benchmarks
python $SPATIALARENA_ROOT/benchmarks/extract_benchmark_archives.py $SPATIALARENA_ROOT/benchmarks --delete
```

> [!IMPORTANT]
> Each dataset folder carries its own `LICENSE_NOTICE.md`; read it before using that data. Third-party datasets,
> benchmarks and model artifacts keep their own licenses and access terms.

The models themselves run in their own environments (one shared `si_bench` env, plus `genception` for GenCeption);
see [`docs/benchmark/evaluator_quickstart.md`](docs/benchmark/evaluator_quickstart.md) for setting them up and for
fetching checkpoints and method code.

## Quick start: evaluate your model

Wrap your model as **one command** that writes depth maps or COLMAP cameras, and the pipeline does the rest: it plans
the runs, executes them, scores them with the benchmark protocol, and writes the report the dashboard reads.

### 1. Install the benchmark package

```console
$ git clone https://github.com/mvp-ai-lab/SpatialArena.git && cd SpatialArena
$ pip install -e .
Successfully installed spatialarena-0.1.0
```

The models themselves run in one shared environment; see [Installation](#installation) and
[`docs/benchmark/evaluator_quickstart.md`](docs/benchmark/evaluator_quickstart.md).

### 2. Describe your model: one command template

```python
# evaluate.py
from spatialarena import BenchmarkEvaluator

bench = BenchmarkEvaluator(
    benchmark_root="/data/spatialarena/benchmarks",   # the downloaded benchmark data
    experiment_name="my_model",
    job_system="native",                              # or "slurm"
)

# SpatialArena's own captures, on three of the five tasks
bench.select_datasets([
    {"task": "MonoDepth", "dataset": "spatialarena"},
    {"task": "SfM", "dataset": "spatialarena"},
    {"task": "SLAM", "dataset": "spatialarena"},
])

# Your model is one command that writes depths/ and sparse/0 (COLMAP)
bench.select_models([{
    "name": "MyModel",
    "track": "rgb",
    "command": "python infer.py --data {data_path} --output {output_path}",
}])

report = bench.validate()
print(f"ok={report.ok}  errors={report.summary['error_count']}")
plan = bench.plan()
print(f"{len(plan.selected_runs)} runs planned")

result = bench.evaluate()          # inference, alignment and scoring
bench.export_report("./report")
```

Besides `{data_path}` and `{output_path}`, the command template can use `{images_dir}`, `{code_path}`,
`{ckpt_path}`, `{resolution}`, `{track}` and more (see `scripts/inference/external_command_adapter.py`). To
evaluate a **registered** model instead, select it by name:

```python
bench.select_models([{"name": "VGGT"}, {"name": "PI3_X", "track": "rgb_cameras"}])
```

### 3. Plan and evaluate, locally or on SLURM

```console
$ python evaluate.py
ok=True  errors=0
3 runs planned
  MonoDepth  spatialarena  MyModel  rgb   inference ... scored
  SfM        spatialarena  MyModel  rgb   inference ... scored
  SLAM       spatialarena  MyModel  rgb   inference ... scored
report written to ./report
```

The full benchmark, task by task, from the shipped configs:

```console
$ python -m spatialarena.run_config configs/benchmark/full_slam.yaml plan
SLAM 7scenes DA3-L rgb short
SLAM 7scenes DA3-L rgb long
...
# 230 runs: SLAM/rgb/large=36, SLAM/rgb/long=63, SLAM/rgb/short=116, SLAM/rgb_depth/long=6, SLAM/rgb_depth/short=9
```

### 4. Compare in the dashboard

```console
$ app/webpage/run.sh 8020
collecting every report root ... results.json
serving the dashboard on port 8020

# Leaderboard: your model ranked per task and overall
# Artifact viewer: its depth maps, trajectories and point clouds next to the others
```

> [!TIP]
> For a 5-minute sanity check of the pipeline, run one small case:
> `bench.select_runs([{"task": "MonoDepth", "dataset": "marigold", "model": "VGGT", "track": "rgb", "scenes": ["nyuv2"], "interval": 200}])`.

The same run can also be described in a YAML file
([`configs/benchmark/evaluation_example.yaml`](configs/benchmark/evaluation_example.yaml) is a commented template):

```python
bench = BenchmarkEvaluator()
bench.configure("configs/benchmark/evaluation_example.yaml", experiment_name="my_second_run")
if bench.validate().ok:
    report = bench.evaluate()      # SLURM: then bench.wait(report)
    bench.export_report()
```

The full API (state, validation messages, custom methods, output layout) is in
[`docs/benchmark/evaluator_quickstart.md`](docs/benchmark/evaluator_quickstart.md).

## Running the full benchmark

There is one config per task under `configs/benchmark/` (`full_monodepth.yaml`, `full_videodepth.yaml`,
`full_mvrecon.yaml`, `full_sfm.yaml`, `full_slam.yaml`; `all_tracks.yaml` holds all of them). Each lists every
registered method on the tracks that task allows:

```bash
python -m spatialarena.run_config configs/benchmark/full_slam.yaml plan       # the planned runs, one per line
python -m spatialarena.run_config configs/benchmark/full_slam.yaml generate   # write the job scripts only, for review
python -m spatialarena.run_config configs/benchmark/full_slam.yaml coverage   # compare with the canonical case lists
python -m spatialarena.run_config configs/benchmark/full_slam.yaml submit     # generate and submit
```

<details>
<summary><b>Submitting the case-list engine directly (SLURM)</b></summary>

<br>

The generated scripts run the case-list engine in `spatialarena/templates/_engine/`, which can also be submitted
directly. Each run is an inference job (GPU) followed by an evaluation job (CPU) on the same method, suite and track:

```bash
ENGINE=spatialarena/templates/_engine
JOB=$(sbatch --parsable --export=NIL $ENGINE/run_method_benchmark_infer.sh VGGT full-depth rgb)
sbatch --export=NIL --dependency=afterok:$JOB $ENGINE/run_method_benchmark_eval.sh VGGT full-depth rgb

# SLAM runs one range track at a time (short = stride 10, long = stride 3):
sbatch --export=NIL,SLAM_RANGE=long $ENGINE/run_method_benchmark_infer.sh PI3_X full-slam rgb
```

`scripts/test/submit_method_case_jobs.sh <suite> <METHOD>...` submits the same job pair for many methods at once.

The suite (2nd argument) is one of `sampled`, `all-task`, `multiscene`, `monodepth`, `sparse`, `full-depth`,
`full-pose` or `full-slam`; each selects its default case list under `scripts/test/` (override with `CASE_FILE`).
Predictions are written under `$BENCHMARK_EVAL_ROOT/artifacts/`. Each full suite keeps **one report folder per
method**, shared by every run:

```text
$BENCHMARK_EVAL_ROOT/reports/benchmark_full_reports/full-depth/<METHOD_ID>/
$BENCHMARK_EVAL_ROOT/reports/benchmark_full_pose_reports/full-pose/<METHOD_ID>/
$BENCHMARK_EVAL_ROOT/reports/benchmark_full_slam_reports/full-slam-<range>/<METHOD_ID>/
```

Reruns, retries and fixes write into that same folder: finished cases are reused and rerun cases overwrite their
results in place. The engine refuses (exit 8) any other report folder inside `reports/`, except one-off test folders
`reports/_<name>/` and the sampled smoke suites. `RUN_ID` (4th argument) only names temporary and checkpoint folders.
Use `--export=NIL` so stale caller environment variables do not leak into the jobs.

Method code and checkpoint paths default to the cluster layout and can be overridden per method with
`<METHOD>_CODE_PATH` / `<METHOD>_CKPT_PATH` (the names are in `spatialarena/methods.py`). Some methods need one-time
setup, for example:

```bash
bash scripts/download/vggt_omega.sh                         # VGGT-Omega Google Drive checkpoints
sbatch --export=NIL scripts/slurm/build_come_extensions.sh  # CoME CUDA extensions
```

</details>

## Data and artifact formats

<details>
<summary><b>Canonical dataset inputs</b></summary>

<br>

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

Only the folders a task needs have to exist:

| Task | Required canonical inputs |
| --- | --- |
| MonoDepth | `images/`, `depths/`; optional `masks/`; `materialized_full.json` for materialized subsets. |
| VideoDepth | `images/`, `depths/`. Bonn is paired by filename or nearest timestamp. |
| SfM | `images/` and GT COLMAP models. ETH3D can use calibration folders; DTU and Tanks-and-Temples use local COLMAP-backed 3DGS copies. (`CameraPose` is an alias of `SfM`.) |
| SLAM | One continuous recording per case with its own GT COLMAP model, in range tracks (short: stride 10, long: stride 3; large: Oxford Spires, VBR and KITTI Odometry). |
| MVRecon | `images/`, `depths/`, GT COLMAP models, and the dataset-specific geometry ground truth in `docs/evaluation.md`. |

</details>

<details>
<summary><b>Prediction artifacts a method writes</b></summary>

<br>

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

Depth tasks read `depths/`, SfM and SLAM read `sparse/0`, and the mesh evaluators read the dataset-specific mesh or
point artifacts. Methods may write additional logs; the evaluators only need the artifacts above.

</details>

## Scoring existing predictions

When the prediction artifacts already exist, only the metrics need to be computed:

```bash
python -m spatialarena.eval.eval_mono_depth \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --experiment_root ~/tmp --experiment_name benchmark_full_depth_VGGT_70074 \
  --methods VGGT --datasets marigold --scenes nyuv2 kitti eth3d diode \
  --output_path ~/tmp/eval_mono_depth.json --strict
```

<details>
<summary><b>VideoDepth, SfM/Pose and artifact verification</b></summary>

<br>

```bash
# VideoDepth
python -m spatialarena.eval.eval_video_depth \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --experiment_root ~/tmp --experiment_name benchmark_full_depth_DA3_70078 \
  --methods DA3 --datasets sintel bonn kitti \
  --output_path ~/tmp/eval_video_depth.json --strict

# SfM/Pose
python -m spatialarena.eval.sparse_reconstruction_evaluator \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --experiment_root ~/tmp --experiment_name benchmark_full_pose_VGGT_70750 \
  --methods VGGT --tasks SfM \
  --datasets eth3d tanks_and_temples imc dtu scannetpp hiroom texturepoorsfm \
  --output_path ~/tmp/eval_sparse_pose.json --strict

# Check prediction artifacts before evaluating them
python scripts/preprocess/verify_prediction_outputs.py \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --experiment_root ~/tmp --experiment_name benchmark_full_depth_DA3_70078 \
  --methods DA3 --tasks MonoDepth --datasets marigold --scenes nyuv2 \
  --strict --output_path ~/tmp/verify_prediction_outputs.json
```

</details>

## Metrics and reports

| Task | Metrics |
| --- | --- |
| Depth | AbsRel, SqRel, RMSE, RMSE-log, log10, δ<1.03, δ<1.25, δ<1.25², δ<1.25³, iRMSE, SILog |
| SfM / Pose | ATE, relative rotation error, relative translation-direction error, translation-axis error, RRA@5/15/30, RTA@5/15/30, mAA@30, AUC@1/3/5/10/20/30 |

ATE uses a pose-aware Sim(3) alignment (LMedS RANSAC over the GTSAM pose fit, then a robust refit;
`spatialarena/eval/pose_alignment.py`). Depth evaluators write `eval_depth.json` and pose evaluation writes
`eval_sparse.json`; each holds the per-scene metrics and the resolved protocol (alignment, GT scale, clipping range,
frame counts).

Aggregate the full-standard summary from the method report folders:

```bash
sbatch --export=NIL scripts/slurm/aggregate_full_standard_reports.sh \
  "$BENCHMARK_EVAL_ROOT"/reports/benchmark_full_reports/full-depth/<METHOD_ID> \
  "$BENCHMARK_EVAL_ROOT"/reports/benchmark_full_pose_reports/full-pose/<METHOD_ID>
```

It writes `full_standard_benchmark_summary.{md,tsv}` to `$BENCHMARK_EVAL_ROOT/reports/full_standard_summary/`
(override with `SUMMARY_DIR`). The dashboard (`app/webpage`) collects results directly from the reports tree.

Dense depth predictions are resized with bilinear interpolation when needed; this is the canonical policy for dense
continuous depth maps.

## Dataset and case utilities

<details>
<summary><b>List scenes, generate case files, verify downloads, regenerate the dataset docs</b></summary>

<br>

```bash
# List the canonical scenes
python spatialarena/evaluate.py --list_scenes --benchmark_root $SPATIALARENA_ROOT/benchmarks

# Full-standard case files
python scripts/test/generate_full_standard_cases.py \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --output scripts/test/full_standard_depth_cases_no_scannet.txt \
  --tasks MonoDepth VideoDepth --exclude_scannet
python scripts/test/generate_full_standard_cases.py \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks \
  --output scripts/test/full_standard_sfm_cases.txt --tasks SfM

# SLAM: one list per range track (short = stride 10, long = stride 3, large = every frame)
for range in short long large; do
  python scripts/test/generate_full_standard_cases.py --slam_range $range \
    --output scripts/test/full_standard_slam_cases_$range.txt
done

# Verify the downloaded benchmark datasets
python scripts/preprocess/verify_benchmark_datasets.py --benchmark_root $SPATIALARENA_ROOT/benchmarks

# Regenerate the dataset documentation
python scripts/preprocess/export_benchmark_data_doc.py \
  --benchmark_root $SPATIALARENA_ROOT/benchmarks --output docs/evaluation.md
```

</details>

## Citation

If you use SpatialArena in your research, please cite:

```bibtex
@article{chen2026spatialarena,
  title   = {SpatialArena: Where Do Geometric Foundation Models Stand?},
  author  = {Chen, Yu and Yang, Haosen and Tan, Su Ee and Zhang, Zhensong and Song, Jifei and Deng, Jiankang},
  journal = {arXiv preprint arXiv:2611.32816},
  year    = {2026}
}
```

## License

SpatialArena is source-available for noncommercial use under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Copyright (c) 2026 Yu Chen.

Commercial use requires separate written authorization from Yu Chen. Third-party datasets, benchmarks and model
artifacts referenced by this repository remain governed by their own licenses and access terms.

SpatialArena was split out of FeedForward-Benchmark, which keeps the training-data side (dataset downloads,
WebDataset conversion, dataloaders, MindCloud processing).
