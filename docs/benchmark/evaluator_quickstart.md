# `BenchmarkEvaluator` Quickstart: From a Fresh Machine to a Full Evaluation Run

This is a step-by-step guide for someone who has cloned this repo onto a
**fresh machine** -- no datasets, no checkpoints, no method source code, no
Python environments set up -- and wants to run the full evaluation
pipeline through the `spatialarena.BenchmarkEvaluator` Python API
(its full contract is the [API reference](#appendix-api-reference) at the end). For
what's actually *in* each dataset (tracks, scene counts, scene type,
evaluated-vs-total image counts, license), see
[`docs/evaluation.md`](../evaluation.md).

It covers, in order:

0. Installing `spatialarena` itself, and the separate conda envs
   methods actually run their inference in (Section 0.5).
1. What you need before you can evaluate anything (three separate asset
   types: datasets, checkpoints, method source code).
2. How to get each of those three asset types onto disk.
3. The `discover -> select -> validate -> plan -> evaluate -> collect`
   API walkthrough, with runnable code, and the same run driven from a
   YAML/JSON config file (Section 5a).
4. Running the full matrix (all registered methods x all registered
   datasets) in one call.
5. The two job systems (`slurm` and `native`) and when to use which.
6. Building a final report with `export_report()`.
7. Troubleshooting `validate()` output.
8. Adding a method that isn't in the registry.
9. Releasing packaged datasets to Hugging Face (for maintainers).

If you only want a quick sanity check that the pipeline itself works
end-to-end (not a real evaluation), see the "5-minute smoke test" box near
the end.

## 0. Install

```bash
cd SpatialArena
pip install -e .   # registers the `spatialarena` package and its
                    # declared dependencies (PyYAML included, for configure())
```

Set the data root once: every default path (datasets, evaluation outputs, method code, checkpoints) is under
`SPATIALARENA_ROOT` (default `~/spatialarena`; see the README's "Data root" and `spatialarena/paths.py`).

```bash
export SPATIALARENA_ROOT=/data/spatialarena   # datasets in $SPATIALARENA_ROOT/benchmarks
```

`spatialarena` is a thin orchestration layer: its own `install_requires`
does not pull in every *method's* own dependencies (PyTorch, `transformers`,
per-method requirements). Those live in separate conda envs built once, up
front -- see Section 0.5 below.

If you're re-installing into an environment that already has everything
else from `requirements.txt` set up and just want to pick up local package
changes without pip re-resolving/upgrading anything, `pip install -e .
--no-deps` is faster -- but skip it on a genuinely fresh environment, since
that also skips PyYAML and everything else `spatialarena` needs.

## 0.5 Setting up the Python environments methods actually run in

The `pip install -e .` above only sets up `spatialarena` itself (the
orchestration layer). Every method's own inference code runs via a plain,
cluster-native `/path/to/env/bin/python` binary path baked into
`spatialarena/templates/_engine/run_method_case_list.sh` -- not
whatever env your own shell/`spatialarena` process happens to be in,
except under `job_system="native"` where you're responsible for the
allocation (Section 6a). This is **one shared env (`si_bench`) for every
method except GenCeption** (a JAX/Flax stack that can't coexist with
`si_bench`'s PyTorch one), not one env per method; `run_method_case_list.sh`
overrides the python path only for GenCeption. Don't assume a method needs
its own env just because its `requirements.txt` looks heavy -- IGGT's turned
out to need nothing beyond what `si_bench` already has once its actual
inference code path (not its full training-env freeze) was traced through;
verify with a real smoke test rather than guessing either way.

| Env | Used for | Built by | Notes |
|---|---|---|---|
| `si_bench` | The **default** for every method except GenCeption | `scripts/slurm/build_si_bench_env.sh` | Python 3.11, torch 2.6.0+cu124. Rebuilt from a package list (`scripts/slurm/si_bench_freeze.txt`) rather than cloned: the earlier Python 3.10 env was a clone of `dogs` whose torch tree kept stale files from the cloned torch, which broke `torch.compile` (SenseNova-Vision). Adds AMB3R's source-built `torch_scatter` / `pytorch3d` / `flash-attn` and the SLAM deps (`gtsam-develop`, cp311+ only, hence Python 3.11; `faiss-gpu-cu12`; `viser`; `sim3solve`), and the official MonoDepth baselines' / DVD's deps (`diffusers` 0.36.0, `xformers` for torch 2.6, MoGe's `utils3d_moge`, FlexGEMM for MoGe-3 on its Triton kernels without the CUDA extension, `tabulate`); numpy stays 1.26.4 although MoGe and UniDepth ask for numpy 2. |
| `genception` | `GENCEPTION_DEPTH` / `GENCEPTION_POSE` only | `scripts/slurm/build_genception_env.sh` | Pure JAX/Flax, deliberately **no torch, no tensorflow** -- verified via real failed runs that torch's mere presence in this env breaks JAX's cuDNN init regardless of torch version, and tensorflow (pulled in transitively by `representations4d`) conflicts with jax's own bundled CUDA 12 plugin. |

Build the envs on a **compute node**, not the login node
(`sbatch`, not run inline):

```bash
sbatch --export=NIL scripts/slurm/build_si_bench_env.sh    # ~1 hour, CPU only; run from the repo root
# Overrides go after the script as KEY=VALUE (ENV_TOOL, ENV_NAME, ENV_PREFIX, BENCHMARK_ROOT, ...; see its header).
# ENV_TOOL=conda|uv (default: conda if found, else uv). uv needs a node g++ or CC=/CXX=; elsewhere point
# the other scripts at the env with PYTHON=<ENV_PREFIX>/bin/python.
sbatch scripts/slurm/build_genception_env.sh # only needed if you're running GenCeption
```

Both scripts are idempotent (existence checks before `conda create`) and
end with an import check of the env's key packages -- read the tail of the
SLURM log/err files to confirm before trusting the env. `build_si_bench_env.sh`
builds the CUDA extensions for sm_80/sm_90 without a GPU, with conda-forge's
GCC 12 (compute nodes have no system g++), installs `faiss` without deps so
it cannot pull numpy 2.x, and logs every package that had to fall back to a
newer cp311 release.

Which env a given method actually runs in is resolved per-`METHOD` inside
`run_method_case_list.sh` (`DEFAULT_BENCHMARK_PYTHON`, overridden to
`GENCEPTION_BENCHMARK_PYTHON` for GenCeption) -- if you're
adding a new method that needs its own env, add it there the same way
rather than fighting `si_bench`'s existing package set. Pointing
generated jobs at envs on your own machine instead of this repo's own
hardcoded cluster paths: set `PYTHON=/your/env/bin/python` as an
environment variable before `evaluate()` submits/runs anything -- it's
read as `PYTHON=${PYTHON:-$DEFAULT_BENCHMARK_PYTHON}`, so it overrides the
default for every method except GenCeption, whose override stays active.

## 1. The three asset types

Every (task, dataset, model, track) run needs three things to exist on
disk, and the evaluator's `discover_*()`/`validate()` methods only *check*
for them. All three can be fetched automatically for the subset each one
covers -- `bench.download_all_supported_datasets()` /
`bench.download_all_supported_models()` /
`bench.download_all_supported_method_code()` (Sections 2, 3, 4); the rest
of each section is the manual fallback for whatever isn't automated (no
verified download script, no verified materializer, or no verified
upstream repo URL -- never guessed at). `bench.download_all_supported()`
runs all three in one go. Every download call shows progress as bars by
default, or on the interactive terminal board with `progress="board"`
(view `5` lists each item with its current step and any failure reason);
see "Progress Display" in the API reference below.

| Asset | Where it must live | How the evaluator finds it | How you get it |
|---|---|---|---|
| **Dataset scenes** | `benchmark_root/<Task>/<dataset>/` (e.g. `MonoDepth/marigold/`) | `discover_datasets()` / `DatasetSpec` in `spatialarena/eval/dataset_registry.py` | `bench.download_all_supported_datasets()`, or `scripts/download/*.sh` + `scripts/preprocess/*` by hand (Section 2) |
| **Model checkpoints** | HF hub cache, a direct URL, or a local file path | `discover_models()`'s `ckpt_default`/`ckpt_env` fields | `bench.download_all_supported_models()`, or manual `wget`/HF fetch (Section 3) |
| **Method source code** | `method_code_root/<code_default relative path>` (e.g. `_model_codes/Pi3`) | `discover_models()`'s `code_default`/`code_env` fields | `bench.download_all_supported_method_code()`, or clone each method's own upstream repo yourself (Section 4) |

`benchmark_root` defaults to `spatialarena.eval.dataset_registry.
DEFAULT_BENCHMARK_ROOT` (`<benchmark_root>/_checkpoints` and
`<benchmark_root>/_model_codes` are the canonical, shared locations
automatic downloads write to); `checkpoint_root` defaults to
`benchmark_root/_checkpoints`, and `method_code_root` defaults to
`benchmark_root/_model_codes`. All three are constructor arguments on
`BenchmarkEvaluator` (or `configure()` keys, via `CHECKPOINT_ROOT`/
`METHOD_CODE_ROOT` env vars), so you don't have to use the defaults --
point them at wherever you've put things on your own machine/cluster.

## 2. Downloading datasets

### 2.0 Automated (most datasets)

```python
from spatialarena import BenchmarkEvaluator

bench = BenchmarkEvaluator(benchmark_root="/path/to/benchmark_root")

for entry in bench.list_supported_datasets():
    print(f"{entry['task']:10s} {entry['dataset']:20s} {entry['status']:14s} {entry['notes']}")

results = bench.download_all_supported_datasets(skip_existing=True)
for r in results:
    print(r["task"], r["dataset"], r["status"])   # already_present | ok | failed | skipped
```

`list_supported_datasets()` reports a `status` for every registered
`(task, dataset)` pair, grounded in whether a real download+materialize
step exists in this repo -- it never claims something is automatable
without a verified script path:

- `"automated"` -- `download_all_supported_datasets()` runs the download
  script and any materialize step for you (22 of the 28 entries as of this
  writing, including the `legacy_relative_paths` fallbacks like
  `SLAM/7scenes` and `SfM/hiroom`, which need zero steps of their own
  because a sibling entry already materializes the data they resolve to).
- `"download_only"` -- a download script exists (`eth3d`, `dtu`) but this
  repo has no verified canonical-layout materializer for it; the raw
  download runs, then you still need Section 2a to reshape it.
- `"manual"` -- no download script was found in this repo at all
  (`tanks_and_temples`); see the table below for what to do by hand.

`download_all_supported_datasets()` always skips (never guesses at
running) anything not `"automated"`, and with `skip_existing=True` (the
default) also skips any dataset `describe_benchmark_scenes()` already
reports as present -- pass `datasets=[("SfM", "imc"), ...]` to restrict it
to specific pairs, or `skip_existing=False` to force a re-download.

### 2.0a Fastest path: download the pre-packaged release

Most of the datasets above (everything Section 10 marks `included`) are
already packaged and hosted at
[`mvp-lab/SpatialArena-Dataset`](https://huggingface.co/datasets/mvp-lab/SpatialArena-Dataset)
-- pulling from there skips re-running any `scripts/download/` +
`scripts/preprocess/` step yourself:

```bash
pip install huggingface_hub
hf download mvp-lab/SpatialArena-Dataset --repo-type dataset \
    --local-dir /path/to/benchmark_root
python /path/to/benchmark_root/extract_benchmark_archives.py /path/to/benchmark_root --delete
```

Each scene is stored as one uncompressed `<Task>/<dataset>/<scene>.tar`
(the raw tree is ~900k files, which the Hub rate-limits), so the release
must be extracted once after downloading. The bundled
`extract_benchmark_archives.py` (also `scripts/release/` in this repo;
standard library only) unpacks every archive in place, restoring the exact
`benchmark_root` layout including relative symlinks, and `--delete` removes
each archive after it is extracted. Rerunning it skips archives already
extracted.

or from Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mvp-lab/SpatialArena-Dataset",
    repo_type="dataset",
    local_dir="/path/to/benchmark_root",
    # allow_patterns=["MVRecon/*", "VideoDepth/*", "*.py"],  # to fetch only some tasks
)
# then: python /path/to/benchmark_root/extract_benchmark_archives.py /path/to/benchmark_root --delete
```

Each dataset folder in that repo carries its own `LICENSE_NOTICE.md` --
read it before using that data; the repo's own `README.md` lists which
datasets are packaged there and which are excluded (unconfirmed license,
see Section 10). `MonoDepth/marigold` ships without its `scannet` subset
(ScanNet's terms forbid redistribution); its `LICENSE_NOTICE.md` gives the two
commands that rebuild it from Marigold's own download. A few registered pairs resolve to the exact same on-disk
data as another entry (e.g. `SfM/eth3d` == `MVRecon/eth3d`) -- those show
up as a `SEE_INSTEAD.md` pointer instead of a second copy, to avoid
duplicating bytes in the download.

### 2.1 Manual reference (what the automated steps above actually run, and the fallback for `download_only`/`manual` entries)

Run `python -c "from spatialarena.eval.dataset_registry import
DEFAULT_DATASET_SPECS; [print(s.task, s.dataset, s.relative_path) for s in
DEFAULT_DATASET_SPECS]"` to see the full, current list (28 entries as of
this writing) -- this table is the ground truth if it and the list below
ever disagree.

| Task | Dataset | Benchmark-root path | Download script | Extra materialize step |
|---|---|---|---|---|
| MonoDepth | marigold | `MonoDepth/marigold` | `scripts/download/marigold.sh` | `scripts/preprocess/materialize_marigold_sample.py` |
| MonoDepth | hammer, ibims1, spring | `MonoDepth/hammer`, `MonoDepth/ibims1`, `MonoDepth/spring` | `scripts/download/moge_eval.sh [HAMMER] [iBims-1] [Spring]` | `scripts/preprocess/materialize_moge_eval.py --datasets hammer ibims1 spring` |
| MonoDepth | synth4k | `MonoDepth/synth4k` | `scripts/download/synth4k.sh [game ...]` | `scripts/preprocess/materialize_synth4k.py` |
| MonoDepth | aerialmetric | `MonoDepth/aerialmetric` | `scripts/download/aerialmetric.sh` | `scripts/preprocess/materialize_aerialmetric.py` |
| VideoDepth | sintel | `VideoDepth/sintel` | `scripts/download/sintel.sh` | -- |
| VideoDepth | bonn | `VideoDepth/bonn` | `scripts/download/bonn.sh` | `scripts/preprocess/bonn_preprocessing.py` |
| VideoDepth | kitti | `VideoDepth/kitti` | `scripts/download/kitti.sh` | `scripts/preprocess/kitti_preprocessing.py` |
| SfM, MVRecon | eth3d | `SfM/eth3d`, `MVRecon/eth3d` | `scripts/download/eth3d.sh`, then `scripts/preprocess/finalize_mvrecon_layout.sh` | -- |
| SfM, MVRecon | tanks_and_temples | `3DGS/tanks_and_temples` (or `SfM/tanks_and_temples`, as the release ships it), `MVRecon/tanks_and_temples` | *(no script found under `scripts/download/`; obtain from the dataset's own official release and lay out per Section 2a below)* | -- |
| SfM | imc | `SfM/imc` | `scripts/download/imc.sh` | -- |
| MVRecon, SLAM | 7scenes | `MVRecon/7scenes`, `SLAM/7scenes` | `scripts/download/7scenes.sh` | `scripts/preprocess/7scenes_preprocessing.py`, then `scripts/preprocess/normalize_benchmark_layout.py <benchmark_root>/MVRecon/7scenes` and `scripts/preprocess/generate_benchmark_metadata.py --dataset_roots <benchmark_root>/MVRecon/7scenes`; for SLAM also `scripts/preprocess/materialize_7scenes_slam.py --source_root <benchmark_root>/MVRecon/7scenes --benchmark_root <benchmark_root>`, then `scripts/preprocess/materialize_slam_inputs.py --datasets 7scenes` |
| SfM, MVRecon | dtu | `SfM/dtu`, `MVRecon/dtu` | `scripts/download/dtu.sh <output_dir>` | `scripts/preprocess/fuse_depth_map_dtu.py` |
| SfM, MVRecon | scannetpp, hiroom | `SfM/scannetpp`, `MVRecon/scannetpp`, `SfM/hiroom`, `MVRecon/hiroom` | `scripts/download/da3_bench.sh` | `scripts/preprocess/materialize_da3_bench.py --source_root <da3_bench output> --benchmark_root <your benchmark_root> --datasets scannetpp hiroom` |
| SfM | texturepoorsfm | `SfM/texturepoorsfm` | `scripts/download/texturepoorsfm.sh` | `scripts/preprocess/materialize_texturepoorsfm.py` |
| MVRecon | nrgbd | `MVRecon/nrgbd` | `scripts/download/nrgbd.sh` | `scripts/preprocess/materialize_nrgbd.py --source_root <nrgbd_raw> --benchmark_root <your benchmark_root>` |
| SLAM | tum_dynamics | `SLAM/tum_dynamics` | `scripts/download/tum_dynamics.sh` | `scripts/preprocess/materialize_tum_dynamics.py --source_root <tum_dynamics_raw> --benchmark_root <your benchmark_root>`, then `scripts/preprocess/materialize_slam_inputs.py --datasets tum_dynamics` |
| SLAM | oxford_spires | `SLAM/oxford_spires` | `scripts/download/oxford_spires.sh` | `scripts/preprocess/materialize_oxford_spires.py --raw_root <oxford_spires_raw> --benchmark_root <your benchmark_root>` (needs the lingbot-map code clone), then `scripts/preprocess/materialize_slam_inputs.py --datasets oxford_spires` |
| SLAM | vbr | `SLAM/vbr` | `scripts/download/vbr.sh` | `scripts/preprocess/materialize_vbr.py --raw_root <vbr_raw> --benchmark_root <your benchmark_root>`, then `scripts/preprocess/materialize_slam_inputs.py --datasets vbr` |
| SLAM | kitti_odometry | `SLAM/kitti_odometry` | `scripts/download/kitti_odometry.sh` | `scripts/preprocess/materialize_kitti_odometry.py --raw_root <kitti_odometry_raw> --benchmark_root <your benchmark_root>`, then `scripts/preprocess/materialize_slam_inputs.py --datasets kitti_odometry` |

Notes:

- Several download scripts (`marigold.sh`, `sintel.sh`, `bonn.sh`, `kitti.sh`,
  `7scenes.sh`, `imc.sh`) carry `#SBATCH` headers and a **hardcoded**
  `DATA_ROOT_PATH`/output directory from this repo's own cluster setup --
  open each script before running it and point its output path at your own
  `benchmark_root`, or `mv`/symlink the result afterward. They're plain bash
  otherwise (`bash scripts/download/marigold.sh` works directly; only
  `sbatch` it if you're also on a SLURM cluster and want it queued rather
  than run inline).
- `dtu.sh` takes its output directory as `$1` (`bash scripts/download/dtu.sh
  /path/to/benchmark_root/SfM/dtu`); most of the others don't take
  arguments -- check each script's top before running it.
- `da3_bench.sh` needs `hf` (the huggingface_hub CLI) and downloads via the
  `depth-anything/DA3-BENCH` HF dataset repo.
- `texturepoorsfm.sh` needs `gdown` (Google Drive). `materialize_texturepoorsfm.py`
  copies (not symlinks) by default, so the canonical `SfM/texturepoorsfm` is
  self-contained -- pass `--symlink` for the old space-saving behavior, which
  keeps a hard dependency on `source_root` staying on disk. The raw pose
  files are world-to-camera and are written to COLMAP unchanged. The current
  benchmark (split v2) is built by `scripts/preprocess/consolidate_texturepoorsfm.py
  benchmark` from the per-object pool `dataset_aux/texturepoorsfm`: 30 bags of
  the 5 held-out objects in `scripts/preprocess/assets/texturepoorsfm_split_v2.json`.
  The other 12 objects are training data only.
- `nrgbd.sh` downloads Neural RGB-D Surface Reconstruction's
  `neural_rgbd_data.zip` (~7.25GB, 9 scenes) plus its published
  `meshes.zip` (~514MB) from `kaldir.vc.in.tum.de`.
  `materialize_nrgbd.py` converts the dataset's own OpenGL-convention
  poses to COLMAP and copies `depth_filtered/` + `gt_mesh.ply` through.
- The MonoDepth extension datasets download from Hugging Face at pinned revisions into
  `<benchmark root>/MonoDepth_source/` (`DATA_ROOT_PATH`; `HF_HOME` defaults there too, off the home quota):
  `moge_eval.sh` fetches MoGe's processed eval sets (`Ruicheng/monocular-geometry-evaluation`: HAMMER, iBims-1,
  Spring, ~2 GB), `synth4k.sh` the five InfiniDepth game tars (`ritianyu/game_4k_data`, ~87 GB, kept as tars),
  `aerialmetric.sh` only the AerialMetric Test-Split (`Kuiee/AerialMetric-ECCV2026`, ~15 GB). The materializers
  reuse the committed frame lists in `spatialarena/eval/monodepth_splits/` (`--write_split` reselects them):
  HAMMER/iBims-1/Spring keep MoGe's exact frames; Synth4K and AerialMetric keep half (`--frame_fraction 0.5`) of
  each scene's frames with GT in range (per site for AerialMetric `decoupled`), uniformly spaced
  (`--frames_per_scene 100` restores the earlier 100-frame selection). MonoDepth evaluation then scores every
  materialized image (no frame cap). Depth is written as float32 `.npy` metres with invalid pixels (and Synth4K's sky) set to 0; each scene's
  depth limits go into `metadata.json`. `scripts/preprocess/check_monodepth_gt.py --datasets <name> --output_dir
  <dir>` then checks GT statistics, runs an oracle evaluation and writes overlay panels.
- `tum_dynamics.sh` downloads the 9 TUM RGB-D "Dynamic Objects" sequences
  (~7GB total) from `vision.in.tum.de` (redirects to `cvg.cit.tum.de`).
  `materialize_tum_dynamics.py` associates rgb/depth/groundtruth by
  timestamp and writes each camera's official per-model intrinsics
  (freiburg2 has real lens distortion; freiburg3 is pre-undistorted).
- `oxford_spires.sh` (~76 GB from HF `ori-drs/oxford_spires_dataset`), `vbr.sh` (~121 GB of LoGeR's processed
  release, HF `Junyi42/vbr_processed`) and `kitti_odometry.sh` (the official ~65 GB color archive, of which only
  `image_2` of sequences 00-10 is extracted, plus calib and poses) write the raw data; their materialize scripts
  build the native-resolution sequences under `SLAM_source/<dataset>/` (Oxford Spires through lingbot-map's own
  `preprocess/oxford.py`; KITTI with its cam0 ground truth moved to the color camera), and
  `materialize_slam_inputs.py` writes the fixed-resolution SLAM inputs every method reads under `SLAM/<dataset>/`
  (see `docs/benchmark/slam_protocol.md`). None of the three has depth, so they run on the `rgb` track only.

### 2a. After downloading: verify, don't guess

Whatever layout a download+materialize pair produces, the canonical shape
`BenchmarkEvaluator`/`describe_benchmark_scenes()` expects is:

```text
benchmark_root/<Task>/<dataset>/
  metadata.json
  <scene>/
    images/
    depths/          # depth tasks
    sparse/0/        # pose/reconstruction tasks (COLMAP model)
```

If a dataset's `metadata.json` is missing after you've laid out the scene
folders, generate it with:

```bash
python scripts/preprocess/generate_benchmark_metadata.py \
    --benchmark_root /path/to/benchmark_root \
    --dataset_roots /path/to/benchmark_root/<Task>/<dataset>
```

(omit `--dataset_roots` to (re)generate metadata for every dataset found
under `--benchmark_root`; add `--dry_run` to preview first). To generically
reshape a raw download into the canonical layout instead of writing a
one-off script, see `scripts/preprocess/normalize_benchmark_layout.py`.

Then verify against the registry's own expectations, independent of the
evaluator:

```bash
python scripts/preprocess/verify_benchmark_datasets.py --benchmark_root /path/to/benchmark_root
```

...or just ask the evaluator directly (this is the check that actually
matters, since it's what `validate()` uses):

```python
from spatialarena import BenchmarkEvaluator

bench = BenchmarkEvaluator(benchmark_root="/path/to/benchmark_root")
result = bench.discover_datasets()
print(result["summary"])   # {"dataset_count": ..., "missing_count": ..., "total_scenes": ...}
for record in result["missing"]:
    print(record["task"], record["dataset"], "->", record["root_directory"])
```

## 3. Getting model checkpoints

### 3.0 Automated (HF-hub and direct-URL checkpoints)

```python
from spatialarena import BenchmarkEvaluator

bench = BenchmarkEvaluator()

for m in bench.list_supported_models():
    print(f"{m['name']:20s} {m['ckpt_kind']:10s} automatable={m['automatable']}")

results = bench.download_all_supported_models()   # or methods=["PI3_X", "DUST3R"]
for r in results:
    print(r["name"], r["status"], r.get("path"))   # ok | skipped
```

`list_supported_models()` classifies every registered method's
`ckpt_default` into `ckpt_kind`: `"hf_hub"`, `"url"`, `"gdrive"`,
`"hf_file"`, and `"shipped_with_code"` are `automatable` (21 of the 27
registered methods as of this writing); `"local_file"` (this cluster's own
private pre-migrated copies -- VGGT, DA3-G, FastVGGT, VGGT-Omega) entries
are always reported `"skipped"` with a `reason`, never guessed at --
`hf_hub` checkpoints are fetched via `huggingface_hub.snapshot_download()`
into the shared HF cache; `url` checkpoints are downloaded directly into
`checkpoint_root/local/<method>/`; `gdrive` checkpoints (CUT3R's, into the
path under `method_code_root` its own README expects; AMB3R's, into
`checkpoint_root/local/amb3r/` like every other local-file checkpoint) are
fetched with `gdown` (see `GDRIVE_CHECKPOINT_URLS` in
`spatialarena/download.py`); `hf_file` checkpoints (currently just
IGGT4D's) fetch one named file from a specific HF repo into that same
`method_code_root`-relative path (see `HF_FILE_CHECKPOINTS` -- IGGT4D's
repo is gated, so this reports `"skipped"` with an actionable reason until
the account behind `HF_TOKEN` has been granted access); and
`shipped_with_code` checkpoints (currently just COME's) are copied from an
already-cloned method code checkout into
`checkpoint_root/local/<method>/`, same as every other local-file checkpoint
(see `SHIPPED_WITH_CODE_CHECKPOINTS`). For anything
`"skipped"`, use the manual categories below.

### 3.1 Manual reference (checkpoint categories `download_all_supported_models()` can't cover)

Run this to see exactly what every registered method needs, grounded in
the live registry rather than this doc going stale:

```python
from spatialarena import BenchmarkEvaluator

bench = BenchmarkEvaluator()
for m in bench.discover_models()["methods"]:
    print(f"{m['name']:20s} ckpt_env={m['ckpt_env']} ckpt_default={m['ckpt_default']}")
```

Each method's `ckpt_default` falls into one of six categories:

1. **Hugging Face hub id** (e.g. `facebook/map-anything`, `yyfz233/Pi3X`) --
   auto-downloads the first time the method's adapter runs, *unless*
   `HF_HUB_OFFLINE=1` is set. The generated SLURM scripts set
   `HF_HUB_OFFLINE=1` by default (see `run_method_benchmark_infer.sh`) to
   avoid slow/flaky network HEAD requests on a cluster where the checkpoint
   is already cached -- **on a fresh machine, export `HF_HUB_OFFLINE=0`** (or
   leave it unset) before your first run of each such method so it can
   actually fetch the weights; you can set it back to `1` afterward once
   everything is cached under `HF_HOME`.
2. **Direct download URL** (e.g. DUSt3R/MASt3R/Pow3R's naverlabs URLs) --
   download it yourself and place it at the path the method's `ckpt_env`
   variable should point to (or pass `ckpt_path` directly via a
   custom-adapter selector -- see Section 8).
3. **Local file path under a shared checkpoint store** (VGGT, DA3-G,
   FastVGGT, VGGT-Omega on this repo's own cluster) -- these are this
   session's private pre-migrated copies and won't exist on a fresh
   machine. Get the underlying checkpoint from its official source (VGGT:
   `facebook/VGGT-1B` on HF is the same model in hub form -- see
   `ROBUSTVGGT`'s entry for reference; DA3-Giant/FastVGGT/VGGT-Omega: check
   each method's own repo/paper for their release, or reuse
   `scripts/download/vggt_omega.sh` as a *pattern* for a Google-Drive-hosted
   checkpoint) and either replicate the expected path, or override it (see
   below). `GENCEPTION_DEPTH`/`GENCEPTION_POSE` share this pattern too --
   `scripts/download/genception.sh` fetches the real, public, multi-file
   bundle (transformer + VAE + officially-released prompt embeddings) into
   `checkpoint_root/local/genception/`; it's a real script, just not
   automated via the generic single-file `download_checkpoint()` mechanism
   since it's several files, not one. `GENCEPTION_POSE`'s pose/"raymap"
   prompt has no official release at all -- run `scripts/preprocess/
   generate_genception_raymap_prompt.py` after `genception.sh` to
   reconstruct one (see the note right after the six categories below).
   `AMB3R`'s checkpoint is also a local-file path in this same
   `checkpoint_root/local/amb3r/` layout, but *is* automatable -- see
   category 4.
4. **`gdrive`-sourced, either `$METHOD_CODE_ROOT`-relative or a plain local
   file** (`CUT3R`'s `cut3r_512_dpt_4_64.pth`, expected inside its own code
   checkout per its README; `AMB3R`'s `amb3r.pt`, a plain canonical
   `checkpoint_root/local/amb3r/` path since AMB3R's own `load_weights()`
   takes an arbitrary path) -- `download_all_supported_models()` fetches
   both automatically via `gdown` (see `GDRIVE_CHECKPOINT_URLS` in
   `spatialarena/download.py`).
5. **`$METHOD_CODE_ROOT`-relative, one named file on the HF hub**
   (`IGGT4D`'s `model.safetensors`, per its own README's instructions to
   save it at `checkpoints/model.safetensors` inside the code checkout) --
   `download_all_supported_models()` fetches it with
   `huggingface_hub.hf_hub_download()` (see `HF_FILE_CHECKPOINTS`).
   `IGGT4D`'s HF repo is gated: until the account behind `HF_TOKEN` has been
   granted access at `https://huggingface.co/HorizonRobotics/IGGT4D`, this
   reports `status="skipped"` with that URL in the `reason`, never a crash.
6. **Ships inside the method's own code clone, canonicalized to the shared
   checkpoint store** (`COME`'s confidence-distill checkpoint is committed
   directly in the `CoMe` repo at
   `output/confidence_distill/vggt/.../checkpoint.pth`) --
   `download_all_supported_models()` copies it from the already-cloned
   `CoMe` checkout into `checkpoint_root/local/come/checkpoint.pth`, the
   same canonical layout every other local-file checkpoint uses (see
   `SHIPPED_WITH_CODE_CHECKPOINTS`), rather than resolving it from inside
   `method_code_root` at run time.

**`GENCEPTION_POSE` follows GenCeption's official camera-pose evaluation.**
The authors answered
[google-deepmind/representations4d#10](https://github.com/google-deepmind/representations4d/issues/10)
(closed 2026-11-10) with `representations4d/genception/evals/eval_camera_pose.py` and the
`raymap.npy` prompt embedding ("Input: RGB. Output: Raymap."), which `scripts/download/genception.sh`
downloads. `spatialarena/geometry/genception_raymap.py` ports that decode (it matches the upstream
functions to ~1e-7 on synthetic raymaps): the 160x256 origin block holds the camera centre,
log-scaled; the rest holds ray directions; both are read with a 2-patch margin; the rotation comes from
the `robust2` solver (`GENCEPTION_POSE_SOLVER=svd` selects the other one upstream reports); one pinhole K
is fitted per clip; poses are smoothed over time (`GENCEPTION_RAW_POSES=1` exports the raw ones). Frames
are resized to cover 480x832 and center-cropped, the 81-frame clip is padded with its last frame, and the
fitted K is mapped back to the original images. Upstream scores Sintel with Sim(3) ATE and RPE at
delta 1 (MegaSaM); this benchmark scores the exported poses with its own SfM metrics.

**Overriding a checkpoint path** without touching the registry: every
method's `ckpt_env` value (e.g. `VGGT_CKPT_PATH`) is a real environment
variable the engine scripts read (`${VGGT_CKPT_PATH:-<ckpt_default>}`), so
`export VGGT_CKPT_PATH=/my/path/vggt.pt` before calling `evaluate()`
overrides it for every run in that process -- no code change needed. All
of these local-file checkpoints, plus the shared HF cache, live under one
configurable root -- `BenchmarkEvaluator`'s `checkpoint_root` (default
`benchmark_root/_checkpoints`, canonical on this repo's cluster at
`$SPATIALARENA_ROOT/benchmarks/_checkpoints`);
point that at your own equivalent directory via the constructor,
`configure()`, or the `CHECKPOINT_ROOT` env var if you replicate the same
`local/<method>/...` layout, or just set each method's own `*_CKPT_PATH`
env var individually.

`validate()` (Section 5) flags exactly which of these four categories each
selected model falls into and whether it currently resolves -- run it
before assuming a checkpoint problem is fixed.

## 4. Getting method source code

### 4.0 Automated (methods with a verified upstream repo)

```python
from spatialarena import BenchmarkEvaluator

bench = BenchmarkEvaluator(method_code_root="/path/to/_model_codes")

for m in bench.list_supported_method_code():
    print(f"{m['name']:20s} {m['leaf_dir']:20s} {m['status']:10s} {m['repo_url'] or m['notes']}")

results = bench.download_all_supported_method_code()   # or methods=["PI3_X", "DUST3R"]
for r in results:
    print(r["name"], r["status"], r.get("path") or r.get("reason"))
```

`list_supported_method_code()` reports `status="automated"` only for
methods whose vendored directory has a verified `git clone` URL in
`spatialarena/download.py`'s `METHOD_CODE_REPOS` (all 27 registered
methods as of this writing -- those URLs were read directly off each
clone's own `git remote get-url origin` and confirmed reachable via
`git ls-remote`, not guessed). Methods that share one upstream repo
(VGGT/FASTVGGT/ROBUSTVGGT -> `RobustVGGT`; DA3-L/DA3-G/DA3-NGL ->
`depth-anything-3`, the official ByteDance-Seed repo, not a third-party
vendored copy) clone it once, so the 27 methods collapse to 22 unique
directories under `_model_codes/` (canonically at
`$SPATIALARENA_ROOT/benchmarks/_model_codes`
on this cluster) -- the rest resolve to `status="already_present"`.
Anything with `status="manual"` has no confirmed source here -- clone it
yourself per the method's own paper/GitHub listing, same as before.

### 4.1 Manual reference

Every non-`native` method's adapter shells out to that method's own
inference code, vendored under `<method_code_root>/<code_default's
relative part>` (e.g. `PI3_X`'s `code_default` is
`$METHOD_CODE_ROOT/Pi3`, so you need that method's official repo cloned
into `_model_codes/Pi3`). This repo does **not** ship those as git
submodules (no `.gitmodules`) -- they're plain vendored clones you add
yourself:

```bash
mkdir -p _model_codes
cd _model_codes
git clone <method's official upstream repo> Pi3   # match the code_default's leaf dirname exactly
# ...and install that method's own requirements -- most methods share the
# si_bench env (Section 0.5); only GenCeption/IGGT/SenseNova need a
# different one.
```

Run the same `discover_models()` loop as Section 3, printing `code_default`
instead of `ckpt_default`, to get the exact expected directory name for
every method. `spatialarena/download.py`'s `METHOD_CODE_REPOS` (Section
4.0) already covers this mapping for the methods it lists as
`"automated"`; for anything `list_supported_method_code()` reports
`"manual"`, this doc deliberately doesn't guess a URL -- go to that
method's own paper/GitHub listing plus `discover_models()`'s live
`code_default` field instead.

You don't need every method's code before you can evaluate *anything* --
`validate()` only checks the methods you've actually `select_models()`-ed,
so start with one or two you have code+checkpoints for (see the 5-minute
smoke test below) and add more incrementally.

## 5. The core API loop

```python
from spatialarena import BenchmarkEvaluator

bench = BenchmarkEvaluator(
    benchmark_root="/path/to/your/benchmark_root",       # Section 2
    checkpoint_root="/path/to/your/_checkpoints",         # Section 3
    method_code_root="/path/to/your/_model_codes",        # Section 4
    experiment_root="/path/to/where/scripts+results/go",
    experiment_name="my_first_run",
    job_system="native",   # or "slurm" -- see Section 6
)

# 1. discover: see what's actually on disk / registered, before selecting anything
datasets = bench.discover_datasets()
print(datasets["summary"])
models = bench.discover_models()
print([m["name"] for m in models["methods"]])

# 2. select: choose what you want to run
bench.select_datasets([
    {"task": "MonoDepth", "dataset": "marigold"},
    {"task": "SfM", "dataset": "eth3d"},
])
bench.select_models([
    {"name": "VGGT"},
    {"name": "PI3_X", "track": "rgb_cameras"},
])

# 3. validate: cheap, actionable, catches problems before you burn GPU time
report = bench.validate()
print(f"ok={report.ok}  errors={report.summary['error_count']}  warnings={report.summary['warning_count']}")
for issue in report.issues:
    print(f"[{issue.level}] {issue.scope}: {issue.message}")
if not report.ok:
    raise SystemExit("fix the errors above before evaluating")

# 4. plan: preview the exact (task, dataset, model, track) runs -- read-only, no side effects
plan = bench.plan()
print(f"{len(plan.selected_runs)} runs planned:")
for run in plan.selected_runs:
    print(" ", run)

# 5. evaluate: actually run it
report = bench.evaluate()

# 6. SLURM only (see Section 6): block with progress bars until finished, then collect
# report = bench.wait(report)
```

`select_datasets()`/`select_models()`/`select_runs()` all **replace** the
prior selection by default; pass `append=True` to add instead. `plan()`'s
default (no explicit `select_runs()`) is the cross product of every
selected dataset x every selected model, automatically filtered by the
[Task/Track Compatibility Matrix](#tasktrack-compatibility-matrix)
intersected with each model's own supported tracks -- so e.g. selecting
`MonoDepth` + a model with `rgb_cameras` support still only produces an
`rgb` run for that task, since `MonoDepth` has nothing meaningful left to
predict once cameras are given as a prior. MonoDepth runs are also single-view: the engine sets
`BENCHMARK_SINGLE_VIEW=1`, so multi-view models predict each image on its own (an independent `[B, 1, 3, H, W]`
batch or one image per forward pass), never the case's frames as one multi-view input, and a scene that mixes
image sizes runs once per size; see the Frame Selection
section of [`docs/evaluation.md`](../evaluation.md).

## 5a. Driving a run from a config file

Everything Section 5 passes to the constructor and the `select_*()` calls
can live in one YAML (or JSON) file instead, loaded with `configure()`.
[`configs/benchmark/evaluation_example.yaml`](../../configs/benchmark/evaluation_example.yaml)
is a commented template with every key it reads:

```yaml
job_system: slurm          # slurm | native
progress_mode: bars        # bars | board | none
benchmark_root: /path/to/your/benchmark_root
# benchmark_eval_root / checkpoint_root / method_code_root: optional, default under benchmark_root
experiment_root: /path/to/where/scripts+results/go
experiment_name: my_first_run

datasets:                  # same items as select_datasets()
  - {task: MonoDepth, dataset: marigold}
  - {task: SfM, dataset: eth3d}
models:                    # same items as select_models(); one item per track
  - {name: VGGT}
  - {name: PI3_X, track: rgb_cameras}
  - {name: CUT3R, env: {SPATIAL_POLICY: shared}}  # per-model env overrides (below)
# runs: [...]              # optional: exact runs, replacing the datasets x models product

report:
  output_path: ./report    # export_report()'s default path
```

```python
from spatialarena import BenchmarkEvaluator

bench = BenchmarkEvaluator()
bench.configure("configs/benchmark/evaluation_example.yaml", experiment_name="my_second_run")  # kwargs override the file
validation = bench.validate()
if validation.ok:
    report = bench.evaluate()     # SLURM: then bench.wait(report)
    bench.export_report()         # writes to report.output_path
```

Precedence, highest first: `configure(**overrides)` > constructor
arguments > the config file > environment variables
(`BENCHMARK_AUX_ROOT`, `BENCHMARK_EVAL_ROOT`, `CHECKPOINT_ROOT`,
`METHOD_CODE_ROOT`, `BENCHMARK_PROGRESS_MODE`) > built-in defaults.
Datasets, models, runs and report settings in the file replace any earlier
selection. `max_in_flight` in the file is `evaluate()`'s default bound on
runs in flight (an explicit `evaluate(max_in_flight=...)` wins); `progress`
is not a config key.

Three keys change what the generated jobs do:

- `evaluate_only: true` submits only the evaluation jobs (CPU, no inference
  job), for runs whose predictions already exist. SfM and SLAM predictions
  made before per-frame depth export (pose-only) are scored too: both
  evaluators read only the poses. A VGGT-Long / Pi-Long case whose chunk
  alignment is still pending is aligned first and that newer prediction is
  scored.
- `reevaluate: true` scores every case again even when its evaluation file
  already has results (the engine's `REEVALUATE=1`) -- use it after an
  evaluator fix. Without it, a case with results is reused as is.
- `force_reinfer: true` bypasses the content-addressed prediction cache and
  re-runs inference even when a matching cached prediction already exists
  (the engine's `FORCE_REINFER=1`, which also deletes the stale shared
  output first) -- use it while testing an adapter/preprocessing change the
  cache's hash key doesn't capture (it hashes method, track, resolution and
  similar run config, not arbitrary adapter-internal code changes).
- `case_parallelism: N` runs N cases of a run at once (the engine's
  `CASE_PARALLELISM`; default 4 for inference, 6 for evaluation). Use 1 when
  concurrent cases run out of GPU memory, e.g. PI3_X `rgb_cameras` on long
  VideoDepth sequences (about 52 GB each on one H200). SLAM inference
  defaults to 2 on the short range and 1 on the long and large ranges, where
  a range admits a method only if its longest recording fits one H200 pass
  (four KITTI Odometry sequences at once ran ZipMap out of memory).
- `slurm_resources: {infer: {...}, eval: {...}}` replaces the engine's
  `#SBATCH` resource lines (`mem`, `cpus-per-task`, `time`, `gres`, ...) in
  the generated scripts, per phase. It can be set on the config, on a
  dataset item and on a model item; each level overrides the one before it
  option by option (config < dataset < model). The engine's eval default
  (320G memory) would leave one SLAM eval job at a time on the single cpu
  node, so `full_slam.yaml` asks 64G / 8 CPUs for SLAM
  evaluation and 128G / 16 for VGGT-Long and Pi-Long, whose eval job also
  runs the chunk alignment; `all_tracks.yaml` sets the same on its SLAM
  dataset items and those two models.

A model item's `env:` (as in `models:` above) sets extra environment
variables for just that model's generated scripts, merged over any
dataset/config-level `env:`. `SPATIAL_POLICY: shared` is the main use: on
the `depth_only`/`export_colmap` export path it restores the exported depth
map and camera intrinsics to the image's true source resolution instead of
leaving them at whatever padded/resized grid the model's own pipeline needs
internally (see `docs/benchmark/method_preprocessing.md` for which methods
support it and why a few don't). The default, `legacy` (or
omitting `env:` entirely), is each method's original, unchanged behavior.

These settings are written into the generated scripts as `export VAR=value`
lines, so `generate` shows exactly what will run. Jobs are always submitted
with plain `sbatch --export=NIL`: adding variables to it
(`--export=NIL,VAR=value`) intermittently leaves jobs held with
`user_env_retrieval_failed_requeued_held` on this cluster.

### Running a config from the command line

`spatialarena.run_config` drives the same steps from a shell, so every
job script comes from the pipeline rather than being written by hand:

```bash
python -m spatialarena.run_config configs/benchmark/full_mvrecon.yaml plan      # the planned runs, one per line
python -m spatialarena.run_config configs/benchmark/full_mvrecon.yaml generate  # write the scripts only, for review
python -m spatialarena.run_config configs/benchmark/full_mvrecon.yaml coverage  # generate, then compare with the canonical case lists
python -m spatialarena.run_config configs/benchmark/full_mvrecon.yaml submit    # generate and submit (max_in_flight from the config)
```

There is one config per task: `configs/benchmark/full_monodepth.yaml`,
`full_videodepth.yaml`, `full_mvrecon.yaml`, `full_sfm.yaml` and
`full_slam.yaml` (all under `configs/benchmark/`). Each lists every registered method on exactly the tracks
the task/track matrix allows for that task (for example no `rgb_cameras` for
SfM or SLAM, which would be given the poses they are scored on, and on
VideoDepth only for SpatialArena, the one dataset whose released GT has camera poses). Their
results still go to the suite report folders: MonoDepth, VideoDepth and
MVRecon to `full-depth`, SfM to `full-pose`, SLAM to `full-slam-<range>`.
`configs/benchmark/all_tracks.yaml` holds every task and track in one config (the same
1,064 runs as the five per-task configs together: 245 MonoDepth, 165
VideoDepth, 201 MVRecon, 192 SfM and 261 SLAM); there a method is listed
once per track and `plan()` pairs each track only with the tasks and
datasets it runs on. Scripts and SLURM logs go to
`<experiment_root>/<experiment_name>/`; results go to the canonical report
folders. `submit` with `max_in_flight` starts a detached dispatcher (see
Section 6a) that carries the config's `evaluate_only` and `reevaluate`.

## 6. Running the full matrix (all methods x all datasets)

```python
from spatialarena import BenchmarkEvaluator
from spatialarena.methods import METHOD_SPECS
from spatialarena.task_track_matrix import TASK_TRACK_MATRIX

bench = BenchmarkEvaluator(
    experiment_root="/path/to/experiments",
    experiment_name="full_standard_all_methods",
)

# Only select datasets that actually exist on disk.
datasets = bench.discover_datasets(tasks=list(TASK_TRACK_MATRIX.keys()))
existing = [r for r in datasets["records"] if r["exists"] and (not r["metadata_required"] or r["metadata_exists"])]
bench.select_datasets([{"task": r["task"], "dataset": r["dataset"]} for r in existing])

# Expand every method into one selector per track it supports -- plan()'s
# default cross product only uses ONE track per model selector (defaulting
# to "rgb"), so a multi-track method (PI3_X, MapAnything, OmniVGGT, DA3-NGL)
# needs one selector per track to get all of its runs planned.
model_items = [{"name": spec.name, "track": track} for spec in METHOD_SPECS for track in spec.tracks]
bench.select_models(model_items)

report = bench.validate()
if not report.ok:
    for issue in report.issues:
        if issue.level == "error":
            print(issue.message)
    raise SystemExit("fix errors first")

plan = bench.plan()
print(f"{len(plan.selected_runs)} runs planned")   # 537 with every dataset+method on this repo's cluster
```

**Before calling `evaluate()` on a plan this size**, read the warning in
Section 6a below -- a plan with dozens or hundreds of runs needs
`max_in_flight`, not the default submit-everything path.

### 6a. `slurm` vs `native`, and not flooding the queue

- **`job_system="slurm"` (the default):** `evaluate()` submits an
  inference job and a dependent evaluation job (`--dependency=afterok`) per
  run and **returns immediately** -- the report comes back with
  `summary["status"] == "submitted"` and job IDs, not results. Call
  `bench.collect(report)` later (poll it in a loop, or just call it again
  whenever you check back) to get a finalized report once the jobs finish;
  while anything's still queued/running, `collect()` returns the same
  submitted-state report unchanged.

  **Progress.** `bench.wait(report)` blocks with three progress bars --
  runs finished, inference cases, evaluation cases -- and returns the
  collected report when every run is done (`poll_interval` defaults to
  60 s; `timeout=` returns early). `bench.wait(report, progress="board")`
  shows the interactive terminal board instead: views by track, method,
  dataset, run and download, switched with `1`-`5` (`f` filters runs, `q`
  quits). Set it once with `BenchmarkEvaluator(progress_mode="board")`, the
  config key `progress_mode`, or `BENCHMARK_PROGRESS_MODE=board`; the board
  needs `pip install .[board]` (textual) and an interactive terminal. See
  "Progress Display" in the API reference below. Without blocking, `bench.progress(report)`
  returns a snapshot: each run's stage (queued -> inference -> evaluation ->
  done | failed) and finished/total cases per stage; `print(snapshot.render())`
  shows it as a table. Both also work for `max_in_flight` runs, including
  runs the dispatcher has not submitted yet. Case counts come from the
  per-case rows the engine writes (`summary_*.tsv`, `case_failures_*.tsv`)
  and the run's case file, stages from `sacct`.

  For anything more than a handful of runs, **don't call
  `evaluate()` with no arguments** -- that submits every run's job pair to
  the queue at once, which floods a shared cluster's scheduler (this
  happened during this pipeline's own development: 537 runs -> 1074 jobs
  submitted in one shot, which needed to be cancelled). Instead:

  ```python
  report = bench.evaluate(max_in_flight=24)
  ```

  This launches a **detached background process**
  (`spatialarena.dispatcher`) that keeps at most 24 runs' worth of jobs
  outstanding in the queue at any time, submitting the next run as soon as
  an earlier one finishes -- `evaluate()` still returns immediately, but the
  dispatcher keeps working independently (it survives your Python process
  exiting) and writes its state to
  `<experiment_root>/<experiment_name>/scripts/dispatcher_state.json`.
  Follow it with `bench.wait(report)` (progress bars, then the collected
  report), poll `bench.progress(report)`, or call `bench.collect(report)`
  once the dispatcher has finished.

- **`job_system="native"`:** `evaluate()` is genuinely synchronous -- it
  runs each run's infer/eval scripts as local `bash` subprocesses, in
  order, on whatever machine your Python process is running on, and
  returns a fully populated report directly (no `collect()` needed),
  showing progress while it runs -- the same bars, or the board with
  `evaluate(progress="board")` (`progress="none"` turns it off). This
  is what you want on a machine that isn't a SLURM cluster (e.g. a
  standalone box with your own GPUs), or to sanity-check the pipeline
  without touching a scheduler at all.

  **Native mode does not allocate compute for you.** If you're on a SLURM
  cluster but want to run natively (e.g. to avoid the two-job GPU/CPU
  split), you're responsible for making sure your Python process itself is
  already running on a compute node -- e.g. via an interactive allocation:

  ```bash
  salloc --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00 \
      srun python my_native_eval_script.py
  ```

  Never run native `evaluate()` directly on a login/head node -- the
  generated scripts do real model loading and inference.

## 7. Building a report

```python
result = bench.evaluate()        # or bench.collect(result) once SLURM jobs finish
bench.export_report("/path/to/where/you/want/the/report")
```

`export_report(output_path)` (the path defaults to the config file's
`report.output_path`, Section 5a) uses the evaluator's own last state
(`evaluate()`/`collect()`'s report, `validate()`'s report if you called it,
`plan()`'s plan) and writes:

```text
output_path/
  summary.json       # EvaluationReport.summary + ok + dataset_results
  plan.json          # EvaluationPlan
  validation.json    # ValidationReport, if validate() was called
  report/
    summary.tsv       # per-dataset aggregated metrics
    report.json        # the same visualization payload app/webpage reads
    report.pdf          # best-effort; a report_pdf_error.txt appears instead if it fails
```

It also (re)writes `<experiment_root>/<experiment_name>/scripts/
gather_metrics.sh` -- a standalone script, pre-filled with this run's own
report roots, that reruns the same three-script aggregation
(`summarize_benchmark_reports.py` -> `benchmark_analyzer.py` ->
`generate_benchmark_pdf_report.py`) later without needing `spatialarena`
or Python-level state at all:

```bash
bash /path/to/experiment_root/my_run/scripts/gather_metrics.sh
```

Run it again any time you want to refresh the report after more runs land
in the same report roots (e.g. after `collect()` finalizes more SLURM
jobs) -- `export_report()` only needs to be called from Python once to
generate it.

## 8. Reading `validate()` output

| Issue | Scope | Meaning | Fix |
|---|---|---|---|
| "No registered dataset spec for task=... dataset=..." | `dataset` | Typo, or a task/dataset combo that doesn't exist in `DEFAULT_DATASET_SPECS` | Check spelling against `discover_datasets()`'s output |
| "Dataset root does not exist for .../..." | `dataset` | Section 2 hasn't been done for that dataset yet | Download + materialize it |
| "Missing metadata.json for .../..." | `dataset` | Scene folders exist but no `metadata.json` | `generate_benchmark_metadata.py` (Section 2a) |
| "Unsupported model name or alias: ..." | `model` | Typo, or a method not in the registry | Check `discover_models()`'s `name`/`aliases`, or use the custom-adapter path (Section 9) |
| "... does not support track ..." | `model` | You asked for a track (`rgb_depth`/`rgb_cameras`) the method doesn't support | Drop the `track` key, or check `discover_models()`'s `tracks` field |
| "...'s checkpoint path does not exist." (error) | `model` | A local-file checkpoint is missing | Section 3 |
| "...'s checkpoint ... isn't cached ..." (warning) | `model` | An HF-hub checkpoint hasn't been fetched yet | Set `HF_HUB_OFFLINE=0` for the first run (Section 3) -- this is a warning, not a blocker, since it'll fetch on demand |
| "No (task, dataset, model, track) combination survived ..." | `run` | Everything you selected got filtered out by the task/track matrix | Check `TASK_TRACK_MATRIX` in `spatialarena/task_track_matrix.py` against what you selected |
| "Output path already has content; ..." (warning) | `run` | `<experiment_root>/<experiment_name>` isn't empty | Fine to ignore if you're intentionally resuming/reusing a run directory |

## 9. Adding a method that isn't in the registry

```python
bench.select_models([{
    "name": "MyCustomModel",
    "track": "rgb",
    "command": "python infer.py --data {data_path} --output {output_path}",
    # optional overrides -- see the API reference's "Custom methods":
    "code_path": "/abs/path/to/my/method/code",
    "ckpt_path": "/abs/path/to/my/checkpoint.pt",
    "resolution": 518,
    "env": {"MY_EXTRA_VAR": "value"},
}])
```

`command` is a template string; the engine substitutes `{data_path}`,
`{output_path}`, `{images_dir}`, `{depth_dir}`, `{sparse_dir}`,
`{camera_dir}`, `{method}`, `{track}`, `{code_path}`, `{ckpt_path}`,
`{resolution}`, `{image_load_resolution}`, `{pixel_limit}`,
`{conf_thresh}`, `{depth_conf_thresh}` (see
`scripts/inference/external_command_adapter.py`'s docstring for the full,
current list). Your command must write standardized artifacts into
`--output_path`: `depths/` for depth tasks, `sparse/0` (COLMAP format) for
pose/SfM tasks. `validate()` skips the checkpoint-resolvability check for
custom-adapter items entirely (it's your command's responsibility to find
its own checkpoint).

## 10. Releasing packaged datasets to Hugging Face (for maintainers)

`scripts/release/pack_and_upload_benchmark_datasets.py` packs this repo's
already-materialized `benchmark_root` datasets (MVRecon, MonoDepth, SfM,
VideoDepth, SLAM) into a staging directory and optionally uploads them to a
Hugging Face dataset repo, so someone else can download preprocessed data
directly instead of re-running Section 2's download+materialize steps
themselves. It never fabricates a license -- every packaged dataset folder
gets its own `LICENSE_NOTICE.md`, sourced from
`spatialarena/dataset_licenses.py`'s `SOURCE_LICENSES` registry (each
entry verified against that source's own official page).

```bash
# 1. See the plan first -- what's on disk, what license status each
#    dataset has, and what would be included/excluded. No files written.
python scripts/release/pack_and_upload_benchmark_datasets.py \
    --benchmark_root /path/to/benchmark_root \
    --staging_root /path/to/staging

# 2-3. Build the method manifests, write LICENSE_NOTICE.md + the dataset card,
#    stage the datasets (one .tar per scene by default) and upload -- very
#    large disk I/O and a real network upload: run it via sbatch, not on a
#    login node. Rerunning it resumes an interrupted upload.
sbatch --export=NIL scripts/release/upload_benchmark_release.sh
# overrides (UPLOAD=0 stages only; also REPO_ID, STAGING_ROOT, ARCHIVE=0 for hard links):
#   sbatch --export=NIL --wrap "env UPLOAD=0 bash scripts/release/upload_benchmark_release.sh"
```

Datasets whose original source has no confirmed, citable license/terms
document (`status="unconfirmed"` in `dataset_licenses.py` -- currently
`marigold` (its NYUv2 split), `sintel`, `imc`, `dtu`, `texturepoorsfm`,
`synth4k` and `aerialmetric` (its Wild frames)) are **excluded by default**, not silently
packaged -- pass `--include_unconfirmed` only after you've personally
verified that source yourself. Sources whose terms forbid redistribution
(`status="no_redistribution"`, currently ScanNet) are never packaged, not even
with `--include_unconfirmed`: a dataset that bundles one ships without the
scenes listed in `WITHHELD_SCENES`, its staged `metadata.json` drops them, and
its `LICENSE_NOTICE.md` says how to rebuild them (Marigold's `scannet` subset:
download Marigold's `scannet_val_sampled_800_1.tar` and run
`materialize_marigold_sample.py --scenes scannet --max_samples -1`, which adds
the scene to the existing `metadata.json`). Datasets with a confirmed but restrictive
license (`status="restricted_noncommercial"` -- KITTI, ETH3D, Bonn,
7-Scenes, Tanks and Temples, ScanNet++, HiRoom) are still packaged, but
their `LICENSE_NOTICE.md` states the restriction plainly; a public HF repo
does not change what the original source permits, and `scannetpp`
specifically requires written redistribution permission from TUM beyond
just holding your own signed data-use agreement -- get that in writing
before including it. Use `--datasets Task/dataset ...` to restrict to a
specific subset (e.g. `--datasets SfM/imc` after you've manually confirmed
IMC's per-scene license terms). Pairs that resolve to the same on-disk
data as another included pair (e.g. `SfM/eth3d` and `MVRecon/eth3d`) are
staged once and get a `SEE_INSTEAD.md` pointer on the duplicate side, so
the upload doesn't carry the same bytes twice.

This exact command (default dataset set, `--repo_id mvp-lab/SpatialArena-Dataset`)
is what produced the live release referenced in Section 2.0a. The MonoDepth extension datasets were added to it
on 2026-10-16 with `--datasets MonoDepth/aerialmetric MonoDepth/hammer MonoDepth/ibims1 MonoDepth/spring
MonoDepth/synth4k --include_unconfirmed` (Synth4K and AerialMetric by the maintainer's decision, with the papers,
GitHub and source links in their `LICENSE_NOTICE.md`). KITTI Odometry (`SLAM/kitti_odometry`, the 11 sequences at
504x154, CC BY-NC-SA 3.0) was added on 2026-10-17 with `--datasets SLAM/kitti_odometry --archive_scenes`, committed
on its own with the KITTI row merged into the existing card. Marigold (`MonoDepth/marigold`, NYUv2, KITTI, ETH3D
and DIODE: 2,531 frames, without the withheld ScanNet subset) was added on 2026-10-21 with
`--datasets MonoDepth/marigold --include_unconfirmed --archive_scenes` (NYUv2 by the maintainer's decision), with
its row merged into the existing card.

Two things to watch when adding datasets to an existing release:

- The staging folder's `README.md` is generated for the staged datasets only, and uploading it replaces the
  repo's dataset card, including its hand-written layout and methods sections. Merge the new rows into the
  current card (download it from the repo) before uploading, or leave `README.md` out of the upload.
- `--upload` (`HfApi.upload_folder`; `upload_large_folder` before huggingface_hub 2.x) can transfer every file and
  then fail its final commit repeatedly ("Failed to commit ... I/O error: failed to fill whole buffer"), retrying
  without progress. The data is already on the Hub's storage at that point, so stop it and commit each staged
  dataset folder on its own with
  `HfApi().upload_folder(folder_path=<staging>/<Task>/<dataset>, path_in_repo="<Task>/<dataset>", ...)`; the
  transfer is then skipped and each commit takes seconds.

## 5-minute smoke test

If you just want to confirm the pipeline itself works before investing in
downloading real datasets/methods, pick the smallest thing you already
have and force a tiny scene subset via `interval`:

```python
bench.select_runs([{
    "task": "MonoDepth", "dataset": "marigold", "model": "VGGT",
    "track": "rgb", "scenes": ["nyuv2"], "interval": 200,  # ~3 images instead of 654
}])
report = bench.evaluate()   # native: finishes directly; slurm: check back with collect()
```

This is exactly the pattern used to verify this pipeline's SLURM and
native execution paths during development (see `spatialarena/
dispatcher.py` and `test/test_evaluator.py` for the equivalent automated
tests).

### DIODE prediction clipping

For MonoDepth `marigold/diode`, the standard and GT-only protocol configuration
uses `post_clip_policy: valid_gt_max`: after alignment, each image's predicted
depth is clipped to `[0.001, max(GT[valid_mask])]` metres. The original GT mask
and the selected alignment solver are unchanged; this does not enable the
Marigold v2 DIODE gradient filter or log-space RANSAC alignment. Invalid GT
pixels do not contribute to the clipping maximum. Reports record the policy
and leave `post_clip_max` null because the upper bound varies by image.

Previously generated DIODE reports used a fixed 80-metre prediction ceiling.
Re-score existing predictions through YAML coverage, script review, and submit
before comparing them with results from this policy. No new inference is needed.

DIODE mask ablation: set `FFB_DIODE_MASK_POLICY: marigold_v2` in the YAML
model environment (or evaluator `--diode_mask_policy marigold_v2`). This
intersects the original mask with `0.6 < GT < 350` and excludes pixels where
`abs(sobel(GT, axis=0)) + abs(sobel(GT, axis=1)) > 0.3`, using float32 native
GT and SciPy's default Sobel boundary handling. The resulting mask is used
for both fitting and scoring. It affects only MonoDepth marigold/diode; the
default remains `original`. Reports record `diode_mask_policy`. This mask-only
ablation retains the selected alignment solver and is not the full V2 protocol.

## Appendix: API reference

The `BenchmarkEvaluator` contract the sections above use: the exact API shape, state rules, validation and run
results, configuration precedence, the task/track matrix, the artifact contract and the output layout. It is built on
the repo's own primitives: datasets come from `spatialarena.eval.dataset_registry`, scenes from canonical
`metadata.json` roots, methods write standardized artifacts, and evaluators stay artifact-only (they never import
external model code). Worked examples are in Sections 5, 5a and 9.

### Primary Objects

The public API is an evaluator object:

```python
from spatialarena import BenchmarkEvaluator, evaluator

bench = BenchmarkEvaluator()
bench.select_job_system("slurm")
bench.discover_datasets("$SPATIALARENA_ROOT/benchmarks")
bench.select_datasets([...])
bench.discover_models(model_types=[...])
bench.select_models([...])
bench.validate()
bench.evaluate()
bench.export_report("./report")
```

`evaluator` may remain a convenience singleton for short scripts and notebooks.
`BenchmarkEvaluator` is the implementation.

### Exact API Shape

The user-facing API should be stable and explicit:

```python
class BenchmarkEvaluator:
    def __init__(
        self,
        *,
        benchmark_root: str | Path | None = None,
        benchmark_eval_root: str | Path | None = None,
        method_code_root: str | Path | None = None,
        checkpoint_root: str | Path | None = None,
        experiment_root: str | Path | None = None,
        experiment_name: str | None = None,
        job_system: str = "slurm",
        case_parallelism: int | None = None,
    ) -> None: ...

    def reset(self) -> None: ...
    def select_job_system(self, type: str) -> None: ...
    def discover_datasets(
        self,
        path: str | Path | None = None,
        *,
        tasks: list[str] | None = None,
        datasets: list[str] | None = None,
        scenes: list[str] | None = None,
        include_scannet: bool = False,
    ) -> dict: ...
    def select_datasets(self, items: list[dict], *, append: bool = False) -> None: ...
    def discover_models(self, model_types: list[str] | None = None) -> dict: ...
    def select_models(self, items: list[dict], *, append: bool = False) -> None: ...
    def select_runs(self, items: list[dict], *, append: bool = False) -> None: ...
    def configure(self, config_file: str | Path, **overrides) -> None: ...
    def print_configuration(self) -> None: ...
    def plan(self) -> "EvaluationPlan": ...
    def validate(self) -> "ValidationReport": ...
    def evaluate(self, *, case_parallelism: int | None = None) -> "EvaluationReport": ...
    def collect(self, report: "EvaluationReport | None" = None) -> "EvaluationReport": ...
    def export_report(self, output_path: str | Path) -> Path: ...
```

### State Semantics

State should be predictable:

- `discover_*`, `validate()`, and `plan()` are read-only and must not mutate
  selections or resolved paths.
- `select_*` replaces prior selection by default.
- `append=True` adds to the current selection.
- `configure()` loads file values first, then applies `**overrides`, then
  leaves any existing explicit constructor values intact.
- `reset()` clears selection state but keeps the object reusable.
- `print_configuration()` prints the resolved state only; it should not mutate.

If a selector is repeated without `append=True`, the later value wins.
If a config file, constructor, and explicit call all set the same field, the
most local explicit value wins.

### Validation Results

`validate()` should return a structured report, not only text.

Suggested shape:

```python
@dataclass
class ValidationIssue:
    level: str            # "error" | "warning" | "info"
    scope: str            # "dataset" | "model" | "run"
    message: str
    path: str | None = None

@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue]
    summary: dict
    selected_runs: list[dict] | None = None
```

Rules:

- `ok` is true only when there are no `error` issues.
- warnings do not block evaluation.
- `validate()` should collect missing roots, missing metadata,
  unsupported aliases, missing checkpoints, and missing artifacts.
- `evaluate()` may call `validate()` implicitly, but explicit validation should
  remain available for dry runs and notebooks.

### Run Results

`evaluate()` should return a structured report:

```python
@dataclass
class EvaluationPlan:
    config: dict
    selected_datasets: list[dict]
    selected_models: list[dict]
    selected_runs: list[dict]
    output_root: Path

@dataclass
class EvaluationReport:
    ok: bool
    config: dict
    plan: EvaluationPlan
    scene_results: list[dict]
    dataset_results: list[dict]
    summary: dict
    report_dir: Path | None = None
```

The report should be machine-readable and stable enough to serialize as JSON.

Suggested JSON keys:

- `config`
- `plan`
- `validation`
- `scene_results`
- `dataset_results`
- `summary`
- `report_dir`
- `ok`

`summary` should include counts for selected datasets, selected models, selected
runs, evaluated scenes, passed scenes, failed scenes, and warnings.

### Method Semantics

#### `select_job_system(type)`

Preferred name. Legacy alias: `set_job_sys`.

- `slurm`: generate batch-job commands
- `native`: generate local commands

#### `discover_datasets(path, *, tasks=None, datasets=None, scenes=None, include_scannet=False)`

Preferred name. Legacy alias: `check_available_datasets`.

Returns structured discovery data instead of only printing it. Printing should
be optional and derived from the returned object.

Suggested top-level keys:

- `root`
- `selected_specs`
- `records`
- `missing`
- `summary`

#### `select_datasets(items, *, append=False)`

Preferred name. Legacy alias: `use_datasets`.

Each item should be one task/dataset selector:

```python
[
    {"task": "MonoDepth", "dataset": "marigold", "scenes": ["nyuv2"]},
    {"task": "VideoDepth", "dataset": "sintel"},
]
```

Rules:

- list input is required for multiple datasets
- each item must include `task` and `dataset`
- repeated selectors replace previous state unless `append=True`
- selectors may include `scenes`, `benchmark_root`, and `include_scannet`

#### `discover_models(model_types=None)`

Preferred name. Legacy alias: `check_available_models`.

Should return a registry view with:

- canonical names
- aliases
- supported tracks
- required paths/checkpoints
- adapter availability
- default resolution profile

#### `select_models(items, *, append=False)`

Preferred name. Legacy alias: `use_models`.

Each item should resolve one runnable model:

```python
[
    {"name": "VGGT"},
    {"name": "DA3-G"},
    {"name": "DepthLM", "track": "rgb"},
]
```

The model record may override:

- `code_path`
- `ckpt_path`
- `command`
- `track`
- `resolution`
- `image_load_resolution`
- `pixel_limit`
- `conf_thresh`
- `depth_conf_thresh`
- `env`
- `args`

For custom methods, `command` is the extension point:

```python
{
    "name": "MyCustomModel",
    "track": "rgb",
    "command": "python infer.py --data {data_path} --output {output_path}",
}
```

Required custom-adapter fields:

- `name`
- `track`
- `command`

Optional adapter fields:

- `code_path`
- `ckpt_path`
- `env`
- `args`
- `resolution`
- `image_load_resolution`
- `pixel_limit`
- `conf_thresh`
- `depth_conf_thresh`

The command template may use placeholders already supported by the repo's
external adapter path, including:

- `{data_path}`
- `{output_path}`
- `{images_dir}`
- `{depth_dir}`
- `{sparse_dir}`
- `{camera_dir}`
- `{method}`
- `{track}`
- `{code_path}`
- `{ckpt_path}`
- `{resolution}`
- `{image_load_resolution}`
- `{pixel_limit}`
- `{conf_thresh}`
- `{depth_conf_thresh}`

Track rules:

- if `track` is omitted and the model has exactly one supported track, use it
- if a model supports multiple tracks, a default track must exist in the registry
  or the selection is invalid
- `discover_models()` should expose the preferred/default track explicitly

### Configuration

`configure(config_file=..., **overrides)` should load YAML or JSON and fully
populate the current evaluator state.

Config precedence should be explicit:

1. explicit method arguments and constructor values
2. config file values
3. environment variables
4. built-in defaults

Minimal config:

```yaml
job_system:
  type: slurm
progress_mode: board   # bars (default) | board | none -- see Progress Display

benchmark_root: $SPATIALARENA_ROOT/benchmarks
experiment_root: ~/tmp
experiment_name: benchmark_full

datasets:
  - task: MonoDepth
    dataset: marigold
  - task: SfM
    dataset: imc

models:
  - name: VGGT
  - name: PI3_X
    track: rgb_depth

report:
  output_path: ./report
```

Recommended config keys:

- `benchmark_root`
- `checkpoint_root`
- `method_code_root`
- `experiment_root`
- `experiment_name`
- `job_system`
- `datasets`
- `models`
- `report`

`checkpoint_root` and `method_code_root` default to `benchmark_root/
_checkpoints` and `benchmark_root/_model_codes` respectively -- the
canonical, shared locations `download_all_supported_models()`/
`download_all_supported_method_code()` write to, and that generated SLURM
scripts read checkpoints/vendored method code from at run time (positional
args into the `_engine` templates, which is also where `HF_HOME` and each
method's own `*_CKPT_PATH`/`*_CODE_PATH` env-var defaults derive from). Env
var overrides: `CHECKPOINT_ROOT`, `METHOD_CODE_ROOT` (also read directly by
the generated `_engine` scripts themselves, independent of Python).

### Validation and Run Phases

The evaluator should expose separate phases:

1. discover
2. select
3. validate
4. plan
5. evaluate
6. export

`validate()` should be cheap and actionable. It should report:

- missing dataset roots
- missing metadata
- unsupported tasks or dataset aliases
- missing checkpoints
- unsupported model tracks
- missing prediction artifacts
- output-path conflicts

`plan()` should resolve the final dataset/model cross product or explicit run
list and return the concrete work items without executing them. It must be
read-only.

`evaluate()` should assume validation has passed, but still fail with a clear
error if the on-disk state changed.

### Async SLURM Semantics

`evaluate()`'s signature implies a synchronous call returning a finished
report, but a SLURM job can queue for hours under cluster contention having
nothing to do with whether the run itself is healthy. The two job systems
have different semantics as a result:

- `job_system="native"`: `evaluate()` is genuinely synchronous. It runs the
  work locally and returns a fully populated `EvaluationReport` directly.
- `job_system="slurm"` (the default -- see Model/Job Semantics): `evaluate()`
  submits the job chain (an inference job, then an evaluation job depending
  on it via `afterok`, mirroring the split established for this pipeline)
  and returns **immediately** with an `EvaluationReport` whose `summary`
  marks it as submitted-but-not-finished and whose `plan` carries the
  submitted job IDs. It does not block.
- `collect(report=None)` polls SLURM (`sacct`) for the job IDs on the given
  report (or the evaluator's last report, if omitted). While jobs are still
  running or queued, it returns the same submitted-state report unchanged.
  Once all jobs reach a terminal state, it reads the resulting per-case
  artifacts and returns a finalized `EvaluationReport` with
  `scene_results`/`dataset_results`/`summary` populated, exactly as
  `evaluate()` would have populated them natively.
- A caller who wants to block calls `wait(report=None, poll_interval=60,
  timeout=None, progress=None)` explicitly: it polls until every run has
  finished, showing progress per pipeline stage, then returns `collect()`'s
  finalized report. The blocking wait stays out of `evaluate()`.
- `progress(report=None)` returns a non-blocking `PipelineProgress` snapshot
  (`spatialarena/progress.py`): each run's stage (`queued` -> `inference`
  -> `evaluation` -> `done` | `failed`) and finished/total cases per stage.
  Native `evaluate()` shows progress while it runs.

### Progress Display

Long operations -- native `evaluate()`, `wait()`, and the `download_all_*()`
calls (plus `download_all_supported()`, which fetches method code,
checkpoints and datasets in one go) -- show progress in one of three modes
(`spatialarena/progress_ui.py`):

- `"bars"` (default): tqdm bars -- runs finished / inference cases /
  evaluation cases, or one bar per download kind. Fine in logs.
- `"board"`: an interactive full-screen terminal board
  (`spatialarena/progress_board.py`, needs `textual`: `pip install
  .[board]`). Views: `1` Tracks (task | track | SLAM range), `2` Methods,
  `3` Datasets, `4` Runs (`f` cycles all / active / failed / done; the
  highlighted run's jobs, report dir and notes below), `5` Downloads (each
  dataset / checkpoint / code item with its current step and failure
  reason). `q` closes it; while work is running it asks for a second `q`.
  For `wait()` that stops watching (the SLURM jobs keep running); for native
  evaluation and downloads the work continues and its progress switches to
  bars. Without an interactive terminal or `textual` it falls back to
  `"bars"` with a warning.
- `"none"`: no progress output.

The mode is chosen per call (`progress="board"`), else by
`BenchmarkEvaluator(progress_mode=...)` / the config file's `progress_mode`
key, else the `BENCHMARK_PROGRESS_MODE` environment variable, else
`"bars"`.

### Generated Scripts

`evaluate()` does not run work from a transient temp path. Before
submitting (or running, for `job_system="native"`) anything, it renders the
resolved plan against this package's shell templates and writes the
concrete, standalone scripts to:

```text
<experiment_root>/<experiment_name>/scripts/
  run_<method>_<suite>_<track>_infer.sh
  run_<method>_<suite>_<track>_eval.sh
  gather_metrics.sh
```

These are real, directly re-runnable scripts -- not documentation of what
ran. `sbatch`/`subprocess.run` invoke these exact files, so a user can hand
the run directory to someone else (or come back to it later) and resubmit
or inspect the same scripts that produced the results, without importing
`spatialarena` at all. `gather_metrics.sh` regenerates
`summary.tsv`/`results.json`/`artifacts.json`/`report.pdf` for that run's
own report roots on demand (equivalent to hand-chaining
`summarize_benchmark_reports.py` -> `benchmark_analyzer.py` ->
`benchmark_artifact_analyzer.py` -> `generate_benchmark_pdf_report.py`).

### Dataset Semantics

Supported canonical task families:

- `MonoDepth`
- `VideoDepth`
- `MVRecon`
- `SfM`
- `CameraPose`
- `SLAM`
- `WeakTextureSfM`

Dataset selection should use the existing registry and accept task aliases.
Dataset support is defined by canonical benchmark roots with `metadata.json`
and per-scene folders.

#### Task/Track Compatibility Matrix

Not every (task, track) combination is meaningful. A track conditions the
model on extra ground-truth input (camera poses for `rgb_cameras`, depth
priors for `rgb_depth`, depth priors with half of every map dropped for
`rgb_masked_depth`, and those plus the camera poses for
`rgb_masked_depth_cameras`); a combination is only worth evaluating if the task
still has something nontrivial left to predict once that input is given.
This is the single source of truth `plan()` filters the dataset x model
cross product against:

```python
TASK_TRACK_MATRIX: dict[str, tuple[str, ...]] = {
    "MonoDepth": ("rgb", "rgb_masked_depth"),
    "VideoDepth": ("rgb", "rgb_cameras", "rgb_masked_depth", "rgb_sparse_depth"),
    "MVRecon": ("rgb", "rgb_cameras", "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras",
                "rgb_sparse_depth", "rgb_sparse_depth_cameras"),
    "SfM": ("rgb", "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras", "rgb_sparse_depth",
            "rgb_sparse_depth_cameras"),
    "SLAM": ("rgb", "rgb_depth", "rgb_masked_depth"),
}
```

Rationale per task:

- `MonoDepth` is single-frame: there is no multi-view relationship for a
  camera pose to condition on (`rgb_cameras` excluded), and being given
  depth while predicting depth is circular (`rgb_depth` excluded too).
  Inference enforces it: multi-view models predict each MonoDepth image on its own
  (`BENCHMARK_SINGLE_VIEW`, see the Frame Selection section of `docs/evaluation.md`).
- `VideoDepth` is multi-frame, so a camera-pose prior is meaningful
  (`rgb_cameras` included), but it runs only on SpatialArena, the one dataset
  whose released GT has camera poses (`sparse/0`; `TRACK_DATASETS`): KITTI has
  only the OXTS vehicle trajectory in the IMU frame (no IMU-to-camera
  calibration), and the released GT has no Sintel or DyCheck cameras. Being
  given depth while predicting depth is circular (`rgb_depth` excluded).
- `MVRecon` fuses pose and depth into one reconstruction, so having either
  piece given (or neither) is still a meaningful evaluation -- all three
  tracks apply. `rgb_depth` hands the method the dataset's GT depth, so it
  runs only on datasets whose GT depth is sparse or incomplete (DTU, ETH3D,
  7-Scenes); on dense GT depth (HiRoom, ScanNet++) only the poses would be
  left to predict, and NRGBD / Tanks and Temples have no depth prior
  (`TRACK_DATASETS` in `spatialarena/task_track_matrix.py`).
- `SfM`/`SLAM` predict camera poses: being given camera poses as input and
  then scoring pose-recovery accuracy is circular (`rgb_cameras` excluded);
  a depth prior is a legitimate aid to pose estimation (`rgb_depth`
  included).
- The masked-depth tracks are depth completion: the method is given only the
  kept half of each GT depth map (`spatialarena/depth_prior_masking.py`:
  32 cells along the longer side, a fixed half dropped per image) and is
  scored on every GT-valid pixel, so they are not circular on any task and
  run on every dataset with depth priors, MonoDepth and VideoDepth included.
  `rgb_masked_depth_cameras` adds the GT cameras on MVRecon and SfM (RGB +
  depth + camera conditioning); on SfM it measures how faithfully the
  given poses come back out rather than pose estimation.
- The sparse-depth tracks give real sparse depth: KITTI's single-scan LiDAR
  (VideoDepth) and each image's COLMAP keypoints (MVRecon/SfM datasets whose
  model has 3D points), `spatialarena/sparse_depth.py`. They run only where
  such a source exists (`TRACK_DATASETS`); on SfM the points come from the
  scored reconstruction, so those results are SfM-assisted.

A (task, track) pair is only ever planned or run if the track is in both
this table's entry for the task **and** the selected model's own supported
`tracks`. Most registered models only support `rgb` at all, so the matrix
only actually widens the plan for the handful of models with extra tracks
(see Method Track Support below).

Two further filters narrow the plan:

- **Method tasks.** A `MethodSpec` with `tasks` takes part only in those tasks.
  The long-sequence SLAM systems (`VGGT_LONG`, `PI_LONG`, `VGGT_SLAM`) are
  SLAM-only.
- **SLAM range tracks.** `SLAM` is offered only to the methods of a SLAM range
  track (`spatialarena/slam_protocol.py`), and `plan()` emits one work item
  per range the method is in (`slam_range: short | long`; an explicit
  `slam_range` on the dataset or model item narrows it). Methods excluded from
  SLAM, such as MapAnything, MASt3R and the depth-only VLMs, get no SLAM work. See
  `docs/benchmark/slam_protocol.md`.

### Model Semantics

Model selection should support three modes:

1. registered canonical name
2. registered alias
3. custom command adapter

This is the extension boundary for new neural models. The evaluator should not
need model internals as long as outputs match the artifact contract.

#### Method Track Support

Track support is per-method, declared in the registry, and most methods
only support `rgb`. As of this writing: `PI3_X`, `MapAnything`, and
`OmniVGGT` support all seven tracks (`rgb`, `rgb_cameras`, `rgb_depth`,
`rgb_masked_depth`, `rgb_masked_depth_cameras`, `rgb_sparse_depth`,
`rgb_sparse_depth_cameras`); `DA3-NGL` supports `rgb` and
`rgb_cameras`; every other registered method is `rgb`-only. `discover_models()` should expose each model's supported
tracks intersected with the Task/Track Compatibility Matrix above, so a
caller can see directly which (task, track) pairs a given model will
actually produce a plan item for.

If `select_models()` receives more than one model and no explicit `select_runs()`
is provided, evaluation should use the cartesian product of selected models and
selected datasets, filtered by task/track compatibility.

### Run Selection

By default, evaluation uses the cartesian product of selected datasets and
selected models, subject to task/track compatibility. If the user wants a
specific pairing, they must provide `select_runs()` entries instead of relying
on the default product.

`select_runs(items, *, append=False)` accepts explicit run items such as:

```python
[
    {"task": "MonoDepth", "dataset": "marigold", "model": "VGGT"},
    {"task": "SfM", "dataset": "eth3d", "model": "DA3-G"},
]
```

Run item schema:

- required: `task`, `dataset`, `model`
- optional: `scenes`, `track`, `benchmark_root`, `experiment_root`,
  `experiment_name`, `job_system`, and adapter overrides that are already
  valid on `select_models()` items

Rules:

- every run item must identify one dataset and one model
- `select_runs()` replaces prior run selection by default unless `append=True`
- if `select_runs()` is used, it overrides the default cartesian product
- run items may include scene filters and adapter overrides when needed

### Artifact Contract

The evaluator consumes standardized outputs only:

```text
EXPERIMENT_ROOT/EXPERIMENT_NAME/TASK/DATASET/METHOD/SCENE/
  depths/
  sparse/0/
  points3D_dense.ply
  metrics.json
```

Rules:

- depth tasks read `depths/`
- pose tasks read `sparse/0`
- geometry tasks may read `points3D_dense.ply`
- report export reads `metrics.json` and per-scene results

### Metrics

Each scene result's `metrics` dict is keyed by task family. A dataset that
doesn't produce a given key (rather than producing it as `null`) means that
metric genuinely wasn't computed for that scene -- callers should not
assume every key in a family is always present.

- **Depth** (`MonoDepth`/`VideoDepth`): `abs_relative_difference`,
  `squared_relative_difference`, `rmse_linear`, `rmse_log`, `log10`,
  `delta103_acc`, `delta125_acc`, `delta125_2_acc`, `delta125_3_acc`,
  `i_rmse`, `silog_rmse`, and their `_percent` display variants. A smaller
  subset of scenes also produce point-cloud-derived geometry metrics
  (`mae_normal_deg`, `point_accuracy`, `point_completion`, `point_chamfer`,
  `point_f1`) -- these require camera intrinsics, which are only resolved
  for scenes with a materialized intrinsics source (dataset manifest,
  `sample.json`, dataset metadata, or a known per-dataset fallback); most
  VideoDepth datasets currently have none of those, so these keys are
  commonly absent there. The dashboard's depth tables show `mae_normal_deg`
  only; the `point_*` metrics stay in the evaluation files.
- **MVRecon**: `accuracy`, `completion`, `overall` always; `precision`,
  `recall`, `fscore` for datasets whose reconstruction protocol defines
  them (most datasets) -- **except `dtu`**, whose official MVSNet protocol
  only ever produces `accuracy`/`completion`/`overall` (Chamfer distance);
  its `precision`/`recall`/`fscore` here are an added-for-parity extension
  at a 1mm threshold, not part of DTU's official benchmark, so a `dtu` row
  missing those three keys is complete, not broken.
- **SfM/SLAM**: `ate_rmse`, `ate_mean`, `ate_median`,
  `relative_rotation_error_deg_mean/median`,
  `relative_translation_error_deg_mean/median`,
  `relative_pose_error_deg_mean/median`, `RRA_at_{5,15,30}`,
  `RTA_at_{5,15,30}`, `mAA_30`, `auc_{3,5,10,20,30}`.

### Output Layout

The implementation should keep a stable output layout:

```text
<experiment_root>/<experiment_name>/
  <task>/<dataset>/<method>/<scene>/
    depths/
    sparse/0/
    points3D_dense.ply
    metrics.json
  summary.json
  validation.json
  report/
    report.html
    report.pdf
```

Notes:

- `summary.json` is the run-level machine-readable summary.
- `summary.json` should mirror `EvaluationReport.summary` plus top-level counts.
- `validation.json` should mirror `ValidationReport` as JSON.
- `plan.json` should mirror `EvaluationPlan` as JSON.
- `report/` is the user-facing export directory.
- HTML and PDF are both optional, but the directory should be consistent.

### Report Schema

The JSON-facing schema should be stable:

- `ValidationReport` JSON keys: `ok`, `issues`, `summary`, `selected_runs`
- `EvaluationPlan` JSON keys: `config`, `selected_datasets`, `selected_models`,
  `selected_runs`, `output_root`
- `EvaluationReport` JSON keys: `ok`, `config`, `plan`, `scene_results`,
  `dataset_results`, `summary`, `report_dir`

Each `issue` entry should include `level`, `scope`, `message`, and optional
`path`.

### UX Rules

- default to the simplest happy path
- print a concise summary before execution
- make every failure message actionable
- show resolved paths, not guessed paths
- avoid requiring shell commands for standard use
- keep notebook use safe by avoiding hidden global state where possible
