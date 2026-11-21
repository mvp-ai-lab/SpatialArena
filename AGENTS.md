# AGENTS.md

Guidance for coding agents and contributors working in `SpatialArena`, the benchmark evaluation repository
(split out of FeedForward-Benchmark, which keeps dataloaders and training-data conversion).

## Scope

This file applies to the full repository unless a more specific `AGENTS.md` is added in a subdirectory. Follow these instructions for code, scripts, documentation, tests, and dataset tooling in this project.

## Repository layout

- `spatialarena/`: importable Python package: the evaluator front end (`evaluator.py`, `run_config.py`,
  `script_generator.py`, `dispatcher.py`), the method registry (`methods.py`), task/track rules
  (`task_track_matrix.py`, `slam_protocol.py`, `sparse_depth.py`, `depth_prior_masking.py`), downloads
  (`download.py`), and the Slurm engine templates (`templates/_engine/`).
- `spatialarena/eval/`: benchmark dataset adapters, result layouts, metrics, alignment and evaluation entry points.
- `spatialarena/geometry/`, `spatialarena/utils/`: geometry and point/depth helpers used by evaluators and adapters.
- `scripts/inference/`: model adapters that write prediction artifacts.
- `scripts/preprocess/`: benchmark dataset materializers, layout normalizers and verifiers.
- `scripts/download/`: benchmark dataset download scripts (wrapped by `spatialarena/download.py`).
- `scripts/test/`: case lists, Slurm submission helpers, report summaries and smoke tests.
- `scripts/release/`: Hugging Face benchmark release packing and upload.
- `configs/benchmark/`: run configs for `python -m spatialarena.run_config`.
- `app/webpage/`: the SpatialArena results dashboard and artifact viewer. This is the public copy;
  FeedForward-Benchmark keeps an internal copy (`app/webpage` there) with features this one drops. That
  copy imports `spatialarena` from this checkout, so a change to code it imports (collector inputs, viewer/demo
  assets) must keep it working too.
- `docs/`: evaluation protocol, benchmark notes and the evaluation API guide.
- `test/`: pytest tests.

## Coding style

- Use Python 3.10+ compatible code.
- Keep line length at or below 120 characters, matching `.pylintrc`.
- Prefer explicit, typed function signatures for public APIs and dataset/conversion helpers.
- Use `pathlib.Path` for filesystem paths in new Python code unless an existing local API expects strings.
- Avoid broad rewrites. Keep changes scoped to the requested dataset, loader, evaluator, or script.
- Preserve existing naming conventions for datasets, splits, shard names, and metadata keys.
- Prefer structured parsers (`json`, `csv`, `yaml`, `tarfile`, `numpy`) over ad hoc string parsing.
- Do not hard-code personal scratch paths in reusable logic. If cluster defaults are needed, expose them as CLI defaults or environment-overridable variables.
- New source files should start with the project copyright and SPDX header:

  ```python
  # Copyright (c) 2026 Yu Chen. All rights reserved.
  # SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
  ```

  For shell scripts, use the same text as comments after the shebang.

## Linting and tests

Use the repository root as the working directory:

```bash
cd ~/Projects/SpatialArena
```

Recommended checks before committing code changes:

```bash
python -m py_compile <changed_python_file.py>
python -m pylint spatialarena scripts test
python -m pytest test
```

A pre-commit hook enforces the first two (py_compile + pylint, on staged files only, so a commit
is never blocked by a pre-existing issue elsewhere) automatically. `.pylintrc` disables the
Convention/Refactor/Warning categories repo-wide, so a plain `pylint` run only enforces Error/
Fatal issues; the hook also checks AGENTS.md's 120-character line limit manually, since
line-too-long is a Convention-category message pylint itself won't flag. One-time setup per
clone (not tracked by git itself):

```bash
git config core.hooksPath .githooks
```

For focused changes, run the narrowest relevant test first, then broaden when the change touches shared behavior:

```bash
python -m pytest test/test_sparse_reconstruction_evaluator.py
```

Test expectations:

- Metric/evaluation changes must include a deterministic synthetic test or a small fixture-based regression test.
- Slurm/download wrapper changes should pass `bash -n <script.sh>`.
- Do not require large private datasets for unit tests. Use tiny synthetic fixtures or smoke tests with explicit opt-in flags.

## Evaluation protocol

Evaluation code lives in:

```text
spatialarena/eval/
```

When editing evaluation logic:

- keep input and output layouts compatible with `result_layout.py` and `prediction_artifacts.py`;
- write per-frame prediction maps (depths/, confidences/, raw_disparities/, raw_log_depths/) only through
  `adapter_utils.save_prediction_map` / `export_depth_predictions`: compressed `<stem>.npz` (depth float32, confidence
  float16), never a raw `.npy`;
- record metrics in machine-readable JSON when possible;
- include enough metadata to reproduce dataset split, scene list, alignment mode, and prediction source;
- for pose metrics, report both absolute and relative errors when supported;
- when alignment is used, document whether it is SE(3), Sim(3), depth-assisted, ICP-refined, or selected by best validation metric;
- visualization exports should not be required for metric computation, but should be available for debugging failed scenes.

## Slurm submission protocol

Formal (standard-pipeline) evaluation -- full suites, retries, re-scoring -- must use the pipeline to generate
its job scripts. Do not write your own sbatch scripts or submit the engine scripts by hand for it:

- Describe what to run in a YAML config (tasks/datasets, methods/tracks, and keys such as `evaluate_only`,
  `reevaluate`, `case_parallelism`, `max_in_flight`, `slurm_exclude`); see `configs/benchmark/full_*.yaml`,
  `configs/benchmark/all_tracks.yaml` and `docs/benchmark/evaluator_quickstart.md`. One-off configs go under
  `<benchmark root>/_eval_job_scripts/configs/`.
- `python -m spatialarena.run_config <config.yaml> coverage` generates the scripts into
  `<benchmark root>/_eval_job_scripts/<experiment_name>/scripts/` (never `/tmp` or home) and checks them against
  the canonical case lists; review them, then `submit`. The agent's job is to submit, track the jobs, and fix
  issues in the pipeline code (then regenerate), not to patch generated scripts.
- A setting the jobs need goes into the config, which writes it into the generated scripts as `export VAR=value`.
  Submit with plain `sbatch --export=NIL`; `--export=NIL,VAR=value` intermittently leaves jobs held with
  `user_env_retrieval_failed_requeued_held` on this cluster.

The engine pair in `spatialarena/templates/_engine/` (what the generated scripts wrap) can be submitted directly
only for debugging or a one-off test outside the canonical report folders:

- Inference: `sbatch --export=NIL spatialarena/templates/_engine/run_method_benchmark_infer.sh <method> <suite>
  [<track> <run_id>]`, then evaluation with `run_method_benchmark_eval.sh` and the same arguments
  (`--dependency=afterok:<infer job>`). Full suites always write to one report folder per suite and method
  (`spatialarena/report_layout.py`); reruns overwrite in place and the engine refuses (exit 8) any other
  `REPORT_DIR` inside `reports/` except one-off test folders `reports/_<name>/`. Never create per-run
  (`*_retry_*`, `*_fix_*`) report folders.
- Both wrappers `exec` `run_method_case_list.sh` (a snapshot copied per job) with `CASE_PHASE=inference` or
  `evaluate`; keep that order stable.
- `scripts/test/engine_pair.sh` wraps this: `submit_engine_pair` (two jobs, used by `submit_method_case_jobs.sh`)
  and `run_engine_pair` (both phases inside one GPU job, used by `run_failed_cases_from_tsv.sh`). CPU-only work (evaluation, SLAM chunk alignment, report maintenance) belongs on
  `--partition=cpu`; `engine_pair.sh` still defaults evaluation jobs to `--partition=gpu --gres=none` from when the
  cpu partition was drained (override with `ENGINE_EVAL_SBATCH_ARGS`); check `sinfo -p cpu` first.
- `scripts/test/run_resolution_consistency_smoke.py` runs each registered method's real adapter
  CLI entry point (real checkpoint, real vendored code, CPU or GPU) against a tiny synthetic
  scene and checks the exported depth/confidence/COLMAP intrinsics land back at the scene's true
  source resolution -- use it after touching `--spatial_policy`/preprocessing code in any
  adapter; mocked adapter tests alone have missed real bugs (missing view-dict keys, unhandled
  tensor shapes) that only show up running the actual model forward pass.
- Do not add extra Slurm flags such as custom `--time` or benchmark-specific environment toggles to a
  retry path unless the change is explicitly documented and required.
- When fixing a failed batch, change only the minimum needed inputs: method, suite, track, checkpoint path,
  or case list.
- If a retry needs a different wrapper, document the reason in the repo before resubmitting.

## Documentation expectations

When changing user-facing behavior, update the relevant docs under `docs/`:

- downloader usage;
- conversion usage;
- evaluation API, metrics, and expected prediction format;
- dataset statistics and storage status when dataset inventory changes.

Prefer prose with concrete examples over large tables when explaining APIs or dataset behavior.

## Storage and data safety

- Treat `~/Data4D` as the canonical converted WebDataset location.
- Treat `$SPATIALARENA_ROOT` as the preferred raw/auxiliary dataset location.
- Do not remove raw data, shards, checkpoints, logs, or symlinks used by active Slurm jobs without checking the live job commands.
- Before deleting or moving dataset paths, check `squeue`, relevant config files, and resolved symlink targets with `readlink -f`.
- Do not replace real dataset directories with compatibility symlinks unless the user explicitly approves that migration strategy.

## Privacy: no user names

Scripts, docs, code, configs, data files we publish (e.g. Hugging Face metadata and manifests) and commit history
must not contain a user name or a path that exposes one (e.g. `/home/<user>/...`, `_scratch/<user>/...`).

- Shell: use `$HOME` and `${USER:-$(id -un)}`; scripts run with `sbatch --export=NIL` get no `HOME`/`USER`, so set
  `HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}` before using it. In `#SBATCH` lines use `%u`
  (e.g. `--output=/home/%u/tmp/<job>_%j.log`); `#SBATCH` lines do not expand variables.
- Python: `os.path.expanduser("~/...")` / `Path.home()` and `getpass.getuser()`, never a literal user path.
- Data locations: derive defaults from `SPATIALARENA_ROOT` (`spatialarena/paths.py`; shell
  `${SPATIALARENA_ROOT:-$HOME/spatialarena}`, configs `${SPATIALARENA_ROOT}/...`), never a cluster mount path.
- The dashboard's committed `data/results*.json` are public: strip them with
  `app/webpage/python/public_results.py` (or collect with `--public`).
- Docs: `~`, `$HOME`, `/path/to/...` or repo-relative paths. Dataset manifests store paths relative to the scene or
  dataset folder, not absolute ones.
- Commit messages, PR text and co-author lines must not name a user account; check `git grep -n "<user>"` and the
  staged diff before committing.

## Commit message tags

Use a bracketed tag at the start of each commit title. Keep the title specific and action-oriented.

Recommended tags:

- `[FEAT]`: new user-facing feature, dataset support, loader, metric, or API.
- `[FIX]`: correction to existing behavior or compatibility issue.
- `[BUG]`: bug reproduction, failing-test addition, or bug-focused diagnostic commit.
- `[SCRIPT]`: download, conversion, Slurm, maintenance, or one-off operational script changes.
- `[DATA]`: dataset metadata, manifests, statistics, storage inventory, or data-manager content updates.
- `[DOC]`: documentation-only updates.
- `[TEST]`: test-only additions or refactors.
- `[LOG]`: logging, metrics recording, progress reporting, or diagnostics.
- `[PERF]`: speed, memory, I/O, or scalability improvements without behavior change.
- `[REFACTOR]`: internal restructuring without intended behavior change.
- `[LICENSE]`: license headers, NOTICE, dependency/license metadata.
- `[REL]`: release/version/changelog updates.

Examples:

```text
[FEAT] Add SpatialVID WebDataset conversion with optional HQ depth
[FIX] Handle degenerate Sim3 alignment in driving pose evaluation
[SCRIPT] Add resumable MatrixCity normal conversion Slurm wrapper
[DOC] Document the SLAM range tracks in the evaluation protocol
[DATA] Update public dataset storage inventory
```

## Agent workflow

Before editing:

1. Inspect the relevant files and local patterns.
2. Check the current git status and avoid overwriting unrelated user changes.
3. For migrations or destructive operations, verify live jobs and resolved paths.

Before finishing:

1. Run focused tests or syntax checks relevant to the change.
2. Report exactly what changed and what was not tested.
3. If tests are skipped, state the concrete reason.
4. Do not claim a benchmark dataset is ready unless verified at the filesystem level (verify_benchmark_datasets.py).
