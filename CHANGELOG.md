# Changelog

SpatialArena changes. The history before the split is in FeedForward-Benchmark's CHANGELOG.md.

## Unreleased

### Changed

- The Hugging Face dataset moves to `mvp-lab/SpatialArena-Dataset` (from `Bluefish2333/SpatialArena-Dataset`): the
  dashboard links and download commands, the download and release scripts' default repository, the dataset license
  homepage, and the benchmark docs use the new repository.
- `scripts/slurm/pack_and_upload_benchmark_datasets.sh` is removed: `scripts/release/upload_benchmark_release.sh`
  does the same stage + upload (plus the method manifests and per-scene archives) and the quickstart now uses it.
  `aggregate_full_standard_reports.sh` and `build_come_extensions.sh` default to the `si_bench` env, not `dogs`.
- Lint: no pylint warnings repo-wide either (with W enabled; the repo config gates E/F). Real fixes: tab indentation
  in `7scenes_preprocessing.py`, `fuse_depth_map_dtu.py` / `eval_tnt_mesh.py` run in `main()` (no argument parsing on
  import), unused imports and variables, self-assignments, chained re-raises, `cv2` fallbacks catching `ImportError`
  only, and single-element unpacking in tests checked by `only()`. Deliberate patterns (best-effort broad excepts,
  tests reaching private members, mocks keeping real signatures) carry inline disables. No behavior changes.
- `scripts/inference/` drops two unused files: `official_depth_utils.py` (a staging helper no adapter imports) and
  `unavailable_method.py` (a not-runnable stub no registry entry or engine branch uses).
- `scripts/inference/da3_mipnerf360.sh` and `da3_tanks_and_temples.sh` are removed: unused one-off loops that wrote
  DA3 COLMAP models into each dataset's own `sparse/0` (a 3DGS experiment, not a benchmark run).
- `docs/design_language.md` is the "API reference" appendix of `docs/benchmark/evaluator_quickstart.md` (its
  refactor-planning sections and duplicate examples dropped); every citation in the code points there. README's SLAM
  case-list example covers the large range, and its outdated note that MVRecon metrics are not implemented is gone.
- `scripts/preprocess/` drops four unused files: `fuse_7scenes_gt_mesh.py` (its all-sequence mesh was never scored),
  and the sbatch wrappers `sync_benchmark_data_doc.sh`, `finalize_7scenes_layout.sh` (the quickstart lists its two
  commands) and `consolidate_texturepoorsfm.sh` (the one-time v2 split build; `consolidate_texturepoorsfm.py` stays).
- The old CSV pipeline is removed: `run_benchmark.py` and its wrappers `scripts/run_benchmark_dtu.sh` /
  `scripts/run_benchmark_eth3d.sh` (and the two tests of its command builders). DTU and ETH3D run through
  `run_config` with the full-suite configs; the engine builds the adapter commands.
- `docs/benchmark/`: the four depth-method notes (`native_depth_alignment.md`, `marigold_versions.md`,
  `point_query_grid_protocol.md`, `videodepth_diffusion_baselines.md`) are one `depth_methods.md`, and
  `mvrecon_scoring.md` is a section of `docs/evaluation.md` (MVRecon scoring); links updated. The MindCloud scene
  selection is referenced as FeedForward-Benchmark's `docs/mindcloud_benchmark_scene_selection.md`.
- `docs/benchmark/plans/output_resolution_alignment.md` (a finished plan) is removed; `method_preprocessing.md` is
  the per-method reference it pointed to.
- Tests: `test/` goes from 91 to 42 files. Small files on one subject are merged (e.g. eleven depth-alignment-protocol
  files into `test_depth_alignment_protocols.py`, six MVRecon/DTU files into `test_mvrecon_evaluator.py`, the dataset
  materializers into `test_materializers.py`, the per-method adapters into `test_model_adapters.py`); every test is
  kept (two renamed: `test_materialize_{nrgbd,tum_dynamics}_end_to_end`) except one duplicate protocol check, now a
  case of `test_protocol`. `scripts/test/` drops 28 unused files: the per-method `run_da3_*` / `run_vggt_*` sbatch
  wrappers and the helpers only they used, five stale or duplicate case lists and three one-off audit/smoke scripts.
- `marigold_v2_eth3d.yaml` is merged into `full_monodepth.yaml`: MARIGOLD_V2 is one of its models, at the native
  size except Marigold's ETH3D subset at 1008x672 (`MARIGOLD_V2_SCENE_SIZES`, new in `marigold_v2.py`: per
  `<dataset>/<scene>`), with one adapter call per size group and 96 GB of inference memory; `all_tracks.yaml` has
  the same entry. `dycheck_videodepth.yaml`
  is merged into `full_videodepth.yaml`, which already ran DyCheck with the same settings (100 frames, seed 42, each
  model's native-space protocol); docs/benchmark/dycheck_videodepth.md points there.
- TRIDENT is removed: its adapter (`scripts/inference/trident.py`), registry entry, engine branch and environment
  defaults, SLAM budget, `trident_only.yaml` and its comparison baseline `vggt_omega_da3_baseline.yaml`, its tests,
  the `"trident"` spatial-policy alias and the dashboard's `UNLISTED_METHODS` (which only hid it). Examples and tests
  that used it as the all-tracks method now use PI3_X; the protocol matrix in `docs/evaluation.md` drops it.
- `scripts/preprocess/mindcloud/` is `scripts/preprocess/spatialarena_benchmark/` and `materialize_mindcloud.py` is
  `materialize_spatialarena.py` (its test `test_materialize_spatialarena.py`): they hold and write the SpatialArena
  benchmark (scene selection, review labels), not MindCloud tooling. Not `scripts/preprocess/spatialarena/`, which
  would shadow the `spatialarena` package for scripts run from `scripts/preprocess/`.
- `scripts/preprocess/dl3dv/` is removed: the helpers `eval_scannetpp_depth.py` used from it (scene listing, COLMAP
  cameras, sidecar files, depth loading) are `spatialarena/eval/scene_files.py`. `scripts/env/` (DetectorFreeSfM setup
  and COLMAP notes) is removed too; FeedForward-Benchmark keeps it.
- `scripts/mapping/database.py` is `scripts/preprocess/data_conversion/colmap_database.py`, next to its only user
  `tnt_to_colmap.py` (as in FeedForward-Benchmark); `scripts/mapping/` is gone. `dtu_to_colmap.py` parses its arguments
  in `main()` instead of on import.
- Dashboard: a case that failed in the method's own code, an operator it uses or an output the scorer rejects (int32
  index limit, CUDA illegal memory access, runtime error, constant prediction, empty evaluation region, too few
  poses, too many frames) is a method limitation (`METHOD_LIMITATION_NOTES`,
  `method_limitations`): its row carries a "Method limitation" badge whose tooltip names each one and its cause, and
  its failed cells are marked (`benchmarkLimitationBadge`, `.benchmark-limitation` in `css/benchmark.css`). No score
  changes. Failures of the evaluation itself (missing dependency, full disk, an adapter that rejects the runner's
  arguments or exports no cameras) stay plain failures. `docs/ffb_20261106_release_issues.md` no longer lists them.
- SLAM `rgb_depth` ([+D]) runs only on sequences with depth maps: not on Oxford Spires, KITTI Odometry or VBR
  (`TRACK_DATASETS`), nor on SpatialArena's sequences without LiDAR depth (new `TRACK_EXCLUDED_SCENES`,
  `track_runs_on_case`). The generated case files and `run_config` coverage leave those cases out, the engine skips
  them as `skipped_not_applicable` (`SLAM_RGB_DEPTH_DATASETS`, `SLAM_RGB_DEPTH_EXCLUDED_SCENES`, kept in step by
  `test_task_track_matrix.py`), and the dashboard neither expects them nor shows their old `skipped_missing_depths`
  records; a [+D] method's SLAM score and coverage count only the sequences it runs on, so the [+D] short / long
  scores rise (e.g. Pi3X [+D] short 66.0 -> 91.1).
- Dashboard failure labels: the causes of the 2026-11-20 per-case triage (`docs/case_failure.csv`) are new
  case-failure statuses (`CASE_FAILURE_LABELS`): "Int32 index limit", "Missing dependency", "Adapter argument error",
  "CUDA illegal access", "Runtime error", "No camera export", "Empty eval region", "Time limit", "Not reached (time
  limit)", "Not scored" (predictions exist, the scoring job was never submitted) and "Not re-scored" (scored only
  under the dropped MVRecon TSDF protocol). The dashboard data marks 439 more cases with their cause; 64 have none
  and stay unlabelled. No score changes. `docs/ffb_20261106_release_issues.md` lists every non-OOM gap with
  its cause.
- SfM `rgb_depth` ([+D]) runs only where the released SfM ground truth has depth maps: ETH3D, HiRoom, IMC, ScanNet++
  and SpatialArena (`TRACK_DATASETS` in `spatialarena/task_track_matrix.py`; the engine skips the other datasets with
  `SFM_RGB_DEPTH_DATASETS`, kept in step by `test_task_track_matrix.py`). DTU, Tanks and Temples and Weak-Texture SfM
  have none, so the track had nothing to give there and was never run; the dashboard no longer lists those cases as
  missing. No score changes.
- Dashboard tables: a table wider than the page can be dragged sideways with the mouse instead of only with the
  bottom scroll bar, and the columns through Method (Rank and Method in the leaderboard) stay in place while the
  metric columns scroll under them (`js/table_drag.js`; sticky columns in `css/benchmark.css`). A drag never sorts the
  column it starts on. The SLAM tables' blank corner above Method and the score is split in two so the Method column's
  header stays in place too.
- ABot-Recon now runs with its loop-closure backend on, its own default (DINO-SALAD retrieval and sparse pose-graph
  optimization, with the SALAD and DINOv2 weights shared with VGGT-Long / Pi-Long); before, its streaming model alone
  was scored. `metrics.json` records the backend's summary (`loop_closure`: candidates, loop edges, timings). Its SLAM
  results need a rerun.
- The dashboard folder `app/benchmark_dashboard` is renamed `app/webpage` (imports `app.webpage.python...`); every
  script, test and doc that named it follows. The page is static files only (`server.py` is a local no-cache file
  server), so it can be published as is, e.g. on GitHub Pages.
- GitHub Pages: `.github/workflows/pages.yml` publishes the webpage on every push to `main` that changes
  `app/webpage/` (or by hand, *Run workflow*). It publishes only what the page loads (`index.html`, `css/`, `js/`,
  `assets/`, `artifacts/`, `data/results.json`, `data/results_live.json`), not the Python tools, docs, local server
  or the data folder's TSVs. One-time setup: Settings -> Pages -> Source: GitHub Actions.
- Dashboard home: the author list (names, MVP Lab, the project-leader / corresponding-author notes) is shown on the
  first slide again (it was hidden for a while with `hidden` on `.home-authors` in `index.html`).
- Evolving curves: SenseNova-Vision is drawn as its logo too (`assets/logos/models/sensenova.svg`, SenseTime's GitHub
  avatar in the same white badge as the others), and the hover tooltip's marker is the point's own (the model's logo,
  or its circle / triangle) instead of Chart.js's colour square.
- Evolving curves: the "N runs" badge beside the title is removed.
- Dashboard home: the hero point cloud's fly-in (from above the roof down into the chapter house) replays every time
  the home view is opened again, not only on the first visit (`js/home_scene.js`); the points are not re-streamed.
- Dashboard Overview: the Task Overview card is removed, together with what only it held: the per-task best-method
  tiles, the data-date badge, the Auxiliary Input filter (the Leaderboard tables, which it also filtered, list every
  method, as they did by default) and the Evaluation Coverage table. The Overview's own filters (tracks, datasets,
  model types) are unchanged.
- Dashboard home: the hero links to the GitHub repository (https://github.com/mvp-ai-lab/SpatialArena) and the Hugging
  Face dataset (https://huggingface.co/datasets/Bluefish2333/benchmark), each with its logo, under the author list
  (`assets/logos/huggingface.svg`, Hugging Face's own logo; the GitHub mark is inline and takes the text colour).
- Dependencies: `requirements.txt` and `setup.py` list every third-party package that the `spatialarena` package and
  its tests import. Added: `torch`, `torchvision`, `safetensors`, `pycolmap`, `scikit-learn`, `joblib`, `imageio` and
  `requests` (plus `easydict` in `requirements.txt`; `jinja2` and `pytest` as the `test` extra). Removed:
  `tensorboard`, which nothing imports. `pyrender` (used only by the Tanks and Temples mesh-culling tool, which the
  evaluator does not import) and the method checkouts' own packages are still not listed.
  Pins moved to the newest releases (tested here): `huggingface_hub` 2.4.0, `matplotlib` 3.11.2, `open3d` 0.20.0,
  `opencv-python` 5.1.0.95, `plyfile` 1.1.5, `trimesh` 5.2.1, `tyro` 1.0.17, `viser` 1.1.1, and `websockets` 16.1.1
  (the newest that `viser` allows); `Brotli` is added because huggingface_hub 2.x's `httpx2` fails downloads with an
  older `brotlicffi`. For huggingface_hub 2.x: the release upload uses `HfApi.upload_folder` (`upload_large_folder`
  is gone), with `--upload_workers` capping hf_xet's upload concurrency (`HF_XET_MAX_CONCURRENT_FILE_INGESTION`,
  `HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY`); `huggingface-cli` (removed) is `hf` in the docs, the release scripts
  and `da3_bench.sh`, which also drops the removed `--resume-download` (downloads resume by default).
- `build_viewer_assets.py`: an MVRecon case whose release has the scored `fused_points.ply` but not the depth
  predictions (most SpatialArena direct-backprojection cases) gets a cloud-only viewer (no cameras, so no frustums or
  confidence) instead of failing with "No MVRecon depth predictions match GT views". Tanks and Temples still needs the
  predictions (its GT crop is registered through the predicted cameras).
- Dashboard, trimmed for the public release (FeedForward-Benchmark keeps the full internal copy): the Datasets
  gallery is replaced by a **Download** view (`#download`, `js/download.js`) that walks through fetching the Hugging
  Face release with `hf download` (a task picker builds the command, SpatialArena-only tracks included), unpacking
  it, linking the five datasets two tasks share, and verifying it, with a release-contents table of every dataset's
  size. The Coverage view and the Artifact Viewer's *Single case* and *Demo replay* modes are removed (the viewer
  opens on trajectory comparison), with their scripts, styles and data (`gallery.js`, `coverage.js`,
  `scene_viewer.js`, `demo_replay.js`, `dataset_gallery.json`, `artifacts.json`), the depth/point-cloud CGI (the
  server now serves static files only), `build_demo_assets.py`, the demo index and the collector's
  `--with-artifacts`. Task Metrics is merged into the Leaderboard: an Overall tab (the ranking) and one tab per task
  with its dataset tabs and metric tables (`#leaderboard?task=<Task>&dataset=<dataset>`). Old `#datasets` /
  `#coverage` / `#tasks` links open Download / Overview / the Leaderboard's first task. The Overview's Camera Pose
  AUC chart is removed (the SfM pose AUCs stay in the Leaderboard's SfM tables), and so are its row of stat cards
  (Methods / Tasks / Metrics / Best Overall) and the method, task and metric reference panels they opened.
- Artifact Viewer reads a released subset committed with the dashboard (`app/webpage/artifacts/`, about
  110 MB) instead of the eval root: three scenes per task (a SpatialArena scene and two public ones), each with the
  task's 10 best-ranked methods (plus GPT-6 Astra, Gemini 4 Argon and Claude Opus 5.5 on MonoDepth), capped to 6
  MonoDepth / 24 VideoDepth frames and 60k MVRecon points. `release_artifacts.json` lists them;
  `python/export_release_artifacts.py` rebuilds the folder from the eval root and writes its README with each source
  dataset's license. Each comparison mode opens on its first released scene, and the viewer loads nothing until it is
  opened.
- Home: "Score vs. release date" is renamed **Evolving Curves for Geometric Foundation Models**, and VLM points are
  drawn as their maker's logo (DepthVLM: Tencent Hunyuan; DepthLM: Meta; Qwen3-VL: Qwen; Gemini 4 Argon, Claude Opus
  5.5, GPT-6 Astra: the Gemini, Anthropic and OpenAI icons openrouter.ai uses; badges in `assets/logos/models/`).
  The plot is taller (up to 640 px), and same-month runs with nearly equal scores are spread sideways so they do
  not cover each other. The trajectory comparison opens SfM
  scenes with camera frustums and SLAM scenes with trajectories. The home page's first slide lists the authors, each
  linked to their homepage: Yu Chen (project leader), Haosen Yang, Su Ee Tan, Zhensong Zhang, Jifei Song and Jiankang
  Deng (corresponding author), MVP Lab (with its logo).
- Home: the Hierarchical composition, Scene coverage and 52 model configurations slides are one slide,
  **SpatialArena at a Glance**, with a tab for each.

### Changed (public release)

- Prediction maps are float32 compressed `.npz` (`np.savez_compressed`): depth in `depths/<stem>.npz` (array
  `depth`), the method's confidence in `confidences/<stem>.npz` (`confidence`, float16, clipped to its range), and the
  raw disparities of disparity models. Every adapter writes them through `adapter_utils.save_prediction_map`, which
  also removes a frame's older `.npy`. On real predictions (DA3-L, iBims-1) a depth map takes 15% of its fp32 `.npy`
  size. The evaluators, viewer builders and FFB's demo builder read both formats (`.npz` first;
  `prediction_artifacts.frame_file` / `load_confidence_map`), so cached and imported `.npy` predictions still score
  unchanged.

- Data paths: every default is under `SPATIALARENA_ROOT` (default `~/spatialarena`; `spatialarena/paths.py`) instead of
  the internal cluster mount. Python defaults use it, shell scripts `${SPATIALARENA_ROOT:-$HOME/spatialarena}`, run
  configs `${SPATIALARENA_ROOT}/...` (expanded by `BenchmarkEvaluator.configure`), and generated Slurm scripts record
  the root they were generated with, since `sbatch --export=NIL` drops the environment. The per-folder variables
  still override. On the cluster, `export SPATIALARENA_ROOT=<the shared dataset_aux folder>`.
- Removed internal-only files: the FFB release-issues doc, the view-curriculum ablation configs and the full-standard
  job submission / monitoring scripts. Personal paths are gone from the TnT mesh culling helper, the resolution
  smoke test (`SI_BENCH_PYTHON`) and the ScanNet++ evaluator (`SCANNETPP_ROOT`, `SCANNETPP_COLMAP_ROOT`); the
  si_bench freeze no longer lists the private FFB repository.
- Dashboard data: `data/results.json` and `results_live.json` are stripped of the collection record, source paths and
  coverage matrix (`python/public_results.py`; the collector's `--public`).

### Fixed

- DA3 `--load_from_colmap` (the camera-conditioned tracks): the dense-points branch used an undefined `manager`
  (`NameError`) after the camera priors moved to `resolve_camera_conditioned_inputs`; it now reads the same COLMAP
  prior model (`load_prior_model`, tested). `da3.py` no longer imports a `find_image_paths` it redefines.
- The pre-commit hook passes on the whole tree. `scripts/inference`, `scripts/test`, `scripts/download`,
  `scripts/release`, `scripts/benchmark`, `scripts/visualization`, `scripts/preprocess/data_conversion`,
  `spatialarena/{utils,geometry,eval/tnt_tools}` and `app/webpage{,/python}` are packages (`__init__.py`), so pylint
  no longer reads a file named after an upstream package (`vggt.py`, `amb3r.py`, `scripts/download/spatialarena.py`)
  as that package (false E0611). Every Python line is within 120 characters.
- DVD failed every VideoDepth case at start-up (argparse exit 2): the engine forwards a configured `SPATIAL_POLICY`
  to every depth-only adapter, and `dvd.py` had no `--spatial_policy`. It now takes `shared`, its only policy (the
  official resize, predictions restored to each frame's source grid), and records it in `metrics.json`.
- AMB3R and RobustVGGT MVRecon cases were never scored ("No camera export"): the engine's `depth_only_exports_poses`
  left them out, so their depth-only inference ran without `--export_colmap` and wrote no `sparse/0` to fuse with,
  although both adapters export the cameras (`amb3r.py`; `robustvggt.py` through `vggt.py`). Both are listed now
  (MVRecon and VideoDepth), covered by `test_task_track_matrix.py` and a new `test_amb3r_adapter.py` test. Their
  MVRecon cases, and DVD's VideoDepth ones, need a re-run.
- Dashboard tables: a method track with no result and no recorded failure on any task gets no "-" rows
  (`add_missing_method_rows`), so the masked- and sparse-depth tracks ([+mD], [+mDC], [+sD], [+sDC]), which have not
  been run, and VLM3, which has no released weights, no longer appear. A track that was run keeps its rows where cases
  are missing.
- Dashboard failure labels: the collector (`report_discovery.read_case_failures`) read only the runner's
  `case_failures*.tsv`, so failures recorded elsewhere showed as plain missing cells -- e.g. DUSt3R and Pow3R on SfM
  IMC / ScanNet++ / Tanks and Temples and on VideoDepth (all CUDA out of memory), and IGGT's GPU OOM on MonoDepth
  AerialMetric wild. It now also reads a scene's `inference_failure.tsv`, failed `eval_*.json` reports (`CUDA_OOM` ->
  GPU OOM, `HOST_MEMORY_OOM` -> RAM OOM, a constant prediction, an empty point cloud) and job watchdog outcomes
  (`OUT_OF_MEMORY` -> OOM), and takes a `case_failures*.tsv` row's `reason` (gpu_oom, ...) over its exit code. The
  dashboard data is rebuilt with them; scores are unchanged.
- Webpage trajectory and reconstruction comparisons: the free view is now fully free. It used an orbit control, which
  keeps the scene's up axis up, so a drag could not turn the view past straight above or below, or roll it; it now uses
  a trackball control (the trajectory comparison's X / Y / Z views keep pan and zoom only, and the reconstruction
  comparison's arrow keys turn the view over the top too). Fitting the view sets it upright again.
- Webpage chart export: the model logos on the evolving curves (the VLMs) were linked by their site path
  (`assets/logos/models/<logo>.svg`), so a saved SVG showed none; they are now embedded as PNGs at 4x their drawn size,
  which the PDF export also picks up.
- DepthLM and Qwen3-VL were scored from a single point query per image: the point-query adapter's grid size
  (`POINT_QUERY_DEPTH_GRID_SIZE`) defaulted to 1, the image centre, so every "dense" map was one constant value and the
  two models, after per-image scale alignment, scored identically on every MonoDepth dataset but SpatialArena. The
  registry now sets `point_grid_size=16` for both (256 queries per image, the grid of their SpatialArena rerun already
  on the dashboard), the engine passes it to the adapter, and it
  keys the prediction cache (old centre-only predictions are not reused; every other method's keys are unchanged)
  and is recorded in the run summary. Their MonoDepth / VideoDepth results stay on the dashboard tagged "rerun
  pending" (`BENCHMARK_METHOD_FLAGS` in `js/benchmark.js`) until the non-SpatialArena datasets and VideoDepth are
  rerun on the cluster. The 2026-11-18 rerun (`MonoDepth/supplement-pointgrid16-standard-20261118`) covers iBims-1,
  the five Synth4K scenes and AerialMetric wild, and `VideoDepth/supplement-pointgrid16-20261118` all 74 VideoDepth
  scenes (Bonn, Sintel, DyCheck, KITTI). `MonoDepth/supplement-pointgrid16-standard-completion-20261118` (imported
  2026-11-19) covers the rest (AerialMetric decoupled / oblique, Hammer, Marigold DIODE / ETH3D / KITTI / NYUv2,
  Spring), so the "rerun pending" flags are removed. Qwen3-VL on AerialMetric oblique has no score: two of its 100
  images got 100 m for all 256 queries, a constant map that the evaluator rejects ("Degenerate constant prediction"),
  which aborts the case; its row reads "2/3 scenes · 1 degenerate constant prediction" instead of keeping the old
  centre-query score.

- SfM Tanks and Temples resolves in a downloaded release: the registry falls back to `SfM/tanks_and_temples` (where
  the Hugging Face release ships it) when `3DGS/tanks_and_temples` does not exist. Roots that have the 3DGS folder
  keep resolving to it, so their prediction cache keys do not change.
- Reconstruction comparison: a window added while a render was queued no longer throws before its first layout.

### Added

- Synced from FFB's `eval_refactor_hsy` (commits since 2026-11-20, ported to this layout):
  - `6010d62` strict point-depth queries: `--require_spatial_queries` / `POINT_QUERY_REQUIRE_SPATIAL=1` in
    `point_query_depth_adapter.py` (a grid of at least 2x2; an image with no valid answer fails instead of borrowing a
    depth), with per-image query evidence (`point_query_outputs.jsonl` written as it goes, `point_query_coverage.json`)
    and coverage in `metrics.json` (`docs/benchmark/point_query_grid_protocol.md`). Size-group merging
    (`single_view_groups.py`) concatenates the new coverage file and list, so grouped MonoDepth runs keep working.
  - `45300bb` Marigold V2 (`MARIGOLD_V2`, Log-stage2, `marigold_v2.py`): the official `scripts/infer.py` on the
    selected images only, raw normalized log depth in `raw_log_depths/` (`.npz`), registered with
    `depth_output_space="log_depth"` so it is scored by `log_huber_irls_gt` (already here); `MARIGOLD_V2_WIDTH` /
    `MARIGOLD_V2_HEIGHT`, per scene `MARIGOLD_V2_SCENE_SIZES` (its `full_monodepth.yaml` entry: ETH3D at 1008x672);
    `MARIGOLD_PROCESSING_RES` for the v1 adapter; size-group merging keeps (and requires) `raw_log_depths/`
    (`docs/benchmark/marigold_versions.md`). MARIGOLD_V2 now joins the MonoDepth `rgb` plan.
  - `03f5c26` bounded MVRecon scoring: exact nearest-neighbour metrics in chunks (`MVRECON_METRIC_WORKERS`,
    `MVRECON_METRIC_CHUNK_POINTS`), opt-in memory tracing (`FFB_SCORE_MEMORY_TRACE=1`), empty or non-finite fused/GT
    clouds and DTU point sets fail explicitly instead of scoring NaN (`eval_dtu_mesh.py`, whose standalone-CLI imports
    are now lazy), and an opt-in SpatialArena-only direct-backprojection backend
    (`SPATIALARENA_MVRECON_BACKEND=direct_backprojection`; TSDF stays the default) recorded in each report's
    `metric_protocol.reconstruction_backend` (`docs/benchmark/mvrecon_scoring.md`).
  - Two fixes their tests needed: `run_config` coverage honours a dataset item's `scenes:`, and validation checks a
    config's own checkpoint override (`<METHOD>_CKPT_PATH` in `env:`) instead of the registry default.
- SpatialArena repository, split out of FeedForward-Benchmark (2026-11-17): the benchmark evaluation code, model
  adapters, benchmark dataset preparation and release scripts, Slurm job generation, reports, the results dashboard
  and their tests. The Python package `benchmark_suite` is renamed `spatialarena` (`import spatialarena`,
  `python -m spatialarena.run_config ...`); job scripts default `PROJECT_ROOT` to `~/Projects/SpatialArena`, and
  `run_benchmark.py` reads `SPATIALARENA_PROJECT_ROOT` (the old `FEEDFORWARD_BENCHMARK_ROOT` still works).
  The geometry and point/depth helpers the evaluators use (`spatialarena/geometry`, `spatialarena/utils`) and a few
  shared scripts (DL3DV mask helpers, DTU / Tanks-and-Temples COLMAP converters, TexturePoorSfM consolidation,
  DetectorFreeSfM setup) are copies of FeedForward-Benchmark's.
