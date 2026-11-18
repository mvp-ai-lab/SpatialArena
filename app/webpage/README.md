# Benchmark Dashboard

A standalone web app for SpatialArena results. It collects metrics from every evaluated
method and track, ranks them, compares prediction artifacts across methods, and explains how to download the
benchmark data. It is a static site: `server.py` only serves files.

## Quick start

```bash
cd ~/Projects/SpatialArena

# 1. Collect metrics from the reports tree into app/webpage/data/
~/anaconda3/envs/dogs/bin/python app/webpage/python/collect_benchmark_metrics.py \
  --reports-root $SPATIALARENA_ROOT/benchmarks/_evaluation/reports

# 2. Serve it (default port 8020)
~/anaconda3/envs/dogs/bin/python app/webpage/server.py 8020
```

Open `http://localhost:8020/`. Views are addressable by hash: `#home`, `#overview`, `#leaderboard`
(`#leaderboard?task=SfM&dataset=eth3d` opens a task's tables), `#artifacts` and `#download`. Old links still work:
`#tasks` opens the Leaderboard's first task, `#datasets` the Download view, `#coverage` the Overview and
`#contributors` the home page.

`run.sh [PORT]` does both steps. Set `PYTHON` to pick the interpreter, `SKIP_COLLECT=1` to serve the
existing data without re-collecting, and `COLLECT_ARGS` to pass extra collector flags, e.g.
`COLLECT_ARGS=--all-scenes app/webpage/run.sh`.

## How metrics are collected

`python/collect_benchmark_metrics.py` does not depend on a hand-maintained summary. For each
`--reports-root` (default: `$BENCHMARK_REPORTS_ROOT`, falling back to
`~/benchmark_evaluation/reports`) it:

1. **Discovers report roots.** Jobs write `reports/[<batch>/]<suite>/<method_id>/<Task>/<dataset>/<scene>/eval_*.json`
   plus a `summary.json` carrying `method`, `base_method`, and `benchmark_track`. Each eval file is
   attributed to its nearest `summary.json` ancestor, or to the directory above its first task-named
   path component when no summary exists. Top-level batches matching `_*`, `*verify*`, `*smoke*`, or
   `*debug*` are skipped; add globs with `--exclude` or disable the defaults with `--no-default-excludes`.
2. **Deduplicates across batches.** The same scene is often evaluated by several batches (original
   run, retries, resumes). For each `(method, task, dataset, scene)` the newest `eval_*.json` wins.
3. **Restricts to the full-standard case lists.** Only scenes listed in
   `scripts/test/full_standard_depth_cases.txt` and `scripts/test/full_standard_sfm_cases.txt` are
   aggregated, so every method is compared on the same scenes. Use `--case-list` to substitute lists or
   `--all-scenes` to aggregate everything. Dropped counts are recorded in `results.json`.
4. **Aggregates and ranks** with `benchmark_analyzer.py` (auxiliary-input tracks excluded from the final
   ranking, `scannetpp`, `spring` and `spatialarena` excluded from leaderboard aggregates). The per-scene metrics go to
   `scenes.tsv` next to `summary.tsv`, and every task but SLAM is scored per scene, each scene weighing the
   same: on a scene, each score metric is min-max normalized to 0-100 among the methods that evaluated it
   (one method alone gets 100); a method's scene score is the mean over the score metrics, one it lacks
   counting 0; its task score is the mean over the task's cases (the full-standard cases on its track's
   datasets that any method evaluated), a case it has no result for counting 0. The depth tasks score only
   AbsRel%, Delta@1.03%, Delta@1.25%, Delta@1.25^2%, Delta@1.25^3% and, on datasets that report it,
   MAE-normal; their other metrics are for reference. Other tasks score all their metrics. SLAM keeps its
   protocol: only methods covering a whole range are ranked, on the normalized ATE RMSE and drift of each
   dataset row. Normal/point-cloud metrics missing from older eval files are recomputed for
   the winning eval file only; skip this with `--skip-geometry-recovery`. Results the task/track matrix
   rules out (`on_matrix` in `spatialarena/task_track_matrix.py`) are dropped from the tables, the
   artifact viewer and the comparison index: e.g. `PI3_X_rgb_depth` on MonoDepth, which the old shell
   pipeline produced but which scores depth the method was given. Dropped rows are counted under
   `summary.excluded_off_matrix_rows`.
5. **Builds the coverage matrix.** Every method in `spatialarena.methods.METHOD_SPECS` is crossed with
   every task/track pair in `spatialarena.task_track_matrix.TASK_TRACK_MATRIX`.

Restrict to specific method IDs with `--method PI3_X --method PI3_X_rgb_cameras`, and write elsewhere
with `--output-dir`.

### Coverage statuses

The page does not show this matrix (FeedForward-Benchmark's internal copy of the dashboard does); it is written to
`results.json` as `coverage`. Each supported cell reports evaluated vs. expected scenes (from the case lists) and one
status:

- `complete`: every expected scene has an evaluation result;
- `partial`: some expected scenes are evaluated;
- `failed`: no results, but the runner recorded case failures in `case_failures*.tsv`;
- `reported_unsupported`: no results, and the method wrote `unsupported_status.json` (the reason is shown);
- `not_run`: nothing found for this method/task/track;
- `unsupported`: the method's registry entry does not declare this track (not counted as pending).

Method IDs that have results but are not in the registry are listed as unregistered.

## Outputs

All written to `data/` (or `--output-dir`):

- `summary.tsv`: per `(method, task, dataset)` aggregated metrics, in the same format as
  `scripts/test/summarize_benchmark_reports.py --output-tsv`.
- `results.json`: dashboard payload. It keeps the `benchmark_analyzer.py` schema (`task_tables`,
  `ranking`, `task_ranking`, `radar`, `method_statuses`, ...) and adds `coverage` (registry matrix)
  and `collection` (reports roots, exclude globs, case lists, every discovered root, dropped scenes).
- `results_live.json` (optional, not written by the collector): if present, the page merges it over
  `results.json`, so a partial in-progress export can be layered on top of a full one.

## Artifact viewer comparison modes

The Artifact Viewer has three comparison modes (trajectory comparison opens first). Each one keeps its
state in the URL (`#artifacts?compare=...`), so a view can be shared as a link. The 3D views (trajectory,
reconstruction) have a *Full screen* button (toolbars and layer list included; Esc leaves), and over the view
Ctrl + wheel changes the point size and Alt + wheel the camera frustum size.
In the reconstruction and depth comparisons, Ctrl + drag (Cmd + drag on macOS) a viewer window onto another one
to move it there, or use the left / right arrows in the side list of windows; the windows' own titles carry no
buttons, so screenshots stay clean. The order is kept in the URL.

- **Trajectory comparison** (SfM, Weak-Texture SfM, SLAM): the GT and several methods' trajectories in one 3D view,
  each aligned exactly as its ATE is computed. The GT is drawn on the frames the visible methods have a pose for (a
  method that registered part of the images or returns keyframes only is compared on its own frames), and a
  trajectory with missing poses is joined across them. Free orbit or views along X / Y / Z, a frame slider with
  markers, and the per-frame position error of every method. *Error lines* link each predicted camera center to its GT center
  (the translation error), in the method's color: at the current frame, for every camera, or off.
- **Depth comparison** (MonoDepth, VideoDepth): *Grid* (the default) shows every method side by side with shared
  zoom, pan and crosshair, and a magnifier (on by default) over every tile at the cursor; *Swipe* stacks two layers
  (GT, RGB or any method) in one block and the cursor moves the divider. Methods show
  their aligned depth or their error map. All methods of a scene use one depth color scale fixed by the GT, so they
  compare directly; frames play back like a video.
- **Reconstruction comparison** (MVRecon): each method's fused cloud, aligned to the GT as scored, in its own window.
  All windows share one camera, so orbiting, zooming or a shortcut key (arrows, +/-, F, 1/2/3) in any of them moves
  all. Points show their fused color or their error (distance to the nearest GT point). *GT window* hides or shows
  the ground-truth window (as does its row's swatch or x), *GT overlay* draws the GT points in every method window,
  and *Cameras* draws the fused views' frustums (in a *Frustum color* of your choice). *Hide low confidence* hides
  each method's least confident points, as a percentile of its own predicted confidence (methods use different
  scales): a fused point's confidence is the mean confidence map value of the views that see it
  (`viewer_assets.point_confidence`, `confidence.f32`), for predictions that saved confidence maps.

The evaluators write the assets these modes read while they hold the aligned data
(`spatialarena/eval/viewer_assets.py`, on by default through `VIEWER_ASSETS=1` in the engine): per case into
`<case report dir>/viewer/`, and the files every method of a scene shares (GT and RGB frames, GT points) into
`<eval root>/viewer_assets/scenes/`. The collector then indexes them into `<eval root>/viewer_assets/index/`
(`python/viewer_index.py`; skip with `--no-viewer-index`), so nothing else has to run after an evaluation. For results
evaluated before the evaluators wrote viewer assets, `python/build_viewer_assets.py {trajectories,depth,recon}`
backfills them on a compute node with the same writers; `python/build_viewer_assets.py confidence` adds
`confidence.f32` to MVRecon viewer clouds written before it (from the saved cloud and cameras, without fusing again).

### Released artifacts

The eval root's viewer assets are tens of GB, so the page does not read them: the Artifact Viewer reads
`artifacts/`, a small subset committed with the dashboard (about 110 MB). `release_artifacts.json` lists its scenes,
three per task (a SpatialArena scene and two public scenes whose licenses allow redistributing derived frames), and
the caps; `python/export_release_artifacts.py` builds it from the eval root:

```bash
python app/webpage/python/export_release_artifacts.py --eval-root /path/to/_evaluation
```

Per scene it keeps the task's `top_methods` (10) best-ranked methods that have assets for the scene, by the
leaderboard's task ranks in `data/results_live.json` (else `data/results.json`), plus the task's `extra_methods`
(the three commercial VLMs on MonoDepth). Depth scenes keep 6 frames spread over the scene (MonoDepth) or the first 24
(VideoDepth, so it still plays), MVRecon clouds 60k points (GT 60k), and trajectory thumbnails are re-encoded at
240 px. The trajectory mode shows SfM cameras as frustums and SLAM as trajectories when a scene opens. The files
keep the evaluators' formats, under `artifacts/index/` (the viewer index), `artifacts/scenes/`
(shared per scene) and `artifacts/cases/` (per method), and `artifacts/README.md` lists the scenes and credits each
source dataset's license. The export is written to `artifacts.tmp/` first and fails, leaving the old folder in
place, if it exceeds `max_total_mb` (200). Rerun it after the leaderboard changes; each mode opens on the first
released scene with its four best methods.

## Leaderboard and task metrics tables

Results that need a caveat are listed in `BENCHMARK_METHOD_FLAGS` (`js/benchmark.js`, by base method): tables tag the
name "rerun pending" (the caveat on hover), charts add `*`, and the leaderboard, task tables and evolving curves print
the caveat underneath. DepthLM and Qwen3-VL are flagged until their non-SpatialArena MonoDepth datasets and VideoDepth are
rerun with the 16x16 query grid their SpatialArena scenes already use; remove their entries then.

The Leaderboard view (`#leaderboard`) has an **Overall** tab, the ranking over all tasks, and one tab per task with
that task's metric tables (this was the separate Task Metrics view). Under a task tab there is one tab per dataset of
that task plus an **Overall** tab. Overall is the consolidated table over all of the task's datasets; for SLAM it is one table per range
track (short, long, large), with each dataset's columns grouped and the range score over them. A dataset tab shows that
dataset alone with every metric (for SLAM, one table per range the dataset takes part in). Each task remembers its
dataset tab, and the URL records the open task and dataset tab. Overall averages the leaderboard datasets only: a
track whose results are all on datasets left out of it (SpatialArena, ScanNet++, Spring), such as VideoDepth [+C] on
SpatialArena, shows "SpatialArena only · not scored" instead of empty cells and follows the ranked rows; its numbers
are in that dataset's tab.

The overall score of a method is the mean of its task scores, a task without a result counting 0. An rgb method is
averaged over every task; a conditioned track ([+C], [+D]) only over the tasks the task/track matrix lets it run on a
leaderboard dataset (`track_ranking_tasks`, the row's `ranking_tasks`), so Pi3X [+C] is scored on MVRecon alone rather
than given 0 for the four tasks it never runs. The Overview's track and dataset filters re-average the same way.
Because that score covers fewer tasks, conditioned tracks are kept out of the overall ranking: they follow the rgb
methods with no overall rank (`rank` is null), and keep their per-task ranks.

## Download view

The Download view (`#download`) explains how to fetch the benchmark from its Hugging Face release
(`Bluefish2333/benchmark`): install the client, build the `hf download` command for the chosen tasks (or the
SpatialArena tracks only), unpack the per-scene archives with the bundled `extract_benchmark_archives.py`, link the
five datasets that two tasks share (packed once; the other task's folder holds a `SEE_INSTEAD.md`), and verify the
result. The task picker and the release-contents table read the per-dataset sizes and shared-folder pointers in
`js/download.js`; update them there when the release changes.

The home page's dataset strip shows `assets/datasets/<dataset>.jpg` (640x400), built by
`python app/webpage/python/build_dataset_gallery.py` from the card list in `python/dataset_catalog.py`.

## Exporting charts

Every chart on the dashboard has an Export bar (top right, on hover) that saves it as PNG,
SVG or PDF (`js/chart_export.js`, the same approach as `docs/webpage`). Exports are always drawn in the light theme
and with the current filters: PNG at 4x resolution, SVG and PDF as vector graphics (canvas2svg, jsPDF and svg2pdf.js,
loaded from jsDelivr on first use). PDF text uses Helvetica, so Greek letters are spelled out there (VGGT-Omega).
Files are named after the card and panel, e.g. `ffb-score-vs-release-date-monodepth.pdf`.

The artifact viewer's panels export the same way: the depth comparison's tiles and legend
(grid or swipe, at the current zoom), the trajectory / reconstruction 3D views with their insets, and the
trajectory mode's pose-accuracy chart. Panels are redrawn as SVG boxes and text with their images and canvas
snapshots (WebGL views keep their last frame for this: `preserveDrawingBuffer`) embedded as PNG; PNG exports are
that SVG at 2x. Panel files are named after the mode and scene, e.g. `ffb-depth-ibims1-ibims1.pdf`.

## Home page

The dashboard opens on **Home** (`#home`; the navbar's SpatialArena logo leads back to it, there is no Home tab), a
slide show introducing SpatialArena (`js/home.js`, `css/home.css`):
the 3D hero scene, SpatialArena at a Glance, the evolving curves (score vs. release date), our scenes,
depth alignment, trajectory alignment and how to evaluate a model. Side dots, the arrow keys, Page Up/Down and Space move between
slides, and `#home?slide=N` (1-based) links to one. Each slide's content animates in as it comes into view; with
reduced motion everything is shown at once.

- **Authors**, on the first slide under the lead: each name links to the author's homepage; † marks the project
  leader (Yu Chen) and ✉ the corresponding author (Jiankang Deng); the lab is MVP Lab, with its logo
  (`assets/logos/mvp_lab.png`, the lab's GitHub avatar). Static markup in `index.html`. Hidden for now (the block's
  `hidden` attribute); remove the attribute to show it again.
- **Hero scene** (`js/home_scene.js`): `assets/scene/yorkminster_chapter_house.sap2`, the SpatialArena
  YorminsterTheChapteraofRoof LiDAR map at 2.4M points, streamed in five chunks (two on small screens, data-saver
  connections and low-memory devices). Packed by `scripts/visualization/pack_home_hero_cloud.py`.
- **SpatialArena at a Glance** (`js/home_atlas.js`): one slide with three tabs, *Hierarchical composition* (the
  456 evaluation cases by environment, motion, scene type and dataset), *Scene coverage* (which conditions each
  dataset tests, and which tasks use it) and *52 model configurations* (the model families). Arrow keys move between
  the tabs; the models donut is resized when its tab opens.
- **Evolving Curves for Geometric Foundation Models**, score vs. release date (`renderBenchmarkReleaseScatterCharts` in `js/benchmark.js`; moved here from the
  Overview): one tab per task, each run's leaderboard task score against its paper date with per-family trend lines.
  VLMs are drawn as their maker's logo (`BENCHMARK_MODEL_LOGOS`, badges in `assets/logos/models/`; the Gemini, Claude
  and GPT marks are the icons openrouter.ai shows for Google Gemini, Anthropic and OpenAI). Runs from the same month
  with nearly the same score are spread sideways within the month (`benchmarkDodgeReleasePoints`; the trends fit the
  paper dates). It plots the
  published leaderboard scores, so the Overview's filters do not apply. Drag a box to zoom; the chart
  width (full / half) and the Export bar are kept for paper figures.
- **Depth alignment** (`js/home_align.js`): real iBims-1 frames (CC BY 4.0, credited on the slide) with MoGe's raw
  prediction from the stored MonoDepth artifacts, and every step of the benchmark's own Huber IRLS fit.
  `assets/align/`, written by `scripts/visualization/export_home_depth_alignment.py`.
- **Trajectory alignment** (`js/home_align.js`): six scenarios of the alignment study
  (`scripts/test/compare_trajectory_alignment.py`: noise, outliers, tracking jump, rotation noise, drift,
  near-collinear), three cases each on TUM RGB-D (CC BY 4.0) and SpatialArena GT or the study's synthetic straight
  path. Every RANSAC sample, the inlier refit and each Cauchy IRLS iteration of `ransac_irls_align_sim3` are recorded
  offline and played back, beside Umeyama (evo) and R-DAGSfM and the study's medians.
  `assets/align/trajectory/`, written by `scripts/visualization/export_home_trajectory_alignment.py`, which checks
  each case's final estimate against `ransac_irls_align_sim3`. The cameras are drawn with three.js in the style of
  the FreeScale project page (floor grid, filled frustums, the step as a pill, scenario pills inside the view): drag
  orbits, Ctrl/Cmd + scroll zooms, double-click or the reset button restores the view. Without WebGL or the CDN the
  slide falls back to a 2D canvas projection.

The curves on both alignment slides share one plot frame: nice ticks on both axes, the current value in the card
header, the rest of the series as a faint preview, and a hover read-out (crosshair and tooltip) of any step.

The three scripts read `--benchmark_root` (default `~/datasets/benchmark`) and reproduce the committed assets byte
for byte.

## Other entry points

- `python/benchmark_analyzer.py --input <summary.tsv> --output data/results.json`: build the payload from
  a curated TSV (e.g. the `full_standard_benchmark_summary.tsv` that
  `scripts/slurm/aggregate_full_standard_reports.sh` writes under `<BENCHMARK_EVAL_ROOT>/reports/full_standard_summary/`), without
  the registry coverage matrix. `BenchmarkEvaluator.export_report()` and generated
  `gather_metrics.sh` scripts use this path for per-run reports.

## Layout

```text
app/webpage/
  index.html                    page shell: top navbar + Home (with the author list), Overview, Leaderboard (with the task tables),
                                Artifact Viewer, Download
  server.py, run.sh             static file server, collect-and-serve launcher
  css/base.css                  theme tokens, cards and tables
  css/benchmark.css             dashboard styles
  css/site.css                  top navbar, section headers, Download view
  css/compare.css               comparison mode styles
  css/home.css                  home page
  js/benchmark.js               data loading, view switching, rankings, charts, task tables
  js/download.js                Download view: task picker, download command, release contents
  js/compare/                   artifact viewer comparison modes (trajectory, depth, reconstruction) and mode switch
  js/home*.js                   home page slides
  js/chart_export.js            PNG / SVG / PDF export of charts and viewer panels
  python/collect_benchmark_metrics.py   discovery + aggregation + coverage entry point
  python/report_discovery.py            report-root discovery, case lists, case failures
  python/benchmark_analyzer.py          ranking and payload builder
  python/benchmark_artifact_analyzer.py artifact-case scanner; build_viewer_assets.py uses its path helpers
  python/dataset_catalog.py             dataset list for the home page thumbnails
  python/build_dataset_gallery.py       home page dataset thumbnails
  python/viewer_index.py                index of the evaluators' viewer assets (run by the collector)
  python/build_viewer_assets.py         backfill of viewer assets for older results
  python/export_release_artifacts.py    the Artifact Viewer's released subset (artifacts/)
  release_artifacts.json        scenes, methods and caps of that subset
  artifacts/                    released viewer assets (committed; generated)
  assets/datasets/              home page dataset thumbnails
  data/                         generated summary.tsv, results.json
  docs/visualization_plan.md    original design notes
```

## Deployment notes

- `BENCHMARK_DASHBOARD_HTTP_TIMEOUT` sets the per-request timeout in seconds (default `1800`).
- `server.py` is only a local preview server (static files with no-cache headers); the page needs no backend.
- **GitHub Pages**: `.github/workflows/pages.yml` publishes the page on every push to `main` that changes
  `app/webpage/` (or by hand from the Actions tab, *Run workflow*). It publishes only what the page loads:
  `index.html`, `css/`, `js/`, `assets/`, `artifacts/`, `data/results.json` and `data/results_live.json`. A new file
  or folder the page loads must be added to the workflow's "Stage the web files" step. One-time setup: Settings ->
  Pages -> Source: GitHub Actions. On a private repository Pages needs a paid GitHub plan, and the site is public.
