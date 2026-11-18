# Benchmark Visualization Plan

> Historical design note. This view started as a Benchmark tab inside `app/data_manager` and now
> lives in the standalone `app/webpage` app. Data no longer comes only from the curated
> TSV: `python/collect_benchmark_metrics.py` discovers every report root and adds registry coverage.
> See `../README.md` for current usage.

## Objective

Add a Benchmark tab to the existing data manager dashboard so full-standard benchmark results can be inspected without reading raw TSV/Markdown reports. The dashboard will visualize metrics grouped by task, rank methods globally, and compare methods with a radar chart.

## Data Source

Use the full-standard summary TSV (`full_standard_benchmark_summary.tsv`, now written under `<BENCHMARK_EVAL_ROOT>/reports/full_standard_summary/`) as the canonical input. This file already contains the final no-ScanNet full-standard depth and pose summaries produced by the artifact-only benchmark evaluators.

The data manager will not scan experiment folders directly for this view. Instead, a small exporter will convert the TSV into `app/webpage/data/results.json`, which keeps the browser-side code simple and reproducible.

## Exported JSON Contents

The exporter will produce:

- metadata: generation time, source TSV path, row counts, methods, tasks, datasets
- metric catalog: display names, metric direction, metric family, precision
- task tables: task-level method rows with metrics as columns
- dataset tables: task/dataset method rows for drill-down
- ranking table: overall rank, normalized score, average rank, first-place wins, metric coverage, per-task scores
- radar data: compact axes and all-metric axes with normalized method scores

## Metric Direction

Lower-is-better metrics:

- depth errors: AbsRel, SqRel, RMSE, RMSE-log, log10, iRMSE, SILog
- pose errors: ATE, relative rotation error, relative translation-direction error, relative translation-axis error, relative pose error, relative pose-axis error

Higher-is-better metrics:

- depth accuracy: Delta@1.03, Delta@1.25, Delta@1.25^2, Delta@1.25^3
- pose accuracy: RRA, RTA, mAA, AUC

## Ranking Protocol

0. Drop any row whose dataset is in `LEADERBOARD_EXCLUDED_DATASETS`
   (`benchmark_analyzer.py` -- currently just `scannetpp`, per
   `docs/benchmark/data_leakage_notes.md`) before every step below. The
   dataset stays fully visible in the per-task/per-dataset tables (step
   "dataset tables" above) -- it's excluded only from the aggregate
   ranking/leaderboard score, not from the evaluation or the raw data.
1. For every comparable `(task, dataset, metric)` group, rank methods using the metric direction.
2. Convert each metric value to a normalized `0-100` score within the same comparable group.
3. Aggregate per method:
   - overall normalized score: mean normalized score over all available comparisons
   - average rank: mean rank over all available comparisons
   - first-place wins: count of rank-1 comparisons
   - per-task normalized scores and ranks
   - coverage count, so missing metrics remain visible
4. Sort global rankings by normalized score descending, then average rank ascending.

## User Interface

Add a `Benchmark` nav item and tab to `app/webpage/index.html`.

The tab will include:

- summary cards for method count, task count, metric count, and best overall method
- task overview cards showing the best method, dataset count, method count, and metric count per task
- a method-by-task evaluation coverage table
- a radar chart with compact and all-metric modes
- a final ranking table with method filtering, sorting, and CSV export
- task tabs, and per task an Overall tab (consolidated table; SLAM: one table per range) plus one tab per dataset
- task-grouped metric tables where rows are methods and columns are metric types
- CSV export for the currently filtered task metric rows

## Reconstruction And Depth Visualization

Add a second benchmark data export,
`app/webpage/data/artifacts.json`, generated from
a run manifest (now: discovery by `collect_benchmark_metrics.py --with-artifacts`). This manifest records the canonical
full-standard artifact roots, report roots, available scenes, metrics, and
lazy visualization URLs. It does not inline frame pairs or large 3D viewer
manifests; those are resolved on demand for the selected case.

Depth-only tasks (`MonoDepth` and `VideoDepth`) use image comparison:

- controls: task, dataset, method, scene, frame
- panels: ground-truth depth, aligned prediction depth, absolute error
- frame pairs are resolved lazily for the selected scene using the same
  depth-file pairing rules as the evaluator
- images are rendered lazily by a constrained CGI endpoint so large `.npy`,
  `.npz`, `.dpt`, `.pfm`, and depth image files are not copied into `docs/`

Pose/reconstruction tasks (`SfM` and `CameraPose`) use the existing 3D viewer:

- compare mode shows ground truth on the left and prediction on the right
- 3D viewer manifests are generated lazily for the selected scene/method so the
  dashboard JSON stays small enough to load quickly
- benchmark reconstruction viewers expose only point-cloud and camera-pose
  controls; bounding-box and splat labels/controls are omitted
- users can adjust point size, camera-frustum size, and the number of sampled
  points visualized
- camera frustums in prediction views are colored by matched camera-center
  error when those errors can be computed
- point clouds can be viewed with original image/vertex colors or error colors;
  error-color PLY output is generated lazily by the same constrained CGI
  endpoint when a reference sparse GT point cloud is available
- point-cloud and camera assets are loaded on demand through the existing lazy
  viewer shell

The viewer remains artifact-only. It reads the same standardized outputs that
the evaluator reads (`depths/`, `sparse/0`, and `points3D_dense.ply`) plus the
canonical ground-truth paths recorded in evaluation reports.

## Files To Add Or Update

- `app/webpage/python/benchmark_analyzer.py`
- `app/webpage/python/benchmark_artifact_analyzer.py`
- `app/webpage/cgi-bin/benchmark_asset.py`
- `app/webpage/data/results.json`
- `app/webpage/data/artifacts.json`
- `app/webpage/js/benchmark.js`
- `app/webpage/js/scene_viewer.js`
- `app/webpage/css/benchmark.css`
- `app/webpage/index.html`
- `app/webpage/server.py`
- `test/test_benchmark_visualization_export.py`

## Validation

- Unit test the exporter and ranking logic.
- Generate `benchmark_analysis/results.json` from the current full-standard TSV.
- Generate `benchmark_analysis/artifacts.json` from the current canonical run
  manifest.
- Smoke-test lazy CGI endpoints for selected depth-frame JSON, rendered depth
  PNGs, reconstruction viewer JSON, sampled GT PLY, and point-error PLY.
- Load the data manager through the local server.
- Verify the Benchmark tab, ranking table, task tables, artifact viewer, and
  radar chart render without console errors.
