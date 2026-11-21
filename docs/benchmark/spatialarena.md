# SpatialArena

SpatialArena is the benchmark's own capture: 30 scenes recorded with a handheld 3-camera LiDAR rig,
indoor and outdoor, across London, Cambridge, Edinburgh and York. It feeds all five tasks. The scene selection, the
sampling of each track and the ground-truth sources are in
FeedForward-Benchmark's `docs/mindcloud_benchmark_scene_selection.md`.

| Task | Cases | Images | Ground truth |
| --- | ---: | ---: | --- |
| MonoDepth | 19 scenes | 1,688 | GGGS dense depth (m), per-frame intrinsics |
| VideoDepth | 15 clips from 7 scenes | 4,510 (4,484 with GT) | GGGS dense depth (m), `sparse/0` (single camera) |
| MVRecon | 6 scenes, every image of the 3 cameras (300 per scene evaluated) | 5,295 | GGGS depth and its validity mask, the LiDAR map cropped to the views (`gt_points.ply`) |
| SfM | 7 scenes, all 3 cameras | 7,680 | Refined, Sim(3)-aligned poses (`sparse/0`); LiDAR depth priors (cm) |
| SLAM | 8 sequences (3 short, 2 long, 3 large) | 15,821 | Same poses, one camera at 5 Hz; LiDAR depth priors for the 3 RGB+D sequences |

The tracks add up to 34,994 images, but MonoDepth samples the VideoDepth and MVRecon scenes, so SpatialArena holds
32,655 unique images (the figure in `docs/evaluation.md`'s unique-dataset table).

**Dense-depth validation** (measured 2026-10-30 on the released MonoDepth and VideoDepth tracks). Every GGGS depth map
is compared with the sparse LiDAR depth of the same image on the pixels where the LiDAR depth lies in `[0.1, 100]` m:

| Track | Images | Pixels within ±25% of LiDAR | AbsRel vs. LiDAR | Per-image share within ±25% (min / 5th percentile) |
| --- | ---: | ---: | ---: | --- |
| MonoDepth | 1,688 | 98.3% | 0.036 | 73.0% / 94.4% |
| VideoDepth | 4,484 | 98.2% | 0.030 | 15.8% / 95.2% |

LiDAR has no returns where GGGS fails most often (near-camera floaters, reflections, weak texture;
FeedForward-Benchmark's `docs/mindcloud_gggs_artifacts.md`), so this check is backed by MonoDepth's render-PSNR
selection and the manual review (FeedForward-Benchmark's `docs/mindcloud_benchmark_scene_selection.md`).

**License:** CC BY-NC 4.0 (`spatialarena/dataset_licenses.py`). In MonoDepth, VideoDepth and MVRecon, people and
vehicles are blacked out of the images and LiDAR heatmaps and zeroed in the LiDAR depth, matching the dense depth;
SfM and SLAM frames are released unmasked.

## Getting the data

The Hugging Face release (`Bluefish2333/benchmark`) holds each track as `<Task>/spatialarena/`, one uncompressed
`<scene>.tar` per scene plus its `metadata.json`:

```bash
python scripts/download/spatialarena.py --benchmark_root /path/to/benchmarks            # all five tracks
python scripts/download/spatialarena.py --benchmark_root /path/to/benchmarks --tasks SLAM
```

`python -m spatialarena.download` runs the same step for each task. The result is
`<benchmark_root>/<Task>/spatialarena/<scene>/`, where every dataset sits.

**From the released recordings.** `scripts/preprocess/materialize_spatialarena.py` writes all five tracks from the
processed recordings (`<released>/<Scene>-<date>/convert_final`), with the tracks holding the scenes directly
(`<root>/<Task>/<scene>/`). To evaluate from such a root as it is, set `SPATIALARENA_ROOT=<root>`; the registry
then reads SpatialArena from `<root>/<Task>` instead of `<benchmark_root>/<Task>/spatialarena`
(`spatialarena/spatialarena.py`).

## Protocol

SpatialArena follows each task's standard protocol (`docs/evaluation.md`), with these dataset settings:

- **Two depth GTs (MonoDepth, VideoDepth):** every scene is scored on the GGGS dense depth (`depths/`, the
  `metrics` of a scene result) and again on the sparse LiDAR depth (`lidar_depths/`, uint16 cm; the same metrics
  prefixed `lidar_`, e.g. `lidar_abs_relative_difference`). Each pass fits its own alignment with the same protocol
  and range; the LiDAR pass reports depth metrics only (no normal or point metrics) and its counts under `lidar_gt`,
  and writes its fit records as `<protocol>_lidar_fit(s)`. Aggregates average each metric over the scenes that
  report it.

- **MonoDepth:** every image is scored; valid GT and prediction clip `[0.1, 100]` m, from each scene's
  metadata.json entry. 203 frames left out by the manual review are not shipped.
- **VideoDepth:** every frame of each clip is fed and scored, with no frame cap (the other datasets keep the
  suites' 100; `max_input_frames=0` in the registry), valid GT `[0.1, 100]` m,
  prediction clip `[0.1, 100]` m, one fit per sequence. 26 frames whose GT the manual review left out keep their
  image as model input but are not scored: `gt_excluded` in the clip's `frame_manifest.json`, read by
  `depth_prediction_evaluator.gt_excluded_frame_keys` and listed as `gt_excluded_frames` in the scene result.
  `rgb_cameras` priors are the clip's `sparse/0`; SpatialArena is the only VideoDepth dataset `rgb_cameras` runs on.
- **MVRecon:** every image of a scene is shipped, and a method is fed 300 of them (`max_input_frames=300`, a
  fixed-seed random subset, re-sorted). The distance threshold is per scene, `eval_threshold` in its metadata.json
  entry (0.05 m indoors, 0.25 m outdoors); the fusion voxel, truncation, downsampling and fusion depth scale with it
  as DA3-BENCH scales them for ETH3D: predicted depth is fused up to 20 m indoors (7.8 mm voxels; fusing to 100 m on
  that grid needed over 330 GB per scene, and at most 0.2% of the indoor GT depth lies beyond 20 m) and 100 m
  outdoors (39 mm). StPancrasStatue1 is scored at the indoor threshold but its geometry reaches 100 m (24% of its GT
  depth beyond 20 m), so it is fused to 100 m on a 2 cm grid (`SPATIALARENA_FUSION_OVERRIDES` in
  `spatialarena/eval/mvrecon_evaluator.py`); no ICP. No `rgb_depth` track (the GT depth is dense).
- **SfM:** up to 300 images per scene from all 3 cameras (a fixed-seed random subset, re-sorted; the other datasets
  keep 100); `rgb_depth` priors are the LiDAR depth PNGs
  (`depth_png_scale: 100`).
- **SLAM:** frames are pre-sized to 504x406 (the 1600x1296 frames resized to 501x406 and padded by 3 columns), the
  intrinsics mapped to it. Each sequence runs only in the range its `scene.json` names (`range`), at that range's
  stride: short 3 (1.7 Hz), long 1, large 1 (`slam_protocol.SLAM_RANGES`, `DATASET_FPS=5`). `rgb_depth` runs only
  on the RGB+D sequences; the engine skips the others.

## Release

```bash
python scripts/release/pack_and_upload_benchmark_datasets.py --benchmark_root /path/to/benchmarks \
  --staging_root /path/to/fresh_staging --datasets MonoDepth/spatialarena VideoDepth/spatialarena \
  MVRecon/spatialarena SfM/spatialarena SLAM/spatialarena --stage --archive_scenes \
  --repo_id Bluefish2333/benchmark --upload
```

With `SPATIALARENA_ROOT` set, `--benchmark_root` can be any folder: the tracks are read from that root.
