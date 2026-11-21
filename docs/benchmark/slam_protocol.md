# SLAM track protocol

Status: **implemented** as the `full-slam` suite (2026-10-07). It replaces the old SLAM track, which evaluated a
7-Scenes *scene* or TUM sequence exactly like SfM: a fixed-seed random 100-frame subset, RANSAC + ICP Sim(3), and
all-pairs angular metrics. That measured wide-baseline SfM, not tracking. 7-Scenes was removed from the SfM track on
2026-10-06 because both tracks produced identical cases. Old SLAM results are not comparable with this protocol.

## Cases

| Dataset | Cases | Frames per case | Ground truth |
|---|---|---|---|
| 7-Scenes | the 18 official `TestSplit.txt` sequences, one case each: chess 3, 5; fire 3, 4; heads 1; office 2, 6, 7, 9; pumpkin 1, 7; redkitchen 3, 4, 6, 12, 14; stairs 1, 4 | 1,000 (stairs 500) | KinectFusion poses shipped with the dataset |
| TUM RGB-D Dynamics | the 9 sequences | 676-3,631 | motion-capture ground truth, timestamp-associated |
| Oxford Spires (short and large range) | the 10 sequences LingBot-Map evaluates: bodleian-library 2; christ-church 2, 3, 5; keble-college 2, 3, 4, 5; observatory-quarter 1, 2 | 3,821-3,840 at 20 Hz (the first 3,840 rectified cam0 frames) | LiDAR-derived trajectory, each frame matched within 0.1 s (LingBot-Map's `preprocess/oxford.py`) |
| VBR (large range only) | the 7 training sequences of LoGeR's processed release: campus_train0, 1; ciampino_train1; colosseo_train0; diag_train0; pincio_train0; spagna_train0 | 8,815-18,846 at 10 Hz (one frame per LiDAR sweep) | LiDAR-inertial poses; row k belongs to the k-th sorted frame |
| KITTI Odometry (large range only) | sequences 00-10, the ones with public ground truth | 271-4,661 at 10 Hz | GPS/INS poses of cam0, moved to the left color camera (cam2) |

- **One case = one continuous recording.** `scripts/preprocess/materialize_7scenes_slam.py` builds one case folder
  per test sequence, `SLAM/7scenes/<scene>_seq-NN/`. Its `images/` and `depths/` are relative symlinks into
  `MVRecon/7scenes`, so no data is duplicated; only the sequence's own `sparse/0` (a few KB) is written. The
  `MVRecon/7scenes` scene folders mix every recording of a scene, which is why SLAM cannot use them directly. MVRecon
  keeps the whole scenes and is unaffected (the SLAM side only reads them).
- **Depth priors** (`rgb_depth`) are converted with each dataset's declared `depth_png_scale` (`metadata.json`):
  1000 per metre for 7-Scenes, 5000 for TUM.
- **KITTI Odometry** follows LoGeR (`eval/run_kitti.sh`), LingBot-Map (`benchmark/datasets/kitti.py`) and
  VGGT-Long: the left color camera (`image_2`) of sequences 00-10, every frame, in the large range only
  (`scripts/download/kitti_odometry.sh`, `scripts/preprocess/materialize_kitti_odometry.py`). The ground truth is
  cam0's trajectory; cam2 is rectified to the same orientation and sits about 6 cm to its left, so the
  materializer moves every pose to cam2 through `P2 = K2 [I | t]` (`T_world_cam2 = T_world_cam0 [I | -t]`) instead
  of leaving the offset to the Sim(3) alignment as LingBot-Map does. It has no depth, so only the `rgb` track runs.
  Its longest sequence (02, 4,661 frames) is below the large range's VBR bound, so no method's track membership
  changes.
- The `full-pose` case list (`full_standard_sfm_cases.txt`) is SfM only.

## Range tracks

The SLAM track is split into three range tracks. Each fixes one temporal stride per dataset and has its own case
list (37 short, 27 long, 28 large cases), experiment name (`benchmark_full-slam-<range>_<METHOD>`) and ranking.

| Range | Stride | Input rate | Case list | 7-Scenes frames | TUM frames | Longest input |
|---|---:|---:|---|---|---|---:|
| **short** | 10 | 3 Hz | `full_standard_slam_cases_short.txt` | 100 (stairs 50) | 68-122, fr2/desk 364 | 364 |
| **long** | 3 | 10 Hz | `full_standard_slam_cases_long.txt` | 334 (stairs 167) | 226-406, fr2/desk 1,211 | 1,211 |

The short range also feeds Oxford Spires at stride 12 (1.7 Hz, LingBot-Map's `oxford.yaml`; 319-320 frames, inside
the TUM-defined bound). The **large** range (`full_standard_slam_cases_large.txt`) feeds Oxford Spires, VBR and
KITTI Odometry at every frame (stride 1): up to 3,840, 18,846 and 4,661 frames; its bound is VBR ciampino_train1's
18,846.

- **Whole sequences, single pass.** Every `s`-th frame is fed in recording order, with no random sampling, no
  100-frame cap (`BENCHMARK_MAX_INPUT_FRAMES=0`) and no subsampling (`MAX_IMAGES_PER_RUN=0`). Feed-forward methods
  are never windowed.
- **A track's methods run every sequence of it.** A method joins a range track when its measured single-pass
  budget on one H200 covers the track's longest input (`spatialarena/slam_protocol.py`), or when it is attempted
  beyond its budget (below); there are no per-sequence skips, a sequence that fails is recorded with its cause, and
  only methods that ran every sequence are ranked. The bound grows from short to long, so **every long-range method
  is also evaluated on the short-range track**.
- **Long-sequence SLAM systems** (VGGT-Long, Pi-Long, VGGT-SLAM) chunk internally and have no budget, so they are in
  both tracks. They run unmodified with their published settings (`scripts/inference/long_sequence_slam.py`).
- **Tracks:** `rgb` and `rgb_depth` (sensor depth as the prior), both scored after Sim(3) alignment. The
  long-sequence systems take RGB only.

### Methods per range track

Budgets are the largest input that ran on one H200 (143 GB GPU memory, 180 GB host RAM) through the real benchmark
inference path. They were measured on 2026-10-06 with `scripts/test/profile_max_input_frames.sh` on TUM
fr2/desk_with_person at 640x480 (the resolution of both datasets), for N in 100, 200, 300, 400, 500, 750, 1000,
1500, 2000 and 3000.

| Method | H200 budget (frames) | Peak at budget (GB) | Short range | Long range |
|---|---:|---:|:---:|:---:|
| PI3_X | >= 3000 | 139.2 | yes | yes |
| CUT3R | 2000 | 107.7 | yes | yes |
| FAST3R | 2000 | 137.7 | yes | yes |
| DA3-L | 1500 | 116.9 | yes | yes |
| COME | 1500 | 138.2 | yes | yes |
| VGGT_OMEGA | 1500 | 128.2 | yes | yes |
| VGGT_LONG, PI_LONG, VGGT_SLAM | unbounded (chunked) | - | yes | yes |
| DA3-G, DA3-NGL | 1000 | 124.0, 125.2 | yes | - |
| VGGT, FASTVGGT, ROBUSTVGGT, IGGT, OMNIVGGT | 400 | 132.7-139.6 | yes | - |
| AMB3R | 400 (500 fails in spconv, not memory) | 86.5 | yes | - |

That is **17 short-range methods and 9 long-range methods** by budget. With the streaming methods and the
methods attempted beyond their budget (below), the `rgb` track has 29 short-range, 21 long-range and 9
large-range methods (`slam_protocol.range_track_methods`); `rgb_depth` has 3, 2 and none.

**Attempted beyond budget (large range):** LoGeR, R3, ZipMap and Scal3R run every large-range sequence although the
longest one (VBR ciampino_train1, 18,846 frames) exceeds their budget, since most Oxford Spires and VBR sequences are
much shorter (`slam_protocol.BEYOND_BUDGET_ATTEMPTS`). A failed sequence is recorded with its cause (`gpu_oom`,
`host_oom` or `error`) in the `reason` column of `case_failures*.tsv` and in the case's `inference_failure.tsv`, next
to its `inference.log`. A method with failed sequences is reported with its coverage but not ranked on the range.
A run that finishes but returns fewer than three poses cannot be aligned and is recorded the same way, with the
evaluator's error ("Only 1 of N input frames have a predicted pose") in its `eval_slam.json`: VGGT-SLAM keeps a
keyframe only after 50 px of disparity, so on near-static recordings (TUM `sitting_static`, `walking_static`) and
the short straight KITTI sequence 04 it never adds a second one.

**Attempted on the short and long ranges (budget below the short range's longest input, or not measured):** these
run every sequence of the short and long ranges (`slam_protocol.ATTEMPTED_SHORT_AND_LONG`), one case at a time; a
sequence that fails is recorded with its cause as above, so the method is reported with its coverage instead of
being left out. They are not in the large range: its sequences (3,821 to 18,846 frames) run out of memory for all
of them, so they are not attempted and have no row in its tables.
- MAPANYTHING (budget >= 300; larger sizes not tested), on `rgb` and `rgb_depth`.
- MAST3R (100).
- DEJAVIEW (100; 200 fails with `input tensor must fit into 32-bit index math`).
- DUST3R and POW3R (< 100: complete-graph global alignment exceeds 180 GB host RAM at 100 frames).
- IGGT4D (not measured).

**Not evaluated on the SLAM track:** GENCEPTION_POSE (always resamples to 81 frames).

## Input resolution

Every method sees the same pixels on a dataset, so methods are compared at one resolution instead of each one's own
resize (`slam_protocol.DATASET_INPUT_SIZE`):

| Datasets | Native frames | Input | Patch grid |
|---|---|---|---|
| TUM RGB-D, 7-Scenes | 640×480 | **504×378** | 36×27 at 14 px, exactly 4:3 |
| Oxford Spires | 1440×1080 (rectified) | **504×378** | 36×27 at 14 px, exactly 4:3 |
| VBR | 1388×700 | **504×252** | 36×18 at 14 px; resized to fit 2:1 preserving aspect, then zero-padded (2 columns each side) -- no content lost |
| KITTI Odometry | 1241×376, 1242×375 (03), 1226×370 (04-10) | **504×154** | 36×11 at 14 px; resized to fit 504:154 preserving aspect, then zero-padded (0-1 rows top/bottom) -- no content lost |

- **Why 504.** 504 is the only width near the 512–518 most methods were trained at that has an exactly 4:3 height on
  the 14 px patch grid (518 would need 388.5 rows). It is R3's and LoGeR's native size, ABot-Recon's width, and
  under LingBot-Map's and LoGeR's 255,000-pixel budgets. No size between 504 and 512 is also a multiple of 16
  (both 14 and 16 divide only multiples of 112), so the three models with 16 px patches (CUT3R, Fast3R,
  VGGT-Omega) center-crop to 496×368 (496×240 on VBR, 496×144 on KITTI) themselves: the same pixels, a 4–6 px border less, never
  rescaled.
- **Written once.** `scripts/preprocess/materialize_slam_inputs.py` writes every sequence at its input size under
  `<benchmark_root>/SLAM/<dataset>/<scene>/` from the native-resolution sequences under `SLAM_source/`: `images/`
  (resized to fit the target aspect preserving the source content, then zero-padded with `cv2.INTER_AREA`),
  `depths/` for the `rgb_depth` priors (the same resize+pad, nearest neighbor, units unchanged) and
  `sparse/0` with the intrinsics mapped through the same scale and pad offset. File names and ground-truth poses are
  unchanged, so `SLAM/` is the benchmark's SLAM data: methods read their frames and the evaluator its ground truth
  there (the released copy; `SLAM_source/` stays local).
- **Fed as they are.** In `full-slam` the engine points every adapter at those frames, sets
  `RESOLUTION=IMAGE_LOAD_RESOLUTION=504` and `FIXED_INPUT_RESOLUTION=1`, and each adapter keeps the frames
  unrescaled. Adapters that require a square canvas (VGGT, RobustVGGT, FastVGGT, OmniVGGT, Co-Me) pad them to
  504×504, which adds no image content. The methods whose own code fixes another size are pinned to the input:
  - AMB3R: its fixed 518×392;
  - LoGeR, ZipMap, Scal3R and LingBot-Map: their 512/518 sizes and pixel budgets;
  - ABot-Recon: its 504×280 canvas (its `height` override);
  - Pi3 on `rgb_depth`: a pixel budget of exactly width × height;
  - IGGT: its square resize;
  - VGGT-Long and VGGT-SLAM: VGGT's loader hardcodes a 518 px width. `scripts/inference/fixed_input_launcher.py`
    swaps that one function at launch, so their repositories stay unedited.
- **Checked.** `scripts/test/probe_slam_input_sizes.py` runs each method's real preprocessing (the pinned upstream
  loaders with the adapter's settings) on frames of every protocol size. It confirms on CPU that every one delivers the
  input unchanged, padded or 16 px-cropped, to within one 8-bit level. Adapters also record the tensor size the
  model received as `model_input_size` in each case's `metrics.json`.
- **Not the papers' resolutions.** The published numbers use each method's own size (e.g. LingBot-Map 518×378 on TUM
  and Oxford, 504×280 on VBR; LoGeR 714×350 on VBR), so results here are not directly comparable with them. They
  are comparable with each other.

## Alignment

- **One Sim(3) alignment for every track: RANSAC + GTSAM pose fit + robust refit**
  (`pose_alignment.ransac_irls_align_sim3`, `slam_trajectory_evaluator --alignment_estimator ransac_gtsam_irls`,
  the default). It runs in three stages:
  1. LMedS RANSAC over the pose-aware cost of GTSAM's `TrajectoryAlignerSim3`
     (`fit_sim3_pose_residuals`: rotation geodesic + translation error per frame). It uses minimal samples of 2
     frames, 25 iterations and seed 42, and refits on the inliers.
  2. A Cauchy-weighted IRLS refit of the same cost on all frames (`irls_refine_sim3`). Each iteration
     re-estimates robust scales (1.4826 × MAD) for the rotation and translation residuals, then re-solves with
     Levenberg-Marquardt until the parameters stop moving.
  3. A guard against scale collapse. When a prediction's camera centers disagree with the ground truth under the
     rotation its orientations choose, the pose cost keeps falling as the scale shrinks, ending with every frame on
     the ground-truth centroid. That optimum says nothing about the prediction. If the fitted scale falls below 1 %
     of a center-only fit's (Umeyama in the same RANSAC harness, minimal sample 3), the center-only fit is used, as
     evo would, and the case records `alignment_scale_collapsed = 1`. It only fires on trajectories that fail
     outright: in the first short-range runs, 3 Oxford Spires cases out of 222 (median rotation residual 67–99°).

  RANSAC escapes tracking-loss jumps and gross outliers, and using rotations keeps the fit well-conditioned on
  near-static and near-collinear trajectories. The refit makes the answer independent of the RANSAC seed and as
  precise as a full least-squares fit on clean data. There is no ICP step: ICP needs ground-truth point clouds to
  do real work, and without them it would only re-fit camera centers whose correspondence is already known. The
  previous estimator (RANSAC over the DAGSfM fit, refit on the inliers) remains available as
  `--alignment_estimator ransac_dagsfm`. The study behind the choice is below.
- **ATE and RPE use every returned frame.** The alignment only decides how frames are weighted in the fit; frames
  it down-weights (e.g. after a tracking failure) still count in ATE. `alignment_inlier_ratio` records the fraction
  of frames whose final normalized error is within the Cauchy constant (weight ≥ 1/2). `rgb_depth` uses Sim(3)
  too: even with metric depth priors, most feed-forward methods return poses whose scale is not consistent with
  the ground truth, and a fixed-scale fit would mostly measure that scale offset rather than the trajectory.
- **Scale is reported separately.** Every case also stores the ATE after the same alignment with the scale fixed
  to 1 (SE(3), `ate_rmse_se3`) and the Sim(3) scale (`alignment_scale`). On `rgb_depth`, a small Sim(3)-vs-SE(3)
  gap and a scale near 1 show that a method adopted the metric scale of its depth priors.

### Choice of estimator: simulation study

Published SLAM numbers (LingBot-Map, ABot-Recon and most TUM / 7-Scenes tables) use evo's Umeyama Sim(3)
(`PoseTrajectory3D.align(correct_scale=True)`: closed-form least squares on all camera centers). We compared it with
the RANSAC alternatives on simulated predictions whose true alignment is known:
`scripts/test/compare_trajectory_alignment.py --mode synthetic --trials 30 --frames 300` (CPU, seed 0; 3,450
trials; 85 min on one core). The estimators:

| Name | Estimator |
|---|---|
| evo | evo Umeyama Sim(3) on all camera centers |
| R-DAGSfM | LMedS RANSAC over `fit_sim3_dagsfm`, sample 2, 25 iterations, refit on inliers (previous default) |
| GTSAM | `fit_sim3_pose_residuals` on all frames, no RANSAC |
| R-GTSAM | the same RANSAC harness and settings around `fit_sim3_pose_residuals` |
| GTSAM+refit | GTSAM followed by the IRLS refit (deterministic, no RANSAC) |
| **R-GTSAM+refit** | R-GTSAM followed by the IRLS refit (**current default**) |

The ground-truth trajectories have four shapes: handheld, near-static, driving and loop. Each one is mapped
through a random Sim(3) (scale 0.05–20) and corrupted in one of these ways:

- **noise:** Gaussian noise on the centers (σ as a fraction of the trajectory extent), plus 1° orientation noise;
- **drift:** odometry random walk: every relative motion is perturbed (translation noise as a fraction of the mean
  step, rotation noise 10× that in degrees) and integrated;
- **scale drift:** relative translations are scaled from 1 to 1 + level along the sequence;
- **jump:** tracking loss: the last fraction of frames is offset by a rigid 20° / 0.3-extent jump;
- **outliers:** a fraction of frames is replaced by random poses within twice the extent (0.5 % noise on the rest);
- **rot noise:** orientation-only noise of the given angle (0.5 % center noise).

A near-collinear family covers a 100 m straight path whose lateral deviation is ε × length (ε = 0 … 0.05). On
it, rotation about the path is barely observable from camera centers.

The metrics are:

- **align error:** RMS distance between the estimated and the true alignment over the ground-truth centers, in
  trajectory extents;
- **rotation error:** angle between the estimated and the true alignment rotation;
- **ATE ratio:** ATE after the estimated alignment divided by ATE after the true one. 1.0 is exact; below 1 the
  alignment absorbs (hides) error; above 1 it inflates it;
- **seed CV:** coefficient of variation of the ATE over 10 RANSAC seeds, on the first 5 trials of each setting.

Medians over the four shapes (120 trials per row). Rotation error is the median in degrees; worst cases are in
the next table.

| Scenario | evo | R-DAGSfM | GTSAM | R-GTSAM | GTSAM+refit | **R-GTSAM+refit** |
|---|---|---|---|---|---|---|
| outliers 45 %: rotation error | 20.4° | 0° | 11.7° | 0.07° | 0° | **0°** |
| outliers 45 %: ATE ratio | 0.44 | 1.00 | 0.44 | 1.00 | 1.00 | **1.00** |
| jump 45 %: rotation error | 33.3° | 0° | 20.1° | 0° | 8.2° | **0°** |
| jump 45 %: ATE ratio | 0.45 | 1.00 | 0.46 | 1.00 | 0.60 | **1.00** |
| jump 5 %: ATE ratio | 0.87 | 1.00 | 0.87 | 1.00 | 1.00 | **1.00** |
| noise 5 %: rotation error | 0.74° | 0.09° | 0.31° | 0.31° | 0.09° | **0.09°** |
| rot noise 30°: ATE ratio | 1.00 | 2.63 | 1.01 | 1.01 | 1.00 | **1.00** |
| rot noise 30°: rotation error | 0.07° | 2.92° | 0.29° | 0.30° | 0.07° | **0.07°** |
| drift 10 %: ATE ratio | 0.34 | 0.45 | 0.36 | 0.44 | 0.35 | **0.35** |
| scale drift 60 %: ATE ratio | 0.36 | 0.49 | 0.36 | 0.48 | 0.40 | **0.40** |
| drift 2–10 %: seed CV (p90) | — | 0.10–0.20 | — | 0.09–0.20 | — | **≤ 1e-5** |
| scale drift 10–60 %: seed CV (p90) | — | 0.25–0.35 | — | 0.27–0.38 | — | **≤ 2e-5** |
| rot noise 30°: seed CV (p90) | — | 0.26 | — | 0.05 | — | **5e-10** |

Worst case over all trials of a scenario, including the collinear family (maximum rotation error / maximum ATE
ratio):

| Scenario | evo | R-DAGSfM | R-GTSAM | GTSAM+refit | **R-GTSAM+refit** |
|---|---|---|---|---|---|
| outliers 15 % | 179° / 0.74 | 0.000003° / 1.00 | 1.66° / 1.00 | 0.000002° / 1.00 | **0.000002° / 1.00** |
| jump 45 % | 84° / 0.71 | 0.000003° / 1.00 | 0.000002° / 1.00 | 10.8° / 0.74 | **0.000002° / 1.00** |
| rot noise 10° | 180° / 1.00 | 2.2° / 2.41 | 1.89° / 1.96 | 1.46° / 1.00 | **1.46° / 1.00** |
| rot noise 30° | 0.34° / 1.00 | 7.0° / 5.67 | 4.85° / 5.31 | 0.34° / 1.00 | **0.34° / 1.00** |
| noise 1 % (collinear) | 167° / 1.00 | 0.20° / 1.01 | 2.62° / 1.00 | 0.16° / 1.00 | **0.16° / 1.00** |
| drift 10 % | 47.6° / 0.76 | 45.9° / 2.02 | 45.7° / 2.02 | 36.9° / 0.91 | **36.9° / 0.91** |

Near-collinear paths, median rotation error (30 trials per row; "crash" = evo raises `GeometryException`, its
covariance has rank < 2):

| ε | Scenario | evo | R-DAGSfM | GTSAM | R-GTSAM | **R-GTSAM+refit** |
|---|---|---|---|---|---|---|
| 0 | noise 1 % | crash (30/30) | 0.09° | 0.07° | 0.07° | **0.07°** |
| 0 | outliers 30 % | crash (30/30) | 0° | 7.0° | 0.09° | **0°** |
| 1e-4 | noise 1 % | 57.2° | 0.08° | 0.11° | 0.11° | **0.08°** |
| 1e-4 | jump 30 % | 131° | 0° | 21.9° | 0° | **0°** |
| 1e-3 | noise 1 % | 11.5° | 0.09° | 0.13° | 0.13° | **0.07°** |
| 1e-3 | outliers 30 % | 86.3° | 0° | 5.1° | 0.13° | **0°** |
| 1e-2 | noise 1 % | 1.62° | 0.11° | 0.61° | 0.61° | **0.09°** |
| 5e-2 | outliers 30 % | 35.8° | 0° | 23.0° | 0.27° | **0°** |

What the numbers show:

- **evo is not robust.** One tracking-loss jump or a few bad frames drag the whole alignment, and the reported ATE
  understates the true error (ATE ratio 0.44–0.87): the method looks better than it is. On exactly straight paths
  evo crashes. On nearly straight ones it returns rotations off by tens of degrees, because roll about the path is
  unobservable from camera centers.
- **The previous default (R-DAGSfM) is robust but noisy.** It averages rotations from minimal samples, so
  orientation noise inflates its ATE (2.6× at 30°). Under drift, where no subset of frames is exactly right, its
  answer depends on the RANSAC seed (ATE varies by 10–35 % across seeds).
- **R-GTSAM fixes the orientation-noise bias but not the seed dependence.** Its 25-iteration sample choice still
  decides the answer under drift.
- **The refit removes the seed dependence (CV ≤ 2e-5) and is as precise as evo on clean data**, while staying exact
  under jumps and outliers and correct on (near-)collinear paths. On drift, it absorbs slightly more error than
  RANSAC alone (ATE ratio ≈ 0.35–0.49 vs evo's 0.33–0.45). Any Sim(3) fitted over all frames absorbs part of a
  smooth drift; this keeps our ATE close to evo-based published numbers on well-behaved runs.
- **RANSAC is still needed before the refit.** Started from the plain all-frames fit, the refit gives the same
  answer in every setting except the heaviest jumps (45 %), where it stays in the wrong basin (8–11° rotation
  error).
- **25 iterations are enough.** R-DAGSfM with 200 iterations differs from 25 only in the seed CV. The refit makes
  the iteration count irrelevant for the answer, as long as one sample lands in the right basin.

A second run (`--mode real_gt`) applies the same corruptions to the real ground-truth trajectories of TUM and
7-Scenes at the short and long strides (10 and 3; 54 sequences, 50–364 frames, 10 trials each, so 540 trials per
row). Median rotation error / maximum rotation error (degrees), median ATE ratio, and the 90th percentile of the
seed CV:

| Scenario | evo | R-DAGSfM | R-GTSAM | **R-GTSAM+refit** |
|---|---|---|---|---|
| outliers 45 % | 25.4° / 179° · 0.38 | 0° / 0.3° · 1.00 | 0.04° / 0.6° · 1.00 | **0° / 0° · 1.00** |
| jump 45 % | 23.4° / 49° · 0.51 | 0° / 0° · 1.00 | 0° / 0° · 1.00 | **0° / 0° · 1.00** |
| jump 5 % | 3.6° / 10.7° · 0.90 | 0° / 0° · 1.00 | 0° / 0° · 1.00 | **0° / 0° · 1.00** |
| noise 5 % | 1.02° / 4.8° · 0.97 | 0.11° / 0.4° · 0.98 | 0.19° / 0.9° · 0.98 | **0.11° / 0.5° · 0.98** |
| rot noise 30° | 0.11° / 0.4° · 0.99 | 3.7° / 13° · 2.89, CV 0.12 | 2.8° / 12.7° · 2.17, CV 0.12 | **0.10° / 0.4° · 0.99, CV 1.5e-9** |
| drift 10 % | 12.4° / 77° · 0.43 | 12.2° / 63° · 0.52, CV 0.08 | 12.2° / 65° · 0.50, CV 0.10 | **12.1° / 68° · 0.45, CV 1.5e-7** |
| scale drift 60 % | 1.32° / 2.2° · 0.32 | 0° / 0° · 0.37, CV 0.10 | 0.14° / 0.9° · 0.36, CV 0.11 | **0° / 0° · 0.33, CV 1e-5** |

The real trajectories confirm the synthetic study. The default is exact under outliers and jumps, where evo
understates the error by up to 2.6×. It matches evo under noise, and it is the only RANSAC variant without
orientation-noise inflation or seed dependence (largest seed CV over all 10,800 trials: 0.004, against 0.37 for
R-DAGSfM). Under smooth drift no Sim(3) alignment recovers the true one: rotation errors of 12° are common to all
estimators, and the refit absorbs the least extra error of the RANSAC variants (ATE ratio 0.45 vs evo's 0.43).
Generated by `scripts/test/compare_trajectory_alignment.py --mode real_gt` (2026-10-09).

## Metrics

All metrics come from `spatialarena/eval/slam_trajectory_evaluator.py`, which writes `eval_slam.json` per case.
The scored frames are exactly the method's input frames: the scene's image files listed and strided as the adapters
do (`find_image_paths`, every stride-th). A sequence is scored only when its images and GT poses match one to one; an
image without a GT pose, or a GT pose without an image, would shift every strided frame after it, so the case fails
with the names of the unmatched frames instead. The case's alignment (scale, rotation, translation) is saved as
`alignment_transform` and reused by the 3D trajectory viewer, which fits the alignment itself only when it is absent.

| Metric | Definition |
|---|---|
| `ate_rmse` (m) | RMSE of aligned camera-center errors over the returned frames (also `ate_mean`, `ate_median`) |
| `drift_pct` | sum of 1-second translational RPE divided by the ground-truth path length over the same windows, in percent (a ratio of sums, so near-static segments cannot blow it up) |
| `rpe_trans_step` (m), `rpe_rot_step` (deg) | RMSE relative pose error between consecutive input frames |
| `rpe_trans_1s` (m), `rpe_rot_1s` (deg) | RMSE relative pose error over one second (30 / stride input frames) |
| `auc_1` ... `auc_30` | pose AUC exactly as the SfM track (`sparse_reconstruction_evaluator.compute_auc`): for frame pairs, the error is max(relative rotation angle, relative translation direction angle) in degrees, and `auc_T` is the area under its recall curve up to T degrees, normalized to [0, 1] (T = 1, 3, 5, 10, 20, 30). Both angles are unchanged by a similarity transform, so the AUC does not depend on the trajectory alignment. Pairs: all pairs of the returned frames, or of 1,000 of them evenly spaced in time on longer sequences (499,500 pairs; `--auc_max_frames`) |
| `completeness` | fraction of input frames with a returned pose |

RPE pairs are formed by input-frame index, so a method that skips frames (e.g. keyframe-based VGGT-SLAM) is scored on
the pairs it returned without distorting the time offset. ATE, RPE and AUC use the returned frames only, which is why
completeness is reported with them. The ranking stays ATE plus drift (below); AUC measures local and global
relative-pose accuracy on the same footing as the SfM track and is reported, not ranked. Runtime and peak GPU memory
come from each case's adapter `metrics.json`; they are reported, not ranked. Long- and large-range cases run one per
GPU (short-range cases two), so those figures are the case's own (`run_method_benchmark_infer.sh`).

## Ranking

The summary is `scripts/test/summarize_benchmark_reports.py` on the `full-slam-short` / `full-slam-long` reports. It
is computed per range track and dataset:

1. Each metric is averaged over the track's sequences, without weighting.
2. Methods are ranked by ATE RMSE and by drift. **Rank score = mean of the two ranks** (lower is better; ties share
   the mean rank).
3. A method that failed on any sequence is reported with its coverage (`sequences run / total`) but not ranked.

## Where the protocol lives in code

`spatialarena/slam_protocol.py` is the single definition of the SLAM protocol. It holds the range tracks and their
strides, the H200 budgets, track membership and exclusions, the alignment per input track, and the dataset frame
rates. Everything else reads from it:

| Consumer | What it takes from `slam_protocol` |
|---|---|
| `run_method_case_list.sh` (`full-slam`) | stride of `SLAM_RANGE`, per-case membership check (`skipped_not_in_slam_track`), alignment for `BENCHMARK_TRACK` |
| `task_track_matrix.runnable_tracks` / `BenchmarkEvaluator.plan()` | SLAM is offered only to range-track methods, with one work item per range the method is in |
| `generate_full_standard_cases.py --slam_range short\|long` | case-list interval = the track's stride (`--tasks SLAM` without `--slam_range` is refused) |
| `slam_trajectory_evaluator.py` | dataset frame rates for the 1-second RPE |

SLAM is evaluated only in `full-slam`. The SfM evaluator refuses `SLAM`, and the engine refuses a SLAM case in any
other suite. Tests: `test/test_slam_protocol.py`, `test/test_slam_trajectory_evaluator.py`, and the SLAM cases in
`test/test_evaluator.py` and `test/test_benchmark_report_summarizer.py`.

## Running

```bash
# Once per dataset (CPU): the fixed-resolution inputs every method reads (see Input resolution).
python scripts/preprocess/materialize_slam_inputs.py --datasets tum_dynamics 7scenes oxford_spires vbr kitti_odometry
# Inference (GPU) then evaluation (CPU) for one method on one range track; canonical artifacts go to
# _evaluation/artifacts/full_standard/benchmark_full-slam-<range>_<METHOD>.
METHOD=PI3_X BENCHMARK_SUITE=full-slam SLAM_RANGE=long bash spatialarena/templates/_engine/run_method_benchmark_infer.sh
METHOD=PI3_X BENCHMARK_SUITE=full-slam SLAM_RANGE=long bash spatialarena/templates/_engine/run_method_benchmark_eval.sh
python -m spatialarena.slam_protocol methods long   # the track's methods
```

A method outside the track is recorded as `skipped_not_in_slam_track` for every case. In the Python API
(`BenchmarkEvaluator`), SLAM work items map to `full-slam`, and the item's `slam_range` (default `short`) selects the
track.

## Open items

- `online` (causal) setting: not part of this version. Only CUT3R has a native streaming mode among the track's
  methods.
- The dashboard UI does not render the SLAM tables yet. The collector already reads `eval_slam.json` through the
  summarizer.
