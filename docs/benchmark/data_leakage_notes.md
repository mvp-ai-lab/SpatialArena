# Data-Leakage Notes

This tracks which of our evaluated datasets show up as *training* data for
the methods this repo benchmarks, based on each method's official
paper/GitHub repo (never guessed, never inferred from a dataset being
popular). It exists to explain why `scannetpp` (and `spring`, see the
MonoDepth extension datasets below) is excluded from the
aggregate ranking/leaderboard score (`LEADERBOARD_EXCLUDED_DATASETS` in
`app/webpage/python/benchmark_analyzer.py`) while staying in the
dataset registry and every per-task/per-dataset table. For scene counts,
scene type, and license per dataset, see
`docs/evaluation.md`.

## Headline finding: ScanNet++

ScanNet++ (v1: 230 official train scenes / 50 val scenes / a hidden test
set, ~1,006 scenes total) is by far the most common leak. Our own
`SfM/scannetpp` and `MVRecon/scannetpp` eval data comes from
`depth-anything/DA3-BENCH`'s 20-scene evaluation package, released by the
Depth Anything 3 (DA3) authors.

### Confirmed / plausible training-set overlap (dataset excluded from ranking because of these)

| Model | Evidence |
|---|---|
| **Pi3 / PI3_X** | No split disclosed; paper's own Table 1 footnote admits "All methods in Table 1 have seen ScanNet or ScanNet++ samples during training time" |
| **OmniVGGT** | No split disclosed anywhere; own eval section evaluates on plain ScanNet (RobustMVD protocol) instead, so authors never self-checked ScanNet++ |
| **DepthLM** | Holds out only ~10 scenes -- ~996 of ~1,006 scenes trained on |
| **SenseNova-Vision** | Training corpus filename says `scannetpp_total_...` -- suggests no split at all |
| **DepthVLM** | No split language found anywhere in the paper |
| **Fast3R** | ScanNet++ named as a training dataset with zero elaboration |
| **IGGT** | Fine-tuned on `InsScene-15K`, which sources raw imagery from ScanNet++ with no train/test discipline claimed |
| **MapAnything** | Reserves its own disjoint 30-scene ScanNet++ v2 test set (real hygiene) -- but it's MapAnything's own custom partition, not proven disjoint from our 20 scenes |
| **Déjà View / DVLT** | Explicitly claims a scene-level split safeguard for ScanNet++ -- but its held-out scenes aren't shown to match ours |
| **Pow3R** | States it copies DUSt3R's training recipe verbatim (no released training code to verify independently) |
| **CUT3R** | Reuses DUSt3R's precomputed ScanNet++ pairs file; that file's own split composition wasn't independently checkable |
| **LoGeR** | ScanNet++ is in the paper's training-data list (Table 4, 14 datasets); no split stated and no released training code |
| **Scal3R** | ScanNet++ is in the paper's training-data list (Sec. 4.3, 18 datasets); no split stated and no released training code (also evaluates on ScanNet++ in its supplementary) |

### Cleaner (kept as a valid zero-shot axis for these methods)

| Model | Evidence |
|---|---|
| **Depth Anything 3 (DA3-L/G/NGL)** | Trains on exactly 230 scenes (matches ScanNet++'s official train-split count precisely) and evaluates on a disjoint 20-scene set, with an explicit claim: "we ensure a strict separation at the scene level, i.e., scenes in training and testing are mutually exclusive." Best-evidenced split hygiene found. |
| **DUSt3R** | Code-verified: official training command uses `ScanNetpp(split='train', ...)` |
| **MASt3R** | Same code-verified `split='train'` usage (initialized from DUSt3R) |
| **VGGT / ROBUSTVGGT / FASTVGGT** | VGGT's own paper training list says "ScanNet" (v1), **not** "ScanNet++" -- confirmed by direct fetch of the paper text. No exposure to this dataset at all. |
| **CoMe** | Training-free token-merging module, no training data of its own -- risk is entirely inherited from its backbone (VGGT in our config), which is clean |

## Other datasets: no training-set overlap found

Across all ~19 distinct model families researched, no model's paper/repo
disclosed training on: NYU Depth V2, KITTI (real, non-synthetic), ETH3D,
DIODE, MPI-Sintel, Bonn RGB-D Dynamic, Microsoft 7-Scenes, DTU MVS,
Tanks-and-Temples, or IMC/Phototourism. Several papers explicitly list
these as evaluation-only in their own work (Pi3, CUT3R, Fast3R, Pow3R,
DUSt3R, MASt3R, VGGT, Depth Anything 3 all self-report KITTI/ETH3D/DTU/
NYUv2/etc. as eval-only). Treat as "no evidence of leakage found," not
"proven clean" -- absence of a training-set mention isn't a guarantee.

**`nrgbd` (Neural RGB-D Surface Reconstruction) and `tum_dynamics` (TUM
RGB-D "Dynamic Objects")**, added after this audit's main research pass,
were not named in any training-dataset list surfaced across all ~19 model
families above -- the same "no evidence of leakage found, not proven
clean" caveat applies, since these two weren't the specific subject of a
dedicated research pass the way ScanNet++ was.

**SLAM streaming methods** (LoGeR, ZipMap, LingBot-Map, R3, Scal3R,
ABot-Recon; checked 2026-10-08 against the repos at their pinned commits in
`spatialarena/download.py`) against the two SLAM datasets, `tum_dynamics`
and `7scenes`: the released training recipes of ZipMap
(`training/data/README.md`, 29 datasets), R3 (`R3/training/datasets/`) and
ABot-Recon (`train` branch, `docs/training_datasets.md`, 18 sources) list
neither; LingBot-Map names TUM-D and 7-Scenes only as evaluation sets of its
own benchmark. LoGeR and Scal3R have not released training code, but their
papers list the training data (checked 2026-10-30 in the arXiv HTML): LoGeR
(arXiv 2603.03269, Table 4) trains on DL3DV, TartanAirV2, OmniWorld-Game,
ARKitScenes (+HighRes), TartanAir, Waymo, MegaDepth, ScanNet, ScanNet++,
Virtual KITTI 2, HyperSim, Spring and UnReal4K; Scal3R (arXiv 2604.08542,
Sec. 4.3) on Co3Dv2, BlendedMVS, DL3DV, MegaDepth, WildRGB, ScanNet++,
HyperSim, Mapillary, Replica, MVS-Synth, Virtual KITTI, Aria Synthetic
Environments, Aria Digital Twin, Taskonomy, TartanAir, Mapfree, SceneNet RGB-D
and MatrixCity. Neither lists any of our SLAM datasets (TUM RGB-D, 7-Scenes,
Oxford Spires, VBR, KITTI Odometry); LoGeR evaluates on KITTI, VBR, 7-Scenes and
TUM-Dynamics, and Scal3R on KITTI Odometry, Oxford Spires and TUM-RGBD.
ZipMap, R3, ABot-Recon, LoGeR and Scal3R do train on ScanNet++
(`scannetpp`, already excluded from the ranking above; not a SLAM dataset), and
LoGeR also on Spring (see below).

**KITTI Odometry** (`kitti_odometry`, SLAM large range; checked 2026-10-17 by
grepping every repo under `<benchmark root>/_model_codes` for KITTI in
training configs and dataset lists): no released training recipe uses real
KITTI. ZipMap, R3, CUT3R, OmniVGGT, IGGT, Déjà View / DVLT and AMB3R train on
*Virtual* KITTI 2 (synthetic re-renders of five KITTI scenes, not the
recordings); ABot-Recon's `train` branch lists VKITTI2 and Waymo. R3's
`r3_long` stage also trains on **KITTI-360**, separate 2013 recordings with
the same vehicle platform in Karlsruhe -- a domain overlap, not data overlap.
LoGeR, Scal3R, LingBot-Map, VGGT-Long and Pi-Long report KITTI Odometry as an
evaluation set; LoGeR and Scal3R list Virtual KITTI (2) but not real KITTI in
their papers' training data (domain overlap only). No evidence of leakage, not
proven clean.

**MonoDepth extension datasets** (`aerialmetric`, `hammer`, `ibims1`,
`spring`, `synth4k`; checked 2026-10-16). Sources: every method repo under
`<benchmark root>/_model_codes` was grepped for the dataset names (training
configs, dataset loaders, docs), and the training-data lists of the papers
without released training code (DA3, VGGT, Pi3, Fast3R, Pow3R, DUSt3R,
MASt3R, DepthLM, IGGT, AMB3R) were searched in their arXiv HTML.

- **Spring: confirmed training overlap.** Our `MonoDepth/spring` frames are
  every 5th frame of the Spring *train* split (the test GT is withheld), and
  that split is training data of **DA3** (DA3-L/G/NGL; SPRING is in the
  paper's training-data list), **MapAnything** (`spring_wai` in its
  `megatrain_*` configs; paper Table of training sets), **CUT3R**
  (`dataset27: Spring(...)` in every training stage config), **OmniVGGT**
  (`9_400 @ Spring(...)` in `configs/train.py`), **Déjà View / DVLT**
  (`spring` in `data/mixed_all.yaml`) and **LoGeR** (Spring is in its paper's
  training-data list, Table 4; LoGeR is scored on SLAM only). `spring` is therefore excluded from the
  aggregate ranking (`LEADERBOARD_EXCLUDED_DATASETS`) like `scannetpp`, and
  stays in every per-task/per-dataset table. IGGT ships a Spring loader but no
  released config or paper list uses it; the DUSt3R-family repos (DUSt3R,
  MASt3R, Pow3R, Fast3R, AMB3R's vendored copy) only contain CroCo's
  stereo-flow Spring loader, which is not part of their depth training.
- **iBims-1, HAMMER, Synth4K, AerialMetric: no training overlap found.**
  iBims-1 appears only as evaluation data (DepthLM, DepthVLM -- whose
  curation script states "used for testing only" --, SenseNova-Vision normal
  evaluation); DepthLM's `train.sh` points at its bundled iBims-1 *example*
  folder only to demonstrate the script. No repo or paper names HAMMER,
  Synth4K / its games, or AerialMetric. AerialMetric's own Train-Split
  (Oblique / Synthetic) is the training data of MoGe2-Aerial, which is not a
  benchmarked method; only the Test-Split is used here. Déjà View and Scal3R
  train on MatrixCity aerial renders, a domain (not data) overlap with
  `aerialmetric`.
  The usual caveat applies: no evidence of leakage, not proven clean.

`HiRoom` is Depth Anything 3's own synthetic benchmark set, never used as
training data by DA3 or (as far as found) anyone else -- effectively no
risk regardless of which model is being evaluated on it.

## TexturePoorSfM: in-repo leak fixed by an object-level split

TexturePoorSfM comes from DetectorFreeSfM (He et al., CVPR 2024). Its 1,020
bags are re-samplings of 17 object captures, and bags of one object share
byte-identical frames and poses. The v1 benchmark took 30 bags across all 17
objects and left the other 990 bags as a training pool, so every v1 benchmark
frame (11-20 per object) was also a training frame.

Split v2 (`scripts/preprocess/assets/texturepoorsfm_split_v2.json`) assigns whole
objects: 12 train objects, and 5 held-out objects (1000, 1003, 1014, 1027, 1030)
that supply all 30 benchmark bags. `scripts/preprocess/consolidate_texturepoorsfm.py`
refuses a split whose benchmark bags come from non-test objects, and the WebDataset
converter writes the held-out objects only to the `val` split, recorded as
`benchmark_objects_do_not_train` in `dataset_info_scene.json`. Keep `val` out of
any training mix that is later evaluated on SfM/texturepoorsfm.

Verified on 2026-09-28: the benchmark
(`dataset_aux/benchmarks/SfM/texturepoorsfm`) contains only the 5 held-out objects,
and both training shard sets (`Data4D/texturepoorsfm/webdataset` and the padded
`webdataset_ppcentered`) contain only the 12 train objects. None of the 586
training images is byte-identical to a benchmark image, and none of the unpadded
training images matches a benchmark image pixel-for-pixel.

Remaining caveats (not frame leakage):

- The training-time `val` split *is* the benchmark's objects. Monitoring `val`
  is fine, but selecting checkpoints or hyperparameters by `val` metrics and then
  reporting SfM/texturepoorsfm tunes to the test set; hold out one of the 12
  train objects for model selection instead.
- Train and test objects share the capture setup (same tabletop and lighting,
  similar object categories), so models trained on the 12 train objects have a
  domain advantage over methods that never saw this setup.
- The benchmark's `sparse/0` points come from DetectorFreeSfM (LoFTR matching)
  triangulated against the GT poses. SfM scoring uses poses only; if points are
  ever scored, DetectorFreeSfM/LoFTR-based methods would be favored.

External leakage is still unknown: DetectorFreeSfM itself tunes on objects 1012,
1026 and 1029 (all in our train split, so not in the benchmark), and no other
model family surveyed here lists TexturePoorSfM as training data.

## Unquantifiable background risk: Qwen3-VL

Qwen3-VL is used here only as a zero-shot point-query depth baseline; its
real training data is a largely proprietary, trillion-token web-scale
multimodal pretraining corpus with no itemized dataset list. Overlap with
any of our benchmarks can neither be confirmed nor ruled out -- this is an
inherent property of evaluating any closed foundation model, not a
specific documented hit, and doesn't map onto the include/exclude
mechanism used for ScanNet++ above.

## What this means for the aggregate ranking

`scannetpp` is excluded from `build_rankings()`'s inputs
(`LEADERBOARD_EXCLUDED_DATASETS` in `benchmark_analyzer.py`) so the
majority-contaminated group above can't systematically outscore the
cleaner group in the overall leaderboard. It remains fully evaluated and
visible per-task/per-dataset (`task_tables`) as a diagnostic -- useful for
seeing how a method performs on ScanNet++-style indoor scenes, just not
folded into the headline score.

If a method above gets a public split-disclosure update (e.g. Pow3R
releases training code, DepthLM coordinates its held-out scenes with
DA3-BENCH's 20), update its row here and reconsider whether
`scannetpp`/this table needs to change.
