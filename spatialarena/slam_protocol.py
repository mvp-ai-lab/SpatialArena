# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""The SLAM track protocol: range tracks, strides, evaluated methods, alignment (docs/benchmark/slam_protocol.md).

Single source of truth for everything SLAM-specific in the pipeline. Its users:

- ``run_method_case_list.sh`` (``full-slam`` suite), via this module's CLI: the stride of each dataset in
  ``SLAM_RANGE``, per-case eligibility, and the alignment for the track.
- ``BenchmarkEvaluator`` (``spatialarena/evaluator.py``) and ``task_track_matrix.runnable_tracks``: which
  (method, track, range, dataset) SLAM work items to plan.
- ``scripts/test/generate_full_standard_cases.py --slam_range``: the case lists' datasets and intervals.
- ``spatialarena/eval/slam_trajectory_evaluator.py``: per-dataset strides and frame rates (for the 1-second RPE).
- ``scripts/preprocess/materialize_slam_inputs.py`` and the engine: the fixed input resolution of each dataset
  (``DATASET_INPUT_SIZE``) and where its pre-sized frames live (``input_root``).

The SLAM track has three range tracks. Each fixes, per dataset, the temporal stride a sequence is fed at:

- ``short``: TUM / 7-Scenes stride 10 (3 Hz); Oxford Spires stride 12 (LingBot-Map's ``oxford.yaml``). Longest
  input: 364 frames (TUM fr2/desk_with_person; 7-Scenes 50-100; Oxford <= 320).
- ``long``: TUM / 7-Scenes stride 3 (10 Hz). Longest input: 1,211 frames (TUM fr2/desk_with_person).
- ``large``: the large-scale outdoor recordings at every frame, as the long-sequence papers evaluate them:
  Oxford Spires stride 1 (the first 3,840 cam0 frames, LingBot-Map's ``oxford_long.yaml`` / ABot-Recon's pose
  protocol), VBR stride 1 (LoGeR's processed sequences, LoGeR / LingBot-Map) and KITTI Odometry stride 1 (sequences
  00-10, the left color camera; LoGeR / LingBot-Map / VGGT-Long). Longest input: 18,846 frames
  (VBR ciampino_train1); Oxford sequences are <= 3,840, KITTI <= 4,661 (sequence 02).

Every method in a range track runs every sequence of it in a single pass -- feed-forward methods are never windowed
or skipped per sequence. A method belongs to a range track when its measured single-pass budget on one H200 covers
the track's longest input; since that bound grows from short to long to large, a method of a longer range is also
in every shorter one. Long-sequence systems that chunk or stream internally (VGGT-Long, Pi-Long, VGGT-SLAM and the
streaming methods of scripts/inference/streaming_slam.py, once measured) have no budget and belong to all three.
Methods in ``EXCLUDED_METHODS`` are not evaluated on the SLAM track at all.

``BEYOND_BUDGET_ATTEMPTS`` adds methods to a range whose longest input exceeds their measured budget: the
long-context methods on the large range, whose sequences mostly fit (Oxford <= 3,840 frames, most VBR sequences well
under the 18,846-frame maximum), and the methods whose budget is below even the short range's longest input (or not
measured), on the short and long ranges (``ATTEMPTED_SHORT_AND_LONG``; the large range's thousands of frames per
sequence are an out-of-memory failure for all of them, so they are not in it). They run every sequence of the range;
a sequence that fails (GPU or host OOM, or another error) is recorded with its cause in the run's
``case_failures*.tsv``, and the method is reported with its coverage but, missing those sequences, not ranked on the
range.

Budgets are the largest input that ran through the real inference path on one H200 (143 GB, 180 GB host RAM),
measured with ``scripts/test/profile_max_input_frames.sh`` on TUM fr2/desk_with_person at 640x480 (the resolution of
both SLAM datasets) on 2026-10-06, testing 100, 200, 300, 400, 500, 750, 1000, 1500, 2000 and 3000 frames.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from spatialarena.spatialarena import SPATIALARENA, track_root_override

UNBOUNDED = math.inf

# Per range: the stride of each dataset in it, and the longest strided sequence over those datasets
# (ceil(frames / stride); TUM fr2/desk_with_person has 3,631 frames, Oxford Spires sequences <= 3,840 after
# preprocessing). The "large" bound is the longest VBR sequence (ciampino_train1, 18,846 frames). SpatialArena
# (5 Hz) is in all three, but each of its sequences only in the range its metadata names (``slam_range`` in
# dataset_registry.describe_benchmark_scenes).
SLAM_RANGES: dict[str, dict] = {
    "short": {"strides": {"7scenes": 10, "tum_dynamics": 10, "oxford_spires": 12, "spatialarena": 3},
              "max_frames": 364},
    "long": {"strides": {"7scenes": 3, "tum_dynamics": 3, "spatialarena": 1}, "max_frames": 1211},
    "large": {"strides": {"oxford_spires": 1, "vbr": 1, "kitti_odometry": 1, "spatialarena": 1}, "max_frames": 18846},
}

H200_FRAME_BUDGET: dict[str, float | None] = {
    "PI3_X": 3000,  # no failure up to the largest size tested
    "CUT3R": 2000,
    "FAST3R": 2000,
    "DA3-L": 1500,
    "COME": 1500,
    "VGGT_OMEGA": 1500,
    "DA3-G": 1000,
    "DA3-NGL": 1000,
    "VGGT": 400,
    "FASTVGGT": 400,
    "ROBUSTVGGT": 400,
    "IGGT": 400,
    "OMNIVGGT": 400,
    "AMB3R": 400,  # 500 fails in spconv (kernel limit, not memory)
    "VGGT_LONG": UNBOUNDED,
    "PI_LONG": UNBOUNDED,
    "VGGT_SLAM": UNBOUNDED,
    # Streaming / long-context methods, measured 2026-10-09 on the fixed inputs (TUM fr2/desk_with_person at 504x378
    # up to its 3,631 frames; VBR ciampino_train1 at 504x252 up to 18,846); peak = nvidia-smi, whole process.
    "LINGBOT_MAP": 18846,  # 44 GB at 18,846
    "ABOT_RECON": 18846,  # 8 GB at 18,846
    "LOGER": 8000,  # 18 GB at 8,000; at 18,846 its eval wrapper exceeded 180 GB host RAM (re-measured without it)
    "R3": 8000,  # 104 GB at 8,000; GPU OOM at 18,846
    "ZIPMAP": 1211,  # 109 GB at 1,211 (one offline pass); GPU OOM at 3,631
    "SCAL3R": 1211,  # 17 GB GPU at 1,211 (2,000 s); its process exceeded 180 GB host RAM at 3,631 and 3,840
    # Below the short range's longest input (364 frames), measured 2026-10-06, or not measured (None): attempted on
    # the short and long ranges anyway (ATTEMPTED_SHORT_AND_LONG), each failed sequence recorded with its cause.
    "MAPANYTHING": 300,  # >= 300; larger sizes not tested
    "MAST3R": 100,
    "DEJAVIEW": 100,  # 200 fails on 32-bit index math
    "DUST3R": None,  # < 100: complete-graph global alignment exceeds 180 GB host RAM at 100 frames
    "POW3R": None,  # < 100, as DUST3R
    "IGGT4D": None,  # not measured yet
}

# Not evaluated on the SLAM track: GENCEPTION_POSE (always resamples to 81 frames). Methods that predict no camera
# pose (depth-only VLMs, GENCEPTION_DEPTH) have no budget entry and are never in a track either.
EXCLUDED_METHODS = frozenset({"GENCEPTION_POSE"})

# Run on the short and long ranges although their budget is below the short range's longest input (or unknown); not
# on the large range (3,821 to 18,846 frames per sequence), where they run out of memory.
ATTEMPTED_SHORT_AND_LONG = frozenset({"MAPANYTHING", "MAST3R", "DEJAVIEW", "DUST3R", "POW3R", "IGGT4D"})

# Per range: methods run on it although their budget is below its longest input (see the module docstring). Every
# sequence is attempted and failures are recorded, never skipped up front.
BEYOND_BUDGET_ATTEMPTS: dict[str, frozenset[str]] = {
    "short": ATTEMPTED_SHORT_AND_LONG,
    "long": ATTEMPTED_SHORT_AND_LONG,
    "large": frozenset({"LOGER", "R3", "ZIPMAP", "SCAL3R"}),
}

# Primary trajectory alignment per input track. Sim(3) for both: monocular input is only defined up to scale, and
# even with metric depth priors most feed-forward methods return poses whose scale is not consistent with the ground
# truth. The evaluator always also reports the ATE after an SE(3) fit (``ate_rmse_se3``), which on rgb_depth shows
# how well a method adopts the metric scale of its depth priors. The SLAM track has no rgb_cameras variant (poses are
# what is evaluated).
# rgb_masked_depth (half of each depth map dropped, spatialarena/depth_prior_masking.py) is aligned the same way.
TRACK_ALIGNMENT: dict[str, str] = {"rgb": "sim3", "rgb_depth": "sim3", "rgb_masked_depth": "sim3"}

# Native frame rate of each SLAM dataset; the input rate of a case is this divided by the stride. 7-Scenes Kinect
# and TUM RGB-D record at 30 Hz, Oxford Spires cam0 at 20 Hz. VBR (LoGeR's processed release) keeps only the camera
# frames within 10 ms of an Ouster sweep, i.e. one frame per 10 Hz sweep: the handheld sequences keep exactly every
# third 30 Hz camera frame, the car sequences every frame. KITTI Odometry records at 10 Hz. SpatialArena's
# sequences are its rig frames at 5 Hz (scripts/preprocess/materialize_spatialarena.py).
DATASET_FPS: dict[str, float] = {"7scenes": 30.0, "tum_dynamics": 30.0, "oxford_spires": 20.0, "vbr": 10.0,
                                 "kitti_odometry": 10.0, "spatialarena": 5.0}


def range_datasets(slam_range: str) -> list[str]:
    """The SLAM datasets of ``slam_range``."""
    if slam_range not in SLAM_RANGES:
        raise ValueError(f"Unknown SLAM range {slam_range!r}; expected one of {sorted(SLAM_RANGES)}")
    return list(SLAM_RANGES[slam_range]["strides"])


def range_stride(slam_range: str, dataset: str) -> int:
    """The stride ``dataset`` is fed at in ``slam_range``."""
    strides = SLAM_RANGES[slam_range]["strides"] if slam_range in SLAM_RANGES else None
    if strides is None:
        raise ValueError(f"Unknown SLAM range {slam_range!r}; expected one of {sorted(SLAM_RANGES)}")
    if dataset.lower() not in strides:
        raise ValueError(f"SLAM dataset {dataset!r} is not part of the {slam_range} range ({sorted(strides)})")
    return strides[dataset.lower()]


def alignment_for_track(track: str) -> str:
    if track not in TRACK_ALIGNMENT:
        raise ValueError(f"The SLAM track has no {track!r} variant; expected one of {sorted(TRACK_ALIGNMENT)}")
    return TRACK_ALIGNMENT[track]


# Fixed input resolution (width, height) per SLAM dataset, the same for every method so the comparison is fair: one
# width (504 = 36 x 14 px patches) and each dataset's own aspect ratio, so frames are only downscaled, never stretched.
# 504x378 is exactly 4:3 (TUM / 7-Scenes 640x480, Oxford Spires 1440x1080 after rectification); VBR's 1388x700 frames
# are center-cropped to 2:1 (1388x694) and downscaled to 504x252; KITTI Odometry's 1241x376 / 1226x370 frames (about
# 3.3:1) lose 10-15 columns to 504:154 (36 x 11 patches) and are downscaled to 504x154. Methods with 16 px patches
# (CUT3R, Fast3R, VGGT-Omega) center-crop these to 496x368 / 496x240 / 496x144 themselves; nothing is rescaled again.
# scripts/preprocess/materialize_slam_inputs.py writes the frames at this size under input_root(); the full-slam
# engine feeds those to every adapter with FIXED_INPUT_RESOLUTION=1 and RESOLUTION=IMAGE_LOAD_RESOLUTION=504.
SLAM_INPUT_WIDTH = 504
DATASET_INPUT_SIZE: dict[str, tuple[int, int]] = {
    "7scenes": (504, 378),
    "tum_dynamics": (504, 378),
    "oxford_spires": (504, 378),
    "vbr": (504, 252),
    "kitti_odometry": (504, 154),
    "spatialarena": (504, 406),  # 1600x1296 frames (about 1.23:1) resized to 501x406 and padded by 3 columns
}
# The benchmark's SLAM data (the released one) is the pre-sized sequences under SLAM/: images, depths and the
# ground-truth sparse/0 (poses as recorded, intrinsics mapped through the resize), so methods read and the evaluator
# scores from one directory. SLAM_source/ keeps the native-resolution sequences they are made from (and the raw
# downloads), the input of the materialize_* scripts.
SLAM_INPUT_DIRNAME = "SLAM"
SLAM_SOURCE_DIRNAME = "SLAM_source"


def input_size(dataset: str) -> tuple[int, int]:
    """(width, height) every method is fed on ``dataset``."""
    if dataset.lower() not in DATASET_INPUT_SIZE:
        raise ValueError(f"No input size for SLAM dataset {dataset!r}; add it to slam_protocol.DATASET_INPUT_SIZE")
    return DATASET_INPUT_SIZE[dataset.lower()]


def input_root(benchmark_root: Path, dataset: str) -> Path:
    """Directory of ``dataset``'s pre-sized SLAM inputs (one ``<scene>/{images,depths,sparse/0}`` per sequence)."""
    input_size(dataset)
    override = track_root_override(SLAM_INPUT_DIRNAME) if dataset.lower() == SPATIALARENA else None
    return override if override is not None else Path(benchmark_root) / SLAM_INPUT_DIRNAME / dataset.lower()


def source_root(benchmark_root: Path, dataset: str) -> Path:
    """Directory of ``dataset``'s native-resolution sequences, which the pre-sized inputs are made from."""
    return Path(benchmark_root) / SLAM_SOURCE_DIRNAME / dataset.lower()


def dataset_fps(dataset: str) -> float:
    if dataset.lower() not in DATASET_FPS:
        raise ValueError(f"No frame rate recorded for SLAM dataset {dataset!r}; add it to slam_protocol.DATASET_FPS")
    return DATASET_FPS[dataset.lower()]


def _supports_track(method: str, track: str) -> bool:
    from spatialarena.methods import resolve_method

    try:
        return track in resolve_method(method).tracks
    except (KeyError, ValueError):
        return False


def in_range_track(method: str, slam_range: str, track: str = "rgb") -> bool:
    """Whether ``method`` (canonical name) is evaluated on the ``slam_range`` track with ``track`` input."""
    range_datasets(slam_range)
    if method in EXCLUDED_METHODS or track not in TRACK_ALIGNMENT:
        return False
    if method not in H200_FRAME_BUDGET:
        return False
    budget = H200_FRAME_BUDGET[method]
    fits = (method in BEYOND_BUDGET_ATTEMPTS.get(slam_range, ())
            or (budget is not None and budget >= SLAM_RANGES[slam_range]["max_frames"]))
    return fits and _supports_track(method, track)


def method_slam_ranges(method: str, track: str = "rgb") -> list[str]:
    """The range tracks ``method`` takes part in with ``track`` input (empty: not evaluated on SLAM)."""
    return [slam_range for slam_range in SLAM_RANGES if in_range_track(method, slam_range, track)]


def range_track_methods(slam_range: str, track: str = "rgb") -> list[str]:
    return [method for method in H200_FRAME_BUDGET if in_range_track(method, slam_range, track)]


def main() -> None:
    parser = argparse.ArgumentParser(description="SLAM protocol lookups for run_method_case_list.sh.")
    sub = parser.add_subparsers(dest="command", required=True)
    stride = sub.add_parser("stride", help="Print the stride of a dataset in a range track.")
    stride.add_argument("slam_range")
    stride.add_argument("dataset")
    sub.add_parser("datasets", help="List the datasets of a range track.").add_argument("slam_range")
    sub.add_parser("alignment", help="Print the alignment (sim3/se3) for an input track.").add_argument("track")
    check = sub.add_parser("check", help="Print ok / not_in_track / exceeds for one case.")
    check.add_argument("method")
    check.add_argument("slam_range")
    check.add_argument("track")
    check.add_argument("num_frames", type=int)
    size = sub.add_parser("input_size", help="Print the fixed input width and height of a dataset.")
    size.add_argument("dataset")
    root = sub.add_parser("input_root", help="Print the directory of a dataset's pre-sized SLAM inputs.")
    root.add_argument("benchmark_root", type=Path)
    root.add_argument("dataset")
    listing = sub.add_parser("methods", help="List the methods of a range track.")
    listing.add_argument("slam_range")
    listing.add_argument("track", nargs="?", default="rgb")
    args = parser.parse_args()
    if args.command == "stride":
        print(range_stride(args.slam_range, args.dataset))
    elif args.command == "datasets":
        print(" ".join(range_datasets(args.slam_range)))
    elif args.command == "alignment":
        print(alignment_for_track(args.track))
    elif args.command == "input_size":
        print(*input_size(args.dataset))
    elif args.command == "input_root":
        print(input_root(args.benchmark_root, args.dataset))
    elif args.command == "methods":
        print(" ".join(range_track_methods(args.slam_range, args.track)))
    elif not in_range_track(args.method, args.slam_range, args.track):
        print("not_in_track")
    elif args.method in BEYOND_BUDGET_ATTEMPTS.get(args.slam_range, ()):
        # Attempted beyond its budget on purpose: run the sequence and let a failure be recorded.
        print("ok")
    elif args.num_frames > H200_FRAME_BUDGET[args.method]:
        # Cannot happen for a sequence of the track (max_frames bounds every one); a new, longer sequence would.
        print(f"exceeds {args.num_frames}>{H200_FRAME_BUDGET[args.method]:g}")
    else:
        print("ok")


if __name__ == "__main__":
    main()
