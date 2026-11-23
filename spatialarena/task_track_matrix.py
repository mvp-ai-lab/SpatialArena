# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Task/track compatibility matrix for the BenchmarkEvaluator API.

A track conditions a method on extra ground-truth input (camera poses for
`rgb_cameras`, depth priors for `rgb_depth`, depth priors with half of every
map dropped for `rgb_masked_depth`, and those plus the camera poses for
`rgb_masked_depth_cameras`; the masking is
`spatialarena/depth_prior_masking.py`). A (task, track) pair is only worth
planning if the task still has something nontrivial left to predict once that
input is given -- see "Task/Track Compatibility Matrix" in the API reference
(docs/benchmark/evaluator_quickstart.md) for the full rationale per task.

Note on `MonoDepth`/`rgb_depth`: the existing shell pipeline
(`run_method_case_list.sh`) does not block this combination today, and it
has already produced real, committed results (e.g. PI3_X on nyuv2). This
matrix intentionally excludes it anyway for the *new* API's planning
decisions -- predicting depth while a depth prior is given is not a
meaningful benchmark axis, matching the same reasoning already applied to
excluding SfM/SLAM from `rgb_cameras`. This is a deliberate divergence:
the old pipeline's existing behavior/results are left untouched, but
`BenchmarkEvaluator.plan()` will never generate a MonoDepth/rgb_depth work
item.

The masked-depth tracks are the depth-completion axis: the method sees only the
kept cells of each GT depth map and is scored on every GT-valid pixel, so they
run on every task, MonoDepth and VideoDepth included, and on every dataset that
has depth priors (dense GT depth is no longer complete once masked, so the
`rgb_depth` dataset restriction of MVRecon does not apply).
`rgb_masked_depth_cameras` (RGB + masked depth + camera conditioning) is on
MVRecon and SfM only; on SfM it measures how faithfully the
given poses come back out, not pose estimation.

The sparse-depth tracks (`rgb_sparse_depth`, `rgb_sparse_depth_cameras`)
give the depth a real system has: KITTI's single-scan LiDAR on VideoDepth, and
each image's COLMAP keypoints on the MVRecon/SfM datasets whose model has 3D
points (`TRACK_DATASETS`; spatialarena/sparse_depth.py). The SfM points come
from the reconstruction the SfM poses are scored against, so SfM on these
tracks is SfM-assisted, not pose estimation.
"""

from __future__ import annotations

from spatialarena import methods

TASK_TRACK_MATRIX: dict[str, tuple[str, ...]] = {
    "MonoDepth": ("rgb", "rgb_masked_depth"),
    "VideoDepth":
        ("rgb", "rgb_cameras", "rgb_masked_depth", "rgb_sparse_depth"),
    "MVRecon": (
        "rgb",
        "rgb_cameras",
        "rgb_depth",
        "rgb_masked_depth",
        "rgb_masked_depth_cameras",
        "rgb_sparse_depth",
        "rgb_sparse_depth_cameras",
    ),
    "SfM": (
        "rgb",
        "rgb_depth",
        "rgb_masked_depth",
        "rgb_masked_depth_cameras",
        "rgb_sparse_depth",
        "rgb_sparse_depth_cameras",
    ),
    "SLAM": ("rgb", "rgb_depth", "rgb_masked_depth"),
}

# (task, track) -> the only datasets that pair runs on, for a track worth
# evaluating on some datasets only. `rgb_depth` hands the method the dataset's
# GT depth maps. MVRecon only has something left to reconstruct when that depth
# is sparse or incomplete: DTU (~1.2k SfM points per image), ETH3D (laser scan,
# ~14% of pixels) and 7-Scenes (Kinect, ~70% of pixels). HiRoom (synthetic,
# ~98%) and ScanNet++ (complete 256x192 maps) are dense, which leaves only the
# poses to predict; NRGBD and Tanks and Temples have no depth prior at all. SfM
# `rgb_depth` runs only where the released SfM ground truth has depth maps:
# not DTU (images and the sparse model only), Tanks and Temples or Weak-Texture
# SfM (texturepoorsfm). `rgb_cameras` hands the method the GT camera poses.
# VideoDepth runs it only on SpatialArena, whose released GT has them
# (`sparse/0`): KITTI has only the OXTS vehicle trajectory in the IMU frame
# (no IMU-to-camera calibration), and the released GT has no Sintel or DyCheck
# cameras.
TRACK_DATASETS: dict[tuple[str, str], frozenset[str]] = {
    ("MVRecon", "rgb_depth"):
        frozenset({"7scenes", "dtu", "eth3d"}),
    ("SfM", "rgb_depth"):
        frozenset({"eth3d", "hiroom", "imc", "scannetpp", "spatialarena"}),
    # SLAM `rgb_depth` runs only where the SLAM inputs have depth maps
    # (`depths/`): 7-Scenes and TUM Dynamics (Kinect) and the SpatialArena
    # sequences with LiDAR depth (TRACK_EXCLUDED_SCENES). Oxford Spires, KITTI
    # Odometry and VBR have images and poses only.
    ("SLAM", "rgb_depth"):
        frozenset({"7scenes", "spatialarena", "tum_dynamics"}),
    ("VideoDepth", "rgb_cameras"):
        frozenset({"spatialarena"}),
    # The sparse-depth tracks run where a real sparse depth source exists
    # (spatialarena/sparse_depth.py): KITTI's LiDAR, and the COLMAP keypoints of
    # the scenes whose model records which points each image observes
    # (ScanNet++'s models mostly do not: 19 of its 20 scenes have 3D points
    # without tracks).
    ("VideoDepth", "rgb_sparse_depth"):
        frozenset({"kitti"}),
    ("MVRecon", "rgb_sparse_depth"):
        frozenset({"dtu", "eth3d", "tanks_and_temples"}),
    ("MVRecon", "rgb_sparse_depth_cameras"):
        frozenset({"dtu", "eth3d", "tanks_and_temples"}),
    ("SfM", "rgb_sparse_depth"):
        frozenset(
            {"dtu", "eth3d", "imc", "tanks_and_temples", "texturepoorsfm"}),
    ("SfM", "rgb_sparse_depth_cameras"):
        frozenset(
            {"dtu", "eth3d", "imc", "tanks_and_temples", "texturepoorsfm"}),
}

# (task, track) -> dataset -> the scenes of a dataset in TRACK_DATASETS the pair
# still does not run on. SpatialArena's SLAM sequences without LiDAR depth (no
# `depths/`) have no depth prior for `rgb_depth`; the engine skips them too
# (`SLAM_RGB_DEPTH_EXCLUDED_SCENES` in
# templates/_engine/run_method_case_list.sh).
TRACK_EXCLUDED_SCENES: dict[tuple[str, str], dict[str, frozenset[str]]] = {
    ("SLAM", "rgb_depth"): {
        "spatialarena":
            frozenset({
                "CambridgeWesthub3",
                "CoalDropsYard2",
                "EdinburghCastle1",
                "NationalHistoryMuseumIndoor1",
                "StPancrasOutside",
            }),
    },
}


def track_runs_on(task: str, track: str, dataset: str) -> bool:
    """Returns whether a (task, track) pair is evaluated on a dataset.

    Args:
        task: Benchmark task.
        track: Input-modality track.
        dataset: Dataset name, in any case.

    Returns:
        False only if `TRACK_DATASETS` restricts the pair to other datasets.
    """
    allowed = TRACK_DATASETS.get((task, track))
    return allowed is None or dataset.lower() in allowed


def track_runs_on_case(task: str, track: str, dataset: str, scene: str) -> bool:
    """Returns whether a (task, track) pair is evaluated on a scene.

    Args:
        task: Benchmark task.
        track: Input-modality track.
        dataset: Dataset name, in any case.
        scene: Scene name.

    Returns:
        Whether `TRACK_DATASETS` allows the dataset and `TRACK_EXCLUDED_SCENES`
        does not exclude the scene.
    """
    excluded_by_dataset = TRACK_EXCLUDED_SCENES.get((task, track), {})
    excluded = excluded_by_dataset.get(dataset.lower(), frozenset())
    return track_runs_on(task, track, dataset) and scene not in excluded


def on_matrix(task: str, track: str, dataset: str) -> bool:
    """Returns whether a result for (task, track, dataset) is in the benchmark.

    Off-matrix results (e.g. MonoDepth given GT depth, from the old shell
    pipeline) are left out of the dashboard. A task the matrix does not list is
    not filtered.

    Args:
        task: Benchmark task.
        track: Input-modality track.
        dataset: Dataset name, in any case.

    Returns:
        Whether the result belongs in the benchmark.
    """
    if task not in TASK_TRACK_MATRIX:
        return True
    return track in TASK_TRACK_MATRIX[task] and track_runs_on(
        task, track, dataset)


def matrix_tracks(task: str) -> tuple[str, ...]:
    """Returns the tracks the matrix allows for a task; `()` if unlisted."""
    return TASK_TRACK_MATRIX.get(task, ())


def runnable_tracks(task: str, method: methods.MethodSpec) -> tuple[str, ...]:
    """Returns the tracks matrix-allowed for a task and supported by a method.

    A (task, track) pair is only ever planned/run if both conditions hold.

    Args:
        task: Benchmark task.
        method: The method.

    Returns:
        The tracks, in the method's order.
    """
    if method.tasks and task not in method.tasks:
        return ()
    allowed = set(matrix_tracks(task))
    tracks = tuple(track for track in method.tracks if track in allowed)
    if task == "SLAM":
        # Only methods of a SLAM range track (spatialarena/slam_protocol.py) are
        # evaluated on SLAM.
        # pylint: disable-next=import-outside-toplevel
        from spatialarena import slam_protocol

        tracks = tuple(track for track in tracks
                       if slam_protocol.method_slam_ranges(method.name, track))
    return tracks


def evaluated_methods(task: str,
                      track: str,
                      slam_range: str | None = None) -> list[str]:
    """Returns the registered methods planned for a (task, track) pair.

    This is what docs/evaluation.md's protocol matrix lists per track.

    Args:
        task: Benchmark task.
        track: Input-modality track.
        slam_range: For SLAM, only the methods of this range track.

    Returns:
        The method names, in registry order.
    """
    names = [
        spec.name
        for spec in methods.METHOD_SPECS
        if track in runnable_tracks(task, spec)
    ]
    if task == "SLAM" and slam_range is not None:
        # pylint: disable-next=import-outside-toplevel
        from spatialarena import slam_protocol

        names = [
            name for name in names
            if slam_range in slam_protocol.method_slam_ranges(name, track)
        ]
    return names
