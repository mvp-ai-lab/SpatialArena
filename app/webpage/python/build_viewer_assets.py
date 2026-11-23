#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Backfills the evaluators' viewer assets for older evaluations.

New evaluations write their viewer assets themselves
(spatialarena/eval/viewer_assets.py, VIEWER_ASSETS=1 in the engine). This
tool gives older results the same files, with the same writers, for every
canonical case that has none yet (or all of them with --force).

Asset kinds:

    trajectories    SfM / Weak-Texture SfM / SLAM: re-derives the aligned
                    trajectory (cheap).
    thumbnails      The input-image thumbnails trajectory mode shows, once
                    per pose scene (all its images).
    depth           MonoDepth / VideoDepth: re-runs the depth evaluation's
                    alignment to render the frames.
    recon           MVRecon: re-runs the fusion to get the cloud in the GT
                    frame (memory-heavy: few workers).
    gt_colors       MVRecon: the GT points' colors from the GT scan, for
                    scenes written before they were saved.
    cameras         MVRecon: like recon, for evaluated cases whose viewer
                    cloud predates cameras.json.
    confidence      MVRecon: each viewer point's predicted confidence
                    (confidence.f32) from the saved cloud and cameras.json,
                    for predictions with confidence maps (reads the views, no
                    fusion).
    imported_depth  Like depth, for the depth cases of imported batches
                    (--import_root, e.g. the HF release batches under
                    <benchmark root>/_ffb_eval/<batch>/). Those folders stay
                    as downloaded: the images go to
                    <eval root>/viewer_assets/imported/<method>/<task>/
                    <dataset>/<scene>/viewer/, which the viewer index reads
                    after the canonical report folders. Per (method, task,
                    dataset, scene) only the newest evaluation is rendered,
                    and only for full-standard cases on the task/track
                    matrix that have no canonical case.
    imported_trajectories, imported_slam, imported_recon
                    The same for imported SfM / Weak-Texture SfM
                    trajectories, SLAM trajectories and MVRecon clouds.
                    Trajectories copy their eval_sparse.json / eval_slam.json
                    next to the viewer folder (the index reads its metrics
                    there). A SLAM case keeps its range:
                    viewer_assets/imported/slam/full-slam-<range>/<method>/
                    SLAM/<dataset>/<scene>/viewer/.
    imported_thumbnails
                    The input-image thumbnails of the imported pose (SfM and
                    SLAM) scenes.

Run it on a compute node; it refreshes the viewer index at the end:

    sbatch --partition=cpu ... \
        python app/webpage/python/build_viewer_assets.py depth --workers 16
"""

from __future__ import annotations

import argparse
from concurrent import futures
import json
import os
import pathlib
import shutil
import sys
import time
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.webpage.python import benchmark_analyzer  # noqa: E402
from app.webpage.python import benchmark_artifact_analyzer  # noqa: E402
from app.webpage.python import report_discovery  # noqa: E402
# Tests patch build_viewer_assets.read_case_lists, so it is called through
# this module's own binding.
from app.webpage.python.report_discovery import read_case_lists  # noqa: E402
from app.webpage.python import viewer_index  # noqa: E402
from spatialarena import report_layout  # noqa: E402
from spatialarena import task_track_matrix  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402
from spatialarena.eval import viewer_assets  # noqa: E402

POSE_TASKS = ("SfM", "WeakTextureSfM")
DEPTH_TASKS = ("MonoDepth", "VideoDepth")
VIEWER_FILES = {
    "eval_sparse.json": viewer_assets.TRAJECTORY_FILE,
    "eval_slam.json": viewer_assets.TRAJECTORY_FILE,
    "eval_depth.json": viewer_assets.DEPTH_FRAMES_FILE,
    "eval_mvrecon.json": viewer_assets.CLOUD_FILE,
}
DEPTH_CASE_LIST = (REPO_ROOT / "scripts" / "test" /
                   "full_standard_depth_cases.txt")
SFM_CASE_LIST = REPO_ROOT / "scripts" / "test" / "full_standard_sfm_cases.txt"
# Release batches name their outputs
# <storage>/evaluation/feedforward/outputs/<batch>/...; locally that is
# <import root>/<batch>/...
RELEASE_OUTPUTS_ROOT = "<storage>/evaluation/feedforward/outputs"

_FULL_REPORT_ROOTS = report_layout.FULL_REPORT_ROOTS


def discover_cases(eval_root: pathlib.Path, kind: str) -> list[pathlib.Path]:
    """Returns the evaluation JSONs of the canonical report folders.

    Args:
        eval_root: The evaluation root holding `reports/`.
        kind: The asset kind: "trajectories", "depth" or anything else for
            the MVRecon cases.
    """
    reports = eval_root / "reports"
    if kind == "trajectories":
        pose = reports / _FULL_REPORT_ROOTS["full-pose"] / "full-pose"
        slam = reports / _FULL_REPORT_ROOTS["full-slam"]
        pose_cases = sorted(
            path for path in pose.glob("*/*/*/*/eval_sparse.json")
            if path.parts[-4] in POSE_TASKS)
        slam_cases = sorted(slam.glob("full-slam-*/*/SLAM/*/*/eval_slam.json"))
        return pose_cases + slam_cases
    depth = reports / _FULL_REPORT_ROOTS["full-depth"] / "full-depth"
    if kind == "depth":
        return sorted(path for task in DEPTH_TASKS
                      for path in depth.glob(f"*/{task}/*/*/eval_depth.json"))
    return sorted(depth.glob("*/MVRecon/*/*/eval_mvrecon.json"))


def needs_assets(eval_path: pathlib.Path, force: bool) -> bool:
    """Whether a case lacks viewer assets (or they predate its evaluation)."""
    target = (eval_path.parent / viewer_assets.VIEWER_DIR_NAME /
              VIEWER_FILES[eval_path.name])
    return (force or not target.is_file() or
            target.stat().st_mtime < eval_path.stat().st_mtime)


def _stored_result(eval_path: pathlib.Path) -> dict:
    """Returns the first scene result of an evaluation JSON.

    Raises:
        ValueError: The evaluation has no scene result (a failed case).
    """
    text = eval_path.read_text(encoding="utf-8")
    results = json.loads(text).get("scene_results") or []
    if not results:
        raise ValueError("no scene result")
    return results[0]


def _scene_info(task: str, dataset: str, scene: str,
                benchmark_root: pathlib.Path):
    """Returns the dataset registry's scene info of one benchmark scene.

    Raises:
        FileNotFoundError: The registry does not list the scene.
    """
    for spec in dataset_registry.select_dataset_specs(tasks=[task],
                                                      datasets=[dataset],
                                                      include_scannet=True):
        dataset_scenes = spec.create_dataset(benchmark_root, scenes=[scene])
        for info in dataset_scenes.list_scenes():
            if info.scene == scene:
                return info
    raise FileNotFoundError(
        f"{task}/{dataset}/{scene} is not in the dataset registry")


def backfill_trajectory(
        eval_path: pathlib.Path,
        benchmark_root: pathlib.Path,
        _scene_root: pathlib.Path,
        *,
        case_dir: pathlib.Path | None = None,
        prediction_sparse_path: pathlib.Path | None = None) -> str:
    """Writes one pose case's trajectory.json.

    By default it goes to the case's report folder, with the paths its
    evaluation names. An imported case (`case_dir` and
    `prediction_sparse_path` given) reads its GT model from the local dataset
    registry instead of the release machine's path, and gets a copy of its
    evaluation next to the viewer folder for the index's metrics.

    Returns:
        The written trajectory file.

    Raises:
        FileNotFoundError: An imported case's scene has no local GT model.
        RuntimeError: The viewer writer reported an error.
    """
    result = dict(_stored_result(eval_path))
    if case_dir is None:
        case_dir = eval_path.parent / viewer_assets.VIEWER_DIR_NAME
        result["prediction_sparse_path"] = str(
            benchmark_artifact_analyzer.relocated_path(
                result["prediction_sparse_path"]))
    else:
        scene = _scene_info(result["task"], result["dataset"], result["scene"],
                            benchmark_root)
        if scene.sparse_gt_path is None:
            raise FileNotFoundError(
                f"{result['task']}/{result['dataset']}/{result['scene']} "
                "has no local GT model")
        result["prediction_sparse_path"] = str(prediction_sparse_path)
        result["sparse_gt_path"] = str(scene.sparse_gt_path)
    scene_info = types.SimpleNamespace(task=result["task"],
                                       dataset=result["dataset"],
                                       scene=result["scene"])
    written = viewer_assets.add_case_trajectory(
        case_dir,
        True,
        result.get("method", eval_path.parts[-5]),
        scene_info,
        result,
    )
    if "viewer_error" in written:
        raise RuntimeError(written["viewer_error"])
    if case_dir != eval_path.parent / viewer_assets.VIEWER_DIR_NAME:
        shutil.copy2(eval_path, case_dir.parent / eval_path.name)
    return written["viewer_trajectory"]


def backfill_depth(eval_path: pathlib.Path,
                   benchmark_root: pathlib.Path,
                   scene_root: pathlib.Path,
                   *,
                   case_dir: pathlib.Path | None = None,
                   prediction_scene_path: pathlib.Path | None = None) -> str:
    """Re-runs one depth case's alignment with the viewer writer.

    The images go to `case_dir` (default: the case's report folder) and the
    prediction is read from `prediction_scene_path` (default: the one its
    evaluation names). A VideoDepth case scored with a GT-only Huber-IRLS
    protocol whose evaluation saved its sequence fit is drawn from that fit one
    frame at a time instead (`write_sequence_viewer_from_fit`: the same
    aligned maps without holding the whole sequence in memory).

    Returns:
        The folder the images were written to.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import depth_prediction_evaluator

    result = _stored_result(eval_path)
    scene_info = _scene_info(result["task"], result["dataset"], result["scene"],
                             benchmark_root)
    # The engine's configuration: the evaluator CLI defaults plus the protocol
    # (and its inverse floor) the case was scored with, so the images show the
    # alignment behind its scores. Geometry metrics do not affect the
    # alignment.
    config = depth_prediction_evaluator.DepthEvaluationConfig(
        protocol=result.get("protocol", "standard"),
        inverse_floor_policy=result.get("inverse_floor_policy", "fixed"),
        compute_geometry_metrics=False,
    )
    case_dir = case_dir or eval_path.parent / viewer_assets.VIEWER_DIR_NAME
    writer = viewer_assets.make_depth_viewer_writer(scene_info, config,
                                                    case_dir, scene_root)
    prediction = (prediction_scene_path or
                  benchmark_artifact_analyzer.relocated_path(
                      result["prediction_scene_path"]))
    stored_fit_protocols = depth_prediction_evaluator.STORED_FIT_PROTOCOLS
    if (result["task"] == "VideoDepth" and
            config.protocol in stored_fit_protocols and
        (pathlib.Path(prediction) / f"{config.protocol}_fit.json").is_file()):
        depth_prediction_evaluator.write_sequence_viewer_from_fit(
            scene_info,
            prediction,
            writer,
            result.get("metrics") or {},
            config=config,
        )
    else:
        depth_prediction_evaluator.evaluate_depth_scene(scene_info,
                                                        prediction,
                                                        config=config,
                                                        viewer=writer)
    return str(case_dir)


def saved_fused_point_path(eval_path: pathlib.Path) -> pathlib.Path | None:
    """Returns the `fused_points.ply` an MVRecon evaluation saved, or None.

    The evaluation saves it next to its `eval_mvrecon.json`, under
    `<case>/artifacts/MVRecon/<dataset>/<scene>/`.
    """
    found = sorted(
        (eval_path.parent / "artifacts").glob("*/*/*/fused_points.ply"))
    return found[0] if len(found) == 1 else None


def _saved_fused_cloud(path: pathlib.Path | None):
    """Returns a saved `fused_points.ply` as (points, colors or None).

    Returns None when the file is not there or holds no points.
    """
    if path is None or not path.is_file():
        return None
    import open3d as o3d  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    if not len(points):
        return None
    return points, (np.asarray(cloud.colors) if cloud.has_colors() else None)


def _dtu_cloud_in_scene_frame(scene_info, saved):
    """Moves a DTU evaluation's saved cloud into the normalized GT frame.

    The evaluator moves the fused cloud into DTU's calibration frame with
    `_fit_dtu_raw_calibration_sim3` before saving it; this applies the
    inverse, so the viewer shows the cloud that was scored without re-fusing
    (Open3D 0.20's ScalableTSDFVolume returns an empty volume on this host).

    Returns:
        (points, colors) in the scene frame, or None when the calibration
        cannot be fitted.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import mvrecon_evaluator

    # pylint: disable-next=protected-access
    calibration = mvrecon_evaluator._fit_dtu_raw_calibration_sim3(
        mvrecon_evaluator.load_colmap_views(scene_info.sparse_gt_path),
        scene_info.root_path)
    if calibration is None:
        return None
    points, colors = saved
    return ((points - calibration.translation)
            @ calibration.rotation) / calibration.scale, colors


def saved_cloud_only(scene_info, saved, protocol):
    """Builds a FusedPrediction of a saved cloud and the GT points, no cameras.

    Args:
        scene_info: The MVRecon scene.
        saved: The evaluation's saved (points, colors or None), already in the
            GT frame.
        protocol: The dataset's MVRecon protocol (for the GT sampling number).

    Raises:
        ValueError: The scene has no registered GT point cloud or mesh.
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np
    from spatialarena.eval import mvrecon_evaluator

    gt_geometry = mvrecon_evaluator.resolve_gt_geometry_path(scene_info)
    if gt_geometry is None:
        raise ValueError("No GT point cloud or mesh registered for MVRecon "
                         f"scene {scene_info.scene}")
    points, colors = saved
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors if colors is not None else [],
                        dtype=np.float64).reshape(-1, 3)
    if len(colors) != len(points):
        colors = np.full((len(points), 3), 0.7)
    return mvrecon_evaluator.FusedPrediction(
        points=points,
        colors=colors,
        gt_points=mvrecon_evaluator.load_point_cloud_array(
            gt_geometry, protocol.sampling_number),
        fusion_metadata={
            "viewer_cameras": "none: the release has no predictions"
        },
        cameras=None)


def backfill_recon(eval_path: pathlib.Path,
                   benchmark_root: pathlib.Path,
                   scene_root: pathlib.Path,
                   *,
                   case_dir: pathlib.Path | None = None,
                   prediction_scene_path: pathlib.Path | None = None,
                   fused_point_path: pathlib.Path | None = None) -> str:
    """Writes one MVRecon case's viewer cloud.

    The cloud goes to `case_dir` (default: the case's report folder) and the
    prediction is read from `prediction_scene_path` (default: the one its
    evaluation names). With `fused_point_path` (an imported case's saved
    `fused_points.ply`, the cloud it was scored on) the points come from
    there instead of re-running the fusion; DTU's saved cloud is in DTU's
    calibration frame and is moved back into the scene frame first.

    Returns:
        The folder the cloud was written to.

    Raises:
        ValueError: The fusion failed without a saved cloud to fall back on,
            or the cloud is empty.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import mvrecon_evaluator

    result = _stored_result(eval_path)
    scene_info = _scene_info("MVRecon", result["dataset"], result["scene"],
                             benchmark_root)
    protocol = mvrecon_evaluator.protocol_for_dataset(result["dataset"],
                                                      scene_info.metadata)
    prediction = (prediction_scene_path or
                  benchmark_artifact_analyzer.relocated_path(
                      result["prediction_scene_path"]))
    saved = _saved_fused_cloud(fused_point_path)
    if saved is not None and scene_info.dataset.lower() == "dtu":
        saved = _dtu_cloud_in_scene_frame(scene_info, saved)
    try:
        fused = mvrecon_evaluator.fuse_prediction_in_gt_frame(
            scene_info,
            prediction,
            result.get("reconstruction_protocol", "unposed"),
            protocol,
            fused_cloud=saved,
        )
    except ValueError:
        # A release that shipped the scored cloud but not the predictions
        # (most SpatialArena direct-backprojection cases): the cloud alone,
        # without the cameras, which are read from the predictions. T&T's GT
        # crop needs the registration of the predicted cameras' frame, so it
        # is not done this way.
        if saved is None or scene_info.dataset.lower() == "tanks_and_temples":
            raise
        fused = saved_cloud_only(scene_info, saved, protocol)
    if not len(fused.points):
        raise ValueError("the fused cloud is empty")
    case_dir = case_dir or eval_path.parent / viewer_assets.VIEWER_DIR_NAME
    viewer_assets.write_recon_viewer(
        case_dir,
        viewer_assets.scene_asset_dir(scene_root, "MVRecon", scene_info.dataset,
                                      scene_info.scene),
        scene_info,
        fused,
        metrics=result.get("metrics") or {},
        eval_threshold=protocol.eval_threshold,
    )
    return str(case_dir)


def backfill_confidence(eval_path: pathlib.Path, benchmark_root: pathlib.Path,
                        _scene_root: pathlib.Path) -> str:
    """Writes one MVRecon case's `confidence.f32`.

    It is computed from the case's saved viewer cloud and cameras and the fused
    views' confidence maps (the views' order comes from the fusion inputs;
    their depths are read but nothing is fused).

    Returns:
        The case's viewer folder, or "no confidence maps".

    Raises:
        ValueError: The number of fused views and saved cameras differ.
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np

    from spatialarena.eval import mvrecon_evaluator

    result = _stored_result(eval_path)
    case_dir = eval_path.parent / viewer_assets.VIEWER_DIR_NAME
    payload = json.loads(
        (case_dir / viewer_assets.CAMERAS_FILE).read_text(encoding="utf-8"))
    scene_info = _scene_info("MVRecon", result["dataset"], result["scene"],
                             benchmark_root)
    *_, metadata = mvrecon_evaluator.prepare_fusion_inputs(
        scene_info=scene_info,
        prediction_scene_path=benchmark_artifact_analyzer.relocated_path(
            result["prediction_scene_path"]),
        reconstruction_protocol=result.get("reconstruction_protocol",
                                           "unposed"))
    depth_paths = metadata[mvrecon_evaluator.VIEWER_DEPTH_PATHS_KEY]
    if len(depth_paths) != len(payload["pred"]):
        raise ValueError(f"{len(depth_paths)} fused views but "
                         f"{len(payload['pred'])} saved cameras")
    points = np.fromfile(case_dir / "points.f32", dtype="<f4").reshape(-1, 3)
    cameras = viewer_assets.cameras_from_payload(payload, depth_paths)
    if not viewer_assets.write_point_confidence(case_dir, points, cameras):
        return "no confidence maps"
    cloud = json.loads(
        (case_dir / viewer_assets.CLOUD_FILE).read_text(encoding="utf-8"))
    cloud["confidence"] = True
    viewer_assets.write_json(case_dir / viewer_assets.CLOUD_FILE,
                             cloud,
                             indent=1)
    return str(case_dir)


def confidence_cases(eval_root: pathlib.Path,
                     force: bool) -> list[pathlib.Path]:
    """Returns the MVRecon cases to give a `confidence.f32`.

    These are the cases with a viewer cloud and cameras whose prediction saved
    confidence maps (and, unless `force`, no confidence file yet).
    """
    cases = []
    for path in discover_cases(eval_root, "recon"):
        viewer = path.parent / viewer_assets.VIEWER_DIR_NAME
        if (not (viewer / viewer_assets.CAMERAS_FILE).is_file() or
                not (viewer / "points.f32").is_file()):
            continue
        if not force and (viewer / viewer_assets.CONFIDENCE_FILE).is_file():
            continue
        try:
            prediction = benchmark_artifact_analyzer.relocated_path(
                _stored_result(path).get("prediction_scene_path"))
        except (ValueError, OSError):
            continue
        if (prediction / "confidences").is_dir():
            cases.append(path)
    return cases


def backfill_thumbnails(eval_path: pathlib.Path, benchmark_root: pathlib.Path,
                        scene_root: pathlib.Path) -> str:
    """Writes thumbnails of every input image of one pose scene.

    Args:
        eval_path: An evaluation of the scene; it stands for the scene.
        benchmark_root: The benchmark data root.
        scene_root: The viewer assets' per-scene folder root.

    Returns:
        The scene's image folder.
    """
    result = _stored_result(eval_path)
    scene_info = _scene_info(result["task"], result["dataset"], result["scene"],
                             benchmark_root)
    image_root = pathlib.Path(scene_info.image_path)
    names = sorted(path.name
                   for path in image_root.iterdir()
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    viewer_assets.write_frame_thumbnails(
        viewer_assets.scene_asset_dir(scene_root, scene_info.task,
                                      scene_info.dataset, scene_info.scene),
        image_root, names)
    return str(image_root)


def backfill_gt_colors(scene_dir: pathlib.Path, benchmark_root: pathlib.Path,
                       scene_root: pathlib.Path) -> str:
    """Writes `gt_colors.u8` of one MVRecon scene's shared GT points.

    Args:
        scene_dir: The scene folder holding its scene.json.
        benchmark_root: The benchmark data root.
        scene_root: Unused: the scene folder is the case here.

    Returns:
        The scene folder.
    """
    del scene_root  # The scene folder is the case here.
    scene = json.loads(
        (scene_dir / viewer_assets.SCENE_FILE).read_text(encoding="utf-8"))
    scene_info = _scene_info("MVRecon", scene["dataset"], scene["scene"],
                             benchmark_root)
    viewer_assets.write_gt_colors(scene_dir, scene_info)
    return str(scene_dir)


def recon_scenes_without_gt_colors(scene_root: pathlib.Path,
                                   force: bool) -> list[pathlib.Path]:
    """Returns the MVRecon scene folders whose GT colors are not recorded.

    Their scene.json does not yet say whether the GT is colored (with
    `force`: every MVRecon scene folder).
    """
    folders = []
    for path in sorted(
        (scene_root / "MVRecon").glob(f"*/*/{viewer_assets.SCENE_FILE}")):
        if force or "gt_colors" not in json.loads(
                path.read_text(encoding="utf-8")):
            folders.append(path.parent)
    return folders


def release_path(stored: str, import_root: pathlib.Path) -> pathlib.Path:
    """Moves a release batch's stored path under `import_root`.

    Args:
        stored: A `<storage>/.../outputs/<batch>/...` path the release
            batch's evaluation names.
        import_root: The local folder of imported release batches.
    """
    local = stored.replace(RELEASE_OUTPUTS_ROOT, str(import_root), 1)
    return pathlib.Path(local)


# Per imported kind: the canonical cases it stands in for, the imported
# reports, the tasks, the case lists that make a case full-standard, and the
# viewer file that marks it rendered.
IMPORTED_KINDS = {
    "imported_depth": (
        "depth",
        f"{_FULL_REPORT_ROOTS['full-depth']}/full-depth/*/*/*/*/"
        "eval_depth.json",
        DEPTH_TASKS,
        (DEPTH_CASE_LIST,),
        viewer_assets.DEPTH_FRAMES_FILE,
    ),
    "imported_trajectories": (
        "trajectories",
        f"{_FULL_REPORT_ROOTS['full-pose']}/full-pose/*/*/*/*/"
        "eval_sparse.json",
        POSE_TASKS,
        (SFM_CASE_LIST,),
        viewer_assets.TRAJECTORY_FILE,
    ),
    "imported_recon": (
        "recon",
        f"{_FULL_REPORT_ROOTS['full-depth']}/full-depth/*/MVRecon/*/*/"
        "eval_mvrecon.json",
        ("MVRecon",),
        (DEPTH_CASE_LIST,),
        viewer_assets.CLOUD_FILE,
    ),
    # SLAM has no full-standard case list: every released sequence of a range
    # is a benchmark case.
    "imported_slam": (
        "trajectories",
        f"{_FULL_REPORT_ROOTS['full-slam']}/full-slam-*/*/SLAM/*/*/"
        "eval_slam.json",
        ("SLAM",),
        (),
        viewer_assets.TRAJECTORY_FILE,
    ),
}


def _wanted_imported_case(key: tuple[str, ...], *, tasks, canonical, cases,
                          slam: bool) -> bool:
    """Whether an imported case is one to render.

    Args:
        key: The case's ([suite,] method, task, dataset, scene) path parts.
        tasks: The imported kind's tasks.
        canonical: The keys of the canonical cases, which take precedence.
        cases: The full-standard (task, dataset, scene) cases, or None for
            all.
        slam: Whether the kind is imported_slam.
    """
    method, task, dataset, scene = key[-4:]
    if task not in tasks or key in canonical:
        return False
    if cases is not None and (task, dataset, scene) not in cases:
        return False
    track = report_discovery.split_method_track(method)[1]
    if not task_track_matrix.on_matrix(task, track, dataset):
        return False
    # A SLAM range the protocol does not evaluate the method on (e.g. the
    # large range since 2026-11-13).
    if slam and not benchmark_analyzer.on_slam_protocol(
            method, task, key[0].removeprefix("full-slam-")):
        return False
    return True


def _imported_options(kind: str, stored: dict, case_dir: pathlib.Path,
                      import_root: pathlib.Path) -> dict:
    """Returns the backfill keyword arguments of one imported case."""
    if kind in ("imported_trajectories", "imported_slam"):
        return {
            "case_dir":
                case_dir,
            "prediction_sparse_path":
                release_path(stored["prediction_sparse_path"], import_root),
        }
    options = {
        "case_dir":
            case_dir,
        "prediction_scene_path":
            release_path(stored["prediction_scene_path"], import_root),
    }
    if kind == "imported_recon" and stored.get("fused_point_path"):
        options["fused_point_path"] = release_path(stored["fused_point_path"],
                                                   import_root)
    return options


def imported_jobs(kind: str, eval_root: pathlib.Path, import_root: pathlib.Path,
                  force: bool) -> list[tuple[pathlib.Path, dict]]:
    """Returns one imported kind's cases to render.

    Per (method, task, dataset, scene) the newest imported evaluation, if the
    case is full-standard, on the task/track matrix and has no canonical case;
    its viewer files go to `<eval root>/viewer_assets/imported/<method>/
    <task>/<dataset>/<scene>/viewer/` and the paths it names are read from
    `import_root`.

    Returns:
        `(eval json, backfill keyword arguments)` per case.
    """
    canonical_kind, pattern, tasks, case_lists, viewer_file = IMPORTED_KINDS[
        kind]
    slam = kind == "imported_slam"
    width = 6 if slam else 5  # [suite,] method, task, dataset, scene
    canonical = {
        path.parts[-width:-1]
        for path in discover_cases(eval_root, canonical_kind)
    }
    cases = read_case_lists(case_lists) if case_lists else None
    newest: dict[tuple[str, ...], pathlib.Path] = {}
    for path in import_root.glob(f"*/reports/{pattern}"):
        key = path.parts[-width:-1]
        if not _wanted_imported_case(
                key, tasks=tasks, canonical=canonical, cases=cases, slam=slam):
            continue
        if (key not in newest or
                path.stat().st_mtime > newest[key].stat().st_mtime):
            newest[key] = path
    if slam:
        imported_root = viewer_index.IMPORTED_SLAM_ROOT
    else:
        imported_root = viewer_index.IMPORTED_ROOT
    jobs = []
    for key, path in sorted(newest.items()):
        case_dir = (eval_root / imported_root / pathlib.Path(*key) /
                    viewer_assets.VIEWER_DIR_NAME)
        target = case_dir / viewer_file
        if (not force and target.is_file() and
                target.stat().st_mtime >= path.stat().st_mtime):
            continue
        try:
            stored = _stored_result(path)
        except ValueError:  # A failed case (no scene result): nothing to draw.
            continue
        jobs.append((path, _imported_options(kind, stored, case_dir,
                                             import_root)))
    return jobs


def imported_depth_jobs(eval_root: pathlib.Path, import_root: pathlib.Path,
                        force: bool) -> list[tuple[pathlib.Path, dict]]:
    """Returns the imported batches' depth cases to render.

    Returns:
        `(eval_depth.json, backfill_depth keyword arguments)` per case.
    """
    return imported_jobs("imported_depth", eval_root, import_root, force)


def imported_pose_scenes(eval_root: pathlib.Path,
                         import_root: pathlib.Path) -> list[pathlib.Path]:
    """Returns one imported pose evaluation per (task, dataset, scene).

    The pose (SfM or SLAM) scenes get their input-image thumbnails from these.
    """
    paths = []
    for kind in ("imported_trajectories", "imported_slam"):
        for path, _ in imported_jobs(kind, eval_root, import_root, force=True):
            paths.append(path)
    return one_case_per_scene(paths)


def one_case_per_scene(cases: list[pathlib.Path]) -> list[pathlib.Path]:
    """Returns the first evaluation of every (task, dataset, scene).

    The method or SLAM range of the evaluation does not matter.
    """
    seen = set()
    picked = []
    for path in cases:
        key = path.parts[-4:-1]
        if key not in seen:
            seen.add(key)
            picked.append(path)
    return picked


BACKFILL = {
    "trajectories": backfill_trajectory,
    "thumbnails": backfill_thumbnails,
    "depth": backfill_depth,
    "recon": backfill_recon,
    "gt_colors": backfill_gt_colors,
    "cameras": backfill_recon,
    "confidence": backfill_confidence,
    "imported_depth": backfill_depth,
    "imported_trajectories": backfill_trajectory,
    "imported_slam": backfill_trajectory,
    "imported_thumbnails": backfill_thumbnails,
    "imported_recon": backfill_recon
}


def _cases_without_cameras(eval_root: pathlib.Path,
                           force: bool) -> list[pathlib.Path]:
    """Returns the evaluated MVRecon cases whose viewer has no cameras.json."""
    cases = []
    for path in discover_cases(eval_root, "recon"):
        cameras = (path.parent / viewer_assets.VIEWER_DIR_NAME /
                   viewer_assets.CAMERAS_FILE)
        if ((force or not cameras.is_file()) and json.loads(
                path.read_text(encoding="utf-8")).get("scene_results")):
            cases.append(path)
    return cases


def _cases_to_backfill(
    kind: str,
    eval_root: pathlib.Path,
    scene_root: pathlib.Path,
    force: bool,
    import_root: pathlib.Path | None,
) -> tuple[list[pathlib.Path], dict[pathlib.Path, dict]]:
    """Returns one asset kind's cases and their backfill keyword arguments.

    Raises:
        SystemExit: An imported kind is asked for without `import_root`.
    """
    options: dict[pathlib.Path, dict] = {}
    if kind == "thumbnails":  # Existing thumbnails are skipped file by file.
        cases = one_case_per_scene(discover_cases(eval_root, "trajectories"))
    elif kind == "gt_colors":  # Per scene folder, not per case.
        cases = recon_scenes_without_gt_colors(scene_root, force)
    elif kind == "cameras":
        cases = _cases_without_cameras(eval_root, force)
    elif kind == "confidence":
        cases = confidence_cases(eval_root, force)
    elif kind.startswith("imported_"):
        if import_root is None:
            raise SystemExit(f"{kind} needs --import_root")
        if kind == "imported_thumbnails":
            # Existing thumbnails are skipped file by file.
            cases = imported_pose_scenes(eval_root, import_root)
        else:
            jobs = imported_jobs(kind, eval_root, import_root, force)
            cases = [path for path, _ in jobs]
            options = dict(jobs)
    else:
        cases = [
            path for path in discover_cases(eval_root, kind)
            if needs_assets(path, force)
        ]
        if kind == "recon":
            # The cloud the evaluation scored (fused, cropped and ICP-refined
            # in the GT frame) instead of re-running the TSDF fusion, which is
            # most of an MVRecon evaluation's time and memory (backfill_recon
            # re-fuses DTU).
            for path in cases:
                saved = saved_fused_point_path(path)
                if saved is not None:
                    options[path] = {"fused_point_path": saved}
            print(
                f"recon: {len(options)} of {len(cases)} cases reuse their "
                "evaluation's fused cloud",
                flush=True)
    return cases, options


def backfill(kind: str,
             eval_root: pathlib.Path,
             benchmark_root: pathlib.Path,
             *,
             workers: int,
             force: bool,
             import_root: pathlib.Path | None = None) -> list[dict]:
    """Backfills one asset kind.

    Args:
        kind: One of `BACKFILL`'s asset kinds.
        eval_root: The evaluation root holding `reports/` and
            `viewer_assets/`.
        benchmark_root: The benchmark data root.
        workers: The number of worker processes.
        force: Rewrite assets that are already up to date.
        import_root: The folder of imported release batches (imported kinds
            only).

    Returns:
        The failures, as {"case": ..., "error": ...}.
    """
    scene_root = eval_root / "viewer_assets" / "scenes"
    cases, options = _cases_to_backfill(kind, eval_root, scene_root, force,
                                        import_root)
    print(f"{kind}: {len(cases)} cases without viewer assets", flush=True)
    failures = []
    started = time.monotonic()
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        case_by_future = {
            pool.submit(BACKFILL[kind], path, benchmark_root, scene_root,
                        **options.get(path, {})):
                path for path in cases
        }
        for done, future in enumerate(futures.as_completed(case_by_future),
                                      start=1):
            try:
                future.result()
            except Exception as exc:  # pylint: disable=broad-except
                failures.append({
                    "case": str(case_by_future[future].parent),
                    "error": f"{type(exc).__name__}: {exc}"
                })
            if done % 25 == 0 or done == len(cases):
                elapsed = time.monotonic() - started
                print(
                    f"  {done}/{len(cases)} ({elapsed:.0f} s, "
                    f"{len(failures)} failed)",
                    flush=True)
    for failure in failures:
        print(f"  failed {failure['case']}: {failure['error']}", flush=True)
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kinds",
                        nargs="+",
                        choices=sorted(BACKFILL),
                        help="Which assets to backfill.")
    parser.add_argument("--eval_root",
                        type=pathlib.Path,
                        default=benchmark_artifact_analyzer.BENCHMARK_EVAL_ROOT)
    parser.add_argument("--benchmark_root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--workers",
                        type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--force",
                        action="store_true",
                        help="Rewrite assets that are already up to date.")
    parser.add_argument(
        "--import_root",
        type=pathlib.Path,
        default=None,
        help=("Folder of imported release batches (<batch>/reports, "
              "<batch>/artifacts) for the imported_* kinds, e.g. "
              "<benchmark root>/_ffb_eval."))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Backfills the asked-for kinds, then refreshes the viewer index."""
    args = parse_args(argv)
    for kind in args.kinds:
        backfill(kind,
                 args.eval_root,
                 args.benchmark_root,
                 workers=args.workers,
                 force=args.force,
                 import_root=args.import_root)
    print(
        f"Viewer index: {viewer_index.build_viewer_index(args.eval_root)} "
        "scenes",
        flush=True)


if __name__ == "__main__":
    main()
