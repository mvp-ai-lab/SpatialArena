# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Index of the viewer assets the evaluators wrote.

The evaluators write the assets with spatialarena/eval/viewer_assets.py. The
artifact viewer's comparison modes list scenes and methods from three small
files, written by the dashboard collector after every collection (no other
step runs after evaluation):

    <eval root>/viewer_assets/index/trajectories.json
        SfM / Weak-Texture SfM / SLAM cases with trajectory.json
    <eval root>/viewer_assets/index/depth.json
        MonoDepth / VideoDepth cases with frames.json
    <eval root>/viewer_assets/index/recon.json
        MVRecon cases with cloud.json

Paths in them are relative to the eval root, which the dashboard serves as
`benchmark_eval/`. The canonical report folders (spatialarena/report_layout.py)
are indexed, and then `viewer_assets/imported/`: the depth images, pose
trajectories and MVRecon clouds build_viewer_assets.py renders for imported
release batches, used where no canonical case exists.
"""

from __future__ import annotations

import json
import pathlib
import time

from app.webpage.python import report_discovery
from spatialarena import report_layout
from spatialarena import task_track_matrix
from spatialarena.eval import viewer_assets

SLAM_RANGE_SEPARATOR = "@"
POSE_TASKS = ("SfM", "WeakTextureSfM")
DEPTH_TASKS = ("MonoDepth", "VideoDepth")
INDEX_DIR = pathlib.Path("viewer_assets") / "index"
SCENE_ROOT = pathlib.Path("viewer_assets") / "scenes"
IMPORTED_ROOT = pathlib.Path("viewer_assets") / "imported"
# Imported SLAM cases keep their range:
# <IMPORTED_SLAM_ROOT>/full-slam-<range>/<method>/SLAM/<dataset>/<scene>/viewer/
IMPORTED_SLAM_ROOT = IMPORTED_ROOT / "slam"


def _read_json(path: pathlib.Path) -> dict:
    """Returns a JSON file's payload, or {} if it cannot be read or parsed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _stored_metrics(eval_path: pathlib.Path) -> dict:
    """Returns the first scene result's metrics of an eval_*.json, or {}."""
    results = _read_json(eval_path).get("scene_results") or [{}]
    return results[0].get("metrics") or {}


def _scene_list(scenes: dict) -> list[dict]:
    """Returns the scene entries sorted by key, each with sorted methods."""
    for scene in scenes.values():
        scene["methods"].sort(key=lambda entry: entry["method"])
    return [scenes[key] for key in sorted(scenes)]


def _method_track(method: str) -> str:
    """Returns the input track of a raw method ID."""
    return report_discovery.split_method_track(method)[1]


def _find_trajectories(eval_root: pathlib.Path) -> list[tuple]:
    """Returns the pose cases that have an aligned trajectory.

    Canonical cases come first; an imported case (with its copied eval file)
    is used only where no canonical case exists.

    Args:
        eval_root: The eval root.

    Returns:
        (task, dataset, SLAM range, scene, method, trajectory path, eval path)
        tuples; the SLAM range is "" outside SLAM.
    """
    reports = eval_root / "reports"
    found = []
    pose_root = (reports / report_layout.FULL_REPORT_ROOTS["full-pose"] /
                 "full-pose")
    seen: set[tuple[str, ...]] = set()
    pose_pattern = (f"*/*/*/*/{viewer_assets.VIEWER_DIR_NAME}/"
                    f"{viewer_assets.TRAJECTORY_FILE}")
    for root in (pose_root, eval_root / IMPORTED_ROOT):
        for path in sorted(root.glob(pose_pattern)):
            method, task, dataset, scene = path.parts[-6:-2]
            if task not in POSE_TASKS or (method, task, dataset, scene) in seen:
                continue
            if root != pose_root and not task_track_matrix.on_matrix(
                    task, _method_track(method), dataset):
                continue
            seen.add((method, task, dataset, scene))
            found.append((task, dataset, "", scene, method, path,
                          path.parent.parent / "eval_sparse.json"))
    slam_root = reports / report_layout.FULL_REPORT_ROOTS["full-slam"]
    slam_pattern = (f"full-slam-*/*/SLAM/*/*/{viewer_assets.VIEWER_DIR_NAME}/"
                    f"{viewer_assets.TRAJECTORY_FILE}")
    for root in (slam_root, eval_root / IMPORTED_SLAM_ROOT):
        for path in sorted(root.glob(slam_pattern)):
            suite, method, task, dataset, scene = path.parts[-7:-2]
            if (suite, method, task, dataset, scene) in seen:
                continue
            seen.add((suite, method, task, dataset, scene))
            found.append(
                (task, dataset, suite.removeprefix("full-slam-"), scene, method,
                 path, path.parent.parent / "eval_slam.json"))
    return found


def index_trajectories(eval_root: pathlib.Path) -> dict:
    """Indexes the pose cases that have an aligned trajectory.

    These are the full-pose SfM tasks and every full-slam range.

    Args:
        eval_root: The eval root.

    Returns:
        The index payload: {"scenes": [...]}.
    """
    scenes: dict[tuple, dict] = {}
    for (task, dataset, slam_range, scene, method, path,
         eval_path) in _find_trajectories(eval_root):
        if slam_range:
            dataset_key = f"{dataset}{SLAM_RANGE_SEPARATOR}{slam_range}"
        else:
            dataset_key = dataset
        metrics = _stored_metrics(eval_path)
        thumbnails = (SCENE_ROOT / task / dataset / scene /
                      viewer_assets.THUMBNAIL_DIR)
        if (eval_root / thumbnails).is_dir():
            thumbnail_dir = str(thumbnails)
        else:
            thumbnail_dir = None
        entry = scenes.setdefault(
            (task, dataset_key, scene), {
                "task": task,
                "dataset": dataset,
                "dataset_key": dataset_key,
                "slam_range": slam_range,
                "scene": scene,
                "thumbnail_dir": thumbnail_dir,
                "methods": [],
            })
        entry["methods"].append({
            "method": method,
            "path": str(path.relative_to(eval_root)),
            "ate_rmse": metrics.get("ate_rmse"),
            "completeness": metrics.get("completeness"),
        })
    return {"scenes": _scene_list(scenes)}


def _scene_entry(eval_root: pathlib.Path, task: str, dataset: str,
                 scene: str) -> dict:
    """Returns a new scene entry with the scene's shared assets.

    The shared assets' scene file (viewer_assets.SCENE_FILE) is merged in after
    the identifying keys; a key it repeats takes its value.
    """
    scene_dir = SCENE_ROOT / task / dataset / scene
    entry = {
        "task": task,
        "dataset": dataset,
        "dataset_key": dataset,
        "scene": scene,
        "scene_path": str(scene_dir),
    }
    entry.update(_read_json(eval_root / scene_dir / viewer_assets.SCENE_FILE))
    return entry


def index_depth(eval_root: pathlib.Path) -> dict:
    """Indexes the MonoDepth / VideoDepth cases with viewer frames.

    Cases are grouped by scene with the scene's shared assets; a canonical
    case wins over an imported one.

    Args:
        eval_root: The eval root.

    Returns:
        The index payload: {"scenes": [...]}.
    """
    roots = (eval_root / "reports" /
             report_layout.FULL_REPORT_ROOTS["full-depth"] / "full-depth",
             eval_root / IMPORTED_ROOT)
    scenes: dict[tuple, dict] = {}
    seen: set[tuple[str, ...]] = set()
    for root in roots:
        for task in DEPTH_TASKS:
            pattern = (f"*/{task}/*/*/{viewer_assets.VIEWER_DIR_NAME}/"
                       f"{viewer_assets.DEPTH_FRAMES_FILE}")
            for path in sorted(root.glob(pattern)):
                method, _, dataset, scene = path.parts[-6:-2]
                if (method, task, dataset, scene) in seen:
                    continue  # A canonical case wins over an imported one.
                if not task_track_matrix.on_matrix(task, _method_track(method),
                                                   dataset):
                    # E.g. MonoDepth given GT depth (see task_track_matrix.py).
                    continue
                frames = _read_json(path)
                if not frames.get("frames"):
                    continue
                seen.add((method, task, dataset, scene))
                new_entry = _scene_entry(eval_root, task, dataset, scene)
                scene_dir = SCENE_ROOT / task / dataset / scene
                new_entry["has_rgb"] = (eval_root / scene_dir / "rgb").is_dir()
                new_entry["methods"] = []
                entry = scenes.setdefault((task, dataset, scene), new_entry)
                metrics = frames.get("metrics") or {}
                entry["methods"].append({
                    "method": method,
                    "path": str(path.parent.relative_to(eval_root)),
                    "absrel": metrics.get("abs_relative_difference"),
                    "delta125": metrics.get("delta125_acc"),
                    "frames": len(frames["frames"]),
                })
    return {"scenes": _scene_list(scenes)}


def index_recon(eval_root: pathlib.Path) -> dict:
    """Indexes the MVRecon cases with a viewer cloud.

    Cases are grouped by scene with the scene's GT points; a canonical case
    wins over an imported one.

    Args:
        eval_root: The eval root.

    Returns:
        The index payload: {"scenes": [...]}.
    """
    roots = (eval_root / "reports" /
             report_layout.FULL_REPORT_ROOTS["full-depth"] / "full-depth",
             eval_root / IMPORTED_ROOT)
    scenes: dict[tuple, dict] = {}
    seen: set[tuple[str, ...]] = set()
    pattern = (f"*/MVRecon/*/*/{viewer_assets.VIEWER_DIR_NAME}/"
               f"{viewer_assets.CLOUD_FILE}")
    for root in roots:
        for path in sorted(root.glob(pattern)):
            method, _, dataset, scene = path.parts[-6:-2]
            if (method, dataset, scene) in seen:
                continue  # A canonical case wins over an imported one.
            cloud = _read_json(path)
            if not cloud.get("points"):
                continue
            seen.add((method, dataset, scene))
            new_entry = _scene_entry(eval_root, "MVRecon", dataset, scene)
            new_entry["methods"] = []
            entry = scenes.setdefault(("MVRecon", dataset, scene), new_entry)
            metrics = cloud.get("metrics") or {}
            entry["methods"].append({
                "method": method,
                "path": str(path.parent.relative_to(eval_root)),
                "points": cloud["points"],
                "fscore": metrics.get("fscore"),
                "accuracy": metrics.get("accuracy"),
                "completion": metrics.get("completion"),
                "error_percentiles": cloud.get("error_percentiles"),
            })
    return {"scenes": _scene_list(scenes)}


def build_viewer_index(eval_root: pathlib.Path) -> dict[str, int]:
    """Writes the index files.

    Args:
        eval_root: The eval root; the files go to `<eval root>/INDEX_DIR`.

    Returns:
        The number of scenes in each index, by index name.
    """
    counts = {}
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for name, build in (("trajectories", index_trajectories),
                        ("depth", index_depth), ("recon", index_recon)):
        index = build(eval_root)
        payload = {"generated_at": generated_at}
        payload.update(index)
        viewer_assets.write_json(eval_root / INDEX_DIR / f"{name}.json",
                                 payload)
        counts[name] = len(index["scenes"])
    return counts
