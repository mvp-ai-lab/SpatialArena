# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Index of the viewer assets the evaluators wrote (spatialarena/eval/viewer_assets.py).

The artifact viewer's comparison modes list scenes and methods from three small files, written by the dashboard
collector after every collection (no other step runs after evaluation):

    <eval root>/viewer_assets/index/trajectories.json   SfM / Weak-Texture SfM / SLAM cases with trajectory.json
    <eval root>/viewer_assets/index/depth.json          MonoDepth / VideoDepth cases with frames.json
    <eval root>/viewer_assets/index/recon.json          MVRecon cases with cloud.json

Paths in them are relative to the eval root, which the dashboard serves as ``benchmark_eval/``. The canonical
report folders (spatialarena/report_layout.py) are indexed, and then ``viewer_assets/imported/``: the depth
images, pose trajectories and MVRecon clouds build_viewer_assets.py renders for imported release batches, used where
no canonical case exists.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from spatialarena.eval.viewer_assets import (
    CLOUD_FILE, DEPTH_FRAMES_FILE, SCENE_FILE, THUMBNAIL_DIR, TRAJECTORY_FILE, VIEWER_DIR_NAME, write_json,
)
from spatialarena.report_layout import FULL_REPORT_ROOTS
from spatialarena.task_track_matrix import on_matrix
from app.webpage.python.report_discovery import split_method_track

SLAM_RANGE_SEPARATOR = "@"
POSE_TASKS = ("SfM", "WeakTextureSfM")
DEPTH_TASKS = ("MonoDepth", "VideoDepth")
INDEX_DIR = Path("viewer_assets") / "index"
SCENE_ROOT = Path("viewer_assets") / "scenes"
IMPORTED_ROOT = Path("viewer_assets") / "imported"
# Imported SLAM cases keep their range: <IMPORTED_SLAM_ROOT>/full-slam-<range>/<method>/SLAM/<dataset>/<scene>/viewer/
IMPORTED_SLAM_ROOT = IMPORTED_ROOT / "slam"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _stored_metrics(eval_path: Path) -> dict:
    results = _read_json(eval_path).get("scene_results") or [{}]
    return results[0].get("metrics") or {}


def _scene_list(scenes: dict) -> list[dict]:
    for scene in scenes.values():
        scene["methods"].sort(key=lambda entry: entry["method"])
    return [scenes[key] for key in sorted(scenes)]


def index_trajectories(eval_root: Path) -> dict:
    """Pose cases (full-pose SfM tasks and every full-slam range) that have an aligned trajectory."""
    reports = eval_root / "reports"
    found = []
    pose_root = reports / FULL_REPORT_ROOTS["full-pose"] / "full-pose"
    seen: set[tuple[str, ...]] = set()
    for root in (pose_root, eval_root / IMPORTED_ROOT):  # imported cases (and their copied eval) after canonical
        for path in sorted(root.glob(f"*/*/*/*/{VIEWER_DIR_NAME}/{TRAJECTORY_FILE}")):
            method, task, dataset, scene = path.parts[-6:-2]
            if task not in POSE_TASKS or (method, task, dataset, scene) in seen:
                continue
            if root != pose_root and not on_matrix(task, split_method_track(method)[1], dataset):
                continue
            seen.add((method, task, dataset, scene))
            found.append((task, dataset, "", scene, method, path, path.parent.parent / "eval_sparse.json"))
    slam_root = reports / FULL_REPORT_ROOTS["full-slam"]
    for root in (slam_root, eval_root / IMPORTED_SLAM_ROOT):  # imported cases (and their copied eval) after canonical
        for path in sorted(root.glob(f"full-slam-*/*/SLAM/*/*/{VIEWER_DIR_NAME}/{TRAJECTORY_FILE}")):
            suite, method, task, dataset, scene = path.parts[-7:-2]
            if (suite, method, task, dataset, scene) in seen:
                continue
            seen.add((suite, method, task, dataset, scene))
            found.append((task, dataset, suite.removeprefix("full-slam-"), scene, method, path,
                          path.parent.parent / "eval_slam.json"))
    scenes: dict[tuple, dict] = {}
    for task, dataset, slam_range, scene, method, path, eval_path in found:
        dataset_key = f"{dataset}{SLAM_RANGE_SEPARATOR}{slam_range}" if slam_range else dataset
        metrics = _stored_metrics(eval_path)
        thumbnails = SCENE_ROOT / task / dataset / scene / THUMBNAIL_DIR
        entry = scenes.setdefault((task, dataset_key, scene), {
            "task": task, "dataset": dataset, "dataset_key": dataset_key, "slam_range": slam_range, "scene": scene,
            "thumbnail_dir": str(thumbnails) if (eval_root / thumbnails).is_dir() else None,
            "methods": [],
        })
        entry["methods"].append({
            "method": method, "path": str(path.relative_to(eval_root)),
            "ate_rmse": metrics.get("ate_rmse"), "completeness": metrics.get("completeness"),
        })
    return {"scenes": _scene_list(scenes)}


def index_depth(eval_root: Path) -> dict:
    """MonoDepth / VideoDepth cases with viewer frames, grouped by scene with the scene's shared assets."""
    roots = (eval_root / "reports" / FULL_REPORT_ROOTS["full-depth"] / "full-depth", eval_root / IMPORTED_ROOT)
    scenes: dict[tuple, dict] = {}
    seen: set[tuple[str, ...]] = set()
    for root, task in ((root, task) for root in roots for task in DEPTH_TASKS):
        for path in sorted(root.glob(f"*/{task}/*/*/{VIEWER_DIR_NAME}/{DEPTH_FRAMES_FILE}")):
            method, _, dataset, scene = path.parts[-6:-2]
            if (method, task, dataset, scene) in seen:
                continue  # a canonical case wins over an imported one
            if not on_matrix(task, split_method_track(method)[1], dataset):
                continue  # e.g. MonoDepth given GT depth (see task_track_matrix.py)
            frames = _read_json(path)
            if not frames.get("frames"):
                continue
            seen.add((method, task, dataset, scene))
            scene_dir = SCENE_ROOT / task / dataset / scene
            entry = scenes.setdefault((task, dataset, scene), {
                "task": task, "dataset": dataset, "dataset_key": dataset, "scene": scene,
                "scene_path": str(scene_dir), **_read_json(eval_root / scene_dir / SCENE_FILE),
                "has_rgb": (eval_root / scene_dir / "rgb").is_dir(), "methods": [],
            })
            metrics = frames.get("metrics") or {}
            entry["methods"].append({
                "method": method, "path": str(path.parent.relative_to(eval_root)),
                "absrel": metrics.get("abs_relative_difference"), "delta125": metrics.get("delta125_acc"),
                "frames": len(frames["frames"]),
            })
    return {"scenes": _scene_list(scenes)}


def index_recon(eval_root: Path) -> dict:
    """MVRecon cases with a viewer cloud, grouped by scene with the scene's GT points."""
    roots = (eval_root / "reports" / FULL_REPORT_ROOTS["full-depth"] / "full-depth", eval_root / IMPORTED_ROOT)
    scenes: dict[tuple, dict] = {}
    seen: set[tuple[str, ...]] = set()
    for path in (path for root in roots for path in sorted(root.glob(f"*/MVRecon/*/*/{VIEWER_DIR_NAME}/{CLOUD_FILE}"))):
        method, _, dataset, scene = path.parts[-6:-2]
        if (method, dataset, scene) in seen:
            continue  # a canonical case wins over an imported one
        cloud = _read_json(path)
        if not cloud.get("points"):
            continue
        seen.add((method, dataset, scene))
        scene_dir = SCENE_ROOT / "MVRecon" / dataset / scene
        entry = scenes.setdefault(("MVRecon", dataset, scene), {
            "task": "MVRecon", "dataset": dataset, "dataset_key": dataset, "scene": scene,
            "scene_path": str(scene_dir), **_read_json(eval_root / scene_dir / SCENE_FILE), "methods": [],
        })
        metrics = cloud.get("metrics") or {}
        entry["methods"].append({
            "method": method, "path": str(path.parent.relative_to(eval_root)), "points": cloud["points"],
            "fscore": metrics.get("fscore"), "accuracy": metrics.get("accuracy"),
            "completion": metrics.get("completion"), "error_percentiles": cloud.get("error_percentiles"),
        })
    return {"scenes": _scene_list(scenes)}



def build_viewer_index(eval_root: Path) -> dict[str, int]:
    """Write the index files; returns the number of scenes in each."""
    counts = {}
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for name, build in (("trajectories", index_trajectories), ("depth", index_depth), ("recon", index_recon)):
        index = build(eval_root)
        write_json(eval_root / INDEX_DIR / f"{name}.json", {"generated_at": generated_at, **index})
        counts[name] = len(index["scenes"])
    return counts
