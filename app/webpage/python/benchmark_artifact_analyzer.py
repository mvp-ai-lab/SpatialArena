#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Build visualization metadata for benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.paths import SPATIALARENA_ROOT  # noqa: E402
from spatialarena.eval.dataset_registry import (
    DEFAULT_BENCHMARK_ROOT,
    canonical_task_name,
    select_dataset_specs,
)
from spatialarena.eval.depth_prediction_evaluator import (
    DepthEvaluationConfig,
    _depth_key,
    find_depth_files,
    list_scene_gt_depth_files,
    pair_depth_files,
    pair_depth_files_by_integer_offset,
    pair_depth_files_by_nearest_timestamp,
    resolve_depth_config,
)
from spatialarena.eval.sparse_reconstruction_evaluator import (
    align_points_similarity,
    read_colmap_poses,
)
from spatialarena.eval.result_layout import resolve_sparse_model_path
from spatialarena.geometry import read_write_model
from spatialarena.task_track_matrix import on_matrix
from app.webpage.python.report_discovery import split_method_track


DEFAULT_OUTPUT_PATH = REPO_ROOT / "app" / "webpage" / "data" / "artifacts.json"
VISUALIZATION_FRAME_LIMIT = 8
SCAN_SKIP_DATASETS = {"scannet"}
DEPTH_TASKS = {"MonoDepth", "VideoDepth", "MVRecon"}
RECON_TASKS = {"SfM", "WeakTextureSfM", "SLAM"}
# Evaluation root now holding the artifacts, and roots older eval_*.json files still name (it moved off home on
# 2026-10-12). Prediction paths under a legacy root are read from the current one.
BENCHMARK_EVAL_ROOT = Path(
    os.environ.get("BENCHMARK_EVAL_ROOT", f"{SPATIALARENA_ROOT}/benchmarks/_evaluation")
)
LEGACY_EVAL_ROOTS = tuple(
    Path(root) for root in os.environ.get(
        "BENCHMARK_LEGACY_EVAL_ROOTS",
        f"{os.path.realpath(os.path.expanduser('~'))}/benchmark_evaluation:"
        f"{os.path.expanduser('~')}/benchmark_evaluation",
    ).split(":") if root
)


def relocated_path(value: str | None) -> Path:
    """``value`` as a Path, moved from a legacy evaluation root to ``BENCHMARK_EVAL_ROOT`` when it lies under one."""
    path = Path(value or "")
    for legacy_root in LEGACY_EVAL_ROOTS:
        if path.is_relative_to(legacy_root):
            return BENCHMARK_EVAL_ROOT / path.relative_to(legacy_root)
    return path
GT_DEPTH_FILE_CACHE: dict[tuple[str, str, str, str], list[Path]] = {}
PRED_DEPTH_FILE_CACHE: dict[str, list[Path]] = {}
PREDICTION_SCENE_FALLBACK_CACHE: dict[tuple[str, str, str, str, tuple[str, ...]], Path | None] = {}
PREDICTION_SCENE_INDEX_CACHE: dict[tuple[str, ...], dict[tuple[str, str, str, str], Path]] = {}
PREDICTION_SPARSE_SCENE_INDEX_CACHE: dict[tuple[str, ...], dict[tuple[str, str, str, str], Path]] = {}


@dataclass(frozen=True)
class RunRoot:
    scope: str
    method: str
    root: Path


def quote_asset_url(script: str, params: dict[str, str | int | float | None]) -> str:
    cleaned = {key: value for key, value in params.items() if value is not None}
    return f"cgi-bin/{script}?{urlencode(cleaned)}"


def file_asset_url(path: Path) -> str:
    return quote_asset_url("serve_image.py", {"path": str(path)})


def depth_asset_url(
    *,
    mode: str,
    path: Path,
    gt_path: Path | None,
    config: DepthEvaluationConfig,
) -> str:
    params: dict[str, str | int | float | None] = {
        "mode": mode,
        "path": str(path),
        "gt": str(gt_path) if gt_path is not None else None,
        "alignment": config.alignment,
        "gt_scale": config.gt_depth_scale,
        "pred_scale": config.pred_depth_scale,
        "min_depth": config.min_depth,
        "max_depth": config.max_depth,
        "post_clip_min": config.post_clip_min,
        "post_clip_max": config.post_clip_max,
        "alignment_max_resolution": config.alignment_max_resolution,
        "filter_gt_depth_range": int(config.filter_gt_depth_range),
    }
    return quote_asset_url("benchmark_asset.py", params)


def depth_frames_url(
    *,
    benchmark_root: Path,
    task: str,
    dataset: str,
    scene: str,
    prediction_scene_path: Path,
    limit: int,
) -> str:
    return quote_asset_url(
        "benchmark_asset.py",
        {
            "mode": "depth_frames",
            "benchmark_root": str(benchmark_root),
            "task": task,
            "dataset": dataset,
            "scene": scene,
            "prediction_scene_path": str(prediction_scene_path),
            "limit": limit,
        },
    )


def point_error_asset_url(pred_path: Path, sparse_gt_path: Path | None, max_points: int = 30000) -> str | None:
    if sparse_gt_path is None:
        return None
    return quote_asset_url(
        "benchmark_asset.py",
        {
            "mode": "point_error_ply",
            "path": str(pred_path),
            "reference_model": str(sparse_gt_path),
            "max_points": max_points,
        },
    )


def reconstruction_viewers_url(
    *,
    task: str,
    dataset: str,
    scene: str,
    method: str,
    prediction_scene_path: Path,
    pred_sparse_path: Path,
    gt_sparse_path: Path,
) -> str:
    return quote_asset_url(
        "benchmark_asset.py",
        {
            "mode": "reconstruction_viewers",
            "task": task,
            "dataset": dataset,
            "scene": scene,
            "method": method,
            "prediction_scene_path": str(prediction_scene_path),
            "prediction_sparse_path": str(pred_sparse_path),
            "sparse_gt_path": str(gt_sparse_path),
        },
    )


def parse_markdown_table_roots(path: Path) -> tuple[dict[tuple[str, str], RunRoot], dict[tuple[str, str], RunRoot]]:
    report_roots: dict[tuple[str, str], RunRoot] = {}
    artifact_roots: dict[tuple[str, str], RunRoot] = {}
    section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or cells[0] in {"Scope", "Report", "Root"}:
            continue
        if section == "Canonical Run Roots" and len(cells) >= 5:
            scope, method, _job, _status, root = cells[:5]
            if root.startswith("`") and root.endswith("`"):
                root = root[1:-1]
            report_roots[(scope, method)] = RunRoot(scope=scope, method=method, root=Path(root))
        elif section == "Artifact Roots" and len(cells) >= 3:
            scope, method, root = cells[:3]
            if root.startswith("`") and root.endswith("`"):
                root = root[1:-1]
            artifact_roots[(scope, method)] = RunRoot(scope=scope, method=method, root=Path(root))
    return report_roots, artifact_roots


def load_scene_infos(benchmark_root: Path) -> dict[tuple[str, str, str], object]:
    scene_infos = {}
    for spec in select_dataset_specs(include_scannet=False):
        if spec.dataset.lower() in SCAN_SKIP_DATASETS:
            continue
        data_path = spec.data_path(benchmark_root)
        if spec.metadata_required and not (data_path / "metadata.json").is_file():
            continue
        try:
            dataset = spec.create_dataset(benchmark_root)
            for scene_info in dataset.list_scenes():
                scene_infos[(scene_info.task, scene_info.dataset, scene_info.scene)] = scene_info
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    return scene_infos


def sort_scene_files(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: str(path).lower())


def limited_exact_depth_pairs(
    gt_files: list[Path],
    pred_files: list[Path],
    limit: int,
) -> list[tuple[Path, Path]]:
    pred_by_key: dict[str, Path] = {}
    for pred_path in pred_files:
        pred_by_key.setdefault(_depth_key(pred_path), pred_path)
    pairs = []
    for gt_path in gt_files:
        pred_path = pred_by_key.get(_depth_key(gt_path))
        if pred_path is None:
            continue
        pairs.append((gt_path, pred_path))
        if len(pairs) >= limit:
            break
    return pairs


def resolve_depth_pairs(
    scene_info,
    prediction_scene_path: Path,
    *,
    limit: int | None = None,
) -> tuple[list[tuple[Path, Path]], DepthEvaluationConfig]:
    config = resolve_depth_config(scene_info, DepthEvaluationConfig())
    gt_key = (scene_info.task, scene_info.dataset, scene_info.scene, str(scene_info.root_path))
    if gt_key not in GT_DEPTH_FILE_CACHE:
        GT_DEPTH_FILE_CACHE[gt_key] = list_scene_gt_depth_files(scene_info)
    gt_files = GT_DEPTH_FILE_CACHE[gt_key]
    pred_key = str(prediction_scene_path)
    if pred_key not in PRED_DEPTH_FILE_CACHE:
        PRED_DEPTH_FILE_CACHE[pred_key] = find_depth_files(prediction_scene_path)
    pred_files = PRED_DEPTH_FILE_CACHE[pred_key]
    pairs = limited_exact_depth_pairs(gt_files, pred_files, limit) if limit else []
    if not pairs:
        pairs, _missing_gt, _missing_pred, _used_index_fallback = pair_depth_files(gt_files, pred_files)
    if not pairs and scene_info.dataset.lower() == "bonn":
        pairs, _missing_gt, _missing_pred = pair_depth_files_by_nearest_timestamp(gt_files, pred_files)
    if not pairs and scene_info.dataset.lower() == "tanks_and_temples":
        pairs, _missing_gt, _missing_pred = pair_depth_files_by_integer_offset(
            gt_files,
            pred_files,
            gt_index_offset=1,
        )
    return (pairs[:limit] if limit else pairs), config


def depth_frames_for_scene(scene_info, prediction_scene_path: Path, limit: int) -> list[dict]:
    pairs, config = resolve_depth_pairs(scene_info, prediction_scene_path, limit=limit)
    frames = []
    for gt_path, pred_path in pairs[:limit]:
        name = pred_path.stem
        frames.append(
            {
                "name": name,
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
                "gt_url": depth_asset_url(mode="gt", path=gt_path, gt_path=None, config=config),
                "pred_url": depth_asset_url(mode="pred", path=pred_path, gt_path=gt_path, config=config),
                "error_url": depth_asset_url(mode="error", path=pred_path, gt_path=gt_path, config=config),
            }
        )
    return frames


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def build_prediction_scene_index(
    artifact_roots: dict[tuple[str, str], RunRoot],
) -> dict[tuple[str, str, str, str], Path]:
    roots = tuple(sorted({str(run_root.root) for run_root in artifact_roots.values()}))
    if roots in PREDICTION_SCENE_INDEX_CACHE:
        return PREDICTION_SCENE_INDEX_CACHE[roots]

    task_names = DEPTH_TASKS | RECON_TASKS
    index: dict[tuple[str, str, str, str], Path] = {}
    for root_value in roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for current_root, _dirnames, _filenames in os.walk(root, followlinks=True):
            path = Path(current_root)
            parts = path.parts
            for index_of_task, task in enumerate(parts):
                canonical = canonical_task_name(task)
                if canonical not in task_names or index_of_task + 3 >= len(parts):
                    continue
                dataset = parts[index_of_task + 1]
                method = parts[index_of_task + 2]
                scene = "/".join(parts[index_of_task + 3:])
                if not scene:
                    continue
                key = (canonical, dataset, method, scene)
                current = index.get(key)
                if current is None or path.stat().st_mtime > current.stat().st_mtime:
                    index[key] = path
    PREDICTION_SCENE_INDEX_CACHE[roots] = index
    return index


def build_prediction_sparse_scene_index(
    artifact_roots: dict[tuple[str, str], RunRoot],
) -> dict[tuple[str, str, str, str], Path]:
    roots = tuple(sorted({str(run_root.root) for run_root in artifact_roots.values()}))
    if roots in PREDICTION_SPARSE_SCENE_INDEX_CACHE:
        return PREDICTION_SPARSE_SCENE_INDEX_CACHE[roots]

    index: dict[tuple[str, str, str, str], Path] = {}
    for root_value in roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for current_root, dirnames, _filenames in os.walk(root, followlinks=True):
            path = Path(current_root)
            parts = path.parts
            for index_of_task, task in enumerate(parts):
                canonical = canonical_task_name(task)
                if canonical not in RECON_TASKS or index_of_task + 3 >= len(parts):
                    continue
                scene = "/".join(parts[index_of_task + 3:])
                if not scene or scene.endswith("/sparse") or "/sparse/" in scene:
                    continue
                if resolve_sparse_model_path(path) is None:
                    continue
                dataset = parts[index_of_task + 1]
                method = parts[index_of_task + 2]
                key = (canonical, dataset, method, scene)
                current = index.get(key)
                if current is None or path.stat().st_mtime > current.stat().st_mtime:
                    index[key] = path
                dirnames[:] = []
    PREDICTION_SPARSE_SCENE_INDEX_CACHE[roots] = index
    return index


def find_prediction_scene_fallback(
    *,
    prediction_scene_path: Path,
    artifact_roots: dict[tuple[str, str], RunRoot],
    task: str,
    dataset: str,
    method: str,
    scene: str,
) -> Path | None:
    if prediction_scene_path.is_dir():
        return prediction_scene_path

    roots = tuple(sorted({str(run_root.root) for run_root in artifact_roots.values()}))
    cache_key = (task, dataset, method, scene, roots)
    if cache_key in PREDICTION_SCENE_FALLBACK_CACHE:
        return PREDICTION_SCENE_FALLBACK_CACHE[cache_key]

    candidate = build_prediction_scene_index(artifact_roots).get((task, dataset, method, scene))
    if candidate is not None and candidate.is_dir():
        PREDICTION_SCENE_FALLBACK_CACHE[cache_key] = candidate
        return candidate
    PREDICTION_SCENE_FALLBACK_CACHE[cache_key] = None
    return None


def find_prediction_sparse_scene_fallback(
    *,
    prediction_scene_path: Path,
    artifact_roots: dict[tuple[str, str], RunRoot],
    task: str,
    dataset: str,
    method: str,
    scene: str,
) -> Path | None:
    if prediction_scene_path.is_dir() and resolve_sparse_model_path(prediction_scene_path) is not None:
        return prediction_scene_path
    return build_prediction_sparse_scene_index(artifact_roots).get((task, dataset, method, scene))


def scan_depth_cases(
    report_roots: dict[tuple[str, str], RunRoot],
    artifact_roots: dict[tuple[str, str], RunRoot],
    *,
    benchmark_root: Path,
    frame_limit: int,
    include_cases: set[tuple[str, str, str]] | None = None,
) -> list[dict]:
    cases = []
    scene_infos = load_scene_infos(benchmark_root)
    for (scope, method), run_root in sorted(report_roots.items()):
        if not scope.startswith("Depth"):
            continue
        for eval_path in sorted(run_root.root.rglob("eval_depth.json")):
            data = load_json(eval_path)
            for result in data.get("scene_results", []):
                task = result.get("task")
                dataset = result.get("dataset")
                scene = result.get("scene")
                task = canonical_task_name(task)
                case_method = str(result.get("method") or method)
                if include_cases is not None and (task, dataset, scene) not in include_cases:
                    continue
                if task not in DEPTH_TASKS or not dataset or dataset.lower() in SCAN_SKIP_DATASETS:
                    continue
                if not on_matrix(task, split_method_track(case_method)[1], dataset):
                    continue  # e.g. MonoDepth given GT depth (see task_track_matrix.py)
                prediction_scene_path = relocated_path(result.get("prediction_scene_path"))
                display_prediction_scene_path = find_prediction_scene_fallback(
                    prediction_scene_path=prediction_scene_path,
                    artifact_roots=artifact_roots,
                    task=task,
                    dataset=dataset,
                    method=case_method,
                    scene=scene,
                )
                if display_prediction_scene_path is None:
                    cases.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "method": case_method,
                            "scene": scene,
                            "kind": "depth",
                            "viewable": False,
                            "artifact_status": "prediction_scene_path_missing",
                            "artifact_status_details": (
                                "The evaluation metrics exist, but the saved prediction scene directory is not "
                                "present in the current artifact tree."
                            ),
                            "report_path": str(eval_path),
                            "prediction_scene_path": str(prediction_scene_path),
                            "source_prediction_scene_path": str(prediction_scene_path),
                            "num_evaluated_frames": result.get("num_evaluated_frames"),
                            "num_valid_pixels": result.get("num_valid_pixels"),
                            "metrics": result.get("metrics", {}),
                            "frames": [],
                        }
                    )
                    continue
                scene_info = scene_infos.get((task, dataset, scene))
                if scene_info is None:
                    cases.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "method": case_method,
                            "scene": scene,
                            "kind": "depth",
                            "viewable": False,
                            "artifact_status": "benchmark_scene_info_missing",
                            "artifact_status_details": (
                                "The evaluation metrics exist, but the benchmark scene metadata could not be "
                                "resolved for viewer pairing."
                            ),
                            "report_path": str(eval_path),
                            "prediction_scene_path": str(display_prediction_scene_path),
                            "source_prediction_scene_path": str(prediction_scene_path),
                            "num_evaluated_frames": result.get("num_evaluated_frames"),
                            "num_valid_pixels": result.get("num_valid_pixels"),
                            "metrics": result.get("metrics", {}),
                            "frames": [],
                        }
                    )
                    continue
                try:
                    depth_pairs, _config = resolve_depth_pairs(
                        scene_info,
                        display_prediction_scene_path,
                        limit=1,
                    )
                    pair_error = ""
                except Exception as error:  # pylint: disable=broad-exception-caught
                    depth_pairs = []
                    pair_error = str(error)
                if not depth_pairs:
                    cases.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "method": case_method,
                            "scene": scene,
                            "kind": "depth",
                            "viewable": False,
                            "artifact_status": "no_pairable_depth_files",
                            "artifact_status_details": pair_error or (
                                "The prediction directory exists, but it does not contain depth files that can be "
                                "paired with the benchmark ground truth."
                            ),
                            "report_path": str(eval_path),
                            "prediction_scene_path": str(display_prediction_scene_path),
                            "source_prediction_scene_path": str(prediction_scene_path),
                            "num_evaluated_frames": result.get("num_evaluated_frames"),
                            "num_valid_pixels": result.get("num_valid_pixels"),
                            "metrics": result.get("metrics", {}),
                            "frames": [],
                        }
                    )
                    continue
                cases.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "method": case_method,
                        "scene": scene,
                        "kind": "depth",
                        "viewable": True,
                        "report_path": str(eval_path),
                        "prediction_scene_path": str(display_prediction_scene_path),
                        "source_prediction_scene_path": str(prediction_scene_path),
                        "num_evaluated_frames": result.get("num_evaluated_frames"),
                        "num_valid_pixels": result.get("num_valid_pixels"),
                        "metrics": result.get("metrics", {}),
                        "frames": [],
                        "frames_url": depth_frames_url(
                            benchmark_root=benchmark_root,
                            task=task,
                            dataset=dataset,
                            scene=scene,
                            prediction_scene_path=display_prediction_scene_path,
                            limit=frame_limit,
                        ),
                    }
                )
    return cases


def camera_params(camera) -> tuple[float, float, float, float]:
    params = [float(value) for value in camera.params]
    width = float(camera.width)
    height = float(camera.height)
    if camera.model == "SIMPLE_PINHOLE":
        return params[0], params[0], params[1], params[2]
    if camera.model == "PINHOLE":
        return params[0], params[1], params[2], params[3]
    if camera.model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        return params[0], params[0], params[1], params[2]
    if camera.model in {"RADIAL", "RADIAL_FISHEYE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV"}:
        return params[0], params[1], params[2], params[3]
    return width, width, width * 0.5, height * 0.5


def extrinsic_to_camera_to_world(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation = read_write_model.qvec2rotmat(qvec)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ tvec
    return c2w


def three_camera_to_world(c2w_cv: np.ndarray) -> np.ndarray:
    cv_from_three = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w_cv @ cv_from_three


def finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def normalize_errors(errors: dict[str, float]) -> dict[str, float]:
    values = [value for value in errors.values() if math.isfinite(value)]
    if not values:
        return {}
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {key: 0.0 for key in errors}
    return {key: (value - low) / (high - low) for key, value in errors.items()}


def detect_colmap_model_ext(model_path: Path) -> str:
    if (model_path / "cameras.bin").is_file() and (model_path / "images.bin").is_file():
        return ".bin"
    if (model_path / "cameras.txt").is_file() and (model_path / "images.txt").is_file():
        return ".txt"
    raise FileNotFoundError(f"No COLMAP cameras/images files in {model_path}")


@lru_cache(maxsize=512)
def read_cameras_images_model(model_path_value: str):
    model_path = Path(model_path_value)
    ext = detect_colmap_model_ext(model_path)
    if ext == ".txt":
        cameras = read_write_model.read_cameras_text(str(model_path / "cameras.txt"))
        images = read_write_model.read_images_text(str(model_path / "images.txt"))
    else:
        cameras = read_write_model.read_cameras_binary(str(model_path / "cameras.bin"))
        images = read_write_model.read_images_binary(str(model_path / "images.bin"))
    return cameras, images


@lru_cache(maxsize=1024)
def read_colmap_poses_cached(model_path_value: str):
    return read_colmap_poses(Path(model_path_value))


def read_pose_manifest(model_path: Path, errors_by_name: dict[str, float] | None = None) -> list[dict]:
    errors_by_name = errors_by_name or {}
    normalized_errors = normalize_errors(errors_by_name)
    cameras, images = read_cameras_images_model(str(model_path))
    entries = []
    for image in sorted(images.values(), key=lambda item: item.name):
        camera = cameras.get(image.camera_id)
        if camera is None:
            continue
        fx, fy, cx, cy = camera_params(camera)
        c2w = extrinsic_to_camera_to_world(np.asarray(image.qvec), np.asarray(image.tvec))
        world_to_camera = np.linalg.inv(c2w)
        center = c2w[:3, 3]
        error = finite_float(errors_by_name.get(image.name))
        entry = {
            "name": image.name,
            "stem": Path(image.name).stem,
            "cameraId": int(image.camera_id),
            "width": int(camera.width),
            "height": int(camera.height),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "worldToCamera": world_to_camera.tolist(),
            "cameraToWorld": c2w.tolist(),
            "threeCameraToWorld": three_camera_to_world(c2w).tolist(),
            "center": center.tolist(),
        }
        if error is not None:
            entry["error"] = error
            entry["errorNormalized"] = normalized_errors.get(image.name, 0.0)
            entry["errorLabel"] = f"{error:.4g}"
        entries.append(entry)
    return entries


def pose_errors_by_name(pred_sparse_path: Path, gt_sparse_path: Path) -> dict[str, float]:
    try:
        pred_poses = read_colmap_poses_cached(str(pred_sparse_path))
        gt_poses = read_colmap_poses_cached(str(gt_sparse_path))
    except Exception:  # pylint: disable=broad-exception-caught
        return {}
    matched_names = sorted(set(pred_poses) & set(gt_poses))
    if len(matched_names) < 1:
        return {}
    pred_centers = np.stack([pred_poses[name].center for name in matched_names], axis=0)
    gt_centers = np.stack([gt_poses[name].center for name in matched_names], axis=0)
    try:
        aligned_pred, _scale, _rotation, _translation = align_points_similarity(pred_centers, gt_centers)
    except Exception:  # pylint: disable=broad-exception-caught
        aligned_pred = pred_centers
    errors = np.linalg.norm(aligned_pred - gt_centers, axis=1)
    return {name: float(error) for name, error in zip(matched_names, errors)}


def scene_viewer_manifest(
    *,
    title: str,
    method: str,
    model_path: Path,
    mesh_assets: list[dict],
    errors_by_name: dict[str, float] | None = None,
    messages: list[str] | None = None,
) -> dict:
    camera_poses = read_pose_manifest(model_path, errors_by_name=errors_by_name)
    frames = [
        {
            "imageName": pose["name"],
            "stem": pose["stem"],
            "cameraPose": pose,
            "metrics": {"poseError": pose.get("error")},
            "assets": {},
        }
        for pose in camera_poses
    ]
    return {
        "sceneName": title,
        "modelType": method,
        "frames": frames,
        "cameraPoses": camera_poses,
        "boxes3D": [],
        "boxEdges": [],
        "meshAssets": mesh_assets,
        "pointCloudAssets": mesh_assets,
        "splatAssets": [],
        "messages": messages or [],
    }


def scan_reconstruction_cases(
    report_roots: dict[tuple[str, str], RunRoot],
    artifact_roots: dict[tuple[str, str], RunRoot],
    include_cases: set[tuple[str, str, str]] | None = None,
) -> list[dict]:
    cases = []
    for (scope, method), run_root in sorted(report_roots.items()):
        if "SfM" not in scope and "WeakTexture" not in scope:
            continue
        eval_paths = sorted(run_root.root.rglob("eval_sparse.json")) + sorted(run_root.root.rglob("eval_slam.json"))
        for eval_path in eval_paths:
            data = load_json(eval_path)
            for result in data.get("scene_results", []):
                task = result.get("task")
                dataset = result.get("dataset")
                scene = result.get("scene")
                task = canonical_task_name(task)
                case_method = str(result.get("method") or method)
                if include_cases is not None and (task, dataset, scene) not in include_cases:
                    continue
                if task not in RECON_TASKS or not dataset or dataset.lower() in SCAN_SKIP_DATASETS:
                    continue
                if not on_matrix(task, split_method_track(case_method)[1], dataset):
                    continue  # e.g. MonoDepth given GT depth (see task_track_matrix.py)
                # SLAM cases are kept per range track, keyed like the metric tables ("7scenes@long").
                if task == "SLAM" and result.get("slam_range"):
                    dataset = f"{dataset}@{result['slam_range']}"
                pred_sparse_path = relocated_path(result.get("prediction_sparse_path"))
                gt_sparse_path = Path(result.get("sparse_gt_path", ""))
                prediction_scene_path = relocated_path(result.get("prediction_scene_path"))
                display_prediction_scene_path = find_prediction_sparse_scene_fallback(
                    prediction_scene_path=prediction_scene_path,
                    artifact_roots=artifact_roots,
                    task=task,
                    dataset=dataset,
                    method=case_method,
                    scene=scene,
                )
                if display_prediction_scene_path is None:
                    cases.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "method": case_method,
                            "scene": scene,
                            "kind": "reconstruction",
                            "viewable": False,
                            "artifact_status": "prediction_sparse_model_missing",
                            "artifact_status_details": (
                                "The evaluation metrics exist, but no usable saved sparse COLMAP prediction model is "
                                "present in the current artifact tree."
                            ),
                            "report_path": str(eval_path),
                            "prediction_scene_path": str(prediction_scene_path),
                            "source_prediction_scene_path": str(prediction_scene_path),
                            "prediction_sparse_path": str(pred_sparse_path),
                            "sparse_gt_path": str(gt_sparse_path),
                            "num_matched_images": result.get("num_matched_images"),
                            "metrics": result.get("metrics", {}),
                            "gt_viewer": None,
                            "pred_viewer": None,
                        }
                    )
                    continue
                display_pred_sparse_path = resolve_sparse_model_path(display_prediction_scene_path)
                if display_pred_sparse_path is None:
                    cases.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "method": case_method,
                            "scene": scene,
                            "kind": "reconstruction",
                            "viewable": False,
                            "artifact_status": "prediction_sparse_model_missing",
                            "artifact_status_details": (
                                "The prediction directory exists, but it does not contain a usable sparse COLMAP "
                                "model."
                            ),
                            "report_path": str(eval_path),
                            "prediction_scene_path": str(display_prediction_scene_path),
                            "source_prediction_scene_path": str(prediction_scene_path),
                            "prediction_sparse_path": str(pred_sparse_path),
                            "sparse_gt_path": str(gt_sparse_path),
                            "num_matched_images": result.get("num_matched_images"),
                            "metrics": result.get("metrics", {}),
                            "gt_viewer": None,
                            "pred_viewer": None,
                        }
                    )
                    continue
                pred_cloud = display_prediction_scene_path / "points3D_dense.ply"
                cases.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "method": case_method,
                        "scene": scene,
                        "kind": "reconstruction",
                        "viewable": True,
                        "report_path": str(eval_path),
                        "prediction_scene_path": str(display_prediction_scene_path),
                        "source_prediction_scene_path": str(prediction_scene_path),
                        "prediction_sparse_path": str(display_pred_sparse_path),
                        "sparse_gt_path": str(gt_sparse_path),
                        "num_matched_images": result.get("num_matched_images"),
                        "metrics": result.get("metrics", {}),
                        "point_color_modes": [
                            mode for mode in ("image", "error")
                            if mode == "image" or (pred_cloud.is_file() and gt_sparse_path.is_dir())
                        ],
                        "gt_viewer": None,
                        "pred_viewer": None,
                        "viewer_url": reconstruction_viewers_url(
                            task=task,
                            dataset=dataset,
                            scene=scene,
                            method=case_method,
                            prediction_scene_path=display_prediction_scene_path,
                            pred_sparse_path=display_pred_sparse_path,
                            gt_sparse_path=gt_sparse_path,
                        ),
                    }
                )
    return cases


def compact_case_options(cases: list[dict]) -> dict[str, list[str]]:
    values = {
        "tasks": sorted({case["task"] for case in cases}),
        "datasets": sorted({case["dataset"] for case in cases}),
        "methods": sorted({case["method"] for case in cases}),
        "scenes": sorted({case["scene"] for case in cases}),
    }
    return values


def build_artifact_visualization(
    *,
    benchmark_root: Path,
    manifest_path: Path,
    frame_limit: int,
) -> dict:
    report_roots, artifact_roots = parse_markdown_table_roots(manifest_path)
    depth_cases = scan_depth_cases(
        report_roots,
        artifact_roots,
        benchmark_root=benchmark_root,
        frame_limit=frame_limit,
    )
    reconstruction_cases = scan_reconstruction_cases(report_roots, artifact_roots)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_manifest": str(manifest_path),
        "benchmark_root": str(benchmark_root),
        "summary": {
            "depth_cases": len(depth_cases),
            "reconstruction_cases": len(reconstruction_cases),
            "methods": len({case["method"] for case in depth_cases + reconstruction_cases}),
            "tasks": len({case["task"] for case in depth_cases + reconstruction_cases}),
        },
        "run_roots": {
            "reports": {
                f"{scope}|{method}": str(run_root.root)
                for (scope, method), run_root in report_roots.items()
            },
            "artifacts": {
                f"{scope}|{method}": str(run_root.root)
                for (scope, method), run_root in artifact_roots.items()
            },
        },
        "depth": {
            "options": compact_case_options(depth_cases),
            "cases": depth_cases,
        },
        "reconstruction": {
            "options": compact_case_options(reconstruction_cases),
            "cases": reconstruction_cases,
        },
    }


def deduplicate_cases(cases: list[dict]) -> list[dict]:
    """Keep one case per (kind, task, dataset, method, scene): the one whose eval report is newest."""
    latest: dict[tuple[str, str, str, str, str], tuple[float, dict]] = {}
    for case in cases:
        key = (case["kind"], case["task"], case["dataset"], case["method"], case["scene"])
        try:
            mtime = Path(case.get("report_path", "")).stat().st_mtime
        except OSError:
            mtime = 0.0
        current = latest.get(key)
        if current is None or mtime > current[0]:
            latest[key] = (mtime, case)
    return [case for _mtime, case in sorted(latest.values(), key=lambda item: _case_sort_key(item[1]))]


def _case_sort_key(case: dict) -> tuple[str, str, str, str]:
    return (case["task"], case["dataset"], case["method"], case["scene"])


def build_artifact_visualization_from_roots(
    *,
    report_roots: list[Path],
    benchmark_root: Path,
    frame_limit: int,
    include_cases: set[tuple[str, str, str]] | None = None,
) -> dict:
    """Build artifact-viewer cases from discovered method report roots instead of a run manifest.

    Each root is scanned for both depth (``eval_depth.json``) and reconstruction
    (``eval_sparse.json``) results; the case method is the raw method ID stored in
    the eval payload, so it matches the IDs in ``results.json``. Prediction paths come
    from the eval payloads, so no artifact-root fallback index is built.
    """
    depth_roots = {(f"Depth {root}", root.name): RunRoot(scope="Depth", method=root.name, root=root)
                   for root in map(Path, report_roots)}
    reconstruction_roots = {(f"SfM {root}", root.name): RunRoot(scope="SfM", method=root.name, root=root)
                            for root in map(Path, report_roots)}
    depth_cases = deduplicate_cases(
        scan_depth_cases(
            depth_roots,
            {},
            benchmark_root=benchmark_root,
            frame_limit=frame_limit,
            include_cases=include_cases,
        )
    )
    reconstruction_cases = deduplicate_cases(
        scan_reconstruction_cases(reconstruction_roots, {}, include_cases=include_cases)
    )
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_manifest": None,
        "benchmark_root": str(benchmark_root),
        "summary": {
            "depth_cases": len(depth_cases),
            "reconstruction_cases": len(reconstruction_cases),
            "methods": len({case["method"] for case in depth_cases + reconstruction_cases}),
            "tasks": len({case["task"] for case in depth_cases + reconstruction_cases}),
        },
        "run_roots": {"reports": {str(root): str(root) for root in report_roots}, "artifacts": {}},
        "depth": {
            "options": compact_case_options(depth_cases),
            "cases": depth_cases,
        },
        "reconstruction": {
            "options": compact_case_options(reconstruction_cases),
            "cases": reconstruction_cases,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Markdown manifest with report/artifact root tables. collect_benchmark_metrics.py "
                        "--with-artifacts builds artifacts.json by discovery instead.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--frame-limit", type=int, default=VISUALIZATION_FRAME_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_artifact_visualization(
        benchmark_root=args.benchmark_root,
        manifest_path=args.manifest,
        frame_limit=max(1, args.frame_limit),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote benchmark artifact visualization data to {args.output}")
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
