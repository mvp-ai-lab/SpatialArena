#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Build visualization metadata for benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import json
import math
import os
import pathlib
import sys
import urllib.parse

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.webpage.python import report_discovery  # noqa: E402
from spatialarena import paths as spatialarena_paths  # noqa: E402
from spatialarena import task_track_matrix  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402
from spatialarena.eval import depth_prediction_evaluator  # noqa: E402
from spatialarena.eval import result_layout  # noqa: E402
from spatialarena.eval import sparse_reconstruction_evaluator  # noqa: E402
from spatialarena.geometry import read_write_model  # noqa: E402

DEFAULT_OUTPUT_PATH = REPO_ROOT / "app" / "webpage" / "data" / "artifacts.json"
VISUALIZATION_FRAME_LIMIT = 8
SCAN_SKIP_DATASETS = {"scannet"}
DEPTH_TASKS = {"MonoDepth", "VideoDepth", "MVRecon"}
RECON_TASKS = {"SfM", "WeakTextureSfM", "SLAM"}
# Evaluation root now holding the artifacts, and roots older eval_*.json files
# still name (it moved off home on 2026-10-12). Prediction paths under a legacy
# root are read from the current one.
BENCHMARK_EVAL_ROOT = pathlib.Path(
    os.environ.get(
        "BENCHMARK_EVAL_ROOT",
        f"{spatialarena_paths.SPATIALARENA_ROOT}/benchmarks/_evaluation"))
LEGACY_EVAL_ROOTS = tuple(
    pathlib.Path(root) for root in os.environ.get(
        "BENCHMARK_LEGACY_EVAL_ROOTS",
        f"{os.path.realpath(os.path.expanduser('~'))}/benchmark_evaluation:"
        f"{os.path.expanduser('~')}/benchmark_evaluation",
    ).split(":") if root)

_DepthEvaluationConfig = depth_prediction_evaluator.DepthEvaluationConfig
# (task, dataset, method, scene) of a prediction scene.
_SceneKey = tuple[str, str, str, str]
_RootTable = dict[tuple[str, str], "RunRoot"]


def relocated_path(value: str | None) -> pathlib.Path:
    """Returns `value` as a Path, moved off a legacy evaluation root.

    A path under one of `LEGACY_EVAL_ROOTS` is moved to the same place under
    `BENCHMARK_EVAL_ROOT`; others are returned as they are.
    """
    path = pathlib.Path(value or "")
    for legacy_root in LEGACY_EVAL_ROOTS:
        if path.is_relative_to(legacy_root):
            return BENCHMARK_EVAL_ROOT / path.relative_to(legacy_root)
    return path


GT_DEPTH_FILE_CACHE: dict[tuple[str, str, str, str], list[pathlib.Path]] = {}
PRED_DEPTH_FILE_CACHE: dict[str, list[pathlib.Path]] = {}
_FallbackKey = tuple[str, str, str, str, tuple[str, ...]]
PREDICTION_SCENE_FALLBACK_CACHE: dict[_FallbackKey, pathlib.Path | None] = {}
PREDICTION_SCENE_INDEX_CACHE: dict[tuple[str, ...], dict[_SceneKey,
                                                         pathlib.Path]] = {}
PREDICTION_SPARSE_SCENE_INDEX_CACHE: dict[tuple[str, ...],
                                          dict[_SceneKey, pathlib.Path]] = {}


@dataclasses.dataclass(frozen=True)
class RunRoot:
    """A report or artifact root of one method's run.

    Attributes:
        scope: The manifest's scope label (e.g. "Depth ...", "SfM ...").
        method: The method ID.
        root: The root folder.
    """
    scope: str
    method: str
    root: pathlib.Path


def quote_asset_url(script: str, params: dict[str,
                                              str | int | float | None]) -> str:
    """Returns a `cgi-bin/<script>` URL with the non-None `params` encoded."""
    cleaned = {key: value for key, value in params.items() if value is not None}
    return f"cgi-bin/{script}?{urllib.parse.urlencode(cleaned)}"


def file_asset_url(path: pathlib.Path) -> str:
    """Returns the URL that serves an image file as it is."""
    return quote_asset_url("serve_image.py", {"path": str(path)})


def depth_asset_url(
    *,
    mode: str,
    path: pathlib.Path,
    gt_path: pathlib.Path | None,
    config: _DepthEvaluationConfig,
) -> str:
    """Returns the URL that renders one depth map.

    Args:
        mode: "gt", "pred" or "error".
        path: The depth map to render.
        gt_path: The GT depth map a prediction is aligned to, or None.
        config: The depth evaluation config whose alignment is applied.
    """
    if gt_path is not None:
        gt_value = str(gt_path)
    else:
        gt_value = None
    params: dict[str, str | int | float | None] = {
        "mode": mode,
        "path": str(path),
        "gt": gt_value,
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
    benchmark_root: pathlib.Path,
    task: str,
    dataset: str,
    scene: str,
    prediction_scene_path: pathlib.Path,
    limit: int,
) -> str:
    """Returns the URL that lists up to `limit` depth frames of a case."""
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


def point_error_asset_url(pred_path: pathlib.Path,
                          sparse_gt_path: pathlib.Path | None,
                          max_points: int = 30000) -> str | None:
    """Returns the URL of a prediction's point-error PLY, or None without GT.

    Args:
        pred_path: The predicted point cloud.
        sparse_gt_path: The GT sparse model the errors are measured against.
        max_points: The most points the PLY holds.
    """
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
    prediction_scene_path: pathlib.Path,
    pred_sparse_path: pathlib.Path,
    gt_sparse_path: pathlib.Path,
) -> str:
    """Returns the URL of a reconstruction case's GT and prediction viewers."""
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


def _unquote_code(root: str) -> str:
    """Strips the backticks around a Markdown code span."""
    if root.startswith("`") and root.endswith("`"):
        return root[1:-1]
    return root


def parse_markdown_table_roots(
        path: pathlib.Path) -> tuple[_RootTable, _RootTable]:
    """Reads the report and artifact roots of a Markdown run manifest.

    The manifest lists them in tables under "## Canonical Run Roots" (scope,
    method, job, status, root) and "## Artifact Roots" (scope, method, root).

    Returns:
        (report roots, artifact roots), each keyed by (scope, method).
    """
    report_roots: _RootTable = {}
    artifact_roots: _RootTable = {}
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
            root = _unquote_code(root)
            report_roots[(scope, method)] = RunRoot(scope=scope,
                                                    method=method,
                                                    root=pathlib.Path(root))
        elif section == "Artifact Roots" and len(cells) >= 3:
            scope, method, root = cells[:3]
            root = _unquote_code(root)
            artifact_roots[(scope, method)] = RunRoot(scope=scope,
                                                      method=method,
                                                      root=pathlib.Path(root))
    return report_roots, artifact_roots


def load_scene_infos(
        benchmark_root: pathlib.Path) -> dict[tuple[str, str, str], object]:
    """Returns the scene info of every loadable benchmark scene.

    Datasets in `SCAN_SKIP_DATASETS`, without required metadata or that fail
    to load are left out.

    Returns:
        The scene infos keyed by (task, dataset, scene).
    """
    scene_infos = {}
    for spec in dataset_registry.select_dataset_specs(include_scannet=False):
        if spec.dataset.lower() in SCAN_SKIP_DATASETS:
            continue
        data_path = spec.data_path(benchmark_root)
        if (spec.metadata_required and
                not (data_path / "metadata.json").is_file()):
            continue
        try:
            dataset = spec.create_dataset(benchmark_root)
            for scene_info in dataset.list_scenes():
                scene_infos[(scene_info.task, scene_info.dataset,
                             scene_info.scene)] = scene_info
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    return scene_infos


def sort_scene_files(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Sorts paths case-insensitively."""
    return sorted(paths, key=lambda path: str(path).lower())


def limited_exact_depth_pairs(
    gt_files: list[pathlib.Path],
    pred_files: list[pathlib.Path],
    limit: int,
) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Pairs GT and predicted depth files by their depth key, up to `limit`.

    Returns:
        (GT path, prediction path) pairs in GT order.
    """
    # pylint: disable=protected-access
    depth_key = depth_prediction_evaluator._depth_key
    pred_by_key: dict[str, pathlib.Path] = {}
    for pred_path in pred_files:
        pred_by_key.setdefault(depth_key(pred_path), pred_path)
    pairs = []
    for gt_path in gt_files:
        pred_path = pred_by_key.get(depth_key(gt_path))
        if pred_path is None:
            continue
        pairs.append((gt_path, pred_path))
        if len(pairs) >= limit:
            break
    return pairs


def resolve_depth_pairs(
    scene_info,
    prediction_scene_path: pathlib.Path,
    *,
    limit: int | None = None,
) -> tuple[list[tuple[pathlib.Path, pathlib.Path]], _DepthEvaluationConfig]:
    """Pairs a scene's GT depth files with a prediction's depth files.

    Exact key matches are tried first (with `limit`), then the evaluator's
    pairing, then nearest timestamps (Bonn) or an index offset (Tanks and
    Temples). File listings are cached per scene and prediction.

    Args:
        scene_info: The benchmark scene.
        prediction_scene_path: The prediction's scene folder.
        limit: The most pairs to return, or None for all.

    Returns:
        The (GT path, prediction path) pairs and the scene's depth config.
    """
    config = depth_prediction_evaluator.resolve_depth_config(
        scene_info, _DepthEvaluationConfig())
    gt_key = (scene_info.task, scene_info.dataset, scene_info.scene,
              str(scene_info.root_path))
    if gt_key not in GT_DEPTH_FILE_CACHE:
        GT_DEPTH_FILE_CACHE[gt_key] = (
            depth_prediction_evaluator.list_scene_gt_depth_files(scene_info))
    gt_files = GT_DEPTH_FILE_CACHE[gt_key]
    pred_key = str(prediction_scene_path)
    if pred_key not in PRED_DEPTH_FILE_CACHE:
        PRED_DEPTH_FILE_CACHE[pred_key] = (
            depth_prediction_evaluator.find_depth_files(prediction_scene_path))
    pred_files = PRED_DEPTH_FILE_CACHE[pred_key]
    if limit:
        pairs = limited_exact_depth_pairs(gt_files, pred_files, limit)
    else:
        pairs = []
    if not pairs:
        pairs, _missing_gt, _missing_pred, _used_index_fallback = (
            depth_prediction_evaluator.pair_depth_files(gt_files, pred_files))
    if not pairs and scene_info.dataset.lower() == "bonn":
        pairs, _missing_gt, _missing_pred = (
            depth_prediction_evaluator.pair_depth_files_by_nearest_timestamp(
                gt_files, pred_files))
    if not pairs and scene_info.dataset.lower() == "tanks_and_temples":
        pairs, _missing_gt, _missing_pred = (
            depth_prediction_evaluator.pair_depth_files_by_integer_offset(
                gt_files,
                pred_files,
                gt_index_offset=1,
            ))
    return (pairs[:limit] if limit else pairs), config


def depth_frames_for_scene(scene_info, prediction_scene_path: pathlib.Path,
                           limit: int) -> list[dict]:
    """Returns up to `limit` depth frames of a case with their render URLs."""
    pairs, config = resolve_depth_pairs(scene_info,
                                        prediction_scene_path,
                                        limit=limit)
    frames = []
    for gt_path, pred_path in pairs[:limit]:
        name = pred_path.stem
        frames.append({
            "name":
                name,
            "gt_path":
                str(gt_path),
            "pred_path":
                str(pred_path),
            "gt_url":
                depth_asset_url(mode="gt",
                                path=gt_path,
                                gt_path=None,
                                config=config),
            "pred_url":
                depth_asset_url(mode="pred",
                                path=pred_path,
                                gt_path=gt_path,
                                config=config),
            "error_url":
                depth_asset_url(mode="error",
                                path=pred_path,
                                gt_path=gt_path,
                                config=config),
        })
    return frames


def load_json(path: pathlib.Path) -> dict:
    """Reads a JSON file; returns {} when it cannot be read or parsed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def _artifact_root_values(artifact_roots: _RootTable) -> tuple[str, ...]:
    """Returns the distinct artifact root folders, sorted, as strings."""
    return tuple(
        sorted({str(run_root.root) for run_root in artifact_roots.values()}))


def build_prediction_scene_index(
    artifact_roots: _RootTable,) -> dict[_SceneKey, pathlib.Path]:
    """Indexes the prediction scene folders under the artifact roots.

    A folder `.../<task>/<dataset>/<method>/<scene...>` under a root is a
    prediction scene; per key the most recently modified folder wins. The
    index is cached per set of roots.

    Returns:
        The folders keyed by (task, dataset, method, scene).
    """
    roots = _artifact_root_values(artifact_roots)
    if roots in PREDICTION_SCENE_INDEX_CACHE:
        return PREDICTION_SCENE_INDEX_CACHE[roots]

    task_names = DEPTH_TASKS | RECON_TASKS
    index: dict[_SceneKey, pathlib.Path] = {}
    for root_value in roots:
        root = pathlib.Path(root_value)
        if not root.is_dir():
            continue
        for current_root, _dirnames, _filenames in os.walk(root,
                                                           followlinks=True):
            path = pathlib.Path(current_root)
            parts = path.parts
            for index_of_task, task in enumerate(parts):
                canonical = dataset_registry.canonical_task_name(task)
                if (canonical not in task_names or
                        index_of_task + 3 >= len(parts)):
                    continue
                dataset = parts[index_of_task + 1]
                method = parts[index_of_task + 2]
                scene = "/".join(parts[index_of_task + 3:])
                if not scene:
                    continue
                key = (canonical, dataset, method, scene)
                current = index.get(key)
                if (current is None or
                        path.stat().st_mtime > current.stat().st_mtime):
                    index[key] = path
    PREDICTION_SCENE_INDEX_CACHE[roots] = index
    return index


def build_prediction_sparse_scene_index(
    artifact_roots: _RootTable,) -> dict[_SceneKey, pathlib.Path]:
    """Indexes the prediction scene folders with a sparse model.

    Like `build_prediction_scene_index`, for the reconstruction tasks and only
    folders holding a sparse COLMAP model (the walk does not descend into
    them, and `.../sparse` folders are not scenes).

    Returns:
        The folders keyed by (task, dataset, method, scene).
    """
    roots = _artifact_root_values(artifact_roots)
    if roots in PREDICTION_SPARSE_SCENE_INDEX_CACHE:
        return PREDICTION_SPARSE_SCENE_INDEX_CACHE[roots]

    index: dict[_SceneKey, pathlib.Path] = {}
    for root_value in roots:
        root = pathlib.Path(root_value)
        if not root.is_dir():
            continue
        for current_root, dirnames, _filenames in os.walk(root,
                                                          followlinks=True):
            path = pathlib.Path(current_root)
            parts = path.parts
            for index_of_task, task in enumerate(parts):
                canonical = dataset_registry.canonical_task_name(task)
                if (canonical not in RECON_TASKS or
                        index_of_task + 3 >= len(parts)):
                    continue
                scene = "/".join(parts[index_of_task + 3:])
                if (not scene or scene.endswith("/sparse") or
                        "/sparse/" in scene):
                    continue
                if result_layout.resolve_sparse_model_path(path) is None:
                    continue
                dataset = parts[index_of_task + 1]
                method = parts[index_of_task + 2]
                key = (canonical, dataset, method, scene)
                current = index.get(key)
                if (current is None or
                        path.stat().st_mtime > current.stat().st_mtime):
                    index[key] = path
                dirnames[:] = []
    PREDICTION_SPARSE_SCENE_INDEX_CACHE[roots] = index
    return index


def find_prediction_scene_fallback(
    *,
    prediction_scene_path: pathlib.Path,
    artifact_roots: _RootTable,
    task: str,
    dataset: str,
    method: str,
    scene: str,
) -> pathlib.Path | None:
    """Returns a case's prediction scene folder, or None when it is missing.

    The stored path is used when it exists; otherwise the artifact roots'
    index is searched for the case. Results are cached.
    """
    if prediction_scene_path.is_dir():
        return prediction_scene_path

    roots = _artifact_root_values(artifact_roots)
    cache_key = (task, dataset, method, scene, roots)
    if cache_key in PREDICTION_SCENE_FALLBACK_CACHE:
        return PREDICTION_SCENE_FALLBACK_CACHE[cache_key]

    candidate = build_prediction_scene_index(artifact_roots).get(
        (task, dataset, method, scene))
    if candidate is not None and candidate.is_dir():
        PREDICTION_SCENE_FALLBACK_CACHE[cache_key] = candidate
        return candidate
    PREDICTION_SCENE_FALLBACK_CACHE[cache_key] = None
    return None


def find_prediction_sparse_scene_fallback(
    *,
    prediction_scene_path: pathlib.Path,
    artifact_roots: _RootTable,
    task: str,
    dataset: str,
    method: str,
    scene: str,
) -> pathlib.Path | None:
    """Returns a case's prediction scene folder with a sparse model, or None.

    The stored path is used when it holds a sparse model; otherwise the
    artifact roots' sparse index is searched for the case.
    """
    if (prediction_scene_path.is_dir() and
            result_layout.resolve_sparse_model_path(prediction_scene_path)
            is not None):
        return prediction_scene_path
    return build_prediction_sparse_scene_index(artifact_roots).get(
        (task, dataset, method, scene))


def _wanted_result(task, dataset, scene, case_method: str, *, tasks,
                   include_cases) -> bool:
    """Whether a scene result is a case of the scanned tasks.

    Args:
        task: The result's canonical task.
        dataset: The result's dataset.
        scene: The result's scene.
        case_method: The result's raw method ID.
        tasks: The tasks to keep.
        include_cases: The (task, dataset, scene) cases to keep, or None for
            all.
    """
    if include_cases is not None and (task, dataset,
                                      scene) not in include_cases:
        return False
    if (task not in tasks or not dataset or
            dataset.lower() in SCAN_SKIP_DATASETS):
        return False
    track = report_discovery.split_method_track(case_method)[1]
    # E.g. MonoDepth given GT depth (see task_track_matrix.py).
    return task_track_matrix.on_matrix(task, track, dataset)


def _depth_case(*,
                task: str,
                dataset: str,
                method: str,
                scene: str,
                viewable: bool,
                eval_path: pathlib.Path,
                prediction_scene_path: pathlib.Path,
                source_prediction_scene_path: pathlib.Path,
                result: dict,
                artifact_status: str | None = None,
                artifact_status_details: str | None = None) -> dict:
    """Returns a depth case record, without its frames URL.

    The artifact status fields are only set when `artifact_status` is given.
    """
    case = {
        "task": task,
        "dataset": dataset,
        "method": method,
        "scene": scene,
        "kind": "depth",
        "viewable": viewable,
    }
    if artifact_status is not None:
        case["artifact_status"] = artifact_status
        case["artifact_status_details"] = artifact_status_details
    case["report_path"] = str(eval_path)
    case["prediction_scene_path"] = str(prediction_scene_path)
    case["source_prediction_scene_path"] = str(source_prediction_scene_path)
    case["num_evaluated_frames"] = result.get("num_evaluated_frames")
    case["num_valid_pixels"] = result.get("num_valid_pixels")
    case["metrics"] = result.get("metrics", {})
    case["frames"] = []
    return case


def _scan_depth_result(result: dict, method: str, eval_path: pathlib.Path, *,
                       artifact_roots: _RootTable, scene_infos: dict,
                       benchmark_root: pathlib.Path, frame_limit: int,
                       include_cases) -> dict | None:
    """Returns the case of one depth scene result, or None to skip it."""
    task = result.get("task")
    dataset = result.get("dataset")
    scene = result.get("scene")
    task = dataset_registry.canonical_task_name(task)
    case_method = str(result.get("method") or method)
    if not _wanted_result(task,
                          dataset,
                          scene,
                          case_method,
                          tasks=DEPTH_TASKS,
                          include_cases=include_cases):
        return None
    prediction_scene_path = relocated_path(result.get("prediction_scene_path"))
    display_prediction_scene_path = find_prediction_scene_fallback(
        prediction_scene_path=prediction_scene_path,
        artifact_roots=artifact_roots,
        task=task,
        dataset=dataset,
        method=case_method,
        scene=scene,
    )
    identity = {
        "task": task,
        "dataset": dataset,
        "method": case_method,
        "scene": scene,
        "eval_path": eval_path,
        "source_prediction_scene_path": prediction_scene_path,
        "result": result,
    }
    if display_prediction_scene_path is None:
        return _depth_case(
            **identity,
            viewable=False,
            prediction_scene_path=prediction_scene_path,
            artifact_status="prediction_scene_path_missing",
            artifact_status_details=(
                "The evaluation metrics exist, but the saved prediction scene "
                "directory is not present in the current artifact tree."),
        )
    scene_info = scene_infos.get((task, dataset, scene))
    if scene_info is None:
        return _depth_case(
            **identity,
            viewable=False,
            prediction_scene_path=display_prediction_scene_path,
            artifact_status="benchmark_scene_info_missing",
            artifact_status_details=(
                "The evaluation metrics exist, but the benchmark scene "
                "metadata could not be resolved for viewer pairing."),
        )
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
        return _depth_case(
            **identity,
            viewable=False,
            prediction_scene_path=display_prediction_scene_path,
            artifact_status="no_pairable_depth_files",
            artifact_status_details=pair_error or
            ("The prediction directory exists, but it does not contain depth "
             "files that can be paired with the benchmark ground truth."),
        )
    case = _depth_case(**identity,
                       viewable=True,
                       prediction_scene_path=display_prediction_scene_path)
    case["frames_url"] = depth_frames_url(
        benchmark_root=benchmark_root,
        task=task,
        dataset=dataset,
        scene=scene,
        prediction_scene_path=display_prediction_scene_path,
        limit=frame_limit,
    )
    return case


def scan_depth_cases(
    report_roots: _RootTable,
    artifact_roots: _RootTable,
    *,
    benchmark_root: pathlib.Path,
    frame_limit: int,
    include_cases: set[tuple[str, str, str]] | None = None,
) -> list[dict]:
    """Lists the depth cases of the "Depth" report roots' eval_depth.json.

    Every on-matrix depth scene result becomes a case; one whose prediction,
    scene metadata or pairable depth files are missing is kept as not
    viewable, with an `artifact_status` saying why.

    Args:
        report_roots: The report roots keyed by (scope, method).
        artifact_roots: The artifact roots to find moved predictions in.
        benchmark_root: The benchmark data root.
        frame_limit: The most frames a case's frames URL lists.
        include_cases: The (task, dataset, scene) cases to keep, or None for
            all.
    """
    cases = []
    scene_infos = load_scene_infos(benchmark_root)
    for (scope, method), run_root in sorted(report_roots.items()):
        if not scope.startswith("Depth"):
            continue
        for eval_path in sorted(run_root.root.rglob("eval_depth.json")):
            data = load_json(eval_path)
            for result in data.get("scene_results", []):
                case = _scan_depth_result(result,
                                          method,
                                          eval_path,
                                          artifact_roots=artifact_roots,
                                          scene_infos=scene_infos,
                                          benchmark_root=benchmark_root,
                                          frame_limit=frame_limit,
                                          include_cases=include_cases)
                if case is not None:
                    cases.append(case)
    return cases


def camera_params(camera) -> tuple[float, float, float, float]:
    """Returns a COLMAP camera's (fx, fy, cx, cy).

    Unknown camera models fall back to a focal length of the image width and
    the image center.
    """
    params = [float(value) for value in camera.params]
    width = float(camera.width)
    height = float(camera.height)
    if camera.model == "SIMPLE_PINHOLE":
        return params[0], params[0], params[1], params[2]
    if camera.model == "PINHOLE":
        return params[0], params[1], params[2], params[3]
    if camera.model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        return params[0], params[0], params[1], params[2]
    if camera.model in {
            "RADIAL", "RADIAL_FISHEYE", "OPENCV", "OPENCV_FISHEYE",
            "FULL_OPENCV", "FOV"
    }:
        return params[0], params[1], params[2], params[3]
    return width, width, width * 0.5, height * 0.5


def extrinsic_to_camera_to_world(qvec: np.ndarray,
                                 tvec: np.ndarray) -> np.ndarray:
    """Returns the 4x4 camera-to-world matrix of a COLMAP pose.

    COLMAP stores the world-to-camera rotation (as a quaternion) and
    translation.
    """
    rotation = read_write_model.qvec2rotmat(qvec)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ tvec
    return c2w


def three_camera_to_world(c2w_cv: np.ndarray) -> np.ndarray:
    """Converts an OpenCV camera-to-world matrix to three.js camera axes."""
    cv_from_three = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w_cv @ cv_from_three


def finite_float(value: float | None) -> float | None:
    """Returns `value` as a float, or None when it is None or not finite."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def normalize_errors(errors: dict[str, float]) -> dict[str, float]:
    """Min-max normalizes errors to [0, 1] over their finite values.

    Returns:
        The normalized errors ({} without finite values; all 0.0 when the
        finite values are equal).
    """
    values = [value for value in errors.values() if math.isfinite(value)]
    if not values:
        return {}
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {key: 0.0 for key in errors}
    return {key: (value - low) / (high - low) for key, value in errors.items()}


def detect_colmap_model_ext(model_path: pathlib.Path) -> str:
    """Returns ".bin" or ".txt", the format of a COLMAP model's files.

    Raises:
        FileNotFoundError: The folder has neither cameras/images file pair.
    """
    if ((model_path / "cameras.bin").is_file() and
        (model_path / "images.bin").is_file()):
        return ".bin"
    if ((model_path / "cameras.txt").is_file() and
        (model_path / "images.txt").is_file()):
        return ".txt"
    raise FileNotFoundError(f"No COLMAP cameras/images files in {model_path}")


@functools.lru_cache(maxsize=512)
def read_cameras_images_model(model_path_value: str):
    """Reads (and caches) a COLMAP model's cameras and images.

    Returns:
        (cameras, images) as read_write_model returns them.
    """
    model_path = pathlib.Path(model_path_value)
    ext = detect_colmap_model_ext(model_path)
    if ext == ".txt":
        cameras = read_write_model.read_cameras_text(
            str(model_path / "cameras.txt"))
        images = read_write_model.read_images_text(
            str(model_path / "images.txt"))
    else:
        cameras = read_write_model.read_cameras_binary(
            str(model_path / "cameras.bin"))
        images = read_write_model.read_images_binary(
            str(model_path / "images.bin"))
    return cameras, images


@functools.lru_cache(maxsize=1024)
def read_colmap_poses_cached(model_path_value: str):
    """Reads (and caches) a COLMAP model's poses."""
    return sparse_reconstruction_evaluator.read_colmap_poses(
        pathlib.Path(model_path_value))


def read_pose_manifest(
        model_path: pathlib.Path,
        errors_by_name: dict[str, float] | None = None) -> list[dict]:
    """Returns the viewer's camera pose entries of a COLMAP model.

    Args:
        model_path: The COLMAP model folder.
        errors_by_name: Per image name, its pose error; an image with a finite
            error gets "error", "errorNormalized" and "errorLabel" fields.

    Returns:
        One entry per image with a known camera, sorted by image name.
    """
    errors_by_name = errors_by_name or {}
    normalized_errors = normalize_errors(errors_by_name)
    cameras, images = read_cameras_images_model(str(model_path))
    entries = []
    for image in sorted(images.values(), key=lambda item: item.name):
        camera = cameras.get(image.camera_id)
        if camera is None:
            continue
        fx, fy, cx, cy = camera_params(camera)
        c2w = extrinsic_to_camera_to_world(np.asarray(image.qvec),
                                           np.asarray(image.tvec))
        world_to_camera = np.linalg.inv(c2w)
        center = c2w[:3, 3]
        error = finite_float(errors_by_name.get(image.name))
        entry = {
            "name": image.name,
            "stem": pathlib.Path(image.name).stem,
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


def pose_errors_by_name(pred_sparse_path: pathlib.Path,
                        gt_sparse_path: pathlib.Path) -> dict[str, float]:
    """Returns each matched image's camera center error after alignment.

    The predicted centers are similarity-aligned to the GT ones (unaligned if
    the alignment fails).

    Returns:
        The errors by image name; {} when a model cannot be read or no image
        matches.
    """
    try:
        pred_poses = read_colmap_poses_cached(str(pred_sparse_path))
        gt_poses = read_colmap_poses_cached(str(gt_sparse_path))
    except Exception:  # pylint: disable=broad-exception-caught
        return {}
    matched_names = sorted(set(pred_poses) & set(gt_poses))
    if len(matched_names) < 1:
        return {}
    pred_centers = np.stack([pred_poses[name].center for name in matched_names],
                            axis=0)
    gt_centers = np.stack([gt_poses[name].center for name in matched_names],
                          axis=0)
    try:
        aligned_pred, _scale, _rotation, _translation = (
            sparse_reconstruction_evaluator.align_points_similarity(
                pred_centers, gt_centers))
    except Exception:  # pylint: disable=broad-exception-caught
        aligned_pred = pred_centers
    errors = np.linalg.norm(aligned_pred - gt_centers, axis=1)
    return {name: float(error) for name, error in zip(matched_names, errors)}


def scene_viewer_manifest(
    *,
    title: str,
    method: str,
    model_path: pathlib.Path,
    mesh_assets: list[dict],
    errors_by_name: dict[str, float] | None = None,
    messages: list[str] | None = None,
) -> dict:
    """Returns the scene viewer's manifest of a COLMAP model.

    Args:
        title: The scene name shown.
        method: The model type shown.
        model_path: The COLMAP model folder.
        mesh_assets: The mesh / point cloud assets to show.
        errors_by_name: Per image name, its pose error.
        messages: Messages shown with the scene.
    """
    camera_poses = read_pose_manifest(model_path, errors_by_name=errors_by_name)
    frames = [{
        "imageName": pose["name"],
        "stem": pose["stem"],
        "cameraPose": pose,
        "metrics": {
            "poseError": pose.get("error")
        },
        "assets": {},
    } for pose in camera_poses]
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


def _reconstruction_case(*,
                         task: str,
                         dataset: str,
                         method: str,
                         scene: str,
                         viewable: bool,
                         eval_path: pathlib.Path,
                         prediction_scene_path: pathlib.Path,
                         source_prediction_scene_path: pathlib.Path,
                         prediction_sparse_path: pathlib.Path,
                         gt_sparse_path: pathlib.Path,
                         result: dict,
                         artifact_status: str | None = None,
                         artifact_status_details: str | None = None,
                         point_color_modes: list[str] | None = None) -> dict:
    """Returns a reconstruction case record, without its viewer URL.

    The artifact status fields are only set when `artifact_status` is given,
    "point_color_modes" only when `point_color_modes` is.
    """
    case = {
        "task": task,
        "dataset": dataset,
        "method": method,
        "scene": scene,
        "kind": "reconstruction",
        "viewable": viewable,
    }
    if artifact_status is not None:
        case["artifact_status"] = artifact_status
        case["artifact_status_details"] = artifact_status_details
    case["report_path"] = str(eval_path)
    case["prediction_scene_path"] = str(prediction_scene_path)
    case["source_prediction_scene_path"] = str(source_prediction_scene_path)
    case["prediction_sparse_path"] = str(prediction_sparse_path)
    case["sparse_gt_path"] = str(gt_sparse_path)
    case["num_matched_images"] = result.get("num_matched_images")
    case["metrics"] = result.get("metrics", {})
    if point_color_modes is not None:
        case["point_color_modes"] = point_color_modes
    case["gt_viewer"] = None
    case["pred_viewer"] = None
    return case


def _scan_reconstruction_result(result: dict, method: str,
                                eval_path: pathlib.Path, *,
                                artifact_roots: _RootTable,
                                include_cases) -> dict | None:
    """Returns the case of one pose scene result, or None to skip it."""
    task = result.get("task")
    dataset = result.get("dataset")
    scene = result.get("scene")
    task = dataset_registry.canonical_task_name(task)
    case_method = str(result.get("method") or method)
    if not _wanted_result(task,
                          dataset,
                          scene,
                          case_method,
                          tasks=RECON_TASKS,
                          include_cases=include_cases):
        return None
    # SLAM cases are kept per range track, keyed like the metric tables
    # ("7scenes@long").
    if task == "SLAM" and result.get("slam_range"):
        dataset = f"{dataset}@{result['slam_range']}"
    pred_sparse_path = relocated_path(result.get("prediction_sparse_path"))
    gt_sparse_path = pathlib.Path(result.get("sparse_gt_path", ""))
    prediction_scene_path = relocated_path(result.get("prediction_scene_path"))
    display_prediction_scene_path = find_prediction_sparse_scene_fallback(
        prediction_scene_path=prediction_scene_path,
        artifact_roots=artifact_roots,
        task=task,
        dataset=dataset,
        method=case_method,
        scene=scene,
    )
    identity = {
        "task": task,
        "dataset": dataset,
        "method": case_method,
        "scene": scene,
        "eval_path": eval_path,
        "source_prediction_scene_path": prediction_scene_path,
        "gt_sparse_path": gt_sparse_path,
        "result": result,
    }
    if display_prediction_scene_path is None:
        return _reconstruction_case(
            **identity,
            viewable=False,
            prediction_scene_path=prediction_scene_path,
            prediction_sparse_path=pred_sparse_path,
            artifact_status="prediction_sparse_model_missing",
            artifact_status_details=(
                "The evaluation metrics exist, but no usable saved sparse "
                "COLMAP prediction model is present in the current artifact "
                "tree."),
        )
    display_pred_sparse_path = result_layout.resolve_sparse_model_path(
        display_prediction_scene_path)
    if display_pred_sparse_path is None:
        return _reconstruction_case(
            **identity,
            viewable=False,
            prediction_scene_path=display_prediction_scene_path,
            prediction_sparse_path=pred_sparse_path,
            artifact_status="prediction_sparse_model_missing",
            artifact_status_details=(
                "The prediction directory exists, but it does not contain a "
                "usable sparse COLMAP model."),
        )
    pred_cloud = display_prediction_scene_path / "points3D_dense.ply"
    point_color_modes = [
        mode for mode in ("image", "error")
        if mode == "image" or (pred_cloud.is_file() and gt_sparse_path.is_dir())
    ]
    case = _reconstruction_case(
        **identity,
        viewable=True,
        prediction_scene_path=display_prediction_scene_path,
        prediction_sparse_path=display_pred_sparse_path,
        point_color_modes=point_color_modes,
    )
    case["viewer_url"] = reconstruction_viewers_url(
        task=task,
        dataset=dataset,
        scene=scene,
        method=case_method,
        prediction_scene_path=display_prediction_scene_path,
        pred_sparse_path=display_pred_sparse_path,
        gt_sparse_path=gt_sparse_path,
    )
    return case


def scan_reconstruction_cases(
    report_roots: _RootTable,
    artifact_roots: _RootTable,
    include_cases: set[tuple[str, str, str]] | None = None,
) -> list[dict]:
    """Lists the pose cases of the SfM / Weak-Texture report roots.

    Every on-matrix scene result of their eval_sparse.json and eval_slam.json
    becomes a case; one without a usable sparse prediction model is kept as
    not viewable, with an `artifact_status` saying why.

    Args:
        report_roots: The report roots keyed by (scope, method).
        artifact_roots: The artifact roots to find moved predictions in.
        include_cases: The (task, dataset, scene) cases to keep, or None for
            all.
    """
    cases = []
    for (scope, method), run_root in sorted(report_roots.items()):
        if "SfM" not in scope and "WeakTexture" not in scope:
            continue
        eval_paths = (sorted(run_root.root.rglob("eval_sparse.json")) +
                      sorted(run_root.root.rglob("eval_slam.json")))
        for eval_path in eval_paths:
            data = load_json(eval_path)
            for result in data.get("scene_results", []):
                case = _scan_reconstruction_result(
                    result,
                    method,
                    eval_path,
                    artifact_roots=artifact_roots,
                    include_cases=include_cases)
                if case is not None:
                    cases.append(case)
    return cases


def compact_case_options(cases: list[dict]) -> dict[str, list[str]]:
    """Returns the sorted distinct tasks, datasets, methods and scenes."""
    values = {
        "tasks": sorted({case["task"] for case in cases}),
        "datasets": sorted({case["dataset"] for case in cases}),
        "methods": sorted({case["method"] for case in cases}),
        "scenes": sorted({case["scene"] for case in cases}),
    }
    return values


def _utc_timestamp() -> str:
    """Returns the current UTC time as YYYY-MM-DDTHH:MM:SSZ."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _cases_summary(depth_cases: list[dict],
                   reconstruction_cases: list[dict]) -> dict:
    """Returns the case, method and task counts of a visualization payload."""
    all_cases = depth_cases + reconstruction_cases
    return {
        "depth_cases": len(depth_cases),
        "reconstruction_cases": len(reconstruction_cases),
        "methods": len({case["method"] for case in all_cases}),
        "tasks": len({case["task"] for case in all_cases}),
    }


def build_artifact_visualization(
    *,
    benchmark_root: pathlib.Path,
    manifest_path: pathlib.Path,
    frame_limit: int,
) -> dict:
    """Builds the artifact viewer's payload from a Markdown run manifest.

    Args:
        benchmark_root: The benchmark data root.
        manifest_path: The manifest listing the report and artifact roots.
        frame_limit: The most frames a depth case's frames URL lists.
    """
    report_roots, artifact_roots = parse_markdown_table_roots(manifest_path)
    depth_cases = scan_depth_cases(
        report_roots,
        artifact_roots,
        benchmark_root=benchmark_root,
        frame_limit=frame_limit,
    )
    reconstruction_cases = scan_reconstruction_cases(report_roots,
                                                     artifact_roots)
    return {
        "generated_at": _utc_timestamp(),
        "source_manifest": str(manifest_path),
        "benchmark_root": str(benchmark_root),
        "summary": _cases_summary(depth_cases, reconstruction_cases),
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
    """Keeps one case per (kind, task, dataset, method, scene).

    The kept case is the one whose eval report is newest (a missing report
    counts as oldest).

    Returns:
        The kept cases sorted by (task, dataset, method, scene).
    """
    latest: dict[tuple[str, str, str, str, str], tuple[float, dict]] = {}
    for case in cases:
        key = (case["kind"], case["task"], case["dataset"], case["method"],
               case["scene"])
        try:
            mtime = pathlib.Path(case.get("report_path", "")).stat().st_mtime
        except OSError:
            mtime = 0.0
        current = latest.get(key)
        if current is None or mtime > current[0]:
            latest[key] = (mtime, case)
    return [
        case for _mtime, case in sorted(
            latest.values(), key=lambda item: _case_sort_key(item[1]))
    ]


def _case_sort_key(case: dict) -> tuple[str, str, str, str]:
    return (case["task"], case["dataset"], case["method"], case["scene"])


def build_artifact_visualization_from_roots(
    *,
    report_roots: list[pathlib.Path],
    benchmark_root: pathlib.Path,
    frame_limit: int,
    include_cases: set[tuple[str, str, str]] | None = None,
) -> dict:
    """Builds artifact-viewer cases from discovered method report roots.

    Unlike `build_artifact_visualization` it needs no run manifest. Each root
    is scanned for both depth (eval_depth.json) and reconstruction
    (eval_sparse.json) results; the case method is the raw method ID stored in
    the eval payload, so it matches the IDs in results.json. Prediction paths
    come from the eval payloads, so no artifact-root fallback index is built.

    Args:
        report_roots: The method report roots.
        benchmark_root: The benchmark data root.
        frame_limit: The most frames a depth case's frames URL lists.
        include_cases: The (task, dataset, scene) cases to keep, or None for
            all.
    """
    depth_roots = {
        (f"Depth {root}", root.name):
            RunRoot(scope="Depth", method=root.name, root=root)
        for root in map(pathlib.Path, report_roots)
    }
    reconstruction_roots = {
        (f"SfM {root}", root.name):
            RunRoot(scope="SfM", method=root.name, root=root)
        for root in map(pathlib.Path, report_roots)
    }
    depth_cases = deduplicate_cases(
        scan_depth_cases(
            depth_roots,
            {},
            benchmark_root=benchmark_root,
            frame_limit=frame_limit,
            include_cases=include_cases,
        ))
    reconstruction_cases = deduplicate_cases(
        scan_reconstruction_cases(reconstruction_roots, {},
                                  include_cases=include_cases))
    return {
        "generated_at": _utc_timestamp(),
        "source_manifest": None,
        "benchmark_root": str(benchmark_root),
        "summary": _cases_summary(depth_cases, reconstruction_cases),
        "run_roots": {
            "reports": {
                str(root): str(root) for root in report_roots
            },
            "artifacts": {}
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


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        required=True,
        help=("Markdown manifest with report/artifact root tables. "
              "collect_benchmark_metrics.py --with-artifacts builds "
              "artifacts.json by discovery instead."))
    parser.add_argument("--output",
                        type=pathlib.Path,
                        default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--frame-limit",
                        type=int,
                        default=VISUALIZATION_FRAME_LIMIT)
    return parser.parse_args()


def main() -> None:
    """Writes the artifact viewer's payload of a run manifest."""
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
