# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Utilities for benchmark method output layouts."""

from __future__ import annotations

import os
from pathlib import Path


COLMAP_STEMS = ("cameras", "images", "points3D")
COLMAP_EXTENSIONS = (".bin", ".txt")


def has_colmap_model_files(path: Path) -> bool:
    """Return True when a directory contains a minimal COLMAP model."""
    path = Path(path)
    if not path.is_dir():
        return False
    return all(
        any((path / f"{stem}{extension}").is_file() for extension in COLMAP_EXTENSIONS)
        for stem in COLMAP_STEMS
    )


def sparse_model_candidates(workspace_path: Path) -> list[Path]:
    sparse_path = Path(workspace_path) / "sparse"
    return [
        sparse_path / "0",
        sparse_path,
    ]


def resolve_sparse_model_path(workspace_path: Path) -> Path | None:
    """Resolve either canonical sparse/0 or legacy sparse method output."""
    for candidate in sparse_model_candidates(workspace_path):
        if has_colmap_model_files(candidate):
            return candidate
    return None


def resolve_sparse_model_path_from_any(path: Path) -> Path | None:
    """Resolve COLMAP model path from a scene root, sparse root, or sparse/0 path."""
    path = Path(path)
    if has_colmap_model_files(path):
        return path

    resolved = resolve_sparse_model_path(path)
    if resolved is not None:
        return resolved

    if path.name == "sparse":
        return resolve_sparse_model_path(path.parent)
    if path.parent.name == "sparse":
        return resolve_sparse_model_path(path.parent.parent)
    return None


def method_scene_output_path(
    experiment_root: Path,
    experiment_name: str,
    dataset_name: str,
    method_name: str,
    scene_name: str,
    task_name: str | None = None,
    order_name: str | None = None,
) -> Path:
    base_path = Path(experiment_root) / experiment_name
    if task_name:
        base_path = base_path / task_name
    scene_path = base_path / dataset_name / method_name / scene_name
    if order_name:
        scene_path = scene_path / order_name
    return scene_path


def method_scene_output_candidates(
    experiment_root: Path,
    experiment_name: str,
    dataset_name: str,
    method_name: str,
    scene_name: str,
    task_name: str | None = None,
    order_name: str | None = None,
) -> list[Path]:
    """Return canonical and legacy locations for a method scene output."""
    candidates = [
        method_scene_output_path(
            experiment_root=experiment_root,
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            method_name=method_name,
            scene_name=scene_name,
            task_name=task_name,
            order_name=order_name,
        )
    ]
    if task_name:
        candidates.append(
            method_scene_output_path(
                experiment_root=experiment_root,
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                method_name=method_name,
                scene_name=scene_name,
            )
        )
    if order_name:
        candidates.append(
            method_scene_output_path(
                experiment_root=experiment_root,
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                method_name=method_name,
                scene_name=scene_name,
                task_name=task_name,
            )
        )
        if task_name:
            candidates.append(
                method_scene_output_path(
                    experiment_root=experiment_root,
                    experiment_name=experiment_name,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    scene_name=scene_name,
                )
            )
    return candidates


def resolve_method_scene_output_path(
    experiment_root: Path,
    experiment_name: str,
    dataset_name: str,
    method_name: str,
    scene_name: str,
    task_name: str | None = None,
    order_name: str | None = None,
) -> Path:
    """Resolve the first existing method output path, preferring task namespacing."""
    candidates = method_scene_output_candidates(
        experiment_root=experiment_root,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        method_name=method_name,
        scene_name=scene_name,
        task_name=task_name,
        order_name=order_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def path_is_relative_to(path: Path, root: Path) -> bool:
    """Return True when path is equal to or nested below root after resolution."""
    resolved_path = Path(path).expanduser().resolve(strict=False)
    resolved_root = Path(root).expanduser().resolve(strict=False)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def validate_output_path_outside_benchmark_root(
    output_path: Path | None,
    benchmark_root: Path,
    *,
    label: str = "output_path",
) -> None:
    """Reject evaluator/report outputs that would write into benchmark input data."""
    if output_path is None:
        return
    allowed_eval_root = os.environ.get("BENCHMARK_EVAL_ROOT")
    if allowed_eval_root and path_is_relative_to(output_path, Path(allowed_eval_root)):
        return
    if path_is_relative_to(output_path, benchmark_root):
        raise ValueError(
            f"{label} must be outside benchmark_root to keep benchmark inputs read-only: "
            f"{output_path} is inside {benchmark_root}"
        )
