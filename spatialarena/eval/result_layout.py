# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Utilities for benchmark method output layouts.

A method writes each scene to

    <experiment_root>/<experiment_name>[/<task>]/<dataset>/<method>/<scene>
    [/<order>]

and its COLMAP model to sparse/0 (canonical) or sparse (legacy). The
evaluators resolve these paths here, and refuse to write their outputs into
the read-only benchmark data.
"""

from __future__ import annotations

import os
import pathlib

COLMAP_STEMS = ("cameras", "images", "points3D")
COLMAP_EXTENSIONS = (".bin", ".txt")


def has_colmap_model_files(path: pathlib.Path) -> bool:
    """Returns whether a directory holds COLMAP cameras, images and points."""
    path = pathlib.Path(path)
    if not path.is_dir():
        return False
    return all(
        any((path / f"{stem}{extension}").is_file()
            for extension in COLMAP_EXTENSIONS)
        for stem in COLMAP_STEMS)


def sparse_model_candidates(workspace_path: pathlib.Path) -> list[pathlib.Path]:
    """Returns the model directories of a workspace, sparse/0 first."""
    sparse_path = pathlib.Path(workspace_path) / "sparse"
    return [
        sparse_path / "0",
        sparse_path,
    ]


def resolve_sparse_model_path(
        workspace_path: pathlib.Path) -> pathlib.Path | None:
    """Returns the canonical sparse/0 or legacy sparse model, or None."""
    for candidate in sparse_model_candidates(workspace_path):
        if has_colmap_model_files(candidate):
            return candidate
    return None


def resolve_sparse_model_path_from_any(
        path: pathlib.Path) -> pathlib.Path | None:
    """Returns the COLMAP model of a scene root, sparse root or sparse/0 path.

    Args:
        path: A model directory, a workspace, or a workspace's sparse or
            sparse/<n> directory.

    Returns:
        The model directory, or None if there is none.
    """
    path = pathlib.Path(path)
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
    experiment_root: pathlib.Path,
    experiment_name: str,
    dataset_name: str,
    method_name: str,
    scene_name: str,
    task_name: str | None = None,
    order_name: str | None = None,
) -> pathlib.Path:
    """Returns the canonical output directory of a method on a scene.

    Args:
        experiment_root: Root of all experiments.
        experiment_name: Experiment directory name.
        dataset_name: Dataset name.
        method_name: Method name.
        scene_name: Scene name.
        task_name: Optional task directory (e.g. MonoDepth) above the dataset.
        order_name: Optional image-order directory below the scene.

    Returns:
        <experiment_root>/<experiment_name>[/<task>]/<dataset>/<method>/<scene>
        [/<order>].
    """
    base_path = pathlib.Path(experiment_root) / experiment_name
    if task_name:
        base_path = base_path / task_name
    scene_path = base_path / dataset_name / method_name / scene_name
    if order_name:
        scene_path = scene_path / order_name
    return scene_path


def method_scene_output_candidates(
    experiment_root: pathlib.Path,
    experiment_name: str,
    dataset_name: str,
    method_name: str,
    scene_name: str,
    task_name: str | None = None,
    order_name: str | None = None,
) -> list[pathlib.Path]:
    """Returns the canonical and legacy locations of a method scene output.

    The arguments are those of `method_scene_output_path`. The canonical path
    comes first, then the paths without the task and/or order directories.
    """
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
            ))
    if order_name:
        candidates.append(
            method_scene_output_path(
                experiment_root=experiment_root,
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                method_name=method_name,
                scene_name=scene_name,
                task_name=task_name,
            ))
        if task_name:
            candidates.append(
                method_scene_output_path(
                    experiment_root=experiment_root,
                    experiment_name=experiment_name,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    scene_name=scene_name,
                ))
    return candidates


def resolve_method_scene_output_path(
    experiment_root: pathlib.Path,
    experiment_name: str,
    dataset_name: str,
    method_name: str,
    scene_name: str,
    task_name: str | None = None,
    order_name: str | None = None,
) -> pathlib.Path:
    """Returns the first existing method output path, preferring the task one.

    The arguments are those of `method_scene_output_path`. If no candidate
    exists, returns the canonical path.
    """
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


def path_is_relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
    """Returns whether path is root or below it, after resolving both."""
    resolved_path = pathlib.Path(path).expanduser().resolve(strict=False)
    resolved_root = pathlib.Path(root).expanduser().resolve(strict=False)
    return (resolved_path == resolved_root or
            resolved_root in resolved_path.parents)


def validate_output_path_outside_benchmark_root(
    output_path: pathlib.Path | None,
    benchmark_root: pathlib.Path,
    *,
    label: str = "output_path",
) -> None:
    """Rejects evaluator and report outputs inside the benchmark input data.

    Paths under BENCHMARK_EVAL_ROOT are always allowed.

    Args:
        output_path: Output path to check; None passes.
        benchmark_root: Root of the read-only benchmark data.
        label: Name of the output in the error message.

    Raises:
        ValueError: output_path is inside benchmark_root.
    """
    if output_path is None:
        return
    allowed_eval_root = os.environ.get("BENCHMARK_EVAL_ROOT")
    if allowed_eval_root and path_is_relative_to(
            output_path, pathlib.Path(allowed_eval_root)):
        return
    if path_is_relative_to(output_path, benchmark_root):
        raise ValueError(
            f"{label} must be outside benchmark_root to keep benchmark "
            f"inputs read-only: {output_path} is inside {benchmark_root}")
