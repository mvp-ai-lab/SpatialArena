# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Validation helpers for model-produced benchmark artifacts.

The benchmark suite evaluates files produced by external methods. This module
checks that those files exist in the standardized layout; it does not import,
load, or run any external model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from spatialarena.eval.result_layout import (
    resolve_method_scene_output_path,
    resolve_sparse_model_path,
)


DEPTH_DIR_NAMES = ("depths", "depth", "pred_depths", "depth_maps")
DEPTH_EXTENSIONS = (
    ".npy",
    ".npz",
    ".pfm",
    ".dpt",
    ".exr",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
)
POINT_CLOUD_FILENAMES = (
    "points3D_dense.ply",
    "points3D.ply",
    "point_cloud.ply",
    "points.ply",
)


@dataclass(frozen=True, kw_only=True)
class PredictionRequirements:
    require_sparse: bool = False
    require_depths: bool = False
    require_points: bool = False
    require_metrics: bool = False


@dataclass(kw_only=True)
class PredictionArtifactStatus:
    task: str
    dataset: str
    method: str
    scene: str
    scene_output_path: Path
    scene_output_exists: bool
    sparse_model_path: Path | None = None
    depth_files: list[Path] = field(default_factory=list)
    point_files: list[Path] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "dataset": self.dataset,
            "method": self.method,
            "scene": self.scene,
            "scene_output_path": str(self.scene_output_path),
            "scene_output_exists": self.scene_output_exists,
            "sparse_model_path": (
                str(self.sparse_model_path) if self.sparse_model_path else None
            ),
            "num_depth_files": len(self.depth_files),
            "point_files": [str(path) for path in self.point_files],
            "missing": list(self.missing),
            "warnings": list(self.warnings),
            "ok": self.ok,
        }


def requirements_for_task(task_name: str) -> PredictionRequirements:
    task = task_name.lower()
    if task in {"monodepth", "video_depth", "videodepth", "rmvd", "mvdepth"}:
        return PredictionRequirements(require_depths=True)
    if task in {"sfm", "slam", "camerapose", "camera_pose", "pose"}:
        return PredictionRequirements(require_sparse=True)
    if task in {"mvrecon", "mv_recon", "multi_view_reconstruction"}:
        return PredictionRequirements(require_sparse=True, require_depths=True)
    return PredictionRequirements()


def find_depth_files(scene_output_path: Path) -> list[Path]:
    scene_output_path = Path(scene_output_path)
    depth_files: list[Path] = []
    for dirname in DEPTH_DIR_NAMES:
        depth_dir = scene_output_path / dirname
        if depth_dir.is_dir():
            depth_files.extend(
                path
                for path in depth_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in DEPTH_EXTENSIONS
            )
    return sorted(depth_files)


# Per-frame prediction maps are float32 compressed .npz (adapter_utils.save_prediction_map); predictions exported
# before 2026-11-17 (and imported releases) are .npy, which is still read.
PREDICTION_MAP_SUFFIXES = (".npz", ".npy")


def frame_file(directory: Path, stem: str) -> Path | None:
    """``<directory>/<stem>.npz`` or, for older predictions, ``<stem>.npy``; None when the frame has neither."""
    for suffix in PREDICTION_MAP_SUFFIXES:
        candidate = Path(directory) / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def confidence_file_for(depth_path: Path) -> Path | None:
    """The predicted confidence map of a depth map: ``<scene>/confidences/<depth stem>.npz`` (``.npy`` for older
    predictions; pixel-aligned, written by ``adapter_utils.export_confidence_predictions``), or None when the
    prediction has none."""
    depth_path = Path(depth_path)
    for parent in depth_path.parents:
        if parent.name in DEPTH_DIR_NAMES:
            return frame_file(parent.parent / "confidences", depth_path.stem)
    return None


def load_confidence_map(path: Path):
    """A confidence map as float32 ``H x W``: the "confidence" array of an .npz (else its first array), or an .npy."""
    # pylint: disable-next=import-outside-toplevel
    import numpy as np

    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            array = data["confidence"] if "confidence" in data.files else data[data.files[0]]
    else:
        array = np.load(path)
    array = np.asarray(array, dtype=np.float32)
    return array[..., 0] if array.ndim == 3 else array


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_adapter_runtime(scene_output_path: Path) -> dict | None:
    """Runtime of the inference run that produced a scene's predictions, from the adapter's ``metrics.json``.

    Adapters instrumented with ``adapter_utils.mark_model_loaded`` write ``inference_seconds``,
    ``model_load_seconds`` and ``peak_gpu_memory_mib`` (inference only, model loading excluded). Older outputs only
    carry each adapter's own ``inference time`` / ``Peak GPU Memory``, whose scope differs per adapter; those are
    returned with ``runtime_scope = "legacy"`` so they are never silently mixed with the standard numbers.
    Returns None when there is no readable ``metrics.json``.
    """
    path = Path(scene_output_path) / "metrics.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(payload, list):
        payload = payload[0] if payload and isinstance(payload[0], dict) else {}
    if not isinstance(payload, dict):
        return None
    if "inference_seconds" in payload:
        runtime = {
            "inference_seconds": _finite_or_none(payload.get("inference_seconds")),
            "model_load_seconds": _finite_or_none(payload.get("model_load_seconds")),
            "peak_gpu_memory_mib": _finite_or_none(payload.get("peak_gpu_memory_mib")),
            "runtime_scope": str(payload.get("runtime_scope") or "unknown"),
        }
    else:
        runtime = {
            "inference_seconds": _finite_or_none(payload.get("inference time")),
            "model_load_seconds": None,
            "peak_gpu_memory_mib": _finite_or_none(payload.get("Peak GPU Memory")),
            "runtime_scope": "legacy",
        }
    images = payload.get("Preprocessed Images")
    runtime["num_input_images"] = int(images) if isinstance(images, (int, float)) else None
    return runtime


def find_point_files(scene_output_path: Path) -> list[Path]:
    scene_output_path = Path(scene_output_path)
    point_files = [
        scene_output_path / filename
        for filename in POINT_CLOUD_FILENAMES
        if (scene_output_path / filename).is_file()
    ]
    if not point_files:
        point_files = sorted(scene_output_path.glob("*.ply"))
    return point_files


def validate_prediction_output(
    *,
    task: str,
    dataset: str,
    method: str,
    scene: str,
    scene_output_path: Path,
    requirements: PredictionRequirements | None = None,
) -> PredictionArtifactStatus:
    requirements = requirements or requirements_for_task(task)
    scene_output_path = Path(scene_output_path)
    status = PredictionArtifactStatus(
        task=task,
        dataset=dataset,
        method=method,
        scene=scene,
        scene_output_path=scene_output_path,
        scene_output_exists=scene_output_path.is_dir(),
    )

    if not status.scene_output_exists:
        status.missing.append("scene_output")
        return status

    status.sparse_model_path = resolve_sparse_model_path(scene_output_path)
    status.depth_files = find_depth_files(scene_output_path)
    status.point_files = find_point_files(scene_output_path)

    if requirements.require_sparse and status.sparse_model_path is None:
        status.missing.append("sparse_colmap_model")
    if requirements.require_depths and not status.depth_files:
        status.missing.append("depth_predictions")
    if requirements.require_points and not status.point_files:
        status.missing.append("point_cloud")
    if requirements.require_metrics and not (scene_output_path / "metrics.json").is_file():
        status.missing.append("metrics.json")

    if status.sparse_model_path == scene_output_path / "sparse":
        status.warnings.append("legacy sparse layout; prefer sparse/0")

    return status


def validate_method_scene_output(
    *,
    experiment_root: Path,
    experiment_name: str,
    task: str,
    dataset: str,
    method: str,
    scene: str,
    order_name: str | None = None,
    requirements: PredictionRequirements | None = None,
) -> PredictionArtifactStatus:
    scene_output_path = resolve_method_scene_output_path(
        experiment_root=experiment_root,
        experiment_name=experiment_name,
        dataset_name=dataset,
        method_name=method,
        scene_name=scene,
        task_name=task,
        order_name=order_name,
    )
    return validate_prediction_output(
        task=task,
        dataset=dataset,
        method=method,
        scene=scene,
        scene_output_path=scene_output_path,
        requirements=requirements,
    )
