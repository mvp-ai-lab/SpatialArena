# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Evaluate standardized sparse COLMAP artifacts against sparse GT poses."""

from __future__ import annotations

import argparse
import itertools
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np

from spatialarena.eval.dataset_registry import (
    DEFAULT_BENCHMARK_ROOT,
    canonical_task_name,
    select_dataset_specs,
)
from spatialarena.eval.prediction_artifacts import read_adapter_runtime
from spatialarena.eval.result_layout import (
    resolve_method_scene_output_path,
    resolve_sparse_model_path,
    validate_output_path_outside_benchmark_root,
)


COLMAP_IMAGE_BINARY_HEADER = "<idddddddi"
DEFAULT_AUC_THRESHOLDS = (1.0, 3.0, 5.0, 10.0, 20.0, 30.0)
DEFAULT_RRA_THRESHOLDS = (5.0, 15.0, 30.0)
DEFAULT_RTA_THRESHOLDS = (5.0, 15.0, 30.0)


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x=x))
    return float(np.trapz(y, x=x))


@dataclass(frozen=True, kw_only=True)
class CameraPose:
    name: str
    qvec: np.ndarray
    tvec: np.ndarray

    @property
    def rotation(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.tvec


@dataclass(frozen=True, kw_only=True)
class SparseEvaluationConfig:
    min_baseline: float = 1e-8
    max_pairs: int = 0
    auc_thresholds: tuple[float, ...] = DEFAULT_AUC_THRESHOLDS
    rra_thresholds: tuple[float, ...] = DEFAULT_RRA_THRESHOLDS
    rta_thresholds: tuple[float, ...] = DEFAULT_RTA_THRESHOLDS


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = np.linalg.norm(qvec)
    if norm <= 0:
        raise ValueError("Invalid zero-norm quaternion")
    qw, qx, qy, qz = qvec / norm
    return np.array(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qz * qw,
                2.0 * qx * qz + 2.0 * qy * qw,
            ],
            [
                2.0 * qx * qy + 2.0 * qz * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                2.0 * qy * qz - 2.0 * qx * qw,
            ],
            [
                2.0 * qx * qz - 2.0 * qy * qw,
                2.0 * qy * qz + 2.0 * qx * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation))
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0 or second_norm <= 0:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def vector_axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    angle = vector_angle_deg(first, second)
    return min(angle, abs(180.0 - angle))


def read_null_terminated_string(file) -> str:
    name_bytes = bytearray()
    while True:
        value = file.read(1)
        if not value:
            raise ValueError("Unexpected end of images.bin while reading image name")
        if value == b"\x00":
            break
        name_bytes.extend(value)
    return name_bytes.decode("utf-8")


def read_images_binary(path: Path) -> dict[str, CameraPose]:
    poses: dict[str, CameraPose] = {}
    with Path(path).open("rb") as file:
        num_images = struct.unpack("<Q", file.read(8))[0]
        for _ in range(num_images):
            payload = file.read(struct.calcsize(COLMAP_IMAGE_BINARY_HEADER))
            if len(payload) != struct.calcsize(COLMAP_IMAGE_BINARY_HEADER):
                raise ValueError(f"Invalid COLMAP images.bin record in {path}")
            values = struct.unpack(COLMAP_IMAGE_BINARY_HEADER, payload)
            qvec = np.array(values[1:5], dtype=np.float64)
            tvec = np.array(values[5:8], dtype=np.float64)
            name = read_null_terminated_string(file)
            num_points2d = struct.unpack("<Q", file.read(8))[0]
            file.seek(24 * num_points2d, 1)
            poses[name] = CameraPose(name=name, qvec=qvec, tvec=tvec)
    return poses


def read_images_text(path: Path) -> dict[str, CameraPose]:
    poses: dict[str, CameraPose] = {}
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    index = 0
    while index < len(lines):
        if not lines[index]:
            index += 1
            continue
        parts = lines[index].split(maxsplit=9)
        if len(parts) < 10:
            index += 1
            continue
        name = parts[9]
        qvec = np.array([float(value) for value in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(value) for value in parts[5:8]], dtype=np.float64)
        poses[name] = CameraPose(name=name, qvec=qvec, tvec=tvec)
        index += 2
    return poses


def read_colmap_poses(sparse_model_path: Path) -> dict[str, CameraPose]:
    sparse_model_path = Path(sparse_model_path)
    images_bin = sparse_model_path / "images.bin"
    images_txt = sparse_model_path / "images.txt"
    if images_bin.is_file():
        return read_images_binary(images_bin)
    if images_txt.is_file():
        return read_images_text(images_txt)
    raise FileNotFoundError(f"No COLMAP images.bin/images.txt in {sparse_model_path}")


def relative_transform(first: CameraPose, second: CameraPose) -> tuple[np.ndarray, np.ndarray]:
    first_rotation = first.rotation
    second_rotation = second.rotation
    rotation = second_rotation @ first_rotation.T
    translation = second.tvec - rotation @ first.tvec
    return rotation, translation


def selected_pairs(names: list[str], max_pairs: int) -> list[tuple[str, str]]:
    pairs = list(itertools.combinations(names, 2))
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs
    indices = np.linspace(0, len(pairs) - 1, num=max_pairs, dtype=np.int64)
    return [pairs[int(index)] for index in indices]


def compute_auc(errors: np.ndarray, thresholds: Iterable[float]) -> dict[str, float]:
    errors = np.sort(np.asarray(errors, dtype=np.float64))
    if errors.size == 0:
        return {f"auc_{threshold:g}": 0.0 for threshold in thresholds}
    recall = np.arange(1, errors.size + 1, dtype=np.float64) / errors.size
    aucs: dict[str, float] = {}
    for threshold in thresholds:
        threshold = float(threshold)
        below = errors <= threshold
        if not np.any(below):
            aucs[f"auc_{threshold:g}"] = 0.0
            continue
        x_values = np.concatenate(([0.0], errors[below], [threshold]))
        y_values = np.concatenate(([0.0], recall[below], [recall[below][-1]]))
        aucs[f"auc_{threshold:g}"] = trapezoid_integral(y_values, x_values) / threshold
    return aucs


def compute_maa(
    rotation_errors: np.ndarray,
    translation_errors: np.ndarray,
    max_threshold: int = 30,
) -> float:
    """Compute mAA over max(relative rotation, translation) errors."""
    if rotation_errors.size == 0 or translation_errors.size == 0:
        return 0.0
    error_matrix = np.stack([rotation_errors, translation_errors], axis=1)
    max_errors = np.max(error_matrix, axis=1)
    histogram, _ = np.histogram(max_errors, bins=np.arange(max_threshold + 1))
    normalized_histogram = histogram.astype(np.float64) / float(len(max_errors))
    return float(np.mean(np.cumsum(normalized_histogram)))


def align_points_similarity(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Align source points to target points with a Sim(3) transform."""
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if source_points.shape != target_points.shape or source_points.ndim != 2:
        raise ValueError("Source and target points must have matching NxD shapes")
    if source_points.shape[0] == 0:
        raise ValueError("Need at least one point for trajectory alignment")

    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance <= 1e-12:
        rotation = np.eye(source_points.shape[1], dtype=np.float64)
        scale = 1.0
        translation = target_mean - source_mean
        return source_points + translation, scale, rotation, translation

    covariance = (target_centered.T @ source_centered) / source_points.shape[0]
    left, singular_values, right_t = np.linalg.svd(covariance)
    sign = np.ones(source_points.shape[1], dtype=np.float64)
    if np.linalg.det(left) * np.linalg.det(right_t) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_t
    scale = float(np.sum(singular_values * sign) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source_points.T)).T + translation
    return aligned, scale, rotation, translation


def alignment_transform(estimate, alignment_type: str = "sim3") -> dict:
    """JSON form of a fitted alignment (x -> scale * rotation @ x + translation), as stored in a scene result's
    ``alignment_transform`` so the 3D viewer can reuse it instead of fitting again."""
    return {"type": alignment_type, "scale": float(estimate.scale),
            "rotation": np.asarray(estimate.rotation, dtype=np.float64).tolist(),
            "translation": np.asarray(estimate.translation, dtype=np.float64).tolist()}


def compute_absolute_trajectory_metrics(
    gt_poses: dict[str, CameraPose],
    pred_poses: dict[str, CameraPose],
    matched_names: list[str],
    transform_out: dict | None = None,
) -> dict[str, float]:
    """Align the predicted trajectory to GT with the repo-wide pose-aware Sim(3)
    (``pose_alignment.ransac_irls_align_sim3``: RANSAC over the GTSAM
    pose-residual fit, then a robust refit on all poses) and report ATE.

    A closed-form Umeyama fit on camera centers alone
    (``align_points_similarity``) is degenerate on near-collinear
    trajectories and has no outlier rejection; this fit also uses each
    pose's *rotation* (``CameraPose.rotation``). The simulation study behind
    the choice is in docs/benchmark/slam_protocol.md ("Choice of estimator").
    ``transform_out``, when given, receives the fitted alignment (``alignment_transform``).
    """
    if len(matched_names) < 2:
        raise ValueError("Need at least two matched images for ATE")

    from spatialarena.eval.pose_alignment import ransac_irls_align_sim3

    gt_centers = np.stack([gt_poses[name].center for name in matched_names], axis=0)
    pred_centers = np.stack([pred_poses[name].center for name in matched_names], axis=0)
    gt_rotations = np.stack([gt_poses[name].rotation for name in matched_names], axis=0)
    pred_rotations = np.stack([pred_poses[name].rotation for name in matched_names], axis=0)
    estimate = ransac_irls_align_sim3(pred_rotations, pred_centers, gt_rotations, gt_centers)
    if transform_out is not None:
        transform_out.update(alignment_transform(estimate))
    aligned_pred_centers = estimate.transform_points(pred_centers)
    errors = np.linalg.norm(aligned_pred_centers - gt_centers, axis=1)
    return {
        "ate_rmse": float(np.sqrt(np.mean(errors**2))),
        "ate_mean": float(np.mean(errors)),
        "ate_median": float(np.median(errors)),
        "ate_std": float(np.std(errors)),
    }


def match_pose_names(
    gt_poses: dict[str, CameraPose],
    pred_poses: dict[str, CameraPose],
) -> tuple[dict[str, CameraPose], dict[str, CameraPose]]:
    """GT and predicted poses keyed by the names they share.

    Exact COLMAP image names when they overlap; otherwise file names without their folders, when those are unique on
    both sides (ETH3D's GT names images ``dslr_images_undistorted/DSC_0286.JPG``, the adapters ``DSC_0286.JPG``).
    """
    if set(gt_poses) & set(pred_poses):
        return gt_poses, pred_poses

    def by_basename(poses: dict[str, CameraPose]) -> dict[str, CameraPose] | None:
        keyed = {PurePosixPath(name).name: pose for name, pose in poses.items()}
        return keyed if len(keyed) == len(poses) else None

    gt_keyed, pred_keyed = by_basename(gt_poses), by_basename(pred_poses)
    if gt_keyed is None or pred_keyed is None:
        return gt_poses, pred_poses
    return gt_keyed, pred_keyed


def compute_relative_pose_metrics(
    gt_poses: dict[str, CameraPose],
    pred_poses: dict[str, CameraPose],
    config: SparseEvaluationConfig | None = None,
    transform_out: dict | None = None,
) -> dict:
    config = config or SparseEvaluationConfig()
    matched_names = sorted(set(gt_poses) & set(pred_poses))
    if len(matched_names) < 2:
        raise ValueError("Need at least two matched images for relative pose metrics")

    rotation_errors = []
    translation_direction_errors = []
    translation_axis_errors = []
    for first_name, second_name in selected_pairs(matched_names, config.max_pairs):
        gt_rotation, gt_translation = relative_transform(
            gt_poses[first_name],
            gt_poses[second_name],
        )
        pred_rotation, pred_translation = relative_transform(
            pred_poses[first_name],
            pred_poses[second_name],
        )
        rotation_errors.append(rotation_angle_deg(pred_rotation.T @ gt_rotation))
        if np.linalg.norm(gt_translation) < config.min_baseline:
            translation_direction_errors.append(0.0)
            translation_axis_errors.append(0.0)
        else:
            direction_error = vector_angle_deg(pred_translation, gt_translation)
            translation_direction_errors.append(direction_error)
            translation_axis_errors.append(
                vector_axis_angle_deg(pred_translation, gt_translation)
            )

    rotation_errors_array = np.asarray(rotation_errors, dtype=np.float64)
    translation_errors_array = np.asarray(translation_direction_errors, dtype=np.float64)
    translation_axis_errors_array = np.asarray(translation_axis_errors, dtype=np.float64)
    relative_errors = np.maximum(rotation_errors_array, translation_errors_array)
    axis_relative_errors = np.maximum(rotation_errors_array, translation_axis_errors_array)
    rotation_mean = float(np.mean(rotation_errors_array))
    rotation_median = float(np.median(rotation_errors_array))
    translation_mean = float(np.mean(translation_errors_array))
    translation_median = float(np.median(translation_errors_array))
    translation_axis_mean = float(np.mean(translation_axis_errors_array))
    translation_axis_median = float(np.median(translation_axis_errors_array))
    threshold_metrics: dict[str, float] = {}
    for threshold in config.rra_thresholds:
        threshold_metrics[f"RRA_at_{threshold:g}"] = float(
            np.mean(rotation_errors_array < float(threshold))
        )
    for threshold in config.rta_thresholds:
        threshold_metrics[f"RTA_at_{threshold:g}"] = float(
            np.mean(translation_axis_errors_array < float(threshold))
        )
    return {
        "num_pairs": int(relative_errors.size),
        **compute_absolute_trajectory_metrics(gt_poses, pred_poses, matched_names, transform_out),
        "relative_rotation_error_deg_mean": rotation_mean,
        "relative_rotation_error_deg_median": rotation_median,
        "relative_translation_error_deg_mean": translation_mean,
        "relative_translation_error_deg_median": translation_median,
        "relative_translation_axis_error_deg_mean": translation_axis_mean,
        "relative_translation_axis_error_deg_median": translation_axis_median,
        "rotation_error_deg_mean": rotation_mean,
        "rotation_error_deg_median": rotation_median,
        "translation_direction_error_deg_mean": translation_mean,
        "translation_direction_error_deg_median": translation_median,
        "relative_pose_error_deg_mean": float(np.mean(relative_errors)),
        "relative_pose_error_deg_median": float(np.median(relative_errors)),
        "relative_pose_axis_error_deg_mean": float(np.mean(axis_relative_errors)),
        "relative_pose_axis_error_deg_median": float(np.median(axis_relative_errors)),
        **compute_auc(relative_errors, config.auc_thresholds),
        **threshold_metrics,
        "mAA_30": compute_maa(
            rotation_errors_array,
            translation_axis_errors_array,
            max_threshold=30,
        ),
    }


def evaluate_sparse_scene(
    scene_info,
    prediction_scene_path: Path,
    config: SparseEvaluationConfig | None = None,
) -> dict:
    if scene_info.sparse_gt_path is None:
        raise ValueError(f"Scene has no canonical sparse GT: {scene_info.scene}")

    prediction_sparse_path = resolve_sparse_model_path(prediction_scene_path)
    if prediction_sparse_path is None:
        raise ValueError(f"No sparse COLMAP prediction in {prediction_scene_path}")

    gt_poses, pred_poses = match_pose_names(read_colmap_poses(scene_info.sparse_gt_path),
                                            read_colmap_poses(prediction_sparse_path))
    transform: dict = {}
    metrics = compute_relative_pose_metrics(gt_poses, pred_poses, config=config, transform_out=transform)
    matched_names = sorted(set(gt_poses) & set(pred_poses))
    return {
        "task": scene_info.task,
        "dataset": scene_info.dataset,
        "scene": scene_info.scene,
        "prediction_scene_path": str(prediction_scene_path),
        "prediction_sparse_path": str(prediction_sparse_path),
        "sparse_gt_path": str(scene_info.sparse_gt_path),
        "num_gt_images": len(gt_poses),
        "num_pred_images": len(pred_poses),
        "num_matched_images": len(matched_names),
        "metrics": metrics,
        "alignment_transform": transform,
        "runtime": read_adapter_runtime(prediction_scene_path),
    }


def aggregate_sparse_results(scene_results: list[dict]) -> dict[str, float]:
    metric_sums: dict[str, float] = {}
    metric_weights: dict[str, int] = {}
    total_pairs = 0
    for result in scene_results:
        metrics = result["metrics"]
        pair_weight = int(metrics["num_pairs"])
        pose_weight = int(result.get("num_matched_images", 0))
        for name, value in metrics.items():
            if name == "num_pairs":
                continue
            weight = pose_weight if name.startswith("ate_") else pair_weight
            if weight <= 0:
                continue
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value) * weight
            metric_weights[name] = metric_weights.get(name, 0) + weight
        total_pairs += pair_weight
    if total_pairs == 0:
        return {}
    return {
        name: value / metric_weights[name]
        for name, value in metric_sums.items()
        if metric_weights.get(name, 0) > 0
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standardized sparse COLMAP prediction artifacts."
    )
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--experiment_name", type=str, default="exp")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="*", default=["SfM", "WeakTextureSfM", "MVRecon"])
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--order_name", default="")
    parser.add_argument("--include_scannet", action="store_true")
    parser.add_argument("--min_baseline", type=float, default=1e-8)
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Maximum image pairs to evaluate per scene. Use 0 for all pairs.",
    )
    parser.add_argument("--auc_thresholds", nargs="+", type=float, default=list(DEFAULT_AUC_THRESHOLDS))
    parser.add_argument("--rra_thresholds", nargs="+", type=float, default=list(DEFAULT_RRA_THRESHOLDS))
    parser.add_argument("--rta_thresholds", nargs="+", type=float, default=list(DEFAULT_RTA_THRESHOLDS))
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--viewer_dir", type=Path, default=None,
                        help="Write each SfM case's aligned trajectory here for the dashboard viewer (viewer_assets).")
    parser.add_argument("--viewer_scene_root", type=Path, default=None,
                        help="Root of the viewer assets every method of a scene shares (frame thumbnails).")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def run_sparse_evaluation(
    *,
    benchmark_root: Path,
    experiment_root: Path,
    experiment_name: str,
    methods: Iterable[str],
    tasks: Iterable[str],
    datasets: Iterable[str],
    scenes: Iterable[str],
    include_scannet: bool,
    config: SparseEvaluationConfig,
    order_name: str | None = None,
    viewer_dir: Path | None = None,
    viewer_scene_root: Path | None = None,
) -> dict:
    from spatialarena.eval.viewer_assets import add_case_trajectory  # pylint: disable=import-outside-toplevel

    tasks = list(tasks)
    if any(canonical_task_name(task) == "SLAM" for task in tasks):
        raise ValueError(
            "SLAM is evaluated by spatialarena.eval.slam_trajectory_evaluator (docs/benchmark/slam_protocol.md)"
        )
    scene_filter = list(scenes)
    methods = list(methods)
    datasets = list(datasets)
    # One case (one method, scene and dataset: the engine's per-case call) writes straight into viewer_dir; a
    # scene name can repeat across datasets.
    flat_viewer = len(methods) == 1 and len(scene_filter) == 1 and len(datasets) == 1 and len(tasks) == 1
    scene_results: list[dict] = []
    errors: list[dict] = []

    for spec in select_dataset_specs(
        tasks=tasks,
        datasets=datasets,
        include_scannet=include_scannet,
    ):
        if spec.metadata_required and not (spec.data_path(benchmark_root) / "metadata.json").is_file():
            errors.append(
                {
                    "task": spec.task,
                    "dataset": spec.dataset,
                    "error": "metadata.json is missing",
                }
            )
            continue

        dataset = spec.create_dataset(benchmark_root, scenes=scene_filter)
        for scene_info in dataset.list_scenes():
            for method in methods:
                prediction_scene_path = resolve_method_scene_output_path(
                    experiment_root=experiment_root,
                    experiment_name=experiment_name,
                    dataset_name=scene_info.dataset,
                    method_name=method,
                    scene_name=scene_info.scene,
                    task_name=scene_info.task,
                    order_name=order_name,
                )
                try:
                    result = evaluate_sparse_scene(
                        scene_info,
                        prediction_scene_path,
                        config=config,
                    ) | {"method": method, "order": order_name or ""}
                    if canonical_task_name(scene_info.task) in {"SfM", "WeakTextureSfM"}:
                        result = add_case_trajectory(viewer_dir, flat_viewer, method, scene_info, result,
                                                     viewer_scene_root)
                    scene_results.append(result)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    errors.append(
                        {
                            "task": scene_info.task,
                            "dataset": scene_info.dataset,
                            "scene": scene_info.scene,
                            "method": method,
                            "order": order_name or "",
                            "prediction_scene_path": str(prediction_scene_path),
                            "error": str(exc),
                        }
                    )

    return {
        "num_scene_results": len(scene_results),
        "num_errors": len(errors),
        "metrics": aggregate_sparse_results(scene_results),
        "scene_results": scene_results,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_output_path_outside_benchmark_root(
        args.output_path,
        args.benchmark_root,
        label="--output_path",
    )
    validate_output_path_outside_benchmark_root(args.viewer_dir, args.benchmark_root, label="--viewer_dir")
    validate_output_path_outside_benchmark_root(args.viewer_scene_root, args.benchmark_root,
                                                label="--viewer_scene_root")
    result = run_sparse_evaluation(
        benchmark_root=args.benchmark_root,
        experiment_root=args.experiment_root,
        experiment_name=args.experiment_name,
        methods=args.methods,
        tasks=args.tasks,
        datasets=args.datasets,
        scenes=args.scenes,
        order_name=args.order_name or None,
        viewer_dir=args.viewer_dir,
        viewer_scene_root=args.viewer_scene_root,
        include_scannet=args.include_scannet,
        config=SparseEvaluationConfig(
            min_baseline=args.min_baseline,
            max_pairs=args.max_pairs,
            auc_thresholds=tuple(args.auc_thresholds),
            rra_thresholds=tuple(args.rra_thresholds),
            rta_thresholds=tuple(args.rta_thresholds),
        ),
    )

    text = json.dumps(result, indent=2)
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if args.strict and result["num_errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
