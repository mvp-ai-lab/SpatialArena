# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Trajectory evaluation for the SLAM track (docs/benchmark/slam_protocol.md).

A SLAM case is one continuous recording fed in temporal order at a fixed
stride. The predicted camera poses (`sparse/0` of the method's output) are
compared with the ground-truth trajectory after a Sim(3) alignment
(`slam_protocol.TRACK_ALIGNMENT`; Sim(3) for both rgb and rgb_depth) estimated
with `pose_alignment.ransac_irls_align_sim3` (`--alignment_estimator
ransac_gtsam_irls`, the default): LMedS RANSAC over the pose-aware GTSAM
`TrajectoryAlignerSim3` cost (camera rotations and centers, so it stays
well-conditioned on the near-static and near-collinear trajectories several
TUM sequences have; minimal sample 2, 25 iterations), then a Cauchy-weighted
IRLS refit of that cost on all frames, which makes the result independent of
the RANSAC seed. `ransac_dagsfm` keeps the previous estimator (RANSAC over the
DAGSfM fit, refit on the inliers). The choice is backed by a 3,450-trial
simulation (scripts/test/compare_trajectory_alignment.py; results in
docs/benchmark/slam_protocol.md, "Trajectory alignment"). ATE and RPE are
computed over *every* returned frame, inlier or not. The ATE after a
fixed-scale (SE(3)) alignment from the same harness and the Sim(3) scale are
reported alongside, showing whether a method recovered metric scale.

Metrics (all on camera centers / camera-to-world poses):

- `ate_rmse` / `ate_mean` / `ate_median` (m): absolute trajectory error after
  alignment.
- `rpe_trans_step` (m) / `rpe_rot_step` (deg): relative pose error between
  consecutive input frames.
- `rpe_trans_1s` (m) / `rpe_rot_1s` (deg): relative pose error over one second
  of the recording.
- `drift_pct`: sum of 1-second translational RPE divided by the ground-truth
  path length over the same windows (ratio of sums, so near-static segments
  cannot blow it up), in percent.
- `auc_1` ... `auc_30`: the SfM track's pose AUC
  (sparse_reconstruction_evaluator) over frame pairs: the area under the
  accuracy curve of max(relative rotation error, relative translation
  direction error) in degrees, up to each threshold, normalized to [0, 1].
  Both errors are invariant to a similarity transform, so no alignment
  enters. Pairs are all pairs of the returned frames, or of `auc_max_frames`
  of them evenly spaced in time on longer sequences (a 18,846-frame VBR
  sequence would have 1.8e8 pairs).
- `completeness`: fraction of the input frames with a returned pose.
  ATE/RPE/AUC use the returned frames only, so a method must also be judged
  by completeness; ranking (`ate_rmse` plus `drift_pct`) is done by the
  report.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import glob
import json
import math
import pathlib
import sys
from typing import Iterable

import numpy as np

from spatialarena import slam_protocol
from spatialarena.eval import dataset_registry
from spatialarena.eval import prediction_artifacts
from spatialarena.eval import result_layout
from spatialarena.eval import sparse_reconstruction_evaluator as sfm_evaluator


@dataclasses.dataclass(frozen=True, kw_only=True)
class SlamEvaluationConfig:
    """Settings of a SLAM trajectory evaluation.

    Attributes:
        stride: Temporal stride the method was run with.
        alignment: "sim3" (rgb) or "se3" (rgb_depth).
        slam_range: Range track label ("short" / "long" / "large"), recorded
            with the results.
        alignment_estimator: One of ALIGNMENT_ESTIMATORS.
        auc_thresholds: Pose AUC thresholds in degrees.
        auc_max_frames: Pose AUC over all pairs of at most this many returned
            frames (~5e5 pairs).
    """
    stride: int = 1
    alignment: str = "sim3"
    slam_range: str = ""
    alignment_estimator: str = "ransac_gtsam_irls"
    auc_thresholds: tuple[float, ...] = sfm_evaluator.DEFAULT_AUC_THRESHOLDS
    auc_max_frames: int = 1000


def camera_to_world(poses: dict, names: list[str]) -> np.ndarray:
    """Returns [N, 4, 4] camera-to-world matrices for `names`.

    Args:
        poses: COLMAP world-to-camera poses by image name.
        names: Image names, in output order.
    """
    matrices = np.tile(np.eye(4), (len(names), 1, 1))
    for index, name in enumerate(names):
        pose = poses[name]
        rotation = pose.rotation
        matrices[index, :3, :3] = rotation.T
        matrices[index, :3, 3] = -rotation.T @ pose.tvec
    return matrices


# Same RANSAC iteration count as the SfM evaluator's ransac_icp_align_sim3 call.
RANSAC_ITERATIONS = 25
# "ransac_gtsam_irls": pose_alignment.ransac_irls_align_sim3 (default);
# "ransac_dagsfm": the previous estimator.
ALIGNMENT_ESTIMATORS = ("ransac_gtsam_irls", "ransac_dagsfm")


def fit_alignment(gt: np.ndarray,
                  pred: np.ndarray,
                  *,
                  with_scale: bool,
                  estimator: str = "ransac_gtsam_irls"):
    """Fits the trajectory alignment mapping `pred` onto `gt`.

    Args:
        gt: [N, 4, 4] GT camera-to-world poses.
        pred: [N, 4, 4] predicted camera-to-world poses.
        with_scale: False fixes the scale to 1 (SE(3)).
        estimator: One of `ALIGNMENT_ESTIMATORS`: `ransac_gtsam_irls` =
            `pose_alignment.ransac_irls_align_sim3` (RANSAC over the GTSAM
            pose-residual fit, then a robust refit on all frames);
            `ransac_dagsfm` = `pose_alignment.ransac_align_sim3` over
            `fit_sim3_dagsfm`.

    Returns:
        The `Sim3Estimate` (scale, rotation, translation, inlier mask).

    Raises:
        ValueError: On an unknown estimator, fewer than 3 poses or (with
            scale) predicted camera centers that do not move.
    """
    from spatialarena.eval import pose_alignment

    if estimator not in ALIGNMENT_ESTIMATORS:
        raise ValueError(f"Unknown alignment estimator {estimator!r}; "
                         f"expected one of {ALIGNMENT_ESTIMATORS}")

    if len(gt) < 3:
        raise ValueError(
            f"Trajectory alignment needs >= 3 matched poses, got {len(gt)}")
    if with_scale and np.ptp(pred[:, :3, 3], axis=0).max() <= 1e-9:
        raise ValueError(
            "Degenerate trajectory: predicted camera centers do not move")
    # pose_alignment works on world-to-camera rotations and camera centers.
    if estimator == "ransac_gtsam_irls":
        return pose_alignment.ransac_irls_align_sim3(
            np.swapaxes(pred[:, :3, :3], 1, 2),
            pred[:, :3, 3],
            np.swapaxes(gt[:, :3, :3], 1, 2),
            gt[:, :3, 3],
            scale=None if with_scale else 1.0,
            num_iterations=RANSAC_ITERATIONS,
        )
    if with_scale:
        fit_fn = pose_alignment.fit_sim3_dagsfm
    else:
        fit_fn = functools.partial(pose_alignment.fit_sim3_dagsfm, scale=1.0)
    return pose_alignment.ransac_align_sim3(
        np.swapaxes(pred[:, :3, :3], 1, 2),
        pred[:, :3, 3],
        np.swapaxes(gt[:, :3, :3], 1, 2),
        gt[:, :3, 3],
        fit_fn=fit_fn,
        sample_size=2,
        num_iterations=RANSAC_ITERATIONS,
    )


def apply_alignment(c2w: np.ndarray, scale: float, rotation: np.ndarray,
                    translation: np.ndarray) -> np.ndarray:
    """Applies a Sim(3) to [N, 4, 4] camera-to-world poses."""
    aligned = c2w.copy()
    aligned[:, :3, :3] = rotation @ c2w[:, :3, :3]
    aligned[:, :3, 3] = scale * (c2w[:, :3, 3] @ rotation.T) + translation
    return aligned


def rotation_angle_deg(rotation: np.ndarray) -> np.ndarray:
    """Returns the rotation angle in degrees of each [..., 3, 3] rotation."""
    cos = (np.trace(rotation, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def relative_pose_errors(
        gt: np.ndarray, pred: np.ndarray, valid: np.ndarray,
        delta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the RPE over the input-frame pairs (i, i + delta).

    Args:
        gt: [N, 4, 4] GT camera-to-world poses of the input frames.
        pred: [N, 4, 4] aligned predicted camera-to-world poses.
        valid: Marks the frames with a predicted pose.
        delta: Offset between the paired frames, in input frames.

    Only pairs where both frames have a predicted pose count. `E_i = (G_i^-1
    G_{i+d})^-1 (P_i^-1 P_{i+d})` on aligned camera-to-world poses (TUM / evo
    convention). Pairing by input index keeps the offset a fixed time even
    when a method (e.g. keyframe-based SLAM) skips frames.

    Returns:
        (translational RPE in m, rotational RPE in deg, GT path length between
        the two frames over every input frame, returned or not).
    """
    empty = np.empty((0,))
    if delta < 1 or len(gt) <= delta:
        return empty, empty, empty
    first = np.flatnonzero(valid[:-delta] & valid[delta:])
    if first.size == 0:
        return empty, empty, empty
    second = first + delta
    gt_rel = np.linalg.inv(gt[first]) @ gt[second]
    pred_rel = np.linalg.inv(pred[first]) @ pred[second]
    error = np.linalg.inv(gt_rel) @ pred_rel
    trans = np.linalg.norm(error[:, :3, 3], axis=1)
    rot = rotation_angle_deg(error[:, :3, :3])
    steps = np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    path = cumulative[second] - cumulative[first]
    return trans, rot, path


def pairwise_pose_errors(gt: np.ndarray,
                         pred: np.ndarray,
                         min_baseline: float = 1e-8) -> np.ndarray:
    """Returns the pose error in degrees of every pair of camera-to-world poses.

    The error is max(relative rotation error, relative translation direction
    error), as in the SfM evaluator's compute_relative_pose_metrics. A pair
    whose GT baseline is below `min_baseline`, or whose predicted one is zero,
    has no translation direction error.
    """
    first, second = np.triu_indices(len(gt), k=1)

    def relative(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Camera `first` to camera `second`: R = R2^T R1 and
        # t = R2^T (c1 - c2) from camera-to-world poses.
        rotation_t = np.swapaxes(c2w[second, :3, :3], 1, 2)
        rotation = rotation_t @ c2w[first, :3, :3]
        translation = np.einsum("nij,nj->ni", rotation_t,
                                c2w[first, :3, 3] - c2w[second, :3, 3])
        return rotation, translation

    gt_rotation, gt_translation = relative(gt)
    pred_rotation, pred_translation = relative(pred)
    rotation_error = rotation_angle_deg(
        np.swapaxes(pred_rotation, 1, 2) @ gt_rotation)
    gt_norm = np.linalg.norm(gt_translation, axis=1)
    pred_norm = np.linalg.norm(pred_translation, axis=1)
    defined = (gt_norm >= min_baseline) & (pred_norm > 0)
    cosine = np.einsum("ni,ni->n", gt_translation, pred_translation) / np.where(
        defined, gt_norm * pred_norm, 1.0)
    translation_error = np.where(
        defined, np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), 0.0)
    return np.maximum(rotation_error, translation_error)


def pose_auc_metrics(gt: np.ndarray, pred: np.ndarray,
                     thresholds: Iterable[float], max_frames: int) -> dict:
    """Computes `auc_<threshold>` over the frame pairs of `gt` / `pred`.

    Both are [N, 4, 4] camera-to-world poses of the returned frames in temporal
    order; at most `max_frames` of them, evenly spaced in time, are used.

    Args:
        gt: GT camera-to-world poses.
        pred: Predicted camera-to-world poses.
        thresholds: AUC thresholds in degrees.
        max_frames: Frame cap; 0 or less uses every frame.
    """
    frames = np.arange(len(gt))
    if max_frames > 0 and len(frames) > max_frames:
        frames = np.unique(
            np.round(np.linspace(0,
                                 len(gt) - 1, max_frames)).astype(int))
    errors = pairwise_pose_errors(gt[frames], pred[frames])
    metrics = dict(sfm_evaluator.compute_auc(errors, thresholds))
    metrics["auc_num_frames"] = float(len(frames))
    metrics["auc_num_pairs"] = float(errors.size)
    return metrics


def _rms(values: np.ndarray) -> float:
    """Returns the root mean square of `values`, or NaN if it is empty."""
    if not values.size:
        return math.nan
    return float(np.sqrt(np.mean(values**2)))


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_rmse": math.nan,
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan
        }
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
    }


def trajectory_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    frames_per_second: float,
    alignment: str,
    valid: np.ndarray | None = None,
    alignment_estimator: str = "ransac_gtsam_irls",
    auc_thresholds: Iterable[float] = sfm_evaluator.DEFAULT_AUC_THRESHOLDS,
    auc_max_frames: int = SlamEvaluationConfig.auc_max_frames,
    transform_out: dict | None = None,
) -> dict:
    """Computes ATE / RPE / drift / pose AUC of `pred` against `gt`.

    Args:
        gt: [N, 4, 4] GT camera-to-world poses of the N input frames in
            temporal order.
        pred: [N, 4, 4] predicted camera-to-world poses of the same frames.
        frames_per_second: Rate of the *input* (native fps / stride), used for
            the 1-second RPE offset.
        alignment: "sim3" or "se3".
        valid: Marks the frames with a predicted pose (default: all; `pred`
            rows elsewhere are ignored).
        alignment_estimator: One of ALIGNMENT_ESTIMATORS.
        auc_thresholds: Pose AUC thresholds in degrees.
        auc_max_frames: Frame cap for the pose AUC pairs.
        transform_out: When given, receives the `alignment` fit
            (`alignment_transform`).

    Returns:
        The metrics dict.

    Raises:
        ValueError: On an unknown alignment, or if the alignment fails.
    """
    if alignment not in {"sim3", "se3"}:
        raise ValueError(f"Unknown alignment {alignment!r}")
    if valid is None:
        valid = np.ones(len(gt), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    pred = np.where(valid[:, None, None], pred, np.eye(4))
    metrics: dict[str, float] = {}
    aligned_by = {}
    inlier_ratio = {}
    scale_collapsed = {}
    estimates = {}
    for name, with_scale in (("sim3", True), ("se3", False)):
        estimate = fit_alignment(gt[valid],
                                 pred[valid],
                                 with_scale=with_scale,
                                 estimator=alignment_estimator)
        estimates[name] = estimate
        aligned_by[name] = (estimate.scale,
                            apply_alignment(pred, estimate.scale,
                                            estimate.rotation,
                                            estimate.translation))
        inlier_ratio[name] = float(np.mean(estimate.inliers))
        scale_collapsed[name] = bool(getattr(estimate, "scale_collapsed",
                                             False))
    scale, aligned = aligned_by[alignment]
    if transform_out is not None:
        transform_out.update(
            sfm_evaluator.alignment_transform(estimates[alignment], alignment))
    metrics["alignment_inlier_ratio"] = inlier_ratio[alignment]
    # 1.0 when the robust refit collapsed the scale and the RANSAC estimate was
    # used (pose_alignment.irls_refine_sim3): a sign that the predicted
    # trajectory does not match the ground truth at all.
    metrics["alignment_scale_collapsed"] = float(scale_collapsed[alignment])
    if alignment == "sim3":
        other = "se3"
    else:
        other = "sim3"
    ate = np.linalg.norm(aligned[valid, :3, 3] - gt[valid, :3, 3], axis=1)
    metrics.update(_stats(ate, "ate"))
    metrics["alignment_scale"] = scale
    other_ate = np.linalg.norm(aligned_by[other][1][valid, :3, 3] -
                               gt[valid, :3, 3],
                               axis=1)
    metrics[f"ate_rmse_{other}"] = float(np.sqrt(np.mean(other_ate**2)))

    trans, rot, _path = relative_pose_errors(gt, aligned, valid, 1)
    metrics["rpe_trans_step"] = _rms(trans)
    metrics["rpe_rot_step"] = _rms(rot)
    delta_1s = max(1, int(round(frames_per_second)))
    trans, rot, path = relative_pose_errors(gt, aligned, valid, delta_1s)
    metrics["rpe_delta_1s_frames"] = float(delta_1s)
    metrics["rpe_trans_1s"] = _rms(trans)
    metrics["rpe_rot_1s"] = _rms(rot)
    if path.size:
        total_path = float(path.sum())
    else:
        total_path = 0.0
    metrics["gt_path_length_1s_windows"] = total_path
    if total_path > 1e-9:
        metrics["drift_pct"] = float(100.0 * trans.sum() / total_path)
    else:
        metrics["drift_pct"] = math.nan
    metrics.update(
        pose_auc_metrics(gt[valid], pred[valid], auc_thresholds,
                         auc_max_frames))
    return metrics


def expected_input_names(gt_names: list[str], stride: int) -> list[str]:
    """Returns the frames a method was given.

    That is every `stride`-th frame in temporal (name) order, as
    select_interval picks them.
    """
    return sorted(gt_names)[::max(1, stride)]


def scene_image_names(scene_root: pathlib.Path) -> list[str]:
    """Returns the scene's image files as the adapters list them.

    The adapters use `find_image_paths` on the canonical image root; the files
    are returned as paths relative to that root, the names COLMAP models use.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import colmap_view_resolver

    root = colmap_view_resolver.canonical_image_root(scene_root)
    paths = sorted(
        path for pattern in colmap_view_resolver.IMAGE_PATTERNS
        for path in glob.glob(str(root / "**" / pattern), recursive=True))
    return [pathlib.Path(path).relative_to(root).as_posix() for path in paths]


def matched_input_names(image_names: list[str], gt_names: Iterable[str],
                        stride: int) -> list[str]:
    """Returns the input frames to score.

    That is the adapters' selection over the scene's images, which must match
    the GT poses one to one. A scene where an image has no GT pose (or a GT
    pose no image) is not scored: striding the two lists would pick different
    frames, so the poses compared would not be the frames the method was
    given.

    Raises:
        ValueError: If the images and GT poses do not match one to one.
    """
    gt_set = set(gt_names)
    without_pose = [name for name in image_names if name not in gt_set]
    without_image = sorted(gt_set - set(image_names))
    if without_pose or without_image:
        raise ValueError(
            "SLAM images and GT poses must match one to one: "
            f"{len(without_pose)} image(s) without a GT pose "
            f"{without_pose[:3]}, {len(without_image)} GT pose(s) without an "
            f"image {without_image[:3]}")
    return expected_input_names(image_names, stride)


def evaluate_slam_scene(scene_info, prediction_scene_path: pathlib.Path, *,
                        config: SlamEvaluationConfig) -> dict:
    """Evaluates one method's predicted trajectory of one SLAM scene.

    Args:
        scene_info: The benchmark scene.
        prediction_scene_path: The method's output folder of the scene.
        config: Evaluation settings.

    Returns:
        The scene result, with the metrics and the alignment transform.

    Raises:
        ValueError: If the GT or predicted model is missing or too few input
            frames have a predicted pose.
    """
    if scene_info.sparse_gt_path is None:
        raise ValueError("No ground-truth sparse model for "
                         f"{scene_info.dataset}/{scene_info.scene}")
    gt_poses = sfm_evaluator.read_colmap_poses(scene_info.sparse_gt_path)
    prediction_sparse_path = result_layout.resolve_sparse_model_path(
        prediction_scene_path)
    if prediction_sparse_path is None:
        raise ValueError(
            f"No sparse COLMAP prediction in {prediction_scene_path}")
    pred_poses = sfm_evaluator.read_colmap_poses(prediction_sparse_path)

    input_names = matched_input_names(scene_image_names(scene_info.root_path),
                                      gt_poses, config.stride)
    valid = np.array([name in pred_poses for name in input_names])
    matched = [name for name in input_names if name in pred_poses]
    if len(matched) < 3:
        raise ValueError(f"Only {len(matched)} of {len(input_names)} input "
                         "frames have a predicted pose")
    fps = slam_protocol.dataset_fps(scene_info.dataset) / max(1, config.stride)
    pred_c2w = np.tile(np.eye(4), (len(input_names), 1, 1))
    pred_c2w[valid] = camera_to_world(pred_poses, matched)
    transform: dict = {}
    metrics = trajectory_metrics(
        camera_to_world(gt_poses, input_names),
        pred_c2w,
        frames_per_second=fps,
        alignment=config.alignment,
        valid=valid,
        alignment_estimator=config.alignment_estimator,
        auc_thresholds=config.auc_thresholds,
        auc_max_frames=config.auc_max_frames,
        transform_out=transform,
    )
    metrics["completeness"] = len(matched) / len(input_names)
    runtime = prediction_artifacts.read_adapter_runtime(prediction_scene_path)
    return {
        "task": scene_info.task,
        "dataset": scene_info.dataset,
        "scene": scene_info.scene,
        "prediction_scene_path": str(prediction_scene_path),
        "prediction_sparse_path": str(prediction_sparse_path),
        "sparse_gt_path": str(scene_info.sparse_gt_path),
        "stride": config.stride,
        "slam_range": config.slam_range,
        "alignment": config.alignment,
        "alignment_estimator": config.alignment_estimator,
        "input_frames_per_second": fps,
        "num_gt_images": len(gt_poses),
        "num_input_frames": len(input_names),
        "num_pred_images": len(pred_poses),
        "num_matched_images": len(matched),
        "metrics": metrics,
        "alignment_transform": transform,
        "runtime": runtime,
    }


def aggregate_slam_results(scene_results: list[dict]) -> dict[str, float]:
    """Averages the scene metrics, unweighted over sequences.

    Each recording counts once, as in TUM/7-Scenes SLAM tables.
    """
    auc_keys = set()
    for result in scene_results:
        for key in result["metrics"]:
            if key.startswith("auc_") and key not in ("auc_num_frames",
                                                      "auc_num_pairs"):
                auc_keys.add(key)
    keys = (
        "ate_rmse",
        "rpe_trans_step",
        "rpe_rot_step",
        "rpe_trans_1s",
        "rpe_rot_1s",
        "drift_pct",
        *sorted(auc_keys, key=lambda key: float(key[4:])),
        "completeness",
    )
    aggregate: dict[str, float] = {"num_sequences": float(len(scene_results))}
    for key in keys:
        values = [
            result["metrics"][key]
            for result in scene_results
            if math.isfinite(result["metrics"].get(key, math.nan))
        ]
        if values:
            aggregate[key] = float(np.mean(values))
        else:
            aggregate[key] = math.nan
    return aggregate


def _prediction_scene_path(experiment_root: pathlib.Path, experiment_name: str,
                           method: str, scene_info) -> pathlib.Path:
    """Returns the method's output folder of a scene."""
    return result_layout.resolve_method_scene_output_path(
        experiment_root=experiment_root,
        experiment_name=experiment_name,
        dataset_name=scene_info.dataset,
        method_name=method,
        scene_name=scene_info.scene,
        task_name=scene_info.task,
    )


def run_slam_evaluation(
    *,
    benchmark_root: pathlib.Path,
    experiment_root: pathlib.Path,
    experiment_name: str,
    methods: Iterable[str],
    datasets: Iterable[str],
    scenes: Iterable[str],
    config: SlamEvaluationConfig,
    viewer_dir: pathlib.Path | None = None,
    viewer_scene_root: pathlib.Path | None = None,
) -> dict:
    """Evaluates every SLAM case of the methods, datasets and scenes.

    Args:
        benchmark_root: Benchmark root folder.
        experiment_root: Root of the method outputs.
        experiment_name: Experiment name under `experiment_root`.
        methods: Methods to evaluate.
        datasets: SLAM datasets; empty selects all.
        scenes: Scene filter; empty selects all.
        config: Evaluation settings.
        viewer_dir: Where to write each case's aligned trajectory for the
            dashboard viewer.
        viewer_scene_root: Root of the viewer assets every method of a scene
            shares.

    Returns:
        The protocol, aggregate metrics, scene results and errors.
    """
    from spatialarena.eval import viewer_assets  # pylint: disable=import-outside-toplevel

    scene_filter = list(scenes)
    methods = list(methods)
    datasets = list(datasets)
    # One case (one method, scene and dataset: the engine's per-case call)
    # writes straight into viewer_dir; a scene name can repeat across datasets.
    flat_viewer = (len(methods) == 1 and len(scene_filter) == 1 and
                   len(datasets) == 1)
    scene_results: list[dict] = []
    errors: list[dict] = []
    for spec in dataset_registry.select_dataset_specs(tasks=["SLAM"],
                                                      datasets=datasets,
                                                      include_scannet=False):
        dataset = spec.create_dataset(benchmark_root, scenes=scene_filter)
        for scene_info in dataset.list_scenes():
            for method in methods:
                prediction_scene_path = _prediction_scene_path(
                    experiment_root, experiment_name, method, scene_info)
                try:
                    result = evaluate_slam_scene(scene_info,
                                                 prediction_scene_path,
                                                 config=config)
                    result = result | {"method": method}
                    scene_results.append(
                        viewer_assets.add_case_trajectory(
                            viewer_dir, flat_viewer, method, scene_info, result,
                            viewer_scene_root))
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    errors.append({
                        "task": scene_info.task,
                        "dataset": scene_info.dataset,
                        "scene": scene_info.scene,
                        "method": method,
                        "prediction_scene_path": str(prediction_scene_path),
                        "error": str(exc),
                    })
    return {
        "protocol": {
            "slam_range": config.slam_range,
            "stride": config.stride,
            "alignment": config.alignment,
            "alignment_estimator": config.alignment_estimator,
            "auc_thresholds": list(config.auc_thresholds),
            "auc_max_frames": config.auc_max_frames,
        },
        "num_scene_results": len(scene_results),
        "num_errors": len(errors),
        "metrics": aggregate_slam_results(scene_results),
        "scene_results": scene_results,
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line (`argv`, default sys.argv)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--experiment_root", type=pathlib.Path, required=True)
    parser.add_argument("--experiment_name", type=str, default="exp")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--stride",
                        type=int,
                        required=True,
                        help="Temporal stride the method was run with.")
    parser.add_argument("--slam_range",
                        default="",
                        help="Range track label recorded with the results.")
    parser.add_argument(
        "--alignment",
        choices=("sim3", "se3"),
        default="sim3",
        help=("sim3 for monocular input (rgb), se3 when metric depth was given "
              "(rgb_depth)."))
    parser.add_argument(
        "--alignment_estimator",
        choices=ALIGNMENT_ESTIMATORS,
        default="ransac_gtsam_irls",
        help=(
            "Trajectory alignment estimator (default: RANSAC + GTSAM pose fit "
            "+ robust refit)."))
    parser.add_argument("--auc_thresholds",
                        nargs="+",
                        type=float,
                        default=list(sfm_evaluator.DEFAULT_AUC_THRESHOLDS))
    parser.add_argument(
        "--auc_max_frames",
        type=int,
        default=SlamEvaluationConfig.auc_max_frames,
        help=("Pose AUC over all pairs of at most this many evenly spaced "
              "returned frames (0: all)."))
    parser.add_argument("--output_path", type=pathlib.Path, default=None)
    parser.add_argument(
        "--viewer_dir",
        type=pathlib.Path,
        default=None,
        help=("Write each case's aligned trajectory here for the dashboard "
              "viewer (viewer_assets)."))
    parser.add_argument(
        "--viewer_scene_root",
        type=pathlib.Path,
        default=None,
        help=("Root of the viewer assets every method of a scene shares (frame "
              "thumbnails)."))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _json_safe(value):
    """Replaces non-finite floats by None, recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> None:
    """Runs the SLAM evaluation CLI and writes or prints the JSON result."""
    args = parse_args(argv)
    result_layout.validate_output_path_outside_benchmark_root(
        args.output_path, args.benchmark_root, label="--output_path")
    result_layout.validate_output_path_outside_benchmark_root(
        args.viewer_dir, args.benchmark_root, label="--viewer_dir")
    result_layout.validate_output_path_outside_benchmark_root(
        args.viewer_scene_root,
        args.benchmark_root,
        label="--viewer_scene_root")
    result = run_slam_evaluation(
        benchmark_root=args.benchmark_root,
        experiment_root=args.experiment_root,
        experiment_name=args.experiment_name,
        methods=args.methods,
        datasets=args.datasets,
        scenes=args.scenes,
        viewer_dir=args.viewer_dir,
        viewer_scene_root=args.viewer_scene_root,
        config=SlamEvaluationConfig(
            stride=args.stride,
            alignment=args.alignment,
            slam_range=args.slam_range,
            alignment_estimator=args.alignment_estimator,
            auc_thresholds=tuple(args.auc_thresholds),
            auc_max_frames=args.auc_max_frames,
        ),
    )
    text = json.dumps(_json_safe(result), indent=2)
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.strict and result["num_errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
