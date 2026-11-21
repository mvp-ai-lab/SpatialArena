# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Aligned camera trajectories for the dashboard's trajectory comparison.

Each trajectory is the ground truth and the prediction mapped into the GT frame by exactly the alignment the evaluator
scores, so what the viewer draws is what the ATE measures. When the evaluator's fit is passed in (``transform``: the
scene result's ``alignment_transform``) it is used as is; otherwise (an older result, or a manual call) the viewer fits
the same alignment itself -- same frames, same estimator, same fixed seed:

- SLAM (``slam_trajectory_evaluator``): the range's input frames (every ``stride``-th GT frame), Sim(3) or SE(3) via
  ``fit_alignment`` with the run's ``alignment_estimator``; frames without a predicted pose are kept as gaps.
- SfM / Weak-Texture SfM (``sparse_reconstruction_evaluator.compute_absolute_trajectory_metrics``): the images both
  models share, ``ransac_irls_align_sim3``.

The payload is plain JSON (positions in metres of the GT frame, camera-to-world rotations as wxyz quaternions,
per-frame ATE), rounded so a city-scale VBR trajectory stays a few MB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

POSITION_DECIMALS = 4
QUATERNION_DECIMALS = 4


def rotation_to_quaternion_wxyz(rotations: np.ndarray) -> np.ndarray:
    """[N, 3, 3] rotation matrices to [N, 4] unit quaternions (w, x, y, z), w >= 0."""
    rotations = np.asarray(rotations, dtype=np.float64)
    quaternions = np.empty((len(rotations), 4))
    for index, matrix in enumerate(rotations):
        trace = np.trace(matrix)
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            quaternions[index] = [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s,
                                  (matrix[1, 0] - matrix[0, 1]) / s]
        else:
            axis = int(np.argmax(np.diag(matrix)))
            i, j, k = axis, (axis + 1) % 3, (axis + 2) % 3
            s = 2.0 * np.sqrt(max(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k], 1e-12))
            quaternion = np.empty(4)
            quaternion[0] = (matrix[k, j] - matrix[j, k]) / s
            quaternion[1 + i] = 0.25 * s
            quaternion[1 + j] = (matrix[j, i] + matrix[i, j]) / s
            quaternion[1 + k] = (matrix[k, i] + matrix[i, k]) / s
            quaternions[index] = quaternion
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions[quaternions[:, 0] < 0] *= -1.0
    return quaternions


def _rounded(array: np.ndarray, decimals: int) -> list:
    return np.round(np.asarray(array, dtype=np.float64), decimals).tolist()


def trajectory_payload(
    names: list[str],
    gt_c2w: np.ndarray,
    aligned_c2w: np.ndarray,
    valid: np.ndarray,
    *,
    alignment: dict,
) -> dict:
    """JSON-ready trajectory: GT and aligned prediction per frame (``None`` where the method returned no pose)."""
    valid = np.asarray(valid, dtype=bool)
    ate = np.linalg.norm(aligned_c2w[:, :3, 3] - gt_c2w[:, :3, 3], axis=1)
    pred_positions = _rounded(aligned_c2w[:, :3, 3], POSITION_DECIMALS)
    pred_quaternions = _rounded(rotation_to_quaternion_wxyz(aligned_c2w[:, :3, :3]), QUATERNION_DECIMALS)
    ate_values = _rounded(ate, POSITION_DECIMALS)
    for index in np.flatnonzero(~valid):
        pred_positions[index] = pred_quaternions[index] = ate_values[index] = None
    return {
        "names": list(names),
        "gt": {
            "positions": _rounded(gt_c2w[:, :3, 3], POSITION_DECIMALS),
            "quaternions": _rounded(rotation_to_quaternion_wxyz(gt_c2w[:, :3, :3]), QUATERNION_DECIMALS),
        },
        "pred": {"positions": pred_positions, "quaternions": pred_quaternions},
        "ate": ate_values,
        "ate_rmse": float(np.sqrt(np.mean(ate[valid] ** 2))) if valid.any() else None,
        "num_frames": len(names),
        "num_valid": int(valid.sum()),
        "alignment": alignment,
    }


def slam_trajectory(
    gt_sparse_path: Path,
    pred_sparse_path: Path,
    *,
    stride: int,
    alignment: str = "sim3",
    alignment_estimator: str = "ransac_gtsam_irls",
    transform: dict | None = None,
) -> dict:
    """The SLAM evaluator's aligned trajectory over the range's input frames (``evaluate_slam_scene``): its
    ``transform`` when given, else fitted here. The evaluator only scores scenes whose images and GT poses match one
    to one, so striding the GT names picks the frames the method was given."""
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval.slam_trajectory_evaluator import (
        apply_alignment,
        camera_to_world,
        expected_input_names,
        fit_alignment,
    )
    from spatialarena.eval.sparse_reconstruction_evaluator import read_colmap_poses

    gt_poses = read_colmap_poses(gt_sparse_path)
    pred_poses = read_colmap_poses(pred_sparse_path)
    names = expected_input_names(list(gt_poses), stride)
    valid = np.array([name in pred_poses for name in names])
    if valid.sum() < 3:
        raise ValueError(f"Only {int(valid.sum())} of {len(names)} input frames have a predicted pose")
    gt_c2w = camera_to_world(gt_poses, names)
    pred_c2w = np.tile(np.eye(4), (len(names), 1, 1))
    pred_c2w[valid] = camera_to_world(pred_poses, [name for name in names if name in pred_poses])
    if transform:
        scale, rotation, translation = _transform_parts(transform)
        source = "evaluator"
    else:
        estimate = fit_alignment(gt_c2w[valid], pred_c2w[valid], with_scale=alignment == "sim3",
                                 estimator=alignment_estimator)
        scale, rotation, translation = estimate.scale, estimate.rotation, estimate.translation
        source = "viewer_fit"
    aligned = apply_alignment(pred_c2w, scale, rotation, translation)
    return trajectory_payload(
        names, gt_c2w, aligned, valid,
        alignment={"type": alignment, "estimator": alignment_estimator, "scale": float(scale), "stride": stride,
                   "source": source},
    )


def _transform_parts(transform: dict) -> tuple[float, np.ndarray, np.ndarray]:
    return (float(transform["scale"]), np.asarray(transform["rotation"], dtype=np.float64),
            np.asarray(transform["translation"], dtype=np.float64))


def sfm_trajectory(gt_sparse_path: Path, pred_sparse_path: Path, transform: dict | None = None) -> dict:
    """The SfM evaluator's aligned trajectory over the images both models share: its ``transform`` when given, else
    fitted here.

    Same matching and estimator as ``sparse_reconstruction_evaluator.compute_absolute_trajectory_metrics``.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval.pose_alignment import Sim3Estimate, ransac_irls_align_sim3
    from spatialarena.eval.sparse_reconstruction_evaluator import match_pose_names, read_colmap_poses

    gt_poses, pred_poses = match_pose_names(read_colmap_poses(gt_sparse_path), read_colmap_poses(pred_sparse_path))
    names = sorted(set(gt_poses) & set(pred_poses))
    if len(names) < 2:
        raise ValueError("Need at least two matched images for an aligned trajectory")
    gt_rotations = np.stack([gt_poses[name].rotation for name in names])
    gt_centers = np.stack([gt_poses[name].center for name in names])
    pred_rotations = np.stack([pred_poses[name].rotation for name in names])
    pred_centers = np.stack([pred_poses[name].center for name in names])
    if transform:
        scale, rotation, translation = _transform_parts(transform)
        estimate = Sim3Estimate(scale=scale, rotation=rotation, translation=translation,
                                inliers=np.ones(len(names), dtype=bool))
        source = "evaluator"
    else:
        estimate = ransac_irls_align_sim3(pred_rotations, pred_centers, gt_rotations, gt_centers)
        source = "viewer_fit"
    aligned_rotations_w2c, aligned_centers = estimate.transform_poses(pred_rotations, pred_centers)

    def c2w(rotations_w2c: np.ndarray, centers: np.ndarray) -> np.ndarray:
        matrices = np.tile(np.eye(4), (len(centers), 1, 1))
        matrices[:, :3, :3] = np.transpose(rotations_w2c, (0, 2, 1))
        matrices[:, :3, 3] = centers
        return matrices

    return trajectory_payload(
        names, c2w(gt_rotations, gt_centers), c2w(aligned_rotations_w2c, aligned_centers),
        np.ones(len(names), dtype=bool),
        alignment={"type": "sim3", "estimator": "ransac_gtsam_irls", "scale": float(estimate.scale), "source": source},
    )
