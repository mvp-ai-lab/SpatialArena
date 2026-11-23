# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Aligned camera trajectories for the dashboard's trajectory comparison.

Each trajectory is the ground truth and the prediction mapped into the GT
frame by exactly the alignment the evaluator scores, so what the viewer draws
is what the ATE measures. When the evaluator's fit is passed in (`transform`:
the scene result's `alignment_transform`) it is used as is; otherwise (an
older result, or a manual call) the viewer fits the same alignment itself --
same frames, same estimator, same fixed seed:

- SLAM (`slam_trajectory_evaluator`): the range's input frames (every
  `stride`-th GT frame), Sim(3) or SE(3) via `fit_alignment` with the run's
  `alignment_estimator`; frames without a predicted pose are kept as gaps.
- SfM / Weak-Texture SfM
  (`sparse_reconstruction_evaluator.compute_absolute_trajectory_metrics`):
  the images both models share, `ransac_irls_align_sim3`.

The payload is plain JSON (positions in metres of the GT frame,
camera-to-world rotations as wxyz quaternions, per-frame ATE), rounded so a
city-scale VBR trajectory stays a few MB.
"""

from __future__ import annotations

import pathlib

import numpy as np

POSITION_DECIMALS = 4
QUATERNION_DECIMALS = 4


def rotation_to_quaternion_wxyz(rotations: np.ndarray) -> np.ndarray:
    """Converts [N, 3, 3] rotations to [N, 4] unit quaternions (w, x, y, z).

    The quaternions have w >= 0.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    quaternions = np.empty((len(rotations), 4))
    for index, matrix in enumerate(rotations):
        trace = np.trace(matrix)
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            quaternions[index] = [
                0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s
            ]
        else:
            axis = int(np.argmax(np.diag(matrix)))
            i, j, k = axis, (axis + 1) % 3, (axis + 2) % 3
            s = 2.0 * np.sqrt(
                max(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k], 1e-12))
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
    """Returns the array rounded to decimals, as nested lists."""
    return np.round(np.asarray(array, dtype=np.float64), decimals).tolist()


def trajectory_payload(
    names: list[str],
    gt_c2w: np.ndarray,
    aligned_c2w: np.ndarray,
    valid: np.ndarray,
    *,
    alignment: dict,
) -> dict:
    """Returns the JSON-ready trajectory: GT and aligned prediction per frame.

    Args:
        names: Frame names.
        gt_c2w: [N, 4, 4] GT camera-to-world matrices.
        aligned_c2w: [N, 4, 4] aligned predicted camera-to-world matrices.
        valid: [N] whether the method returned a pose for the frame; the
            prediction and ATE of the other frames are None.
        alignment: Description of the alignment, stored as is.

    Returns:
        The payload: names, gt and pred positions and quaternions, per-frame
        ate, ate_rmse, num_frames, num_valid and alignment.
    """
    valid = np.asarray(valid, dtype=bool)
    ate = np.linalg.norm(aligned_c2w[:, :3, 3] - gt_c2w[:, :3, 3], axis=1)
    pred_positions = _rounded(aligned_c2w[:, :3, 3], POSITION_DECIMALS)
    pred_quaternions = _rounded(
        rotation_to_quaternion_wxyz(aligned_c2w[:, :3, :3]),
        QUATERNION_DECIMALS)
    ate_values = _rounded(ate, POSITION_DECIMALS)
    for index in np.flatnonzero(~valid):
        pred_positions[index] = None
        pred_quaternions[index] = None
        ate_values[index] = None
    if valid.any():
        ate_rmse = float(np.sqrt(np.mean(ate[valid]**2)))
    else:
        ate_rmse = None
    return {
        "names": list(names),
        "gt": {
            "positions":
                _rounded(gt_c2w[:, :3, 3], POSITION_DECIMALS),
            "quaternions":
                _rounded(rotation_to_quaternion_wxyz(gt_c2w[:, :3, :3]),
                         QUATERNION_DECIMALS),
        },
        "pred": {
            "positions": pred_positions,
            "quaternions": pred_quaternions
        },
        "ate": ate_values,
        "ate_rmse": ate_rmse,
        "num_frames": len(names),
        "num_valid": int(valid.sum()),
        "alignment": alignment,
    }


def slam_trajectory(
    gt_sparse_path: pathlib.Path,
    pred_sparse_path: pathlib.Path,
    *,
    stride: int,
    alignment: str = "sim3",
    alignment_estimator: str = "ransac_gtsam_irls",
    transform: dict | None = None,
) -> dict:
    """Returns the SLAM evaluator's aligned trajectory over the input frames.

    Uses the evaluator's `transform` when given, else fits it here, as
    `evaluate_slam_scene` does. The evaluator only scores scenes whose images
    and GT poses match one to one, so striding the GT names picks the frames
    the method was given.

    Args:
        gt_sparse_path: GT COLMAP model.
        pred_sparse_path: Predicted COLMAP model.
        stride: Input-frame stride of the range.
        alignment: "sim3" or "se3".
        alignment_estimator: Estimator of `fit_alignment`.
        transform: The scene result's alignment_transform, if any.

    Returns:
        The `trajectory_payload`.

    Raises:
        ValueError: Fewer than three input frames have a predicted pose.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import slam_trajectory_evaluator
    from spatialarena.eval import sparse_reconstruction_evaluator

    gt_poses = sparse_reconstruction_evaluator.read_colmap_poses(gt_sparse_path)
    pred_poses = sparse_reconstruction_evaluator.read_colmap_poses(
        pred_sparse_path)
    names = slam_trajectory_evaluator.expected_input_names(
        list(gt_poses), stride)
    valid = np.array([name in pred_poses for name in names])
    if valid.sum() < 3:
        raise ValueError(f"Only {int(valid.sum())} of {len(names)} input "
                         "frames have a predicted pose")
    gt_c2w = slam_trajectory_evaluator.camera_to_world(gt_poses, names)
    pred_c2w = np.tile(np.eye(4), (len(names), 1, 1))
    pred_c2w[valid] = slam_trajectory_evaluator.camera_to_world(
        pred_poses, [name for name in names if name in pred_poses])
    if transform:
        scale, rotation, translation = _transform_parts(transform)
        source = "evaluator"
    else:
        estimate = slam_trajectory_evaluator.fit_alignment(
            gt_c2w[valid],
            pred_c2w[valid],
            with_scale=alignment == "sim3",
            estimator=alignment_estimator)
        scale = estimate.scale
        rotation = estimate.rotation
        translation = estimate.translation
        source = "viewer_fit"
    aligned = slam_trajectory_evaluator.apply_alignment(pred_c2w, scale,
                                                        rotation, translation)
    return trajectory_payload(
        names,
        gt_c2w,
        aligned,
        valid,
        alignment={
            "type": alignment,
            "estimator": alignment_estimator,
            "scale": float(scale),
            "stride": stride,
            "source": source
        },
    )


def _transform_parts(transform: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns the scale, rotation and translation of an alignment_transform."""
    return (float(transform["scale"]),
            np.asarray(transform["rotation"], dtype=np.float64),
            np.asarray(transform["translation"], dtype=np.float64))


def _camera_to_world(rotations_w2c: np.ndarray,
                     centers: np.ndarray) -> np.ndarray:
    """Returns [N, 4, 4] camera-to-world matrices of w2c rotations, centers."""
    matrices = np.tile(np.eye(4), (len(centers), 1, 1))
    matrices[:, :3, :3] = np.transpose(rotations_w2c, (0, 2, 1))
    matrices[:, :3, 3] = centers
    return matrices


def sfm_trajectory(gt_sparse_path: pathlib.Path,
                   pred_sparse_path: pathlib.Path,
                   transform: dict | None = None) -> dict:
    """Returns the SfM evaluator's aligned trajectory over the shared images.

    Uses the evaluator's `transform` when given, else fits it here, with the
    same matching and estimator as
    `sparse_reconstruction_evaluator.compute_absolute_trajectory_metrics`.

    Args:
        gt_sparse_path: GT COLMAP model.
        pred_sparse_path: Predicted COLMAP model.
        transform: The scene result's alignment_transform, if any.

    Returns:
        The `trajectory_payload`.

    Raises:
        ValueError: Fewer than two images are matched.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import pose_alignment
    from spatialarena.eval import sparse_reconstruction_evaluator

    gt_poses, pred_poses = sparse_reconstruction_evaluator.match_pose_names(
        sparse_reconstruction_evaluator.read_colmap_poses(gt_sparse_path),
        sparse_reconstruction_evaluator.read_colmap_poses(pred_sparse_path))
    names = sorted(set(gt_poses) & set(pred_poses))
    if len(names) < 2:
        raise ValueError(
            "Need at least two matched images for an aligned trajectory")
    gt_rotations = np.stack([gt_poses[name].rotation for name in names])
    gt_centers = np.stack([gt_poses[name].center for name in names])
    pred_rotations = np.stack([pred_poses[name].rotation for name in names])
    pred_centers = np.stack([pred_poses[name].center for name in names])
    if transform:
        scale, rotation, translation = _transform_parts(transform)
        estimate = pose_alignment.Sim3Estimate(scale=scale,
                                               rotation=rotation,
                                               translation=translation,
                                               inliers=np.ones(len(names),
                                                               dtype=bool))
        source = "evaluator"
    else:
        estimate = pose_alignment.ransac_irls_align_sim3(
            pred_rotations, pred_centers, gt_rotations, gt_centers)
        source = "viewer_fit"
    aligned_rotations_w2c, aligned_centers = estimate.transform_poses(
        pred_rotations, pred_centers)
    return trajectory_payload(
        names,
        _camera_to_world(gt_rotations, gt_centers),
        _camera_to_world(aligned_rotations_w2c, aligned_centers),
        np.ones(len(names), dtype=bool),
        alignment={
            "type": "sim3",
            "estimator": "ransac_gtsam_irls",
            "scale": float(estimate.scale),
            "source": source
        },
    )
