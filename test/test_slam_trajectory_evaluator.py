# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Synthetic checks for the SLAM trajectory evaluator (known trajectories, known transforms, known drift).

aligned_trajectory: Aligned trajectories for the dashboard's trajectory comparison
(spatialarena.eval.aligned_trajectory).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import numpy as np
import pytest
from spatialarena.eval.slam_trajectory_evaluator import (
    SlamEvaluationConfig,
    aggregate_slam_results,
    evaluate_slam_scene,
    expected_input_names,
    trajectory_metrics,
    fit_alignment,
    pose_auc_metrics,
)
from spatialarena.eval.sparse_reconstruction_evaluator import (
    CameraPose,
    DEFAULT_AUC_THRESHOLDS,
    compute_relative_pose_metrics,
)
from spatialarena.geometry.read_write_model import BaseImage, Camera, rotmat2qvec, write_model
from pathlib import Path
from spatialarena.eval.aligned_trajectory import (
    rotation_to_quaternion_wxyz,
    sfm_trajectory,
    slam_trajectory,
)

# ---- slam_trajectory_evaluator -----------------------------------------------------------------------------


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def circle_trajectory(n: int = 60) -> np.ndarray:
    """Camera-to-world poses on a 1 m radius circle, yawing with the motion and bobbing in z."""
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        angle = 2 * math.pi * i / n
        poses[i, :3, :3] = rot_z(angle)
        poses[i, :3, 3] = [
            math.cos(angle),
            math.sin(angle), 0.1 * math.sin(3 * angle)
        ]
    return poses


def similarity(poses: np.ndarray, scale: float, rotation: np.ndarray,
               translation: np.ndarray) -> np.ndarray:
    out = poses.copy()
    out[:, :3, :3] = rotation @ poses[:, :3, :3]
    out[:, :3, 3] = scale * poses[:, :3, 3] @ rotation.T + translation
    return out


def test_fit_recovers_known_similarity_including_on_a_collinear_trajectory():
    rotation = rot_z(0.7) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                                     dtype=float)
    translation = np.array([1.0, -2.0, 3.0])
    for gt in (circle_trajectory(), _line_trajectory()):
        # pred = S^-1(gt): the fit must recover S = (2.5, rotation, translation).
        pred = similarity(gt, 1 / 2.5, rotation.T,
                          -(rotation.T @ translation) / 2.5)
        estimate = fit_alignment(gt, pred, with_scale=True)
        scale, est_rotation, est_translation = estimate.scale, estimate.rotation, estimate.translation
        assert scale == pytest.approx(2.5)
        # On the line, a camera-center-only (Umeyama) fit cannot determine the rotation about the line's axis; the
        # pose-aware fit recovers it from the camera orientations.
        np.testing.assert_allclose(est_rotation, rotation, atol=1e-6)
        np.testing.assert_allclose(est_translation, translation, atol=1e-6)


def _line_trajectory(n: int = 30) -> np.ndarray:
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        poses[i, :3, :3] = rot_z(0.05 *
                                 i)  # turning in place while moving straight
        poses[i, :3, 3] = [0.1 * i, 0.0, 0.0]
    return poses


def test_similarity_transformed_prediction_has_zero_error():
    gt = circle_trajectory()
    # A prediction in its own arbitrary frame and scale -- pred = S^-1 * gt, so Sim(3) alignment must undo S.
    pred = similarity(gt, 0.25, rot_z(-1.1), np.array([5.0, 0.0, -2.0]))
    metrics = trajectory_metrics(gt,
                                 pred,
                                 frames_per_second=6.0,
                                 alignment="sim3")
    for key in ("ate_rmse", "rpe_trans_step", "rpe_rot_step", "rpe_trans_1s",
                "rpe_rot_1s", "drift_pct"):
        assert metrics[key] == pytest.approx(0.0, abs=1e-6), key
    assert metrics["alignment_scale"] == pytest.approx(4.0)
    # SE(3) cannot undo the 4x scale error, so its ATE stays large: the gap exposes scale errors.
    assert metrics["ate_rmse_se3"] > 0.5


def test_se3_alignment_reports_scale_error_for_metric_tracks():
    gt = circle_trajectory()
    pred = similarity(gt, 1.0, rot_z(0.3), np.array([0.0, 1.0, 0.0]))
    exact = trajectory_metrics(gt, pred, frames_per_second=6.0, alignment="se3")
    assert exact["ate_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert exact["alignment_scale"] == 1.0
    scaled = trajectory_metrics(gt,
                                similarity(gt, 1.2, np.eye(3), np.zeros(3)),
                                frames_per_second=6.0,
                                alignment="se3")
    assert scaled["ate_rmse"] > 0.05
    assert scaled["ate_rmse_sim3"] == pytest.approx(0.0, abs=1e-9)


def test_constant_drift_gives_expected_rpe_and_drift_pct():
    # Straight line, 0.1 m per frame along x; the prediction overshoots every step by 1 cm (10 % drift).
    n = 40
    gt = np.tile(np.eye(4), (n, 1, 1))
    gt[:, 0, 3] = 0.1 * np.arange(n)
    pred = gt.copy()
    pred[:, 0, 3] = 0.11 * np.arange(n)
    metrics = trajectory_metrics(gt,
                                 pred,
                                 frames_per_second=5.0,
                                 alignment="se3")
    assert metrics["rpe_trans_step"] == pytest.approx(0.01)
    assert metrics["rpe_delta_1s_frames"] == 5.0
    assert metrics["rpe_trans_1s"] == pytest.approx(0.05)
    assert metrics["drift_pct"] == pytest.approx(10.0)
    assert metrics["rpe_rot_step"] == pytest.approx(0.0, abs=1e-9)
    # Sim(3) absorbs a uniform overshoot entirely: drift is a scale error here.
    drift = trajectory_metrics(gt,
                               pred,
                               frames_per_second=5.0,
                               alignment="sim3")["drift_pct"]
    assert drift == pytest.approx(0.0, abs=1e-6)


def _write_poses(path, names, c2w):
    path.mkdir(parents=True, exist_ok=True)
    cameras = {
        1:
            Camera(id=1,
                   model="PINHOLE",
                   width=64,
                   height=48,
                   params=np.array([50.0, 50.0, 32.0, 24.0]))
    }
    images = {}
    for index, (name, pose) in enumerate(zip(names, c2w), start=1):
        w2c = np.linalg.inv(pose)
        images[index] = BaseImage(id=index,
                                  qvec=rotmat2qvec(w2c[:3, :3]),
                                  tvec=w2c[:3, 3],
                                  camera_id=1,
                                  name=name,
                                  xys=np.empty((0, 2)),
                                  point3D_ids=np.empty((0,), dtype=np.int64))
    write_model(cameras, images, {}, str(path), ext=".txt")


def test_evaluate_scene_uses_strided_frames_and_reports_completeness(tmp_path):
    names = [f"{i:06d}.png" for i in range(1, 61)]
    gt = circle_trajectory(60)
    _write_poses(tmp_path / "gt" / "sparse" / "0", names, gt)
    inputs = expected_input_names(names, 5)
    assert inputs == names[::5] and len(inputs) == 12
    # The method returns 9 of its 12 input frames, in a scaled/rotated frame.
    keep = [i for i in range(0, 60, 5)][:9]
    pred = similarity(gt[keep], 3.0, rot_z(0.4), np.array([1.0, 2.0, 3.0]))
    _write_poses(tmp_path / "pred" / "sparse" / "0", [names[i] for i in keep],
                 pred)

    (tmp_path / "gt" / "images").mkdir()
    for name in names:
        (tmp_path / "gt" / "images" / name).write_bytes(b"")
    scene_info = SimpleNamespace(task="SLAM",
                                 dataset="tum_dynamics",
                                 scene="seq",
                                 root_path=tmp_path / "gt",
                                 sparse_gt_path=tmp_path / "gt" / "sparse" /
                                 "0")
    result = evaluate_slam_scene(scene_info,
                                 tmp_path / "pred",
                                 config=SlamEvaluationConfig(stride=5))
    assert result["alignment_transform"]["scale"] == pytest.approx(1 / 3.0)
    assert result["num_input_frames"] == 12
    assert result["num_matched_images"] == 9
    assert result["input_frames_per_second"] == 6.0
    assert result["metrics"]["completeness"] == pytest.approx(0.75)
    assert result["metrics"]["ate_rmse"] == pytest.approx(0.0, abs=1e-6)

    aggregate = aggregate_slam_results([result, result])
    assert aggregate["num_sequences"] == 2.0

    # Images and GT poses must match one to one: an extra image would shift every strided frame after it.
    (tmp_path / "gt" / "images" / "000000.png").write_bytes(b"")
    with pytest.raises(ValueError, match="one to one"):
        evaluate_slam_scene(scene_info,
                            tmp_path / "pred",
                            config=SlamEvaluationConfig(stride=5))
    assert aggregate["completeness"] == pytest.approx(0.75)


def test_rpe_pairs_frames_by_input_index_when_frames_are_missing():
    # Same 10 % overshoot line as above, but every third frame has no prediction (keyframe-style output).
    n = 40
    gt = np.tile(np.eye(4), (n, 1, 1))
    gt[:, 0, 3] = 0.1 * np.arange(n)
    pred = gt.copy()
    pred[:, 0, 3] = 0.11 * np.arange(n)
    valid = np.arange(n) % 3 != 2
    pred[~valid] = 1e6  # garbage rows must be ignored
    metrics = trajectory_metrics(gt,
                                 pred,
                                 frames_per_second=5.0,
                                 alignment="se3",
                                 valid=valid)
    assert metrics["rpe_trans_step"] == pytest.approx(0.01)
    assert metrics["rpe_trans_1s"] == pytest.approx(0.05)
    assert metrics["drift_pct"] == pytest.approx(10.0)


def test_alignment_ignores_gross_outlier_frames_but_ate_still_counts_them():
    gt = circle_trajectory(60)
    pred = similarity(gt, 0.5, rot_z(0.3), np.array([1.0, 0.0, 0.0]))
    pred[[5, 30, 47], :3, 3] += 10.0  # three catastrophic frames
    metrics = trajectory_metrics(gt,
                                 pred,
                                 frames_per_second=6.0,
                                 alignment="sim3")
    assert metrics["alignment_scale"] == pytest.approx(
        2.0)  # recovered from the 57 good frames
    assert metrics["alignment_inlier_ratio"] == pytest.approx(57 / 60)
    assert metrics["ate_median"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["ate_rmse"] > 1.0  # the outlier frames still count in ATE


def test_alignment_estimator_choice_is_validated_and_the_previous_one_stays_available(
):
    """Both estimators align an exact Sim(3); unknown names are refused; the refit is the default."""
    gt = circle_trajectory(60)
    pred = similarity(gt, 0.5, rot_z(0.3), np.array([1.0, 0.0, 0.0]))
    for estimator in ("ransac_gtsam_irls", "ransac_dagsfm"):
        metrics = trajectory_metrics(gt,
                                     pred,
                                     frames_per_second=6.0,
                                     alignment="sim3",
                                     alignment_estimator=estimator)
        assert metrics["ate_rmse"] == pytest.approx(0.0, abs=1e-6), estimator
        assert metrics["alignment_scale"] == pytest.approx(2.0), estimator
    with pytest.raises(ValueError, match="Unknown alignment estimator"):
        fit_alignment(gt, pred, with_scale=True, estimator="evo")
    assert SlamEvaluationConfig().alignment_estimator == "ransac_gtsam_irls"


def _camera_poses(c2w: np.ndarray) -> dict[str, CameraPose]:
    poses = {}
    for index, pose in enumerate(c2w):
        rotation = pose[:3, :3].T
        poses[f"{index:04d}.png"] = CameraPose(name=f"{index:04d}.png",
                                               qvec=rotmat2qvec(rotation),
                                               tvec=-rotation @ pose[:3, 3])
    return poses


def test_pose_auc_matches_the_sfm_evaluator_and_ignores_the_alignment():
    """SLAM pose AUC is the SfM track's AUC over the same pairs, and a similarity transform leaves it unchanged."""
    rng = np.random.default_rng(0)
    gt = circle_trajectory(40)
    noisy = gt.copy()
    for index in range(len(noisy)):
        axis = rng.normal(size=3)
        angle = math.radians(rng.uniform(0.0, 8.0))
        axis = axis / np.linalg.norm(axis) * angle
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]]) / angle
        noisy[
            index, :3, :3] = (np.eye(3) + math.sin(angle) * k +
                              (1 - math.cos(angle)) * k @ k) @ gt[index, :3, :3]
        noisy[index, :3, 3] += rng.normal(scale=0.05, size=3)
    ours = pose_auc_metrics(gt, noisy, DEFAULT_AUC_THRESHOLDS, max_frames=0)
    sfm = compute_relative_pose_metrics(_camera_poses(gt), _camera_poses(noisy))
    assert ours["auc_num_pairs"] == sfm["num_pairs"] == 40 * 39 / 2
    for threshold in DEFAULT_AUC_THRESHOLDS:
        assert ours[f"auc_{threshold:g}"] == pytest.approx(
            sfm[f"auc_{threshold:g}"], abs=1e-9)
    moved = similarity(noisy, 3.7, rot_z(1.1), np.array([5.0, -2.0, 1.0]))
    assert pose_auc_metrics(gt, moved, DEFAULT_AUC_THRESHOLDS,
                            0)["auc_30"] == pytest.approx(ours["auc_30"])
    assert pose_auc_metrics(gt, gt, DEFAULT_AUC_THRESHOLDS,
                            0)["auc_1"] == pytest.approx(1.0)


def test_pose_auc_caps_the_frames_and_is_reported_per_scene_and_aggregated():
    gt = circle_trajectory(60)
    capped = pose_auc_metrics(gt, gt, DEFAULT_AUC_THRESHOLDS, max_frames=10)
    assert capped["auc_num_frames"] == 10 and capped["auc_num_pairs"] == 45
    metrics = trajectory_metrics(gt,
                                 gt,
                                 frames_per_second=10.0,
                                 alignment="sim3")
    assert metrics["auc_30"] == pytest.approx(
        1.0) and metrics["auc_num_pairs"] == 60 * 59 / 2
    aggregate = aggregate_slam_results([{
        "metrics": metrics
    }, {
        "metrics": metrics | {
            "auc_30": 0.5
        }
    }])
    assert aggregate["auc_30"] == pytest.approx(0.75)
    assert "auc_num_pairs" not in aggregate


def test_matched_input_names_requires_images_and_gt_poses_to_match():
    from spatialarena.eval.slam_trajectory_evaluator import matched_input_names

    names = [f"{i:06d}.png" for i in range(10)]
    assert matched_input_names(names, set(names), 3) == names[::3]
    with pytest.raises(ValueError, match="1 GT pose"):
        matched_input_names(names[1:], set(names), 3)
    with pytest.raises(ValueError, match="1 image"):
        matched_input_names(names, set(names[:-1]), 3)


# ---- aligned_trajectory ------------------------------------------------------------------------------------


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about ``axis`` by ``angle`` radians."""
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a wxyz unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def gt_cameras(count: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, camera-to-world rotation, center) along a curved, turning path."""
    cameras = []
    for index in range(count):
        angle = 0.15 * index
        center = np.array(
            [np.cos(angle) * 3.0,
             np.sin(angle) * 3.0, 0.1 * index])
        rotation = rotation_about([0.2, 0.3, 1.0], angle) @ rotation_about(
            [1.0, 0.0, 0.0], 0.05 * index)
        cameras.append((f"{index:04d}.png", rotation, center))
    return cameras


def write_colmap_model(
        path: Path, cameras: list[tuple[str, np.ndarray, np.ndarray]]) -> Path:
    """COLMAP text model of camera-to-world poses (COLMAP stores world-to-camera)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "cameras.txt").write_text("1 SIMPLE_PINHOLE 640 480 500 320 240\n",
                                      encoding="utf-8")
    lines = []
    for image_id, (name, rotation_c2w, center) in enumerate(cameras, start=1):
        rotation_w2c = rotation_c2w.T
        translation = -rotation_w2c @ center
        qw, qx, qy, qz = rotation_to_quaternion_wxyz(rotation_w2c[None])[0]
        lines += [
            f"{image_id} {qw} {qx} {qy} {qz} {translation[0]} {translation[1]} {translation[2]} 1 {name}",
            ""
        ]
    (path / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path / "points3D.txt").write_text("", encoding="utf-8")
    return path


def transformed(cameras, scale: float, rotation: np.ndarray,
                translation: np.ndarray):
    """The same cameras in another frame: x' = scale * rotation @ x + translation."""
    return [(name, rotation @ rot, scale * rotation @ center + translation)
            for name, rot, center in cameras]


def test_quaternion_round_trip_and_sign():
    """Quaternions reproduce their rotation and have w >= 0, including the 180-degree case."""
    rotations = [
        rotation_about(axis, angle)
        for axis, angle in (([0, 0, 1], 0.3), ([1, 1, 0], 2.9),
                            ([0, 1, 0], np.pi), ([1, 2, 3], -1.2))
    ]
    quaternions = rotation_to_quaternion_wxyz(np.stack(rotations))
    assert np.all(quaternions[:, 0] >= 0)
    for rotation, quaternion in zip(rotations, quaternions):
        assert np.allclose(quaternion_to_rotation(quaternion),
                           rotation,
                           atol=1e-9)


def test_sfm_trajectory_recovers_a_similarity_transform(tmp_path):
    """A prediction that is the GT in another Sim(3) frame aligns back onto the GT with zero ATE."""
    cameras = gt_cameras(24)
    gt = write_colmap_model(tmp_path / "gt", cameras)
    pred = write_colmap_model(
        tmp_path / "pred",
        transformed(cameras, 0.4, rotation_about([1, -1, 2], 0.7),
                    np.array([5.0, -2.0, 1.0])))
    payload = sfm_trajectory(gt, pred)
    assert payload["num_frames"] == payload["num_valid"] == 24
    assert payload["ate_rmse"] < 1e-6
    assert abs(payload["alignment"]["scale"] - 1 / 0.4) < 1e-6
    assert np.allclose(payload["pred"]["positions"],
                       payload["gt"]["positions"],
                       atol=2e-4)
    assert np.allclose(payload["pred"]["quaternions"],
                       payload["gt"]["quaternions"],
                       atol=2e-4)


def test_slam_trajectory_uses_the_strided_inputs_and_keeps_gaps(tmp_path):
    """SLAM frames are every stride-th GT frame; a frame the method did not return stays a gap."""
    cameras = gt_cameras(30)
    gt = write_colmap_model(tmp_path / "gt", cameras)
    inputs = cameras[::3]
    returned = [camera for index, camera in enumerate(inputs) if index != 4]
    pred = write_colmap_model(
        tmp_path / "pred",
        transformed(returned, 2.0, rotation_about([0, 0, 1], -0.4),
                    np.array([0.0, 1.0, 0.0])))
    payload = slam_trajectory(gt, pred, stride=3, alignment="sim3")
    assert payload["names"] == [name for name, _, _ in inputs]
    assert payload["num_frames"] == 10 and payload["num_valid"] == 9
    assert payload["pred"]["positions"][4] is None and payload["ate"][4] is None
    assert payload["ate_rmse"] < 1e-6
    assert payload["alignment"]["stride"] == 3


def test_trajectories_reuse_the_evaluators_alignment_and_fit_only_without_it(
        tmp_path):
    """With the evaluator's alignment_transform the viewer applies it as is; without one it fits the same Sim(3)."""
    from spatialarena.eval.sparse_reconstruction_evaluator import (
        evaluate_sparse_scene,)

    cameras = gt_cameras(24)
    gt = write_colmap_model(tmp_path / "gt", cameras)
    pred = write_colmap_model(
        tmp_path / "pred" / "sparse" / "0",
        transformed(cameras, 0.4, rotation_about([1, -1, 2], 0.7),
                    np.array([5.0, -2.0, 1.0])))
    scene = SimpleNamespace(task="SfM",
                            dataset="eth3d",
                            scene="s",
                            sparse_gt_path=gt)
    result = evaluate_sparse_scene(scene, tmp_path / "pred")
    transform = result["alignment_transform"]
    assert abs(transform["scale"] - 1 / 0.4) < 1e-6

    fitted = sfm_trajectory(gt, pred)
    reused = sfm_trajectory(gt, pred, transform=transform)
    assert fitted["alignment"]["source"] == "viewer_fit" and reused[
        "alignment"]["source"] == "evaluator"
    assert np.allclose(reused["pred"]["positions"],
                       fitted["pred"]["positions"],
                       atol=2e-4)
    # The given transform is what is drawn, not a refit: the identity leaves the prediction where it is.
    identity = {
        "scale": 1.0,
        "rotation": np.eye(3).tolist(),
        "translation": [0.0, 0.0, 0.0]
    }
    assert not np.allclose(sfm_trajectory(
        gt, pred, transform=identity)["pred"]["positions"],
                           fitted["pred"]["positions"],
                           atol=1e-2)
