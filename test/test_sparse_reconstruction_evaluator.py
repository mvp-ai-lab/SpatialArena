# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import struct
from pathlib import Path

import numpy as np

from spatialarena.eval.canonical_dataset import BenchmarkSceneInfo
from spatialarena.eval.sparse_reconstruction_evaluator import (
    align_points_similarity,
    compute_relative_pose_metrics,
    evaluate_sparse_scene,
    match_pose_names,
    read_colmap_poses,
)


def write_colmap_text_model(path: Path,
                            translation_scale: float = 1.0,
                            prefix: str = "") -> None:
    """Write a small COLMAP text model with two cameras' poses."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "cameras.txt").write_text(
        "1 SIMPLE_PINHOLE 640 480 500 320 240\n",
        encoding="utf-8",
    )
    (path / "images.txt").write_text(
        "\n".join([
            f"1 1 0 0 0 0 0 0 1 {prefix}000.png",
            "",
            f"2 1 0 0 0 {translation_scale} 0 0 1 {prefix}001.png",
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    (path / "points3D.txt").write_text("", encoding="utf-8")


def write_colmap_binary_images(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "cameras.bin").write_bytes(b"stub")
    (path / "points3D.bin").write_bytes(b"stub")
    with (path / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(
            struct.pack(
                "<idddddddi",
                1,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1,
            ))
        file.write(b"000.png\x00")
        file.write(struct.pack("<Q", 0))


def test_read_colmap_poses_reads_text_model(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    write_colmap_text_model(sparse)

    poses = read_colmap_poses(sparse)

    assert sorted(poses) == ["000.png", "001.png"]
    np.testing.assert_allclose(poses["000.png"].qvec, [1.0, 0.0, 0.0, 0.0])


def test_read_colmap_poses_reads_binary_images(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    write_colmap_binary_images(sparse)

    poses = read_colmap_poses(sparse)

    assert list(poses) == ["000.png"]
    np.testing.assert_allclose(poses["000.png"].tvec, [0.0, 0.0, 0.0])


def test_compute_relative_pose_metrics_is_zero_for_scaled_identical_layout(
        tmp_path):
    gt_sparse = tmp_path / "gt"
    pred_sparse = tmp_path / "pred"
    write_colmap_text_model(gt_sparse, translation_scale=1.0)
    write_colmap_text_model(pred_sparse, translation_scale=2.0)

    metrics = compute_relative_pose_metrics(
        read_colmap_poses(gt_sparse),
        read_colmap_poses(pred_sparse),
    )

    assert metrics["num_pairs"] == 1
    assert metrics["relative_rotation_error_deg_mean"] == 0.0
    assert metrics["relative_translation_error_deg_mean"] == 0.0
    assert metrics["rotation_error_deg_mean"] == 0.0
    assert metrics["translation_direction_error_deg_mean"] == 0.0
    assert metrics["relative_pose_error_deg_mean"] == 0.0
    assert metrics["ate_rmse"] == 0.0
    assert metrics["RRA_at_5"] == 1.0
    assert metrics["RTA_at_5"] == 1.0
    assert metrics["mAA_30"] == 1.0
    assert metrics["auc_1"] == 1.0
    assert metrics["auc_3"] == 1.0
    assert metrics["auc_30"] == 1.0


def test_rta_and_maa_use_translation_axis_error_while_direct_error_is_retained(
        tmp_path):
    gt_sparse = tmp_path / "gt"
    pred_sparse = tmp_path / "pred"
    write_colmap_text_model(gt_sparse, translation_scale=1.0)
    write_colmap_text_model(pred_sparse, translation_scale=-1.0)

    metrics = compute_relative_pose_metrics(
        read_colmap_poses(gt_sparse),
        read_colmap_poses(pred_sparse),
    )

    assert metrics["relative_translation_error_deg_mean"] == 180.0
    assert metrics["relative_translation_axis_error_deg_mean"] == 0.0
    assert metrics["relative_pose_error_deg_mean"] == 180.0
    assert metrics["relative_pose_axis_error_deg_mean"] == 0.0
    assert metrics["RTA_at_5"] == 1.0
    assert metrics["mAA_30"] == 1.0


def test_similarity_alignment_removes_scale_rotation_and_translation():
    source = np.array(
        [
            [3.0, 1.0, 0.0],
            [3.0, 3.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    aligned, scale, _, _ = align_points_similarity(source, target)

    np.testing.assert_allclose(aligned, target, atol=1e-12)
    assert scale > 0


def test_evaluate_sparse_scene_reads_standardized_sparse_zero(tmp_path):
    scene_root = tmp_path / "scene"
    gt_sparse = scene_root / "sparse" / "0"
    pred_scene = tmp_path / "exp" / "SfM" / "imc" / "DA3" / "scene"
    pred_sparse = pred_scene / "sparse" / "0"
    write_colmap_text_model(gt_sparse)
    write_colmap_text_model(pred_sparse)

    scene_info = BenchmarkSceneInfo(
        dataset="imc",
        task="SfM",
        scene="scene",
        root_path=scene_root,
        sparse_gt_path=gt_sparse,
    )
    result = evaluate_sparse_scene(scene_info, pred_scene)

    assert result["num_gt_images"] == 2
    assert result["num_pred_images"] == 2
    assert result["num_matched_images"] == 2
    assert result["metrics"]["num_pairs"] == 1
    assert "ate_rmse" in result["metrics"]
    assert "relative_rotation_error_deg_mean" in result["metrics"]
    assert "relative_translation_error_deg_mean" in result["metrics"]
    assert "RRA_at_5" in result["metrics"]
    assert "RTA_at_5" in result["metrics"]
    assert "mAA_30" in result["metrics"]
    assert "auc_1" in result["metrics"]
    assert "auc_30" in result["metrics"]


def test_gt_names_with_folders_match_predicted_file_names(tmp_path):
    """ETH3D's GT names images with their folder (dslr_images_undistorted/DSC_x.JPG); adapters write DSC_x.JPG."""
    scene_root = tmp_path / "scene"
    gt_sparse = scene_root / "sparse" / "0"
    pred_scene = tmp_path / "exp" / "SfM" / "eth3d" / "M" / "scene"
    write_colmap_text_model(gt_sparse, prefix="dslr_images_undistorted/")
    write_colmap_text_model(pred_scene / "sparse" / "0", translation_scale=2.0)
    scene_info = BenchmarkSceneInfo(dataset="eth3d",
                                    task="SfM",
                                    scene="scene",
                                    root_path=scene_root,
                                    sparse_gt_path=gt_sparse)
    result = evaluate_sparse_scene(scene_info, pred_scene)
    assert result["num_matched_images"] == 2 and result["metrics"][
        "num_pairs"] == 1


def test_name_matching_keeps_exact_names_and_refuses_ambiguous_basenames(
        tmp_path):
    """Exact names win when they overlap; duplicate file names in different folders are not merged."""
    write_colmap_text_model(tmp_path / "a")
    write_colmap_text_model(tmp_path / "b", prefix="x/")
    exact = read_colmap_poses(tmp_path / "a")
    gt, pred = match_pose_names(exact, exact)
    assert gt is exact and pred is exact
    prefixed = read_colmap_poses(tmp_path / "b")
    ambiguous = {
        **prefixed,
        **{
            name.replace("x/", "y/"): pose for name, pose in prefixed.items()
        }
    }
    gt, pred = match_pose_names(ambiguous, exact)
    assert not set(gt) & set(pred)
