# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""dtu_import_dependencies: The standard reconstruction scorer must not require legacy CLI dependencies.

mvrecon_parallel_metrics: Exact metric equivalence and query-memory bounds for large reconstructions.

spatialarena_backprojection: SpatialArena-only direct reconstruction, with explicit camera and voxel semantics.
"""

from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import spatialarena.eval.mvrecon_evaluator as mvrecon_eval
from spatialarena.eval.canonical_dataset import BenchmarkSceneInfo
from spatialarena.eval.dataset_registry import select_dataset_specs
from spatialarena.eval.mvrecon_evaluator import (
    ColmapImageRecord,
    ColmapMappingDiagnostics,
    MVReconEvaluationConfig,
    MVReconView,
    MVReconViewManifest,
    _align_da3_poses,
    _prepare_da3_fusion_inputs,
    _valid_depth_mask,
    compute_reconstruction_metrics,
    evaluate_mvrecon_scene,
    prepare_fusion_inputs,
    protocol_for_dataset,
    voxel_downsample_points,
)
from spatialarena.eval.pose_alignment import Sim3Estimate
from scripts.test.generate_full_standard_cases import scene_is_evaluable
from types import SimpleNamespace
from spatialarena.eval import eval_dtu_mesh as dtu
import os
import subprocess
import sys
from scipy.spatial import cKDTree
import json
import yaml

# ---- mvrecon_evaluator -------------------------------------------------------------------------------------


def test_reconstruction_metrics_are_perfect_for_identical_pointclouds():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    metrics = compute_reconstruction_metrics(
        points,
        points,
        threshold=0.01,
        down_sample=None,
    )

    assert metrics["accuracy"] == 0.0
    assert metrics["completion"] == 0.0
    assert metrics["overall"] == 0.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["fscore"] == 1.0


def test_reconstruction_metrics_report_directional_accuracy_completion():
    pred = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
    gt = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    metrics = compute_reconstruction_metrics(
        pred,
        gt,
        threshold=0.05,
        down_sample=None,
    )

    assert metrics["accuracy"] == 5.0
    assert metrics["completion"] == 0.0
    assert metrics["overall"] == 2.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["fscore"] == 2.0 / 3.0


def test_voxel_downsample_points_keeps_one_point_per_voxel():
    points = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    downsampled = voxel_downsample_points(points, 0.1)

    assert downsampled.shape == (2, 3)


def test_da3_bench_mvrecon_datasets_are_registered():
    specs = select_dataset_specs(
        tasks=["MVRecon"],
        datasets=["scannetpp", "hiroom", "tanks_and_temples"])
    assert {(spec.task, spec.dataset) for spec in specs} == {
        ("MVRecon", "scannetpp"),
        ("MVRecon", "hiroom"),
        ("MVRecon", "tanks_and_temples"),
    }
    tnt_specs = [spec for spec in specs if spec.dataset == "tanks_and_temples"]
    assert len(tnt_specs) == 1
    assert tnt_specs[0].relative_path == Path("MVRecon/tanks_and_temples")


def test_mvrecon_registry_uses_only_canonical_mvrecon_roots():
    specs = select_dataset_specs(tasks=["MVRecon"])

    assert all(spec.relative_path.parts[:1] == ("MVRecon",) for spec in specs)
    assert all(not spec.legacy_relative_paths for spec in specs)


def test_mvrecon_protocols_match_da3_bench_thresholds():
    scannetpp = protocol_for_dataset("scannetpp")
    eth3d = protocol_for_dataset("eth3d")
    hiroom = protocol_for_dataset("hiroom")

    assert scannetpp.voxel_length == 0.02
    assert scannetpp.sdf_trunc == 0.15
    assert scannetpp.eval_threshold == 0.05
    assert eth3d.eval_threshold == 0.25
    assert hiroom.eval_threshold == 0.05


def test_dtu_mvrecon_uses_mvsnet_evaluator_without_artifact_root(
        tmp_path, monkeypatch):
    scene_root = tmp_path / "scan1"
    scene_root.mkdir()
    scene_info = BenchmarkSceneInfo(
        dataset="dtu",
        task="MVRecon",
        scene="scan1",
        root_path=scene_root,
        sparse_gt_path=scene_root / "sparse" / "0",
        metadata={},
    )
    calls = []

    monkeypatch.setattr(
        mvrecon_eval,
        "prepare_fusion_inputs",
        lambda **_kwargs: (
            np.ones((1, 2, 2), dtype=np.float32),
            np.zeros((1, 2, 2, 3), dtype=np.uint8),
            np.repeat(np.eye(3, dtype=np.float64)[None], 1, axis=0),
            np.repeat(np.eye(4, dtype=np.float64)[None], 1, axis=0),
            {},
        ),
    )

    class DummyPointCloud:
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    monkeypatch.setattr(mvrecon_eval, "fuse_depths_to_pointcloud",
                        lambda *_args: DummyPointCloud())
    monkeypatch.setattr(mvrecon_eval, "resolve_gt_geometry_path",
                        lambda _scene_info: scene_root / "stl001_total.ply")
    monkeypatch.setattr(mvrecon_eval, "load_point_cloud_array",
                        lambda *_args: np.array([[0.0, 0.0, 0.0]]))
    icp_calls = []

    def fake_icp_refine_sim3(pred_points,
                             _gt_points,
                             initial,
                             *,
                             max_correspondence_distance=None):
        icp_calls.append((np.asarray(pred_points).copy(), initial,
                          max_correspondence_distance))
        return Sim3Estimate(scale=1.0,
                            rotation=np.eye(3),
                            translation=np.array([1.0, 1.0, 1.0]),
                            inliers=np.zeros(0, dtype=bool))

    monkeypatch.setattr(mvrecon_eval, "icp_refine_sim3", fake_icp_refine_sim3)
    monkeypatch.setattr(
        mvrecon_eval,
        "dtu_eval_paths",
        lambda _scene_info: (
            1,
            scene_root / "gt.ply",
            scene_root / "mask.mat",
            scene_root / "plane.mat",
        ),
    )

    def fake_write_point_cloud(path, points):
        calls.append(("write", Path(path), np.asarray(points).copy()))

    def fake_evaluate_dtu(_scene_info, pred_pcd_path, output_dir):
        calls.append(("dtu", Path(pred_pcd_path), Path(output_dir)))
        return {"accuracy": 1.0, "completion": 2.0, "overall": 1.5}

    monkeypatch.setattr(mvrecon_eval, "write_point_cloud",
                        fake_write_point_cloud)
    monkeypatch.setattr(mvrecon_eval, "evaluate_dtu_mvsnet", fake_evaluate_dtu)

    result = evaluate_mvrecon_scene(
        scene_info,
        tmp_path / "pred",
        config=MVReconEvaluationConfig(artifact_root=None),
    )

    assert result["metrics"] == {
        "accuracy": 1.0,
        "completion": 2.0,
        "overall": 1.5
    }
    assert [call[0] for call in calls] == ["write", "dtu"]
    np.testing.assert_array_equal(calls[0][2], np.array([[1.0, 1.0, 1.0]]))
    assert len(icp_calls) == 1
    _seeded_points, seeded_initial, threshold = icp_calls[0]
    # The fused points are already in the GT frame: ICP starts from the identity, pairs within 2 eval thresholds.
    assert threshold == pytest.approx(
        2 * mvrecon_eval.protocol_for_dataset("dtu").eval_threshold)
    assert seeded_initial.scale == 1.0
    np.testing.assert_array_equal(seeded_initial.rotation, np.eye(3))
    np.testing.assert_array_equal(seeded_initial.translation, np.zeros(3))


def test_fit_dtu_raw_calibration_sim3_bridges_normalized_and_raw_frames(
        tmp_path, monkeypatch):
    """DTU's evaluate_dtu_mvsnet expects points in the per-rig calibration
    frame (Calibration/cal18/pos_NNN.txt), not the normalized/materialized
    GT frame everything else in this module operates in. This must recover
    the exact known Sim(3) between the two, reusing eval_dtu_mesh's own
    best_fit_transform rather than a separate fitter."""

    scene_root = tmp_path / "scan1"
    scene_root.mkdir()
    (scene_root.parent / "Calibration").mkdir()

    rng = np.random.default_rng(0)
    raw_centers = rng.uniform(-300.0, 300.0, size=(5, 3)) + np.array(
        [0.0, 0.0, 600.0])
    raw_c2w = np.repeat(np.eye(4)[None], 5, axis=0)
    raw_c2w[:, :3, 3] = raw_centers
    monkeypatch.setattr(dtu, "load_dtu_camera", lambda _path: raw_c2w)

    true_scale = 314.0
    true_rotation = Rotation.from_euler("xyz", [10, 20, 30],
                                        degrees=True).as_matrix()
    true_translation = np.array([1.0, 2.0, 3.0])
    # Inverse of transform_points's scale * (points @ rotation.T) + translation.
    norm_centers = (
        (raw_centers - true_translation) / true_scale) @ true_rotation

    class FakeView:

        def __init__(self, center):
            self.center = center

    gt_views = {str(i): FakeView(norm_centers[i]) for i in range(5)}

    result = mvrecon_eval._fit_dtu_raw_calibration_sim3(gt_views, scene_root)  # pylint: disable=protected-access

    assert result is not None
    np.testing.assert_allclose(result.scale, true_scale, rtol=1e-4)
    np.testing.assert_allclose(result.transform_points(norm_centers),
                               raw_centers,
                               atol=1e-6)


def test_fit_dtu_raw_calibration_sim3_returns_none_without_calibration_dir(
        tmp_path):
    scene_root = tmp_path / "scan1"
    scene_root.mkdir()

    result = mvrecon_eval._fit_dtu_raw_calibration_sim3({}, scene_root)  # pylint: disable=protected-access

    assert result is None


def test_mvrecon_case_generation_accepts_sparse_only_geometry_scenes():
    scene = {"has_images": True, "has_depths": False, "has_sparse": True}

    assert scene_is_evaluable("MVRecon",
                              scene,
                              include_mvrecon_depth_compat=False) is False
    assert scene_is_evaluable("MVRecon",
                              scene,
                              include_mvrecon_depth_compat=True) is True


def test_mvrecon_case_generation_keeps_depth_backed_scenes_without_compat_flag(
):
    scene = {"has_images": True, "has_depths": True, "has_sparse": True}

    assert scene_is_evaluable("MVRecon",
                              scene,
                              include_mvrecon_depth_compat=False) is True


def _record(name: str,
            *,
            width: int = 4,
            height: int = 3,
            intrinsic=None,
            w2c=None):
    return ColmapImageRecord(
        image_id=1,
        name=name,
        camera_id=1,
        model="PINHOLE",
        width=width,
        height=height,
        params=np.array([2.0, 2.0, 1.5, 1.0], dtype=np.float32),
        intrinsic=np.array(
            intrinsic if intrinsic is not None else
            [[2.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        w2c=np.array(w2c if w2c is not None else np.eye(4), dtype=np.float32),
    )


# pylint: disable-next=unused-argument
def _manifest_for_test(dataset: str,
                       image_path: Path,
                       gt_depth: Path | None = None,
                       mask: Path | None = None):
    record = _record(image_path.name)
    view = MVReconView(
        image_path=image_path,
        gt_record=record,
        pred_record=record,
        pred_depth_path=image_path.parent / "pred.npy",
        gt_depth_path=gt_depth,
        mask_path=mask,
        image_match_kind="exact_relative",
    )
    empty = ColmapMappingDiagnostics(mapped=(), missing=(), ambiguous=())
    return MVReconViewManifest(
        views=(view,),
        gt_mapping=empty,
        pred_mapping=empty,
        missing_prediction_depths=(),
        missing_prediction_poses=(),
    )


def test_da3_7scenes_uses_fixed_simple_recon_intrinsics_and_masks_invalid_depth(
        tmp_path, monkeypatch):
    image_path = tmp_path / "frame-000000.color.png"
    image_path.touch()
    gt_path = tmp_path / "frame-000000.depth.proj.png"
    gt_path.touch()
    pred_path = image_path.parent / "pred.npy"
    pred_path.touch()
    scene = BenchmarkSceneInfo(
        dataset="7scenes",
        task="MVRecon",
        scene="chess",
        root_path=tmp_path,
        image_path=tmp_path,
        depth_gt_path=tmp_path,
        sparse_gt_path=tmp_path / "sparse",
    )
    monkeypatch.setattr(
        mvrecon_eval, "_build_view_manifest",
        lambda **_: _manifest_for_test("7scenes", image_path, gt_path))
    monkeypatch.setattr(mvrecon_eval, "_read_rgb", lambda _path: np.zeros(
        (3, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(
        mvrecon_eval, "load_depth_map", lambda path:
        (np.array([[1, 65535, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4]],
                  dtype=np.float32)
         if Path(path) == gt_path else np.ones((3, 4), dtype=np.float32)))
    depths, _images, intrinsics, _extrinsics, _metadata = _prepare_da3_fusion_inputs(
        scene_info=scene,
        prediction_scene_path=tmp_path / "pred",
        reconstruction_protocol="posed")
    np.testing.assert_allclose(intrinsics[0],
                               [[525, 0, 320], [0, 525, 240], [0, 0, 1]])
    assert depths[0, 0, 1] == 0
    assert depths[0, 0, 0] == 1


def test_da3_eth3d_views_of_different_sizes_are_kept_per_view(
        tmp_path, monkeypatch):
    """ETH3D's undistorted images differ by a few pixels per view: each view keeps its own size (no np.stack)."""
    shapes = {"a.jpg": (3, 4), "b.jpg": (4, 5)}
    views = []
    for name in shapes:
        image_path = tmp_path / name
        image_path.touch()
        manifest = _manifest_for_test("eth3d", image_path)
        views.append(manifest.views[0])
    manifest = MVReconViewManifest(views=tuple(views),
                                   gt_mapping=manifest.gt_mapping,
                                   pred_mapping=manifest.pred_mapping,
                                   missing_prediction_depths=(),
                                   missing_prediction_poses=())
    scene = BenchmarkSceneInfo(dataset="eth3d",
                               task="MVRecon",
                               scene="courtyard",
                               root_path=tmp_path,
                               image_path=tmp_path,
                               depth_gt_path=tmp_path,
                               sparse_gt_path=tmp_path / "sparse")
    monkeypatch.setattr(mvrecon_eval, "_build_view_manifest",
                        lambda **_: manifest)
    monkeypatch.setattr(
        mvrecon_eval, "_align_da3_poses", lambda gt_w2c, pred_w2c:
        (pred_w2c, 1.0, np.eye(3), np.zeros(3)))
    monkeypatch.setattr(
        mvrecon_eval, "_read_rgb",
        lambda path: np.zeros(shapes[Path(path).name] + (3,), dtype=np.uint8))
    monkeypatch.setattr(mvrecon_eval, "load_depth_map", lambda _path: np.ones(
        (3, 4), dtype=np.float32))
    depths, images, intrinsics, _extrinsics, _metadata = _prepare_da3_fusion_inputs(
        scene_info=scene,
        prediction_scene_path=tmp_path / "pred",
        reconstruction_protocol="posed")
    assert [depth.shape for depth in depths] == [(3, 4), (4, 5)]
    assert [image.shape[:2] for image in images] == [(3, 4), (4, 5)]
    assert intrinsics.shape == (2, 3, 3)


def test_da3_valid_masks_match_eth3d_and_hiroom_protocols(tmp_path):
    gt = tmp_path / "gt.npy"
    mask = tmp_path / "mask.npy"
    np.save(gt, np.array([[1, 0], [2, 3]], dtype=np.float32))
    np.save(mask, np.array([[0, 1], [0, 0]], dtype=np.uint8))
    eth_mask = _valid_depth_mask("eth3d",
                                 gt_depth_path=gt,
                                 mask_path=mask,
                                 target_shape=(2, 2))
    assert eth_mask.tolist() == [[True, False], [True, True]]

    alias = tmp_path / "alias.npy"
    np.save(alias, np.array([[0, 1], [0, 0]], dtype=np.uint8))
    hi_mask = _valid_depth_mask("hiroom",
                                gt_depth_path=gt,
                                mask_path=alias,
                                target_shape=(2, 2))
    assert hi_mask.tolist() == [[True, False], [True, True]]


def test_da3_scannetpp_mask_uses_two_stage_target_resolution(tmp_path):
    gt = tmp_path / "gt.npy"
    np.save(gt, np.ones((2, 2), dtype=np.float32))
    mask = _valid_depth_mask("scannetpp",
                             gt_depth_path=gt,
                             mask_path=None,
                             target_shape=(768, 1024))
    assert mask.shape == (768, 1024)
    assert mask.all()


def test_mvrecon_dispatches_da3_only_for_four_datasets(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        mvrecon_eval, "_prepare_da3_fusion_inputs",
        lambda **kwargs: calls.append(
            ("da3", kwargs["scene_info"].dataset)) or (None,) * 5)
    monkeypatch.setattr(
        mvrecon_eval, "_prepare_generic_fusion_inputs",
        lambda **kwargs: calls.append(
            ("generic", kwargs["scene_info"].dataset)) or (None,) * 5)
    for dataset in ("7scenes", "eth3d", "hiroom", "scannetpp", "dtu",
                    "tanks_and_temples"):
        scene = BenchmarkSceneInfo(dataset=dataset,
                                   task="MVRecon",
                                   scene="scene",
                                   root_path=tmp_path)
        prepare_fusion_inputs(scene_info=scene,
                              prediction_scene_path=tmp_path / "pred",
                              reconstruction_protocol="posed")
    assert calls == [
        ("da3", "7scenes"),
        ("da3", "eth3d"),
        ("da3", "hiroom"),
        ("da3", "scannetpp"),
        ("generic", "dtu"),
        ("generic", "tanks_and_temples"),
    ]


def test_da3_pose_alignment_is_deterministic():
    gt = np.stack([
        np.eye(4),
        np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    ])
    pred = gt.copy()
    aligned_a = _align_da3_poses(gt, pred, random_state=42)
    aligned_b = _align_da3_poses(gt, pred, random_state=42)
    for left, right in zip(aligned_a, aligned_b):
        np.testing.assert_allclose(left, right)


def _make_colmap_views(
        centers: np.ndarray,
        rotations: np.ndarray) -> dict[str, mvrecon_eval.ColmapView]:
    views = {}
    for index, (center, rotation) in enumerate(zip(centers, rotations)):
        w2c = np.eye(4)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = -rotation @ center
        key = f"view_{index}"
        views[key] = mvrecon_eval.ColmapView(name=key,
                                             key=key,
                                             intrinsic=np.eye(3),
                                             w2c=w2c)
    return views


def test_fit_generic_pose_sim3_recovers_known_scale_and_rotation():
    """DTU/tanks_and_temples pose alignment previously used a plain
    closed-form Umeyama fit on camera centers only
    (sparse_reconstruction_evaluator.align_points_similarity). It now goes
    through _fit_generic_pose_sim3, the same pose-aware
    ransac_irls_align_sim3 pipeline _align_da3_poses already uses for the
    four DA3 datasets -- this recovers a known Sim(3) exactly, including
    rotation, which the old centers-only fit could get wrong on
    near-collinear trajectories.
    """
    rng = np.random.default_rng(20)
    gt_centers = rng.normal(scale=5.0, size=(6, 3))
    gt_rotations = Rotation.random(6, random_state=rng).as_matrix()
    true_scale = 2.0
    true_rotation = Rotation.random(random_state=rng).as_matrix()
    true_translation = rng.normal(scale=2.0, size=3)

    # gt_center = scale * (pred_center @ rotation.T) + translation
    pred_centers = (
        (gt_centers - true_translation) / true_scale) @ true_rotation
    pred_rotations = gt_rotations @ true_rotation

    gt_views = _make_colmap_views(gt_centers, gt_rotations)
    pred_views = _make_colmap_views(pred_centers, pred_rotations)
    keys = sorted(gt_views)

    estimate = mvrecon_eval._fit_generic_pose_sim3(pred_views, gt_views, keys)  # pylint: disable=protected-access

    assert estimate.scale == pytest.approx(true_scale, rel=1e-2)
    relative = estimate.rotation.T @ true_rotation
    rotation_error_deg = np.degrees(
        np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    assert rotation_error_deg < 1.0
    np.testing.assert_allclose(estimate.translation, true_translation, atol=0.1)


def test_icp_refinement_keeps_an_already_aligned_cloud_in_place():
    """The fused cloud is already in the GT frame: ICP must not re-apply the camera Sim(3).

    Re-applying it rotated DTU by ~60 deg."""
    rng = np.random.default_rng(0)
    gt_points = rng.uniform(-1.0, 1.0, size=(2000, 3)) * np.array(
        [1.0, 0.6, 0.3])
    pred_points = gt_points + rng.normal(scale=0.002, size=gt_points.shape)

    estimate = mvrecon_eval.refine_fused_points_in_gt_frame(
        pred_points, gt_points)

    assert estimate.scale == pytest.approx(1.0, abs=0.01)
    np.testing.assert_allclose(estimate.rotation, np.eye(3), atol=0.01)
    np.testing.assert_allclose(estimate.transform_points(pred_points),
                               gt_points,
                               atol=0.02)


def test_prediction_depth_index_ignores_confidence_maps(tmp_path):
    """confidences/<stem>.npy shares its depth map's name; indexing the prediction must not see two files per view."""
    for subdir in ("depths", "confidences"):
        (tmp_path / subdir).mkdir()
        np.save(tmp_path / subdir / "seq-01_frame-000082.color.npy",
                np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="Ambiguous depth files"):
        mvrecon_eval._index_depth_files(tmp_path)  # pylint: disable=protected-access
    # pylint: disable-next=protected-access
    indexed = mvrecon_eval._index_depth_paths(
        mvrecon_eval.find_depth_files(tmp_path))

    assert list(indexed.values()) == [
        tmp_path / "depths" / "seq-01_frame-000082.color.npy"
    ]


def test_mvrecon_case_requires_gt_geometry():
    """ETH3D meadow/terrace have GT depth but no GT mesh (not in DA3-BENCH), so MVRecon cannot score them."""
    scene = {
        "has_images": True,
        "has_depths": True,
        "has_sparse": True,
        "has_gt_geometry": False
    }

    assert not scene_is_evaluable(
        "MVRecon", scene, include_mvrecon_depth_compat=True)
    assert scene_is_evaluable("MVRecon", {
        **scene, "has_gt_geometry": True
    },
                              include_mvrecon_depth_compat=False)
    assert scene_is_evaluable("SfM", scene, include_mvrecon_depth_compat=False)


def test_icp_threshold_ignores_background_the_gt_does_not_cover():
    """An aligned object plus off-object background: only pairs within the threshold may move the fit."""
    rng = np.random.default_rng(1)
    gt_points = rng.uniform(-1.0, 1.0, size=(3000, 3)) * np.array(
        [1.0, 0.6, 0.3])
    background = rng.uniform(-1.0, 1.0, size=(3000, 3)) * np.array(
        [4.0, 4.0, 0.2]) + np.array([3.0, 0.0, -1.0])
    pred_points = np.concatenate(
        [gt_points + rng.normal(scale=0.002, size=gt_points.shape), background])

    unbounded = mvrecon_eval.refine_fused_points_in_gt_frame(
        pred_points, gt_points)
    bounded = mvrecon_eval.refine_fused_points_in_gt_frame(
        pred_points, gt_points, max_correspondence_distance=0.05)

    assert abs(unbounded.scale - 1.0) > 0.05 or np.abs(unbounded.rotation -
                                                       np.eye(3)).max() > 0.05
    assert bounded.scale == pytest.approx(1.0, abs=0.01)
    np.testing.assert_allclose(bounded.rotation, np.eye(3), atol=0.01)


def test_image_alpha_mask_reads_the_rgba_alpha_at_the_depth_shape(tmp_path):
    """DTU's evaluation mask is its images' alpha channel; RGB images have none."""
    from PIL import Image  # pylint: disable=import-outside-toplevel

    rgba = np.zeros((4, 6, 4), dtype=np.uint8)
    rgba[:, :3, 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "0000.png")
    Image.fromarray(rgba[..., :3], "RGB").save(tmp_path / "0001.png")

    mask = mvrecon_eval.image_alpha_mask(tmp_path / "0000.png", (2, 3))

    np.testing.assert_array_equal(
        mask, np.array([[True, True, False], [True, True, False]]))
    assert mvrecon_eval.image_alpha_mask(tmp_path / "0001.png", (2, 3)) is None


def _write_tanks_and_temples_registration(root: Path, scene: str,
                                          matrix: np.ndarray) -> None:
    np.savetxt(root / f"{scene}_trans.txt", matrix)
    (root / f"{scene}.json").write_text(
        '{"class_name": "SelectionPolygonVolume", "orthogonal_axis": "Z", "axis_min": -1.0, "axis_max": 1.0,'
        ' "bounding_polygon": [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], "version_major": 1, "version_minor": 0}',
        encoding="utf-8",
    )


def test_tanks_and_temples_crop_volume_matches_open3d(tmp_path):
    """The `<scene>.json` crop keeps exactly the points Open3D's `SelectionPolygonVolume` keeps."""
    o3d = pytest.importorskip("open3d")
    _write_tanks_and_temples_registration(tmp_path, "Barn", np.eye(4))
    points = np.random.default_rng(0).uniform(-1.0, 3.0, size=(2000, 3))

    crop = mvrecon_eval.read_crop_volume(tmp_path / "Barn.json")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    expected = np.asarray(
        o3d.visualization.read_selection_polygon_volume(
            str(tmp_path / "Barn.json")).crop_point_cloud(cloud).points)

    kept = points[crop.contains(points)]
    assert len(kept) == len(expected)
    np.testing.assert_allclose(np.sort(kept, axis=0), np.sort(expected, axis=0))


def test_tanks_and_temples_registration_reads_scaled_trans(tmp_path):
    """`<scene>_trans.txt` is a scaled similarity: the registration recovers scale, rotation and translation."""
    rotation = Rotation.from_euler("xyz", [10, -40, 75],
                                   degrees=True).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = 2.5 * rotation
    matrix[:3, 3] = [1.0, -2.0, 3.0]
    _write_tanks_and_temples_registration(tmp_path, "Truck", matrix)
    scene = BenchmarkSceneInfo(dataset="tanks_and_temples",
                               task="MVRecon",
                               scene="Truck",
                               root_path=tmp_path)

    transform, _crop = mvrecon_eval.tanks_and_temples_registration(scene)

    points = np.random.default_rng(1).normal(size=(5, 3))
    expected = points @ matrix[:3, :3].T + matrix[:3, 3]
    assert transform.scale == pytest.approx(2.5)
    np.testing.assert_allclose(transform.transform_points(points),
                               expected,
                               atol=1e-9)
    assert mvrecon_eval.tanks_and_temples_registration(
        BenchmarkSceneInfo(dataset="tanks_and_temples",
                           task="MVRecon",
                           scene="Truck",
                           root_path=tmp_path / "x")) is None


# ---- mvrecon_empty_geometry --------------------------------------------------------------------------------


@pytest.mark.parametrize("bad,which,message", [
    (np.empty((0, 3)), "points", "predicted point cloud is empty"),
    (np.empty((0, 3)), "gt_points", "ground-truth point cloud is empty"),
    (np.array([[np.nan, 1., 2.]]), "points", "non-finite coordinates"),
])
def test_invalid_fused_geometry_is_failure_before_metric_or_artifact(
        tmp_path, monkeypatch, bad, which, message):
    fused = SimpleNamespace(points=np.ones((2, 3)),
                            gt_points=np.ones((2, 3)),
                            colors=None,
                            fusion_metadata={})
    setattr(fused, which, bad)
    monkeypatch.setattr(mvrecon_eval, "fuse_prediction_in_gt_frame",
                        lambda *a, **kw: fused)

    def unexpected(*_args, **_kwargs):
        pytest.fail(
            "Invalid geometry must not be scored or exported as a successful result"
        )

    monkeypatch.setattr(mvrecon_eval, "compute_reconstruction_metrics",
                        unexpected)
    monkeypatch.setattr(mvrecon_eval, "write_point_cloud", unexpected)
    scene = BenchmarkSceneInfo(dataset="scannetpp",
                               task="MVRecon",
                               scene="test",
                               root_path=tmp_path)
    with pytest.raises(ValueError, match=message):
        mvrecon_eval.evaluate_mvrecon_scene(
            scene, tmp_path,
            mvrecon_eval.MVReconEvaluationConfig(artifact_root=tmp_path /
                                                 "artifacts"))


# ---- dtu_empty_masks ---------------------------------------------------------------------------------------


def test_dtu_observation_mask_empty_is_explicit_failure(tmp_path, monkeypatch):
    cloud = dtu.o3d.geometry.PointCloud()
    cloud.points = dtu.o3d.utility.Vector3dVector(
        np.array([[1., 1., 1.], [2., 2., 2.]]))
    path = tmp_path / "prediction.ply"
    dtu.o3d.io.write_point_cloud(str(path), cloud)
    monkeypatch.setattr(
        dtu, "loadmat", lambda _: {
            "ObsMask": np.zeros((5, 5, 5)),
            "BB": np.array([[0, 0, 0], [4, 4, 4]]),
            "Res": np.ones((1, 3))
        })
    with pytest.raises(ValueError, match="observation mask"):
        dtu.eval(1,
                 str(path),
                 "unused.ply",
                 "mask.mat",
                 "plane.mat",
                 eval_modal="pcd",
                 output_dir=str(tmp_path))
    assert not (tmp_path / "results.json").exists()


def test_dtu_truncation_without_neighbors_must_not_return_nan():
    with pytest.raises(ValueError,
                       match="no prediction-to-reference distances"):
        dtu.truncated_distance_mean(np.array([[20.], [30.]]), 20.,
                                    "prediction-to-reference")


def test_dtu_valid_truncated_mean_is_unchanged():
    assert dtu.truncated_distance_mean(np.array([[1.], [3.], [25.]]), 20.,
                                       "prediction-to-reference") == 2.


# ---- dtu_import_dependencies -------------------------------------------------------------------------------


def test_dtu_numeric_helpers_without_legacy_dataset_imports():
    code = """
import importlib.abc
import sys
import numpy as np
class RejectLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'spatialarena.eval.dtu_dataset',
                        'spatialarena.utils.colmap_manager', 'easydict'}:
            raise ImportError('legacy CLI dependency must not be imported: ' + fullname)
sys.meta_path.insert(0, RejectLegacy())
from spatialarena.eval.eval_dtu_mesh import best_fit_transform
points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
transform, rotation, translation = best_fit_transform(points, points + [1., 2., 3.])
np.testing.assert_allclose(rotation, np.eye(3), atol=1e-10)
np.testing.assert_allclose(translation, [1., 2., 3.], atol=1e-10)
"""
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, '-c', code],
                            env=env,
                            capture_output=True,
                            text=True,
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr


# ---- mvrecon_parallel_metrics ------------------------------------------------------------------------------


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('chunk_points', [1, 17, 10000])
def test_metrics_match_original_full_query(dtype, workers, chunk_points):
    rng = np.random.default_rng(341)
    pred = rng.normal(size=(103, 3)).astype(dtype)
    gt = rng.normal(size=(89, 3)).astype(dtype)
    pred = np.concatenate([pred, pred[:10]])
    pred_ref = mvrecon_eval.voxel_downsample_points(pred, .2)
    gt_ref = mvrecon_eval.voxel_downsample_points(gt, .2)
    a = cKDTree(gt_ref).query(pred_ref, k=1)[0]
    b = cKDTree(pred_ref).query(gt_ref, k=1)[0]
    threshold = .7
    precision, recall = np.mean(a < threshold), np.mean(b < threshold)
    expected = dict(accuracy=np.mean(a),
                    completion=np.mean(b),
                    overall=(np.mean(a) + np.mean(b)) / 2,
                    precision=precision,
                    recall=recall,
                    fscore=2 * precision * recall / (precision + recall))
    actual = mvrecon_eval.compute_reconstruction_metrics(
        pred,
        gt,
        threshold=threshold,
        down_sample=.2,
        workers=workers,
        chunk_points=chunk_points)
    assert actual == pytest.approx(expected, abs=1e-14)


def test_threshold_is_strict_and_empty_retains_original_contract():
    result = mvrecon_eval.compute_reconstruction_metrics(np.array([[0., 0.,
                                                                    0.]]),
                                                         np.array([[1., 0.,
                                                                    0.]]),
                                                         threshold=1.,
                                                         down_sample=None,
                                                         workers=2,
                                                         chunk_points=1)
    assert result['accuracy'] == 1
    assert result['precision'] == result['recall'] == result['fscore'] == 0
    result = mvrecon_eval.compute_reconstruction_metrics(np.empty((0, 3)),
                                                         np.ones((1, 3)),
                                                         threshold=1.,
                                                         down_sample=None)
    assert np.isinf(result['accuracy'])


def test_batches_respect_limit_and_worker_count(monkeypatch):
    calls = []

    class ObservedTree:

        def __init__(self, target):
            self.tree = cKDTree(target)

        def query(self, source, **kwargs):
            calls.append((len(source), kwargs))
            return self.tree.query(source, **kwargs)

    monkeypatch.setattr(mvrecon_eval, 'KDTree', ObservedTree)
    mvrecon_eval.compute_reconstruction_metrics(np.ones((33, 3)),
                                                np.zeros((25, 3)),
                                                threshold=1,
                                                down_sample=None,
                                                workers=2,
                                                chunk_points=10)
    assert [n for n, _ in calls] == [10, 10, 10, 3, 10, 10, 5]
    assert all(k == dict(k=1, eps=0, workers=2) for _, k in calls)


@pytest.mark.parametrize('workers,chunk', [(0, 2), (-1, 2), (1, 0)])
def test_invalid_execution_parameters(workers, chunk):
    with pytest.raises(ValueError, match='positive'):
        mvrecon_eval.compute_reconstruction_metrics(np.ones((1, 3)),
                                                    np.ones((1, 3)),
                                                    threshold=1,
                                                    down_sample=None,
                                                    workers=workers,
                                                    chunk_points=chunk)


def test_cli_and_engine_expose_execution_parameters():
    args = mvrecon_eval.parse_args([
        '--experiment_root', '/tmp/pred', '--methods', 'VGGT',
        '--metric_workers', '8', '--metric_chunk_points', '1000'
    ])
    assert (args.metric_workers, args.metric_chunk_points) == (8, 1000)
    engine = (
        Path(__file__).resolve().parents[1] /
        'spatialarena/templates/_engine/run_method_case_list.sh').read_text(
            encoding='utf-8')
    assert '--metric_workers "${MVRECON_METRIC_WORKERS:-1}"' in engine
    assert '--metric_chunk_points "${MVRECON_METRIC_CHUNK_POINTS:-262144}"' in engine


# ---- mvrecon_scoring_integration ---------------------------------------------------------------------------


def test_memory_trace_is_opt_in(monkeypatch, capsys):
    monkeypatch.delenv("FFB_SCORE_MEMORY_TRACE", raising=False)
    mvrecon_eval._memory_trace("probe", processed=3)  # pylint: disable=protected-access
    assert not capsys.readouterr().out
    monkeypatch.setenv("FFB_SCORE_MEMORY_TRACE", "1")
    mvrecon_eval._memory_trace("probe", processed=3)  # pylint: disable=protected-access
    record = json.loads(capsys.readouterr().out)
    assert record["score_memory_stage"] == "probe"
    assert record["peak_rss_kib"] > 0
    assert record["processed"] == 3


@pytest.mark.parametrize("backend", ["tsdf", "direct_backprojection"])
def test_report_distinguishes_backends(tmp_path, monkeypatch, backend):
    points = np.array([[0., 0., 1.], [1., 0., 1.]])

    def fused(*_args, **kwargs):
        assert kwargs["spatialarena_backend"] == backend
        return SimpleNamespace(
            points=points,
            gt_points=points,
            colors=None,
            fusion_metadata={"reconstruction_backend": backend})

    monkeypatch.setattr(mvrecon_eval, "fuse_prediction_in_gt_frame", fused)
    scene = BenchmarkSceneInfo(dataset="spatialarena",
                               task="MVRecon",
                               scene="test",
                               root_path=tmp_path)
    result = mvrecon_eval.evaluate_mvrecon_scene(
        scene, tmp_path,
        mvrecon_eval.MVReconEvaluationConfig(spatialarena_backend=backend))
    protocol = result["metric_protocol"]
    assert protocol["reconstruction_backend"] == backend
    for key in ["voxel_length", "sdf_trunc", "sampling_number"]:
        assert (protocol[key] is None) == (backend == "direct_backprojection")
    assert mvrecon_eval.reconstruction_backend("dtu", backend) == "tsdf"


def test_spatialarena_standard_yaml_coverage(tmp_path):
    from spatialarena.evaluator import BenchmarkEvaluator
    from spatialarena.run_config import coverage_problems, read_cases
    root = Path(__file__).resolve().parents[1]
    cases = sorted(
        line
        for line in read_cases(root /
                               "scripts/test/full_standard_depth_cases.txt")
        if line.startswith("MVRecon|spatialarena|"))
    assert len(cases) == 6
    dataset = tmp_path / "data/MVRecon/spatialarena"
    dataset.mkdir(parents=True)
    scenes = []
    for line in cases:
        name = line.split("|")[2]
        (dataset / name).mkdir()
        scenes.append({"scene_name": name, "eval_threshold": .05})
    (dataset / "metadata.json").write_text(
        json.dumps({
            "dataset_name": "spatialarena",
            "scenes": scenes
        }))
    config = tmp_path / "run.yaml"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config.write_text(
        yaml.safe_dump({
            "benchmark_root":
                str(tmp_path / "data"),
            "benchmark_eval_root":
                str(tmp_path / "scores-direct"),
            "experiment_root":
                str(tmp_path / "jobs"),
            "experiment_name":
                "backprojection",
            "evaluate_only":
                True,
            "datasets": [{
                "task": "MVRecon",
                "dataset": "spatialarena",
                "scenes": [s["scene_name"] for s in scenes]
            }],
            "models": [{
                "name": "VGGT",
                "env": {
                    "VGGT_CKPT_PATH": str(checkpoint),
                    "SPATIALARENA_MVRECON_BACKEND": "direct_backprojection",
                    "MVRECON_METRIC_WORKERS": "2",
                    "MVRECON_METRIC_CHUNK_POINTS": "17"
                }
            }],
        }))
    bench = BenchmarkEvaluator()
    bench.configure(config)
    generated = bench.generate("check")
    assert coverage_problems(bench, generated) == []
    assert len(generated) == 1
    text = generated[0][1].eval_script.read_text()
    assert "SPATIALARENA_MVRECON_BACKEND=direct_backprojection" in text
    assert "MVRECON_METRIC_CHUNK_POINTS=17" in text


# ---- spatialarena_backprojection ---------------------------------------------------------------------------


@pytest.mark.parametrize('stacked', [False, True])
def test_backprojection_w2c_intrinsics_and_invalid_pixels(stacked):
    depth = np.array([[2, 4, np.nan], [0, -1, 11]], dtype=np.float32)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, :2] = [[255, 0, 0], [0, 255, 0]]
    k = np.array([[2, 0, 1], [0, 3, 0], [0, 0, 1]], dtype=float)
    c2w = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]],
                   dtype=float)
    depths, images = [depth], [rgb]
    if stacked:
        depths, images = np.stack(depths), np.stack(images)
    points, colors = mvrecon_eval.backproject_depths_to_pointcloud(
        depths,
        images,
        np.array([k]),
        np.array([np.linalg.inv(c2w)]),
        mvrecon_eval.MVReconProtocol(max_depth=10, down_sample=.01),
        chunk_pixels=2,
    )
    np.testing.assert_allclose(points, [[10, 19, 32], [10, 20, 34]])
    np.testing.assert_allclose(colors, [[1, 0, 0], [0, 1, 0]])


def test_variable_shapes_incremental_dedup_matches_full_cloud():
    depths = [np.ones((2, 3)), np.ones((3, 4)), np.ones((1, 2))]
    images = [
        np.full((*d.shape, 3), i * 70, dtype=np.uint8)
        for i, d in enumerate(depths)
    ]
    raw = []
    for d in depths:
        y, x = np.indices(d.shape)
        raw.append(np.column_stack([x.ravel(), y.ravel(), d.ravel()]))
    expected = mvrecon_eval.voxel_downsample_points(np.concatenate(raw), .1)
    points, colors = mvrecon_eval.backproject_depths_to_pointcloud(
        depths,
        images,
        np.repeat(np.eye(3)[None], 3, axis=0),
        np.repeat(np.eye(4)[None], 3, axis=0),
        mvrecon_eval.MVReconProtocol(down_sample=.1),
        chunk_pixels=3,
    )
    np.testing.assert_array_equal(points, expected)
    np.testing.assert_array_equal(colors[:6], 0)
    metrics = mvrecon_eval.compute_reconstruction_metrics(points,
                                                          expected,
                                                          threshold=.01,
                                                          down_sample=.1)
    assert metrics['fscore'] == 1
    assert metrics['accuracy'] == 0


def test_empty_and_mismatched_inputs():
    protocol = mvrecon_eval.MVReconProtocol(down_sample=.1)
    points, colors = mvrecon_eval.backproject_depths_to_pointcloud(
        [np.array([[np.inf, 0]])],
        [np.zeros((1, 2, 3), dtype=np.uint8)],
        [np.eye(3)],
        [np.eye(4)],
        protocol,
    )
    assert points.shape == colors.shape == (0, 3)
    with pytest.raises(ValueError, match='counts'):
        mvrecon_eval.backproject_depths_to_pointcloud([], [], [np.eye(3)], [],
                                                      protocol)
    with pytest.raises(ValueError, match='positive'):
        mvrecon_eval.backproject_depths_to_pointcloud(
            [], [], [], [], mvrecon_eval.MVReconProtocol(down_sample=None))


@pytest.mark.parametrize('dataset,backend,expected', [
    ('spatialarena', 'direct_backprojection', 'direct_backprojection'),
    ('spatialarena', 'tsdf', 'tsdf'),
    ('hiroom', 'direct_backprojection', 'tsdf'),
])
def test_fusion_dispatch_preserves_other_datasets(monkeypatch, dataset, backend,
                                                  expected):
    scene = SimpleNamespace(dataset=dataset, scene='test')
    protocol = mvrecon_eval.MVReconProtocol(icp_refinement=False,
                                            down_sample=.1)
    monkeypatch.setattr(
        mvrecon_eval, 'prepare_fusion_inputs', lambda **kw: (
            [np.ones((1, 1))],
            [np.zeros((1, 1, 3), dtype=np.uint8)],
            np.eye(3)[None],
            np.eye(4)[None],
            {},
        ))
    calls = []

    def tsdf(*_args):
        calls.append('tsdf')
        return SimpleNamespace(points=np.array([[0, 0, 1]]),
                               colors=np.zeros((1, 3)))

    monkeypatch.setattr(mvrecon_eval, 'fuse_depths_to_pointcloud', tsdf)
    monkeypatch.setattr(mvrecon_eval, 'resolve_gt_geometry_path',
                        lambda _: Path('gt.ply'))
    monkeypatch.setattr(mvrecon_eval, 'load_point_cloud_array',
                        lambda *args: np.array([[0, 0, 1]]))
    result = mvrecon_eval.fuse_prediction_in_gt_frame(
        scene, Path('pred'), 'unposed', protocol, spatialarena_backend=backend)
    assert result.fusion_metadata['reconstruction_backend'] == expected
    assert calls == (['tsdf'] if expected == 'tsdf' else [])
    np.testing.assert_allclose(result.points, [[0, 0, 1]])


def test_cli_exposes_opt_in_without_changing_default():
    base = ['--experiment_root', '/tmp/pred', '--methods', 'VGGT']
    assert mvrecon_eval.parse_args(base).spatialarena_backend == 'tsdf'
    parsed = mvrecon_eval.parse_args(
        base + ['--spatialarena_backend', 'direct_backprojection'])
    assert parsed.spatialarena_backend == 'direct_backprojection'
