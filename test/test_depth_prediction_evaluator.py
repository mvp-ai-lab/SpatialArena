# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
from pathlib import Path

import numpy as np
import pytest

import spatialarena.eval.depth_prediction_evaluator as depth_eval
from spatialarena.eval.canonical_dataset import CanonicalDataset
from spatialarena.eval.depth_prediction_evaluator import (
    DepthEvaluationConfig,
    align_prediction_depth,
    compute_depth_geometry_metrics,
    evaluate_depth_scene,
    load_depth_map,
    pair_depth_files,
    pair_depth_files_by_integer_offset,
    pair_depth_files_by_nearest_timestamp,
    point_map_to_normals,
    resolve_depth_config,
)


def only(items):
    """The one element of ``items``."""
    items = list(items)
    assert len(items) == 1, items
    return items[0]


def write_metadata(root: Path, dataset_name: str, scene_root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_name":
            dataset_name,
        "root_directory":
            str(root),
        "total_images":
            1,
        "average_images_per_scene":
            1,
        "scenes": [{
            "dataset_name": dataset_name,
            "scene_name": scene_root.name,
            "root_directory": str(scene_root),
            "total_images": 1,
        }],
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_pair_depth_files_matches_common_depth_prefixes(tmp_path):
    gt = tmp_path / "gt"
    pred = tmp_path / "pred"
    gt.mkdir()
    pred.mkdir()
    gt_file = gt / "depth_map_0001.npy"
    pred_file = pred / "0001.npy"
    gt_file.write_bytes(b"gt")
    pred_file.write_bytes(b"pred")

    pairs, missing_gt, missing_pred, used_fallback = pair_depth_files(
        [gt_file],
        [pred_file],
    )

    assert pairs == [(gt_file, pred_file)]
    assert missing_gt == []
    assert missing_pred == []
    assert not used_fallback


def test_pair_depth_files_prefers_numeric_depth_over_visualization(tmp_path):
    gt = tmp_path / "gt"
    pred = tmp_path / "pred"
    gt.mkdir()
    pred.mkdir()
    gt_file = gt / "0001.png"
    pred_npy = pred / "0001.npy"
    pred_png = pred / "0001.png"
    gt_file.write_bytes(b"gt")
    pred_npy.write_bytes(b"numeric")
    pred_png.write_bytes(b"visualization")

    pairs, missing_gt, missing_pred, used_fallback = pair_depth_files(
        [gt_file],
        [pred_png, pred_npy],
    )

    assert pairs == [(gt_file, pred_npy)]
    assert missing_gt == []
    assert missing_pred == []
    assert not used_fallback


def test_pair_depth_files_normalizes_7scenes_color_and_projected_depth_stems(
        tmp_path):
    gt = tmp_path / "gt"
    pred = tmp_path / "pred"
    gt.mkdir()
    pred.mkdir()
    gt_file = gt / "seq-01_frame-000000.depth.proj.png"
    pred_file = pred / "seq-01_frame-000000.color.npy"
    gt_file.write_bytes(b"gt")
    pred_file.write_bytes(b"pred")

    pairs, missing_gt, missing_pred, used_fallback = pair_depth_files(
        [gt_file],
        [pred_file],
    )

    assert pairs == [(gt_file, pred_file)]
    assert missing_gt == []
    assert missing_pred == []
    assert not used_fallback


def test_pair_depth_files_by_nearest_timestamp_supports_bonn_partial_predictions(
        tmp_path):
    gt = tmp_path / "gt"
    pred = tmp_path / "pred"
    gt.mkdir()
    pred.mkdir()
    gt_files = [
        gt / "1548263177.85200.png",
        gt / "1548263177.88536.png",
        gt / "1548263179.69562.png",
    ]
    pred_files = [
        pred / "1548263177.69641.npy",
        pred / "1548263179.54013.npy",
    ]
    for path in gt_files + pred_files:
        path.write_bytes(b"depth")

    pairs, missing_gt, missing_pred = pair_depth_files_by_nearest_timestamp(
        gt_files,
        pred_files,
    )

    assert pairs == [(gt_files[0], pred_files[0]), (gt_files[2], pred_files[1])]
    assert missing_gt == []
    assert missing_pred == ["1548263177.88536"]


def test_pair_depth_files_by_integer_offset_supports_tnt_zero_indexed_images(
        tmp_path):
    gt = tmp_path / "gt"
    pred = tmp_path / "pred"
    gt.mkdir()
    pred.mkdir()
    gt_files = [gt / "000001.npz", gt / "000206.npz"]
    pred_files = [pred / "00000000.npy", pred / "00000205.npy"]
    for path in gt_files + pred_files:
        path.write_bytes(b"depth")

    pairs, missing_gt, missing_pred = pair_depth_files_by_integer_offset(
        gt_files,
        pred_files,
        gt_index_offset=1,
    )

    assert pairs == [(gt_files[0], pred_files[0]), (gt_files[1], pred_files[1])]
    assert missing_gt == []
    assert missing_pred == []


def test_evaluate_depth_scene_reads_standardized_artifacts(tmp_path):
    dataset_root = tmp_path / "bonn"
    scene_root = dataset_root / "scene_a"
    depth_root = scene_root / "depths"
    pred_root = tmp_path / "exp" / "VideoDepth" / "bonn" / "DA3" / "scene_a" / "depths"
    depth_root.mkdir(parents=True)
    pred_root.mkdir(parents=True)

    gt_depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.save(depth_root / "000.npy", gt_depth)
    np.save(pred_root / "000.npy", gt_depth * 2.0)
    write_metadata(dataset_root, "bonn", scene_root)

    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="VideoDepth",
        ).list_scenes())
    result = evaluate_depth_scene(
        scene_info,
        pred_root.parent,
        config=DepthEvaluationConfig(alignment="median"),
    )

    assert result["num_evaluated_frames"] == 1
    assert result["num_valid_pixels"] == 4
    assert result["metrics"]["abs_relative_difference"] == 0.0
    assert result["metrics"]["delta103_acc"] == 1.0
    assert result["metrics"]["delta103_percent"] == 100.0
    assert result["metrics"]["delta125_acc"] == 1.0


def test_standard_monodepth_protocol_uses_least_squares_alignment_and_nyu_limits(
        tmp_path):
    dataset_root = tmp_path / "marigold"
    scene_root = dataset_root / "nyuv2"
    depth_root = scene_root / "depths"
    pred_root = tmp_path / "exp" / "MonoDepth" / "marigold" / "DA3" / "nyuv2" / "depths"
    depth_root.mkdir(parents=True)
    pred_root.mkdir(parents=True)

    gt_depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred_depth = gt_depth * 0.5 + 2.0
    np.save(depth_root / "000.npy", gt_depth)
    np.save(pred_root / "000.npy", pred_depth)
    write_metadata(dataset_root, "marigold", scene_root)

    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="MonoDepth",
        ).list_scenes())
    result = evaluate_depth_scene(
        scene_info,
        pred_root.parent,
        config=DepthEvaluationConfig(alignment="standard"),
    )

    assert result["alignment"] == "least_squares"
    assert result["min_depth"] == 1e-3
    assert result["max_depth"] == 10.0
    assert result["gt_depth_scale"] == 1e-3
    assert result["post_clip_max"] == 10.0
    assert result["metrics"]["abs_relative_difference"] < 1e-6


def test_standard_monodepth_protocol_recovers_geometry_metrics_without_manifest_intrinsics(
        tmp_path):
    dataset_root = tmp_path / "marigold"
    scene_root = dataset_root / "nyuv2"
    depth_root = scene_root / "depths"
    pred_root = tmp_path / "exp" / "MonoDepth" / "marigold" / "DA3" / "nyuv2" / "depths"
    image_root = scene_root / "images"
    depth_root.mkdir(parents=True)
    pred_root.mkdir(parents=True)
    image_root.mkdir(parents=True)

    gt_depth = np.full((3, 3), 2000.0, dtype=np.float32)
    np.save(depth_root / "000.npy", gt_depth)
    np.save(pred_root / "000.npy", gt_depth.copy())
    from PIL import Image

    Image.new("RGB", (640, 480),
              color=(20, 40, 60)).save(image_root / "000.png")
    (scene_root / "materialized_full.json").write_text(
        json.dumps({
            "dataset":
                "marigold",
            "scene":
                "nyuv2",
            "num_samples":
                1,
            "records": [{
                "key": "000",
                "depth": str(depth_root / "000.npy"),
                "image": str(image_root / "000.png")
            }],
        }),
        encoding="utf-8",
    )
    write_metadata(dataset_root, "marigold", scene_root)

    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="MonoDepth",
        ).list_scenes())
    result = evaluate_depth_scene(
        scene_info,
        pred_root.parent,
        config=DepthEvaluationConfig(alignment="standard"),
    )

    assert result["metrics"]["mae_normal_deg"] == 0.0
    assert result["metrics"]["point_chamfer"] == pytest.approx(0.0, abs=1e-12)
    assert result["metrics"]["point_f1"] == 1.0


def test_standard_eth3d_monodepth_protocol_uses_alignment_downsample(tmp_path):
    dataset_root = tmp_path / "marigold"
    scene_root = dataset_root / "eth3d"
    depth_root = scene_root / "depths"
    pred_root = tmp_path / "exp" / "MonoDepth" / "marigold" / "DA3" / "eth3d" / "depths"
    depth_root.mkdir(parents=True)
    pred_root.mkdir(parents=True)

    gt_depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred_depth = gt_depth * 2.0
    np.save(depth_root / "000.npy", gt_depth)
    np.save(pred_root / "000.npy", pred_depth)
    write_metadata(dataset_root, "marigold", scene_root)

    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="MonoDepth",
        ).list_scenes())
    result = evaluate_depth_scene(
        scene_info,
        pred_root.parent,
        config=DepthEvaluationConfig(alignment="standard"),
    )

    assert result["alignment"] == "least_squares"
    assert result["alignment_max_resolution"] == 1024
    assert result["metrics"]["abs_relative_difference"] < 1e-6


def test_pi3_video_protocol_uses_lad_scale_shift_and_cubic_resize(tmp_path):
    dataset_root = tmp_path / "bonn"
    scene_root = dataset_root / "scene_a"
    write_metadata(dataset_root, "bonn", scene_root)
    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="VideoDepth",
        ).list_scenes())

    resolved = resolve_depth_config(
        scene_info,
        DepthEvaluationConfig(protocol="pi3_video"),
    )

    assert resolved.alignment == "pi3_scale_shift_lad"
    assert resolved.resize_interpolation == "cubic"
    assert resolved.max_depth == 10.0
    assert resolved.gt_depth_scale == 1.0 / 5000.0  # TUM-format PNGs (metres x 5000)
    assert resolved.post_clip_max == 10.0


def test_default_video_depth_protocol_uses_pi3_sequence_alignment(tmp_path):
    dataset_root = tmp_path / "sintel"
    scene_root = dataset_root / "alley_1"
    write_metadata(dataset_root, "sintel", scene_root)
    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="VideoDepth",
        ).list_scenes())

    resolved = resolve_depth_config(
        scene_info,
        DepthEvaluationConfig(),
    )

    assert resolved.alignment == "pi3_scale_shift_lad"
    assert resolved.resize_interpolation == "cubic"
    assert resolved.max_depth == 70.0
    assert resolved.post_clip_max == 70.0


def test_pi3_mono_protocol_uses_scale_alignment_and_cubic_resize(tmp_path):
    dataset_root = tmp_path / "marigold"
    scene_root = dataset_root / "kitti"
    write_metadata(dataset_root, "marigold", scene_root)
    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="MonoDepth",
        ).list_scenes())

    resolved = resolve_depth_config(
        scene_info,
        DepthEvaluationConfig(protocol="pi3_mono"),
    )

    assert resolved.alignment == "pi3_median_scale"
    assert resolved.resize_interpolation == "cubic"
    assert resolved.max_depth == 80.0
    assert resolved.gt_depth_scale == 1.0 / 256.0


def test_alignment_max_resolution_only_affects_scale_fit():
    gt = np.ones((4, 4), dtype=np.float32)
    pred = np.ones((4, 4), dtype=np.float32) * 2.0
    gt[:, -1] = 10.0
    pred[:, -1] = 1.0
    valid = np.ones((4, 4), dtype=bool)

    full_fit = align_prediction_depth(pred, gt, valid, "least_squares")
    downsampled_fit = align_prediction_depth(
        pred,
        gt,
        valid,
        "least_squares",
        max_resolution=2,
    )

    assert not np.allclose(full_fit, downsampled_fit)


def test_monodepth_manifest_filters_stale_depth_files(tmp_path):
    dataset_root = tmp_path / "marigold"
    scene_root = dataset_root / "nyuv2"
    depth_root = scene_root / "depths"
    pred_root = tmp_path / "exp" / "MonoDepth" / "marigold" / "DA3" / "nyuv2" / "depths"
    depth_root.mkdir(parents=True)
    pred_root.mkdir(parents=True)

    gt_depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.save(depth_root / "000.npy", gt_depth)
    np.save(depth_root / "stale.npy", gt_depth * 10.0)
    np.save(pred_root / "000.npy", gt_depth)
    np.save(pred_root / "stale.npy", gt_depth * 10.0)
    (scene_root / "materialized_full.json").write_text(
        json.dumps({
            "dataset": "marigold",
            "scene": "nyuv2",
            "num_samples": 1,
            "records": [{
                "key": "000",
                "depth": str(depth_root / "000.npy")
            }],
        }),
        encoding="utf-8",
    )
    write_metadata(dataset_root, "marigold", scene_root)

    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="MonoDepth",
        ).list_scenes())
    result = evaluate_depth_scene(
        scene_info,
        pred_root.parent,
        config=DepthEvaluationConfig(alignment="standard", gt_depth_scale=1.0),
    )

    assert result["num_gt_depth_files"] == 1
    assert result["num_evaluated_frames"] == 1
    assert result["missing_gt_predictions"] == ["stale"]


def test_monodepth_manifest_record_mask_is_applied(tmp_path):
    dataset_root = tmp_path / "marigold"
    scene_root = dataset_root / "nyuv2"
    depth_root = scene_root / "depths"
    protocol_mask_root = scene_root / "protocol_masks"
    pred_root = tmp_path / "exp" / "MonoDepth" / "marigold" / "DA3" / "nyuv2" / "depths"
    depth_root.mkdir(parents=True)
    protocol_mask_root.mkdir(parents=True)
    pred_root.mkdir(parents=True)

    gt_depth = np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32)
    pred_depth = gt_depth * 0.001
    mask = np.array([[True, False], [False, False]])
    np.save(depth_root / "000.npy", gt_depth)
    np.save(protocol_mask_root / "000_valid.npy", mask)
    np.save(pred_root / "000.npy", pred_depth)
    (scene_root / "materialized_full.json").write_text(
        json.dumps({
            "dataset":
                "marigold",
            "scene":
                "nyuv2",
            "num_samples":
                1,
            "records": [{
                "key": "000",
                "depth": str(depth_root / "000.npy"),
                "mask": str(protocol_mask_root / "000_valid.npy"),
            }],
        }),
        encoding="utf-8",
    )
    write_metadata(dataset_root, "marigold", scene_root)

    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="MonoDepth",
        ).list_scenes())
    result = evaluate_depth_scene(
        scene_info,
        pred_root.parent,
        config=DepthEvaluationConfig(alignment="standard", gt_depth_scale=1.0),
    )

    assert result["num_valid_pixels"] == 1


def test_standard_diode_protocol_uses_provided_mask_without_gt_range_filter(
        tmp_path):
    dataset_root = tmp_path / "marigold"
    scene_root = dataset_root / "diode"
    depth_root = scene_root / "depths"
    mask_root = scene_root / "masks"
    pred_root = tmp_path / "exp" / "MonoDepth" / "marigold" / "DA3" / "diode" / "depths"
    depth_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    pred_root.mkdir(parents=True)

    gt_depth = np.array([[1.0, 120.0], [3.0, 4.0]], dtype=np.float32)
    pred_depth = gt_depth.copy()
    mask = np.ones((2, 2), dtype=bool)
    np.save(depth_root / "000.npy", gt_depth)
    np.save(mask_root / "000.npy", mask)
    np.save(pred_root / "000.npy", pred_depth)
    write_metadata(dataset_root, "marigold", scene_root)

    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="MonoDepth",
        ).list_scenes())
    result = evaluate_depth_scene(
        scene_info,
        pred_root.parent,
        config=DepthEvaluationConfig(alignment="standard"),
    )

    assert not result["filter_gt_depth_range"]
    assert result["max_depth"] == 80.0
    assert result["num_valid_pixels"] == 4


def test_mvrecon_compatibility_protocol_uses_median_alignment_and_clipping(
        tmp_path):
    dataset_root = tmp_path / "eth3d"
    scene_root = dataset_root / "courtyard" / "key000001"
    depth_root = scene_root / "depths"
    pred_root = (tmp_path / "exp" / "MVRecon" / "eth3d" / "DA3" / "courtyard" /
                 "key000001" / "depths")
    depth_root.mkdir(parents=True)
    pred_root.mkdir(parents=True)

    gt_depth = np.full((2, 2), 1.0, dtype=np.float32)
    np.save(depth_root / "000.npy", gt_depth)
    np.save(pred_root / "000.npy", gt_depth * 2.0)
    write_metadata(dataset_root, "eth3d", scene_root)

    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="MVRecon",
        ).list_scenes())
    result = evaluate_depth_scene(
        scene_info,
        pred_root.parent,
        config=DepthEvaluationConfig(alignment="standard"),
    )

    assert result["alignment"] == "median"
    assert result["post_clip_min"] == 0.1
    assert result["post_clip_max"] == 100.0
    assert result["metrics"]["absrel_percent"] == 0.0


def test_depth_geometry_metrics_are_zero_for_identical_plane():
    depth = np.ones((3, 3), dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    intrinsics = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    metrics = compute_depth_geometry_metrics(
        depth,
        depth,
        valid,
        intrinsics,
        DepthEvaluationConfig(
            alignment="none",
            geometry_max_points=9,
            geometry_f1_threshold=0.01,
        ),
    )

    assert metrics["mae_normal_deg"] == 0.0
    assert metrics["point_accuracy"] == 0.0
    assert metrics["point_completion"] == 0.0
    assert metrics["point_chamfer"] == 0.0
    assert metrics["point_f1"] == 1.0


def test_surge_style_normals_use_local_neighborhoods_at_borders():
    depth = np.ones((2, 2), dtype=np.float32)
    intrinsics = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    points = depth_eval.depth_to_camera_points(depth, intrinsics)
    normals, mask = point_map_to_normals(points, np.ones_like(depth,
                                                              dtype=bool))

    assert mask.all()
    np.testing.assert_allclose(normals[mask],
                               np.array([[0.0, 0.0, 1.0]] * 4),
                               atol=1e-12)


def test_rmvd_sample_json_intrinsics_enable_depth_geometry_metrics(tmp_path):
    dataset_root = tmp_path / "dtu"
    scene_root = dataset_root / "scan1" / "key000000"
    depth_root = scene_root / "depths"
    pred_root = tmp_path / "exp" / "RMVD" / "dtu" / "DA3" / "scan1" / "key000000" / "depths"
    depth_root.mkdir(parents=True)
    pred_root.mkdir(parents=True)

    gt_depth = np.full((3, 3), 1000.0, dtype=np.float32)
    np.save(depth_root / "000000_key.npy", gt_depth)
    np.save(pred_root / "000000_key.npy", gt_depth)
    (scene_root / "sample.json").write_text(
        json.dumps({
            "dataset_name":
                "dtu",
            "task":
                "RMVD",
            "keyview_idx":
                0,
            "depth": {
                "canonical_path": "depths/000000_key.npy"
            },
            "intrinsics": [[
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]],
        }),
        encoding="utf-8",
    )
    write_metadata(dataset_root, "dtu", scene_root)

    scene_info = only(
        CanonicalDataset(
            dataset_root,
            task_name="RMVD",
        ).list_scenes())
    result = evaluate_depth_scene(
        scene_info,
        pred_root.parent,
        config=DepthEvaluationConfig(alignment="standard", gt_depth_scale=1.0),
    )

    assert result["metrics"]["mae_normal_deg"] == 0.0
    assert result["metrics"]["point_chamfer"] == 0.0
    assert result["metrics"]["point_f1"] == 1.0


def test_load_depth_map_reads_sintel_dpt(tmp_path):
    path = tmp_path / "frame_0001.dpt"
    depth = np.array([[1.0, 2.0, 3.0], [4.0, 1.0e11, 6.0]], dtype=np.float32)
    with path.open("wb") as file:
        np.array([202021.25], dtype=np.float32).tofile(file)
        np.array([depth.shape[1]], dtype=np.int32).tofile(file)
        np.array([depth.shape[0]], dtype=np.int32).tofile(file)
        depth.tofile(file)

    loaded = load_depth_map(path)

    assert np.isnan(loaded[1, 1])
    np.testing.assert_array_equal(loaded[np.isfinite(loaded)],
                                  [1.0, 2.0, 3.0, 4.0, 6.0])


def test_load_depth_map_reads_eth3d_raw_float_jpg(tmp_path, monkeypatch):
    depth = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    path = tmp_path / "DSC_0001.JPG"
    depth.tofile(path)
    monkeypatch.setattr(depth_eval, "ETH3D_RAW_DEPTH_SHAPE", depth.shape)
    monkeypatch.setattr(depth_eval, "ETH3D_RAW_DEPTH_BYTES", depth.nbytes)

    loaded = load_depth_map(path)

    np.testing.assert_array_equal(loaded, depth)


def test_error_artifacts_record_alignment_and_absrel_maps(tmp_path):
    """Per-frame error maps are |aligned - gt| / gt on valid pixels (NaN elsewhere), downsampled to max_side."""
    gt = np.full((40, 80), 2.0, dtype=np.float32)
    gt[:, :8] = 0.0  # invalid GT column
    prediction = gt / 4.0 + 0.5  # recovered exactly by a scale/shift fit
    prediction[10, 20] = 1.5  # one wrong pixel: aligned depth 4 vs GT 2
    valid = gt > 0
    pair = depth_eval.DepthPairData(gt_path=tmp_path / "frame_0007.png",
                                    pred_path=tmp_path / "frame_0007.npy",
                                    ground_truth=gt,
                                    prediction=prediction,
                                    valid_mask=valid)
    writer = depth_eval.DepthErrorArtifactWriter(root=tmp_path / "out",
                                                 max_side=40)
    config = depth_eval.DepthEvaluationConfig(alignment="least_squares",
                                              compute_geometry_metrics=False)
    metrics = depth_eval.evaluate_single_depth_pair(pair, config, writer)

    error = np.load(tmp_path / "out" / "error_maps" / "frame_0007.npy")
    assert error.dtype == np.float16 and error.shape == (20, 40)
    assert np.isnan(error[:, :4]).all() and np.nanmin(error) == 0.0
    assert (tmp_path / "out" / "error_visualizations" /
            "frame_0007.png").is_file()
    record = writer.records[0]
    assert record["gt"] == "frame_0007.png" and record["metrics"] == metrics
    assert record["num_valid_pixels"] == int(valid.sum())
    assert not (tmp_path / "out" / "aligned_depths").exists()


def test_bonn_video_depth_reads_tum_format_depth(tmp_path):
    """Bonn depth PNGs are metres x 5000 (TUM format): a 3 m pixel stored as 15000 must read as 3 m, under the cap."""
    dataset_root = tmp_path / "bonn"
    scene_root = dataset_root / "scene_a"
    (scene_root / "depths").mkdir(parents=True)
    write_metadata(dataset_root, "bonn", scene_root)
    scene_info = only(
        CanonicalDataset(dataset_root, task_name="VideoDepth").list_scenes())
    config = resolve_depth_config(scene_info, DepthEvaluationConfig())
    assert config.gt_depth_scale == depth_eval.BONN_DEPTH_SCALE == 1.0 / 5000.0
    assert 15000 * config.gt_depth_scale == 3.0 < config.max_depth
