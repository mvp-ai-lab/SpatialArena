# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
from pathlib import Path

import numpy as np
import pytest

from spatialarena import slam_protocol
from spatialarena.eval.canonical_dataset import CanonicalDataset
from spatialarena.eval.dataset_registry import describe_benchmark_scenes, find_dataset_spec
from spatialarena.eval.depth_prediction_evaluator import (
    DepthEvaluationConfig,
    evaluate_depth_scene,
    resolve_depth_config,
)
from spatialarena.eval.mvrecon_evaluator import protocol_for_dataset
from spatialarena.evaluator import BenchmarkEvaluator
from spatialarena.spatialarena import SPATIALARENA_ROOT_ENV, TASKS



def only(items):
    """The one element of ``items``."""
    items = list(items)
    assert len(items) == 1, items
    return items[0]

def write_track(track_root: Path, task: str, scenes: list[dict]) -> None:
    """A SpatialArena track as materialize_spatialarena.py writes it: scenes directly under the track folder."""
    for scene in scenes:
        for sub in ("images", "depths", "sparse/0"):
            (track_root / scene["scene_name"] / sub).mkdir(parents=True, exist_ok=True)
    (track_root / "metadata.json").write_text(json.dumps(
        {"dataset_name": "spatialarena", "task": task, "root_directory": ".", "scenes": scenes}), encoding="utf-8")


def test_every_task_resolves_to_the_benchmark_root_or_the_spatialarena_root(tmp_path, monkeypatch):
    monkeypatch.delenv(SPATIALARENA_ROOT_ENV, raising=False)
    for task in TASKS:
        assert find_dataset_spec(task, "spatialarena").data_path(tmp_path) == tmp_path / task / "spatialarena"
    monkeypatch.setenv(SPATIALARENA_ROOT_ENV, str(tmp_path / "arena"))
    for task in TASKS:
        assert find_dataset_spec(task, "spatialarena").data_path(tmp_path) == tmp_path / "arena" / task
    assert slam_protocol.input_root(tmp_path, "spatialarena") == tmp_path / "arena" / "SLAM"
    assert slam_protocol.input_root(tmp_path, "vbr") == tmp_path / "SLAM" / "vbr"


def test_slam_sequences_run_only_in_their_own_range(tmp_path, monkeypatch):
    monkeypatch.setenv(SPATIALARENA_ROOT_ENV, str(tmp_path))
    write_track(tmp_path / "SLAM", "SLAM", [
        {"scene_name": "Short1", "total_images": 300, "range": "short"},
        {"scene_name": "Large1", "total_images": 3000, "range": "large"},
    ])
    record = only(describe_benchmark_scenes(tmp_path, tasks=["SLAM"], datasets=["spatialarena"]))
    assert {scene["scene"]: scene["slam_range"] for scene in record["scenes"]} == {"Short1": "short",
                                                                                    "Large1": "large"}
    bench = BenchmarkEvaluator(benchmark_root=tmp_path)
    for slam_range, expected in (("short", ["Short1"]), ("large", ["Large1"]), ("long", [])):
        # pylint: disable-next=protected-access
        entries = bench._resolve_case_entries({"task": "SLAM", "dataset": "spatialarena", "slam_range": slam_range})
        assert [entry.scene for entry in entries] == expected
    assert slam_protocol.range_stride("short", "spatialarena") == 3
    assert slam_protocol.range_stride("long", "spatialarena") == 1
    assert slam_protocol.dataset_fps("spatialarena") == 5.0


def test_videodepth_leaves_reviewed_frames_out_of_the_metrics(tmp_path):
    track = tmp_path / "VideoDepth"
    scene = track / "Clip_clip00_cam1"
    write_track(track, "VideoDepth", [{"scene_name": scene.name, "total_images": 3}])
    pred = tmp_path / "pred" / scene.name / "depths"
    pred.mkdir(parents=True)
    rng = np.random.default_rng(0)
    manifest = []
    for index in range(3):
        depth = rng.uniform(1.0, 20.0, size=(8, 8)).astype(np.float32)
        excluded = index == 1
        if not excluded:
            np.save(scene / "depths" / f"{index:06d}.npy", depth)
        np.save(pred / f"{index:06d}.npy", depth if not excluded else depth * 7.0)
        manifest.append({"frame": index, "rgb": f"images/{index:06d}.jpg",
                         "depth": None if excluded else f"depths/{index:06d}.npy", **({"gt_excluded": True}
                                                                                       if excluded else {})})
    (scene / "frame_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    scene_info = only(CanonicalDataset(track, task_name="VideoDepth").list_scenes())
    assert scene_info.dataset == "spatialarena"
    result = evaluate_depth_scene(scene_info, pred.parent, config=DepthEvaluationConfig(alignment="median"))
    assert result["gt_excluded_frames"] == ["000001"] and result["missing_gt_predictions"] == []
    assert result["metrics"]["abs_relative_difference"] == pytest.approx(0.0, abs=1e-6)
    resolved = resolve_depth_config(scene_info, DepthEvaluationConfig())
    assert (resolved.min_depth, resolved.max_depth, resolved.gt_depth_scale) == (0.1, 100.0, 1.0)


def test_mvrecon_thresholds_follow_each_scenes_setting():
    indoor = protocol_for_dataset("spatialarena", {"setting": "indoor", "eval_threshold": 0.05})
    outdoor = protocol_for_dataset("spatialarena", {"setting": "outdoor", "eval_threshold": 0.25})
    assert indoor.eval_threshold == pytest.approx(0.05) and indoor.voxel_length == pytest.approx(4.0 / 512.0)
    assert outdoor.eval_threshold == pytest.approx(0.25) and outdoor.voxel_length == pytest.approx(4.0 / 512.0 * 5)
    assert outdoor.max_depth == 100.0 and not outdoor.icp_refinement
    # Indoor scenes fuse up to 20 m (100 m on the 7.8 mm grid ran out of 330 GB), with the same five-voxel truncation.
    assert indoor.max_depth == pytest.approx(20.0)
    assert indoor.sdf_trunc == pytest.approx(0.04) and outdoor.sdf_trunc == pytest.approx(0.2)


def test_mvrecon_fusion_override_for_a_large_indoor_scene():
    """StPancrasStatue1 keeps the indoor threshold but is fused to 100 m on a 2 cm grid (its geometry reaches 100 m)."""
    scene = protocol_for_dataset("spatialarena", {"scene_name": "StPancrasStatue1", "eval_threshold": 0.05})
    assert scene.eval_threshold == pytest.approx(0.05) and scene.down_sample == pytest.approx(4.0 / 512.0)
    assert scene.voxel_length == pytest.approx(0.02) and scene.max_depth == pytest.approx(100.0)
    assert scene.sdf_trunc == pytest.approx(0.02 * 5.12)
    other = protocol_for_dataset("spatialarena", {"scene_name": "NationalScotlandMuseum7", "eval_threshold": 0.05})
    assert other.voxel_length == pytest.approx(4.0 / 512.0) and other.max_depth == pytest.approx(20.0)


def test_videodepth_is_uncapped_and_mvrecon_and_sfm_capped_at_300_in_the_engine():
    """The engine's per-case frame cap comes from the registry: SpatialArena's own, other datasets the suite's."""
    import re  # pylint: disable=import-outside-toplevel
    import subprocess  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    repo = Path(__file__).resolve().parents[1]
    engine = (repo / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh").read_text()
    helper = re.search(r"^dataset_frame_cap\(\) \{.*?^\}$", engine, flags=re.S | re.M).group(0)
    caps = {}
    for task, dataset in (("VideoDepth", "spatialarena"), ("SfM", "spatialarena"), ("MVRecon", "spatialarena"),
                          ("VideoDepth", "bonn")):
        result = subprocess.run(["bash", "-c", f'{helper}\ndataset_frame_cap "$0" "$1"', task, dataset],
                                env={"PROJECT_ROOT": str(repo), "PYTHON": sys.executable, "PATH": "/usr/bin:/bin"},
                                capture_output=True, text=True, check=True)
        caps[(task, dataset)] = result.stdout.strip()
    assert caps == {("VideoDepth", "spatialarena"): "0", ("SfM", "spatialarena"): "300",
                    ("MVRecon", "spatialarena"): "300", ("VideoDepth", "bonn"): ""}


def _sparse_lidar_png(path: Path, depth_m: np.ndarray) -> None:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    lidar = np.zeros(depth_m.shape, dtype=np.uint16)
    lidar[::2, ::2] = np.rint(depth_m[::2, ::2] * 100.0).astype(np.uint16)  # centimetres, a sparse pattern
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lidar).save(path)


def test_videodepth_is_scored_on_both_the_dense_and_the_sparse_lidar_depth(tmp_path):
    track = tmp_path / "VideoDepth"
    scene = track / "Clip_clip00_cam1"
    write_track(track, "VideoDepth", [{"scene_name": scene.name, "total_images": 3}])
    pred = tmp_path / "pred" / scene.name / "depths"
    pred.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for index in range(3):
        depth = rng.uniform(1.0, 20.0, size=(8, 8)).astype(np.float32)
        np.save(scene / "depths" / f"{index:06d}.npy", depth)
        _sparse_lidar_png(scene / "lidar_depths" / f"{index:06d}.png", depth * 1.1)  # LiDAR disagrees with GGGS
        np.save(pred / f"{index:06d}.npy", depth)
    scene_info = only(CanonicalDataset(track, task_name="VideoDepth").list_scenes())
    result = evaluate_depth_scene(scene_info, pred.parent, config=DepthEvaluationConfig(alignment="none"))
    assert result["metrics"]["abs_relative_difference"] == pytest.approx(0.0, abs=1e-6)
    assert result["metrics"]["lidar_abs_relative_difference"] == pytest.approx(1 / 11, abs=2e-3)
    assert result["lidar_gt"]["num_evaluated_frames"] == 3 and result["lidar_gt"]["num_valid_pixels"] == 3 * 16
    assert result["lidar_gt"]["gt_depth_scale"] == pytest.approx(0.01)


def test_monodepth_lidar_gt_comes_from_the_records_and_aggregates_only_over_its_scenes(tmp_path):
    from spatialarena.eval.depth_prediction_evaluator import aggregate_depth_results  # pylint: disable=C0415

    track = tmp_path / "MonoDepth"
    scene = track / "Mono"
    write_track(track, "MonoDepth", [{"scene_name": "Mono", "total_images": 2, "min_depth": 0.1, "max_depth": 100.0,
                                      "depth_scale": 1.0}])
    pred = tmp_path / "pred" / "Mono" / "depths"
    pred.mkdir(parents=True)
    records = []
    for key in ("cam0__img0_000001", "cam1__img1_000002"):
        depth = np.full((8, 8), 4.0, dtype=np.float32)
        np.save(scene / "depths" / f"{key}.npy", depth)
        _sparse_lidar_png(scene / "lidar_depths" / f"{key}.png", depth)
        np.save(pred / f"{key}.npy", depth)
        records.append({"key": key, "image": f"images/{key}.jpg", "depth": f"depths/{key}.npy",
                        "lidar_depth": f"lidar_depths/{key}.png", "lidar_depth_scale": 100.0})
    (scene / "materialized_full.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    scene_info = only(CanonicalDataset(track, task_name="MonoDepth").list_scenes())
    result = evaluate_depth_scene(scene_info, pred.parent, config=DepthEvaluationConfig(alignment="none"))
    assert result["lidar_gt"]["num_gt_depth_files"] == 2
    assert result["metrics"]["lidar_abs_relative_difference"] == pytest.approx(0.0, abs=1e-6)
    other = {"metrics": {"abs_relative_difference": 0.5}, "num_metric_units": 2}
    aggregate = aggregate_depth_results([result, other])
    assert aggregate["lidar_abs_relative_difference"] == pytest.approx(0.0, abs=1e-6)  # SpatialArena's scene only
    assert aggregate["abs_relative_difference"] == pytest.approx(0.25, abs=1e-6)


def test_mvrecon_prediction_is_cropped_to_the_gt_clouds_reach_from_the_cameras():
    """SpatialArena's GT cloud covers a few metres around the cameras: predicted points beyond its reach are dropped."""
    from spatialarena.eval.mvrecon_evaluator import gt_camera_reach_mask  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(0)
    cameras = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    direction = rng.normal(size=(5000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    gt = cameras[rng.integers(0, 2, 5000)] + direction * rng.uniform(0.5, 4.0, (5000, 1))  # within 4 m of a camera
    near = cameras[[0]] + np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    far = cameras[[1]] + np.array([[0.0, 0.0, 40.0], [0.0, 80.0, 0.0]])  # e.g. sky / distant facades
    keep, reach = gt_camera_reach_mask(np.concatenate([near, far]), gt, cameras)
    assert 3.9 < reach <= 4.0
    assert keep.tolist() == [True, True, False, False]
