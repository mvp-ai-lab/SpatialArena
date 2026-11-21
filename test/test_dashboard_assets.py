# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Dashboard and viewer assets: dataset gallery, evaluator viewer assets, demo replays, release export and the viewer
backfill.

benchmark_dashboard_gallery: Tests for the benchmark dashboard's dataset gallery builder and catalog.

viewer_assets: Viewer assets written by the evaluators (spatialarena/eval/viewer_assets.py) and the dashboard's index
of them.

demo_assets: Demo-mode replay assets (spatialarena/eval/demo_assets.py).

export_release_artifacts: The Artifact Viewer's release export (app/webpage/python/export_release_artifacts.py).

build_viewer_assets: The depth viewer backfill re-renders a case with the protocol it was scored with, not the
evaluator default.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image
from app.webpage.python.build_dataset_gallery import (
    THUMBNAIL_SIZE,
    candidate_images,
    center_crop_thumbnail,
    tasks_for_dataset,
)
from app.webpage.python.dataset_catalog import DATASET_CATALOG, DATASET_CATALOG_BY_KEY
from app.webpage.python.report_discovery import read_case_lists
import json
from types import SimpleNamespace
import numpy as np
from app.webpage.python.viewer_index import build_viewer_index
from spatialarena.eval import viewer_assets
from spatialarena.eval import demo_assets
import pytest
from app.webpage.python import export_release_artifacts as export
import os
from app.webpage.python import build_viewer_assets
from spatialarena.eval import depth_prediction_evaluator


# ---- benchmark_dashboard_gallery ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_LISTS = (
    REPO_ROOT / "scripts" / "test" / "full_standard_depth_cases.txt",
    REPO_ROOT / "scripts" / "test" / "full_standard_sfm_cases.txt",
)


def test_catalog_covers_every_full_standard_dataset():
    datasets = {dataset for _task, dataset, _scene in read_case_lists(CASE_LISTS)}

    assert datasets <= set(DATASET_CATALOG_BY_KEY)
    assert len(DATASET_CATALOG_BY_KEY) == len(DATASET_CATALOG)
    for card in DATASET_CATALOG:
        assert card.intro and card.url.startswith("http") and card.scenes > 0


def test_tasks_for_dataset_orders_tasks():
    assert tasks_for_dataset("7scenes") == ["MVRecon", "SLAM"]
    assert tasks_for_dataset("marigold") == ["MonoDepth"]
    assert tasks_for_dataset("not_a_dataset") == []


def test_candidate_images_skips_non_rgb_and_prefers_hint(tmp_path):
    for name in ("0001.png", "0001_depth.png", "0002_rgb.png", "mask_0003.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"")

    assert [path.name for path in candidate_images(tmp_path)] == ["0001.png", "0002_rgb.png"]
    assert [path.name for path in candidate_images(tmp_path, hint="rgb")] == ["0002_rgb.png"]
    assert not candidate_images(tmp_path / "missing")


def test_center_crop_thumbnail_writes_fixed_size_jpeg(tmp_path):
    source = tmp_path / "tall.png"
    Image.new("RGB", (300, 900), (10, 200, 30)).save(source)
    destination = tmp_path / "out" / "thumb.jpg"

    center_crop_thumbnail(source, destination)

    with Image.open(destination) as thumbnail:
        assert thumbnail.format == "JPEG"
        assert thumbnail.size == THUMBNAIL_SIZE


# ---- viewer_assets -----------------------------------------------------------------------------------------

def test_depth_writer_uses_one_scale_and_writes_shared_frames_once(tmp_path):
    """Two methods of a scene share the GT scale; GT frames are written once, error is grey where GT is invalid."""
    gt = np.linspace(1.0, 5.0, 40 * 60, dtype=np.float32).reshape(40, 60)
    valid = np.ones_like(gt, dtype=bool)
    valid[:, :10] = False
    pair = SimpleNamespace(gt_path=Path("frame_0001.png"), ground_truth=gt, valid_mask=valid)
    scene_info = SimpleNamespace(task="VideoDepth", dataset="toy", scene="s0")
    scene_dir = tmp_path / "scenes" / "VideoDepth" / "toy" / "s0"
    for method, aligned in (("A", gt), ("B", gt * 1.5)):
        writer = viewer_assets.DepthViewerWriter(case_dir=tmp_path / method / "viewer", scene_dir=scene_dir,
                                                 depth_range=(1.0, 5.0), max_side=30)
        writer.write(pair, aligned, 1.0, 0.0, {"abs_relative_difference": 0.1, "delta125_acc": 0.9})
        writer.finish(scene_info, {"abs_relative_difference": 0.1})
    frames = json.loads((tmp_path / "A" / "viewer" / "frames.json").read_text())
    assert frames["frames"][0]["frame"] == "frame_0001" and frames["size"] == {"height": 20, "width": 30}
    assert frames["depth_range"] == [1.0, 5.0]
    assert json.loads((scene_dir / "scene.json").read_text())["depth_range"] == [1.0, 5.0]
    error_a = np.asarray(Image.open(tmp_path / "A" / "viewer" / "error" / "frame_0001.webp"))
    assert error_a.shape == (20, 30, 3)
    assert np.allclose(error_a[:, :4].astype(float), 128, atol=6)  # no valid GT: grey
    depth_a = np.asarray(Image.open(tmp_path / "A" / "viewer" / "depth" / "frame_0001.webp")).astype(float)
    depth_b = np.asarray(Image.open(tmp_path / "B" / "viewer" / "depth" / "frame_0001.webp")).astype(float)
    assert np.abs(depth_a - depth_b).mean() > 10  # same scale, so a 1.5x deeper prediction looks different
    assert (scene_dir / "gt" / "frame_0001.webp").is_file()


def test_viewer_index_lists_cases_by_scene_with_relative_paths(tmp_path):
    """The collector's index groups each kind's cases by scene; SLAM keys carry the range."""
    reports = tmp_path / "reports"
    slam_case = reports / "benchmark_full_slam_reports" / "full-slam-short" / "M1" / "SLAM" / "tum" / "s1"
    (slam_case / "viewer").mkdir(parents=True)
    (slam_case / "viewer" / "trajectory.json").write_text("{}")
    (slam_case / "eval_slam.json").write_text(json.dumps({"scene_results": [{"metrics": {"ate_rmse": 0.2}}]}))
    depth_case = reports / "benchmark_full_reports" / "full-depth" / "M2" / "VideoDepth" / "bonn" / "s2"
    (depth_case / "viewer").mkdir(parents=True)
    (depth_case / "viewer" / "frames.json").write_text(json.dumps(
        {"frames": [{"frame": "f0"}], "metrics": {"abs_relative_difference": 0.05}}))
    scene_dir = tmp_path / "viewer_assets" / "scenes" / "VideoDepth" / "bonn" / "s2"
    scene_dir.mkdir(parents=True)
    (scene_dir / "scene.json").write_text(json.dumps({"depth_range": [0.5, 4.0]}))
    recon_case = reports / "benchmark_full_reports" / "full-depth" / "M3" / "MVRecon" / "dtu" / "scan1"
    (recon_case / "viewer").mkdir(parents=True)
    (recon_case / "viewer" / "cloud.json").write_text(json.dumps({"points": 10, "metrics": {"fscore": 0.7}}))

    assert build_viewer_index(tmp_path) == {"trajectories": 1, "depth": 1, "recon": 1}
    index_dir = tmp_path / "viewer_assets" / "index"
    trajectories = json.loads((index_dir / "trajectories.json").read_text())["scenes"]
    assert trajectories[0]["dataset_key"] == "tum@short"
    assert trajectories[0]["methods"][0] == {
        "method": "M1", "path": str(slam_case.relative_to(tmp_path) / "viewer" / "trajectory.json"),
        "ate_rmse": 0.2, "completeness": None,
    }
    depth = json.loads((index_dir / "depth.json").read_text())["scenes"][0]
    assert depth["depth_range"] == [0.5, 4.0] and depth["methods"][0]["absrel"] == 0.05
    assert depth["scene_path"] == "viewer_assets/scenes/VideoDepth/bonn/s2"
    recon = json.loads((index_dir / "recon.json").read_text())["scenes"][0]
    assert recon["methods"][0]["fscore"] == 0.7


def test_gt_point_colors_come_from_the_nearest_vertex_of_the_colored_scan(tmp_path):
    """A colored GT cloud gives each GT sample its nearest vertex's RGB; an uncolored one gives None."""
    import open3d as o3d  # pylint: disable=import-outside-toplevel

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float64))
    colored = tmp_path / "colored.ply"
    o3d.io.write_point_cloud(str(colored), cloud)
    samples = np.array([[0.4, 0, 0], [9.0, 0.5, 0]], dtype=np.float32)
    np.testing.assert_array_equal(viewer_assets.gt_point_colors(colored, samples), [[255, 0, 0], [0, 0, 255]])

    plain = tmp_path / "plain.ply"
    o3d.io.write_point_cloud(str(plain), o3d.geometry.PointCloud(cloud.points))
    assert viewer_assets.gt_point_colors(plain, samples) is None


def test_cameras_payload_keeps_sizes_intrinsics_and_both_poses():
    """Each fused view: (width, height), (fx, fy, cx, cy), and [x, y, z, qw, qx, qy, qz] for prediction and GT."""
    turn = np.eye(4)
    turn[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 degrees about z
    turn[:3, 3] = [1, 2, 3]
    payload = viewer_assets.cameras_payload({
        "pred_c2w": np.stack([np.eye(4), turn]), "gt_c2w": np.stack([np.eye(4), np.eye(4)]),
        "intrinsics": np.array([[[500, 0, 320], [0, 510, 240], [0, 0, 1]]] * 2, dtype=float),
        "sizes": np.array([[640, 480], [640, 480]]),
    })
    assert payload["sizes"] == [[640, 480], [640, 480]]
    assert payload["intrinsics"][0] == [500, 510, 320, 240]
    assert payload["pred"][0] == [0, 0, 0, 1, 0, 0, 0]
    np.testing.assert_allclose(payload["pred"][1], [1, 2, 3, np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-4)
    assert payload["gt"][1] == [0, 0, 0, 1, 0, 0, 0]


def test_point_confidence_averages_the_maps_of_the_views_that_see_a_point(tmp_path):
    """A point takes the confidence at its pixel in each view that sees it; occluded or unseen points get NaN."""
    (tmp_path / "depths").mkdir()
    (tmp_path / "confidences").mkdir()
    for name in ("a", "b"):
        np.save(tmp_path / "depths" / f"{name}.npy", np.ones((8, 8), dtype=np.float32))
    confidence = np.full((8, 8), 4.0, dtype=np.float16)  # half the 16 x 16 fused size
    confidence[:, 4:] = 8.0
    np.save(tmp_path / "confidences" / "a.npy", confidence)  # view b has no confidence map
    intrinsic = np.array([[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]])
    cameras = {"pred_c2w": np.stack([np.eye(4), np.eye(4)]), "intrinsics": np.stack([intrinsic, intrinsic]),
               "sizes": [[16, 16], [16, 16]],
               "depth_paths": [tmp_path / "depths" / "a.npy", tmp_path / "depths" / "b.npy"]}
    points = np.array([[-0.2, 0.0, 1.0],  # left half: 4
                       [0.3, 0.0, 1.0],  # right half: 8
                       [-0.4, 0.0, 2.0],  # same pixel as the first, behind it: occluded
                       [0.0, 0.0, -1.0]])  # behind the camera
    values = viewer_assets.point_confidence(points, cameras)
    np.testing.assert_allclose(values[:2], [4.0, 8.0])
    assert np.all(np.isnan(values[2:]))
    assert viewer_assets.point_confidence(points, {**cameras, "depth_paths": [None, None]}) is None


def test_cameras_from_payload_inverts_the_saved_cameras():
    """cameras.json's poses and intrinsics come back as the matrices they were saved from."""
    turn = np.eye(4)
    turn[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    turn[:3, 3] = [1, 2, 3]
    intrinsics = np.array([[[500, 0, 320], [0, 510, 240], [0, 0, 1]]] * 2, dtype=float)
    payload = viewer_assets.cameras_payload({"pred_c2w": np.stack([np.eye(4), turn]), "intrinsics": intrinsics,
                                             "sizes": np.array([[640, 480], [640, 480]])})
    cameras = viewer_assets.cameras_from_payload(payload, ["a", "b"])
    np.testing.assert_allclose(cameras["pred_c2w"][1], turn, atol=1e-4)
    np.testing.assert_allclose(cameras["intrinsics"], intrinsics)
    assert cameras["depth_paths"] == ["a", "b"] and cameras["sizes"] == [[640, 480], [640, 480]]


# ---- demo_assets -------------------------------------------------------------------------------------------

def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _frames(tmp_path: Path, count: int, posed: bool = True) -> list[demo_assets.DemoFrame]:
    intrinsic = np.array([[40.0, 0.0, 32.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]])
    frames = []
    for index in range(count):
        image = tmp_path / f"{index:03d}.png"
        Image.fromarray(np.full((48, 64, 3), (10 * index, 100, 200), dtype=np.uint8)).save(image)
        c2w = np.eye(4)
        c2w[:3, :3] = _rotation_y(0.1 * index)
        c2w[:3, 3] = [0.2 * index, 0.0, 0.0]
        frames.append(demo_assets.DemoFrame(
            name=image.name, image_path=image, load_depth=lambda: np.full((48, 64), 2.0, dtype=np.float32),
            intrinsic=intrinsic, size=(96, 128) if index == 0 else (48, 64), c2w=c2w if posed else None,
        ))
    return frames


def test_write_demo_places_every_frame_in_its_camera(tmp_path):
    """Each frame's slice of the point buffer is its depth unprojected by its own camera; images per frame."""
    out = tmp_path / "demo"
    frames = _frames(tmp_path, 3)
    payload = demo_assets.write_demo(out, frames, metadata={"method": "M"}, posed=True,
                                     intrinsics_source="predicted")
    demo = json.loads((out / demo_assets.DEMO_FILE).read_text())
    assert demo == payload and demo["method"] == "M" and demo["posed"] and len(demo["frames"]) == 3
    points = np.fromfile(out / demo_assets.POINTS_FILE, dtype="<u2").reshape(-1, 3).astype(np.float64)
    low, high = (np.array(bound) for bound in demo["bounds"])
    points = low + points / 65535.0 * (high - low)
    colors = np.fromfile(out / demo_assets.COLORS_FILE, dtype=np.uint8).reshape(-1, 3)
    assert points.shape == colors.shape and demo["points"] == points.shape[0]
    for index, entry in enumerate(demo["frames"]):
        assert entry["start"] == sum(item["count"] for item in demo["frames"][:index]) and entry["count"] > 0
        c2w = np.array(entry["c2w"]).reshape(4, 4)
        local = (points[entry["start"]:entry["start"] + entry["count"]] - c2w[:3, 3]) @ c2w[:3, :3]
        assert np.allclose(local[:, 2], 2.0, atol=1e-3 * (high - low).max() + 1e-6)  # depth 2 in its own camera
        assert np.all(colors[entry["start"]:entry["start"] + entry["count"], 0] == 10 * index)
        assert (out / "rgb" / f"{index:04d}.webp").is_file() and (out / "depth" / f"{index:04d}.webp").is_file()
    # Frame 0's intrinsics were given for a 2x larger image: scaled to the depth it is unprojected with.
    assert demo["frames"][0]["intrinsics"] == [20.0, 20.0, 16.0, 12.0]
    assert demo["frames"][1]["intrinsics"] == [40.0, 40.0, 32.0, 24.0]
    assert np.allclose(demo["up"], [0.0, -1.0, 0.0])


def test_unposed_demo_stays_in_the_camera_frame(tmp_path):
    """A VideoDepth replay has identity cameras and one up direction."""
    demo = demo_assets.write_demo(tmp_path / "demo", _frames(tmp_path, 2, posed=False), metadata={}, posed=False,
                                  intrinsics_source="nominal")
    assert not demo["posed"] and demo["frames"][1]["c2w"] == np.eye(4).reshape(-1).tolist()
    assert demo["up"] == [0.0, -1.0, 0.0]


def test_edge_mask_drops_depth_discontinuities():
    """Pixels next to a depth jump (flying points) and invalid depth are dropped; smooth surfaces kept."""
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    depth[:, 5:] = 4.0
    depth[0, 0] = 0.0
    mask = demo_assets.edge_free_mask(depth)
    assert not mask[:, 4:6].any() and mask[5, 2] and mask[5, 8] and not mask[0, 0]


def test_frame_selection_and_nominal_intrinsics():
    """At most MAX_FRAMES evenly spaced frames, first and last kept; nominal intrinsics scale with the image."""
    picked = demo_assets.select_frames(1000, 240)
    assert len(picked) == 240 and picked[0] == 0 and picked[-1] == 999
    assert list(demo_assets.select_frames(5, 240)) == [0, 1, 2, 3, 4]
    bonn = demo_assets.nominal_intrinsics("bonn", (240, 320))
    assert np.isclose(bonn[0, 0], 542.822841 / 2) and np.isclose(bonn[1, 2], 237.756098 / 2)
    other = demo_assets.nominal_intrinsics("unknown", (100, 200))
    assert np.isclose(other[0, 0], 100.0 / np.tan(np.radians(30.0)))


def test_demo_points_carry_their_frames_confidence(tmp_path):
    """confidence.f32 holds each point's confidence (NaN for a frame without a map); no file when none has one."""
    frames = _frames(tmp_path, 2)
    frames[0] = demo_assets.DemoFrame(**{**frames[0].__dict__,
                                          "load_confidence": lambda: np.full((96, 128), 7.0, dtype=np.float16)})
    demo = demo_assets.write_demo(tmp_path / "demo", frames, metadata={}, posed=True, intrinsics_source="predicted")
    confidence = np.fromfile(tmp_path / "demo" / demo_assets.CONFIDENCE_FILE, dtype="<f4")
    first, second = demo["frames"]
    assert demo["confidence"] and confidence.shape == (demo["points"],)
    assert np.all(confidence[:first["count"]] == 7.0)
    assert np.all(np.isnan(confidence[second["start"]:second["start"] + second["count"]]))

    plain = demo_assets.write_demo(tmp_path / "plain", _frames(tmp_path, 2), metadata={}, posed=True,
                                   intrinsics_source="predicted")
    assert not plain["confidence"] and not (tmp_path / "plain" / demo_assets.CONFIDENCE_FILE).exists()


# ---- export_release_artifacts ------------------------------------------------------------------------------

def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _webp(path: Path, width: int = 64, height: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (90, 120, 200)).save(path, "WEBP")


def _eval_root(root: Path) -> None:
    """One MonoDepth scene (methods A, B, C, VLM; 10 frames), one MVRecon scene (A, B; 1,000 points), one SLAM scene."""
    frames = [f"f{i:02d}" for i in range(10)]
    depth_methods = []
    for method in ("A", "B", "C", "VLM"):
        case = f"viewer_assets/imported/{method}/MonoDepth/ds/s0/viewer"
        _write(root / case / "frames.json", {"frames": [{"frame": f, "absrel": 0.1} for f in frames], "metrics": {}})
        for frame in frames:
            for layer in ("depth", "error"):
                _webp(root / case / layer / f"{frame}.webp")
        depth_methods.append({"method": method, "path": case, "frames": len(frames)})
    for frame in frames:
        for layer in ("gt", "rgb"):
            _webp(root / "viewer_assets/scenes/MonoDepth/ds/s0" / layer / f"{frame}.webp")
    _write(root / "viewer_assets/scenes/MonoDepth/ds/s0/scene.json", {"task": "MonoDepth"})

    rng = np.random.default_rng(1)
    recon_methods = []
    for method in ("A", "B"):
        case = root / f"viewer_assets/imported/{method}/MVRecon/ds/s0/viewer"
        case.mkdir(parents=True)
        rng.random((1000, 3), dtype=np.float32).tofile(case / "points.f32")
        rng.integers(0, 255, (1000, 3), dtype=np.uint8).tofile(case / "colors.u8")
        rng.random(1000, dtype=np.float32).tofile(case / "error.f32")
        _write(case / "cloud.json", {"points": 1000})
        recon_methods.append({"method": method, "path": str(case.relative_to(root)), "points": 1000})
    shared = root / "viewer_assets/scenes/MVRecon/ds/s0"
    shared.mkdir(parents=True)
    rng.random((500, 3), dtype=np.float32).tofile(shared / "gt_points.f32")
    _write(shared / "scene.json", {"gt_points": 500})

    names = ["img0.png", "img1.png"]
    traj_methods = []
    for method in ("A", "B"):
        path = f"viewer_assets/imported/slam/full-slam-short/{method}/SLAM/ds/s0/viewer/trajectory.json"
        _write(root / path, {"names": names, "gt": {"positions": [[0, 0, 0]] * 2},
                             "pred": {"positions": [[0, 0, 0]] * 2}})
        traj_methods.append({"method": method, "path": path, "ate_rmse": 0.1})
    for name in names:
        _webp(root / "viewer_assets/scenes/SLAM/ds/s0/thumbs" / f"{name}.webp", width=320, height=240)

    index = root / "viewer_assets/index"
    _write(index / "depth.json", {"scenes": [{"task": "MonoDepth", "dataset": "ds", "dataset_key": "ds", "scene": "s0",
                                              "scene_path": "viewer_assets/scenes/MonoDepth/ds/s0",
                                              "methods": depth_methods}]})
    _write(index / "recon.json", {"scenes": [{"task": "MVRecon", "dataset": "ds", "dataset_key": "ds", "scene": "s0",
                                              "scene_path": "viewer_assets/scenes/MVRecon/ds/s0", "gt_points": 500,
                                              "methods": recon_methods}]})
    _write(index / "trajectories.json", {"scenes": [{"task": "SLAM", "dataset": "ds", "dataset_key": "ds@short",
                                                     "scene": "s0",
                                                     "thumbnail_dir": "viewer_assets/scenes/SLAM/ds/s0/thumbs",
                                                     "methods": traj_methods}]})


def _results(data_dir: Path) -> None:
    def row(method, ranks):
        return {"method": method, "task_scores": {task: {"rank": rank} for task, rank in ranks.items()}}
    _write(data_dir / "results.json", {"task_ranking": {"tasks": ["MonoDepth", "MVRecon", "SLAM"], "rows": [
        row("C", {"MonoDepth": 1}), row("A", {"MonoDepth": 2, "MVRecon": 2, "SLAM": 1}),
        row("B", {"MonoDepth": 3, "MVRecon": 1, "SLAM": 2}), row("VLM", {"MonoDepth": 9}),
    ]}})


def _config(path: Path, max_total_mb: float = 0) -> Path:
    _write(path, {
        "top_methods": 2,
        "extra_methods": {"MonoDepth": ["VLM"]},
        "caps": {"MonoDepth": {"frames": 4, "frame_pick": "spread"}, "MVRecon": {"points": 100, "gt_points": 50},
                 "thumbnail_width": 160, "thumbnail_quality": 70},
        "max_total_mb": max_total_mb,
        "scenes": [
            {"task": "MonoDepth", "dataset_key": "ds", "scene": "s0", "license": "ibims1"},
            {"task": "MVRecon", "dataset_key": "ds", "scene": "s0", "license": "eth3d"},
            {"task": "SLAM", "dataset_key": "ds@short", "scene": "s0", "license": "tum_dynamics"},
        ],
    })
    return path


def test_export_keeps_the_top_methods_and_caps_frames_points_and_thumbnails(tmp_path):
    eval_root, data_dir, out = tmp_path / "eval", tmp_path / "data", tmp_path / "artifacts"
    _eval_root(eval_root)
    _results(data_dir)
    export.main(["--eval-root", str(eval_root), "--config", str(_config(tmp_path / "config.json")),
                 "--output", str(out), "--data-dir", str(data_dir)])

    depth = json.loads((out / "index/depth.json").read_text())["scenes"][0]
    assert [item["method"] for item in depth["methods"]] == ["C", "A", "VLM"]  # top 2 by rank, then the extra
    frames = json.loads((out / depth["methods"][0]["path"] / "frames.json").read_text())["frames"]
    kept = [frame["frame"] for frame in frames]
    assert kept == ["f00", "f03", "f06", "f09"] and depth["methods"][0]["frames"] == 4
    for frame in kept:
        assert (out / depth["scene_path"] / "gt" / f"{frame}.webp").is_file()
        assert (out / depth["methods"][2]["path"] / "error" / f"{frame}.webp").is_file()
    assert not (out / depth["scene_path"] / "gt" / "f01.webp").exists()

    recon = json.loads((out / "index/recon.json").read_text())["scenes"][0]
    assert [item["method"] for item in recon["methods"]] == ["B", "A"] and recon["gt_points"] == 50
    case = out / recon["methods"][0]["path"]
    assert (case / "points.f32").stat().st_size == 100 * 12 and (case / "colors.u8").stat().st_size == 100 * 3
    assert (case / "error.f32").stat().st_size == 100 * 4
    assert json.loads((case / "cloud.json").read_text())["points"] == 100
    assert (out / recon["scene_path"] / "gt_points.f32").stat().st_size == 50 * 12

    slam = json.loads((out / "index/trajectories.json").read_text())["scenes"][0]
    assert [item["method"] for item in slam["methods"]] == ["A", "B"]
    with Image.open(out / slam["thumbnail_dir"] / "img0.png.webp") as thumb:
        assert thumb.width == 160
    readme = (out / "README.md").read_text()
    assert "iBims-1" in readme and "| MonoDepth | ds | s0 | 3 |" in readme


def test_export_refuses_to_exceed_the_size_budget(tmp_path):
    eval_root, data_dir, out = tmp_path / "eval", tmp_path / "data", tmp_path / "artifacts"
    _eval_root(eval_root)
    _results(data_dir)
    with pytest.raises(SystemExit, match="over max_total_mb"):
        export.main(["--eval-root", str(eval_root), "--config", str(_config(tmp_path / "config.json", 0.001)),
                     "--output", str(out), "--data-dir", str(data_dir)])
    assert not out.exists()


# ---- build_viewer_assets -----------------------------------------------------------------------------------

@pytest.mark.parametrize(("stored", "protocol", "floor"), [
    ({"protocol": "depth_huber_irls_gt", "inverse_floor_policy": "gt_max"}, "depth_huber_irls_gt", "gt_max"),
    ({}, "standard", "fixed"),  # results written before the protocol was recorded
])
def test_backfill_depth_reuses_the_scored_protocol(tmp_path, monkeypatch, stored, protocol, floor):
    eval_path = tmp_path / "eval_depth.json"
    eval_path.write_text(json.dumps({"scene_results": [
        {"task": "MonoDepth", "dataset": "ibims1", "scene": "ibims1", "prediction_scene_path": str(tmp_path)} | stored
    ]}))
    seen = {}
    monkeypatch.setattr(build_viewer_assets, "_scene_info", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(viewer_assets, "make_depth_viewer_writer", lambda *_args: "writer")
    monkeypatch.setattr(depth_prediction_evaluator, "evaluate_depth_scene",
                        lambda _info, _path, config, viewer: seen.update(config=config, viewer=viewer))

    build_viewer_assets.backfill_depth(eval_path, tmp_path, tmp_path)
    assert (seen["config"].protocol, seen["config"].inverse_floor_policy) == (protocol, floor)
    assert seen["viewer"] == "writer" and not seen["config"].compute_geometry_metrics


def _eval_json(path, prediction, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scene_results": [{"prediction_scene_path": prediction}]}))
    os.utime(path, (mtime, mtime))


def test_imported_depth_jobs_render_the_newest_imported_case_only(tmp_path, monkeypatch):
    """Newest batch per case, full-standard and on-matrix only, none with a canonical case; prediction paths local."""
    eval_root, imports = tmp_path / "eval", tmp_path / "imports"
    monkeypatch.setattr(build_viewer_assets, "read_case_lists",
                        lambda _paths: {("MonoDepth", "ibims1", "s0"), ("MonoDepth", "ibims1", "s1")})
    rel = "reports/benchmark_full_reports/full-depth"
    release = build_viewer_assets.RELEASE_OUTPUTS_ROOT
    for batch, mtime in (("old", 1.0), ("new", 2.0)):
        _eval_json(imports / batch / rel / "FOO/MonoDepth/ibims1/s0/eval_depth.json",
                   f"{release}/{batch}/artifacts/FOO/s0", mtime)
    _eval_json(imports / "new" / rel / "FOO/MonoDepth/ibims1/s9/eval_depth.json", "x", 1.0)  # not full-standard
    _eval_json(imports / "new" / rel / "FOO_rgb_depth/MonoDepth/ibims1/s0/eval_depth.json", "x", 1.0)  # off-matrix
    _eval_json(imports / "new" / rel / "BAR/MonoDepth/ibims1/s1/eval_depth.json", "x", 1.0)
    _eval_json(eval_root / rel / "BAR/MonoDepth/ibims1/s1/eval_depth.json", "x", 1.0)  # canonical case exists

    jobs = build_viewer_assets.imported_depth_jobs(eval_root, imports, force=False)
    assert [(path.parts[-9], options) for path, options in jobs] == [("new", {
        "case_dir": eval_root / "viewer_assets/imported/FOO/MonoDepth/ibims1/s0/viewer",
        "prediction_scene_path": imports / "new/artifacts/FOO/s0",
    })]


def test_viewer_index_prefers_the_canonical_case_over_an_imported_one(tmp_path):
    from app.webpage.python.viewer_index import index_depth

    canonical = tmp_path / "reports/benchmark_full_reports/full-depth"
    for root in (canonical, tmp_path / "viewer_assets/imported"):
        for method in ("FOO", "BAR"):
            if root == canonical and method == "BAR":
                continue
            frames = root / method / "MonoDepth/ibims1/s0/viewer/frames.json"
            frames.parent.mkdir(parents=True)
            frames.write_text(json.dumps({"frames": [{"frame": "0"}]}))
    (scene,) = index_depth(tmp_path)["scenes"]
    paths = {entry["method"]: entry["path"] for entry in scene["methods"]}
    assert paths == {"FOO": "reports/benchmark_full_reports/full-depth/FOO/MonoDepth/ibims1/s0/viewer",
                     "BAR": "viewer_assets/imported/BAR/MonoDepth/ibims1/s0/viewer"}


def _sparse_json(path, sparse, mtime=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scene_results": [{"prediction_sparse_path": sparse, "metrics": {"ate_rmse": 0.5}}]}))
    os.utime(path, (mtime, mtime))


def test_imported_trajectory_jobs_read_the_sparse_model_from_the_import_root(tmp_path, monkeypatch):
    eval_root, imports = tmp_path / "eval", tmp_path / "imports"
    monkeypatch.setattr(build_viewer_assets, "read_case_lists", lambda _paths: {("SfM", "eth3d", "s0")})
    rel = "reports/benchmark_full_pose_reports/full-pose"
    release = build_viewer_assets.RELEASE_OUTPUTS_ROOT
    _sparse_json(imports / "b" / rel / "FOO/SfM/eth3d/s0/eval_sparse.json", f"{release}/b/artifacts/FOO/s0/sparse/0")
    _sparse_json(imports / "b" / rel / "FOO/MonoDepth/eth3d/s0/eval_sparse.json", "x")  # not a pose task

    jobs = build_viewer_assets.imported_jobs("imported_trajectories", eval_root, imports, force=False)
    assert [options for _path, options in jobs] == [{
        "case_dir": eval_root / "viewer_assets/imported/FOO/SfM/eth3d/s0/viewer",
        "prediction_sparse_path": imports / "b/artifacts/FOO/s0/sparse/0",
    }]


def test_imported_recon_jobs_skip_cases_with_a_canonical_cloud(tmp_path, monkeypatch):
    eval_root, imports = tmp_path / "eval", tmp_path / "imports"
    monkeypatch.setattr(build_viewer_assets, "read_case_lists",
                        lambda _paths: {("MVRecon", "7scenes", "s0"), ("MVRecon", "7scenes", "s1")})
    rel = "reports/benchmark_full_reports/full-depth"
    release = build_viewer_assets.RELEASE_OUTPUTS_ROOT
    for scene in ("s0", "s1"):
        _eval_json(imports / "b" / rel / f"FOO/MVRecon/7scenes/{scene}/eval_mvrecon.json",
                   f"{release}/b/artifacts/FOO/{scene}", 1.0)
    _eval_json(eval_root / rel / "FOO/MVRecon/7scenes/s1/eval_mvrecon.json", "x", 1.0)  # canonical case exists

    jobs = build_viewer_assets.imported_jobs("imported_recon", eval_root, imports, force=False)
    assert [options for _path, options in jobs] == [{
        "case_dir": eval_root / "viewer_assets/imported/FOO/MVRecon/7scenes/s0/viewer",
        "prediction_scene_path": imports / "b/artifacts/FOO/s0",
    }]


def test_viewer_index_lists_imported_trajectories_and_clouds_after_canonical_ones(tmp_path):
    from app.webpage.python.viewer_index import index_recon, index_trajectories

    pose = tmp_path / "reports/benchmark_full_pose_reports/full-pose"
    depth = tmp_path / "reports/benchmark_full_reports/full-depth"
    imported = tmp_path / "viewer_assets/imported"
    for root, method in ((pose, "FOO"), (imported, "FOO"), (imported, "BAR")):
        case = root / method / "SfM/eth3d/s0"
        (case / "viewer").mkdir(parents=True)
        (case / "viewer/trajectory.json").write_text("{}")
        _sparse_json(case / "eval_sparse.json", "x")
    for root, method in ((depth, "FOO"), (imported, "FOO"), (imported, "BAR")):
        cloud = root / method / "MVRecon/7scenes/s0/viewer/cloud.json"
        cloud.parent.mkdir(parents=True)
        cloud.write_text(json.dumps({"points": 10, "metrics": {"fscore": 0.5}}))

    (scene,) = index_trajectories(tmp_path)["scenes"]
    assert {entry["method"]: entry["path"] for entry in scene["methods"]} == {
        "FOO": "reports/benchmark_full_pose_reports/full-pose/FOO/SfM/eth3d/s0/viewer/trajectory.json",
        "BAR": "viewer_assets/imported/BAR/SfM/eth3d/s0/viewer/trajectory.json"}
    assert {entry["ate_rmse"] for entry in scene["methods"]} == {0.5}
    (scene,) = index_recon(tmp_path)["scenes"]
    assert {entry["method"]: entry["path"] for entry in scene["methods"]} == {
        "FOO": "reports/benchmark_full_reports/full-depth/FOO/MVRecon/7scenes/s0/viewer",
        "BAR": "viewer_assets/imported/BAR/MVRecon/7scenes/s0/viewer"}


def test_imported_slam_jobs_keep_the_range_and_skip_failed_cases(tmp_path):
    eval_root, imports = tmp_path / "eval", tmp_path / "imports"
    rel = "reports/benchmark_full_slam_reports"
    release = build_viewer_assets.RELEASE_OUTPUTS_ROOT
    for suite in ("full-slam-short", "full-slam-long"):
        _sparse_json(imports / "b" / rel / suite / "FOO/SLAM/7scenes/s0/eval_slam.json",
                     f"{release}/b/artifacts/{suite}/FOO/s0/sparse/0")
    failed = imports / "b" / rel / "full-slam-short/BAR/SLAM/7scenes/s0/eval_slam.json"
    failed.parent.mkdir(parents=True)
    failed.write_text(json.dumps({"scene_results": [], "errors": ["insufficient poses"]}))

    jobs = build_viewer_assets.imported_jobs("imported_slam", eval_root, imports, force=False)
    assert [options for _path, options in jobs] == [{
        "case_dir": eval_root / f"viewer_assets/imported/slam/{suite}/FOO/SLAM/7scenes/s0/viewer",
        "prediction_sparse_path": imports / f"b/artifacts/{suite}/FOO/s0/sparse/0",
    } for suite in ("full-slam-long", "full-slam-short")]


def test_viewer_index_lists_imported_slam_cases_per_range_after_canonical_ones(tmp_path):
    from app.webpage.python.viewer_index import index_trajectories

    canonical = tmp_path / "reports/benchmark_full_slam_reports"
    imported = tmp_path / "viewer_assets/imported/slam"
    for root, suite, method in ((canonical, "full-slam-short", "FOO"), (imported, "full-slam-short", "FOO"),
                                (imported, "full-slam-short", "BAR"), (imported, "full-slam-long", "BAR")):
        case = root / suite / method / "SLAM/7scenes/s0"
        (case / "viewer").mkdir(parents=True)
        (case / "viewer/trajectory.json").write_text("{}")
        _sparse_json(case / "eval_slam.json", "x")

    scenes = {scene["dataset_key"]: scene for scene in index_trajectories(tmp_path)["scenes"]}
    assert {entry["method"]: entry["path"] for entry in scenes["7scenes@short"]["methods"]} == {
        "FOO": "reports/benchmark_full_slam_reports/full-slam-short/FOO/SLAM/7scenes/s0/viewer/trajectory.json",
        "BAR": "viewer_assets/imported/slam/full-slam-short/BAR/SLAM/7scenes/s0/viewer/trajectory.json"}
    assert [entry["method"] for entry in scenes["7scenes@long"]["methods"]] == ["BAR"]
