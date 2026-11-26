# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from spatialarena.geometry.read_write_model import Camera, read_model, rotmat2qvec, write_model
from spatialarena.geometry.read_write_model import Image as ColmapImage
from spatialarena import slam_protocol
from scripts.preprocess import materialize_spatialarena as mm

WIDTH, HEIGHT = 32, 24
FOCAL = 20.0


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_release(root: Path,
                 name: str,
                 numbers: list[int],
                 *,
                 gggs: bool = True) -> Path:
    """A tiny released scene: 3 cameras looking down +z at a wall 5 m away, rig frames ``numbers``."""
    scene = root / f"{name}-20261001-120000" / "convert_final"
    cameras = {
        cam + 1:
            Camera(id=cam + 1,
                   model="PINHOLE",
                   width=WIDTH,
                   height=HEIGHT,
                   params=np.array([FOCAL, FOCAL, WIDTH / 2, HEIGHT / 2]))
        for cam in range(3)
    }
    images = {}
    rng = np.random.default_rng(1)
    image_id = 0
    manifest_frames = []
    for cam in range(3):
        for number in numbers:
            image_id += 1
            stem = f"img{cam}_{number}"
            rotation = _rotation_z(0.01 * cam)
            center = np.array([0.1 * number, 0.0, 0.0])
            images[image_id] = ColmapImage(id=image_id,
                                           qvec=rotmat2qvec(rotation),
                                           tvec=-rotation @ center,
                                           camera_id=cam + 1,
                                           name=f"cam{cam}/{stem}.JPEG",
                                           xys=np.zeros((0, 2)),
                                           point3D_ids=np.zeros(0,
                                                                dtype=np.int64))
            photo = rng.integers(0,
                                 256,
                                 size=(HEIGHT, WIDTH, 3),
                                 dtype=np.uint8)
            (scene / "images" / f"cam{cam}").mkdir(parents=True, exist_ok=True)
            Image.fromarray(photo).save(scene / "images" / f"cam{cam}" /
                                        f"{stem}.JPEG")
            (scene / "lidar_depth" / f"cam{cam}").mkdir(parents=True,
                                                        exist_ok=True)
            lidar = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
            lidar[::4, ::4] = 500  # 5 m in centimetres
            Image.fromarray(lidar).save(scene / "lidar_depth" / f"cam{cam}" /
                                        f"{stem}.png")
            vehicles = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            vehicles[:4, :4] = 255
            (scene / "privacy" / "masks" / "vehicles" / f"cam{cam}").mkdir(
                parents=True, exist_ok=True)
            Image.fromarray(vehicles).save(scene / "privacy" / "masks" /
                                           "vehicles" / f"cam{cam}" /
                                           f"{stem}.png")
            if gggs:
                evaluation = scene / "training" / f"{scene.parent.name}{mm.GGGS_RUN_SUFFIX}" / "evaluation"
                for sub in ("depths", "rendered_images", "rgb_gt", "masks"):
                    (evaluation / sub / f"cam{cam}").mkdir(parents=True,
                                                           exist_ok=True)
                depth = np.full((HEIGHT, WIDTH), 5.0, dtype=np.float32)
                np.save(evaluation / "depths" / f"cam{cam}" / f"{stem}.npy",
                        depth)
                Image.fromarray(photo).save(evaluation / "rgb_gt" /
                                            f"cam{cam}" / f"{stem}.png")
                noise = rng.integers(
                    0, 2 + (number * 7 + cam) % 40,
                    size=photo.shape)  # a different PSNR per image
                rendered = np.clip(photo.astype(int) + noise, 0,
                                   255).astype(np.uint8)
                Image.fromarray(rendered).save(evaluation / "rendered_images" /
                                               f"cam{cam}" / f"{stem}.png")
                Image.fromarray(np.full((HEIGHT, WIDTH), 255, np.uint8)).save(
                    evaluation / "masks" / f"cam{cam}" / f"{stem}.png")
                manifest_frames.append({
                    "image":
                        f"cam{cam}/{stem}.JPEG",
                    "depth":
                        str(evaluation / "depths" / f"cam{cam}" / f"{stem}.npy"
                           ),
                    "rendered_rgb":
                        str(evaluation / "rendered_images" / f"cam{cam}" /
                            f"{stem}.png"),
                    "rgb_gt":
                        str(evaluation / "rgb_gt" / f"cam{cam}" / f"{stem}.png"
                           ),
                    "mask":
                        str(evaluation / "masks" / f"cam{cam}" / f"{stem}.png"),
                })
    (scene / "colmap" / "sparse" / "0").mkdir(parents=True)
    write_model(cameras,
                images, {},
                str(scene / "colmap" / "sparse" / "0"),
                ext=".bin")
    if gggs:
        evaluation = scene / "training" / f"{scene.parent.name}{mm.GGGS_RUN_SUFFIX}" / "evaluation"
        (evaluation / "mindcloud_full_generation_manifest.json").write_text(
            json.dumps({
                "status": "done",
                "frames": manifest_frames
            }),
            encoding="utf-8")
    # The LiDAR map: a wall at z = 5 (seen) and points 3 m behind it (hidden).
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20)),
                    -1).reshape(-1, 2)
    wall = np.c_[grid, np.full(len(grid), 5.0)]
    hidden = np.c_[grid, np.full(len(grid), 8.0)]
    mm.write_ply_points(scene / "dense_map" / "points3D_dense.ply",
                        np.r_[wall, hidden],
                        np.full((2 * len(grid), 3), 128, dtype=np.uint8))
    manifest = scene / "refined_depth_manifest.json"
    manifest.write_text(json.dumps({
        "status": "done",
        "depth_scale": 100.0
    }),
                        encoding="utf-8")
    later = (scene / "colmap" / "sparse" / "0" /
             "images.bin").stat().st_mtime + 10
    os.utime(manifest, (later, later))
    return scene


def write_selection(path: Path, **overrides) -> Path:
    selection = {
        "seed": 0,
        "lidar_depth_scale": 100.0,
        "monodepth": {
            "scenes": ["Mono"],
            "sample_from": ["videodepth"],
            "fraction": 0.2,
            "min_depth": 0.1,
            "max_depth": 100.0
        },
        "videodepth": {
            "scenes": ["Video"],
            "clip_frames": [3, 5],
            "whole": []
        },
        "mvrecon": {
            "views": 6,
            "scenes": {
                "Recon": {
                    "setting": "indoor",
                    "eval_threshold": 0.05
                }
            }
        },
        "sfm": {
            "scenes": ["Sfm"]
        },
        "slam": {
            "frame_rate_hz":
                5.0,
            "sequences": [
                {
                    "scene": "Slam",
                    "range": "short",
                    "camera": 2,
                    "input": "rgb_depth"
                },
                {
                    "scene": "SlamRgb",
                    "range": "large",
                    "camera": 0,
                    "input": "rgb"
                },
            ]
        },
    }
    selection.update(overrides)
    path.write_text(yaml.safe_dump(selection), encoding="utf-8")
    return path


def run(tmp_path: Path, tracks: list[str], *extra: str, **overrides) -> Path:
    benchmark = tmp_path / "benchmark"
    selection = write_selection(tmp_path / "selection.yaml", **overrides)
    mm.main([
        "--released_root",
        str(tmp_path / "released"), "--output_root",
        str(benchmark), "--selection",
        str(selection), "--review",
        str(tmp_path / "review.json"), "--tracks", *tracks, *extra
    ])
    return benchmark


def lidar_heatmap(depth_png: Path) -> np.ndarray:
    """The heatmap the materializer draws of a LiDAR depth PNG (cm): turbo over its 1st to 99th percentile."""
    depth = np.asarray(Image.open(depth_png), dtype=np.float32) / 100.0
    lower, upper = np.percentile(depth[depth > 0], [1.0, 99.0])
    return mm.depth_heatmap(depth, float(lower), float(upper))


def test_sfm_keeps_every_posed_image_with_its_lidar_depth_and_pose(
        tmp_path: Path):
    source = make_release(tmp_path / "released", "Sfm", [2, 10, 1], gggs=False)
    benchmark = run(tmp_path, ["sfm"])
    scene = benchmark / "SfM" / "Sfm"
    names = sorted(path.name for path in (scene / "images").iterdir())
    assert names[:3] == [
        "img0_000001.jpg", "img0_000002.jpg", "img0_000010.jpg"
    ] and len(names) == 9
    assert sorted(path.stem for path in (scene / "depths").iterdir()) == [
        Path(name).stem for name in names
    ]
    heatmap = np.asarray(
        Image.open(scene / mm.HEATMAP_DIRNAME / "img2_000010.png"))
    np.testing.assert_array_equal(
        heatmap, lidar_heatmap(scene / "depths" / "img2_000010.png"))
    cameras, images, points = read_model(str(scene / "sparse" / "0"))
    assert len(cameras) == 3 and not points
    _, source_images, _ = read_model(str(source / "colmap" / "sparse" / "0"))
    by_name = {image.name: image for image in source_images.values()}
    for image in images.values():
        cam, number = image.name[3], int(image.name[5:11])
        original = by_name[f"cam{cam}/img{cam}_{number}.JPEG"]
        np.testing.assert_allclose(image.tvec, original.tvec)
        np.testing.assert_allclose(image.qvec, original.qvec)
    metadata = json.loads((benchmark / "SfM" / "metadata.json").read_text())
    assert metadata["depth_png_scale"] == 100.0 and metadata["scenes"][0][
        "total_images"] == 9
    assert not (benchmark / "SfM" / mm.STAGING_DIRNAME).exists()


def test_slam_sequences_are_one_camera_in_time_order_with_depth_only_for_rgbd(
        tmp_path: Path):
    make_release(tmp_path / "released", "Slam", [2, 10, 1], gggs=False)
    make_release(tmp_path / "released", "SlamRgb", [0, 1], gggs=False)
    benchmark = run(tmp_path, ["slam"])
    sequence = benchmark / "SLAM" / "Slam"
    rows = (sequence / "frames.tsv").read_text().splitlines()[1:]
    assert [row.split("\t")[1] for row in rows
           ] == ["cam2/img2_1.JPEG", "cam2/img2_2.JPEG", "cam2/img2_10.JPEG"]
    assert sorted(path.name for path in (sequence / "depths").iterdir()) == [
        "000000.png", "000001.png", "000002.png"
    ]
    # Pre-sized like every SLAM dataset: the 32x24 frames resized to 504x378 and padded by 14 rows top and bottom.
    target = slam_protocol.input_size("spatialarena")
    heatmap = np.asarray(
        Image.open(sequence / mm.HEATMAP_DIRNAME / "000002.png"))
    assert heatmap.shape[:2] == (target[1], target[0])
    source_depth = (tmp_path / "released" / "Slam-20261001-120000" /
                    "convert_final" / "lidar_depth" / "cam2" / "img2_10.png")
    np.testing.assert_array_equal(
        heatmap, mm.fit_image(lidar_heatmap(source_depth), target))
    assert not heatmap[0, 0].any()
    assert Image.open(sequence / "images" / "000000.jpg").size == target
    depth = np.asarray(Image.open(sequence / "depths" / "000000.png"))
    assert depth.dtype == np.uint16 and depth.shape == (
        target[1], target[0]) and depth.max() == 500
    cameras, images, _ = read_model(str(sequence / "sparse" / "0"))
    assert list(cameras) == [1] and sorted(
        image.name for image in images.values())[0] == "000000.jpg"
    camera = cameras[1]
    assert (camera.width, camera.height) == target
    assert camera.params[0] == pytest.approx(FOCAL * 15.75)  # 504 / 32
    assert camera.params[3] == pytest.approx((HEIGHT / 2 + 0.5) * 15.75 - 0.5 +
                                             14)
    rgb_only = benchmark / "SLAM" / "SlamRgb"
    assert (rgb_only / "images").is_dir() and not (rgb_only / "depths").exists()
    assert not (rgb_only / mm.HEATMAP_DIRNAME).exists()
    metadata = json.loads((benchmark / "SLAM" / "metadata.json").read_text())
    assert metadata["frame_rate_hz"] == 5.0 and metadata[
        "depth_png_scale"] == 100.0
    ranges = {
        entry["scene_name"]: entry["range"] for entry in metadata["scenes"]
    }
    assert ranges == {
        "Slam": "short",
        "SlamRgb": "large"
    } and metadata["dataset_name"] == "spatialarena"


def test_stale_lidar_depth_is_refused(tmp_path: Path):
    scene = make_release(tmp_path / "released", "Sfm", [0, 1], gggs=False)
    poses = scene / "colmap" / "sparse" / "0" / "images.bin"
    later = (scene / "refined_depth_manifest.json").stat().st_mtime + 10
    os.utime(poses, (later, later))
    with pytest.raises(RuntimeError, match="predates"):
        run(tmp_path, ["sfm"])


def test_videodepth_clips_tile_the_timeline_within_the_length_range(
        tmp_path: Path):
    scene = mm.SourceScene(
        "Video",
        make_release(tmp_path / "released",
                     "Video",
                     list(range(23)),
                     gggs=False), 100.0)
    config = {"clip_frames": [3, 5], "whole": []}
    clips = mm.videodepth_clips(scene, config, seed=0)
    assert clips == mm.videodepth_clips(scene, config, seed=0)  # reproducible
    covered = [
        number for clip in clips
        for number in range(clip["first_rig_frame"], clip["last_rig_frame"] + 1)
    ]
    assert covered == sorted(
        set(covered)) and covered[0] == 0  # consecutive and non-overlapping
    assert all(3 <= clip["rig_frames"] <= 5 for clip in clips)
    whole = mm.videodepth_clips(scene, {
        "clip_frames": [3, 5],
        "whole": ["Video"]
    },
                                seed=0)
    assert len(whole) == 1 and whole[0]["rig_frames"] == 23 and whole[0][
        "scene_name"].startswith("Video_full_cam")


def test_monodepth_waits_for_gggs_and_then_writes_metric_depth(tmp_path: Path):
    make_release(tmp_path / "released", "Mono", list(range(10)), gggs=False)
    make_release(tmp_path / "released", "Video", list(range(8)))
    benchmark = run(tmp_path, ["monodepth"])
    assert not (benchmark / "MonoDepth" /
                "Mono").exists()  # no GGGS yet: skipped
    mono_scene = benchmark / "MonoDepth" / "Video"  # sampled from the VideoDepth scene
    manifest = json.loads((mono_scene / "materialized_full.json").read_text())
    assert 0 < manifest["num_samples"] <= round(
        0.2 * 24)  # a stretch of the timeline without good renders is skipped
    median = manifest["provenance"]["median_render_psnr"]
    assert all(record["render_psnr"] > median for record in manifest["records"])
    record = manifest["records"][0]
    depth = np.load(mono_scene / record["depth"])
    assert depth[0, 0] == 0.0 and depth[-1, -1] == pytest.approx(
        5.0)  # vehicles removed, metres kept
    assert record["intrinsics"][0][0] == FOCAL and (
        mono_scene / record["lidar_depth"]).is_file()
    heatmaps = sorted(
        path.stem for path in (mono_scene / mm.DENSE_HEATMAP_DIRNAME).iterdir())
    assert heatmaps == sorted(
        path.stem for path in (mono_scene / "depths").iterdir())
    assert json.loads((
        mono_scene /
        "scene.json").read_text())["depth_heatmaps"] == mm.DENSE_HEATMAP_DIRNAME


def test_monodepth_sample_is_spread_over_each_cameras_timeline(tmp_path: Path):
    make_release(tmp_path / "released", "Mono", list(range(60)))
    scene = mm.SourceScene(
        "Mono",
        next((tmp_path / "released").glob("Mono-*")) / "convert_final", 100.0,
        False)
    chosen, psnr = mm.monodepth_selection(scene, 0.2, seed=0)
    median = float(np.median(list(psnr.values())))
    assert all(psnr[frame.name] > median for frame in chosen)
    assert len(
        chosen
    ) > 0.5 * 36  # 12 stretches of 5 frames per camera; most hold a well-spaced good render
    for cam in range(3):
        numbers = sorted(frame.number for frame in chosen if frame.cam == cam)
        assert numbers and min(b - a for a, b in zip(numbers, numbers[1:])
                              ) >= 3  # never neighbouring frames


def add_gggs(tmp_path: Path, name: str, numbers: list[int]) -> None:
    """Simulate GGGS finishing: rebuild the release with its full-generation output (same images and poses)."""
    import shutil  # pylint: disable=import-outside-toplevel
    shutil.rmtree(next((tmp_path / "released").glob(f"{name}-*")))
    make_release(tmp_path / "released", name, numbers)


def test_videodepth_is_written_now_and_completed_with_gggs_depth_later(
        tmp_path: Path):
    make_release(tmp_path / "released", "Video", list(range(8)), gggs=False)
    benchmark = run(tmp_path, ["videodepth"])
    clips = sorted((benchmark / "VideoDepth").glob("Video_clip*"))
    assert clips and all(
        (clip / "sparse" / "0" / "images.txt").is_file() for clip in clips)
    assert not (clips[0] / "depths").exists() and (clips[0] / "lidar_depths" /
                                                   "000000.png").is_file()
    assert sorted(
        path.name
        for path in (clips[0] / mm.HEATMAP_DIRNAME).iterdir()) == sorted(
            path.name for path in (clips[0] / "lidar_depths").iterdir())
    entry = json.loads(
        (benchmark / "VideoDepth" / "metadata.json").read_text())["scenes"][0]
    assert entry["dense_depth"] is False
    add_gggs(tmp_path, "Video", list(range(8)))
    run(tmp_path, ["videodepth"])
    frames = json.loads((clips[0] / "frame_manifest.json").read_text())
    assert frames[0]["rgb"] == "images/000000.jpg" and frames[0][
        "depth"] == "depths/000000.npy"
    assert np.load(clips[0] / frames[0]["depth"]).dtype == np.float32
    entries = json.loads(
        (benchmark / "VideoDepth" / "metadata.json").read_text())["scenes"]
    assert all(entry["dense_depth"] for entry in entries)
    assert sorted(
        path.stem
        for path in (clips[0] / mm.DENSE_HEATMAP_DIRNAME).iterdir()) == sorted(
            path.stem for path in (clips[0] / "depths").iterdir())


def test_mvrecon_crops_the_lidar_map_to_what_the_views_see_once_gggs_is_done(
        tmp_path: Path):
    make_release(tmp_path / "released", "Recon", list(range(4)), gggs=False)
    benchmark = run(tmp_path, ["mvrecon"])
    scene = benchmark / "MVRecon" / "Recon"
    views = sorted(path.name for path in (scene / "images").iterdir())
    assert len(views) == 6 and not (scene / "gt_points.ply").exists() and not (
        scene / "depths").exists()
    assert len(list((scene / mm.HEATMAP_DIRNAME).iterdir())) == 6
    add_gggs(tmp_path, "Recon", list(range(4)))
    run(tmp_path, ["mvrecon"])
    assert sorted(
        path.name
        for path in (scene / "images").iterdir()) == views  # the same views
    assert len(list((scene / "masks").iterdir())) == 6 and len(
        list((scene / "depths").iterdir())) == 6
    xyz, _ = mm.read_ply_points(scene / "gt_points.ply")
    assert len(xyz) > 0 and np.allclose(
        xyz[:, 2], 5.0)  # the wall stays, the points behind it go
    entry = json.loads(
        (benchmark / "MVRecon" / "metadata.json").read_text())["scenes"][0]
    assert entry["gt_point_path"] == "gt_points.ply" and entry[
        "eval_threshold"] == 0.05 and entry["dense_depth"]


def test_dense_depth_heatmap_is_turbo_near_warm_with_invalid_black():
    depth = np.array([[0.0, 1.0, 3.0, 9.0]], dtype=np.float32)
    image = mm.depth_heatmap(depth, 1.0, 3.0)
    assert image.dtype == np.uint8 and image.shape == (1, 4, 3)
    assert tuple(image[0, 0]) == (0, 0, 0)  # invalid
    assert image[0, 1, 0] > image[0, 1, 2] and image[0, 2, 2] > image[
        0, 2, 0]  # near red, far blue
    assert tuple(image[0, 3]) == tuple(image[0, 2])  # clipped to the range


def test_heatmaps_are_added_to_scenes_written_before_them(tmp_path: Path):
    make_release(tmp_path / "released", "Sfm", [0, 1], gggs=False)
    benchmark = run(tmp_path, ["sfm"])
    scene = benchmark / "SfM" / "Sfm"
    import shutil  # pylint: disable=import-outside-toplevel
    shutil.rmtree(scene / mm.HEATMAP_DIRNAME)
    run(tmp_path, ["sfm"])  # the scene is complete; only its heatmaps are drawn
    assert len(list((scene / mm.HEATMAP_DIRNAME).iterdir())) == 6
    assert json.loads((
        scene /
        "scene.json").read_text())["lidar_depth_heatmaps"] == mm.HEATMAP_DIRNAME


def test_scenes_dropped_from_the_selection_are_removed(tmp_path: Path):
    make_release(tmp_path / "released", "Mono", list(range(10)))
    make_release(tmp_path / "released", "Video", list(range(8)))
    benchmark = run(tmp_path, ["monodepth"])
    assert (benchmark / "MonoDepth" /
            "Mono").is_dir() and (benchmark / "MonoDepth" / "Video").is_dir()
    run(tmp_path, ["monodepth"],
        videodepth={
            "scenes": [],
            "clip_frames": [3, 5],
            "whole": []
        })
    assert (benchmark / "MonoDepth" / "Mono"
           ).is_dir() and not (benchmark / "MonoDepth" / "Video").exists()
    entries = json.loads(
        (benchmark / "MonoDepth" / "metadata.json").read_text())["scenes"]
    assert [entry["scene_name"] for entry in entries] == ["Mono"]


def test_scenes_are_rebuilt_when_their_gggs_run_is_retrained(tmp_path: Path):
    make_release(tmp_path / "released", "Mono", list(range(10)))
    make_release(tmp_path / "released", "Recon", list(range(4)))
    benchmark = run(tmp_path, ["monodepth", "mvrecon"],
                    monodepth={
                        "scenes": ["Mono"],
                        "sample_from": [],
                        "fraction": 0.2,
                        "min_depth": 0.1,
                        "max_depth": 100.0
                    })
    run_dir = next(
        (tmp_path / "released").glob("Mono-*")) / "convert_final" / "training"
    stamp = json.loads((benchmark / "MonoDepth" / "Mono" /
                        "scene.json").read_text())["gggs_generated"]
    # Retraining: the old run is moved aside and the new one has no full generation yet.
    run_dir.joinpath(
        f"{run_dir.parent.parent.name}{mm.GGGS_RUN_SUFFIX}").rename(run_dir /
                                                                    "old_run")
    recon_run = next(
        (tmp_path / "released").glob("Recon-*")) / "convert_final" / "training"
    recon_run.joinpath(
        f"{recon_run.parent.parent.name}{mm.GGGS_RUN_SUFFIX}").rename(
            recon_run / "old_run")
    keys = [
        entry["key"] for entry in json.loads((
            benchmark / "MonoDepth" / "Mono" /
            "materialized_full.json").read_text())["records"]
    ]
    run(tmp_path, ["monodepth", "mvrecon"],
        monodepth={
            "scenes": ["Mono"],
            "sample_from": [],
            "fraction": 0.2,
            "min_depth": 0.1,
            "max_depth": 100.0
        })
    kept = json.loads(
        (benchmark / "MonoDepth" / "Mono" / "scene.json").read_text())
    assert kept[
        "gggs_generated"] == stamp  # waits for the new generation with its current depth
    recon = json.loads(
        (benchmark / "MVRecon" / "Recon" / "scene.json").read_text())
    assert not recon["dense_depth"] and not (benchmark / "MVRecon" / "Recon" /
                                             "depths").exists()
    add_gggs(tmp_path, "Mono", list(range(10)))
    later = time.time() + 100
    manifest = next(run_dir.glob(f"*{mm.GGGS_RUN_SUFFIX}")
                   ) / "evaluation" / "mindcloud_full_generation_manifest.json"
    os.utime(manifest, (later, later))
    run(tmp_path, ["monodepth"],
        monodepth={
            "scenes": ["Mono"],
            "sample_from": [],
            "fraction": 0.2,
            "min_depth": 0.1,
            "max_depth": 100.0
        })
    record = json.loads(
        (benchmark / "MonoDepth" / "Mono" / "scene.json").read_text())
    assert record["gggs_generated"] != stamp and (benchmark / "MonoDepth" /
                                                  "Mono" / "depths").is_dir()
    manifest = json.loads((benchmark / "MonoDepth" / "Mono" /
                           "materialized_full.json").read_text())
    assert [entry["key"] for entry in manifest["records"]
           ] == keys  # the same frames, so the same review
    assert manifest["provenance"][
        "sampled_with"] == f"{run_dir.parent.parent.name}{mm.GGGS_RUN_SUFFIX}"


def test_monodepth_takes_the_compacted_models_depth_and_keeps_its_frames(
        tmp_path: Path):
    make_release(tmp_path / "released", "Mono", list(range(10)))
    selection = {
        "scenes": ["Mono"],
        "sample_from": [],
        "fraction": 0.2,
        "min_depth": 0.1,
        "max_depth": 100.0
    }
    benchmark = run(tmp_path, ["monodepth"], monodepth=selection)
    scene = benchmark / "MonoDepth" / "Mono"
    keys = [
        entry["key"] for entry in json.loads((
            scene / "materialized_full.json").read_text())["records"]
    ]
    # Compaction: the run's generation moves to <run>_compact, with new depth.
    training = next(
        (tmp_path / "released").glob("Mono-*")) / "convert_final" / "training"
    run_dir = training / f"{training.parent.parent.name}{mm.GGGS_RUN_SUFFIX}"
    compact = run_dir.with_name(run_dir.name + mm.GGGS_COMPACT_SUFFIX)
    run_dir.rename(compact)
    legacy_path = compact / "evaluation" / "mindcloud_full_generation_manifest.json"
    generation = json.loads(legacy_path.read_text().replace(
        str(run_dir) + "/",
        str(compact) + "/"))
    for frame in generation["frames"]:
        np.save(frame["depth"], np.full((HEIGHT, WIDTH), 4.0, dtype=np.float32))
    # DOGE's generation writes full_generation_manifest.json; older runs have the legacy name.
    legacy_path.unlink()
    manifest_path = compact / "evaluation" / "full_generation_manifest.json"
    manifest_path.write_text(json.dumps(generation))
    later = time.time() + 100
    os.utime(manifest_path, (later, later))
    run(tmp_path, ["monodepth"], monodepth=selection)
    manifest = json.loads((scene / "materialized_full.json").read_text())
    assert [entry["key"] for entry in manifest["records"]] == keys
    assert np.load(
        scene / manifest["records"][0]["depth"])[-1, -1] == pytest.approx(4.0)
    record = json.loads((scene / "scene.json").read_text())
    assert record["gggs"] == compact.name and record["provenance"][
        "sampled_with"] == run_dir.name
    assert len(list(
        (scene / mm.DENSE_HEATMAP_DIRNAME).glob("*.png"))) == len(keys)


def test_frames_the_review_left_out_can_be_put_back_for_another_review(
        tmp_path: Path):
    make_release(tmp_path / "released", "Mono", list(range(10)))
    selection = {
        "scenes": ["Mono"],
        "sample_from": [],
        "fraction": 0.2,
        "min_depth": 0.1,
        "max_depth": 100.0
    }
    benchmark = run(tmp_path, ["monodepth"], monodepth=selection)
    scene = benchmark / "MonoDepth" / "Mono"
    keys = [
        entry["key"] for entry in json.loads((
            scene / "materialized_full.json").read_text())["records"]
    ]
    (tmp_path / "review.json").write_text(
        json.dumps({"tracks": {
            "MonoDepth": {
                "Mono": {
                    "drop": keys[:2]
                }
            }
        }}))
    run(tmp_path, ["monodepth"], monodepth=selection)
    assert not list(scene.rglob(f"{keys[0]}.*"))
    run(tmp_path, ["monodepth"], "--restore_review_drops", monodepth=selection)
    manifest = json.loads((scene / "materialized_full.json").read_text())
    assert [entry["key"] for entry in manifest["records"]] == keys
    for folder in ("images", "depths", "lidar_depths", mm.HEATMAP_DIRNAME,
                   mm.DENSE_HEATMAP_DIRNAME):
        assert len(list((scene / folder).iterdir())) == len(keys)
    record = json.loads((scene / "scene.json").read_text())
    assert record["total_images"] == len(
        keys) and record["manual_review"]["excluded_frames"] == []
    # The review keeps the second frame after all: the next run removes only the first.
    (tmp_path / "review.json").write_text(
        json.dumps({"tracks": {
            "MonoDepth": {
                "Mono": {
                    "drop": keys[:1]
                }
            }
        }}))
    run(tmp_path, ["monodepth"], monodepth=selection)
    manifest = json.loads((scene / "materialized_full.json").read_text())
    assert [entry["key"] for entry in manifest["records"]] == keys[1:]
    assert json.loads(
        (scene / "scene.json"
        ).read_text())["manual_review"]["excluded_frames"] == keys[:1]


def masked_to_black(image: np.ndarray) -> bool:
    """Whether the fixture's masked top-left block of a JPEG photo is black.

    The fixture photos are random noise, on which JPEG (quality 95) rings into
    a black block by up to ~70 levels at single pixels, so the block's mean is
    checked: at most ~11 when masked, ~127 when not.
    """
    return image[:3, :3].astype(float).mean() <= 32


def test_depth_tracks_mask_people_and_vehicles_out_of_every_input(
        tmp_path: Path):
    make_release(tmp_path / "released", "Video", list(range(8)))
    make_release(tmp_path / "released", "Recon", list(range(4)))
    make_release(tmp_path / "released", "Sfm", [0, 1], gggs=False)
    benchmark = run(tmp_path, ["monodepth", "videodepth", "mvrecon", "sfm"],
                    monodepth={
                        "scenes": [],
                        "sample_from": ["videodepth"],
                        "fraction": 0.2,
                        "min_depth": 0.1,
                        "max_depth": 100.0
                    })
    scenes = [
        benchmark / "MonoDepth" / "Video",
        next((benchmark / "VideoDepth").glob("Video_clip*")),
        benchmark / "MVRecon" / "Recon"
    ]
    for scene in scenes:  # the fixture's vehicle mask is the top-left 4x4 pixels of every frame
        assert json.loads(
            (scene / "scene.json").read_text())["dynamic_objects_masked"]
        image = np.asarray(Image.open(next((scene / "images").iterdir())))
        lidar = np.asarray(Image.open(next((scene / "lidar_depths").iterdir())))
        heatmap = np.asarray(
            Image.open(next((scene / mm.HEATMAP_DIRNAME).iterdir())))
        assert masked_to_black(image)
        assert (lidar[:4, :4]
                == 0).all() and lidar[4, 4] == 500 and (heatmap[:4, :4]
                                                        == 0).all()
        assert heatmap[4, 4].any()  # the first LiDAR point outside the mask
    sfm_image = np.asarray(
        Image.open(benchmark / "SfM" / "Sfm" / "images" / "img0_000000.jpg"))
    assert sfm_image[:4, :4].any()  # SfM keeps the photos whole


def test_depth_scenes_written_before_masking_are_masked_in_place(
        tmp_path: Path):
    make_release(tmp_path / "released", "Recon", list(range(4)))
    benchmark = run(tmp_path, ["mvrecon"])
    scene = benchmark / "MVRecon" / "Recon"
    record = json.loads((scene / "scene.json").read_text())
    del record["dynamic_objects_masked"]
    (scene / "scene.json").write_text(json.dumps(record))
    image = next((scene / "images").iterdir())
    lidar = next((scene / "lidar_depths").iterdir())
    source = next((tmp_path / "released").glob("Recon-*")) / "convert_final"
    match = mm.FRAME_PATTERN.match(image.stem)
    original = f"cam{match.group(1)}/img{match.group(1)}_{int(match.group(2))}"
    mm.copy_file(source / "images" / f"{original}.JPEG",
                 image)  # as an unmasked scene had them
    mm.copy_file(source / "lidar_depth" / f"{original}.png", lidar)
    depths = sorted(
        path.stat().st_mtime_ns for path in (scene / "depths").iterdir())
    run(tmp_path, ["mvrecon"])
    assert json.loads(
        (scene / "scene.json").read_text())["dynamic_objects_masked"]
    assert masked_to_black(np.asarray(Image.open(image))) and (np.asarray(
        Image.open(lidar))[:4, :4] == 0).all()
    assert sorted(
        path.stat().st_mtime_ns
        for path in (scene / "depths").iterdir()) == depths  # not rebuilt


def test_the_manual_review_is_applied_on_every_run(tmp_path: Path):
    make_release(tmp_path / "released", "Mono", list(range(10)))
    make_release(tmp_path / "released", "Video", list(range(8)))
    make_release(tmp_path / "released", "Sfm", [0, 1], gggs=False)
    benchmark = run(tmp_path, ["monodepth", "videodepth", "sfm"])
    mono = benchmark / "MonoDepth" / "Mono"
    keys = [
        record["key"] for record in json.loads((
            mono / "materialized_full.json").read_text())["records"]
    ]
    clips = sorted(
        path.name for path in (benchmark / "VideoDepth").glob("Video_clip*"))
    assert len(keys) >= 2 and len(clips) >= 2
    (tmp_path / "review.json").write_text(
        json.dumps({
            "tracks": {
                "MonoDepth": {
                    "Mono": {
                        "drop": [keys[0]],
                        "reviewed": True
                    }
                },
                "VideoDepth": {
                    clips[0]: {
                        "drop": ["000001"]
                    },
                    clips[1]: {
                        "keep": False
                    }
                },
                "SfM": {
                    "Sfm": {
                        "keep": False
                    }
                },
            }
        }))

    for extra in (
        [], ["--overwrite"]
    ):  # an existing scene is reviewed in place; a rebuilt one comes out reviewed
        run(tmp_path, ["monodepth", "videodepth", "sfm"], *extra)
        manifest = json.loads((mono / "materialized_full.json").read_text())
        assert [record["key"] for record in manifest["records"]] == keys[1:]
        assert manifest["num_samples"] == len(keys) - 1
        assert not list(mono.rglob(f"{keys[0]}.*"))
        record = json.loads((mono / "scene.json").read_text())
        assert record["total_images"] == len(keys) - 1 and record[
            "manual_review"]["excluded_frames"] == [keys[0]]
        clip = benchmark / "VideoDepth" / clips[0]
        assert sorted(path.parent.name for path in clip.rglob("000001.*")) == [
            "images"
        ]  # the clip stays whole
        frames = json.loads((clip / "frame_manifest.json").read_text())
        assert frames[1]["gt_excluded"] and frames[1][
            "depth"] is None and frames[0]["depth"] == "depths/000000.npy"
        assert json.loads(
            (clip / "scene.json").read_text())["gt_frames"] == len(frames) - 1
        assert not (benchmark / "VideoDepth" / clips[1]
                   ).exists()  # a clip left out is pruned; the rest stay
        assert all(
            (benchmark / "VideoDepth" / name).is_dir() for name in clips[2:])
        plan = json.loads((benchmark / "VideoDepth" /
                           "clips.json").read_text())["clips"]["Video"]
        assert [entry["scene_name"] for entry in plan] == [clips[0], *clips[2:]]
        assert not (benchmark / "SfM" / "Sfm").exists()
        metadata = json.loads(
            (benchmark / "MonoDepth" / "metadata.json").read_text())
        assert {
            entry["scene_name"]: entry["total_images"]
            for entry in metadata["scenes"]
        }["Mono"] == len(keys) - 1


def test_mvrecon_uses_every_image_with_views_0_and_rebuilds_when_the_view_count_changes(
        tmp_path: Path):
    make_release(tmp_path / "released", "Recon", list(range(4)))
    benchmark = run(tmp_path, ["mvrecon"])  # 6 of the 12 images
    scene = benchmark / "MVRecon" / "Recon"
    assert len(list((scene / "images").iterdir())) == 6
    run(tmp_path, ["mvrecon"],
        mvrecon={
            "views": 0,
            "scenes": {
                "Recon": {
                    "setting": "indoor",
                    "eval_threshold": 0.05
                }
            }
        })
    record = json.loads((scene / "scene.json").read_text())
    assert record["total_images"] == 12 and record["dense_depth"]
    assert len(list(
        (scene / "images").iterdir())) == len(list(
            (scene / "depths").iterdir())) == 12
    _, images, _ = read_model(str(scene / "sparse" / "0"))
    assert len(images) == 12
