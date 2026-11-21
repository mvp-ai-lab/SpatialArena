# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Benchmark dataset materializers (scripts/preprocess/materialize_*.py, consolidate_texturepoorsfm.py).

materialize_dycheck: Canonical DyCheck materialization and source pairing regressions.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from PIL import Image
from spatialarena.eval.canonical_dataset import CanonicalDataset
from scripts.preprocess.materialize_dycheck import materialize_dataset as materialize_dycheck_dataset
from scripts.preprocess.materialize_nrgbd import (
    _alphanum_key,
    export_sparse_model,
    load_focal,
    load_poses,
    materialize_dataset as materialize_nrgbd_dataset,
    opengl_c2w_to_colmap_w2c,
)
import os
from scripts.preprocess.materialize_texturepoorsfm import link_path
from pathlib import Path
from spatialarena.geometry.read_write_model import BaseImage, Camera, qvec2rotmat, read_model, rotmat2qvec, write_model
from scripts.preprocess.consolidate_texturepoorsfm import (
    DEFAULT_SPLIT_PATH,
    consolidate,
    load_split,
    materialize_benchmark,
    parse_bag_name,
    select_benchmark_bags,
    to_world_to_camera,
)
from scripts.preprocess.materialize_tum_dynamics import (
    associate,
    camera_for_model,
    freiburg_model,
    materialize_dataset as materialize_tum_dataset,
    read_file_list,
    read_groundtruth,
)
import tarfile
from scripts.preprocess.materialize_marigold_sample import materialize_nyuv2, write_marigold_metadata
from scripts.preprocess.materialize_da3_bench import (
    materialize_7scenes_gt_mesh,
    materialize_eth3d_gt_mesh,
    materialize_hiroom,
    materialize_scannetpp,
)


# ---- materialize_dycheck -----------------------------------------------------------------------------------

def make_scene(root):
    scene = root / "apple"
    (scene / "dense/rgb/1x").mkdir(parents=True)
    (scene / "depth/2x").mkdir(parents=True)
    for stem in ["0_10", "0_2", "1_0"]:
        Image.new("RGB", (8, 12)).save(scene / "dense/rgb/1x" / f"{stem}.png")
        np.save(scene / "depth/2x" / f"{stem}.npy", np.full((6, 4, 1), 2.5, dtype=np.float32))
    return scene


def test_materialize_pair_order_native_shapes_and_resume(tmp_path):
    source = tmp_path / "source"
    scene = make_scene(source)
    output = tmp_path / "benchmark"
    materialize_dycheck_dataset(source, output, scenes=["apple"])
    root = output / "VideoDepth/dycheck"
    frames = json.loads((root / "apple/frame_manifest.json").read_text())
    assert [f["source_frame"] for f in frames] == ["0_2", "0_10"]
    info = CanonicalDataset(root, task_name="VideoDepth").list_scenes()[0]
    assert info.scene == "apple"
    assert Image.open(info.image_path / "000000.png").size == (8, 12)
    np.testing.assert_array_equal(np.load(info.depth_gt_path / "000000.npy"), np.full((6, 4, 1), 2.5))
    assert (info.depth_gt_path / "000000.npy").resolve() == scene / "depth/2x/0_2.npy"
    assert str(tmp_path) not in (root / "metadata.json").read_text()
    materialize_dycheck_dataset(source, output, scenes=["apple"])
    assert len(list(info.image_path.iterdir())) == 2


def test_missing_pair_fails_before_output(tmp_path):
    source = tmp_path / "source"
    scene = make_scene(source)
    (scene / "depth/2x/0_2.npy").unlink()
    with pytest.raises(ValueError, match="Unmatched"):
        materialize_dycheck_dataset(source, tmp_path / "out", scenes=["apple"])
    assert not (tmp_path / "out/VideoDepth/dycheck/metadata.json").exists()


def test_source_change_cannot_replace_existing_dataset(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    make_scene(first)
    make_scene(second)
    materialize_dycheck_dataset(first, tmp_path / "out", scenes=["apple"])
    with pytest.raises(FileExistsError):
        materialize_dycheck_dataset(second, tmp_path / "out", scenes=["apple"])


def test_explicit_mask_preserved(tmp_path):
    source = tmp_path / "source"
    scene = make_scene(source)
    (scene / "validity").mkdir()
    for stem in ["0_2", "0_10"]:
        np.save(scene / "validity" / f"{stem}.npy", np.ones((6, 4), dtype=bool))
    materialize_dycheck_dataset(source, tmp_path / "out", scenes=["apple"], mask_subdir="validity")
    mask = tmp_path / "out/VideoDepth/dycheck/apple/masks/000000.npy"
    assert mask.is_symlink()
    assert np.load(mask).all()


def test_bad_shape_and_duplicate_stems_rejected(tmp_path):
    source = tmp_path / "source"
    scene = make_scene(source)
    Image.new("RGB", (8, 12)).save(scene / "dense/rgb/1x/0_2.jpg")
    with pytest.raises(ValueError, match="Duplicate"):
        materialize_dycheck_dataset(source, tmp_path / "out", scenes=["apple"])
    (scene / "dense/rgb/1x/0_2.jpg").unlink()
    np.save(scene / "depth/2x/0_2.npy", np.ones((4, 4)))
    with pytest.raises(ValueError, match="aspect ratios"):
        materialize_dycheck_dataset(source, tmp_path / "out", scenes=["apple"])


def test_registration_cases_and_yaml_agree():
    import yaml
    from spatialarena.eval.dataset_registry import DEFAULT_DATASET_SPECS
    from scripts.preprocess.materialize_dycheck import SCENES

    root = Path(__file__).resolve().parents[1]
    spec = next(s for s in DEFAULT_DATASET_SPECS if s.task == "VideoDepth" and s.dataset == "dycheck")
    assert spec.relative_path == Path("VideoDepth/dycheck")
    cases = (root / "scripts/test/full_standard_depth_cases.txt").read_text().splitlines()
    assert {line.split("|")[2] for line in cases if line.startswith("VideoDepth|dycheck|")} == set(SCENES)
    config = yaml.safe_load((root / "configs/benchmark/full_videodepth.yaml").read_text())
    assert {"task": "VideoDepth", "dataset": "dycheck"} in config["datasets"]


# ---- materialize_nrgbd -------------------------------------------------------------------------------------

def _write_png(path, size=(4, 4)):

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def test_alphanum_key_sorts_numeric_suffixes_correctly():
    names = ["img10.png", "img2.png", "img1.png", "img20.png"]
    assert sorted(names, key=_alphanum_key) == ["img1.png", "img2.png", "img10.png", "img20.png"]


def test_opengl_c2w_to_colmap_w2c_flips_y_and_z_for_identity_pose():
    c2w_gl = np.eye(4)
    R, t = opengl_c2w_to_colmap_w2c(c2w_gl)
    assert np.allclose(R, np.diag([1.0, -1.0, -1.0]))
    assert np.allclose(t, [0.0, 0.0, 0.0])
    assert np.isclose(np.linalg.det(R), 1.0)  # a valid rotation, not a reflection


def test_opengl_c2w_to_colmap_w2c_round_trips_camera_position():
    # A camera translated 2 units along OpenGL's +Y (up) with identity rotation.
    c2w_gl = np.eye(4)
    c2w_gl[:3, 3] = [0.0, 2.0, 0.0]
    R, t = opengl_c2w_to_colmap_w2c(c2w_gl)
    # World-to-camera: camera position recovered via -R^T is undone by R@(-t)... verify by reconstructing c2w.
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    c2w_recovered = np.linalg.inv(w2c)
    assert np.allclose(c2w_recovered[:3, 3], [0.0, 2.0, 0.0])


def test_load_poses_marks_nan_rows_invalid(tmp_path):
    poses_path = tmp_path / "poses.txt"
    poses_path.write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
        "nan nan nan nan\nnan nan nan nan\nnan nan nan nan\nnan nan nan nan\n"
        "1 0 0 1\n0 1 0 2\n0 0 1 3\n0 0 0 1\n"
    )
    poses = load_poses(poses_path)
    assert len(poses) == 3
    assert poses[0][1] is True
    assert poses[1] == (None, False)
    assert poses[2][1] is True
    assert np.allclose(poses[2][0][:3, 3], [1.0, 2.0, 3.0])


def test_load_focal_reads_single_float(tmp_path):
    path = tmp_path / "focal.txt"
    path.write_text("554.2562584220408\n")
    assert load_focal(path) == 554.2562584220408


def test_export_sparse_model_skips_invalid_poses(tmp_path):
    scene_root = tmp_path / "breakfast_room"
    _write_png(scene_root / "images" / "img0.png")
    _write_png(scene_root / "images" / "img1.png")
    poses = [(np.eye(4), True), (None, False)]
    num = export_sparse_model(
        scene_root, tmp_path / "out_sparse", focal=500.0, poses=poses,
        image_names=["img0.png", "img1.png"], overwrite=True,
    )
    assert num == 1
    assert (tmp_path / "out_sparse" / "cameras.txt").exists()
    assert (tmp_path / "out_sparse" / "images.txt").exists()


def test_materialize_nrgbd_end_to_end(tmp_path):
    source_root = tmp_path / "source"
    scene_root = source_root / "breakfast_room"
    for i in range(2):
        _write_png(scene_root / "images" / f"img{i}.png")
        _write_png(scene_root / "depth_filtered" / f"depth{i}.png")
    (scene_root / "focal.txt").write_text("500.0\n")
    (scene_root / "poses.txt").write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
        "1 0 0 1\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
    )
    mesh_dir = source_root / "meshes" / "breakfast_room"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "gt_mesh.ply").write_bytes(b"fake-ply")

    benchmark_root = tmp_path / "benchmark_root"
    result = materialize_nrgbd_dataset(source_root, benchmark_root, overwrite=True)

    assert result["num_scenes"] == 1
    out_scene = benchmark_root / "MVRecon" / "nrgbd" / "breakfast_room"
    assert (out_scene / "images" / "img0.png").exists()
    assert (out_scene / "depths" / "depth0.png").exists()
    assert (out_scene / "gt_mesh.ply").read_bytes() == b"fake-ply"
    assert (out_scene / "sparse" / "0" / "images.txt").exists()

    metadata = json.loads((benchmark_root / "MVRecon" / "nrgbd" / "metadata.json").read_text())
    assert metadata["scenes"][0]["gt_mesh_path"] == "gt_mesh.ply"
    assert metadata["scenes"][0]["total_images"] == 2


def test_materialize_dataset_skips_missing_scenes(tmp_path):
    source_root = tmp_path / "source"  # no scene directories at all
    benchmark_root = tmp_path / "benchmark_root"
    result = materialize_nrgbd_dataset(source_root, benchmark_root, overwrite=True)
    assert result["num_scenes"] == 0


# ---- materialize_texturepoorsfm ----------------------------------------------------------------------------

def _make_source_dir(tmp_path):
    src = tmp_path / "source" / "images"
    src.mkdir(parents=True)
    (src / "a.jpg").write_bytes(b"fake-image-bytes")
    return src


def test_link_path_copies_by_default(tmp_path):
    src = _make_source_dir(tmp_path)
    dst = tmp_path / "out" / "images"

    status = link_path(src, dst)

    assert status == "copied"
    assert dst.is_dir()
    assert not dst.is_symlink()
    assert (dst / "a.jpg").read_bytes() == b"fake-image-bytes"


def test_link_path_symlinks_when_copy_is_false(tmp_path):
    src = _make_source_dir(tmp_path)
    dst = tmp_path / "out" / "images"

    status = link_path(src, dst, copy=False)

    assert status == "linked"
    assert dst.is_symlink()
    assert os.path.realpath(dst) == os.path.realpath(src)


def test_link_path_copy_survives_source_removal(tmp_path):
    src = _make_source_dir(tmp_path)
    dst = tmp_path / "out" / "images"
    link_path(src, dst, copy=True)

    import shutil

    shutil.rmtree(src.parent)  # delete the whole staging source

    assert (dst / "a.jpg").read_bytes() == b"fake-image-bytes"  # still readable -- genuinely self-contained


def test_link_path_missing_source_reports_missing(tmp_path):
    status = link_path(tmp_path / "does-not-exist", tmp_path / "out")
    assert status == "missing_source"


def test_link_path_existing_dst_without_overwrite_is_skipped(tmp_path):
    src = _make_source_dir(tmp_path)
    dst = tmp_path / "out" / "images"
    dst.mkdir(parents=True)

    status = link_path(src, dst, overwrite=False)
    assert status == "exists"


def test_link_path_existing_dst_with_overwrite_replaces_it(tmp_path):
    src = _make_source_dir(tmp_path)
    dst = tmp_path / "out" / "images"
    dst.mkdir(parents=True)
    (dst / "stale.txt").write_text("old")

    status = link_path(src, dst, overwrite=True)
    assert status == "copied"
    assert (dst / "a.jpg").exists()
    assert not (dst / "stale.txt").exists()


def test_export_sparse_model_keeps_world_to_camera_pose_files(tmp_path):
    """Raw TexturePoorSfM poses/<frame>.txt are world-to-camera; they must reach COLMAP uninverted."""

    from scripts.preprocess.materialize_texturepoorsfm import export_sparse_model as export_texturepoor_model

    scene = tmp_path / "scene"
    for name in ("images", "poses", "intrins"):
        (scene / name).mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(scene / "images" / "0.png")
    angle = 0.3
    w2c = np.eye(4)
    w2c[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    w2c[:3, 3] = [0.1, -0.2, 0.5]
    np.savetxt(scene / "poses" / "0.txt", w2c)
    np.savetxt(scene / "intrins" / "0.txt", np.array([[10.0, 0, 4], [0, 10.0, 3], [0, 0, 1]]))

    assert export_texturepoor_model(scene, tmp_path / "sparse", ["0.png"], overwrite=False) == 1

    _cameras, images, _points = read_model(str(tmp_path / "sparse"), ext=".txt")
    (image,) = images.values()
    np.testing.assert_allclose(qvec2rotmat(image.qvec), w2c[:3, :3], atol=1e-9)
    np.testing.assert_allclose(image.tvec, w2c[:3, 3], atol=1e-9)


# ---- consolidate_texturepoorsfm ----------------------------------------------------------------------------

def _frame_pose(frame: int) -> tuple[np.ndarray, np.ndarray]:
    angle = 0.01 * frame
    qvec = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
    return qvec, np.array([0.1 * frame, 0.0, 1.0])


def _write_bag(root: Path, name: str, frames: list[int], tweak_frame: int | None = None) -> None:
    bag = root / name
    (bag / "images").mkdir(parents=True)
    cameras, images = {}, {}
    for image_id, frame in enumerate(frames, start=1):
        (bag / "images" / f"{frame}.png").write_bytes(f"{name.split('__')[0]}-{frame}".encode())
        qvec, tvec = _frame_pose(frame)
        if frame == tweak_frame:
            tvec = tvec + 0.5
        cameras[image_id] = Camera(image_id, "PINHOLE", 8, 8, np.array([10.0, 10.0, 4.0, 4.0]))
        images[image_id] = BaseImage(
            image_id, qvec, tvec, image_id, f"{frame}.png", np.empty((0, 2)), np.empty((0,), dtype=np.int64)
        )
    (bag / "sparse" / "0").mkdir(parents=True)
    write_model(cameras, images, {}, str(bag / "sparse" / "0"), ext=".txt")


def _write_split(path: Path, benchmark_scenes: list[str]) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "train_objects": ["1000"],
                "val_objects": ["1001"],
                "test_objects": ["1002"],
                "benchmark_scenes": benchmark_scenes,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(name="bag_roots")
def fixture_bag_roots(tmp_path: Path) -> list[Path]:
    train_root, bench_root = tmp_path / "train", tmp_path / "bench_v1"
    for object_id in ("1000", "1001", "1002"):
        _write_bag(train_root, f"{object_id}__5bag001", [0, 10, 20, 30, 40])
        _write_bag(train_root, f"{object_id}__5bag002", [20, 30, 50, 60, 70])
    # The old benchmark holds another overlapping bag of object 1000 -- the leak being fixed.
    _write_bag(bench_root, "1000__5bag019", [10, 20, 80, 90, 100])
    return [train_root, bench_root]


def test_parse_bag_name() -> None:
    assert parse_bag_name("1000__10bag017") == ("1000", 10, 17)
    assert parse_bag_name("metadata.json") is None


def test_consolidate_dedups_frames_across_roots(tmp_path: Path, bag_roots: list[Path]) -> None:
    split = _write_split(tmp_path / "split.json", [])
    pool = tmp_path / "pool"

    summary = consolidate(bag_roots, pool, split, overwrite=False, source_pose_convention="w2c")

    assert summary["num_bags"] == 7 and summary["num_objects"] == 3
    object_root = pool / "1000"
    assert sorted(p.name for p in (object_root / "images").iterdir()) == sorted(
        f"{f}.png" for f in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    )
    _cameras, images, points = read_model(str(object_root / "sparse" / "0"), ext=".txt")
    assert [image.name for image in sorted(images.values(), key=lambda i: i.id)][:3] == ["0.png", "10.png", "20.png"]
    assert points == {}
    by_name = {image.name: image for image in images.values()}
    np.testing.assert_allclose(by_name["80.png"].tvec, _frame_pose(80)[1])
    bags = json.loads((object_root / "bags.json").read_text())
    assert set(bags) == {"1000__5bag001", "1000__5bag002", "1000__5bag019"}
    assert not (tmp_path / "pool.staging").exists()


def test_consolidate_rejects_pose_conflicts(tmp_path: Path) -> None:
    root = tmp_path / "train"
    for object_id in ("1000", "1001", "1002"):
        _write_bag(root, f"{object_id}__5bag001", [0, 10])
    _write_bag(root, "1000__5bag002", [10, 20], tweak_frame=10)
    split = _write_split(tmp_path / "split.json", [])

    with pytest.raises(ValueError, match="translation differs"):
        consolidate([root], tmp_path / "pool", split, overwrite=False, source_pose_convention="w2c")
    assert not (tmp_path / "pool").exists()


def test_split_rejects_overlap_and_non_test_benchmark(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps({"train_objects": ["1000"], "val_objects": [], "test_objects": ["1000"], "benchmark_scenes": []})
    )
    with pytest.raises(ValueError, match="both train and test"):
        load_split(path)
    with pytest.raises(ValueError, match="does not belong to a test object"):
        load_split(_write_split(path, ["1000__5bag001"]))


def test_benchmark_materializes_only_listed_test_bags(tmp_path: Path, bag_roots: list[Path]) -> None:
    split = _write_split(tmp_path / "split.json", ["1002__5bag002"])
    pool = tmp_path / "pool"
    consolidate(bag_roots, pool, split, overwrite=False, source_pose_convention="w2c")

    summary = materialize_benchmark(pool, tmp_path / "benchmarks", split, overwrite=False)

    out = tmp_path / "benchmarks" / "SfM" / "texturepoorsfm"
    assert summary["num_scenes"] == 1
    assert sorted(p.name for p in out.iterdir()) == ["1002__5bag002", "metadata.json"]
    cameras, images, _ = read_model(str(out / "1002__5bag002" / "sparse" / "0"), ext=".txt")
    assert sorted(image.name for image in images.values()) == sorted(f"{f}.png" for f in (20, 30, 50, 60, 70))
    assert set(cameras) == {image.camera_id for image in images.values()}
    by_name = {image.name: image for image in images.values()}
    np.testing.assert_allclose(by_name["50.png"].tvec, _frame_pose(50)[1])
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["test_objects"] == ["1002"] and metadata["total_images"] == 5


def test_select_benchmark_bags_is_deterministic() -> None:
    bags = {"1002": [f"1002__{size}bag{i:03d}" for size in (5, 10) for i in range(6)]}
    first = select_benchmark_bags(bags, ["1002"], {5: 2, 10: 1}, seed=0)
    assert first == select_benchmark_bags(bags, ["1002"], {5: 2, 10: 1}, seed=0)
    assert [parse_bag_name(bag)[1] for bag in first] == [5, 5, 10]


def test_checked_in_split_is_object_disjoint() -> None:
    split = load_split(DEFAULT_SPLIT_PATH)
    objects = split["train_objects"] + split["val_objects"] + split["test_objects"]
    assert len(objects) == len(set(objects)) == 17
    assert not split["val_objects"]
    assert split["benchmark_scenes"], "benchmark_scenes must be populated"
    assert {parse_bag_name(bag)[0] for bag in split["benchmark_scenes"]} == set(split["test_objects"])


def test_inverted_source_poses_are_restored_to_world_to_camera() -> None:
    qvec, tvec = _frame_pose(30)
    rotation = qvec2rotmat(qvec)
    stored_q, stored_t = rotmat2qvec(rotation.T), -rotation.T @ tvec  # what the pre-fix materializer wrote
    restored_q, restored_t = to_world_to_camera(stored_q, stored_t, "inverted_w2c")
    np.testing.assert_allclose(qvec2rotmat(restored_q), rotation, atol=1e-12)
    np.testing.assert_allclose(restored_t, tvec, atol=1e-12)
    with pytest.raises(ValueError, match="Unknown pose convention"):
        to_world_to_camera(qvec, tvec, "c2w")


def test_benchmark_bags_carry_triangulated_points_restricted_to_bag(tmp_path: Path, bag_roots: list[Path]) -> None:
    from spatialarena.geometry.read_write_model import Point3D

    split = _write_split(tmp_path / "split.json", ["1002__5bag002"])
    pool = tmp_path / "pool"
    consolidate(bag_roots, pool, split, overwrite=False, source_pose_convention="w2c")
    cameras, images, _ = read_model(str(pool / "1002" / "sparse" / "0"), ext=".txt")
    by_name = {image.name: image for image in images.values()}
    # Point 1 is seen by frames 0 (not in the bag), 20 and 30; point 2 only by 20 inside the bag, plus frame 0.
    obs = {"0.png": [1, 2], "20.png": [1, 2], "30.png": [1, -1]}
    tri_images = {
        image.id: image._replace(
            xys=np.zeros((len(obs.get(image.name, [])), 2)),
            point3D_ids=np.array(obs.get(image.name, []), dtype=np.int64),
        )
        for image in images.values()
    }
    tracks = {1: [], 2: []}
    for image in tri_images.values():
        for idx, pid in enumerate(image.point3D_ids):
            if pid >= 0:
                tracks[int(pid)].append((image.id, idx))
    points = {
        pid: Point3D(pid, np.array([0.0, 0.0, 1.0]), np.array([1, 2, 3]), 0.1,
                     np.array([t[0] for t in track]), np.array([t[1] for t in track]))
        for pid, track in tracks.items()
    }
    (pool / "1002" / "sparse" / "dfsfm_tri").mkdir()
    write_model(cameras, tri_images, points, str(pool / "1002" / "sparse" / "dfsfm_tri"), ext=".txt")

    summary = materialize_benchmark(pool, tmp_path / "benchmarks", split, overwrite=False, sparse_model="dfsfm_tri")

    assert summary["num_scenes"] == 1
    _c, bag_images, bag_points = read_model(
        str(tmp_path / "benchmarks" / "SfM" / "texturepoorsfm" / "1002__5bag002" / "sparse" / "0"), ext=".txt"
    )
    assert set(bag_points) == {1}  # point 2 has a single in-bag observation
    bag_by_name = {image.name: image for image in bag_images.values()}
    assert sorted(bag_by_name[n].id for n in ("20.png", "30.png")) == sorted(bag_points[1].image_ids.tolist())
    assert list(bag_by_name["20.png"].point3D_ids) == [1, -1]
    np.testing.assert_allclose(bag_by_name["20.png"].tvec, by_name["20.png"].tvec)


# ---- materialize_tum_dynamics ------------------------------------------------------------------------------

def _write_tum_png(path, size=(4, 4)):

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(64, 64, 64)).save(path)


def test_freiburg_model_detects_camera_from_sequence_name():
    assert freiburg_model("rgbd_dataset_freiburg2_desk_with_person") == "freiburg2"
    assert freiburg_model("rgbd_dataset_freiburg3_walking_xyz") == "freiburg3"


def test_freiburg_model_raises_for_unknown_sequence():

    with pytest.raises(ValueError):
        freiburg_model("rgbd_dataset_freiburg1_xyz")


def test_camera_for_model_uses_full_opencv_for_distorted_freiburg2():
    cam = camera_for_model("freiburg2", width=640, height=480, camera_id=1)
    assert cam.model == "FULL_OPENCV"
    assert len(cam.params) == 12


def test_camera_for_model_uses_pinhole_for_undistorted_freiburg3():
    cam = camera_for_model("freiburg3", width=640, height=480, camera_id=1)
    assert cam.model == "PINHOLE"
    assert len(cam.params) == 4


def test_associate_matches_nearest_timestamps_within_threshold():
    first = {1.000: "a", 2.000: "b", 5.000: "c"}
    second = {1.010: "a2", 2.500: "b2"}  # b2 is 0.5s off, outside default threshold
    matches = associate(first, second, max_difference=0.02)
    assert matches == [(1.000, 1.010)]


def test_associate_does_not_reuse_a_timestamp_twice():
    first = {1.000: "a", 1.005: "b"}
    second = {1.002: "x"}
    matches = associate(first, second, max_difference=0.02)
    assert len(matches) == 1  # only one of {a, b} can claim x


def test_read_file_list_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "rgb.txt"
    path.write_text("# comment\n\n1.000000 rgb/1.png\n2.000000 rgb/2.png\n")
    result = read_file_list(path)
    assert result == {1.0: "rgb/1.png", 2.0: "rgb/2.png"}


def test_read_groundtruth_identity_quaternion_gives_identity_rotation(tmp_path):
    path = tmp_path / "groundtruth.txt"
    path.write_text("1.000000 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
    poses = read_groundtruth(path)
    c2w = poses[1.0]
    assert np.allclose(c2w[:3, :3], np.eye(3))
    assert np.allclose(c2w[:3, 3], [0.0, 0.0, 0.0])


def test_materialize_tum_dynamics_end_to_end(tmp_path):
    source_root = tmp_path / "source"
    seq_root = source_root / "rgbd_dataset_freiburg3_walking_xyz"
    for ts in [1.000, 1.033, 1.066]:
        _write_tum_png(seq_root / "rgb" / f"{ts:.6f}.png")
        _write_tum_png(seq_root / "depth" / f"{ts:.6f}.png")
    (seq_root / "rgb.txt").write_text(
        "\n".join(f"{ts:.6f} rgb/{ts:.6f}.png" for ts in [1.000, 1.033, 1.066])
    )
    (seq_root / "depth.txt").write_text(
        "\n".join(f"{ts:.6f} depth/{ts:.6f}.png" for ts in [1.000, 1.033, 1.066])
    )
    (seq_root / "groundtruth.txt").write_text(
        "\n".join(f"{ts:.6f} 0.0 0.0 0.0 0.0 0.0 0.0 1.0" for ts in [1.000, 1.033, 1.066])
    )

    benchmark_root = tmp_path / "benchmark_root"
    result = materialize_tum_dataset(source_root, benchmark_root, overwrite=True)

    assert result["num_scenes"] == 1
    out_scene = benchmark_root / "SLAM_source" / "tum_dynamics" / "rgbd_dataset_freiburg3_walking_xyz"
    assert (out_scene / "images" / "000001.png").exists()
    assert (out_scene / "depths" / "000001.png").exists()
    assert (out_scene / "sparse" / "0" / "images.txt").exists()
    assert (out_scene / "sparse" / "0" / "cameras.txt").read_text().count("PINHOLE") == 3

    metadata = json.loads((benchmark_root / "SLAM_source" / "tum_dynamics" / "metadata.json").read_text())
    assert metadata["scenes"][0]["total_images"] == 3
    assert metadata["scenes"][0]["camera_model"] == "freiburg3"


def test_materialize_dataset_skips_sequences_missing_required_files(tmp_path):
    source_root = tmp_path / "source"
    (source_root / "rgbd_dataset_freiburg3_walking_xyz").mkdir(parents=True)  # no rgb.txt/depth.txt/groundtruth.txt
    benchmark_root = tmp_path / "benchmark_root"
    result = materialize_tum_dataset(source_root, benchmark_root, overwrite=True)
    assert result["num_scenes"] == 0


# ---- marigold_materialization ------------------------------------------------------------------------------

def add_png(tar: tarfile.TarFile, tmp_path: Path, arcname: str, color: tuple[int, int, int]) -> None:
    image_path = tmp_path / arcname.replace("/", "_")
    Image.new("RGB", (2, 2), color=color).save(image_path)
    tar.add(image_path, arcname=arcname)


def test_materialize_nyuv2_writes_matching_canonical_files(tmp_path):
    marigold_root = tmp_path / "marigold"
    archive = marigold_root / "nyuv2" / "nyu_labeled_extracted.tar"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as tar:
        add_png(tar, tmp_path, "./test/kitchen_0004/rgb_0001.png", (255, 0, 0))
        add_png(tar, tmp_path, "./test/kitchen_0004/depth_0001.png", (1, 0, 0))
        add_png(tar, tmp_path, "./test/kitchen_0004/filled_0001.png", (2, 0, 0))
        add_png(tar, tmp_path, "./test/office_0005/rgb_0009.png", (0, 255, 0))
        add_png(tar, tmp_path, "./test/office_0005/depth_0009.png", (3, 0, 0))

    manifest = materialize_nyuv2(
        marigold_root=marigold_root,
        max_samples=1,
        overwrite=False,
    )

    assert manifest["num_samples"] == 1
    record = manifest["records"][0]
    assert not Path(record["image"]).is_absolute() and not Path(manifest["source_archive"]).is_absolute()
    image_path = marigold_root / "nyuv2" / record["image"]  # manifests store scene-relative paths
    depth_path = marigold_root / "nyuv2" / record["depth"]
    assert image_path.name == depth_path.name
    assert image_path.is_file()
    assert depth_path.is_file()
    assert record["intrinsics"][0][0] > 0
    assert (marigold_root / "nyuv2" / "materialized_sample.json").is_file()


def test_write_marigold_metadata_adds_a_scene_to_existing_metadata(tmp_path):
    def manifest(scene, count):
        return {"scene": scene, "num_samples": count, "source_archive": f"{scene}.tar"}

    write_marigold_metadata(tmp_path, [manifest("nyuv2", 2), manifest("kitti", 3)])
    metadata = write_marigold_metadata(tmp_path, [manifest("scannet", 4)])

    assert [s["scene_name"] for s in metadata["scenes"]] == ["nyuv2", "kitti", "scannet"]
    assert metadata["total_images"] == 9
    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata


# ---- materialize_da3_bench ---------------------------------------------------------------------------------

def test_materialize_hiroom_exports_canonical_layout_and_sparse_model(tmp_path):
    source = tmp_path / "da3"
    scene_rel = Path("building/floor/room")
    scene_root = source / "hiroom" / "data" / scene_rel
    image_root = scene_root / "image"
    depth_root = scene_root / "depth"
    mask_root = scene_root / "aliasing_mask"
    pose_root = scene_root / "pose"
    gt_root = source / "hiroom" / "fused_pcd"
    for path in (image_root, depth_root, mask_root, pose_root, gt_root):
        path.mkdir(parents=True)

    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(image_root / "000.png")
    Image.fromarray(np.ones((2, 3), dtype=np.uint16)).save(depth_root / "000.png")
    Image.fromarray(np.ones((2, 3), dtype=np.uint8)).save(mask_root / "000.png")
    np.save(scene_root / "cam_K.npy", np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]))
    np.save(pose_root / "000.npy", np.eye(4))
    (gt_root / "building-floor-room.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
        "end_header\n",
        encoding="utf-8",
    )
    (source / "hiroom" / "selected_scene_list_val.txt").write_text(f"{scene_rel.as_posix()}\n", encoding="utf-8")

    summary = materialize_hiroom(source, tmp_path / "benchmarks", overwrite=False)

    scene_out = tmp_path / "benchmarks" / "MVRecon" / "hiroom" / "building__floor__room"
    metadata = json.loads((scene_out.parent / "metadata.json").read_text(encoding="utf-8"))
    assert summary["num_scenes"] == 1
    assert (scene_out / "images").is_dir() and not (scene_out / "images").is_symlink()
    assert (scene_out / "depths").is_dir() and not (scene_out / "depths").is_symlink()
    assert (scene_out / "masks").is_dir() and not (scene_out / "masks").is_symlink()
    assert (scene_out / "images" / "000.png").is_file()
    assert (scene_out / "sparse" / "0" / "images.txt").is_file()
    assert metadata["scenes"][0]["gt_point_path"] == "gt_pointcloud.ply"
    assert (scene_out / "gt_pointcloud.ply").is_file() and not (scene_out / "gt_pointcloud.ply").is_symlink()


def test_materialize_scannetpp_links_da3_bench_scene_layout(tmp_path):
    source = tmp_path / "da3"
    scene = "09c1414f1b"
    scene_root = source / "scannetpp" / scene
    input_root = scene_root / "merge_dslr_iphone"
    image_root = input_root / "images"
    depth_root = input_root / "render_depth"
    sparse_root = input_root / "colmap" / "sparse_render_rgb"
    mesh_root = scene_root / "scans"
    for path in (image_root, depth_root, sparse_root, mesh_root):
        path.mkdir(parents=True)

    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(image_root / "000.png")
    Image.fromarray(np.ones((2, 3), dtype=np.uint16)).save(depth_root / "000.png")
    (sparse_root / "cameras.txt").write_text("# cameras\n", encoding="utf-8")
    (sparse_root / "images.txt").write_text("# images\n", encoding="utf-8")
    (sparse_root / "points3D.txt").write_text("# points\n", encoding="utf-8")
    (mesh_root / "mesh_aligned_0.05.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
        "end_header\n",
        encoding="utf-8",
    )

    summary = materialize_scannetpp(source, tmp_path / "benchmarks", overwrite=False)

    scene_out = tmp_path / "benchmarks" / "MVRecon" / "scannetpp" / scene
    metadata = json.loads((scene_out.parent / "metadata.json").read_text(encoding="utf-8"))
    assert summary["num_scenes"] == 1
    assert (scene_out / "images").is_dir() and not (scene_out / "images").is_symlink()
    assert (scene_out / "depths").is_dir() and not (scene_out / "depths").is_symlink()
    assert (scene_out / "sparse" / "0").is_dir() and not (scene_out / "sparse" / "0").is_symlink()
    assert (scene_out / "sparse" / "0" / "cameras.txt").is_file()
    assert (scene_out / "gt_mesh.ply").is_file() and not (scene_out / "gt_mesh.ply").is_symlink()
    assert metadata["scenes"][0]["gt_mesh_path"] == "gt_mesh.ply"


def test_materialize_7scenes_gt_mesh_attaches_to_existing_layout(tmp_path):
    # 7scenes images/depths/sparse are already materialized by
    # scripts/download/7scenes.sh from the original Microsoft release, not
    # by this script; materialize_7scenes_gt_mesh only attaches the
    # DA3-BENCH ground-truth mesh to that existing metadata.json.
    benchmark_root = tmp_path / "benchmarks"
    scene_out = benchmark_root / "MVRecon" / "7scenes" / "heads"
    scene_out.mkdir(parents=True)
    metadata_path = scene_out.parent / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_name": "7scenes",
                "scenes": [
                    {
                        "dataset_name": "7scenes",
                        "scene_name": "heads",
                        "root_directory": str(scene_out),
                        "total_images": 2000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source = tmp_path / "da3" / "7scenes"
    mesh_root = source / "7Scenes" / "meshes"
    mesh_root.mkdir(parents=True)
    (mesh_root / "heads.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
        "end_header\n",
        encoding="utf-8",
    )

    summary = materialize_7scenes_gt_mesh(source, benchmark_root, overwrite=False)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert summary["num_scenes"] == 1
    assert (scene_out / "gt_mesh.ply").is_file() and not (scene_out / "gt_mesh.ply").is_symlink()
    assert metadata["scenes"][0]["gt_mesh_path"] == "gt_mesh.ply"
    # Fields unrelated to the mesh are preserved untouched.
    assert metadata["scenes"][0]["total_images"] == 2000


def test_materialize_eth3d_gt_mesh_attaches_available_scenes_only(tmp_path):
    # ETH3D images/depths/calibration are already materialized by
    # scripts/download/eth3d.sh and scripts/preprocess/finalize_mvrecon_layout.sh,
    # not by this script.
    # DA3-BENCH does not cover every ETH3D scene (e.g. "meadow"), so a
    # scene missing from the source must be reported, not fail the batch.
    benchmark_root = tmp_path / "benchmarks"
    output_root = benchmark_root / "MVRecon" / "eth3d"
    for scene in ("courtyard", "meadow"):
        (output_root / scene).mkdir(parents=True)
    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_name": "eth3d",
                "scenes": [
                    {"dataset_name": "eth3d", "scene_name": "courtyard", "total_images": 76},
                    {"dataset_name": "eth3d", "scene_name": "meadow", "total_images": 30},
                ],
            }
        ),
        encoding="utf-8",
    )

    source = tmp_path / "da3" / "eth3d"
    (source / "courtyard").mkdir(parents=True)
    (source / "courtyard" / "combined_mesh.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
        "end_header\n",
        encoding="utf-8",
    )

    summary = materialize_eth3d_gt_mesh(source, benchmark_root, overwrite=False)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scenes_by_name = {s["scene_name"]: s for s in metadata["scenes"]}
    assert summary["num_scenes"] == 2
    assert (output_root / "courtyard" / "gt_mesh.ply").is_file()
    assert not (output_root / "courtyard" / "gt_mesh.ply").is_symlink()
    assert scenes_by_name["courtyard"]["gt_mesh_path"] == "gt_mesh.ply"
    assert "gt_mesh_path" not in scenes_by_name["meadow"]
    assert not (output_root / "meadow" / "gt_mesh.ply").exists()
