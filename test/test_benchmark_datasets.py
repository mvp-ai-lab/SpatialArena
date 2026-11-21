# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Benchmark dataset layout: canonical dataset adapters, metadata/layout normalization and verification.
"""

import json
from pathlib import Path
import pytest
from spatialarena.eval.bonn_dataset import DatasetBonn
from spatialarena.eval.canonical_dataset import CanonicalDataset
from spatialarena.eval.imc_dataset import DatasetIMC
from spatialarena.eval.kitti_dataset import DatasetKitti
from spatialarena.eval.marigold_dataset import DatasetMarigold
from spatialarena.eval.pose_datasets import DatasetEth3DMVRecon
from spatialarena.eval.seven_scenes_dataset import DatasetSevenScenes
from spatialarena.eval.sintel_dataset import DatasetSintel
import tarfile
from scripts.preprocess.generate_benchmark_metadata import build_metadata
from scripts.preprocess.normalize_benchmark_layout import normalize_bonn, normalize_imc
from scripts.preprocess.verify_benchmark_datasets import verify_records
from spatialarena.eval.dataset_registry import describe_benchmark_scenes
from scripts.preprocess.verify_benchmark_datasets import READY_REQUIREMENTS


def only(items):
    """The one element of ``items``."""
    items = list(items)
    assert len(items) == 1, items
    return items[0]


# ---- canonical_dataset_adapters ----------------------------------------------------------------------------

def write_metadata(root: Path, dataset_name: str, scene_roots: list[Path]) -> None:
    metadata = {
        "dataset_name": dataset_name,
        "root_directory": str(root),
        "total_images": len(scene_roots),
        "average_images_per_scene": 1,
        "scenes": [
            {
                "dataset_name": dataset_name,
                "scene_name": scene_root.name,
                "root_directory": str(scene_root),
                "total_images": 1,
            }
            for scene_root in scene_roots
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_canonical_dataset_lists_materialized_scene_paths(tmp_path):
    root = tmp_path / "custom"
    scene = root / "scene_a"
    (scene / "images").mkdir(parents=True)
    (scene / "depths").mkdir()
    (scene / "masks").mkdir()
    (scene / "sparse" / "0").mkdir(parents=True)
    write_metadata(root, "custom", [scene])

    scene_info = only(CanonicalDataset(root, task_name="VideoDepth").list_scenes())

    assert scene_info.dataset == "custom"
    assert scene_info.task == "VideoDepth"
    assert scene_info.scene == "scene_a"
    assert scene_info.root_path == scene
    assert scene_info.image_path == scene / "images"
    assert scene_info.depth_gt_path == scene / "depths"
    assert scene_info.mask_path == scene / "masks"
    assert scene_info.sparse_gt_path == scene / "sparse" / "0"
    assert scene_info.metadata["total_images"] == 1


def test_canonical_dataset_filters_scenes(tmp_path):
    root = tmp_path / "custom"
    keep = root / "keep"
    skip = root / "skip"
    keep.mkdir(parents=True)
    skip.mkdir()
    write_metadata(root, "custom", [keep, skip])

    scenes = CanonicalDataset(root, scenes=["keep"]).list_scenes()

    assert [scene.scene for scene in scenes] == ["keep"]


def test_canonical_dataset_prefers_scene_root_under_dataset(tmp_path):
    root = tmp_path / "bonn"
    raw_scene = root / "rgbd_bonn_dataset" / "scene_a"
    canonical_scene = root / "scene_a"
    (raw_scene / "rgb_110").mkdir(parents=True)
    (canonical_scene / "images").mkdir(parents=True)
    write_metadata(root, "bonn", [raw_scene])

    scene_info = only(CanonicalDataset(root).list_scenes())

    assert scene_info.root_path == canonical_scene
    assert scene_info.image_path == canonical_scene / "images"


def test_eth3d_mvrecon_uses_calibration_colmap_model(tmp_path):
    root = tmp_path / "eth3d"
    scene = root / "courtyard"
    (scene / "images").mkdir(parents=True)
    (scene / "images" / "dslr_images_undistorted").mkdir()
    calibration = scene / "dslr_calibration_undistorted"
    calibration.mkdir()
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        (calibration / filename).touch()
    write_metadata(root, "eth3d", [scene])

    scene_info = only(DatasetEth3DMVRecon(root).list_scenes())

    assert scene_info.task == "MVRecon"
    assert scene_info.sparse_gt_path == calibration
    assert scene_info.image_path == scene / "images" / "dslr_images_undistorted"


@pytest.mark.parametrize(
    ("dataset_cls", "dataset_name", "task_name"),
    [
        (DatasetBonn, "bonn", "VideoDepth"),
        (DatasetIMC, "imc", "SfM"),
        (DatasetKitti, "kitti", "VideoDepth"),
        (DatasetMarigold, "marigold", "MonoDepth"),
        (DatasetSevenScenes, "7scenes", "MVRecon"),
        (DatasetSintel, "sintel", "VideoDepth"),
    ],
)
def test_dataset_wrappers_set_dataset_and_task_names(
    tmp_path,
    dataset_cls,
    dataset_name,
    task_name,
):
    root = tmp_path / dataset_name
    scene = root / "scene_a"
    (scene / "images").mkdir(parents=True)
    write_metadata(root, dataset_name, [scene])

    scene_info = only(dataset_cls(root).list_scenes())

    assert scene_info.dataset == dataset_name
    assert scene_info.task == task_name
    assert scene_info.image_path == scene / "images"


# ---- benchmark_metadata_layout -----------------------------------------------------------------------------

def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def test_metadata_prefers_canonical_bonn_images(tmp_path):
    root = tmp_path / "bonn"
    scene = root / "rgbd_bonn_dataset" / "rgbd_bonn_test"

    touch(scene / "rgb" / "raw_000.png")
    touch(scene / "rgb" / "raw_001.png")
    touch(scene / "rgb_110" / "eval_000.png")
    touch(scene / "depth_110" / "eval_000.png")

    metadata = build_metadata(root)

    assert metadata["dataset_name"] == "bonn"
    assert metadata["total_images"] == 1
    assert metadata["average_images_per_scene"] == 1
    assert metadata["scenes"][0]["scene_name"] == "rgbd_bonn_test"


def test_metadata_prefers_imc_dense_images(tmp_path):
    root = tmp_path / "imc"
    scene = root / "scene_a"

    touch(scene / "dense" / "images" / "000.jpg")
    touch(scene / "dense" / "images" / "001.jpg")
    touch(scene / "dense" / "stereo" / "depth_maps_clean_300_th_0.10" / "000.png")

    metadata = build_metadata(root)

    assert metadata["total_images"] == 2
    assert metadata["scenes"][0]["total_images"] == 2


def test_normalize_bonn_creates_canonical_links(tmp_path):
    root = tmp_path / "bonn"
    scene = root / "rgbd_bonn_dataset" / "rgbd_bonn_test"
    (scene / "rgb_110").mkdir(parents=True)
    (scene / "depth_110").mkdir(parents=True)

    stats = normalize_bonn(root)

    assert stats["copied"] == 2
    assert (root / "rgbd_bonn_test" / "images").is_dir()
    assert not (root / "rgbd_bonn_test" / "images").is_symlink()
    assert (root / "rgbd_bonn_test" / "depths").is_dir()
    assert not (root / "rgbd_bonn_test" / "depths").is_symlink()


def test_normalize_imc_links_images_depths_and_sparse(tmp_path):
    root = tmp_path / "imc"
    scene = root / "scene_a"
    (scene / "dense" / "images").mkdir(parents=True)
    (scene / "dense" / "sparse").mkdir(parents=True)
    (scene / "dense" / "stereo" / "depth_maps_clean_300_th_0.10").mkdir(parents=True)

    stats = normalize_imc(root)

    assert stats["copied"] == 3
    assert (scene / "images").is_dir() and not (scene / "images").is_symlink()
    assert (scene / "depths").is_dir() and not (scene / "depths").is_symlink()
    assert (scene / "sparse" / "0").is_dir() and not (scene / "sparse" / "0").is_symlink()


def test_metadata_json_shape_is_serializable(tmp_path):
    root = tmp_path / "imc"
    touch(root / "scene_a" / "dense" / "images" / "000.jpg")

    metadata = build_metadata(root)

    encoded = json.dumps(metadata)
    assert "average_images_per_scene" in encoded


def test_metadata_counts_images_inside_tar_archives(tmp_path):
    root = tmp_path / "marigold"
    subset = root / "kitti"
    subset.mkdir(parents=True)
    tar_path = subset / "kitti_test.tar"

    image_file = tmp_path / "image.png"
    depth_file = tmp_path / "depth.png"
    image_file.write_bytes(b"image")
    depth_file.write_bytes(b"depth")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(image_file, arcname="./scene/image_02/data/0000000001.png")
        tar.add(depth_file, arcname="./scene/depth/data/0000000001.png")

    metadata = build_metadata(root)

    assert metadata["total_images"] == 1
    assert metadata["scenes"][0]["scene_name"] == "kitti"


# ---- verify_benchmark_datasets -----------------------------------------------------------------------------

def write_verify_metadata(root: Path, dataset_name: str, scene_roots: list[Path]) -> None:
    metadata = {
        "dataset_name": dataset_name,
        "root_directory": str(root),
        "total_images": len(scene_roots),
        "average_images_per_scene": 1,
        "scenes": [
            {
                "dataset_name": dataset_name,
                "scene_name": scene_root.name,
                "root_directory": str(scene_root),
                "total_images": 1,
            }
            for scene_root in scene_roots
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_verify_ready_video_dataset_passes(tmp_path):
    dataset_root = tmp_path / "VideoDepth" / "bonn"
    scene = dataset_root / "scene_a"
    (scene / "images").mkdir(parents=True)
    (scene / "depths").mkdir()
    write_verify_metadata(dataset_root, "bonn", [scene])

    records = describe_benchmark_scenes(
        benchmark_root=tmp_path,
        tasks=["VideoDepth"],
        datasets=["bonn"],
    )

    assert verify_records(records) == []


def test_verify_ready_video_dataset_reports_missing_depths(tmp_path):
    dataset_root = tmp_path / "VideoDepth" / "bonn"
    scene = dataset_root / "scene_a"
    (scene / "images").mkdir(parents=True)
    write_verify_metadata(dataset_root, "bonn", [scene])

    records = describe_benchmark_scenes(
        benchmark_root=tmp_path,
        tasks=["VideoDepth"],
        datasets=["bonn"],
    )
    issues = verify_records(records)

    assert [issue.level for issue in issues] == ["error"]
    assert "missing canonical depths" in issues[0].message


def test_removed_rmvd_task_is_not_registered(tmp_path):
    records = describe_benchmark_scenes(
        benchmark_root=tmp_path,
        tasks=["RMVD"],
        datasets=["dtu"],
    )
    assert records == []


def test_missing_mvrecon_dataset_root_is_reported(tmp_path):
    records = describe_benchmark_scenes(
        benchmark_root=tmp_path,
        tasks=["MVRecon"],
        datasets=["dtu"],
    )
    issues = verify_records(records)

    assert len(issues) == 1
    assert issues[0].level == "error"
    assert issues[0].message == f"dataset root missing: {tmp_path / 'MVRecon' / 'dtu'}"


def test_verify_only_ready_reports_expected_issues(tmp_path):
    records = describe_benchmark_scenes(benchmark_root=tmp_path)

    issues = verify_records(records, only_ready=True)

    assert issues
    assert all((issue.task, issue.dataset) in READY_REQUIREMENTS for issue in issues)
