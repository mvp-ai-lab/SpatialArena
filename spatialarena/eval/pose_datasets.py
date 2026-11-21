# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Dataset adapters for pose-only benchmark cases."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from spatialarena.eval.canonical_dataset import BenchmarkSceneInfo, CanonicalDataset
from spatialarena.eval.result_layout import has_colmap_model_files


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def count_images(image_path: Path | None) -> int:
    if image_path is None or not image_path.is_dir():
        return 0
    return sum(
        1
        for path in image_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def optional_colmap_model(path: Path) -> Path | None:
    return path if has_colmap_model_files(path) else None


class DatasetEth3DPose(CanonicalDataset):
    """Expose ETH3D calibration folders as sparse COLMAP ground truth."""

    dataset_name = "eth3d"
    task_name = "SfM"

    def list_scenes(self) -> list[BenchmarkSceneInfo]:
        scene_infos = []
        for scene_info in super().list_scenes():
            sparse_gt_path = (
                optional_colmap_model(scene_info.root_path / "sparse" / "0")
                or optional_colmap_model(scene_info.root_path / "dslr_calibration_undistorted")
                or optional_colmap_model(scene_info.root_path / "dslr_calibration_jpg")
            )
            image_path = scene_info.image_path
            eth3d_undistorted = scene_info.root_path / "images" / "dslr_images_undistorted"
            if eth3d_undistorted.is_dir():
                image_path = eth3d_undistorted
            scene_infos.append(
                replace(scene_info, image_path=image_path, sparse_gt_path=sparse_gt_path)
            )
        return scene_infos


class DatasetEth3DMVRecon(DatasetEth3DPose):
    """Use ETH3D calibration folders for MVRecon evaluation."""

    task_name = "MVRecon"


class DatasetColmapSceneRoots(CanonicalDataset):
    """List child scene directories that already contain images and sparse/0."""

    dataset_name = ""
    task_name = "SfM"

    def list_scenes(self) -> list[BenchmarkSceneInfo]:
        scene_infos: list[BenchmarkSceneInfo] = []
        if not self.data_path.is_dir():
            return scene_infos

        for scene_root in sorted(child for child in self.data_path.iterdir() if child.is_dir()):
            scene_name = scene_root.name
            if self.scenes and scene_name not in self.scenes:
                continue

            image_path = self._optional_path(scene_root / "images")
            sparse_gt_path = next(
                (
                    candidate
                    for candidate in (
                        scene_root / "sparse" / "0",
                        scene_root / "distorted" / "sparse" / "0",
                    )
                    if optional_colmap_model(candidate) is not None
                ),
                None,
            )
            if image_path is None or sparse_gt_path is None:
                continue

            total_images = count_images(image_path)
            scene_infos.append(
                BenchmarkSceneInfo(
                    dataset=self.dataset_name or self.data_path.name,
                    task=self.task_name,
                    scene=scene_name,
                    root_path=scene_root,
                    image_path=image_path,
                    depth_gt_path=self._optional_path(scene_root / "depths"),
                    mask_path=self._optional_path(scene_root / "masks"),
                    sparse_gt_path=sparse_gt_path,
                    metadata={
                        "dataset_name": self.dataset_name or self.data_path.name,
                        "scene_name": scene_name,
                        "root_directory": str(scene_root),
                        "total_images": total_images,
                    },
                )
            )
        return scene_infos


class DatasetTanksAndTemplesPose(DatasetColmapSceneRoots):
    dataset_name = "tanks_and_temples"
    task_name = "SfM"


class DatasetTanksAndTemplesMVRecon(DatasetColmapSceneRoots):
    dataset_name = "tanks_and_temples"
    task_name = "MVRecon"

    def list_scenes(self) -> list[BenchmarkSceneInfo]:
        scene_infos = []
        for scene_info in super().list_scenes():
            metadata = dict(scene_info.metadata or {})
            gt_mesh_path = scene_info.root_path / f"{scene_info.scene}.ply"
            if gt_mesh_path.is_file():
                metadata["gt_mesh_path"] = gt_mesh_path.name
            scene_infos.append(replace(scene_info, metadata=metadata))
        return scene_infos


class DatasetDTUPose(DatasetColmapSceneRoots):
    dataset_name = "dtu"
    task_name = "SfM"


class DatasetSevenScenesPose(CanonicalDataset):
    """7Scenes pose adapter using generated sparse/0 COLMAP ground truth."""

    dataset_name = "7scenes"
    task_name = "SfM"

    def list_scenes(self) -> list[BenchmarkSceneInfo]:
        scene_infos = []
        for scene_info in super().list_scenes():
            scene_infos.append(
                replace(
                    scene_info,
                    task=self.task_name,
                    sparse_gt_path=optional_colmap_model(scene_info.root_path / "sparse" / "0"),
                )
            )
        return scene_infos


class DatasetSevenScenesSLAM(DatasetSevenScenesPose):
    """7Scenes SLAM adapter: one case per official test sequence (``SLAM/7scenes/<scene>_seq-NN``, written by
    scripts/preprocess/materialize_7scenes_slam.py), each a single continuous recording with its own sparse/0."""

    task_name = "SLAM"


class DatasetTUMDynamicsSLAM(CanonicalDataset):
    """TUM RGB-D dynamic-objects SLAM adapter using generated sparse/0
    COLMAP ground truth (converted from groundtruth.txt's camera-to-world
    poses by materialize_tum_dynamics.py)."""

    dataset_name = "tum_dynamics"
    task_name = "SLAM"

    def list_scenes(self) -> list[BenchmarkSceneInfo]:
        scene_infos = []
        for scene_info in super().list_scenes():
            scene_infos.append(
                replace(
                    scene_info,
                    task=self.task_name,
                    sparse_gt_path=optional_colmap_model(scene_info.root_path / "sparse" / "0"),
                )
            )
        return scene_infos


class DatasetOxfordSpiresSLAM(DatasetTUMDynamicsSLAM):
    """Oxford Spires SLAM adapter: one case per sequence (``SLAM/oxford_spires/<sequence>``, written by
    scripts/preprocess/materialize_oxford_spires.py with LingBot-Map's preprocessing), rectified cam0 frames with
    their sparse/0 GT."""

    dataset_name = "oxford_spires"


class DatasetVBRSLAM(DatasetTUMDynamicsSLAM):
    """VBR (Vision Benchmark in Rome) SLAM adapter: one case per training sequence (``SLAM/vbr/<sequence>``, written by
    scripts/preprocess/materialize_vbr.py from LoGeR's processed release), with its sparse/0 GT."""

    dataset_name = "vbr"


class DatasetKITTIOdometrySLAM(DatasetTUMDynamicsSLAM):
    """KITTI Odometry SLAM adapter: one case per sequence with public GT, 00-10 (``SLAM/kitti_odometry/<NN>``, written
    by scripts/preprocess/materialize_kitti_odometry.py), the left color camera with its cam2 sparse/0 GT."""

    dataset_name = "kitti_odometry"
