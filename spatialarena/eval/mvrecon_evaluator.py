# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Evaluate canonical multi-view reconstruction artifacts.

MVRecon emits reconstruction-only metrics: accuracy, completion, overall,
precision, recall, and fscore. DTU uses the existing MVSNet-style
ObsMask/Plane evaluator when the required DTU metadata is available.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import sys
import tempfile
from typing import Iterable

import cv2
import numpy as np
# Imported by name: tests patch mvrecon_evaluator.KDTree.
from scipy.spatial import cKDTree as KDTree

from spatialarena import spatialarena
from spatialarena.eval import canonical_dataset
from spatialarena.eval import colmap_view_resolver
from spatialarena.eval import dataset_registry
from spatialarena.eval import depth_prediction_evaluator
from spatialarena.eval import pose_alignment
from spatialarena.eval import prediction_artifacts
from spatialarena.eval import result_layout
# The names below are imported by name because tests and other modules use or
# patch them through this module (e.g. mvrecon_evaluator.load_depth_map).
from spatialarena.eval.colmap_view_resolver import ColmapImageRecord
from spatialarena.eval.colmap_view_resolver import ColmapMappingDiagnostics
from spatialarena.eval.depth_prediction_evaluator import load_depth_map
from spatialarena.eval.pose_alignment import icp_refine_sim3
from spatialarena.eval.prediction_artifacts import find_depth_files
from spatialarena.geometry import read_write_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DA3_MVRECON_DATASETS = frozenset({"7scenes", "eth3d", "hiroom", "scannetpp"})
# The fusion inputs' GT world-to-camera poses per fused view;
# fuse_prediction_in_gt_frame takes them out of the metadata (which is written
# to the evaluation JSON) for the viewer's cameras.
VIEWER_GT_W2C_KEY = "_viewer_gt_w2c"
# The fused views' prediction depth files, in fusion order (the viewer reads
# their confidence maps).
VIEWER_DEPTH_PATHS_KEY = "_viewer_depth_paths"
# Per-view depths or images: one stacked array, or a list when the views' sizes
# differ (see _stack_views).
ViewArrays = np.ndarray | list[np.ndarray]


@dataclasses.dataclass(frozen=True, kw_only=True)
class MVReconProtocol:
    """Fusion and scoring settings of a dataset's MVRecon protocol.

    Attributes:
        voxel_length: TSDF voxel size in metres.
        sdf_trunc: TSDF truncation distance in metres.
        max_depth: Depth beyond which predicted depth is not fused.
        sampling_number: Points sampled from the fused (or GT) mesh.
        eval_threshold: Precision/recall distance threshold in metres.
        down_sample: Voxel size of the downsampling before scoring.
        icp_refinement: ICP-refine the fused cloud against the GT.
    """
    voxel_length: float = 4.0 / 512.0
    sdf_trunc: float = 0.04
    max_depth: float = 1_000_000.0
    sampling_number: int = 1_000_000
    eval_threshold: float = 0.05
    down_sample: float | None = 4.0 / 512.0
    icp_refinement: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True)
class MVReconEvaluationConfig:
    """Settings of an MVRecon evaluation run.

    Attributes:
        reconstruction_protocol: "unposed" (aligned predicted poses) or
            "posed" (GT poses).
        spatialarena_backend: "tsdf" or "direct_backprojection" (SpatialArena
            only).
        compute_geometry_metrics: Fuse and score the geometry.
        metric_workers: KD-tree query workers.
        metric_chunk_points: Points per KD-tree query batch.
        artifact_root: Where to write the fused clouds, if given.
    """
    reconstruction_protocol: str = "unposed"
    spatialarena_backend: str = "tsdf"
    compute_geometry_metrics: bool = True
    metric_workers: int = 1
    metric_chunk_points: int = 262144
    artifact_root: pathlib.Path | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ColmapView:
    """One image of a COLMAP model, keyed like the depth files.

    Attributes:
        name: COLMAP image name.
        key: View key (`depth_key` of the name).
        intrinsic: 3x3 pinhole intrinsic matrix.
        w2c: 4x4 world-to-camera matrix.
    """
    name: str
    key: str
    intrinsic: np.ndarray
    w2c: np.ndarray

    @property
    def c2w(self) -> np.ndarray:
        """The 4x4 camera-to-world matrix."""
        return np.linalg.inv(self.w2c)

    @property
    def center(self) -> np.ndarray:
        """The camera center in world coordinates."""
        return self.c2w[:3, 3]


@dataclasses.dataclass(frozen=True)
class MVReconView:
    """One RGB/depth/camera unit in the evaluator's authoritative order.

    Attributes:
        image_path: The RGB image.
        gt_record: The view's GT COLMAP record.
        pred_record: The view's predicted COLMAP record.
        pred_depth_path: The predicted depth file.
        gt_depth_path: The GT depth file, if any.
        mask_path: The mask file, if any.
        image_match_kind: How the GT record was matched to the image.
    """

    image_path: pathlib.Path
    gt_record: ColmapImageRecord
    pred_record: ColmapImageRecord | None
    pred_depth_path: pathlib.Path
    gt_depth_path: pathlib.Path | None
    mask_path: pathlib.Path | None
    image_match_kind: str


@dataclasses.dataclass(frozen=True)
class MVReconViewManifest:
    """The views of a scene, with the mapping diagnostics.

    Attributes:
        views: The views in authoritative order.
        gt_mapping: GT COLMAP-to-image mapping diagnostics.
        pred_mapping: Predicted COLMAP-to-image mapping diagnostics.
        missing_prediction_depths: GT views without a predicted depth.
        missing_prediction_poses: GT views without a predicted pose.
    """
    views: tuple[MVReconView, ...]
    gt_mapping: ColmapMappingDiagnostics
    pred_mapping: ColmapMappingDiagnostics
    missing_prediction_depths: tuple[str, ...]
    missing_prediction_poses: tuple[str, ...]

    def to_metadata(self) -> dict:
        """Returns the manifest summary for the fusion metadata."""
        return {
            "num_views": len(self.views),
            "gt_mapping": self.gt_mapping.to_dict(),
            "prediction_mapping": self.pred_mapping.to_dict(),
            "missing_prediction_depths": list(self.missing_prediction_depths),
            "missing_prediction_poses": list(self.missing_prediction_poses),
        }


# SpatialArena MVRecon fusion depth on the indoor (0.05 m) grid; outdoor scenes
# (0.25 m) get 5x, i.e. 100 m. At most 0.2% of the GT depth of
# NationalScotlandMuseum7 and YorminsterTheChapteraofRoof lies beyond 20 m.
SPATIALARENA_INDOOR_FUSION_DEPTH = 20.0
# StPancrasStatue1 is scored at the indoor 0.05 m threshold but its geometry
# reaches 100 m (GT depth p99.9 100.8 m, 24% beyond 20 m): it is fused up to
# 100 m on a 2 cm grid (about 16x less volume memory than 7.8 mm), so no real
# geometry is cut while precision/recall keep the strict 0.05 m threshold and
# 7.8 mm downsampling.
SPATIALARENA_FUSION_OVERRIDES: dict[str, dict[str, float]] = {
    "StPancrasStatue1": {
        "voxel_length": 0.02,
        "max_depth": 100.0
    },
}


def protocol_for_dataset(dataset: str,
                         scene_metadata: dict | None = None) -> MVReconProtocol:
    """Returns DA3-BENCH-compatible reconstruction thresholds for a dataset.

    SpatialArena sets the distance threshold per scene (`eval_threshold` in
    its metadata.json entry: 0.05 m indoors, 0.25 m outdoors); the fusion
    voxel, truncation, downsampling and the depth predicted views are fused up
    to all scale with it as DA3-BENCH scales them for ETH3D: 7.8 mm voxels and
    20 m indoors, 39 mm and 100 m (the depth tracks' range) outdoors. Fusing
    to 100 m on the indoor grid needed over 330 GB per scene. Scenes whose
    geometry does not fit that rule are overridden in
    `SPATIALARENA_FUSION_OVERRIDES`.

    Args:
        dataset: Dataset name.
        scene_metadata: The scene's metadata.json entry (SpatialArena).

    Returns:
        The protocol.
    """
    name = dataset.lower()
    if name == spatialarena.SPATIALARENA:
        scene_metadata = scene_metadata or {}
        scale = float(scene_metadata.get("eval_threshold", 0.05)) / 0.05
        voxel_length = 4.0 / 512.0 * scale
        max_depth = SPATIALARENA_INDOOR_FUSION_DEPTH * scale
        override = SPATIALARENA_FUSION_OVERRIDES.get(
            str(scene_metadata.get("scene_name", "")))
        if override is not None:
            voxel_length = override["voxel_length"]
            max_depth = override["max_depth"]
        return MVReconProtocol(
            voxel_length=voxel_length,
            # 0.04 m on the 4/512 m grid: the same five voxels everywhere.
            sdf_trunc=voxel_length * 5.12,
            max_depth=max_depth,
            eval_threshold=0.05 * scale,
            down_sample=4.0 / 512.0 * scale,
            icp_refinement=False,
        )
    if name == "eth3d" or name == "tanks_and_temples":
        scale = 5.0
        return MVReconProtocol(
            voxel_length=4.0 / 512.0 * scale,
            sdf_trunc=0.04 * scale,
            max_depth=100_000.0,
            eval_threshold=0.05 * scale,
            down_sample=4.0 / 512.0 * scale,
            icp_refinement=name == "tanks_and_temples",
        )
    if name == "scannetpp":
        return MVReconProtocol(
            voxel_length=0.02,
            sdf_trunc=0.15,
            max_depth=5.0,
            eval_threshold=0.05,
            down_sample=0.02,
            icp_refinement=False,
        )
    if name in {"7scenes", "hiroom"}:
        return MVReconProtocol(icp_refinement=False)
    return MVReconProtocol()


def depth_key(path: pathlib.Path | str) -> str:
    """Returns the view key of a depth, image or COLMAP name.

    The key is the lower-case stem without depth/color affixes, e.g.
    "depth_0001.depth" -> "0001".
    """
    stem = pathlib.Path(path).stem.lower()
    for suffix in (".depth.proj", ".depth", ".color"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    for prefix in ("depth_map_", "depth_", "disp_", "frame_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    if stem.endswith("_depth"):
        stem = stem[:-len("_depth")]
    return stem


def camera_to_matrix(camera) -> np.ndarray:
    """Returns the 3x3 pinhole intrinsic matrix of a COLMAP camera.

    Raises:
        ValueError: On an unsupported camera model.
    """
    params = np.asarray(camera.params, dtype=np.float64)
    model = str(camera.model).upper()
    if model == "PINHOLE" or model.startswith(
            "OPENCV") or model == "FULL_OPENCV":
        fx, fy, cx, cy = params[:4]
    elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "FOV"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model for MVRecon: {camera.model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float64)


def colmap_image_to_w2c(image) -> np.ndarray:
    """Returns the 4x4 world-to-camera matrix of a COLMAP image."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = image.qvec2rotmat()
    w2c[:3, 3] = image.tvec
    return w2c


def load_colmap_views(
        sparse_path: pathlib.Path | None) -> dict[str, ColmapView]:
    """Reads a COLMAP model into views by key (empty if it is unreadable)."""
    if sparse_path is None:
        return {}
    resolved = result_layout.resolve_sparse_model_path_from_any(sparse_path)
    if resolved is None:
        return {}
    model = read_write_model.read_model(str(resolved))
    if model is None:
        return {}
    cameras, images, _ = model
    views: dict[str, ColmapView] = {}
    for image in images.values():
        view = ColmapView(
            name=image.name,
            key=depth_key(image.name),
            intrinsic=camera_to_matrix(cameras[image.camera_id]),
            w2c=colmap_image_to_w2c(image),
        )
        views[view.key] = view
    return views


def find_image_files(
    scene_info: canonical_dataset.BenchmarkSceneInfo
) -> dict[str, pathlib.Path]:
    """Returns the scene's image files by view key (first file per key)."""
    if scene_info.image_path is None:
        return {}
    files = {}
    for path in sorted(scene_info.image_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.setdefault(depth_key(path), path)
    return files


def find_depth_like_files(
        directory: pathlib.Path | None) -> dict[str, pathlib.Path]:
    """Returns the depth-like files of a folder by view key."""
    if directory is None:
        return {}
    return {
        depth_key(path): path for path in
        depth_prediction_evaluator.list_depth_files_in_directory(directory)
    }


def image_alpha_mask(path: pathlib.Path | None,
                     shape: tuple[int, int]) -> np.ndarray | None:
    """Returns the object mask in an RGBA image's alpha channel, at `shape`.

    The mask is alpha > 127. DTU's materialized images carry the evaluation
    mask as alpha (MVRecon/dtu/README.md), aligned with the cropped,
    undistorted 1554x1162 frames -- unlike its mask/NNN.png files, which are
    the raw 1600x1200 frames.

    Returns:
        The mask, or None without an image or an alpha channel.
    """
    if path is None or not path.is_file():
        return None
    from PIL import Image

    with Image.open(path) as image:
        if image.mode not in ("RGBA", "LA"):
            return None
        alpha = np.asarray(image.getchannel("A"), dtype=np.float32)
    return depth_prediction_evaluator._resize_to_match(alpha, shape,
                                                       "nearest") > 127


def load_image(path: pathlib.Path | None, shape: tuple[int, int]) -> np.ndarray:
    """Loads an RGB image at `shape` (black if there is none)."""
    if path is None or not path.is_file():
        return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if image.shape[:2] == shape:
        return image
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def scale_intrinsics(intrinsic: np.ndarray, source_shape: tuple[int, int],
                     target_shape: tuple[int, int]) -> np.ndarray:
    """Scales an intrinsic matrix from `source_shape` to `target_shape`."""
    scaled = np.asarray(intrinsic, dtype=np.float64).copy()
    source_h, source_w = source_shape
    target_h, target_w = target_shape
    if source_h > 0 and source_w > 0:
        scaled[0, :] *= float(target_w) / float(source_w)
        scaled[1, :] *= float(target_h) / float(source_h)
    return scaled


def apply_sim3_to_c2w(c2w: np.ndarray, scale: float, rotation: np.ndarray,
                      translation: np.ndarray) -> np.ndarray:
    """Applies a Sim(3) to a 4x4 camera-to-world pose."""
    aligned = np.asarray(c2w, dtype=np.float64).copy()
    aligned[:3, :3] = rotation @ c2w[:3, :3]
    aligned[:3, 3] = scale * (rotation @ c2w[:3, 3]) + translation
    return aligned


def _fit_generic_pose_sim3(
    pred_views: dict[str, ColmapView],
    gt_views: dict[str, ColmapView],
    alignment_keys: list[str],
) -> pose_alignment.Sim3Estimate:
    """Fits a pose-aware Sim(3) for the generic (non-DA3) MVRecon datasets.

    These are currently `dtu` and `tanks_and_temples`. Mirrors
    `_align_da3_poses` exactly (same `ransac_irls_align_sim3` call: RANSAC
    over the GTSAM pose-residual fit, then a robust refit on all poses).
    """
    pred_centers = np.stack([pred_views[key].center for key in alignment_keys])
    gt_centers = np.stack([gt_views[key].center for key in alignment_keys])
    pred_rotations = np.stack(
        [pred_views[key].w2c[:3, :3] for key in alignment_keys])
    gt_rotations = np.stack(
        [gt_views[key].w2c[:3, :3] for key in alignment_keys])
    return pose_alignment.ransac_irls_align_sim3(pred_rotations, pred_centers,
                                                 gt_rotations, gt_centers)


def _stack_views(arrays: list[np.ndarray]) -> ViewArrays:
    """Stacks the views when they share a size, else returns the per-view list.

    ETH3D's undistorted DSLR images differ by a few pixels per view (e.g.
    6205x4135 and 6198x4129 in `courtyard`), and TSDF fusion takes each view
    at its own size anyway.
    """
    if len({array.shape for array in arrays}) == 1:
        return np.stack(arrays)
    return arrays


def _generic_view_depth(depth_path: pathlib.Path, *,
                        gt_depth_path: pathlib.Path | None,
                        mask_path: pathlib.Path | None,
                        image_path: pathlib.Path | None, dataset: str,
                        scale: float) -> np.ndarray:
    """Loads a view's scaled predicted depth, zeroed outside the valid pixels.

    Valid pixels have a positive finite prediction and, where given, a
    positive finite GT depth and a nonzero mask; DTU views without a mask
    file use the image's alpha mask.
    """
    resize_to_match = depth_prediction_evaluator._resize_to_match
    depth = load_depth_map(depth_path).astype(np.float32) * float(scale)
    valid = np.isfinite(depth) & (depth > 0)
    if gt_depth_path is not None:
        gt_depth = resize_to_match(load_depth_map(gt_depth_path), depth.shape,
                                   "nearest")
        valid &= np.isfinite(gt_depth) & (gt_depth > 0)
    if mask_path is not None:
        mask = resize_to_match(load_depth_map(mask_path), depth.shape,
                               "nearest")
        valid &= mask > 0
    elif dataset.lower() == "dtu":
        object_mask = image_alpha_mask(image_path, depth.shape)
        if object_mask is not None:
            valid &= object_mask
    return np.where(valid, depth, 0.0).astype(np.float32)


def _prepare_generic_fusion_inputs(
    *,
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    prediction_scene_path: pathlib.Path,
    reconstruction_protocol: str,
) -> tuple[ViewArrays, ViewArrays, np.ndarray, np.ndarray, dict]:
    """Prepares the fusion inputs of the generic (non-DA3) datasets.

    Predicted depths are matched to the GT views by key; the predicted poses
    are aligned to the GT ones by a Sim(3) ("unposed"), or the GT poses are
    used ("posed").

    Returns:
        (depths, images, intrinsics, w2c extrinsics, fusion metadata).

    Raises:
        ValueError: If no views match or the protocol is unknown.
    """
    gt_views = load_colmap_views(scene_info.sparse_gt_path)
    if not gt_views:
        raise ValueError(
            f"Scene has no readable GT sparse model: {scene_info.scene}")

    pred_sparse = result_layout.resolve_sparse_model_path(prediction_scene_path)
    pred_views = load_colmap_views(pred_sparse)
    pred_depths = {
        depth_key(path): path
        for path in find_depth_files(prediction_scene_path)
    }
    image_files = find_image_files(scene_info)
    gt_depths = find_depth_like_files(scene_info.depth_gt_path)
    masks = find_depth_like_files(scene_info.mask_path)
    common_keys = sorted(set(gt_views) & set(pred_depths))
    if not common_keys:
        raise ValueError("No MVRecon depth predictions match GT views in "
                         f"{prediction_scene_path}")

    mode = reconstruction_protocol.lower()
    if mode not in {"unposed", "posed"}:
        raise ValueError("Unknown MVRecon reconstruction protocol: "
                         f"{reconstruction_protocol}")

    scale = 1.0
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    if pred_views:
        alignment_keys = sorted(set(gt_views) & set(pred_views))
        if len(alignment_keys) >= 3:
            pose_estimate = _fit_generic_pose_sim3(pred_views, gt_views,
                                                   alignment_keys)
            scale = pose_estimate.scale
            rotation = pose_estimate.rotation
            translation = pose_estimate.translation

    depths = []
    images = []
    intrinsics = []
    extrinsics = []
    gt_extrinsics = []
    depth_paths = []
    skipped = []

    for key in common_keys:
        gt_view = gt_views[key]
        pred_view = pred_views.get(key)
        depth = _generic_view_depth(pred_depths[key],
                                    gt_depth_path=gt_depths.get(key),
                                    mask_path=masks.get(key),
                                    image_path=image_files.get(key),
                                    dataset=scene_info.dataset,
                                    scale=scale)
        image = load_image(image_files.get(key), depth.shape)

        if mode == "unposed":
            if pred_view is None:
                skipped.append(key)
                continue
            c2w = apply_sim3_to_c2w(pred_view.c2w, scale, rotation, translation)
            intrinsic = pred_view.intrinsic
        else:
            c2w = gt_view.c2w
            intrinsic = gt_view.intrinsic

        intrinsic = scale_intrinsics(intrinsic, image.shape[:2], depth.shape)
        depths.append(depth)
        images.append(image)
        intrinsics.append(intrinsic)
        extrinsics.append(np.linalg.inv(c2w))
        gt_extrinsics.append(np.linalg.inv(gt_view.c2w))
        depth_paths.append(pred_depths[key])

    if not depths:
        raise ValueError("No usable MVRecon views after pose/depth matching in "
                         f"{prediction_scene_path}")

    metadata = {
        "num_matched_views": len(depths),
        "num_skipped_views": len(skipped),
        "skipped_view_keys": skipped,
        "pose_alignment": "dagsfm_sim3_ransac_icp",
        "sim3_scale": float(scale),
        "sim3_rotation": rotation.tolist(),
        "sim3_translation": translation.tolist(),
        "used_predicted_sparse": bool(pred_views),
        VIEWER_GT_W2C_KEY: np.stack(gt_extrinsics),
        VIEWER_DEPTH_PATHS_KEY: depth_paths,
    }
    return (
        _stack_views(depths),
        _stack_views(images),
        np.stack(intrinsics),
        np.stack(extrinsics),
        metadata,
    )


def _index_depth_files(
        directory: pathlib.Path | None) -> dict[str, pathlib.Path]:
    """Indexes a folder's depth-like files by view key."""
    if directory is None:
        return {}
    return _index_depth_paths(
        depth_prediction_evaluator.list_depth_files_in_directory(directory))


def _index_depth_paths(paths: list[pathlib.Path]) -> dict[str, pathlib.Path]:
    """Indexes depth-like files by view key, refusing two files for one view.

    Raises:
        ValueError: If two files share a view key.
    """
    indexed: dict[str, pathlib.Path] = {}
    duplicates: dict[str, list[pathlib.Path]] = {}
    for path in paths:
        key = depth_key(path)
        if key in indexed:
            duplicates.setdefault(key, [indexed[key]]).append(path)
        else:
            indexed[key] = path
    if duplicates:
        preview = ", ".join(f"{key}: {[str(path) for path in paths[:3]]}"
                            for key, paths in list(duplicates.items())[:3])
        raise ValueError(
            f"Ambiguous depth files for MVRecon view keys: {preview}")
    return indexed


def _eth3d_mask_root(
        scene_info: canonical_dataset.BenchmarkSceneInfo
) -> pathlib.Path | None:
    """Returns ETH3D's per-image mask folder, else the scene's mask path."""
    candidate = scene_info.root_path / "masks_for_images" / "dslr_images"
    if candidate.is_dir():
        return candidate
    return scene_info.mask_path


def _build_view_manifest(
    *,
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    prediction_scene_path: pathlib.Path,
) -> MVReconViewManifest:
    """Matches the GT views to the real images and the predicted depth/poses.

    Raises:
        ValueError: If the scene or prediction lacks a model or images, or no
            view matches.
    """
    if scene_info.sparse_gt_path is None:
        raise ValueError(f"Scene has no GT COLMAP model: {scene_info.scene}")
    if scene_info.image_path is None:
        raise ValueError(
            f"Scene has no RGB image directory: {scene_info.scene}")

    resolve_mappings = colmap_view_resolver.resolve_colmap_image_mappings
    real_images = colmap_view_resolver.list_real_image_paths(
        scene_info.root_path)
    gt_mappings, gt_diagnostics = resolve_mappings(
        scene_info.root_path,
        real_images,
        colmap_path=scene_info.sparse_gt_path,
        strict=True,
    )
    pred_sparse = result_layout.resolve_sparse_model_path(prediction_scene_path)
    if pred_sparse is None:
        raise ValueError(
            f"Prediction has no readable sparse model: {prediction_scene_path}")
    pred_mappings, pred_diagnostics = resolve_mappings(
        scene_info.root_path,
        real_images,
        colmap_path=pred_sparse,
        strict=True,
    )
    pred_by_image = {
        mapping.image_path.resolve(): mapping for mapping in pred_mappings
    }
    # Only the depth directories: confidences/<stem>.npz shares each depth
    # map's name.
    pred_depths = _index_depth_paths(find_depth_files(prediction_scene_path))
    gt_depths = _index_depth_files(scene_info.depth_gt_path)
    if scene_info.dataset.lower() == "eth3d":
        mask_root = _eth3d_mask_root(scene_info)
    else:
        mask_root = scene_info.mask_path
    masks = _index_depth_files(mask_root)

    views: list[MVReconView] = []
    missing_depths: list[str] = []
    missing_poses: list[str] = []
    for mapping in gt_mappings:
        image_path = mapping.image_path
        key_candidates = (depth_key(mapping.record.name), depth_key(image_path))
        pred_depth_path = next(
            (pred_depths[key] for key in key_candidates if key in pred_depths),
            None,
        )
        pred_mapping = pred_by_image.get(image_path.resolve())
        if pred_depth_path is None:
            missing_depths.append(mapping.record.name)
            continue
        if pred_mapping is None:
            missing_poses.append(mapping.record.name)
            continue
        gt_depth_path = next(
            (gt_depths[key] for key in key_candidates if key in gt_depths),
            None,
        )
        mask_path = next(
            (masks[key] for key in key_candidates if key in masks),
            None,
        )
        views.append(
            MVReconView(
                image_path=image_path,
                gt_record=mapping.record,
                pred_record=pred_mapping.record,
                pred_depth_path=pred_depth_path,
                gt_depth_path=gt_depth_path,
                mask_path=mask_path,
                image_match_kind=mapping.match_kind,
            ))

    if not views:
        raise ValueError(
            f"No prediction depth/pose records match GT COLMAP views in "
            f"{prediction_scene_path}; missing_depths={len(missing_depths)} "
            f"missing_poses={len(missing_poses)}")
    return MVReconViewManifest(
        views=tuple(views),
        gt_mapping=gt_diagnostics,
        pred_mapping=pred_diagnostics,
        missing_prediction_depths=tuple(missing_depths),
        missing_prediction_poses=tuple(missing_poses),
    )


def _apply_similarity_to_c2w(c2w: np.ndarray, scale: float,
                             rotation: np.ndarray,
                             translation: np.ndarray) -> np.ndarray:
    """Applies a similarity transform to a 4x4 camera-to-world pose."""
    aligned = np.asarray(c2w, dtype=np.float64).copy()
    aligned[:3, :3] = rotation @ aligned[:3, :3]
    aligned[:3, 3] = scale * (rotation @ aligned[:3, 3]) + translation
    return aligned


def _align_da3_poses(
    gt_w2c: np.ndarray,
    pred_w2c: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Aligns predicted to GT camera poses with a pose-aware Sim(3).

    Uses `spatialarena/eval/pose_alignment.py`'s
    `ransac_irls_align_sim3`, the repo-wide Sim(3) pose alignment: LMedS
    RANSAC over the GTSAM pose-residual fit (camera rotations and centers,
    so it is not degenerate on near-collinear trajectories and rejects
    outlier poses), then a Cauchy-weighted refit on all poses that makes the
    result independent of the RANSAC seed. The simulation study behind the
    choice is in docs/benchmark/slam_protocol.md ("Choice of estimator").

    Args:
        gt_w2c: [N, 4, 4] GT world-to-camera poses.
        pred_w2c: [N, 4, 4] predicted world-to-camera poses.
        random_state: RANSAC seed.

    Returns:
        (aligned predicted w2c as float32, scale, rotation, translation).

    Raises:
        ValueError: If the pose arrays are empty or differ in length.
    """

    gt_w2c = np.asarray(gt_w2c, dtype=np.float64)
    pred_w2c = np.asarray(pred_w2c, dtype=np.float64)
    gt_c2w = np.linalg.inv(gt_w2c)
    pred_c2w = np.linalg.inv(pred_w2c)
    gt_centers = gt_c2w[:, :3, 3]
    pred_centers = pred_c2w[:, :3, 3]
    if len(gt_centers) != len(pred_centers) or len(gt_centers) == 0:
        raise ValueError(
            "GT and predicted pose arrays must have matching non-empty lengths")

    estimate = pose_alignment.ransac_irls_align_sim3(
        pred_w2c[:, :3, :3],
        pred_centers,
        gt_w2c[:, :3, :3],
        gt_centers,
        random_state=random_state,
    )

    aligned_w2c = []
    for pose in pred_c2w:
        aligned_c2w = _apply_similarity_to_c2w(pose, estimate.scale,
                                               estimate.rotation,
                                               estimate.translation)
        aligned_w2c.append(np.linalg.inv(aligned_c2w))
    return (
        np.stack(aligned_w2c).astype(np.float32),
        float(estimate.scale),
        estimate.rotation,
        estimate.translation,
    )


def _read_rgb(path: pathlib.Path) -> np.ndarray:
    """Reads an image as uint8 RGB."""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _valid_depth_mask(
    dataset: str,
    *,
    gt_depth_path: pathlib.Path | None,
    mask_path: pathlib.Path | None,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    """Returns the DA3 datasets' valid-GT mask at `target_shape`.

    Returns:
        The mask, or None where the dataset (or a missing GT depth) gives none.
    """
    name = dataset.lower()
    if name == "7scenes":
        if gt_depth_path is None:
            return None
        gt_depth = load_depth_map(gt_depth_path)
        gt_depth = np.asarray(gt_depth)
        valid = gt_depth != 65535
        valid &= gt_depth > 0
    elif name == "hiroom":
        if gt_depth_path is None:
            return None
        gt_depth = np.asarray(load_depth_map(gt_depth_path))
        valid = gt_depth > 0
        if mask_path is not None:
            aliasing = np.asarray(load_depth_map(mask_path)) > 0
            valid &= ~aliasing
    elif name == "eth3d":
        if gt_depth_path is not None:
            gt_depth = np.asarray(load_depth_map(gt_depth_path))
        else:
            gt_depth = np.ones(target_shape, dtype=np.float32)
        valid = np.isfinite(gt_depth) & (gt_depth != 0)
        if mask_path is not None:
            invalid = np.asarray(load_depth_map(mask_path)) == 1
            valid &= ~invalid
    elif name == "scannetpp":
        if gt_depth_path is None:
            return None
        gt_depth = np.asarray(load_depth_map(gt_depth_path))
        gt_depth = cv2.resize(gt_depth, (1024, 768),
                              interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(gt_depth) & (gt_depth > 0)
    else:
        return None
    if valid.shape != target_shape:
        valid = cv2.resize(
            valid.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return valid


def _mask_prediction_depth(depth: np.ndarray,
                           valid_gt: np.ndarray | None) -> np.ndarray:
    """Zeroes the non-finite, non-positive or GT-invalid prediction pixels."""
    depth = np.asarray(depth, dtype=np.float32).copy()
    invalid = ~np.isfinite(depth)
    if valid_gt is None:
        depth[invalid | (depth <= 0)] = 0.0
    else:
        depth *= (valid_gt & ~invalid).astype(np.float32)
    return depth


def _prepare_da3_fusion_inputs(
    *,
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    prediction_scene_path: pathlib.Path,
    reconstruction_protocol: str,
) -> tuple[ViewArrays, ViewArrays, np.ndarray, np.ndarray, dict]:
    """Prepares the fusion inputs of the DA3 datasets as DA3-BENCH does.

    Returns:
        (depths, images, intrinsics, w2c extrinsics, fusion metadata).
    """
    manifest = _build_view_manifest(
        scene_info=scene_info,
        prediction_scene_path=prediction_scene_path,
    )
    dataset = scene_info.dataset.lower()
    pred_w2c = np.stack([view.pred_record.w2c for view in manifest.views])
    gt_w2c = np.stack([view.gt_record.w2c for view in manifest.views])
    aligned_pred_w2c, scale, _rotation, _translation = _align_da3_poses(
        gt_w2c, pred_w2c)

    depths: list[np.ndarray] = []
    images: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    for index, view in enumerate(manifest.views):
        pred_depth = np.asarray(load_depth_map(view.pred_depth_path),
                                dtype=np.float32)
        pred_intrinsic = view.pred_record.intrinsic.astype(np.float64)
        gt_intrinsic = view.gt_record.intrinsic.astype(np.float64)
        if dataset == "7scenes":
            gt_intrinsic = np.array(
                [[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        image = _read_rgb(view.image_path)
        model_shape = pred_depth.shape

        if dataset == "scannetpp":
            camera_width = view.gt_record.width
            camera_height = view.gt_record.height
            raw_intrinsic = view.gt_record.intrinsic.astype(np.float64).copy()
            raw_intrinsic[:2, 2] -= 0.5
            dist = np.zeros(5, dtype=np.float32)
            if view.gt_record.model.upper() == "OPENCV":
                dist[:4] = view.gt_record.params[4:8]
            undistorted_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
                raw_intrinsic,
                dist,
                (camera_width, camera_height),
                1,
                (camera_width, camera_height),
            )
            image = cv2.undistort(
                image,
                raw_intrinsic,
                dist,
                newCameraMatrix=undistorted_intrinsic,
            )
            x, y, width, height = roi
            image = image[y:y + height, x:x + width]
            image = cv2.resize(image, (1024, 768), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(pred_depth, (camera_width, camera_height),
                               interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (1024, 768),
                               interpolation=cv2.INTER_LINEAR).astype(
                                   np.float32)
            valid = _valid_depth_mask(
                dataset,
                gt_depth_path=view.gt_depth_path,
                mask_path=None,
                target_shape=depth.shape,
            )
            depth = _mask_prediction_depth(depth, valid) * float(scale)
            if reconstruction_protocol.lower() == "unposed":
                intrinsic = scale_intrinsics(pred_intrinsic, model_shape,
                                             (768, 1024))
                extrinsic = aligned_pred_w2c[index]
            else:
                intrinsic = undistorted_intrinsic.astype(np.float64)
                intrinsic[0, 2] += 0.5
                intrinsic[0, :] *= 1024.0 / camera_width
                intrinsic[1, :] *= 768.0 / camera_height
                extrinsic = gt_w2c[index]
        else:
            target_shape = image.shape[:2]
            depth = cv2.resize(
                pred_depth,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
            valid = _valid_depth_mask(
                dataset,
                gt_depth_path=view.gt_depth_path,
                mask_path=view.mask_path,
                target_shape=depth.shape,
            )
            depth = _mask_prediction_depth(depth, valid) * float(scale)
            if reconstruction_protocol.lower() == "unposed":
                intrinsic = scale_intrinsics(pred_intrinsic, model_shape,
                                             depth.shape)
                extrinsic = aligned_pred_w2c[index]
            else:
                intrinsic = gt_intrinsic
                extrinsic = gt_w2c[index]

        if image.shape[:2] != depth.shape:
            image = cv2.resize(image, (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_AREA)
        depths.append(depth)
        images.append(image)
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)

    metadata = manifest.to_metadata()
    depth_paths = [view.pred_depth_path for view in manifest.views]
    metadata.update({
        "dataset_preprocessing": "depth_anything_3",
        "pose_alignment": "dagsfm_sim3_ransac_icp",
        "pose_alignment_random_state": 42,
        "sim3_scale": float(scale),
        "reconstruction_protocol": reconstruction_protocol,
        VIEWER_GT_W2C_KEY: np.asarray(gt_w2c, dtype=np.float64),
        VIEWER_DEPTH_PATHS_KEY: depth_paths,
    })
    return (
        _stack_views(depths),
        _stack_views(images),
        np.stack(intrinsics),
        np.stack(extrinsics),
        metadata,
    )


def prepare_fusion_inputs(
    *,
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    prediction_scene_path: pathlib.Path,
    reconstruction_protocol: str,
) -> tuple[ViewArrays, ViewArrays, np.ndarray, np.ndarray, dict]:
    """Prepares the fusion inputs with the dataset's declared protocol.

    Returns:
        (depths, images, intrinsics, w2c extrinsics, fusion metadata).
    """

    if scene_info.dataset.lower() in DA3_MVRECON_DATASETS:
        return _prepare_da3_fusion_inputs(
            scene_info=scene_info,
            prediction_scene_path=prediction_scene_path,
            reconstruction_protocol=reconstruction_protocol,
        )
    return _prepare_generic_fusion_inputs(
        scene_info=scene_info,
        prediction_scene_path=prediction_scene_path,
        reconstruction_protocol=reconstruction_protocol,
    )


def require_open3d():
    """Imports Open3D.

    Raises:
        RuntimeError: If Open3D cannot be imported.
    """
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError("Open3D is required for MVRecon TSDF fusion and "
                           "point-cloud evaluation") from exc
    return o3d


def _memory_trace(stage: str, **details):
    """Prints opt-in memory diagnostics (FFB_SCORE_MEMORY_TRACE=1).

    They distinguish the input stacks, TSDF and metric memory peaks.
    """
    import os
    if os.environ.get("FFB_SCORE_MEMORY_TRACE") != "1":
        return
    import resource
    record = {
        "score_memory_stage": stage,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    record.update(details)
    print(json.dumps(record), flush=True)


def reconstruction_backend(dataset: str, spatialarena_backend: str) -> str:
    """Returns the reconstruction backend; the alternative is SpatialArena's.

    Raises:
        ValueError: On an unknown backend.
    """
    if spatialarena_backend not in {"tsdf", "direct_backprojection"}:
        raise ValueError(
            f"Unknown SpatialArena backend: {spatialarena_backend}")
    if dataset.lower() == spatialarena.SPATIALARENA:
        return spatialarena_backend
    return "tsdf"


def backproject_depths_to_pointcloud(
    depths: ViewArrays,
    images: ViewArrays,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    protocol: MVReconProtocol,
    *,
    chunk_pixels: int = 262144,
) -> tuple[np.ndarray, np.ndarray]:
    """Backprojects z-depth with w2c poses, keeping the first point per voxel.

    Inputs already carry the evaluator's scale/pose alignment. Pixel chunks
    and incremental voxel deduplication avoid retaining all overlapping raw
    points. Unlike TSDF this neither estimates a surface nor samples a mesh.

    Args:
        depths: Per-view HxW z-depth.
        images: Per-view HxWx3 RGB.
        intrinsics: [V, 3, 3] intrinsics.
        extrinsics: [V, 4, 4] world-to-camera poses.
        protocol: Sets the maximum depth and the voxel size.
        chunk_pixels: Pixels backprojected at once.

    Returns:
        (points, colors in [0, 1]).

    Raises:
        ValueError: On mismatched inputs, invalid settings or cameras.
    """
    if not (len(depths) == len(images) == len(intrinsics) == len(extrinsics)):
        raise ValueError("Depth, RGB and camera counts must match")
    if (chunk_pixels <= 0 or protocol.down_sample is None or
            protocol.down_sample <= 0):
        raise ValueError("Direct backprojection requires positive chunk_pixels "
                         "and down_sample")
    points = np.empty((0, 3), dtype=np.float64)
    colors = np.empty((0, 3), dtype=np.float64)
    for view_index, (depth, image, intrinsic, w2c) in enumerate(
            zip(depths, images, intrinsics, extrinsics)):
        depth = np.asarray(depth)
        image = np.asarray(image)
        if depth.ndim != 2 or image.shape != (*depth.shape, 3):
            raise ValueError("Expected HxW depth and matching HxWx3 RGB")
        inverse_k = np.linalg.inv(np.asarray(intrinsic, dtype=np.float64))
        c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
        if not (np.isfinite(inverse_k).all() and np.isfinite(c2w).all()):
            raise ValueError("Camera matrices must be finite and invertible")
        width = depth.shape[1]
        view_points, view_colors = [], []
        for start in range(0, depth.size, chunk_pixels):
            values = depth.reshape(-1)[start:start + chunk_pixels]
            valid = (np.isfinite(values) & (values > 0) &
                     (values <= protocol.max_depth))
            indices = np.flatnonzero(valid) + start
            if not len(indices):
                continue
            pixels = np.column_stack(
                (indices % width, indices // width, np.ones(len(indices))))
            camera_points = (pixels @ inverse_k.T) * values[valid, None]
            world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
            view_points.append(world_points)
            view_colors.append(
                image.reshape(-1, 3)[indices].astype(np.float64) / 255.0)
        if view_points:
            combined = np.concatenate([points, *view_points])
            combined_colors = np.concatenate([colors, *view_colors])
            bins = np.floor(combined / protocol.down_sample).astype(np.int64)
            _, indices = np.unique(bins, axis=0, return_index=True)
            indices.sort()
            points, colors = combined[indices], combined_colors[indices]
            del combined, combined_colors, bins, view_points, view_colors
        if (view_index + 1) % 10 == 0:
            _memory_trace("backprojection_view",
                          views=view_index + 1,
                          retained_points=len(points))
    return points, colors


def fuse_depths_to_pointcloud(
    depths: ViewArrays,
    images: ViewArrays,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    protocol: MVReconProtocol,
):
    """Fuses depths with Open3D TSDF and samples the mesh.

    Adapted from Depth-Anything-3 bench/utils.py.
    """
    o3d = require_open3d()
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=protocol.voxel_length,
        sdf_trunc=protocol.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    _memory_trace("tsdf_start",
                  views=len(depths),
                  voxel_length=protocol.voxel_length)
    for index, (depth, image, intrinsic, extrinsic) in enumerate(
            zip(depths, images, intrinsics, extrinsics)):
        height, width = depth.shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(image.astype(np.uint8)),
            o3d.geometry.Image(depth.astype(np.float32)),
            depth_trunc=protocol.max_depth,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        volume.integrate(rgbd, pinhole, extrinsic)
        if index % 10 == 0:
            _memory_trace("tsdf_integrate", view=index)
    _memory_trace("mesh_extract_start")
    mesh = volume.extract_triangle_mesh()
    del volume
    _memory_trace("mesh_extract_done",
                  vertices=len(mesh.vertices),
                  triangles=len(mesh.triangles))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return o3d.geometry.PointCloud()
    return mesh.sample_points_uniformly(
        number_of_points=protocol.sampling_number)


def load_point_cloud_array(path: pathlib.Path,
                           sample_points: int) -> np.ndarray:
    """Loads finite [N, 3] points from .npy/.npz or a PLY cloud or mesh.

    A mesh is sampled uniformly with `sample_points` points.

    Raises:
        ValueError: On an unsupported file type.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if "points" in data:
            key = "points"
        else:
            key = data.files[0]
        points = data[key]
    elif suffix == ".ply":
        o3d = require_open3d()
        geometry = o3d.io.read_triangle_mesh(str(path))
        if len(geometry.triangles) > 0:
            pcd = geometry.sample_points_uniformly(
                number_of_points=sample_points)
        else:
            pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported point-cloud/mesh file: {path}")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def voxel_downsample_points(points: np.ndarray,
                            voxel_size: float | None) -> np.ndarray:
    """Keeps the first point (in input order) of each voxel."""
    if voxel_size is None or voxel_size <= 0 or len(points) == 0:
        return points
    bins = np.floor(points / float(voxel_size)).astype(np.int64)
    _, unique_indices = np.unique(bins, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def compute_reconstruction_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    threshold: float,
    down_sample: float | None,
    workers: int = 1,
    chunk_points: int = 262144,
) -> dict[str, float]:
    """Computes the exact nearest-neighbor metrics in bounded query batches.

    Args:
        pred_points: Predicted points.
        gt_points: GT points.
        threshold: Precision/recall distance threshold.
        down_sample: Voxel size of the downsampling, or None.
        workers: KD-tree query workers.
        chunk_points: Points per query batch.

    Returns:
        accuracy, completion, overall, precision, recall and fscore.

    Raises:
        ValueError: If workers or chunk_points is not positive.
    """
    if workers < 1 or chunk_points < 1:
        raise ValueError("workers and chunk_points must be positive")
    pred = voxel_downsample_points(pred_points, down_sample)
    gt = voxel_downsample_points(gt_points, down_sample)
    if len(pred) == 0 or len(gt) == 0:
        return {
            "accuracy": float("inf"),
            "completion": float("inf"),
            "overall": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "fscore": 0.0,
        }
    accuracy, precision = _nearest_neighbor_statistics(pred, gt, threshold,
                                                       workers, chunk_points)
    completion, recall = _nearest_neighbor_statistics(gt, pred, threshold,
                                                      workers, chunk_points)
    if precision + recall <= 0:
        fscore = 0.0
    else:
        fscore = float(2.0 * precision * recall / (precision + recall))
    return {
        "accuracy": accuracy,
        "completion": completion,
        "overall": (accuracy + completion) / 2.0,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def _nearest_neighbor_statistics(
    source: np.ndarray,
    target: np.ndarray,
    threshold: float,
    workers: int,
    chunk_points: int,
) -> tuple[float, float]:
    """Returns the mean k=1 distance and the fraction within `threshold`.

    Only the batch distances and scalar accumulators are retained.
    """
    tree = KDTree(target)
    total_distance, matches = 0.0, 0
    for start in range(0, len(source), chunk_points):
        distances, _ = tree.query(source[start:start + chunk_points],
                                  k=1,
                                  eps=0,
                                  workers=workers)
        total_distance += float(np.sum(distances, dtype=np.float64))
        matches += int(np.count_nonzero(distances < threshold))
        _memory_trace("nearest_neighbor_batch",
                      processed=min(start + chunk_points, len(source)),
                      total=len(source),
                      workers=workers)
    return total_distance / len(source), matches / len(source)


def resolve_gt_geometry_path(
        scene_info: canonical_dataset.BenchmarkSceneInfo
) -> pathlib.Path | None:
    """Returns the scene's GT point cloud or mesh, or None if there is none.

    The metadata's GT paths come first, then the usual file names.
    """
    metadata = scene_info.metadata or {}
    for key in ("gt_point_path", "gt_pcd_path", "gt_mesh_path",
                "dense_gt_path"):
        value = metadata.get(key)
        if value:
            path = pathlib.Path(value)
            if not path.is_absolute():
                path = scene_info.root_path / path
            if path.is_file():
                return path
    for name in (
            "gt_pointcloud.ply",
            "gt_points.ply",
            "gt_mesh.ply",
            "mesh_aligned_0.05.ply",
            "points3D_dense.ply",
            "points.ply",
            f"{scene_info.scene}.ply",
    ):
        path = scene_info.root_path / name
        if path.is_file():
            return path
    return None


@dataclasses.dataclass(frozen=True)
class CropVolume:
    """Tanks and Temples' `<scene>.json` crop (Open3D `SelectionPolygonVolume`).

    A polygon in the plane orthogonal to `axis`, extruded between `axis_min`
    and `axis_max`.

    Attributes:
        axis: The orthogonal axis (0, 1 or 2).
        axis_min: Lower bound along the axis.
        axis_max: Upper bound along the axis.
        polygon: (K, 2), the two in-plane coordinates in axis order.
    """

    axis: int
    axis_min: float
    axis_max: float
    polygon: np.ndarray

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Returns the boolean mask of the (N, 3) points inside the volume."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        along = points[:, self.axis]
        inside = (along >= self.axis_min) & (along <= self.axis_max)
        plane = np.delete(points, self.axis, axis=1)
        x, y = plane[:, 0], plane[:, 1]
        crossings = np.zeros(len(points), dtype=bool)
        for (x0, y0), (x1, y1) in zip(self.polygon,
                                      np.roll(self.polygon, -1, axis=0)):
            # Even-odd ray casting along +x.
            straddles = (y0 > y) != (y1 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            crossings ^= straddles & (x < x_cross)
        return inside & crossings


def read_crop_volume(path: pathlib.Path) -> CropVolume:
    """Reads a Tanks and Temples crop `<scene>.json`."""
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    axis = "XYZ".index(str(payload["orthogonal_axis"]).upper())
    bounding_polygon = np.asarray(payload["bounding_polygon"], dtype=np.float64)
    polygon = np.delete(bounding_polygon, axis, axis=1)
    return CropVolume(axis=axis,
                      axis_min=float(payload["axis_min"]),
                      axis_max=float(payload["axis_max"]),
                      polygon=polygon)


def tanks_and_temples_registration(
    scene_info: canonical_dataset.BenchmarkSceneInfo
) -> tuple[pose_alignment.Sim3Estimate, CropVolume] | None:
    """Returns the official Tanks and Temples registration of a scene.

    `<scene>_trans.txt` maps the COLMAP frame (the GT cameras, so the fused
    prediction) onto the laser scan `<scene>.ply`, and `<scene>.json` crops
    both clouds to the evaluated region.

    Returns:
        (registration, crop volume), or None when either file is missing.
    """
    root = scene_info.root_path
    trans_path = root / f"{scene_info.scene}_trans.txt"
    crop_path = root / f"{scene_info.scene}.json"
    if not (trans_path.is_file() and crop_path.is_file()):
        return None
    matrix = np.loadtxt(trans_path, dtype=np.float64).reshape(4, 4)
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    registration = pose_alignment.Sim3Estimate(scale=scale,
                                               rotation=matrix[:3, :3] / scale,
                                               translation=matrix[:3, 3],
                                               inliers=np.zeros(0, dtype=bool))
    return registration, read_crop_volume(crop_path)


# ICP pairs a predicted point with its nearest GT point only within this many
# evaluation thresholds.
ICP_CORRESPONDENCE_THRESHOLDS = 2.0


def refine_fused_points_in_gt_frame(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_correspondence_distance: float | None = None,
) -> pose_alignment.Sim3Estimate:
    """ICP-refines a fused cloud that is already in the GT frame.

    prepare_fusion_inputs has already applied the camera-pose Sim(3) (depths
    scaled, cameras moved), so ICP starts from the identity. Seeding it with
    that Sim(3) again applied it twice (the 2026-09-17 to 2026-10-15 DTU,
    NRGBD and Tanks-and-Temples MVRecon results): every view ended up rotated
    by the alignment's own angle (50-80 degrees).

    Args:
        pred_points: The fused points, in the GT frame.
        gt_points: The GT points ICP aligns to.
        max_correspondence_distance: ICP correspondence distance bound.

    Returns:
        The Sim(3) that maps the fused points onto the GT points.
    """
    identity = pose_alignment.Sim3Estimate(scale=1.0,
                                           rotation=np.eye(3),
                                           translation=np.zeros(3),
                                           inliers=np.zeros(0, dtype=bool))
    return icp_refine_sim3(
        pred_points,
        gt_points,
        identity,
        max_correspondence_distance=max_correspondence_distance)


def icp_target_points(scene_info: canonical_dataset.BenchmarkSceneInfo,
                      gt_points: np.ndarray,
                      protocol: MVReconProtocol) -> np.ndarray:
    """Returns the GT points ICP aligns to, in the GT (normalized) frame.

    DTU: the official laser scan (Points/stl/stlNNN_total.ply, DTU's
    calibration frame) brought into the normalized frame by the inverse of
    _fit_dtu_raw_calibration_sim3 -- the scene's `points.ply` is only a sparse
    COLMAP cloud (~30k points, with outliers). Other datasets: their
    registered GT geometry (`gt_points`).
    """
    if scene_info.dataset.lower() != "dtu":
        return gt_points
    paths = dtu_eval_paths(scene_info)
    if paths is None:
        return gt_points
    calibration = _fit_dtu_raw_calibration_sim3(
        load_colmap_views(scene_info.sparse_gt_path), scene_info.root_path)
    if calibration is None:
        return gt_points
    scan = load_point_cloud_array(paths[1], protocol.sampling_number)
    # The scan is millions of sub-millimetre points; the fusion voxel is
    # enough for ICP's nearest neighbors.
    normalized_scan = ((scan - calibration.translation)
                       @ calibration.rotation) / calibration.scale
    return voxel_downsample_points(normalized_scan, protocol.down_sample)


def gt_bbox_mask(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    margin: float = 0.1,
) -> np.ndarray:
    """Returns the mask of the prediction points inside the GT's grown AABB.

    The AABB is grown by `margin` (the ScanNet++ DA3 crop).
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.ones(len(pred_points), dtype=bool)
    minimum = np.min(gt_points, axis=0) - float(margin)
    maximum = np.max(gt_points, axis=0) + float(margin)
    return np.all((pred_points >= minimum) & (pred_points <= maximum), axis=1)


def crop_points_to_gt_bbox(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    margin: float = 0.1,
) -> np.ndarray:
    """Applies the ScanNet++ DA3 ground-truth AABB crop to predictions."""
    return pred_points[gt_bbox_mask(pred_points, gt_points, margin=margin)]


def write_point_cloud(path: pathlib.Path,
                      points,
                      colors: np.ndarray | None = None) -> None:
    """Writes points (an Open3D cloud or an [N, 3] array) as PLY.

    Array points get `colors` ([N, 3] in [0, 1]) when given.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d = require_open3d()
    if hasattr(points, "points"):
        pcd = points
    else:
        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
        if colors is not None and len(colors) == len(pcd.points):
            pcd.colors = o3d.utility.Vector3dVector(
                np.asarray(colors, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


@dataclasses.dataclass
class FusedPrediction:
    """A method's fused reconstruction of one scene in the scene's GT frame.

    The GT frame is the scene's materialized one.

    Attributes:
        points: [N, 3], after the ScanNet++ crop and the T&T ICP refinement.
        colors: [N, 3] in [0, 1], the fused image colors.
        gt_points: [M, 3] sampled from the GT geometry.
        fusion_metadata: The fusion metadata of the evaluation JSON.
        cameras: The fused views' cameras in the same frame as `points`:
            {"pred_c2w", "gt_c2w": [V, 4, 4], "intrinsics": [V, 3, 3],
            "sizes": [V, 2] (width, height of the fused depth), "depth_paths":
            the V prediction depth files}. pred_c2w is what was fused: the
            aligned prediction ("unposed") or the GT poses ("posed").
    """

    points: np.ndarray
    colors: np.ndarray
    gt_points: np.ndarray
    fusion_metadata: dict
    cameras: dict | None = None


GT_REACH_PERCENTILE = 99.9


def gt_camera_reach_mask(
        pred_points: np.ndarray,
        gt_points: np.ndarray,
        camera_centers: np.ndarray,
        percentile: float = GT_REACH_PERCENTILE) -> tuple[np.ndarray, float]:
    """Masks the predicted points within the GT cloud's reach from the cameras.

    The reach is the `percentile` of the GT points' distance to their nearest
    camera (the same cameras), so the prediction is scored on the region the
    GT covers, as ScanNet++'s bounding-box crop does for its scans.

    Returns:
        (mask of the predicted points no farther from their nearest camera
        than the reach, the reach).
    """
    if len(pred_points) == 0 or len(gt_points) == 0 or len(camera_centers) == 0:
        return np.ones(len(pred_points), dtype=bool), float("inf")
    tree = KDTree(np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3))
    gt_distance, _ = tree.query(np.asarray(gt_points, dtype=np.float64), k=1)
    reach = float(np.percentile(gt_distance, percentile))
    pred_distance, _ = tree.query(np.asarray(pred_points, dtype=np.float64),
                                  k=1)
    return pred_distance <= reach, reach


def _reconstruct_points(depths: ViewArrays, images: ViewArrays,
                        intrinsics: np.ndarray, extrinsics: np.ndarray,
                        protocol: MVReconProtocol, backend: str,
                        fusion_metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reconstructs the views' points with `backend`.

    Direct backprojection records its settings in `fusion_metadata`.

    Returns:
        (points, colors in [0, 1]).
    """
    if backend == "direct_backprojection":
        pred_points, pred_colors = backproject_depths_to_pointcloud(
            depths, images, intrinsics, extrinsics, protocol)
        fusion_metadata["pre_icp_voxel_size"] = protocol.down_sample
        fusion_metadata["voxel_representative"] = (
            "first_valid_point_in_view_pixel_order")
        fusion_metadata["mesh_sampling"] = False
    else:
        fused_pcd = fuse_depths_to_pointcloud(depths, images, intrinsics,
                                              extrinsics, protocol)
        pred_points = np.asarray(fused_pcd.points)
        pred_colors = np.asarray(getattr(fused_pcd, "colors", []),
                                 dtype=np.float64).reshape(-1, 3)
    return pred_points, pred_colors


def _kept_fraction(keep: np.ndarray) -> float:
    """Returns the kept fraction of a crop mask (0 for an empty mask)."""
    if len(keep):
        return float(np.mean(keep))
    return 0.0


def fuse_prediction_in_gt_frame(
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    prediction_scene_path: pathlib.Path,
    reconstruction_protocol: str,
    protocol: MVReconProtocol,
    *,
    fused_cloud: tuple[np.ndarray, np.ndarray | None] | None = None,
    spatialarena_backend: str = "tsdf",
) -> FusedPrediction:
    """Reconstructs a method's depths and brings the cloud into the GT frame.

    The depths are fused by TSDF, or on SpatialArena optionally by direct
    backprojection (`spatialarena_backend`), and the cloud is brought into the
    GT frame as the evaluator scores it. Shared by `evaluate_mvrecon_scene`
    and the dashboard's viewer assets. DTU's evaluation additionally moves the
    points into DTU's raw calibration frame afterwards
    (`_fit_dtu_raw_calibration_sim3`); that is not done here.

    Args:
        scene_info: The benchmark scene.
        prediction_scene_path: The method's output folder of the scene.
        reconstruction_protocol: "unposed" or "posed".
        protocol: The dataset's fusion and scoring settings.
        fused_cloud: (points, colors) of an evaluation's saved
            `fused_points.ply` -- already fused, in the GT frame, cropped and
            ICP-refined -- used instead of re-fusing (the dashboard's imported
            viewer clouds: Open3D 0.20's ScalableTSDFVolume returns an empty
            volume on this host). The cameras still get the T&T registration,
            but not the ICP refinement, which needs the unrefined cloud.
        spatialarena_backend: "tsdf" or "direct_backprojection".

    Returns:
        The fused prediction.

    Raises:
        ValueError: If the scene has no GT geometry.
    """
    _memory_trace("prepare_inputs_start", scene=scene_info.scene)
    (depths, images, intrinsics, extrinsics,
     fusion_metadata) = prepare_fusion_inputs(
         scene_info=scene_info,
         prediction_scene_path=prediction_scene_path,
         reconstruction_protocol=reconstruction_protocol,
     )
    _memory_trace("prepare_inputs_done", scene=scene_info.scene)
    gt_w2c = fusion_metadata.pop(VIEWER_GT_W2C_KEY, None)
    depth_paths = fusion_metadata.pop(VIEWER_DEPTH_PATHS_KEY, None)
    pred_c2w = np.linalg.inv(np.asarray(extrinsics, dtype=np.float64))
    sizes = np.array([(depth.shape[1], depth.shape[0]) for depth in depths])
    backend = reconstruction_backend(scene_info.dataset, spatialarena_backend)
    fusion_metadata["reconstruction_backend"] = backend
    if fused_cloud is not None:
        pred_points = np.asarray(fused_cloud[0],
                                 dtype=np.float64).reshape(-1, 3)
        if fused_cloud[1] is not None:
            saved_colors = fused_cloud[1]
        else:
            saved_colors = []
        pred_colors = np.asarray(saved_colors, dtype=np.float64).reshape(-1, 3)
    else:
        pred_points, pred_colors = _reconstruct_points(depths, images,
                                                       intrinsics, extrinsics,
                                                       protocol, backend,
                                                       fusion_metadata)
    del depths, images
    if len(pred_colors) != len(pred_points):
        pred_colors = np.full((len(pred_points), 3), 0.7)
    gt_geometry = resolve_gt_geometry_path(scene_info)
    if gt_geometry is None:
        raise ValueError("No GT point cloud or mesh registered for MVRecon "
                         f"scene {scene_info.scene}")
    _memory_trace("gt_geometry_start", path=str(gt_geometry))
    gt_points = load_point_cloud_array(gt_geometry, protocol.sampling_number)
    _memory_trace("gt_geometry_done", points=len(gt_points))
    if gt_w2c is not None:
        gt_c2w = np.linalg.inv(gt_w2c)
    else:
        gt_c2w = None
    if scene_info.dataset.lower() == "tanks_and_temples":
        registration = tanks_and_temples_registration(scene_info)
    else:
        registration = None
    if registration is not None:
        # The GT cameras (and so the fused cloud) are in the COLMAP frame, the
        # laser scan is not: without this the prediction lay tens of metres
        # from the scan, and the 2026-10-16 bounded ICP could not bridge that.
        transform, crop = registration
        if fused_cloud is None:
            pred_points = transform.transform_points(pred_points)
        for c2w in (pred_c2w, gt_c2w):
            if c2w is not None:
                c2w[:, :3, :3] = transform.rotation @ c2w[:, :3, :3]
                c2w[:, :3, 3] = transform.transform_points(c2w[:, :3, 3])
        if fused_cloud is None:
            keep = crop.contains(pred_points)
        else:
            keep = np.ones(len(pred_points), dtype=bool)
        pred_points, pred_colors = pred_points[keep], pred_colors[keep]
        gt_points = gt_points[crop.contains(gt_points)]
        fusion_metadata["gt_registration"] = "tanks_and_temples_trans_crop"
        fusion_metadata["crop_kept_fraction"] = _kept_fraction(keep)
    if scene_info.dataset.lower() == "scannetpp" and fused_cloud is None:
        keep = gt_bbox_mask(pred_points, gt_points)
        pred_points, pred_colors = pred_points[keep], pred_colors[keep]
    if (scene_info.dataset.lower() == spatialarena.SPATIALARENA and
            fused_cloud is None and gt_c2w is not None and len(gt_points)):
        # The GT cloud is the LiDAR map cropped to what the views see, a few
        # metres around the cameras (ChrispAlley2: 90% of its points within
        # 3.8 m), while predicted depth is fused out to 100 m: correct far
        # geometry was scored as error (masked-depth precision 1%, accuracy
        # 40 m) and depth too short scored best. Keep only the predicted
        # points the GT cloud's own reach from the cameras covers.
        keep, reach = gt_camera_reach_mask(pred_points, gt_points, gt_c2w[:, :3,
                                                                          3])
        pred_points, pred_colors = pred_points[keep], pred_colors[keep]
        fusion_metadata["gt_reach_crop_m"] = reach
        fusion_metadata["crop_kept_fraction"] = _kept_fraction(keep)
    if protocol.icp_refinement and fused_cloud is None:
        # icp_refine_sim3's scaled ICP (Open3D's rigid ICP has no scale DOF
        # for the pose fit's residual scale error), against the dense fused
        # points, which are already in the GT frame.
        refined_estimate = refine_fused_points_in_gt_frame(
            pred_points,
            icp_target_points(scene_info, gt_points, protocol),
            max_correspondence_distance=ICP_CORRESPONDENCE_THRESHOLDS *
            protocol.eval_threshold,
        )
        pred_points = refined_estimate.transform_points(pred_points)
        # The cameras move with the points: rotate the orientations, scale and
        # move the centers.
        pred_c2w[:, :3, :3] = refined_estimate.rotation @ pred_c2w[:, :3, :3]
        pred_c2w[:, :3, 3] = refined_estimate.transform_points(pred_c2w[:, :3,
                                                                        3])
    cameras = {
        "pred_c2w": pred_c2w,
        "gt_c2w": gt_c2w,
        "intrinsics": np.asarray(intrinsics, dtype=np.float64),
        "sizes": sizes,
        "depth_paths": depth_paths,
    }
    return FusedPrediction(points=pred_points,
                           colors=pred_colors,
                           gt_points=gt_points,
                           fusion_metadata=fusion_metadata,
                           cameras=cameras)


def _fit_dtu_raw_calibration_sim3(
        gt_views: dict[str, ColmapView],
        scene_root: pathlib.Path) -> pose_alignment.Sim3Estimate | None:
    """Fits the Sim(3) from the normalized GT frame to DTU's calibration frame.

    The source is this scene's materialized/normalized GT frame (`gt_views`,
    the COLMAP model under `sparse_gt_path` -- unit-ish scale, the frame
    everything above this point in the pipeline operates in), the target
    DTU's original per-rig calibration frame
    (`Calibration/cal18/pos_NNN.txt`, hundreds-of-mm scale; also what
    ObsMask/Plane/STL in `evaluate_dtu_mvsnet` are defined in).

    These are two unrelated coordinate systems that nothing previously
    reconciled: pose-based Sim(3) alignment and (for DTU, the default
    `icp_refinement=True`) mesh-ICP refinement both only ever target the
    normalized frame, so `pred_points` reaching `evaluate_dtu_mvsnet`
    was never in DTU's calibration frame at all -- it landed nowhere near
    the ObsMask bounding box, and the masking step filtered out every
    point (job 132501, 2026-09-16: 13/15 dtu scenes failed with sklearn's
    "Found array with 0 sample(s)").

    DTU's 64-position rig is shared by every scan and each materialized
    view is named by its 0-based rig index (`"0035"` -> `pos_036.txt`),
    so this fit uses every matched view, not a RANSAC subset -- empirically
    a near-exact match (<1e-3 relative residual against a ~1000mm scene),
    confirming it's a real fixed transform rather than an approximation.

    Reuses `best_fit_transform` -- this file's own scale-then-rigid
    Procrustes fit, already used by `evaluate_mesh`'s
    `align_coordinate_frame` for the same predicted-frame-to-calibration-
    frame problem (just never wired into this module's pipeline) -- rather
    than a second, parallel Sim(3) fitter.

    Returns:
        The Sim(3), or None without the calibration or 3 matched views.
    """
    calibration_root = scene_root.parent / "Calibration"
    if not calibration_root.is_dir():
        return None
    from spatialarena.eval import eval_dtu_mesh

    try:
        raw_c2w = eval_dtu_mesh.load_dtu_camera(str(scene_root.parent))
    except (OSError, ValueError):
        return None
    raw_centers_all = raw_c2w[:, :3, 3]

    norm_centers = []
    raw_centers = []
    for key, view in gt_views.items():
        if key.isdigit():
            index = int(key)
            if 0 <= index < len(raw_centers_all):
                norm_centers.append(view.center)
                raw_centers.append(raw_centers_all[index])
    if len(norm_centers) < 3:
        return None

    norm_centers_arr = np.stack(norm_centers)
    raw_centers_arr = np.stack(raw_centers)

    # Same decomposition as evaluate_mesh's align_coordinate_frame path:
    # best_fit_transform is a pure rigid (rotation + translation) Procrustes
    # fit with no scale DOF, so the scale ratio is estimated first from each
    # set's mean camera-center spread around its own centroid.
    scale_norm = np.linalg.norm(norm_centers_arr -
                                norm_centers_arr.mean(axis=0),
                                axis=1).mean()
    scale_raw = np.linalg.norm(raw_centers_arr - raw_centers_arr.mean(axis=0),
                               axis=1).mean()
    scale = float(scale_raw / scale_norm)
    _, rotation, translation = eval_dtu_mesh.best_fit_transform(
        norm_centers_arr * scale, raw_centers_arr)
    return pose_alignment.Sim3Estimate(
        scale=scale,
        rotation=rotation,
        translation=translation,
        inliers=np.ones(len(norm_centers_arr), dtype=bool),
    )


def dtu_eval_paths(
    scene_info: canonical_dataset.BenchmarkSceneInfo
) -> tuple[int, pathlib.Path, pathlib.Path, pathlib.Path] | None:
    """Returns the DTU MVSNet evaluation inputs of a scene.

    Returns:
        (scan id, STL laser scan, ObsMask, Plane mask), or None if a file is
        missing.
    """
    metadata = scene_info.metadata or {}
    match = re.search(r"\d+", scene_info.scene)
    if match is None:
        return None
    scan_id = int(match.group())
    # evaluate_dtu_mvsnet needs DTU's official high-res laser scan (Points/
    # stl/stlNNN_total.ply, hundreds-of-mm scale -- the frame ObsMask/Plane
    # are defined in), not resolve_gt_geometry_path's "points.ply" (this
    # scene's materialized/normalized sparse COLMAP model, an unrelated
    # unit-ish scale used to seed pose alignment above). Using the wrong one
    # here made every distance exceed max_distance after the raw-calibration
    # fix landed points correctly, giving NaN mean_d2s/mean_s2d for every
    # scene (job 132575, 2026-09-16) instead of the "0 sample" crash from
    # before that fix.
    stl = metadata.get("dtu_stl_path") or metadata.get("stl_path")
    if stl:
        gt_path = pathlib.Path(stl)
        if not gt_path.is_absolute():
            gt_path = scene_info.root_path / gt_path
    else:
        gt_path = (scene_info.root_path.parent / "Points" / "stl" /
                   f"stl{scan_id:03d}_total.ply")
    if not gt_path.is_file():
        gt_path = None
    obj_mask_value = metadata.get("dtu_obj_mask_path") or metadata.get(
        "obj_mask_path")
    plane = metadata.get("dtu_plane_mask_path") or metadata.get(
        "plane_mask_path")
    if obj_mask_value:
        obj_path = pathlib.Path(obj_mask_value)
        if not obj_path.is_absolute():
            obj_path = scene_info.root_path / obj_path
    else:
        obj_path = (scene_info.root_path.parent / "ObsMask" /
                    f"ObsMask{scan_id}_10.mat")
    if plane:
        plane_path = pathlib.Path(plane)
        if not plane_path.is_absolute():
            plane_path = scene_info.root_path / plane_path
    else:
        plane_path = (scene_info.root_path.parent / "ObsMask" /
                      f"Plane{scan_id}.mat")
    if gt_path is None or not obj_path.is_file() or not plane_path.is_file():
        return None
    return scan_id, gt_path, obj_path, plane_path


def evaluate_dtu_mvsnet(scene_info: canonical_dataset.BenchmarkSceneInfo,
                        pred_pcd_path: pathlib.Path,
                        output_dir: pathlib.Path) -> dict[str, float]:
    """Scores a fused cloud with the DTU MVSNet (ObsMask/Plane) evaluator.

    Raises:
        ValueError: If the scene lacks the DTU evaluation inputs.
    """
    paths = dtu_eval_paths(scene_info)
    if paths is None:
        raise ValueError("DTU MVSNet evaluation requires GT point cloud and "
                         "ObsMask/Plane metadata")
    from spatialarena.eval import eval_dtu_mesh

    scan_id, gt_path, obj_mask_path, plane_mask_path = paths
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dtu_mesh.eval(
        scan_id,
        str(pred_pcd_path),
        str(gt_path),
        str(obj_mask_path),
        str(plane_mask_path),
        eval_modal="pcd",
        output_dir=str(output_dir),
    )
    result_path = output_dir / "results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "accuracy": float(payload["mean_d2s"]),
        "completion": float(payload["mean_s2d"]),
        "overall": float(payload["overall"]),
        "precision": float(payload["precision"]),
        "recall": float(payload["recall"]),
        "fscore": float(payload["fscore"]),
    }


def evaluate_mvrecon_scene(
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    prediction_scene_path: pathlib.Path,
    config: MVReconEvaluationConfig | None = None,
    *,
    viewer_dir: pathlib.Path | None = None,
    viewer_scene_root: pathlib.Path | None = None,
) -> dict:
    """Fuses, aligns and scores one scene.

    Args:
        scene_info: The benchmark scene.
        prediction_scene_path: The method's output folder of the scene.
        config: Evaluation settings; defaults to `MVReconEvaluationConfig()`.
        viewer_dir: Where to write the dashboard viewer's cloud, if given.
        viewer_scene_root: Root of the viewer assets every method of a scene
            shares.

    Returns:
        The scene result, with the metrics and the metric protocol.

    Raises:
        ValueError: If the fused or GT cloud is empty, malformed or not
            finite.
    """
    config = config or MVReconEvaluationConfig()
    protocol = protocol_for_dataset(scene_info.dataset, scene_info.metadata)
    artifact_dir = None
    if config.artifact_root is not None:
        artifact_dir = (config.artifact_root / scene_info.task /
                        scene_info.dataset / scene_info.scene)

    geometry_metrics = {}
    fusion_metadata = {}
    fused_path = None
    if config.compute_geometry_metrics:
        fused = fuse_prediction_in_gt_frame(
            scene_info,
            prediction_scene_path,
            config.reconstruction_protocol,
            protocol,
            spatialarena_backend=config.spatialarena_backend,
        )
        pred_points = fused.points
        gt_points = fused.gt_points
        fusion_metadata = fused.fusion_metadata
        for name, points in (("predicted", pred_points), ("ground-truth",
                                                          gt_points)):
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                raise ValueError(
                    f"MVRecon {name} point cloud is empty or malformed after "
                    "fusion/alignment/cropping; no valid reconstruction "
                    "metrics can be computed")
            if not np.isfinite(points).all():
                raise ValueError(f"MVRecon {name} point cloud contains "
                                 "non-finite coordinates")
        if scene_info.dataset.lower() == "dtu" and dtu_eval_paths(
                scene_info) is not None:
            # Everything above operates in this scene's materialized/normalized
            # GT frame; evaluate_dtu_mvsnet below needs DTU's own per-rig
            # calibration frame instead -- see _fit_dtu_raw_calibration_sim3.
            gt_views_for_calibration = load_colmap_views(
                scene_info.sparse_gt_path)
            calibration_estimate = _fit_dtu_raw_calibration_sim3(
                gt_views_for_calibration, scene_info.root_path)
            if calibration_estimate is not None:
                pred_points = calibration_estimate.transform_points(pred_points)
        if artifact_dir is not None:
            fused_path = artifact_dir / "fused_points.ply"
            write_point_cloud(fused_path, pred_points, fused.colors)
        if scene_info.dataset.lower() == "dtu" and dtu_eval_paths(
                scene_info) is not None:
            if fused_path is not None and artifact_dir is not None:
                geometry_metrics = evaluate_dtu_mvsnet(
                    scene_info,
                    fused_path,
                    artifact_dir / "dtu_mvsnet",
                )
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    temp_root = pathlib.Path(tmpdir)
                    temp_fused_path = temp_root / "fused_points.ply"
                    write_point_cloud(temp_fused_path, pred_points)
                    geometry_metrics = evaluate_dtu_mvsnet(
                        scene_info,
                        temp_fused_path,
                        temp_root / "dtu_mvsnet",
                    )
        else:
            geometry_metrics = compute_reconstruction_metrics(
                pred_points,
                gt_points,
                threshold=protocol.eval_threshold,
                down_sample=protocol.down_sample,
                workers=config.metric_workers,
                chunk_points=config.metric_chunk_points,
            )

    metrics = geometry_metrics
    viewer_info = {}
    if viewer_dir is not None and config.compute_geometry_metrics:
        # pylint: disable=import-outside-toplevel
        from spatialarena.eval import viewer_assets

        try:
            if viewer_scene_root:
                scene_dir = viewer_assets.scene_asset_dir(
                    viewer_scene_root, scene_info.task, scene_info.dataset,
                    scene_info.scene)
            else:
                scene_dir = None
            viewer_assets.write_recon_viewer(
                viewer_dir,
                scene_dir,
                scene_info,
                fused,
                metrics=metrics,
                eval_threshold=protocol.eval_threshold,
            )
            viewer_info = {"viewer_dir": str(viewer_dir)}
        except Exception as exc:  # pylint: disable=broad-exception-caught
            viewer_info = {"viewer_error": f"{type(exc).__name__}: {exc}"}

    runtime = prediction_artifacts.read_adapter_runtime(prediction_scene_path)
    if fused_path is not None:
        fused_point_path = str(fused_path)
    else:
        fused_point_path = None
    return viewer_info | {
        "task": scene_info.task,
        "dataset": scene_info.dataset,
        "scene": scene_info.scene,
        "prediction_scene_path": str(prediction_scene_path),
        "reconstruction_protocol": config.reconstruction_protocol,
        "num_metric_units": 1,
        "runtime": runtime,
        "fusion_metadata": fusion_metadata,
        "fused_point_path": fused_point_path,
        "metric_protocol": _metric_protocol(scene_info, config, protocol),
        "metrics": metrics,
    }


def _metric_protocol(scene_info: canonical_dataset.BenchmarkSceneInfo,
                     config: MVReconEvaluationConfig,
                     protocol: MVReconProtocol) -> dict:
    """Returns the scene result's description of how it was scored."""
    backend = reconstruction_backend(scene_info.dataset,
                                     config.spatialarena_backend)
    dataset = scene_info.dataset.lower()
    if dataset in DA3_MVRECON_DATASETS:
        evaluator = "depth_anything_3"
        preprocessing = "depth_anything_3"
    else:
        if dataset == "dtu" and dtu_eval_paths(scene_info) is not None:
            evaluator = "repository_dtu_mvsnet"
        else:
            evaluator = "repository_mvrecon"
        preprocessing = "repository"
    # The TSDF settings do not apply to direct backprojection.
    if backend == "tsdf":
        voxel_length = protocol.voxel_length
        sdf_trunc = protocol.sdf_trunc
        sampling_number = protocol.sampling_number
    else:
        voxel_length = None
        sdf_trunc = None
        sampling_number = None
    return {
        "reconstruction_backend": backend,
        "evaluator": evaluator,
        "preprocessing": preprocessing,
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "max_depth": protocol.max_depth,
        "sampling_number": sampling_number,
        "eval_threshold": protocol.eval_threshold,
        "down_sample": protocol.down_sample,
        "icp_refinement": protocol.icp_refinement,
    }


def aggregate_scene_results(scene_results: list[dict]) -> dict[str, float]:
    """Averages each metric over the scenes that report it."""
    if not scene_results:
        return {}
    keys = ("accuracy", "completion", "overall", "precision", "recall",
            "fscore")
    aggregated = {}
    for key in keys:
        values = [
            float(scene.get("metrics", {}).get(key))
            for scene in scene_results
            if scene.get("metrics", {}).get(key) is not None
        ]
        if values:
            aggregated[key] = float(sum(values) / len(values))
    return aggregated


def _prediction_scene_path(
        experiment_root: pathlib.Path, experiment_name: str, method: str,
        scene_info: canonical_dataset.BenchmarkSceneInfo) -> pathlib.Path:
    """Returns the method's output folder of a scene."""
    return result_layout.resolve_method_scene_output_path(
        experiment_root=experiment_root,
        experiment_name=experiment_name,
        dataset_name=scene_info.dataset,
        method_name=method,
        scene_name=scene_info.scene,
        task_name=scene_info.task,
    )


def run_mvrecon_evaluation(
    *,
    benchmark_root: pathlib.Path,
    experiment_root: pathlib.Path,
    experiment_name: str,
    methods: Iterable[str],
    datasets: Iterable[str],
    scenes: Iterable[str],
    include_scannet: bool,
    config: MVReconEvaluationConfig,
    viewer_dir: pathlib.Path | None = None,
    viewer_scene_root: pathlib.Path | None = None,
) -> dict:
    """Evaluates every MVRecon case of the methods, datasets and scenes.

    Args:
        benchmark_root: Benchmark root folder.
        experiment_root: Root of the method outputs.
        experiment_name: Experiment name under `experiment_root`.
        methods: Methods to evaluate.
        datasets: MVRecon datasets; empty selects all.
        scenes: Scene filter; empty selects all.
        include_scannet: Also evaluate ScanNet.
        config: Evaluation settings.
        viewer_dir: Where to write each case's viewer cloud.
        viewer_scene_root: Root of the viewer assets every method of a scene
            shares.

    Returns:
        The aggregate metrics, scene results and errors.
    """
    from spatialarena.eval import viewer_assets  # pylint: disable=import-outside-toplevel

    scene_results: list[dict] = []
    errors: list[dict] = []
    scene_filter = list(scenes)
    methods = list(methods)
    datasets = list(datasets)
    # One case (one method, scene and dataset: the engine's per-case call)
    # writes straight into viewer_dir; a scene name can repeat across datasets.
    flat_viewer = (len(methods) == 1 and len(scene_filter) == 1 and
                   len(datasets) == 1)
    for spec in dataset_registry.select_dataset_specs(
            tasks=["MVRecon"], datasets=datasets,
            include_scannet=include_scannet):
        metadata_path = spec.data_path(benchmark_root) / "metadata.json"
        if spec.metadata_required and not metadata_path.is_file():
            errors.append({
                "task": spec.task,
                "dataset": spec.dataset,
                "error": "metadata.json is missing"
            })
            continue
        dataset = spec.create_dataset(benchmark_root, scenes=scene_filter)
        for scene_info in dataset.list_scenes():
            for method in methods:
                prediction_scene_path = _prediction_scene_path(
                    experiment_root, experiment_name, method, scene_info)
                try:
                    if viewer_dir is not None:
                        case_viewer = viewer_assets.case_viewer_dir(
                            viewer_dir, flat_viewer, method, scene_info.task,
                            scene_info.dataset, scene_info.scene)
                    else:
                        case_viewer = None
                    scene_result = evaluate_mvrecon_scene(
                        scene_info,
                        prediction_scene_path,
                        config=config,
                        viewer_dir=case_viewer,
                        viewer_scene_root=viewer_scene_root)
                    scene_results.append(scene_result | {"method": method})
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    errors.append({
                        "task": scene_info.task,
                        "dataset": scene_info.dataset,
                        "scene": scene_info.scene,
                        "method": method,
                        "prediction_scene_path": str(prediction_scene_path),
                        "error": str(exc),
                    })
    return {
        "num_scene_results": len(scene_results),
        "num_errors": len(errors),
        "metrics": aggregate_scene_results(scene_results),
        "scene_results": scene_results,
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line (`argv`, default sys.argv)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--experiment_root", type=pathlib.Path, required=True)
    parser.add_argument("--experiment_name", type=str, default="exp")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--include_scannet", action="store_true")
    parser.add_argument("--reconstruction_protocol",
                        choices=["unposed", "posed"],
                        default="unposed")
    parser.add_argument(
        "--spatialarena_backend",
        choices=["tsdf", "direct_backprojection"],
        default="tsdf",
        help=("SpatialArena-only reconstruction backend; all other datasets "
              "keep TSDF."))
    parser.add_argument("--metric_workers", type=int, default=1)
    parser.add_argument("--metric_chunk_points", type=int, default=262144)
    parser.add_argument("--artifact_root", type=pathlib.Path, default=None)
    parser.add_argument("--output_path", type=pathlib.Path, default=None)
    parser.add_argument("--no_geometry_metrics", action="store_true")
    parser.add_argument(
        "--viewer_dir",
        type=pathlib.Path,
        default=None,
        help=("Write the dashboard viewer's cloud (GT frame, colors, error) "
              "there (viewer_assets)."))
    parser.add_argument(
        "--viewer_scene_root",
        type=pathlib.Path,
        default=None,
        help=("Root of the viewer assets every method of a scene shares (GT "
              "points)."))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Runs the MVRecon evaluation CLI and writes or prints the JSON result."""
    args = parse_args(argv)
    result_layout.validate_output_path_outside_benchmark_root(
        args.output_path, args.benchmark_root, label="--output_path")
    artifact_root = args.artifact_root
    if artifact_root is None and args.output_path is not None:
        artifact_root = args.output_path.parent / "artifacts"
    result_layout.validate_output_path_outside_benchmark_root(
        artifact_root, args.benchmark_root, label="--artifact_root")
    for label in ("viewer_dir", "viewer_scene_root"):
        result_layout.validate_output_path_outside_benchmark_root(
            getattr(args, label), args.benchmark_root, label=f"--{label}")

    config = MVReconEvaluationConfig(
        reconstruction_protocol=args.reconstruction_protocol,
        spatialarena_backend=args.spatialarena_backend,
        compute_geometry_metrics=not args.no_geometry_metrics,
        metric_workers=args.metric_workers,
        metric_chunk_points=args.metric_chunk_points,
        artifact_root=artifact_root,
    )
    result = run_mvrecon_evaluation(
        benchmark_root=args.benchmark_root,
        experiment_root=args.experiment_root,
        experiment_name=args.experiment_name,
        methods=args.methods,
        datasets=args.datasets,
        scenes=args.scenes,
        include_scannet=args.include_scannet,
        config=config,
        viewer_dir=args.viewer_dir,
        viewer_scene_root=args.viewer_scene_root,
    )
    text = json.dumps(result, indent=2)
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.strict and result["num_errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
