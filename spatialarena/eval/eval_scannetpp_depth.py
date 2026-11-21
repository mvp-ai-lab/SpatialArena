#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Strict ScanNet++ depth and point-cloud evaluation.

This evaluator follows the ScanNet++ path used by CoMe's ``eval_SCN.py``:
the predicted geometry is transformed into the ScanNet++ mesh frame, refined
with ICP against ``scans/mesh_aligned_0.05.ply``, and evaluated with
precision/recall/F-score at a metric distance threshold.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spatialarena.eval.tnt_tools.evaluation import write_color_distances
from spatialarena.eval.scene_files import (
    DEPTH_EXTENSIONS,
    MASK_EXTENSIONS,
    SceneCamera,
    find_sidecar_file,
    list_scene_dirs,
    load_depth_map,
    load_scene_cameras,
    parse_scene_ids,
    scale_intrinsics_between_sizes,
    write_json,
)

MAX_POINT_NUMBER = 4e6


@dataclass(frozen=True)
class SourcePointCloud:
    name: str
    pcd: o3d.geometry.PointCloud
    sampling: Dict[str, object]


def needs_pointcloud_sources(args: argparse.Namespace) -> bool:
    return args.depth_geometry in {"pointcloud", "both"}


def needs_tsdf_sources(args: argparse.Namespace) -> bool:
    return args.depth_geometry in {"tsdf", "both"}


def source_variant_name(base_name: str, geometry_name: str, args: argparse.Namespace) -> str:
    if args.depth_geometry == "both":
        return f"{geometry_name}_{base_name}"
    return base_name


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def stable_scene_seed(seed: int, scene_id: str) -> int:
    return (int(seed) + zlib.adler32(scene_id.encode("utf-8"))) % (2**32)


def select_frame_indices(num_frames: int, max_frames: Optional[int]) -> List[int]:
    if max_frames is None or max_frames >= num_frames:
        return list(range(num_frames))
    if max_frames <= 0:
        return []
    return np.linspace(0, num_frames - 1, max_frames, dtype=np.int64).tolist()


def read_scene_scale(scene_id: str, args: argparse.Namespace) -> float:
    if args.scale_source == "none":
        return 1.0
    if not args.scale_json_root:
        raise ValueError("--scale_json_root is required unless --scale_source none")
    scene_json = Path(args.scale_json_root) / "scenes" / f"{scene_id}.json"
    if not scene_json.exists():
        raise FileNotFoundError(f"Missing scale JSON for {scene_id}: {scene_json}")
    with scene_json.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    key = "gt_oracle_scale" if args.scale_source == "oracle" else "estimated_scale"
    if payload.get(key) is None:
        raise ValueError(f"Missing scale key {key!r} in {scene_json}")
    return float(payload[key])


def resolve_scene_dirs(args: argparse.Namespace) -> List[Path]:
    return list_scene_dirs(
        dataset_root=Path(args.colmap_root),
        scene_ids=parse_scene_ids(args.scene_ids),
        scene_ids_file=args.scene_ids_file,
    )


def scannetpp_gt_mesh_path(dataset_dir: Path) -> Path:
    gt_path = dataset_dir / "scans" / "mesh_aligned_0.05.ply"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ScanNet++ GT mesh: {gt_path}")
    return gt_path


def load_gt_point_cloud(dataset_dir: Path) -> o3d.geometry.PointCloud:
    gt_path = scannetpp_gt_mesh_path(dataset_dir)
    gt_pcd = o3d.io.read_point_cloud(str(gt_path))
    if len(gt_pcd.points) > 0:
        return gt_pcd

    mesh = o3d.io.read_triangle_mesh(str(gt_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"ScanNet++ GT mesh has no vertices: {gt_path}")
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = mesh.vertices
    if len(mesh.vertex_colors) == len(mesh.vertices):
        gt_pcd.colors = mesh.vertex_colors
    return gt_pcd


def load_reconstruction_point_cloud(
    ply_path: Path,
    add_face_centers: bool = True,
    input_scale: float = 1.0,
) -> SourcePointCloud:
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing reconstruction PLY: {ply_path}")

    loaded = trimesh.load(str(ply_path), process=False)
    points = None
    if isinstance(loaded, trimesh.Scene):
        geometries = [geom for geom in loaded.geometry.values()]
    else:
        geometries = [loaded]

    point_chunks = []
    for geometry in geometries:
        vertices = getattr(geometry, "vertices", None)
        if vertices is None or len(vertices) == 0:
            continue
        vertices = np.asarray(vertices, dtype=np.float64)
        point_chunks.append(vertices)
        faces = getattr(geometry, "faces", None)
        if add_face_centers and faces is not None and len(faces) > 0:
            point_chunks.append(vertices[np.asarray(faces)].mean(axis=1))

    if point_chunks:
        points = np.concatenate(point_chunks, axis=0)
    else:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points, dtype=np.float64)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite] * float(input_scale)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return SourcePointCloud(
        name="ply",
        pcd=pcd,
        sampling={
            "source_path": str(ply_path),
            "num_points": int(points.shape[0]),
            "input_scale": float(input_scale),
            "add_face_centers": bool(add_face_centers),
        },
    )


def camera_intrinsics_for_depth(camera: SceneCamera, depth_shape: Tuple[int, int]) -> np.ndarray:
    height, width = depth_shape
    intrinsics = camera.intrinsics
    if (height, width) != (camera.height, camera.width):
        intrinsics = scale_intrinsics_between_sizes(
            intrinsics,
            source_size=(camera.width, camera.height),
            target_size=(width, height),
        )
    return intrinsics.astype(np.float32, copy=False)


def load_mask(mask_path: Path, target_shape: Tuple[int, int]) -> np.ndarray:
    mask_image = Image.open(mask_path).convert("L")
    target_height, target_width = target_shape
    if mask_image.size != (target_width, target_height):
        mask_image = mask_image.resize((target_width, target_height), Image.NEAREST)
    return np.asarray(mask_image) > 0


def resize_depth_to_shape(
    depth: np.ndarray,
    target_shape: Tuple[int, int],
    mode: str,
) -> np.ndarray:
    if depth.shape == target_shape:
        return depth.astype(np.float32, copy=False)

    target_height, target_width = target_shape
    resampling = (
        Image.Resampling.NEAREST
        if mode == "nearest"
        else Image.Resampling.BILINEAR
    )
    depth_image = Image.fromarray(depth.astype(np.float32), mode="F")
    resized = depth_image.resize((target_width, target_height), resample=resampling)
    return np.asarray(resized, dtype=np.float32)


def resize_bool_mask_to_shape(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool, copy=False)
    target_height, target_width = target_shape
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_image = mask_image.resize(
        (target_width, target_height),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(mask_image) > 0


def load_scannetpp_depth_meters(depth_path: Path, depth_scale: float) -> np.ndarray:
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing ScanNet++ GT depth: {depth_path}")
    depth = np.asarray(Image.open(depth_path), dtype=np.float32)
    return depth / float(depth_scale)


def valid_depth_mask(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > min_depth)
    if max_depth > 0:
        valid &= depth < max_depth
    return valid


def valid_depth_pair_mask(
    predicted_depth: np.ndarray,
    gt_depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    extra_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    valid = (
        np.isfinite(predicted_depth)
        & np.isfinite(gt_depth)
        & (predicted_depth > min_depth)
        & (gt_depth > min_depth)
    )
    if max_depth > 0:
        valid &= (predicted_depth < max_depth) & (gt_depth < max_depth)
    if extra_mask is not None:
        valid &= extra_mask
    return valid


def new_depth_metric_accumulator() -> Dict[str, float]:
    return {
        "num_frames": 0,
        "num_valid_frames": 0,
        "num_pixels": 0,
        "sum_abs_rel": 0.0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_log_sq": 0.0,
        "sum_delta1": 0.0,
    }


def update_depth_metric_accumulator(
    accumulator: Dict[str, float],
    predicted_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, object]:
    accumulator["num_frames"] += 1
    num_pixels = int(valid.sum())
    if num_pixels == 0:
        return {
            "num_pixels": 0,
            "abs_rel": None,
            "mae": None,
            "rmse": None,
            "log_rmse": None,
            "delta1": None,
        }

    pred = predicted_depth[valid].astype(np.float64)
    gt = gt_depth[valid].astype(np.float64)
    diff = pred - gt
    abs_diff = np.abs(diff)
    ratio = pred / gt
    delta = np.maximum(ratio, 1.0 / ratio)
    log_diff = np.log(pred) - np.log(gt)

    sum_abs_rel = float(np.sum(abs_diff / gt))
    sum_abs = float(np.sum(abs_diff))
    sum_sq = float(np.sum(diff * diff))
    sum_log_sq = float(np.sum(log_diff * log_diff))
    sum_delta1 = float(np.sum(delta < 1.25))

    accumulator["num_valid_frames"] += 1
    accumulator["num_pixels"] += num_pixels
    accumulator["sum_abs_rel"] += sum_abs_rel
    accumulator["sum_abs"] += sum_abs
    accumulator["sum_sq"] += sum_sq
    accumulator["sum_log_sq"] += sum_log_sq
    accumulator["sum_delta1"] += sum_delta1

    return {
        "num_pixels": num_pixels,
        "abs_rel": sum_abs_rel / num_pixels,
        "mae": sum_abs / num_pixels,
        "rmse": float(np.sqrt(sum_sq / num_pixels)),
        "log_rmse": float(np.sqrt(sum_log_sq / num_pixels)),
        "delta1": sum_delta1 / num_pixels,
    }


def finalize_depth_metric_accumulator(
    accumulator: Dict[str, float],
    include_sums: bool,
) -> Dict[str, object]:
    num_pixels = int(accumulator["num_pixels"])
    result = {
        "num_frames": int(accumulator["num_frames"]),
        "num_valid_frames": int(accumulator["num_valid_frames"]),
        "num_pixels": num_pixels,
        "abs_rel": None,
        "mae": None,
        "rmse": None,
        "log_rmse": None,
        "delta1": None,
    }
    if num_pixels > 0:
        result.update({
            "abs_rel": float(accumulator["sum_abs_rel"] / num_pixels),
            "mae": float(accumulator["sum_abs"] / num_pixels),
            "rmse": float(np.sqrt(accumulator["sum_sq"] / num_pixels)),
            "log_rmse": float(np.sqrt(accumulator["sum_log_sq"] / num_pixels)),
            "delta1": float(accumulator["sum_delta1"] / num_pixels),
        })
    if include_sums:
        result["_sums"] = {
            "sum_abs_rel": float(accumulator["sum_abs_rel"]),
            "sum_abs": float(accumulator["sum_abs"]),
            "sum_sq": float(accumulator["sum_sq"]),
            "sum_log_sq": float(accumulator["sum_log_sq"]),
            "sum_delta1": float(accumulator["sum_delta1"]),
        }
    return result


def unproject_depth_points(
    depth: np.ndarray,
    valid: np.ndarray,
    camera: SceneCamera,
    points_per_frame: int,
    scale: float,
    scale_application: str,
    rng: np.random.Generator,
) -> np.ndarray:
    flat_indices = np.flatnonzero(valid.reshape(-1))
    if flat_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if points_per_frame > 0 and flat_indices.size > points_per_frame:
        flat_indices = rng.choice(flat_indices, size=points_per_frame, replace=False)

    width = depth.shape[1]
    ys, xs = np.divmod(flat_indices, width)
    z = depth[ys, xs].astype(np.float32)
    if scale_application == "depth":
        z = z * float(scale)
    intrinsics = camera_intrinsics_for_depth(camera, depth.shape)
    x_cam = (xs.astype(np.float32) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y_cam = (ys.astype(np.float32) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points_cam = np.stack([x_cam, y_cam, z], axis=1)

    rotation = camera.w2c[:3, :3].astype(np.float32, copy=False)
    translation = camera.w2c[:3, 3].astype(np.float32, copy=False)
    points_world = (points_cam - translation[None]) @ rotation
    if scale_application == "world":
        points_world = points_world * float(scale)
    return points_world.astype(np.float32, copy=False)


def downsample_points(
    points: np.ndarray,
    max_points: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    return pcd


def mesh_points_with_face_centers(
    mesh: o3d.geometry.TriangleMesh,
    max_points: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    chunks = []
    if vertices.shape[0] > 0:
        chunks.append(vertices)
    if vertices.shape[0] > 0 and triangles.shape[0] > 0:
        chunks.append(vertices[triangles].mean(axis=1))
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    points = np.concatenate(chunks, axis=0)
    finite = np.isfinite(points).all(axis=1)
    return downsample_points(points[finite], max_points, rng)


def make_tsdf_volume(args: argparse.Namespace):
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.tsdf_voxel_length),
        sdf_trunc=float(args.tsdf_sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )


def scaled_depth_for_geometry(
    depth: np.ndarray,
    scale: float,
    scale_application: str,
) -> np.ndarray:
    depth_metric = depth.astype(np.float32, copy=True)
    if scale_application in {"depth", "world"}:
        depth_metric *= float(scale)
    return depth_metric


def tsdf_extrinsic(
    camera: SceneCamera,
    scale: float,
    scale_application: str,
) -> np.ndarray:
    w2c = camera.w2c.astype(np.float64, copy=True)
    if scale_application == "world":
        w2c[:3, 3] *= float(scale)
    return w2c


def integrate_depth_into_tsdf(
    volume,
    depth: np.ndarray,
    valid: np.ndarray,
    camera: SceneCamera,
    scale: float,
    args: argparse.Namespace,
) -> None:
    depth_metric = scaled_depth_for_geometry(
        depth=depth,
        scale=scale,
        scale_application=args.scale_application,
    )
    depth_metric[~valid] = 0.0
    height, width = depth_metric.shape
    intrinsics = camera_intrinsics_for_depth(camera, depth_metric.shape)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2]),
        float(intrinsics[1, 2]),
    )
    color = np.zeros((height, width, 3), dtype=np.uint8)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(np.ascontiguousarray(depth_metric)),
        depth_scale=1.0,
        depth_trunc=float(args.tsdf_depth_trunc),
        convert_rgb_to_intensity=False,
    )
    volume.integrate(
        rgbd,
        intrinsic,
        tsdf_extrinsic(
            camera=camera,
            scale=scale,
            scale_application=args.scale_application,
        ),
    )


def tsdf_volume_to_source(
    volume,
    name: str,
    sampling: Dict[str, object],
    args: argparse.Namespace,
    rng: np.random.Generator,
    mesh_output_path: Optional[Path] = None,
) -> SourcePointCloud:
    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    if mesh_output_path is not None:
        mesh_output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mesh_output_path), mesh)
        sampling["mesh_path"] = str(mesh_output_path)
    points = mesh_points_with_face_centers(
        mesh=mesh,
        max_points=args.max_source_points,
        rng=rng,
    )
    sampling.update({
        "geometry": "tsdf",
        "num_mesh_vertices": int(len(mesh.vertices)),
        "num_mesh_triangles": int(len(mesh.triangles)),
        "num_points": int(points.shape[0]),
        "tsdf_voxel_length": float(args.tsdf_voxel_length),
        "tsdf_sdf_trunc": float(args.tsdf_sdf_trunc),
        "tsdf_depth_trunc": float(args.tsdf_depth_trunc),
    })
    return SourcePointCloud(name=name, pcd=make_point_cloud(points), sampling=sampling)


def scannetpp_gt_depth_path(dataset_dir: Path, image_name: str) -> Path:
    return dataset_dir / "iphone" / "depth" / Path(image_name).with_suffix(".png").name


def compute_scene_depth_metrics(
    scene_dir: Path,
    dataset_dir: Path,
    scale: float,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    scene_id = scene_dir.name
    cameras = load_scene_cameras(
        scene_dir=scene_dir,
        image_folder=args.image_folder,
        model_folder=args.model_folder,
    )
    depth_dir = scene_dir / args.depth_folder
    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing predicted depth folder: {depth_dir}")
    mask_dir = scene_dir / args.mask_folder if args.mask_folder else None
    if mask_dir is not None and not mask_dir.exists():
        raise FileNotFoundError(f"Missing depth mask folder: {mask_dir}")

    accumulators = {"unmasked": new_depth_metric_accumulator()}
    if mask_dir is not None:
        accumulators["masked"] = new_depth_metric_accumulator()

    frame_indices = select_frame_indices(len(cameras), args.max_frames)
    frame_stats = []
    for camera_index in frame_indices:
        camera = cameras[camera_index]
        depth_path = find_sidecar_file(depth_dir, camera.image_name, DEPTH_EXTENSIONS)
        if depth_path is None:
            raise FileNotFoundError(
                f"Missing predicted depth for {scene_id}/{camera.image_name} under {depth_dir}"
            )
        depth = load_depth_map(
            depth_path,
            npz_key=args.npz_key,
            png_depth_scale=args.predicted_png_depth_scale,
            allow_rgb_depth_png=args.allow_rgb_depth_png,
        )
        predicted_metric_depth = scaled_depth_for_geometry(
            depth=depth,
            scale=scale,
            scale_application=args.scale_application,
        )

        gt_depth_path = scannetpp_gt_depth_path(dataset_dir, camera.image_name)
        gt_metric_depth = load_scannetpp_depth_meters(
            gt_depth_path,
            depth_scale=args.gt_depth_scale,
        )
        predicted_metric_depth = resize_depth_to_shape(
            predicted_metric_depth,
            target_shape=gt_metric_depth.shape,
            mode=args.depth_metric_resize_mode,
        )

        valid = valid_depth_pair_mask(
            predicted_metric_depth,
            gt_metric_depth,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        frame_record = {
            "image_name": camera.image_name,
            "depth_path": str(depth_path),
            "gt_depth_path": str(gt_depth_path),
            "unmasked": update_depth_metric_accumulator(
                accumulators["unmasked"],
                predicted_depth=predicted_metric_depth,
                gt_depth=gt_metric_depth,
                valid=valid,
            ),
        }

        if mask_dir is not None:
            mask_path = find_sidecar_file(mask_dir, camera.image_name, MASK_EXTENSIONS)
            if mask_path is None:
                raise FileNotFoundError(
                    f"Missing depth mask for {scene_id}/{camera.image_name} under {mask_dir}"
                )
            mask = load_mask(mask_path, target_shape=depth.shape)
            mask = resize_bool_mask_to_shape(mask, target_shape=gt_metric_depth.shape)
            masked_valid = valid & mask
            frame_record["mask_path"] = str(mask_path)
            frame_record["masked"] = update_depth_metric_accumulator(
                accumulators["masked"],
                predicted_depth=predicted_metric_depth,
                gt_depth=gt_metric_depth,
                valid=masked_valid,
            )
            frame_record["mask_keep_ratio"] = (
                float(masked_valid.sum() / max(valid.sum(), 1))
                if valid.any()
                else None
            )
        else:
            frame_record["mask_path"] = None

        frame_stats.append(frame_record)

    depth_metrics = {
        name: finalize_depth_metric_accumulator(accumulator, include_sums=True)
        for name, accumulator in accumulators.items()
    }
    return depth_metrics, frame_stats


def crop_point_cloud(
    pcd: o3d.geometry.PointCloud,
    crop_volume: o3d.geometry.AxisAlignedBoundingBox,
) -> o3d.geometry.PointCloud:
    if hasattr(crop_volume, "crop_point_cloud"):
        return crop_volume.crop_point_cloud(pcd)
    return pcd.crop(crop_volume)


def crop_and_downsample(
    pcd: o3d.geometry.PointCloud,
    crop_volume: o3d.geometry.AxisAlignedBoundingBox,
    downsample_method: str = "voxel",
    voxel_size: float = 0.01,
    trans: np.ndarray = np.eye(4),
) -> o3d.geometry.PointCloud:
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(trans)
    pcd_crop = crop_point_cloud(pcd_copy, crop_volume)
    if downsample_method == "voxel":
        return pcd_crop.voxel_down_sample(voxel_size)
    if downsample_method == "uniform":
        n_points = len(pcd_crop.points)
        if n_points > MAX_POINT_NUMBER:
            ds_rate = int(round(n_points / float(MAX_POINT_NUMBER)))
            return pcd_crop.uniform_down_sample(ds_rate)
    return pcd_crop


def registration_vol_ds(
    source: o3d.geometry.PointCloud,
    gt_target: o3d.geometry.PointCloud,
    init_trans: np.ndarray,
    crop_volume: o3d.geometry.AxisAlignedBoundingBox,
    voxel_size: float,
    threshold: float,
    max_itr: int,
    verbose: bool = True,
):
    if verbose:
        print(f"[Registration] voxel_size: {voxel_size:f}, threshold: {threshold:f}")
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    source_eval = crop_and_downsample(
        source,
        crop_volume,
        downsample_method="voxel",
        voxel_size=voxel_size,
        trans=init_trans,
    )
    target_eval = crop_and_downsample(
        gt_target,
        crop_volume,
        downsample_method="voxel",
        voxel_size=voxel_size,
    )
    reg = o3d.pipelines.registration.registration_icp(
        source_eval,
        target_eval,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        o3d.pipelines.registration.ICPConvergenceCriteria(1e-6, max_iteration=max_itr),
    )
    reg.transformation = reg.transformation @ init_trans
    return reg


def registration_unif(
    source: o3d.geometry.PointCloud,
    gt_target: o3d.geometry.PointCloud,
    init_trans: np.ndarray,
    crop_volume: o3d.geometry.AxisAlignedBoundingBox,
    threshold: float,
    max_itr: int,
    verbose: bool = True,
):
    if verbose:
        print(f"[Registration] threshold: {threshold:f}")
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    source_eval = crop_and_downsample(
        source,
        crop_volume,
        downsample_method="uniform",
        trans=init_trans,
    )
    target_eval = crop_and_downsample(
        gt_target,
        crop_volume,
        downsample_method="uniform",
    )
    reg = o3d.pipelines.registration.registration_icp(
        source_eval,
        target_eval,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        o3d.pipelines.registration.ICPConvergenceCriteria(1e-6, max_iteration=max_itr),
    )
    reg.transformation = reg.transformation @ init_trans
    return reg


def collect_depth_source_point_clouds(
    scene_dir: Path,
    scale: float,
    args: argparse.Namespace,
) -> List[SourcePointCloud]:
    scene_id = scene_dir.name
    cameras = load_scene_cameras(
        scene_dir=scene_dir,
        image_folder=args.image_folder,
        model_folder=args.model_folder,
    )
    depth_dir = scene_dir / args.depth_folder
    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing predicted depth folder: {depth_dir}")
    mask_dir = scene_dir / args.mask_folder if args.mask_folder else None
    if mask_dir is not None and not mask_dir.exists():
        raise FileNotFoundError(f"Missing depth mask folder: {mask_dir}")

    rng = np.random.default_rng(stable_scene_seed(args.seed, scene_id))
    frame_indices = select_frame_indices(len(cameras), args.max_frames)
    pointcloud_enabled = needs_pointcloud_sources(args)
    tsdf_enabled = needs_tsdf_sources(args)
    unmasked_chunks: List[np.ndarray] = []
    masked_chunks: List[np.ndarray] = []
    unmasked_volume = make_tsdf_volume(args) if tsdf_enabled else None
    masked_volume = make_tsdf_volume(args) if tsdf_enabled and mask_dir is not None else None
    frame_stats = []

    for camera_index in frame_indices:
        camera = cameras[camera_index]
        depth_path = find_sidecar_file(depth_dir, camera.image_name, DEPTH_EXTENSIONS)
        if depth_path is None:
            raise FileNotFoundError(
                f"Missing predicted depth for {scene_id}/{camera.image_name} under {depth_dir}"
            )
        depth = load_depth_map(
            depth_path,
            npz_key=args.npz_key,
            png_depth_scale=args.predicted_png_depth_scale,
            allow_rgb_depth_png=args.allow_rgb_depth_png,
        )
        valid = valid_depth_mask(depth, args.min_depth, args.max_depth)
        mask = None
        if mask_dir is not None:
            mask_path = find_sidecar_file(mask_dir, camera.image_name, MASK_EXTENSIONS)
            if mask_path is None:
                raise FileNotFoundError(
                    f"Missing depth mask for {scene_id}/{camera.image_name} under {mask_dir}"
                )
            mask = load_mask(mask_path, target_shape=depth.shape)
        else:
            mask_path = None

        unmasked_points_count = None
        masked_points_count = None
        if pointcloud_enabled:
            unmasked_points = unproject_depth_points(
                depth=depth,
                valid=valid,
                camera=camera,
                points_per_frame=args.points_per_frame,
                scale=scale,
                scale_application=args.scale_application,
                rng=rng,
            )
            unmasked_chunks.append(unmasked_points)
            unmasked_points_count = int(unmasked_points.shape[0])

        if tsdf_enabled and unmasked_volume is not None:
            integrate_depth_into_tsdf(
                volume=unmasked_volume,
                depth=depth,
                valid=valid,
                camera=camera,
                scale=scale,
                args=args,
            )

        if mask is not None:
            masked_valid = valid & mask
            if pointcloud_enabled:
                masked_points = unproject_depth_points(
                    depth=depth,
                    valid=masked_valid,
                    camera=camera,
                    points_per_frame=args.points_per_frame,
                    scale=scale,
                    scale_application=args.scale_application,
                    rng=rng,
                )
                masked_chunks.append(masked_points)
                masked_points_count = int(masked_points.shape[0])
            if tsdf_enabled and masked_volume is not None:
                integrate_depth_into_tsdf(
                    volume=masked_volume,
                    depth=depth,
                    valid=masked_valid,
                    camera=camera,
                    scale=scale,
                    args=args,
                )
            masked_count = int((valid & mask).sum())
        else:
            masked_count = None

        valid_count = int(valid.sum())
        frame_stats.append({
            "image_name": camera.image_name,
            "depth_path": str(depth_path),
            "mask_path": str(mask_path) if mask_path is not None else None,
            "valid_pixels": valid_count,
            "masked_valid_pixels": masked_count,
            "mask_keep_ratio": (
                float(masked_count / max(valid_count, 1))
                if masked_count is not None
                else None
            ),
            "unmasked_points": unmasked_points_count,
            "masked_points": masked_points_count,
        })

    sources = []
    base_sampling = {
        "num_cameras": len(cameras),
        "frame_indices": [int(index) for index in frame_indices],
        "scale": float(scale),
        "scale_source": args.scale_source,
        "scale_application": args.scale_application,
        "scale_definition": (
            "metric_depth = scale * predicted_depth"
            if args.scale_application == "depth"
            else "metric_points = scale * colmap_points"
        ),
        "frame_stats": frame_stats,
    }

    if pointcloud_enabled:
        unmasked_all = (
            np.concatenate(unmasked_chunks, axis=0)
            if unmasked_chunks
            else np.zeros((0, 3), dtype=np.float32)
        )
        unmasked_all = downsample_points(unmasked_all, args.max_source_points, rng)
        unmasked_sampling = dict(base_sampling)
        unmasked_sampling.update({
            "geometry": "pointcloud",
            "num_points": int(unmasked_all.shape[0]),
            "points_per_frame": int(args.points_per_frame),
        })
        sources.append(
            SourcePointCloud(
                name=source_variant_name("unmasked", "pointcloud", args),
                pcd=make_point_cloud(unmasked_all),
                sampling=unmasked_sampling,
            )
        )

    if pointcloud_enabled and mask_dir is not None:
        masked_all = (
            np.concatenate(masked_chunks, axis=0)
            if masked_chunks
            else np.zeros((0, 3), dtype=np.float32)
        )
        masked_all = downsample_points(masked_all, args.max_source_points, rng)
        keep_ratios = [
            frame["mask_keep_ratio"]
            for frame in frame_stats
            if frame["mask_keep_ratio"] is not None
        ]
        masked_sampling = dict(base_sampling)
        masked_sampling.update({
            "geometry": "pointcloud",
            "num_points": int(masked_all.shape[0]),
            "points_per_frame": int(args.points_per_frame),
            "mask_folder": args.mask_folder,
            "mean_mask_keep_ratio": (
                float(np.mean(keep_ratios)) if keep_ratios else None
            ),
        })
        sources.append(
            SourcePointCloud(
                name=source_variant_name("masked", "pointcloud", args),
                pcd=make_point_cloud(masked_all),
                sampling=masked_sampling,
            )
        )

    if tsdf_enabled and unmasked_volume is not None:
        mesh_dir = (
            Path(args.output_dir) / "scenes" / scene_id / "tsdf_meshes"
            if args.write_tsdf_meshes
            else None
        )
        sources.append(
            tsdf_volume_to_source(
                volume=unmasked_volume,
                name=source_variant_name("unmasked", "tsdf", args),
                sampling=dict(base_sampling),
                args=args,
                rng=rng,
                mesh_output_path=(mesh_dir / "unmasked.ply") if mesh_dir else None,
            )
        )

    if tsdf_enabled and masked_volume is not None:
        keep_ratios = [
            frame["mask_keep_ratio"]
            for frame in frame_stats
            if frame["mask_keep_ratio"] is not None
        ]
        masked_sampling = dict(base_sampling)
        masked_sampling.update({
            "mask_folder": args.mask_folder,
            "mean_mask_keep_ratio": (
                float(np.mean(keep_ratios)) if keep_ratios else None
            ),
        })
        mesh_dir = (
            Path(args.output_dir) / "scenes" / scene_id / "tsdf_meshes"
            if args.write_tsdf_meshes
            else None
        )
        sources.append(
            tsdf_volume_to_source(
                volume=masked_volume,
                name=source_variant_name("masked", "tsdf", args),
                sampling=masked_sampling,
                args=args,
                rng=rng,
                mesh_output_path=(mesh_dir / "masked.ply") if mesh_dir else None,
            )
        )
    return sources


def come_initial_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        o3d.geometry.TriangleMesh.create_coordinate_frame()
        .get_rotation_matrix_from_xyz((0.0, np.pi, np.pi / 2.0))
    )
    return transform


def initial_transform(args: argparse.Namespace) -> np.ndarray:
    if args.initial_transform == "auto":
        if args.mode == "ply":
            return come_initial_transform()
        return np.eye(4, dtype=np.float64)
    if args.initial_transform == "none":
        return np.eye(4, dtype=np.float64)
    if args.initial_transform == "come":
        return come_initial_transform()
    raise ValueError(f"Unsupported initial transform: {args.initial_transform}")


def compute_histogram_metrics(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
    crop_volume: o3d.geometry.AxisAlignedBoundingBox,
    voxel_size: float,
    thresholds: Sequence[float],
    output_dir: Optional[Path],
    scene_name: str,
    variant_name: str,
    write_visualizations: bool,
) -> Dict[str, object]:
    source_eval = copy.deepcopy(source)
    source_eval.transform(transform)
    source_eval = crop_point_cloud(source_eval, crop_volume)
    source_eval = source_eval.voxel_down_sample(voxel_size)

    target_eval = crop_point_cloud(copy.deepcopy(target), crop_volume)
    target_eval = target_eval.voxel_down_sample(voxel_size)

    if len(source_eval.points) == 0 or len(target_eval.points) == 0:
        return {
            "status": "empty",
            "num_source_points": int(len(source.points)),
            "num_target_points": int(len(target.points)),
            "num_source_eval_points": int(len(source_eval.points)),
            "num_target_eval_points": int(len(target_eval.points)),
        }

    source_to_target = np.asarray(
        source_eval.compute_point_cloud_distance(target_eval), dtype=np.float64
    )
    target_to_source = np.asarray(
        target_eval.compute_point_cloud_distance(source_eval), dtype=np.float64
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / f"{scene_name}_{variant_name}.source_to_target.txt", source_to_target)
        np.savetxt(output_dir / f"{scene_name}_{variant_name}.target_to_source.txt", target_to_source)
        if write_visualizations:
            write_color_distances(
                str(output_dir / f"{scene_name}_{variant_name}.precision.ply"),
                source_eval,
                source_to_target,
                max_distance=3.0 * max(thresholds),
            )
            write_color_distances(
                str(output_dir / f"{scene_name}_{variant_name}.recall.ply"),
                target_eval,
                target_to_source,
                max_distance=3.0 * max(thresholds),
            )

    threshold_metrics = {}
    for threshold in thresholds:
        precision = float(np.mean(source_to_target < threshold))
        recall = float(np.mean(target_to_source < threshold))
        denom = precision + recall
        fscore = 0.0 if denom <= 0.0 else 2.0 * precision * recall / denom
        threshold_metrics[f"{threshold:g}m"] = {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "fscore": float(fscore),
        }

    return {
        "status": "ok",
        "num_source_points": int(len(source.points)),
        "num_target_points": int(len(target.points)),
        "num_source_eval_points": int(len(source_eval.points)),
        "num_target_eval_points": int(len(target_eval.points)),
        "source_to_target_mean": float(source_to_target.mean()),
        "source_to_target_median": float(np.median(source_to_target)),
        "source_to_target_p90": float(np.percentile(source_to_target, 90)),
        "target_to_source_mean": float(target_to_source.mean()),
        "target_to_source_median": float(np.median(target_to_source)),
        "target_to_source_p90": float(np.percentile(target_to_source, 90)),
        "chamfer_mean": float(0.5 * (source_to_target.mean() + target_to_source.mean())),
        "thresholds": threshold_metrics,
    }


def evaluate_source_against_scannetpp(
    dataset_dir: Path,
    source: SourcePointCloud,
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    gt_pcd = load_gt_point_cloud(dataset_dir)
    crop_volume = gt_pcd.get_axis_aligned_bounding_box()
    init_transform = initial_transform(args)
    source_for_icp = copy.deepcopy(source.pcd)
    source_for_icp.transform(init_transform)

    if args.skip_icp:
        trajectory_transform = init_transform
        icp = {"enabled": False}
    else:
        d_tau = float(args.distance_tau)
        r2 = registration_vol_ds(
            source_for_icp,
            gt_pcd,
            np.eye(4),
            crop_volume,
            d_tau,
            d_tau * 80.0,
            args.icp_iterations,
            verbose=args.verbose,
        )
        r3 = registration_vol_ds(
            source_for_icp,
            gt_pcd,
            r2.transformation,
            crop_volume,
            d_tau / 2.0,
            d_tau * 20.0,
            args.icp_iterations,
            verbose=args.verbose,
        )
        r = registration_unif(
            source_for_icp,
            gt_pcd,
            r3.transformation,
            crop_volume,
            2.0 * d_tau,
            args.icp_iterations,
            verbose=args.verbose,
        )
        trajectory_transform = r.transformation @ init_transform
        icp = {
            "enabled": True,
            "fitness": float(r.fitness),
            "inlier_rmse": float(r.inlier_rmse),
            "coarse_transform": r2.transformation.tolist(),
            "medium_transform": r3.transformation.tolist(),
            "final_transform_after_initial": r.transformation.tolist(),
        }

    metrics_dir = out_dir / "artifacts" if args.write_distance_artifacts else None
    metrics = compute_histogram_metrics(
        source=source.pcd,
        target=gt_pcd,
        transform=trajectory_transform,
        crop_volume=crop_volume,
        voxel_size=float(args.distance_tau) / 2.0,
        thresholds=args.fscore_thresholds_m,
        output_dir=metrics_dir,
        scene_name=dataset_dir.name,
        variant_name=source.name,
        write_visualizations=args.write_visualizations,
    )
    metrics["initial_transform"] = init_transform.tolist()
    metrics["trajectory_transform"] = trajectory_transform.tolist()
    metrics["icp"] = icp
    metrics["sampling"] = source.sampling
    return metrics


def process_depth_scene(scene_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    start = time.time()
    scene_id = scene_dir.name
    dataset_dir = Path(args.scannetpp_root) / scene_id
    scale = read_scene_scale(scene_id, args)
    scene_out_dir = Path(args.output_dir) / "scenes" / scene_id
    sources = collect_depth_source_point_clouds(scene_dir, scale=scale, args=args)
    depth_metrics = None
    depth_metric_frame_stats = None
    if args.compute_depth_metrics:
        depth_metrics, depth_metric_frame_stats = compute_scene_depth_metrics(
            scene_dir=scene_dir,
            dataset_dir=dataset_dir,
            scale=scale,
            args=args,
        )

    evaluations = {}
    for source in sources:
        evaluations[source.name] = evaluate_source_against_scannetpp(
            dataset_dir=dataset_dir,
            source=source,
            out_dir=scene_out_dir / source.name,
            args=args,
        )

    result = {
        "scene_id": scene_id,
        "status": "ok",
        "mode": "depth",
        "dataset_dir": str(dataset_dir),
        "scene_dir": str(scene_dir),
        "depth_folder": args.depth_folder,
        "mask_folder": args.mask_folder,
        "depth_geometry": args.depth_geometry,
        "scale": float(scale),
        "scale_source": args.scale_source,
        "evaluations": evaluations,
        "seconds": round(time.time() - start, 4),
    }
    if depth_metrics is not None:
        result["depth_metric_metadata"] = {
            "gt_depth_scale": float(args.gt_depth_scale),
            "resize_mode": args.depth_metric_resize_mode,
            "metric_definition": "predicted_metric_depth = scale * predicted_depth",
            "delta1_definition": "mean(max(pred / gt, gt / pred) < 1.25)",
        }
        result["depth_metrics"] = depth_metrics
        result["depth_metric_frame_stats"] = depth_metric_frame_stats
    write_json(scene_out_dir / "metrics.json", result)
    return result


def resolve_scene_ply(scene_id: str, args: argparse.Namespace) -> Path:
    if args.input_ply:
        return Path(args.input_ply)
    if not args.input_ply_root:
        raise ValueError("--input_ply or --input_ply_root is required in ply mode")
    return Path(args.input_ply_root) / scene_id / args.input_ply_name


def process_ply_scene(scene_id: str, args: argparse.Namespace) -> Dict[str, object]:
    start = time.time()
    dataset_dir = Path(args.scannetpp_root) / scene_id
    ply_path = resolve_scene_ply(scene_id, args)
    source = load_reconstruction_point_cloud(
        ply_path=ply_path,
        add_face_centers=args.add_face_centers,
        input_scale=args.input_scale,
    )
    scene_out_dir = Path(args.output_dir) / "scenes" / scene_id
    metrics = evaluate_source_against_scannetpp(
        dataset_dir=dataset_dir,
        source=source,
        out_dir=scene_out_dir / "ply",
        args=args,
    )
    result = {
        "scene_id": scene_id,
        "status": "ok",
        "mode": "ply",
        "dataset_dir": str(dataset_dir),
        "input_ply": str(ply_path),
        "evaluations": {"ply": metrics},
        "seconds": round(time.time() - start, 4),
    }
    write_json(scene_out_dir / "metrics.json", result)
    return result


def average_evaluation_metrics(
    scene_stats: Sequence[Dict[str, object]],
    variant: str,
) -> Dict[str, object]:
    metrics = []
    for scene in scene_stats:
        if scene.get("status") != "ok":
            continue
        variant_metrics = scene.get("evaluations", {}).get(variant)
        if variant_metrics and variant_metrics.get("status") == "ok":
            metrics.append(variant_metrics)
    if not metrics:
        return {}

    summary = {
        "num_scenes": len(metrics),
        "mean_source_to_target": float(np.mean([m["source_to_target_mean"] for m in metrics])),
        "mean_target_to_source": float(np.mean([m["target_to_source_mean"] for m in metrics])),
        "mean_chamfer": float(np.mean([m["chamfer_mean"] for m in metrics])),
        "mean_num_source_eval_points": float(
            np.mean([m["num_source_eval_points"] for m in metrics])
        ),
        "mean_num_target_eval_points": float(
            np.mean([m["num_target_eval_points"] for m in metrics])
        ),
        "thresholds": {},
    }
    for threshold_key in metrics[0].get("thresholds", {}):
        summary["thresholds"][threshold_key] = {
            "threshold": metrics[0]["thresholds"][threshold_key]["threshold"],
            "precision": float(
                np.mean([m["thresholds"][threshold_key]["precision"] for m in metrics])
            ),
            "recall": float(
                np.mean([m["thresholds"][threshold_key]["recall"] for m in metrics])
            ),
            "fscore": float(
                np.mean([m["thresholds"][threshold_key]["fscore"] for m in metrics])
            ),
        }
    return summary


def add_depth_metric_sums(
    accumulator: Dict[str, float],
    metric: Dict[str, object],
) -> None:
    sums = metric.get("_sums")
    if not isinstance(sums, dict):
        return
    accumulator["num_frames"] += int(metric.get("num_frames") or 0)
    accumulator["num_valid_frames"] += int(metric.get("num_valid_frames") or 0)
    accumulator["num_pixels"] += int(metric.get("num_pixels") or 0)
    for key in ("sum_abs_rel", "sum_abs", "sum_sq", "sum_log_sq", "sum_delta1"):
        accumulator[key] += float(sums.get(key, 0.0) or 0.0)


def average_depth_metric_variant(
    scene_stats: Sequence[Dict[str, object]],
    variant: str,
) -> Dict[str, object]:
    accumulator = new_depth_metric_accumulator()
    num_scenes = 0
    for scene in scene_stats:
        if scene.get("status") != "ok":
            continue
        metric = scene.get("depth_metrics", {}).get(variant)
        if isinstance(metric, dict) and isinstance(metric.get("_sums"), dict):
            add_depth_metric_sums(accumulator, metric)
            num_scenes += 1
    if num_scenes == 0:
        return {}

    summary = finalize_depth_metric_accumulator(accumulator, include_sums=False)
    summary["num_scenes"] = num_scenes
    return summary


def depth_metric_delta(
    unmasked: Dict[str, object],
    masked: Dict[str, object],
) -> Dict[str, object]:
    delta = {
        "num_pixels_delta": int(masked["num_pixels"]) - int(unmasked["num_pixels"]),
        "num_valid_frames_delta": (
            int(masked["num_valid_frames"]) - int(unmasked["num_valid_frames"])
        ),
    }
    for key in ("abs_rel", "mae", "rmse", "log_rmse", "delta1"):
        if masked.get(key) is None or unmasked.get(key) is None:
            delta[f"{key}_delta"] = None
        else:
            delta[f"{key}_delta"] = float(masked[key] - unmasked[key])
    return delta


def aggregate_depth_metrics(scene_stats: Sequence[Dict[str, object]]) -> Dict[str, object]:
    ok = [scene for scene in scene_stats if scene.get("status") == "ok"]
    variants = sorted({
        variant
        for scene in ok
        for variant in scene.get("depth_metrics", {}).keys()
    })
    aggregate = {
        variant: average_depth_metric_variant(ok, variant)
        for variant in variants
    }
    if aggregate.get("unmasked") and aggregate.get("masked"):
        aggregate["masked_minus_unmasked"] = depth_metric_delta(
            unmasked=aggregate["unmasked"],
            masked=aggregate["masked"],
        )
    return aggregate


def aggregate_results(scene_stats: Sequence[Dict[str, object]]) -> Dict[str, object]:
    ok = [scene for scene in scene_stats if scene.get("status") == "ok"]
    variants = sorted({
        variant
        for scene in ok
        for variant in scene.get("evaluations", {}).keys()
    })
    aggregate = {
        "num_scenes": len(scene_stats),
        "num_ok": len(ok),
        "num_failed": len(scene_stats) - len(ok),
        "variants": {
            variant: average_evaluation_metrics(ok, variant)
            for variant in variants
        },
    }
    delta_pairs = [("unmasked", "masked")]
    for prefix in ("pointcloud", "tsdf"):
        delta_pairs.append((f"{prefix}_unmasked", f"{prefix}_masked"))
    for unmasked_key, masked_key in delta_pairs:
        if masked_key not in aggregate["variants"] or unmasked_key not in aggregate["variants"]:
            continue
        masked = aggregate["variants"][masked_key]
        unmasked = aggregate["variants"][unmasked_key]
        if masked and unmasked:
            delta = {
                "mean_chamfer_delta": masked["mean_chamfer"] - unmasked["mean_chamfer"],
                "mean_source_to_target_delta": (
                    masked["mean_source_to_target"] - unmasked["mean_source_to_target"]
                ),
                "mean_target_to_source_delta": (
                    masked["mean_target_to_source"] - unmasked["mean_target_to_source"]
                ),
                "thresholds": {},
            }
            for key in masked["thresholds"]:
                if key not in unmasked["thresholds"]:
                    continue
                delta["thresholds"][key] = {
                    "threshold": masked["thresholds"][key]["threshold"],
                    "precision_delta": (
                        masked["thresholds"][key]["precision"]
                        - unmasked["thresholds"][key]["precision"]
                    ),
                    "recall_delta": (
                        masked["thresholds"][key]["recall"]
                        - unmasked["thresholds"][key]["recall"]
                    ),
                    "fscore_delta": (
                        masked["thresholds"][key]["fscore"]
                        - unmasked["thresholds"][key]["fscore"]
                    ),
                }
            delta_key = (
                "masked_minus_unmasked"
                if unmasked_key == "unmasked"
                else f"{masked_key}_minus_{unmasked_key}"
            )
            aggregate[delta_key] = delta
    depth_metrics = aggregate_depth_metrics(ok)
    if depth_metrics:
        aggregate["depth_metrics"] = depth_metrics
    return aggregate


def run_evaluation(
    dataset_dir: str,
    ply_path: str,
    out_dir: str,
    view_crop: bool = False,
    distance_tau: float = 0.05,
) -> Dict[str, object]:
    """CoMe-style single-scene ScanNet++ PLY evaluation entry point."""
    args = argparse.Namespace(
        distance_tau=distance_tau,
        fscore_thresholds_m=[distance_tau],
        initial_transform="come",
        mode="ply",
        skip_icp=False,
        icp_iterations=20,
        verbose=True,
        write_distance_artifacts=True,
        write_visualizations=bool(view_crop),
    )
    dataset_path = Path(dataset_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source = load_reconstruction_point_cloud(Path(ply_path))
    metrics = evaluate_source_against_scannetpp(
        dataset_dir=dataset_path,
        source=source,
        out_dir=output_path,
        args=args,
    )
    write_json(output_path / "results.json", {
        "scene_id": dataset_path.name,
        "input_ply": str(ply_path),
        "evaluations": {"ply": metrics},
    })
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scannetpp_root",
        default=os.path.expanduser(os.environ.get("SCANNETPP_ROOT", "~/datasets/scannetpp/raw")),
    )
    parser.add_argument(
        "--colmap_root",
        default=os.path.expanduser(os.environ.get("SCANNETPP_COLMAP_ROOT", "~/datasets/scannetpp/colmap")),
    )
    parser.add_argument("--scene_ids", default=None)
    parser.add_argument("--scene_ids_file", default=None)
    parser.add_argument("--mode", choices=["depth", "ply"], default="depth")
    parser.add_argument(
        "--depth_geometry",
        choices=["pointcloud", "tsdf", "both"],
        default="pointcloud",
        help="Geometry to build from depth sidecars before ScanNet++ mesh evaluation.",
    )
    parser.add_argument("--image_folder", default="images_4")
    parser.add_argument("--model_folder", default="0")
    parser.add_argument("--depth_folder", default="depths_4_3dgs")
    parser.add_argument("--mask_folder", default=None)
    parser.add_argument("--scale_json_root", default=None)
    parser.add_argument(
        "--scale_source",
        choices=["estimated", "oracle", "none"],
        default="estimated",
    )
    parser.add_argument(
        "--scale_application",
        choices=["depth", "world"],
        default="depth",
        help="Apply the scene scale to predicted depths only, or to fully unprojected COLMAP points.",
    )
    parser.add_argument("--npz_key", default=None)
    parser.add_argument("--predicted_png_depth_scale", type=float, default=1.0)
    parser.add_argument("--allow_rgb_depth_png", action="store_true")
    parser.add_argument("--min_depth", type=float, default=1e-6)
    parser.add_argument("--max_depth", type=float, default=0.0)
    parser.add_argument(
        "--compute_depth_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare predicted metric depth maps against ScanNet++ GT depth PNGs.",
    )
    parser.add_argument("--gt_depth_scale", type=float, default=1000.0)
    parser.add_argument(
        "--depth_metric_resize_mode",
        choices=["bilinear", "nearest"],
        default="bilinear",
    )
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--points_per_frame", type=int, default=4096)
    parser.add_argument("--max_source_points", type=int, default=300000)
    parser.add_argument("--tsdf_voxel_length", type=float, default=0.02)
    parser.add_argument("--tsdf_sdf_trunc", type=float, default=0.08)
    parser.add_argument("--tsdf_depth_trunc", type=float, default=10.0)
    parser.add_argument("--write_tsdf_meshes", action="store_true")
    parser.add_argument("--input_ply", default=None)
    parser.add_argument("--input_ply_root", default=None)
    parser.add_argument("--input_ply_name", default="point_cloud.ply")
    parser.add_argument("--input_scale", type=float, default=1.0)
    parser.add_argument("--add_face_centers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distance_tau", type=float, default=0.05)
    parser.add_argument(
        "--fscore_thresholds_m",
        type=parse_float_list,
        default=parse_float_list("0.02,0.05,0.1"),
    )
    parser.add_argument(
        "--initial_transform",
        choices=["auto", "come", "none"],
        default="auto",
        help="Use 'auto' to keep raw COLMAP depth points unchanged and apply CoMe's axis transform to PLY inputs.",
    )
    parser.add_argument("--skip_icp", action="store_true")
    parser.add_argument("--icp_iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--write_distance_artifacts", action="store_true")
    parser.add_argument("--write_visualizations", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 0:
        args.max_frames = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "depth":
        scene_dirs = resolve_scene_dirs(args)
        scene_stats = []
        for scene_dir in scene_dirs:
            try:
                print(f"Evaluating {scene_dir.name} depth {args.depth_geometry}")
                scene_stats.append(process_depth_scene(scene_dir, args))
            except Exception as exc:  # pylint: disable=broad-except
                print(f"FAILED {scene_dir.name}: {exc}")
                scene_stats.append({
                    "scene_id": scene_dir.name,
                    "status": "failed",
                    "error": repr(exc),
                })
    else:
        scene_ids = parse_scene_ids(args.scene_ids)
        if args.scene_ids_file:
            with Path(args.scene_ids_file).open("r", encoding="utf-8") as file:
                scene_ids.extend(line.strip() for line in file if line.strip())
        if not scene_ids:
            if not args.input_ply:
                raise ValueError("PLY mode requires --scene_ids, --scene_ids_file, or --input_ply")
            scene_ids = [Path(args.input_ply).parent.name]
        scene_stats = []
        for scene_id in sorted(set(scene_ids)):
            try:
                print(f"Evaluating {scene_id} PLY")
                scene_stats.append(process_ply_scene(scene_id, args))
            except Exception as exc:  # pylint: disable=broad-except
                print(f"FAILED {scene_id}: {exc}")
                scene_stats.append({
                    "scene_id": scene_id,
                    "status": "failed",
                    "error": repr(exc),
                })

    payload = {
        "parameters": {
            "mode": args.mode,
            "depth_geometry": args.depth_geometry,
            "scannetpp_root": str(Path(args.scannetpp_root).resolve()),
            "colmap_root": str(Path(args.colmap_root).resolve()),
            "image_folder": args.image_folder,
            "model_folder": args.model_folder,
            "depth_folder": args.depth_folder,
            "mask_folder": args.mask_folder,
            "scale_json_root": (
                str(Path(args.scale_json_root).resolve())
                if args.scale_json_root
                else None
            ),
            "scale_source": args.scale_source,
            "scale_application": args.scale_application,
            "compute_depth_metrics": args.compute_depth_metrics,
            "gt_depth_scale": args.gt_depth_scale,
            "depth_metric_resize_mode": args.depth_metric_resize_mode,
            "distance_tau": args.distance_tau,
            "fscore_thresholds_m": args.fscore_thresholds_m,
            "initial_transform": args.initial_transform,
            "skip_icp": args.skip_icp,
            "max_frames": args.max_frames,
            "points_per_frame": args.points_per_frame,
            "max_source_points": args.max_source_points,
            "tsdf_voxel_length": args.tsdf_voxel_length,
            "tsdf_sdf_trunc": args.tsdf_sdf_trunc,
            "tsdf_depth_trunc": args.tsdf_depth_trunc,
            "write_tsdf_meshes": args.write_tsdf_meshes,
            "seed": args.seed,
        },
        "aggregate": aggregate_results(scene_stats),
        "scene_stats": scene_stats,
    }
    write_json(output_dir / "metrics.json", payload)
    print(f"Metrics written to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
