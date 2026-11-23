# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Aligns monocular depth predictions to a COLMAP sparse reconstruction.

The fit is done in inverse depth: COLMAP points projected into each image
give inverse depths, the monocular inverse depth maps are sampled at the same
pixels, and `inv_colmap = scale * inv_mono + shift` is solved by least
squares. `estimate_and_apply_global_alignment` fits one scale and shift for
all images; `estimate_and_apply_local_alignment` fits one per image.
"""

from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np

from spatialarena.geometry import rotation


def collect_paired_measurements(
    key: str,
    cameras: Dict,
    images: Dict,
    points3d_ordered: np.ndarray,
    inv_depths: Dict,
    model_h: int,
    model_w: int,
) -> Dict:
    """Collects paired inverse depth measurements of a single image.

    Args:
        key: Image identifier.
        cameras: COLMAP camera dictionary.
        images: COLMAP image dictionary.
        points3d_ordered: 3D points array indexed by point id.
        inv_depths: Dictionary of monocular inverse depth maps by image name.
        model_h: Model input height.
        model_w: Model input width.

    Returns:
        The image name and the paired COLMAP and monocular inverse depths, or
        None if the image has too few valid points or too little depth
        variation.
    """
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    # The valid 3D points of this image.
    pts_idx = image_meta.point3D_ids
    mask = (pts_idx >= 0) & (pts_idx < len(points3d_ordered))

    if not mask.any():
        return None

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) == 0:
        return None

    # Transform the 3D points to camera coordinates.
    pts = points3d_ordered[pts_idx]
    R = rotation.qvec2rotmat(image_meta.qvec)
    tvec = image_meta.tvec
    pts_cam = np.dot(pts, R.T) + tvec

    # Inverse COLMAP depths (the ground truth).
    inv_colmap_depth = 1.0 / (pts_cam[..., 2] + 1e-8)

    # The monocular inverse depth map.
    inv_mono_depth_map = inv_depths.get(image_meta.name)
    if inv_mono_depth_map is None:
        return None

    # Ensure the monocular depth map is 2D.
    if inv_mono_depth_map.ndim != 2:
        inv_mono_depth_map = inv_mono_depth_map[..., 0]

    # Scale factor from the original to the model resolution.
    original_h, original_w = cam_intrinsic.height, cam_intrinsic.width
    scale_h = model_h / original_h
    scale_w = model_w / original_w

    # Scale the 2D points to the model resolution.
    maps = (valid_xys * np.array([scale_w, scale_h])).astype(np.float32)

    # Valid mask.
    valid = ((maps[..., 0] >= 0) & (maps[..., 1] >= 0) &
             (maps[..., 0] < model_w) & (maps[..., 1] < model_h) &
             (inv_colmap_depth > 0))

    # Require enough valid points with sufficient variation.
    if valid.sum() <= 10:
        return None

    inv_colmap_depth_range = (inv_colmap_depth[valid].max() -
                              inv_colmap_depth[valid].min())
    if inv_colmap_depth_range <= 1e-3:
        return None

    # Keep the valid points.
    maps = maps[valid, :]
    inv_colmap_depth = inv_colmap_depth[valid]

    # Sample the monocular inverse depth at these locations.
    inv_mono_depth = cv2.remap(inv_mono_depth_map.astype(np.float32),
                               maps[..., 0].reshape(-1, 1),
                               maps[..., 1].reshape(-1, 1),
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE).flatten()

    return {
        "image_name": image_meta.name,
        "inv_colmap_depth": inv_colmap_depth,
        "inv_mono_depth": inv_mono_depth,
    }


def collect_all_paired_measurements(
    cameras: Dict,
    images: Dict,
    points3d: Dict,
    inv_depths: Dict,
    model_resolution: Tuple[int, int],
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Collects paired measurements of all images in parallel threads.

    Args:
        cameras: COLMAP camera dictionary.
        images: COLMAP image dictionary.
        points3d: COLMAP 3D points dictionary.
        inv_depths: Dictionary of monocular inverse depth maps.
        model_resolution: Model input resolution (H, W).
        n_jobs: Number of parallel jobs (-1 for all cores).

    Returns:
        The concatenated inverse COLMAP depths, the concatenated inverse
        monocular depths and the names of the images that contributed.

    Raises:
        ValueError: If no image gave valid measurements.
    """
    # 3D points in an array indexed by point id, for fast indexing.
    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    model_h, model_w = model_resolution

    # Collect the measurements in parallel.
    print(f"Collecting paired measurements from {len(images)} images...")
    measurement_list = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
        joblib.delayed(collect_paired_measurements)
        (key, cameras, images, points3d_ordered, inv_depths, model_h, model_w)
        for key in images)

    # Drop the images without measurements.
    all_inv_colmap = []
    all_inv_mono = []
    valid_image_names = []

    for measurement in measurement_list:
        if measurement is not None:
            all_inv_colmap.append(measurement["inv_colmap_depth"])
            all_inv_mono.append(measurement["inv_mono_depth"])
            valid_image_names.append(measurement["image_name"])

    if len(all_inv_colmap) == 0:
        raise ValueError("No valid measurements collected from any image")

    inv_colmap_all = np.concatenate(all_inv_colmap)
    inv_mono_all = np.concatenate(all_inv_mono)

    print(f"Collected {len(inv_colmap_all)} paired measurements from "
          f"{len(valid_image_names)} images")

    return inv_colmap_all, inv_mono_all, valid_image_names


def estimate_global_scale_shift(
    inv_colmap_depths: np.ndarray,
    inv_mono_depths: np.ndarray,
    outlier_rejection: bool = True,
    outlier_threshold: float = 2.0,
) -> Tuple[float, float, Dict]:
    """Estimates `gt = scale * pred + shift` by least squares.

    With outlier rejection, up to 5 fits run, each dropping the residuals
    above outlier_threshold standard deviations before the next.

    Args:
        inv_colmap_depths: Ground-truth inverse depths from COLMAP.
        inv_mono_depths: Predicted inverse depths from the monocular model.
        outlier_rejection: Whether to perform iterative outlier rejection.
        outlier_threshold: Number of standard deviations of an outlier.

    Returns:
        The scale, the shift and the fit statistics (point counts, rmse,
        mae, r2, scale and shift).
    """
    pred = inv_mono_depths.copy()
    gt = inv_colmap_depths.copy()

    if outlier_rejection:
        # Iterative outlier rejection.
        max_iterations = 5
        for iteration in range(max_iterations):
            # Solve least squares: gt = scale * pred + shift.
            _ones = np.ones_like(pred)
            A = np.column_stack([pred, _ones])
            X = np.linalg.lstsq(A, gt, rcond=None)[0]
            scale, shift = X

            # Residuals.
            aligned_pred = pred * scale + shift
            residuals = np.abs(gt - aligned_pred)

            # Outliers.
            residual_std = np.std(residuals)
            outlier_mask = residuals > (outlier_threshold * residual_std)

            num_outliers = np.sum(outlier_mask)
            if num_outliers == 0 or iteration == max_iterations - 1:
                break

            # Remove the outliers.
            print(f"  Iteration {iteration + 1}: Removing {num_outliers} "
                  f"outliers ({num_outliers/len(pred)*100:.1f}%)")
            pred = pred[~outlier_mask]
            gt = gt[~outlier_mask]

    # Final least squares solution.
    _ones = np.ones_like(pred)
    A = np.column_stack([pred, _ones])
    X = np.linalg.lstsq(A, gt, rcond=None)[0]
    scale, shift = X

    # Statistics.
    aligned_pred = pred * scale + shift
    residuals = gt - aligned_pred

    stats = {
        'num_points': len(pred),
        'original_num_points': len(inv_mono_depths),
        'rmse': float(np.sqrt(np.mean(residuals**2))),
        'mae': float(np.mean(np.abs(residuals))),
        'r2': float(1 - np.sum(residuals**2) / np.sum((gt - np.mean(gt))**2)),
        'scale': float(scale),
        'shift': float(shift),
    }

    return scale, shift, stats


def align_depth_with_global_params(
    inv_depth_maps: Dict,
    scale: float,
    shift: float,
) -> Dict:
    """Applies a global scale and shift to inverse depth maps.

    Args:
        inv_depth_maps: Dictionary of predicted inverse depth maps.
        scale: Global scale factor.
        shift: Global shift.

    Returns:
        The aligned inverse depth maps, clipped to [1e-6, 1e6].
    """
    aligned_maps = {}

    for img_name, inv_depth in inv_depth_maps.items():
        inv_depth_aligned = scale * inv_depth + shift

        # Clip to a reasonable range.
        inv_depth_aligned = np.clip(inv_depth_aligned, 1e-6, 1e6)
        aligned_maps[img_name] = inv_depth_aligned

    return aligned_maps


def depth_to_inv_depth(depth_maps: Dict) -> Dict:
    """Converts a dictionary of predicted depth maps to inverse depth."""
    inv_depth_maps = {}

    for img_name, depth in depth_maps.items():
        inv_depth = 1.0 / (depth + 1e-8)
        inv_depth_maps[img_name] = inv_depth

    return inv_depth_maps


def inv_depth_to_depth(inv_depth_maps: Dict) -> Dict:
    """Converts a dictionary of inverse depth maps to depth."""
    depth_maps = {}

    for img_name, inv_depth in inv_depth_maps.items():
        depth = 1.0 / (inv_depth + 1e-8)
        depth_maps[img_name] = depth

    return depth_maps


def estimate_and_apply_global_alignment(
    cameras: Dict,
    images: Dict,
    points3d: Dict,
    depth_predictions: Dict,
    model_resolution: Tuple[int, int],
    outlier_rejection: bool = True,
    n_jobs: int = -1,
) -> Tuple[Dict, Dict]:
    """Estimates one global scale and shift and aligns all depth maps.

    Args:
        cameras: COLMAP camera dictionary.
        images: COLMAP image dictionary.
        points3d: COLMAP 3D points dictionary.
        depth_predictions: Dictionary of depth maps from DepthAnything3.
        model_resolution: Model input resolution (H, W).
        outlier_rejection: Whether to perform outlier rejection.
        n_jobs: Number of parallel jobs.

    Returns:
        The aligned metric depth maps and the alignment parameters and
        statistics.
    """
    # Step 1: convert depth to inverse depth.
    print("Converting depth maps to inverse depth...")
    inv_depth_predictions = depth_to_inv_depth(depth_predictions)

    # Step 2: collect all paired measurements.
    print("\nCollecting paired measurements across all images...")
    (inv_colmap_all, inv_mono_all,
     valid_images) = collect_all_paired_measurements(
         cameras=cameras,
         images=images,
         points3d=points3d,
         inv_depths=inv_depth_predictions,
         model_resolution=model_resolution,
         n_jobs=n_jobs,
     )

    # Step 3: estimate the global scale and shift.
    print("\nEstimating global scale and shift...")
    scale, shift, stats = estimate_global_scale_shift(
        inv_colmap_all,
        inv_mono_all,
        outlier_rejection=outlier_rejection,
    )

    print("\n=== Global Alignment Results ===")
    print(f"Scale: {scale:.6f}")
    print(f"Shift: {shift:.6f}")
    print(f"RMSE: {stats['rmse']:.6f}")
    print(f"MAE: {stats['mae']:.6f}")
    print(f"R²: {stats['r2']:.6f}")
    print(f"Points used: {stats['num_points']}/{stats['original_num_points']}")

    # Step 4: apply the global alignment to all inverse depth maps.
    print("\nApplying global alignment to all depth maps...")
    aligned_inv_depth = align_depth_with_global_params(
        inv_depth_predictions,
        scale,
        shift,
    )

    # Step 5: convert back to depth.
    aligned_depth_maps = inv_depth_to_depth(aligned_inv_depth)

    # Step 6: alignment info.
    alignment_info = {
        'scale': scale,
        'shift': shift,
        'stats': stats,
        'valid_images': valid_images,
    }

    return aligned_depth_maps, alignment_info


def estimate_per_frame_scale_shift(image_id, cameras, images, points3d_ordered,
                                   inv_depths: Dict):
    """Estimates the scale and offset of one image's inverse depth.

    Images with too few valid points or too little depth variation get scale
    1 and offset 0.

    Args:
        image_id: The image id.
        cameras: COLMAP camera dictionary.
        images: COLMAP image dictionary.
        points3d_ordered: 3D points array indexed by point id.
        inv_depths: Dictionary of monocular inverse depth maps by image name.

    Returns:
        The image name, scale and offset, or None if the image has no
        inverse depth map.
    """
    image_meta = images[image_id]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = image_meta.point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = rotation.qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    inv_colmap_depth_map = 1. / (pts[..., 2] + 1e-8)
    inv_mono_depth_map = inv_depths.get(image_meta.name, None)

    if inv_mono_depth_map is None:
        return None
    print(f'found image: {image_meta.name}')

    if inv_mono_depth_map.ndim != 2:
        inv_mono_depth_map = inv_mono_depth_map[..., 0]

    s = inv_mono_depth_map.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = ((maps[..., 0] >= 0) * (maps[..., 1] >= 0) *
             (maps[..., 0] < cam_intrinsic.width * s) *
             (maps[..., 1] < cam_intrinsic.height * s) *
             (inv_colmap_depth_map > 0))

    if valid.sum() > 10 and (inv_colmap_depth_map.max() -
                             inv_colmap_depth_map.min()) > 1e-3:
        maps = maps[valid, :]
        inv_colmap_depth_map = inv_colmap_depth_map[valid]
        inv_mono_depth = cv2.remap(inv_mono_depth_map,
                                   maps[..., 0],
                                   maps[..., 1],
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)[..., 0]
        scale, offset, _ = estimate_global_scale_shift(inv_colmap_depth_map,
                                                       inv_mono_depth)
    else:
        scale = 1.0
        offset = 0

    return {"image_name": image_meta.name, "scale": scale, "offset": offset}


def estimate_and_apply_local_alignment(
    cameras: Dict,
    images: Dict,
    points3d: Dict,
    depth_predictions: Dict,
    model_resolution: Tuple[int, int],
    outlier_rejection: bool = True,
    n_jobs: int = -1,
) -> Tuple[Dict, Dict]:
    """Estimates a scale and shift per image and aligns its depth map.

    model_resolution and outlier_rejection are accepted for symmetry with
    `estimate_and_apply_global_alignment` and are not used.

    Args:
        cameras: COLMAP camera dictionary.
        images: COLMAP image dictionary.
        points3d: COLMAP 3D points dictionary.
        depth_predictions: Dictionary of depth maps from DepthAnything3.
        model_resolution: Model input resolution (H, W); unused.
        outlier_rejection: Unused.
        n_jobs: Number of parallel jobs.

    Returns:
        The aligned depth maps of the images that have a depth map, and the
        scale and shift of the last of them.
    """
    # Step 1: convert depth to inverse depth.
    print("Converting depth maps to inverse depth...")
    inv_depth_predictions = depth_to_inv_depth(depth_predictions)

    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    # Step 2: estimate the per-image scale and shift.
    print(f"Estimating scale and shift from {len(images)} images...")
    depth_param_list = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
        joblib.delayed(estimate_per_frame_scale_shift)(
            key, cameras, images, points3d_ordered, inv_depth_predictions)
        for key in images)

    aligned_inv_depths = {}
    for depth_param in depth_param_list:
        if depth_param is None:
            continue

        image_name = depth_param["image_name"]
        scale = depth_param["scale"]
        shift = depth_param["offset"]
        print(f'scale: {scale}, shift: {shift}')
        inv_depth_aligned = scale * inv_depth_predictions[image_name] + shift
        inv_depth_aligned = np.clip(inv_depth_aligned, 1e-6, 1e6)
        aligned_inv_depths[image_name] = inv_depth_aligned

    # Step 3: convert back to depth.
    aligned_depth_maps = inv_depth_to_depth(aligned_inv_depths)

    # Step 4: alignment info.
    alignment_info = {
        'scale': scale,
        'shift': shift,
    }

    return aligned_depth_maps, alignment_info
