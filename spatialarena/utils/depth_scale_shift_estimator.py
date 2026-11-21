# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import numpy as np
import cv2
from typing import Dict, Tuple, List
from joblib import Parallel, delayed

from spatialarena.geometry.rotation import qvec2rotmat


def collect_paired_measurements(
    key: str,
    cameras: Dict,
    images: Dict,
    points3d_ordered: np.ndarray,
    inv_depths: Dict,
    model_h: int,
    model_w: int,
) -> Dict:
    """
    Collect paired inverse depth measurements for a single image.
    
    Args:
        key: Image identifier
        cameras: COLMAP camera dictionary
        images: COLMAP image dictionary
        points3d_ordered: Ordered 3D points array
        inv_depths: Dictionary of monocular inverse depth maps
        model_h, model_w: Model input resolution
    
    Returns:
        Dictionary containing paired measurements or None if invalid
    """
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]
    
    # Get valid 3D points for this image
    pts_idx = image_meta.point3D_ids
    mask = (pts_idx >= 0) & (pts_idx < len(points3d_ordered))
    
    if not mask.any():
        return None
    
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]
    
    if len(pts_idx) == 0:
        return None
    
    # Get 3D points and transform to camera coordinates
    pts = points3d_ordered[pts_idx]
    R = qvec2rotmat(image_meta.qvec)
    tvec = image_meta.tvec
    pts_cam = np.dot(pts, R.T) + tvec
    
    # Compute inverse COLMAP depths (ground truth)
    inv_colmap_depth = 1.0 / (pts_cam[..., 2] + 1e-8)
    
    # Get monocular inverse depth map
    inv_mono_depth_map = inv_depths.get(image_meta.name)
    if inv_mono_depth_map is None:
        return None
    
    # Ensure monocular depth map is 2D
    if inv_mono_depth_map.ndim != 2:
        inv_mono_depth_map = inv_mono_depth_map[..., 0]
    
    # Scale factor from original to model resolution
    original_h, original_w = cam_intrinsic.height, cam_intrinsic.width
    scale_h = model_h / original_h
    scale_w = model_w / original_w
    
    # Scale 2D points to model resolution
    maps = (valid_xys * np.array([scale_w, scale_h])).astype(np.float32)
    
    # Create valid mask
    valid = (
        (maps[..., 0] >= 0) &
        (maps[..., 1] >= 0) &
        (maps[..., 0] < model_w) &
        (maps[..., 1] < model_h) &
        (inv_colmap_depth > 0)
    )
    
    # Check if we have enough valid points with sufficient variation
    if valid.sum() <= 10:
        return None
    
    inv_colmap_depth_range = inv_colmap_depth[valid].max() - inv_colmap_depth[valid].min()
    if inv_colmap_depth_range <= 1e-3:
        return None
    
    # Filter valid points
    maps = maps[valid, :]
    inv_colmap_depth = inv_colmap_depth[valid]
    
    # Sample monocular inverse depth at these locations
    inv_mono_depth = cv2.remap(
        inv_mono_depth_map.astype(np.float32),
        maps[..., 0].reshape(-1, 1),
        maps[..., 1].reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    ).flatten()
    
    # Return collected measurements
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
    """
    Collect paired measurements from all images using parallel processing.
    
    Args:
        cameras: COLMAP camera dictionary
        images: COLMAP image dictionary
        points3d: COLMAP 3D points dictionary
        inv_depths: Dictionary of monocular inverse depth maps
        model_resolution: Model input resolution (H, W)
        n_jobs: Number of parallel jobs (-1 for all cores)
    
    Returns:
        all_inv_colmap_depths: Concatenated inverse COLMAP depths
        all_inv_mono_depths: Concatenated inverse monocular depths
        valid_image_names: List of image names that contributed measurements
    """
    # Organize 3D points in ordered array for fast indexing
    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
    points3d_ordered[pts_indices] = pts_xyzs
    
    model_h, model_w = model_resolution
    
    # Collect measurements in parallel
    print(f"Collecting paired measurements from {len(images)} images...")
    measurement_list = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(collect_paired_measurements)(
            key, cameras, images, points3d_ordered, inv_depths, model_h, model_w
        ) for key in images
    )
    
    # Filter out None results and collect data
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
    
    # Concatenate all measurements
    inv_colmap_all = np.concatenate(all_inv_colmap)
    inv_mono_all = np.concatenate(all_inv_mono)
    
    print(f"Collected {len(inv_colmap_all)} paired measurements from {len(valid_image_names)} images")
    
    return inv_colmap_all, inv_mono_all, valid_image_names


def estimate_global_scale_shift(
    inv_colmap_depths: np.ndarray,
    inv_mono_depths: np.ndarray,
    outlier_rejection: bool = True,
    outlier_threshold: float = 2.0,
) -> Tuple[float, float, Dict]:
    """
    Estimate global scale and shift using least squares with optional outlier rejection.
    
    Args:
        inv_colmap_depths: Ground truth inverse depths from COLMAP
        inv_mono_depths: Predicted inverse depths from monocular model
        outlier_rejection: Whether to perform iterative outlier rejection
        outlier_threshold: Number of standard deviations for outlier detection
    
    Returns:
        scale: Global scale factor
        shift: Global shift
        stats: Dictionary with statistics
    """
    
    pred = inv_mono_depths.copy()
    gt = inv_colmap_depths.copy()
    
    if outlier_rejection:
        # Iterative outlier rejection
        max_iterations = 5
        for iteration in range(max_iterations):
            # Solve least squares: gt = scale * pred + shift
            _ones = np.ones_like(pred)
            A = np.column_stack([pred, _ones])
            X = np.linalg.lstsq(A, gt, rcond=None)[0]
            scale, shift = X
            
            # Compute residuals
            aligned_pred = pred * scale + shift
            residuals = np.abs(gt - aligned_pred)
            
            # Identify outliers
            residual_std = np.std(residuals)
            outlier_mask = residuals > (outlier_threshold * residual_std)
            
            num_outliers = np.sum(outlier_mask)
            if num_outliers == 0 or iteration == max_iterations - 1:
                break
            
            # Remove outliers
            print(f"  Iteration {iteration + 1}: Removing {num_outliers} outliers ({num_outliers/len(pred)*100:.1f}%)")
            pred = pred[~outlier_mask]
            gt = gt[~outlier_mask]
    
    # Final least squares solution
    _ones = np.ones_like(pred)
    A = np.column_stack([pred, _ones])
    X = np.linalg.lstsq(A, gt, rcond=None)[0]
    scale, shift = X
    
    # Compute statistics
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
    inv_depth_maps: Dict,  # Dictionary of inverse depth maps
    scale: float,
    shift: float,
) -> Dict:
    """
    Apply global scale and shift to align all monocular depth maps.
    
    Args:
        inv_depth_maps: Dictionary of inverse depth maps (prediction)
        scale: Global scale factor
        shift: Global shift
    
    Returns:
        aligned_inv_depth_maps: Dictionary of aligned inverse depth maps
    """
    aligned_maps = {}
    
    for img_name, inv_depth in inv_depth_maps.items():
        # Apply alignment: inv_depth_aligned = scale * inv_depth + shift
        inv_depth_aligned = scale * inv_depth + shift
        
        # Clip to reasonable range
        inv_depth_aligned = np.clip(inv_depth_aligned, 1e-6, 1e6)
        aligned_maps[img_name] = inv_depth_aligned
    
    return aligned_maps


def depth_to_inv_depth(
    depth_maps: Dict,  # Dictionary of depth maps (prediction)
) -> Dict:
    """Convert depth maps to inverse depth maps."""
    inv_depth_maps = {}
    
    for img_name, depth in depth_maps.items():
        # Convert to inverse depth
        inv_depth = 1.0 / (depth + 1e-8)
        inv_depth_maps[img_name] = inv_depth
    
    return inv_depth_maps


def inv_depth_to_depth(
    inv_depth_maps: Dict,  # Dictionary of inverse depth maps
) -> Dict:
    """Convert inverse depth maps to depth maps."""
    depth_maps = {}
    
    for img_name, inv_depth in inv_depth_maps.items():
        # Convert to depth
        depth = 1.0 / (inv_depth + 1e-8)
        depth_maps[img_name] = depth
    
    return depth_maps


def estimate_and_apply_global_alignment(
    cameras: Dict,
    images: Dict,
    points3d: Dict,
    depth_predictions: Dict,  # Dictionary of depth maps from DepthAnything3
    model_resolution: Tuple[int, int],
    outlier_rejection: bool = True,
    n_jobs: int = -1,
) -> Tuple[Dict, Dict]:
    """
    Complete pipeline to estimate global scale/shift and align all depth maps.
    
    Args:
        cameras: COLMAP camera dictionary
        images: COLMAP image dictionary
        points3d: COLMAP 3D points dictionary
        depth_predictions: Dictionary of depth maps from DepthAnything3
        model_resolution: Model input resolution (H, W)
        outlier_rejection: Whether to perform outlier rejection
        n_jobs: Number of parallel jobs
    
    Returns:
        aligned_depth_maps: Dictionary of aligned metric depth maps
        alignment_info: Dictionary with alignment parameters and statistics
    """
    
    # Step 1: Convert depth to inverse depth
    print("Converting depth maps to inverse depth...")
    inv_depth_predictions = depth_to_inv_depth(depth_predictions)
    
    # Step 2: Collect all paired measurements
    print("\nCollecting paired measurements across all images...")
    inv_colmap_all, inv_mono_all, valid_images = collect_all_paired_measurements(
        cameras=cameras,
        images=images,
        points3d=points3d,
        inv_depths=inv_depth_predictions,
        model_resolution=model_resolution,
        n_jobs=n_jobs,
    )
    
    # Step 3: Estimate global scale and shift
    print("\nEstimating global scale and shift...")
    scale, shift, stats = estimate_global_scale_shift(
        inv_colmap_all,
        inv_mono_all,
        outlier_rejection=outlier_rejection,
    )
    
    print(f"\n=== Global Alignment Results ===")
    print(f"Scale: {scale:.6f}")
    print(f"Shift: {shift:.6f}")
    print(f"RMSE: {stats['rmse']:.6f}")
    print(f"MAE: {stats['mae']:.6f}")
    print(f"R²: {stats['r2']:.6f}")
    print(f"Points used: {stats['num_points']}/{stats['original_num_points']}")
    
    # Step 4: Apply global alignment to all inverse depth maps
    print("\nApplying global alignment to all depth maps...")
    aligned_inv_depth = align_depth_with_global_params(
        inv_depth_predictions,
        scale,
        shift,
    )
    
    # Step 5: Convert back to depth
    aligned_depth_maps = inv_depth_to_depth(aligned_inv_depth)
    
    # Step 6: Prepare alignment info
    alignment_info = {
        'scale': scale,
        'shift': shift,
        'stats': stats,
        'valid_images': valid_images,
    }
    
    return aligned_depth_maps, alignment_info


def estimate_per_frame_scale_shift(
    image_id,
    cameras,
    images,
    points3d_ordered,
    inv_depths: Dict
):
    image_meta = images[image_id]
    cam_intrinsic = cameras[image_meta.camera_id]

    # pts_idx = images[key].point3D_ids
    pts_idx = image_meta.point3D_ids # image_meta[key].point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    inv_colmap_depth_map = 1. / (pts[..., 2] + 1e-8)
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    inv_mono_depth_map = inv_depths.get(image_meta.name, None)

    if inv_mono_depth_map is None:
        return None
    else:
        print(f'found image: {image_meta.name}')

    if inv_mono_depth_map.ndim != 2:
        inv_mono_depth_map = inv_mono_depth_map[..., 0]

    s = inv_mono_depth_map.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) *
        (maps[..., 1] >= 0) *
        (maps[..., 0] < cam_intrinsic.width * s) *
        (maps[..., 1] < cam_intrinsic.height * s) * (inv_colmap_depth_map > 0))

    if valid.sum() > 10 and (inv_colmap_depth_map.max() - inv_colmap_depth_map.min()) > 1e-3:
        maps = maps[valid, :]
        inv_colmap_depth_map = inv_colmap_depth_map[valid]
        inv_mono_depth = cv2.remap(
            inv_mono_depth_map,
            maps[..., 0],
            maps[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )[..., 0]
        scale, offset, _ = estimate_global_scale_shift(inv_colmap_depth_map, inv_mono_depth)
    else:
        scale = 1.0
        offset = 0

    return {
        "image_name": image_meta.name,
        "scale": scale,
        "offset": offset
    }


def estimate_and_apply_local_alignment(
    cameras: Dict,
    images: Dict,
    points3d: Dict,
    depth_predictions: Dict,  # Dictionary of depth maps from DepthAnything3
    model_resolution: Tuple[int, int],
    outlier_rejection: bool = True,
    n_jobs: int = -1,
) -> Tuple[Dict, Dict]:
    # Step 1: Convert depth to inverse depth
    print("Converting depth maps to inverse depth...")
    inv_depth_predictions = depth_to_inv_depth(depth_predictions)

    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    # Step 2: Estimate local scale and shift
    print(f"Estimating scale and shift from {len(images)} images...")
    depth_param_list = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(estimate_per_frame_scale_shift)(
            key, cameras, images, points3d_ordered, inv_depth_predictions
        ) for key in images
    )

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

    # Step 4: Convert back to depth
    aligned_depth_maps = inv_depth_to_depth(aligned_inv_depths)
    
    # Step 5: Prepare alignment info
    alignment_info = {
        'scale': scale,
        'shift': shift,
    }
    
    return aligned_depth_maps, alignment_info
