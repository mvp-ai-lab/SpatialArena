# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Conversion of predicted cameras and points to COLMAP models.

`write_colmap_formats` writes a COLMAP sparse model (cameras, images and
points) from per-image extrinsics and intrinsics and a point cloud. The
module also has Open3D voxel downsampling and a random cap on mask pixels.
"""

from typing import Dict, List, Sequence, Union

import numpy as np


def _open3d():
    """Import Open3D only when point-cloud downsampling is requested."""
    import open3d as o3d

    return o3d


from spatialarena.geometry.read_write_model import (
    Camera,
    BaseImage,
    Point3D,
    rotmat2qvec,
    write_model,
)


def to_colmap_camera(id: int,
                     width: int,
                     height: int,
                     params: List[float],
                     model: str = "PINHOLE"):
    """Returns a COLMAP camera with the given intrinsic parameters."""
    return Camera(id=id, model=model, width=width, height=height, params=params)


def to_colmap_cameras(
    intrinsics: np.ndarray,
    width: Union[int, Sequence[int]],
    height: Union[int, Sequence[int]],
    shared_intrinsics: bool = False,
) -> Dict:
    """Returns one PINHOLE COLMAP camera per image, or one shared camera.

    Args:
        intrinsics: (N, 3, 3) intrinsic matrices.
        width: One width for every image, or one per image for scenes whose
            frames differ in resolution (e.g. ETH3D after undistortion).
        height: One height for every image, or one per image.
        shared_intrinsics: Whether all images share the first camera.

    Returns:
        The cameras by camera id, starting at 1.

    Raises:
        ValueError: If the number of widths or heights is not 1 or N.
    """
    cameras = {}
    N = intrinsics.shape[0]
    num_cameras = 1 if shared_intrinsics else N
    widths = [int(width)] * N if np.isscalar(width) else [int(w) for w in width]
    heights = [int(height)] * N if np.isscalar(height) else [
        int(h) for h in height
    ]
    if len(widths) != N or len(heights) != N:
        raise ValueError(f"expected one width/height or {N} of each, got "
                         f"{len(widths)} and {len(heights)}")
    for i in range(num_cameras):
        params = [
            intrinsics[i, 0, 0], intrinsics[i, 1, 1], intrinsics[i, 0, 2],
            intrinsics[i, 1, 2]
        ]
        camera = to_colmap_camera(
            i + 1,
            width=widths[i],
            height=heights[i],
            params=params,
        )
        cameras[camera.id] = camera

    return cameras


def to_colmap_images(
    scene_name: str,
    extrinsics: np.ndarray,
    image_paths: List[str],
    shared_intrinsics: bool = False,
) -> Dict:
    """Returns the COLMAP images of camera-from-world extrinsics.

    Image names are the image paths after the scene name.
    """
    images = {}
    N = extrinsics.shape[0]
    for i in range(N):
        camera_id = 1 if shared_intrinsics else i + 1
        image_path = image_paths[i]
        rotation = extrinsics[i, :3, :3]
        qvec = rotmat2qvec(rotation).astype(np.float32)
        tvec = extrinsics[i, :3, 3].astype(np.float32)
        image_name_start_index = image_path.find(scene_name) + len(scene_name)
        image_name = image_path[image_name_start_index + 1:]
        image = BaseImage(id=i + 1,
                          qvec=qvec,
                          tvec=tvec,
                          camera_id=camera_id,
                          name=image_name,
                          xys=[],
                          point3D_ids=[])
        images[image.id] = image

    return images


def to_colmap_points(points: np.ndarray, rgbs: np.ndarray) -> Dict:
    """Returns COLMAP points (without tracks) of positions and colors."""
    points3D = {}
    N = points.shape[0]
    for i in range(N):
        point = Point3D(
            id=i,
            xyz=points[i],
            rgb=rgbs[i],
            error=0,
            image_ids=np.array([]),
            point2D_idxs=np.array([]),
        )
        points3D[i] = point

    return points3D


def write_colmap_formats(
    seq_name: str,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_paths: List[str],
    points: np.ndarray,
    colors: np.ndarray,
    width: Union[int, Sequence[int]],
    height: Union[int, Sequence[int]],
    output_path: str,
    ext: str = '.bin',
    shared_intrinsics: bool = False,
):
    """Writes a COLMAP sparse model of predicted cameras and points.

    Args:
        seq_name: The scene name; image names are the paths after it.
        extrinsics: (N, 3 or 4, 4) camera-from-world extrinsics.
        intrinsics: (N, 3, 3) intrinsic matrices.
        image_paths: The N image paths.
        points: (M, 3) point positions.
        colors: (M, 3) point colors.
        width: One image width, or one per image.
        height: One image height, or one per image.
        output_path: The model directory.
        ext: The model format, '.bin' or '.txt'.
        shared_intrinsics: Whether all images share one camera.
    """
    colmap_cameras = to_colmap_cameras(
        intrinsics,
        width=width,
        height=height,
        shared_intrinsics=shared_intrinsics,
    )
    colmap_images = to_colmap_images(
        seq_name,
        extrinsics=extrinsics,
        image_paths=image_paths,
        shared_intrinsics=shared_intrinsics,
    )
    colmap_points3D = to_colmap_points(points, colors)
    write_model(colmap_cameras,
                colmap_images,
                colmap_points3D,
                output_path,
                ext=ext)


def downsample_points_and_cols(points, colors, voxel_size=0.05):
    """Downsamples a point cloud by Open3D voxel downsampling.

    Args:
        points: (N, 3) array of point locations.
        colors: (N, 3) array of point colors (RGB, 0-1 range).
        voxel_size: The voxel size.

    Returns:
        (downsampled_points, downsampled_colors) as numpy arrays.
    """
    o3d = _open3d()
    # Create Open3D point cloud
    if o3d is None:
        raise ImportError("open3d is required for point cloud downsampling")
    if o3d is None:
        raise ImportError("open3d is required for point cloud downsampling")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Original points: {len(points)}")

    # Perform voxel downsampling
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    print(f"Downsampled points: {len(downpcd.points)}")

    # Convert back to numpy arrays
    downsampled_points = np.asarray(downpcd.points)
    downsampled_colors = np.asarray(downpcd.colors)

    return downsampled_points, downsampled_colors


# Alternative version with automatic voxel size calculation
def downsample_auto_voxel(points, colors, target_reduction=0.1):
    """Downsamples with a voxel size derived from a target reduction.

    Args:
        points: (N, 3) array of point locations.
        colors: (N, 3) array of point colors.
        target_reduction: Target reduction ratio (0-1).

    Returns:
        (downsampled_points, downsampled_colors) as numpy arrays.
    """
    o3d = _open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Calculate automatic voxel size based on bounding box and target reduction
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    avg_extent = np.mean(extent)

    # Empirical formula for voxel size based on target reduction
    voxel_size = avg_extent * (target_reduction**(1 / 3))

    print(f"Auto-calculated voxel size: {voxel_size:.4f}")

    return downsample_points_and_cols(points, colors, voxel_size)


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """Randomly keeps at most max_trues True values of a mask.

    Args:
        mask: A bool array.
        max_trues: The largest number of True values to keep.

    Returns:
        The mask itself if it is within the budget, otherwise a new mask with
        max_trues randomly chosen True values.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    # randomly pick which True positions to keep
    sampled_indices = np.random.choice(true_indices,
                                       size=max_trues,
                                       replace=False)  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)
