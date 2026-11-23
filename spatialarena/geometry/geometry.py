# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Depth unprojection and camera projection helpers (from VGGT).

Numpy helpers unproject depth maps to camera and world points; torch helpers
project world points into cameras. Extrinsics are OpenCV camera-from-world
(3, 4) or (4, 4) matrices.
"""

import numpy as np
import torch


def unproject_depth_map_to_point_map(depth_map: np.ndarray,
                                     extrinsics_cam: np.ndarray,
                                     intrinsics_cam: np.ndarray) -> np.ndarray:
    """Unprojects a batch of depth maps to 3D world coordinates.

    Args:
        depth_map: Depth maps of shape (S, H, W, 1) or (S, H, W).
        extrinsics_cam: Camera-from-world extrinsics of shape (S, 3, 4).
        intrinsics_cam: Intrinsic matrices of shape (S, 3, 3).

    Returns:
        The world coordinates, of shape (S, H, W, 3).
    """
    if len(depth_map.shape) == 4:
        depth_map = depth_map.squeeze(-1)

    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx], extrinsics_cam[frame_idx],
            intrinsics_cam[frame_idx])
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts a depth map to world coordinates.

    Args:
        depth_map: Depth map of shape (H, W).
        extrinsic: Camera extrinsic matrix of shape (3, 4), OpenCV camera
            convention, camera from world.
        intrinsic: Camera intrinsic matrix of shape (3, 3).
        eps: Depths at or below this value are invalid.

    Returns:
        The world coordinates (H, W, 3), the camera coordinates (H, W, 3) and
        the valid depth mask (H, W); three Nones if depth_map is None.
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask.
    point_mask = depth_map > eps

    # Convert the depth map to camera coordinates.
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of the extrinsic matrix to transform to world
    # coordinates. closed_form_inverse_se3 is batched: its output is (N, 4, 4).
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates.
    world_coords_points = np.dot(
        cam_coords_points,
        R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(
        depth_map: np.ndarray,
        intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts a depth map to camera coordinates.

    Args:
        depth_map: Depth map of shape (H, W).
        intrinsic: Camera intrinsic matrix of shape (3, 3), without skew.

    Returns:
        The float32 camera coordinates, of shape (H, W, 3).
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[
        1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters.
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Grid of pixel coordinates.
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates.
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form the camera coordinates.
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """Computes the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must be the rotation and translation
    components of `se3`; otherwise they are extracted from `se3`.

    Args:
        se3: (N, 4, 4) or (N, 3, 4) array or tensor of SE3 matrices.
        R: Optional (N, 3, 3) array or tensor of rotation matrices.
        T: Optional (N, 3, 1) array or tensor of translation vectors.

    Returns:
        The (N, 4, 4) inverted matrices, of the same type and device as `se3`.

    Raises:
        ValueError: If se3 is not of shape (N, 4, 4) or (N, 3, 4).
    """
    # Whether se3 is a numpy array or a torch tensor.
    is_numpy = isinstance(se3, np.ndarray)

    # Validate the shapes.
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided.
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # The inverse is [R^T, -R^T t].
    if is_numpy:
        R_transposed = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def project_world_points_to_camera_points_batch(world_points, cam_extrinsics):
    """Transforms world points to camera points with per-frame extrinsics.

    Args:
        world_points: torch.Tensor of 3D points, of shape (B, S, H, W, 3).
        cam_extrinsics: torch.Tensor of extrinsics, of shape (B, S, 3, 4).

    Returns:
        The camera points, of shape (B, S, H, W, 3).
    """
    # TODO: merge this into project_world_points_to_cam.
    ones = torch.ones_like(world_points[..., :1])  # shape: (B, S, H, W, 1)
    # shape: (B, S, H, W, 4)
    world_points_h = torch.cat([world_points, ones], dim=-1)

    # extrinsics: (B, S, 3, 4) -> (B, S, 1, 1, 3, 4)
    extrinsics_exp = cam_extrinsics.unsqueeze(2).unsqueeze(3)

    # world_points_h: (B, S, H, W, 4) -> (B, S, H, W, 4, 1)
    world_points_h_exp = world_points_h.unsqueeze(-1)

    # (B, S, 1, 1, 3, 4) @ (B, S, H, W, 4, 1) broadcasts to (B, S, H, W, 3, 1).
    camera_points = torch.matmul(extrinsics_exp, world_points_h_exp).squeeze(-1)

    return camera_points


def project_world_points_to_cam(
    world_points,
    cam_extrinsics,
    cam_intrinsics=None,
    distortion_params=None,
    default=0,
    only_points_cam=False,
):
    """Transforms 3D points to 2D using extrinsic and intrinsic parameters.

    Args:
        world_points: torch.Tensor of 3D points, of shape (P, 3).
        cam_extrinsics: torch.Tensor of extrinsics, of shape (B, 3, 4).
        cam_intrinsics: torch.Tensor of intrinsics, of shape (B, 3, 3).
        distortion_params: Radial distortion parameters of shape (B, N);
            currently ignored.
        default: The value that replaces NaN pixel coordinates.
        only_points_cam: Whether to return only the camera points.

    Returns:
        (image_points, cam_points): the (B, P, 2) pixel coordinates (None if
        only_points_cam) and the (B, 3, P) camera points.
    """
    device = world_points.device
    with torch.autocast(device_type=device.type, enabled=False):
        N = world_points.shape[0]  # Number of points
        B = cam_extrinsics.shape[0]  # Batch size, i.e., number of cameras
        world_points_homogeneous = torch.cat(
            [world_points,
             torch.ones_like(world_points[..., 0:1])], dim=1)  # Nx4
        # Reshape for batch processing
        world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
            B, -1, -1)  # BxNx4

        # Step 1: transform the points to the coordinate system of every
        # camera.
        cam_points = torch.bmm(cam_extrinsics,
                               world_points_homogeneous.transpose(-1, -2))

        if only_points_cam:
            return None, cam_points

        # Step 2: apply the intrinsic parameters and the (optional)
        # distortion.
        image_points = img_from_cam(cam_intrinsics,
                                    cam_points,
                                    distortion_params,
                                    default=default)

        return image_points, cam_points


def img_from_cam(cam_intrinsics,
                 cam_points,
                 distortion_params=None,
                 default=0.0):
    """Applies intrinsic parameters and optional distortion to 3D points.

    Args:
        cam_intrinsics: torch.Tensor of intrinsics, of shape (B, 3, 3).
        cam_points: torch.Tensor of camera points, of shape (B, 3, N).
        distortion_params: Optional distortion parameters of shape (B, N),
            where N can be 1, 2, or 4; currently ignored.
        default: The value that replaces NaNs in the output.

    Returns:
        The pixel coordinates, of shape (B, N, 2).
    """
    # Normalized device coordinates (NDC).
    cam_points = cam_points / cam_points[:, 2:3, :]
    ndc_xy = cam_points[:, :2, :]

    # Distortion is not ported (VGGT's apply_distortion), so distortion_params
    # leave the points unchanged.
    distorted_xy = ndc_xy

    # Homogeneous coordinates for the batch matrix multiplication.
    cam_coords_homo = torch.cat(
        (distorted_xy, torch.ones_like(distorted_xy[:, :1, :])), dim=1)  # Bx3xN
    # Apply the intrinsic parameters.
    pixel_coords = torch.bmm(cam_intrinsics, cam_coords_homo)  # Bx3xN

    # Extract the x and y coordinates.
    pixel_coords = pixel_coords[:, :2, :]  # Bx2xN

    # Replace NaNs with the default value.
    pixel_coords = torch.nan_to_num(pixel_coords, nan=default)

    return pixel_coords.transpose(1, 2)  # BxNx2
