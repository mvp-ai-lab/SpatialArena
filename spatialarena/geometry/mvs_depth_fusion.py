# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Fuses multi-view depth maps into a dense colored point cloud.

A pixel of a reference view is kept when its point agrees, within a distance
threshold, with the points of at least `num_consist` views warped into the
reference view; the kept point is the average of the agreeing points. The
fusion code is borrowed from https://github.com/doubleZ0108/GeoMVSNet.
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    """Warps source features into the reference view at the given depths.

    Args:
        src_fea: [B, C, H, W] source features.
        src_proj: [B, 4, 4] source projection matrices.
        ref_proj: [B, 4, 4] reference projection matrices.
        depth_values: [B, Ndepth] or [B, Ndepth, H, W] reference depths.

    Returns:
        The warped features, reshaped to [B, C, H, W].
    """
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([
            torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
            torch.arange(0, width, dtype=torch.float32, device=src_fea.device)
        ])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_values.view(
            -1, 1, 1, height * width)  # [B, 3, 1, H*W]

        # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
        # [B, 2, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        # [B, Ndepth, H*W, 2]
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)
        grid = proj_xy

    warped_src_fea = F.grid_sample(
        src_fea,
        grid.view(batch, height, width, 2),
        mode='bilinear',
        padding_mode='zeros',
    )
    warped_src_fea = warped_src_fea.view(batch, channels, height, width)

    return warped_src_fea


def generate_points_from_depth(depth, proj):
    """Unprojects depth maps to world points with the inverse projections.

    Args:
        depth: (B, 1, H, W) depths, a torch tensor or a numpy array.
        proj: (B, 4, 4) projection matrices, a torch tensor or a numpy array.

    Returns:
        The (B, 3, H, W) point cloud.
    """
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth).float()

    if isinstance(proj, np.ndarray):
        proj = torch.from_numpy(proj).float()

    batch, height, width = depth.shape[0], depth.shape[2], depth.shape[3]
    inv_proj = torch.inverse(proj)

    rot = inv_proj[:, :3, :3]  # [B,3,3]
    trans = inv_proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([
        torch.arange(0, height, dtype=torch.float32, device=depth.device),
        torch.arange(0, width, dtype=torch.float32, device=depth.device)
    ])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = rot_xyz * depth.view(batch, 1, -1)
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)  # [B, 3, H*W]
    proj_xyz = proj_xyz.view(batch, 3, height, width)

    return proj_xyz


def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
    """Compares the reference points with the source points warped onto it.

    Args:
        ref_depth: (1, 1, H, W) reference depth.
        src_depths: (B, 1, H, W) source depths.
        ref_proj: (1, 4, 4) reference projection.
        src_projs: (B, 4, 4) source projections.

    Returns:
        ref_pc (1, 3, H, W), aligned_pcs (B, 3, H, W) and their distances
        dist (B, 1, H, W).
    """
    ref_pc = generate_points_from_depth(ref_depth, ref_proj)
    src_pcs = generate_points_from_depth(src_depths, src_projs)

    aligned_pcs = homo_warping(src_pcs, src_projs, ref_proj, ref_depth)

    x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0])**2
    y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1])**2
    z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2])**2
    dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)

    return ref_pc, aligned_pcs, dist


def extract_points(pc, mask, rgb):
    """Returns the (M, 6) points and colors of the pixels where mask is set."""
    pc = pc.cpu().numpy()
    mask = mask.cpu().numpy()

    mask = np.reshape(mask, (-1,))
    pc = np.reshape(pc, (-1, 3))
    rgb = np.reshape(rgb, (-1, 3))

    points = pc[np.where(mask)]
    colors = rgb[np.where(mask)]

    points_with_color = np.concatenate([points, colors], axis=1)

    return points_with_color


@torch.no_grad()
def depth_fusion(
    depths: torch.Tensor,
    projs: torch.Tensor,
    rgbs: List,
    valid_masks: torch.Tensor = None,
    dist_thresh: float = 0.2,
    num_consist: int = 4,
) -> np.ndarray:
    """Fuses per-view depth maps into one colored point cloud.

    Args:
        depths: (N, 1, H, W) depth maps.
        projs: (N, 4, 4) projection matrices.
        rgbs: The N per-view colors, each reshapeable to (H * W, 3).
        valid_masks: Optional (N, H, W) bool masks of the pixels to keep.
        dist_thresh: The largest distance of an agreeing point.
        num_consist: The least number of agreeing views of a kept pixel.

    Returns:
        The (M, 6) fused points (xyz then the color).
    """
    batch_size = 20
    num_frames = depths.shape[0]
    height, width = depths.shape[2], depths.shape[3]
    points = []

    for i in range(num_frames):
        pc_buff = torch.zeros((3, height, width),
                              device=depths.device,
                              dtype=depths.dtype)
        val_cnt = torch.zeros((1, height, width),
                              device=depths.device,
                              dtype=depths.dtype)
        j = 0

        while True:
            ref_pc, pcs, dist = filter_depth(
                ref_depth=depths[i:i + 1],
                src_depths=depths[j:min(j + batch_size, num_frames)],
                ref_proj=projs[i:i + 1],
                src_projs=projs[j:min(j + batch_size, num_frames)])
            masks = (dist < dist_thresh).float()
            masked_pc = pcs * masks
            pc_buff += masked_pc.sum(dim=0, keepdim=False)
            val_cnt += masks.sum(dim=0, keepdim=False)

            j += batch_size
            if j >= num_frames:
                break

        final_mask = (val_cnt >= num_consist).squeeze(0)
        if valid_masks is not None:
            final_mask = final_mask & valid_masks[i]

        avg_points = torch.div(pc_buff, val_cnt).permute(1, 2, 0)
        final_pc = extract_points(avg_points, final_mask, rgbs[i])
        points.append(final_pc)

    final_points = np.concatenate(points, axis=0)
    return final_points
