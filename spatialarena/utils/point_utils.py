# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Point cloud export to PLY files."""

import numpy as np
import plyfile


def export_ply(path: str, xyz: np.ndarray, color: np.ndarray):
    """Writes a colored point cloud as a binary PLY file with zero normals.

    Args:
        path: The output .ply path.
        xyz: (N, 3) point positions.
        color: (N, 3) RGB colors, stored as uint8.

    Raises:
        ValueError: If xyz or color does not have shape (N, 3).
    """
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    xyz = np.asarray(xyz, dtype=np.float32)
    color = np.asarray(color)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape Nx3, got {xyz.shape}")
    if color.ndim != 2 or color.shape[1] != 3 or color.shape[0] != xyz.shape[0]:
        raise ValueError(
            f"color must have shape Nx3 aligned with xyz, got {color.shape}")

    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["red"] = color[:, 0].astype(np.uint8, copy=False)
    elements["green"] = color[:, 1].astype(np.uint8, copy=False)
    elements["blue"] = color[:, 2].astype(np.uint8, copy=False)

    vertex_element = plyfile.PlyElement.describe(elements, "vertex")
    plyfile.PlyData([vertex_element]).write(path)
