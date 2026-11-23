# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Quaternion to rotation matrix conversion (COLMAP convention)."""

import numpy as np


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Converts a quaternion to a rotation matrix.

    Args:
        qvec: (4,) quaternion (w, x, y, z), as COLMAP stores it.

    Returns:
        The (3, 3) rotation matrix.
    """
    return np.array([[
        1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
        2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
        2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
    ],
                     [
                         2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
                         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
                     ],
                     [
                         2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
                     ]])
