# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Alias of spatialarena.models.slam_depth (the SLAM models' depth export).

Kept for the code that imports scripts.inference.slam_depth_export.
"""

import pathlib
import sys

_ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# pylint: disable-next=wrong-import-position
from spatialarena.models import slam_depth  # noqa: E402

WRITE_BATCH = slam_depth.WRITE_BATCH
DepthSink = slam_depth.DepthSink
chunk_scale = slam_depth.chunk_scale
chunk_ranges = slam_depth.chunk_ranges
export_long_chunks = slam_depth.export_long_chunks
export_numbered_maps = slam_depth.export_numbered_maps
read_exr_depth = slam_depth.read_exr_depth
export_vggt_slam_dumps = slam_depth.export_vggt_slam_dumps
