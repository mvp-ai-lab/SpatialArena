# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Pack a SpatialArena LiDAR map (``gt_points.ply``) as the dashboard home page's streamed hero point cloud.

Format (little endian): ``b"SAP2"``, uint32 total, uint32 chunks, float32 lo[3], float32 span[3], then per chunk
uint32 count, uint16 xyz[count*3] (position = lo + q / 65535 * span), uint8 rgb[count*3], zero padding to 4 bytes.
Points are cropped to the 0.5-99.5 percentile box, voxel-downsampled to about ``--target`` points and shuffled, so
the first chunk alone is an even sample of the whole scene and every later chunk adds density
(``js/home_scene.js`` plays the first chunk's assembly animation and fades the rest in).

The current asset, ``app/webpage/assets/scene/yorkminster_chapter_house.sap2``, is
``MVRecon/spatialarena/YorminsterTheChapteraofRoof/gt_points.ply`` with the default ``--target`` and ``--chunks``.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """xyz (float64) and rgb (uint8) of a binary little-endian PLY with float xyz and uchar rgb vertices."""
    with open(path, "rb") as handle:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: no PLY header end")
            header += line
        count = int(next(row for row in header.decode().splitlines() if row.startswith("element vertex")).split()[-1])
        vertices = np.fromfile(handle, dtype=PLY_DTYPE, count=count)
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], 1).astype(np.float64)
    return points, np.stack([vertices["r"], vertices["g"], vertices["b"]], 1)


def voxel_size_for(points: np.ndarray, origin: np.ndarray, target: int) -> float:
    """Bisect the voxel edge whose occupied-cell count is about ``target`` (the last edge tried)."""
    low, high, size = 0.002, 1.0, 0.5
    for _ in range(28):
        size = (low + high) / 2
        cells = np.unique(np.floor((points - origin) / size).astype(np.int64), axis=0).shape[0]
        low, high = (size, high) if cells > target else (low, size)
    return size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply", type=Path, help="Input gt_points.ply.")
    parser.add_argument("output", type=Path, help="Output .sap2 file.")
    parser.add_argument("--target", type=int, default=2_400_000, help="Approximate point count after downsampling.")
    parser.add_argument("--chunks", default="400000,500000,500000,500000,500000",
                        help="Comma-separated chunk sizes; the last chunk takes any remainder.")
    args = parser.parse_args()
    chunk_sizes = [int(v) for v in args.chunks.split(",")]

    points, colors = read_ply(args.ply)
    low, high = np.percentile(points, 0.5, 0), np.percentile(points, 99.5, 0)
    keep = ((points >= low) & (points <= high)).all(1)
    points, colors = points[keep], colors[keep]
    voxel = voxel_size_for(points, low, args.target)
    _, first = np.unique(np.floor((points - low) / voxel).astype(np.int64), axis=0, return_index=True)
    points, colors = points[first], colors[first]
    order = np.random.default_rng(0).permutation(len(points))
    points, colors = points[order], colors[order]
    low = points.min(0)
    span = np.maximum(points.max(0) - low, 1e-6)
    quantized = np.round((points - low) / span * 65535).astype("<u2")

    bounds, start = [], 0
    for size in chunk_sizes:
        if start >= len(points):
            break
        bounds.append((start, min(start + size, len(points))))
        start += size
    if start < len(points):
        bounds[-1] = (bounds[-1][0], len(points))
    with open(args.output, "wb") as handle:
        handle.write(b"SAP2")
        handle.write(struct.pack("<II", len(points), len(bounds)))
        handle.write(struct.pack("<6f", *low, *span))
        for begin, end in bounds:
            handle.write(struct.pack("<I", end - begin))
            handle.write(quantized[begin:end].tobytes())
            handle.write(colors[begin:end].astype("u1").tobytes())
            handle.write(b"\0" * (-(9 * (end - begin)) % 4))
    print(json.dumps({"points": len(points), "voxel_m": round(voxel, 4), "chunks": [e - b for b, e in bounds],
                      "span": span.round(2).tolist(), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
