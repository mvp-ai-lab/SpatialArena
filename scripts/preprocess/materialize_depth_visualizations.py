#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Convert numeric depth prediction artifacts to visualization PNGs.

Each depth map is normalized to its own 1st-99th percentile of valid values and colored with matplotlib's
``Spectral`` colormap, near red and far blue (as Marigold renders depth); invalid pixels (non-finite or <= 0) are black.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image


NUMERIC_DEPTH_EXTENSIONS = {".npy", ".npz"}
DEPTH_COLORMAP = matplotlib.colormaps["Spectral"]


def load_depth(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32, copy=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            for key in ("depth", "pred", "prediction", "arr_0"):
                if key in archive:
                    return archive[key].astype(np.float32, copy=False)
            if len(archive.files) == 1:
                return archive[archive.files[0]].astype(np.float32, copy=False)
        raise ValueError(f"No depth array found in {path}")
    raise ValueError(f"Unsupported depth extension: {path}")


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    depth = np.squeeze(depth).astype(np.float32, copy=False)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")

    valid = np.isfinite(depth) & (depth > 0)  # adapters mark masked pixels with 0 depth
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    values = depth[valid]
    low, high = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    if high <= low:
        high = low + 1e-6

    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    normalized[~valid] = 0.0

    rgb = DEPTH_COLORMAP(normalized)[..., :3]
    rgb[~valid] = 0.0
    return (rgb * 255.0).astype(np.uint8)


def iter_numeric_depths(prediction_root: Path) -> list[Path]:
    depth_dir = prediction_root / "depths"
    if not depth_dir.is_dir():
        return []
    return sorted(
        path
        for path in depth_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in NUMERIC_DEPTH_EXTENSIONS
    )


def materialize_visualizations(
    prediction_root: Path,
    output_dir: Path,
    *,
    remove_numeric_depths: bool,
) -> dict:
    depth_files = iter_numeric_depths(prediction_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    removed: list[str] = []
    errors: list[dict] = []
    for depth_path in depth_files:
        relative = depth_path.relative_to(prediction_root / "depths")
        png_path = output_dir / relative.with_suffix(".png")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            Image.fromarray(depth_to_rgb(load_depth(depth_path))).save(png_path)
            written.append(str(png_path))
            if remove_numeric_depths:
                depth_path.unlink()
                removed.append(str(depth_path))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append({"path": str(depth_path), "error": str(exc)})

    return {
        "prediction_root": str(prediction_root),
        "output_dir": str(output_dir),
        "num_depth_files": len(depth_files),
        "num_png_written": len(written),
        "num_numeric_removed": len(removed),
        "written": written,
        "removed": removed,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visualization PNGs from numeric benchmark depth predictions."
    )
    parser.add_argument("--prediction_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--remove_numeric_depths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_visualizations(
        args.prediction_root,
        args.output_dir,
        remove_numeric_depths=args.remove_numeric_depths,
    )
    print(
        f"depth visualizations: {result['num_png_written']} written, "
        f"{result['num_numeric_removed']} numeric removed, "
        f"{len(result['errors'])} errors"
    )
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
