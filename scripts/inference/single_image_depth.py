# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Shared loop for the official single-image MonoDepth baselines (MoGe, Marigold, Lotus, Depth Pro, UniDepth V2,
InfiniDepth).

Each of these models takes one RGB image and predicts depth on that image's own pixel grid, so an adapter only
supplies ``predict(rgb) -> (depth, raw_disparity | None)``. This module selects the frames
(``select_interval``, which honours the engine's ``BENCHMARK_IMAGE_SUBSET`` size groups), checks every prediction
is exactly the source image's (H, W) -- a model that changed the grid (e.g. an EXIF-rotated input) fails the case
instead of being silently resized -- and exports ``depths/<stem>.npz`` (plus ``raw_disparities/<stem>.npz`` for
disparity models, read by the inverse-depth protocols) and ``metrics.json``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from adapter_utils import (
    canonical_image_root,
    export_depth_predictions,
    save_prediction_map,
    find_image_paths,
    mark_model_loaded,
    print_dry_run,
    select_interval,
    write_adapter_metrics,
)

# Shared method checkouts (the engine's METHOD_CODE_ROOT); the engine also passes --code_path.
# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
METHOD_CODE_ROOT = Path(
    os.environ.get("METHOD_CODE_ROOT", f"{SPATIALARENA_ROOT}/benchmarks/_model_codes")
)

Predictor = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray | None]"]


def parse_args(description: str, methods: dict[str, str], code_dirname: str) -> argparse.Namespace:
    """The adapter CLI: ``methods`` maps each method name this adapter serves to its default checkpoint."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--method", choices=sorted(methods), default=next(iter(methods)))
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", default="", help="HF repo id, directory or file; default: the method's own")
    parser.add_argument("--code_path", type=Path, default=METHOD_CODE_ROOT / code_dirname)
    parser.add_argument("--track", choices=("rgb",), default="rgb")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.ckpt_path = args.ckpt_path or methods[args.method]
    args.output_path = args.output_path.resolve()
    return args


def select_images(args: argparse.Namespace) -> list[str] | None:
    """The frames this call predicts (absolute paths); None after a --dry_run report."""
    if args.export_colmap or not args.export_depth:
        raise ValueError(f"{args.method} predicts dense depth only: pass --export_depth, not --export_colmap")
    images_dir = canonical_image_root(args.data_path)
    image_paths = [str(Path(path).resolve()) for path in select_interval(find_image_paths(images_dir), args.interval)]
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, args.method, len(image_paths))
        return None
    return image_paths


def resolve_checkpoint_file(ckpt_path: str, filename: str) -> str:
    """A checkpoint file given as itself, as a directory holding ``filename``, or as an HF repo id (cached)."""
    path = Path(ckpt_path).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        return str(path / filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(ckpt_path, filename)


def load_rgb(path: str) -> np.ndarray:
    """H x W x 3 uint8, in the file's stored orientation (no EXIF transpose), like the GT depth maps."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def run_single_image_depth(
    args: argparse.Namespace,
    image_paths: list[str],
    predict: Predictor,
    *,
    metrics_extra: dict[str, Any] | None = None,
) -> None:
    """Predict each image on its own and export its depth on the source grid."""
    args.output_path.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_path / "raw_disparities"
    started_at = mark_model_loaded()
    nonfinite = 0
    for index, path in enumerate(image_paths):
        rgb = load_rgb(path)
        depth, raw = predict(rgb)
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        for name, array in (("depth", depth), ("raw disparity", raw)):
            if array is not None and np.asarray(array).squeeze().shape != rgb.shape[:2]:
                raise ValueError(f"{args.method} {name} for {path} has shape {np.asarray(array).squeeze().shape}, "
                                 f"not the source image's {rgb.shape[:2]}")
        nonfinite += int((~np.isfinite(depth)).sum())
        export_depth_predictions(args.output_path, [path], depth[None])
        if raw is not None:
            raw_dir.mkdir(exist_ok=True)
            save_prediction_map(raw_dir, Path(path).stem, np.asarray(raw, dtype=np.float32).squeeze())
        print(f"[{args.method}] {index + 1}/{len(image_paths)} {Path(path).name}", flush=True)
    write_adapter_metrics(
        output_path=args.output_path,
        method=args.method,
        image_paths=image_paths,
        started_at=started_at,
        checkpoint=args.ckpt_path,
        depth_only=True,
        metrics_extra={"track": args.track, "seed": args.seed, "nonfinite_depth_pixels": nonfinite,
                       **(metrics_extra or {})},
    )
