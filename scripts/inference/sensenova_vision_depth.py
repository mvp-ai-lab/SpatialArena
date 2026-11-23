# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run SenseNova-Vision dense-perception depth output for benchmark evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from adapter_utils import (
    SpatialTransform,
    canonical_image_root,
    mark_model_loaded,
    add_code_path,
    export_depth_predictions,
    find_image_paths,
    print_dry_run,
    restore_prediction,
    select_interval,
    write_adapter_metrics,
)

METHOD = "SENSENOVA_VISION"
PROMPT = (
    "<image> Estimate relative depth for each pixel in the image, with closer "
    "objects appearing brighter and distant objects appearing darker. Output "
    "is a grayscale image with pixel values ranging from 0-255.")

# Shared method checkouts (the engine's METHOD_CODE_ROOT); the engine also passes the *_CODE_PATH below.
# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(
    os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
METHOD_CODE_ROOT = Path(
    os.environ.get("METHOD_CODE_ROOT",
                   f"{SPATIALARENA_ROOT}/benchmarks/_model_codes"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path",
                        default=os.environ.get(
                            "SENSENOVA_VISION_CKPT_PATH",
                            "sensenova/SenseNova-Vision-7B-MoT"))
    parser.add_argument(
        "--sensenova_vision_code_path",
        type=Path,
        default=Path(os.environ["SENSENOVA_VISION_CODE_PATH"])
        if "SENSENOVA_VISION_CODE_PATH" in os.environ else METHOD_CODE_ROOT /
        "SenseNova-Vision",
    )
    parser.add_argument("--track", choices=("rgb",), default="rgb")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--image_load_resolution", type=int, default=224)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--max_images_per_run", type=int, default=1)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=0.0)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", default="0")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def relative_depth_from_output(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    depth = 1.0 - gray
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32, copy=False)


def run() -> None:
    args = parse_args()
    if args.export_colmap and not args.depth_only:
        raise ValueError(
            "SenseNova-Vision depth adapter only exports dense depth maps.")

    images_dir = canonical_image_root(args.data_path)
    image_paths = select_interval(find_image_paths(images_dir), args.interval)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, METHOD, len(image_paths))
        return
    if not args.export_depth:
        raise ValueError(
            "SenseNova-Vision depth adapter requires --export_depth.")

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    add_code_path(args.sensenova_vision_code_path,
                  ("inference", "modeling", "data"))
    from inference.sensenova_vision import SenseNovaVisionModel

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES",
                                      "") != "" else "auto"
    model = SenseNovaVisionModel(model_path=args.ckpt_path,
                                 device=device,
                                 dtype="bf16")

    args.output_path.mkdir(parents=True, exist_ok=True)
    started_at = mark_model_loaded()
    for image_path in image_paths:
        with Image.open(image_path) as probe:
            source_wh = probe.size
        result = model.generate(
            question=PROMPT,
            images=[image_path],
            mode="dense_perception",
            return_intermediate_outputs=True,
        )
        output_image = result.get("image") if isinstance(result,
                                                         dict) else result
        if not isinstance(output_image, Image.Image):
            raise RuntimeError(
                f"SenseNova-Vision did not return an image for {image_path}")
        depth = relative_depth_from_output(output_image)
        # ImageTransform's MaxLongEdgeMinShortEdgeResize (data/transforms.py) is a pure resize --
        # no crop, no pad -- so restoring this generated depth map back to the image's true
        # source resolution is the plain-resize degenerate case restore_prediction already
        # handles (same as depthvlm.py/point_query_depth_adapter.py).
        transform = SpatialTransform(source_wh=source_wh,
                                     resized_wh=output_image.size)
        restored = restore_prediction(depth, transform)
        export_depth_predictions(args.output_path, [image_path], restored[None])

    write_adapter_metrics(
        output_path=args.output_path,
        method=METHOD,
        image_paths=image_paths,
        started_at=started_at,
        checkpoint=args.ckpt_path,
        depth_only=True,
        metrics_extra={
            "track": args.track,
            "adapter": "SenseNovaVisionModel.generate(mode=dense_perception)",
            "depth_type": "relative_depth_from_generated_grayscale",
        },
    )


if __name__ == "__main__":
    run()
