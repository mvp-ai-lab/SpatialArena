# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official DepthCrafter/ChronoDepth pipelines on canonical selected image sequences."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from adapter_utils import (
    add_code_path, canonical_image_root, export_depth_predictions, find_image_paths,
    mark_model_loaded, print_dry_run, save_prediction_map, select_interval, write_adapter_metrics,
)

REPOS = {"DEPTHCRAFTER": "tencent/DepthCrafter", "CHRONODEPTH": "jhshao/ChronoDepth-v1"}


def processing_size(width: int, height: int, max_res: int) -> tuple[int, int]:
    """Official open-video rounding to 64, with a long-edge limit."""
    if min(width, height, max_res) <= 0:
        raise ValueError("Image dimensions and max_res must be positive")
    w, h = round(width / 64) * 64, round(height / 64) * 64
    if max(w, h) > max_res:
        scale = max_res / max(width, height)
        w, h = round(width * scale / 64) * 64, round(height * scale / 64) * 64
    if min(w, h) < 64:
        raise ValueError("Official preprocessing would produce a zero-size image")
    return w, h


def restore_output(output, method: str, count: int, source_wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Validate actual pipeline layouts, restore native representation, then convert for FFB."""
    raw = np.asarray(output, dtype=np.float32)
    if method == "DEPTHCRAFTER":
        if raw.ndim != 5 or raw.shape[0] != 1 or raw.shape[-1] != 3:
            raise ValueError(f"Unexpected DepthCrafter output {raw.shape}")
        raw = raw[0].mean(axis=-1)
    elif method == "CHRONODEPTH":
        if raw.ndim != 4 or raw.shape[1] != 1:
            raise ValueError(f"Unexpected ChronoDepth output {raw.shape}")
        raw = raw[:, 0]
    else:
        raise ValueError(method)
    if raw.shape[0] != count or not np.isfinite(raw).all():
        raise ValueError("Prediction count mismatch or nonfinite predictions")
    if method == "DEPTHCRAFTER":
        span = float(raw.max() - raw.min())
        if span <= 1e-8:
            raise ValueError("Constant DepthCrafter prediction")
        raw = (raw - raw.min()) / span  # official sequence-wide normalization, not per-frame
    raw = np.stack([cv2.resize(frame, source_wh, interpolation=cv2.INTER_LINEAR) for frame in raw])
    if method == "DEPTHCRAFTER":
        depth = 1.0 / np.maximum(raw, 1e-3)
    else:
        raw = np.clip(raw, 0.0, 1.0)
        depth = np.maximum(raw, 1e-6)
    return raw, depth.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=REPOS, required=True)
    p.add_argument("--data_path", type=Path, required=True)
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument("--code_path", type=Path, required=True)
    p.add_argument("--ckpt_path", default="")
    p.add_argument("--base_path",
                   default=os.environ.get("SVD_BASE_PATH", "stabilityai/stable-video-diffusion-img2vid-xt"))
    p.add_argument("--track", choices=["rgb"], default="rgb")
    p.add_argument("--spatial_policy", choices=["shared"], default="shared")
    p.add_argument("--interval", type=int, default=1)
    p.add_argument("--max_images_per_run", type=int, default=0)
    p.add_argument("--max_res", type=int, default=1024)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window_size", type=int, default=110)
    p.add_argument("--overlap", type=int, default=25)
    p.add_argument("--n_tokens", type=int, default=10)
    p.add_argument("--chunk_size", type=int, default=5)
    p.add_argument("--cpu_offload", choices=["none", "model", "sequential"], default="none")
    p.add_argument("--export_depth", action="store_true")
    p.add_argument("--export_colmap", action="store_true")
    p.add_argument("--depth_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or not 0 <= args.overlap < args.window_size:
        p.error("Invalid steps/window/overlap")
    if not 0 < args.chunk_size <= args.n_tokens:
        p.error("Require 0 < chunk_size <= n_tokens")
    return args


def run() -> None:
    args = parse_args()
    if args.export_colmap or not args.export_depth:
        raise ValueError("These methods export depth only")
    paths = [str(Path(p).absolute()) for p in select_interval(
        find_image_paths(canonical_image_root(args.data_path)), args.interval, args.max_images_per_run)]
    if not paths or len({Path(p).stem for p in paths}) != len(paths):
        raise ValueError("Empty selection or duplicate frame identifiers")
    if args.dry_run:
        print_dry_run(args, args.method, len(paths))
        return
    sizes = set()
    for path in paths:
        with Image.open(path) as image:
            sizes.add(image.size)
    if len(sizes) != 1:
        raise ValueError("Video frames must share a source resolution")
    source_wh = sizes.pop()
    model_wh = processing_size(*source_wh, args.max_res)
    add_code_path(args.code_path)
    import torch
    from transformers import set_seed

    set_seed(args.seed)
    checkpoint = args.ckpt_path or REPOS[args.method]
    if args.method == "DEPTHCRAFTER":
        from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline as Pipeline
        from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter as UNet
    else:
        from chronodepth.chronodepth_pipeline import ChronoDepthPipeline as Pipeline
        from chronodepth.unet_chronodepth import DiffusersUNetSpatioTemporalConditionModelChronodepth as UNet
    unet = UNet.from_pretrained(checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    pipe = Pipeline.from_pretrained(args.base_path, unet=unet, torch_dtype=torch.float16, variant="fp16")
    if args.cpu_offload == "model":
        pipe.enable_model_cpu_offload()
    elif args.cpu_offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    if args.method == "DEPTHCRAFTER":
        pipe.enable_attention_slicing()
    else:
        pipe.n_tokens, pipe.chunk_size = args.n_tokens, args.chunk_size
    started_at = mark_model_loaded()
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(cv2.resize(np.asarray(image.convert("RGB")), model_wh, interpolation=cv2.INTER_LINEAR))
    frames = np.stack(frames).astype(np.float32) / 255.0
    generator = torch.Generator("cuda").manual_seed(args.seed)
    with torch.inference_mode():
        if args.method == "DEPTHCRAFTER":
            output = pipe(frames, height=model_wh[1], width=model_wh[0], output_type="np",
                          guidance_scale=1.0, num_inference_steps=args.steps, window_size=args.window_size,
                          overlap=args.overlap, generator=generator, track_time=False).frames
        else:
            output = pipe(frames, num_inference_steps=args.steps, decode_chunk_size=8, motion_bucket_id=127,
                          fps=7, noise_aug_strength=0.0, generator=generator,
                          infer_mode="ours", sigma_epsilon=-4.0).frames
    raw, depth = restore_output(output, args.method, len(paths), source_wh)
    export_depth_predictions(args.output_path, paths, depth)
    if args.method == "DEPTHCRAFTER":
        folder = args.output_path / "raw_disparities"
        folder.mkdir(parents=True, exist_ok=True)
        for path, frame in zip(paths, raw):
            save_prediction_map(folder, Path(path).stem, frame)
    write_adapter_metrics(output_path=args.output_path, method=args.method, image_paths=paths,
                          started_at=started_at, checkpoint=checkpoint, depth_only=True, metrics_extra={
                              "source_wh": source_wh, "model_wh": model_wh, "seed": args.seed,
                              "steps": args.steps, "base_model": args.base_path,
                              "raw_representation": "disparity" if args.method == "DEPTHCRAFTER" else "depth",
                              "window_size": args.window_size if args.method == "DEPTHCRAFTER" else args.n_tokens,
                              "overlap": (args.overlap if args.method == "DEPTHCRAFTER"
                                          else args.n_tokens - args.chunk_size),
                              "clamped_fraction": float(
                                  (raw < (1e-3 if args.method == "DEPTHCRAFTER" else 1e-6)).mean()),
                          })


if __name__ == "__main__":
    run()
