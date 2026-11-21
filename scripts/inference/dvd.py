# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Official DVD v1.1 (EnVision-Research/DVD) VideoDepth, RGB only.

Ported from PR #21's dvd.py (eval_refactor_hsy), depth-only variant: the official Wan2.1-T2V-1.3B DiT + VAE with the
DVD LoRA merged, the official training-scale resize (at least 640x480, each side rounded up to 16 --
adapter_utils.plan_spatial_transform(method="dvd")), and the official sliding-window stitching
(test_single_video.generate_depth_sliced, window 45, overlap 9). The stitched normalized disparity is restored to
each frame's source grid before inversion: raw_disparities/ keeps it, depths/ holds the relative-depth proxy
1 / max(q, 0.001) (not metric depth). No GT is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from adapter_utils import (
    add_code_path,
    apply_spatial_transform_tensor,
    save_prediction_map,
    canonical_image_root,
    export_depth_predictions,
    find_image_paths,
    mark_model_loaded,
    plan_spatial_transform,
    print_dry_run,
    restore_prediction,
    select_interval,
    write_adapter_metrics,
)
from single_image_depth import METHOD_CODE_ROOT, resolve_checkpoint_file

METHOD = "DVD"
WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", default="FayeHongfeiZhang/DVD", help="HF repo id, directory or file")
    parser.add_argument("--code_path", type=Path, default=METHOD_CODE_ROOT / "DVD")
    parser.add_argument("--wan_path", default=os.environ.get("WAN_MODEL_PATH", WAN_REPO),
                        help="Wan2.1-T2V-1.3B directory or HF repo id (DiT safetensors + Wan2.1_VAE.pth)")
    parser.add_argument("--track", choices=("rgb",), default="rgb")
    # The engine forwards a configured SPATIAL_POLICY to every depth-only adapter. DVD has one policy, the benchmark's
    # "shared" one: the official resize, with every prediction restored to its frame's source grid.
    parser.add_argument("--spatial_policy", choices=("shared",), default="shared")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=45)
    parser.add_argument("--overlap", type=int, default=9)
    parser.add_argument("--inverse_epsilon", type=float, default=1e-3)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.overlap < args.window_size and args.window_size % 4 == 1):
        raise ValueError("--window_size must be 4n+1 and --overlap in [0, window_size)")
    args.output_path = args.output_path.resolve()
    return args


def wan_directory(wan_path: str) -> Path:
    """The Wan2.1 base model directory, given as one or as an HF repo id (resolved from the cache)."""
    if Path(wan_path).is_dir():
        return Path(wan_path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(wan_path, allow_patterns=["diffusion_pytorch_model*.safetensors",
                                                            "Wan2.1_VAE.pth", "config.json"]))


def run() -> None:
    args = parse_args()
    if args.export_colmap or not args.export_depth:
        raise ValueError("DVD predicts dense depth only: pass --export_depth, not --export_colmap")
    images_dir = canonical_image_root(args.data_path)
    paths = [str(Path(p).resolve())
             for p in select_interval(find_image_paths(images_dir), args.interval, args.max_images_per_run)]
    if not paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, METHOD, len(paths))
        return
    sizes = set()
    for path in paths:
        with Image.open(path) as image:
            sizes.add(image.size)
    if len(sizes) != 1:
        raise ValueError(f"DVD needs one frame size per sequence, got {sorted(sizes)}")
    transform = plan_spatial_transform(sizes.pop(), method="dvd")
    model_w, model_h = transform.model_wh
    checkpoint = resolve_checkpoint_file(args.ckpt_path, "dvd_1.1.safetensors")
    wan = wan_directory(args.wan_path)

    code_path = args.code_path.resolve()
    add_code_path(code_path, ("examples", "test_script", "diffsynth"))
    os.chdir(code_path)  # the official pipeline resolves its bundled tokenizer relative to the repository root
    import torch
    from accelerate import Accelerator
    from examples.wanvideo.model_training.WanTrainingModule import WanTrainingModule
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from test_script.test_single_video import generate_depth_sliced

    cfg = OmegaConf.load(code_path / "ckpt" / "model_config.yaml")
    model = WanTrainingModule(
        args=cfg,
        accelerator=Accelerator(),
        model_paths=json.dumps([str(wan / "diffusion_pytorch_model.safetensors"), str(wan / "Wan2.1_VAE.pth")]),
        trainable_models=None,
        lora_rank=cfg.lora_rank,
        lora_base_model=cfg.lora_base_model,
        use_gradient_checkpointing=False,
    )
    weights = load_file(checkpoint)
    dit = {key.removeprefix("pipe.dit."): value for key, value in weights.items() if key.startswith("pipe.dit.")}
    if not dit:
        raise ValueError(f"{checkpoint} contains no pipe.dit parameters")
    model.pipe.dit.load_state_dict(dit, strict=True)
    del weights, dit
    model.merge_lora_layer()
    model = model.to("cuda").eval()

    started_at = mark_model_loaded()
    rgb = np.stack([np.asarray(Image.open(path).convert("RGB")) for path in paths])
    video = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
    video = apply_spatial_transform_tensor(video, transform)[None]
    if tuple(video.shape[-2:]) != (model_h, model_w):
        raise ValueError(f"Preprocessed frames are {tuple(video.shape[-2:])}, planned {(model_h, model_w)}")
    with torch.inference_mode():
        disparity = generate_depth_sliced(model, video, window_size=args.window_size, overlap=args.overlap)[0]
    disparity = np.asarray(disparity).mean(-1)
    if disparity.shape != (len(paths), model_h, model_w) or not np.isfinite(disparity).all():
        raise ValueError(f"Invalid stitched disparity: shape {disparity.shape}")

    # Interpolate disparity (DVD's representation) to the source grid, then invert.
    disparity = np.stack([restore_prediction(frame, transform) for frame in disparity]).astype(np.float32)
    raw_dir = args.output_path / "raw_disparities"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for path, frame in zip(paths, disparity):
        save_prediction_map(raw_dir, Path(path).stem, frame)
    export_depth_predictions(args.output_path, paths, 1.0 / np.maximum(disparity, args.inverse_epsilon))
    (args.output_path / "spatial_transforms.json").write_text(
        json.dumps({Path(path).name: transform.metadata() for path in paths}, indent=2), encoding="utf-8")
    write_adapter_metrics(
        output_path=args.output_path,
        method=METHOD,
        image_paths=paths,
        started_at=started_at,
        checkpoint=checkpoint,
        depth_only=True,
        metrics_extra={
            "track": args.track, "spatial_policy": args.spatial_policy, "model_wh": [model_w, model_h],
            "window_size": args.window_size, "overlap": args.overlap, "raw_representation": "normalized_disparity",
            "depth_conversion": f"1 / max(q, {args.inverse_epsilon})",
            "clamped_fraction": float((disparity < args.inverse_epsilon).mean()),
        },
    )


if __name__ == "__main__":
    run()
