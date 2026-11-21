# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Official InfiniDepth (zju3dv/InfiniDepth) RGB-only depth.

Uses the official ``inference_depth.run_depth_inference`` with the RGB model (``InfiniDepth``, not the depth-sensor
variant): its default 768x1024 model input, metric scale from the bundled MoGe-2 checkpoint (no input depth, no GT
intrinsics), no sky segmentation, and ``output_resolution_mode="original"`` so the implicit field is queried on the
source pixel grid. The official loader applies EXIF orientation; a rotated input therefore comes back transposed and
fails the shared shape check instead of being scored misaligned.
"""

from __future__ import annotations

import os

from adapter_utils import add_code_path
from single_image_depth import parse_args, resolve_checkpoint_file, run_single_image_depth, select_images

METHODS = {"INFINIDEPTH": "ritianyu/InfiniDepth"}


def run() -> None:
    args = parse_args(__doc__, METHODS, "InfiniDepth")
    image_paths = select_images(args)
    if image_paths is None:
        return
    depth_ckpt = resolve_checkpoint_file(args.ckpt_path, "infinidepth.ckpt")
    moge2_ckpt = resolve_checkpoint_file(args.ckpt_path, "moge2.pt")
    add_code_path(args.code_path, ("InfiniDepth",))
    os.chdir(args.code_path)  # the official code resolves some of its own files relative to the repository root
    import torch
    from inference_depth import DepthInferenceArgs, load_depth_model, run_depth_inference

    torch.manual_seed(args.seed)
    options = DepthInferenceArgs(input_image_path=image_paths[0], model_type="InfiniDepth",
                                 depth_model_path=depth_ckpt, moge2_pretrained=moge2_ckpt,
                                 output_resolution_mode="original", save_pcd=False, enable_skyseg_model=False)
    model, device = load_depth_model(options)

    paths = iter(image_paths)

    def predict(_rgb):
        result = run_depth_inference(options, model=model, device=device, input_image_path=next(paths))
        return result.pred_depthmap.float().cpu().numpy(), None

    run_single_image_depth(args, image_paths, predict, metrics_extra={
        "model_type": "InfiniDepth", "input_size": list(options.input_size), "output_resolution_mode": "original",
        "metric_scale": "MoGe-2 (bundled moge2.pt)",
    })


if __name__ == "__main__":
    run()
