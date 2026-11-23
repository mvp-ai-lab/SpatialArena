# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official Marigold depth (prs-eth): v1-0 (MARIGOLD) and v1-1 (MARIGOLD_V1_1), via diffusers' MarigoldDepthPipeline.

Each image is predicted on its own with the checkpoint's own default denoising steps and processing resolution, one
ensemble member, fp32 and a fixed seed, and returned at the input resolution (``match_input_resolution``). The
prediction is affine-invariant depth in [0, 1].
"""

from __future__ import annotations

import os

from PIL import Image

from single_image_depth import parse_args, run_single_image_depth, select_images

METHODS = {
    "MARIGOLD": "prs-eth/marigold-depth-v1-0",
    "MARIGOLD_V1_1": "prs-eth/marigold-depth-v1-1"
}


def processing_options() -> dict[str, int]:
    """The pipeline's processing_resolution from MARIGOLD_PROCESSING_RES (0: the input's own size); unset keeps the
    checkpoint's default (768)."""
    resolution = os.environ.get("MARIGOLD_PROCESSING_RES")
    kwargs = {} if resolution is None else {
        "processing_resolution": int(resolution)
    }
    if kwargs.get("processing_resolution", 0) < 0:
        raise ValueError("MARIGOLD_PROCESSING_RES must be nonnegative")
    return kwargs


def run() -> None:
    args = parse_args(__doc__, METHODS, "Marigold")
    image_paths = select_images(args)
    if image_paths is None:
        return
    import torch
    from diffusers import MarigoldDepthPipeline

    pipe = MarigoldDepthPipeline.from_pretrained(
        args.ckpt_path, torch_dtype=torch.float32).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    kwargs = processing_options()

    def predict(rgb):
        generator = torch.Generator("cuda").manual_seed(args.seed)
        output = pipe(Image.fromarray(rgb),
                      ensemble_size=1,
                      match_input_resolution=True,
                      generator=generator,
                      **kwargs)
        return output.prediction, None

    run_single_image_depth(args,
                           image_paths,
                           predict,
                           metrics_extra={
                               "denoising_steps":
                                   pipe.default_denoising_steps,
                               "processing_resolution":
                                   kwargs.get(
                                       "processing_resolution",
                                       pipe.default_processing_resolution),
                               "ensemble_size":
                                   1,
                           })


if __name__ == "__main__":
    run()
