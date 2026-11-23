# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official Lotus-D v2 disparity (EnVision-Research/Lotus), in-process.

Mirrors the official infer.py regression path (and PR #21's lotus_official.py settings): one image per forward,
single step at timestep 999, fp16, seed 42, processing resolution 768 (long side), bilinear resampling, output at the
input resolution. infer.py itself is not used because it only globs *.png/*.jpg. The prediction is normalized
disparity q in [0, 1]; depths/ holds 1 / max(q, 0.001) and raw_disparities/ keeps q unmodified.
"""

from __future__ import annotations

import numpy as np

from adapter_utils import add_code_path
from single_image_depth import parse_args, run_single_image_depth, select_images

METHODS = {"LOTUS_D": "jingheya/lotus-depth-d-v2-0-disparity"}
PROCESSING_RES = 768
DISPARITY_FLOOR = 1e-3


def run() -> None:
    args = parse_args(__doc__, METHODS, "Lotus")
    image_paths = select_images(args)
    if image_paths is None:
        return
    add_code_path(args.code_path, ("pipeline", "utils"))
    import torch
    from pipeline import LotusDPipeline

    torch.manual_seed(args.seed)
    pipe = LotusDPipeline.from_pretrained(args.ckpt_path,
                                          torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).cuda()
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    def predict(rgb):
        image = torch.from_numpy(rgb.astype(np.float32)).permute(
            2, 0, 1).unsqueeze(0).cuda() / 127.5 - 1.0
        generator = torch.Generator("cuda").manual_seed(args.seed)
        with torch.no_grad(), torch.autocast("cuda"):
            output = pipe(rgb_in=image,
                          prompt="",
                          num_inference_steps=1,
                          generator=generator,
                          output_type="np",
                          timesteps=[999],
                          task_emb=task_emb,
                          processing_res=PROCESSING_RES,
                          match_input_res=True,
                          resample_method="bilinear").images[0]
        disparity = output.mean(axis=-1).astype(np.float32)
        return 1.0 / np.maximum(disparity, DISPARITY_FLOOR), disparity

    run_single_image_depth(
        args,
        image_paths,
        predict,
        metrics_extra={
            "raw_representation": "normalized_disparity",
            "depth_conversion": f"1 / max(q, {DISPARITY_FLOOR})",
            "processing_res": PROCESSING_RES,
            "timestep": 999,
        })


if __name__ == "__main__":
    run()
