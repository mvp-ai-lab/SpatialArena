# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official Lotus-D v2 disparity (EnVision-Research/Lotus), in-process.

Mirrors the official infer.py regression path (and PR #21's lotus_official.py
settings): one image per forward, single step at timestep 999, fp16, seed 42,
processing resolution 768 (long side), bilinear resampling, output at the
input resolution. infer.py itself is not used because it only globs
*.png/*.jpg. The prediction is normalized disparity q in [0, 1]; depths/ holds
1 / max(q, 0.001) and raw_disparities/ keeps q unmodified.
"""

from __future__ import annotations

import numpy as np

from spatialarena.models import registry
from spatialarena.models import single_image

# The adapter's command-line description.
DESCRIPTION = __doc__
PROCESSING_RES = 768
DISPARITY_FLOOR = 1e-3


@registry.register_model
class LotusD(single_image.SingleImageDepthModel):
    """Lotus-D v2 (disparity)."""

    name = "LOTUS_D"
    aliases = ("LOTUS", "LOTUS-D")
    description = DESCRIPTION
    code_packages = ("pipeline", "utils")

    def __init__(self, args):
        super().__init__(args)
        self.task_emb = None

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        import pipeline

        del device  # CUDA only.
        torch.manual_seed(self.args.seed)
        pipe = pipeline.LotusDPipeline.from_pretrained(
            self.args.ckpt_path, torch_dtype=torch.float16).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).cuda()
        self.task_emb = torch.cat(
            [torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
        return pipe

    def infer(self, rgb, image_path):
        import torch  # pylint: disable=import-outside-toplevel

        image = torch.from_numpy(rgb.astype(np.float32)).permute(
            2, 0, 1).unsqueeze(0).cuda() / 127.5 - 1.0
        generator = torch.Generator("cuda").manual_seed(self.args.seed)
        with torch.no_grad(), torch.autocast("cuda"):
            output = self.model(rgb_in=image,
                                prompt="",
                                num_inference_steps=1,
                                generator=generator,
                                output_type="np",
                                timesteps=[999],
                                task_emb=self.task_emb,
                                processing_res=PROCESSING_RES,
                                match_input_res=True,
                                resample_method="bilinear").images[0]
        disparity = output.mean(axis=-1).astype(np.float32)
        return 1.0 / np.maximum(disparity, DISPARITY_FLOOR), disparity

    def metrics_extra(self, depth_only=True, images_per_run=None):
        del depth_only, images_per_run  # Always one image per call.
        return {
            "raw_representation": "normalized_disparity",
            "depth_conversion": f"1 / max(q, {DISPARITY_FLOOR})",
            "processing_res": PROCESSING_RES,
            "timestep": 999,
        }
