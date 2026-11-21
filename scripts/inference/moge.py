# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Official MoGe / MoGe-2 / MoGe-3 single-image depth (microsoft/MoGe).

Each image goes through ``MoGeModel.infer`` at its own resolution (the model resizes internally and returns depth on
the input grid), with ``resolution_level=9``, fp32 and ``apply_mask=False`` so every pixel keeps a depth (as in the
official evaluation); no GT intrinsics are given. MoGe predicts affine-invariant geometry; MoGe-2/-3 metric.
"""

from __future__ import annotations

import importlib

from adapter_utils import add_code_path
from single_image_depth import parse_args, run_single_image_depth, select_images

METHODS = {"MOGE": "Ruicheng/moge-vitl", "MOGE2": "Ruicheng/moge-2-vitl", "MOGE3": "Ruicheng/moge-3-vitl"}
VERSIONS = {"MOGE": "v1", "MOGE2": "v2", "MOGE3": "v3"}


def run() -> None:
    args = parse_args(__doc__, METHODS, "MoGe")
    image_paths = select_images(args)
    if image_paths is None:
        return
    add_code_path(args.code_path, ("moge",))
    import torch

    torch.manual_seed(args.seed)
    model_class = importlib.import_module(f"moge.model.{VERSIONS[args.method]}").MoGeModel
    model = model_class.from_pretrained(args.ckpt_path).cuda().eval()

    def predict(rgb):
        image = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda() / 255.0
        with torch.inference_mode():
            output = model.infer(image, resolution_level=9, apply_mask=False, use_fp16=False)
        return output["depth"].float().cpu().numpy(), None

    run_single_image_depth(args, image_paths, predict, metrics_extra={
        "moge_version": VERSIONS[args.method], "resolution_level": 9, "apply_mask": False, "use_fp16": False,
    })


if __name__ == "__main__":
    run()
