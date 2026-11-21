# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Official UniDepth V2 (lpiccinelli-eth/UniDepth) metric depth.

Each image goes through ``UniDepthV2.infer`` without a camera (rgb track) and with ``resolution_level`` unset, the
documented default (the full training pixel range); the model pads/resizes internally and returns depth at the input
resolution.
"""

from __future__ import annotations

from adapter_utils import add_code_path
from single_image_depth import parse_args, run_single_image_depth, select_images

METHODS = {"UNIDEPTH_V2": "lpiccinelli/unidepth-v2-vitl14"}


def run() -> None:
    args = parse_args(__doc__, METHODS, "UniDepth")
    image_paths = select_images(args)
    if image_paths is None:
        return
    add_code_path(args.code_path, ("unidepth",))
    import torch
    from unidepth.models import UniDepthV2  # pylint: disable=no-name-in-module

    torch.manual_seed(args.seed)
    model = UniDepthV2.from_pretrained(args.ckpt_path).cuda().eval()

    def predict(rgb):
        with torch.no_grad():
            output = model.infer(torch.from_numpy(rgb).permute(2, 0, 1))
        return output["depth"].float().cpu().numpy(), None

    run_single_image_depth(args, image_paths, predict, metrics_extra={"camera": None})


if __name__ == "__main__":
    run()
