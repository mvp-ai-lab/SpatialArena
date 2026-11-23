# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official Depth Pro (apple/ml-depth-pro) metric depth.

Each image goes through the official transform and ``DepthPro.infer`` in fp32 with ``f_px=None``: the focal length
comes from the model's own FoV head, not from EXIF (the rgb track gives no camera). Depth is returned at the input
resolution.
"""

from __future__ import annotations

from dataclasses import replace

from adapter_utils import add_code_path
from single_image_depth import parse_args, resolve_checkpoint_file, run_single_image_depth, select_images

METHODS = {"DEPTH_PRO": "apple/DepthPro"}


def run() -> None:
    args = parse_args(__doc__, METHODS, "ml-depth-pro")
    image_paths = select_images(args)
    if image_paths is None:
        return
    add_code_path(args.code_path / "src", ("depth_pro",))
    import torch
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT  # pylint: disable=no-name-in-module

    config = replace(DEFAULT_MONODEPTH_CONFIG_DICT,
                     checkpoint_uri=resolve_checkpoint_file(
                         args.ckpt_path, "depth_pro.pt"))
    model, transform = depth_pro.create_model_and_transforms(
        config=config, device=torch.device("cuda"))
    model.eval()

    def predict(rgb):
        with torch.no_grad():
            output = model.infer(transform(rgb), f_px=None)
        return output["depth"].float().cpu().numpy(), None

    run_single_image_depth(args,
                           image_paths,
                           predict,
                           metrics_extra={"f_px": "predicted (FoV head)"})


if __name__ == "__main__":
    run()
