# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official InfiniDepth (zju3dv/InfiniDepth) RGB-only depth.

Uses the official ``inference_depth.run_depth_inference`` with the RGB model
(``InfiniDepth``, not the depth-sensor variant): its default 768x1024 model
input, metric scale from the bundled MoGe-2 checkpoint (no input depth, no GT
intrinsics), no sky segmentation, and ``output_resolution_mode="original"`` so
the implicit field is queried on the source pixel grid. The official loader
applies EXIF orientation; a rotated input therefore comes back transposed and
fails the shared shape check instead of being scored misaligned.
"""

from __future__ import annotations

import os

from spatialarena.models import registry
from spatialarena.models import single_image
from spatialarena.models import utils

# The adapter's command-line description.
DESCRIPTION = __doc__


@registry.register_model
class InfiniDepth(single_image.SingleImageDepthModel):
    """InfiniDepth (RGB model)."""

    name = "INFINIDEPTH"
    aliases = ("INFINI_DEPTH", "INFINI-DEPTH")
    description = DESCRIPTION
    code_packages = ("InfiniDepth",)
    # The official loader picks its own device.
    requires_cuda = False

    def __init__(self, args):
        super().__init__(args)
        self.checkpoints = None
        self.options = None
        self.inference_device = None

    def add_code_path(self):
        """Resolves the checkpoints, then enters the official checkout.

        The official code resolves some of its own files relative to the
        repository root, so the working directory becomes --code_path.
        """
        self.checkpoints = (
            single_image.resolve_checkpoint_file(self.args.ckpt_path,
                                                 "infinidepth.ckpt"),
            single_image.resolve_checkpoint_file(self.args.ckpt_path,
                                                 "moge2.pt"),
        )
        utils.add_code_path(self.args.code_path, self.code_packages)
        os.chdir(self.args.code_path)

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        import inference_depth

        del device  # load_depth_model picks it.
        depth_ckpt, moge2_ckpt = self.checkpoints
        torch.manual_seed(self.args.seed)
        self.options = inference_depth.DepthInferenceArgs(
            input_image_path=self.image_paths[0] if self.image_paths else "",
            model_type="InfiniDepth",
            depth_model_path=depth_ckpt,
            moge2_pretrained=moge2_ckpt,
            output_resolution_mode="original",
            save_pcd=False,
            enable_skyseg_model=False)
        model, self.inference_device = inference_depth.load_depth_model(
            self.options)
        return model

    def infer(self, rgb, image_path):
        import inference_depth  # pylint: disable=import-outside-toplevel

        result = inference_depth.run_depth_inference(
            self.options,
            model=self.model,
            device=self.inference_device,
            input_image_path=image_path)
        return result.pred_depthmap.float().cpu().numpy(), None

    def metrics_extra(self, depth_only=True, images_per_run=None):
        del depth_only, images_per_run  # Always one image per call.
        return {
            "model_type": "InfiniDepth",
            "input_size": list(self.options.input_size),
            "output_resolution_mode": "original",
            "metric_scale": "MoGe-2 (bundled moge2.pt)",
        }
