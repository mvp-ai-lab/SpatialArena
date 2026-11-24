# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official Depth Pro (apple/ml-depth-pro) metric depth.

Each image goes through the official transform and ``DepthPro.infer`` in fp32
with ``f_px=None``: the focal length comes from the model's own FoV head, not
from EXIF (the rgb track gives no camera). Depth is returned at the input
resolution.
"""

from __future__ import annotations

import dataclasses

from spatialarena.models import registry
from spatialarena.models import single_image
from spatialarena.models import utils

# The adapter's command-line description.
DESCRIPTION = __doc__


@registry.register_model
class DepthPro(single_image.SingleImageDepthModel):
    """Depth Pro."""

    name = "DEPTH_PRO"
    aliases = ("DEPTHPRO", "DEPTH-PRO")
    description = DESCRIPTION
    code_packages = ("depth_pro",)

    def __init__(self, args):
        super().__init__(args)
        self.transform = None

    def add_code_path(self):
        """Puts the checkout's src/ on sys.path."""
        utils.add_code_path(self.args.code_path / "src", self.code_packages)

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        import depth_pro
        from depth_pro import depth_pro as depth_pro_lib

        del device  # CUDA only.
        config = dataclasses.replace(
            depth_pro_lib.DEFAULT_MONODEPTH_CONFIG_DICT,
            checkpoint_uri=single_image.resolve_checkpoint_file(
                self.args.ckpt_path, "depth_pro.pt"))
        model, self.transform = depth_pro.create_model_and_transforms(
            config=config, device=torch.device("cuda"))
        model.eval()
        return model

    def infer(self, rgb, image_path):
        import torch  # pylint: disable=import-outside-toplevel

        with torch.no_grad():
            output = self.model.infer(self.transform(rgb), f_px=None)
        return output["depth"].float().cpu().numpy(), None

    def metrics_extra(self, depth_only=True, images_per_run=None):
        del depth_only, images_per_run  # Always one image per call.
        return {"f_px": "predicted (FoV head)"}
