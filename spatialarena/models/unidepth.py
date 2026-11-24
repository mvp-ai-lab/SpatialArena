# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official UniDepth V2 (lpiccinelli-eth/UniDepth) metric depth.

Each image goes through ``UniDepthV2.infer`` without a camera (rgb track) and
with ``resolution_level`` unset, the documented default (the full training
pixel range); the model pads/resizes internally and returns depth at the input
resolution.
"""

from __future__ import annotations

from spatialarena.models import registry
from spatialarena.models import single_image

# The adapter's command-line description.
DESCRIPTION = __doc__


@registry.register_model
class UniDepthV2(single_image.SingleImageDepthModel):
    """UniDepth V2."""

    name = "UNIDEPTH_V2"
    aliases = ("UNIDEPTH", "UNIDEPTHV2", "UNIDEPTH-V2")
    description = DESCRIPTION
    code_packages = ("unidepth",)

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        from unidepth import models

        del device  # CUDA only.
        torch.manual_seed(self.args.seed)
        return models.UniDepthV2.from_pretrained(
            self.args.ckpt_path).cuda().eval()

    def infer(self, rgb, image_path):
        import torch  # pylint: disable=import-outside-toplevel

        with torch.no_grad():
            output = self.model.infer(torch.from_numpy(rgb).permute(2, 0, 1))
        return output["depth"].float().cpu().numpy(), None

    def metrics_extra(self, depth_only=True, images_per_run=None):
        del depth_only, images_per_run  # Always one image per call.
        return {"camera": None}
