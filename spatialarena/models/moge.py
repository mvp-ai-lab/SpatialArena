# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official MoGe / MoGe-2 / MoGe-3 single-image depth (microsoft/MoGe).

Each image goes through ``MoGeModel.infer`` at its own resolution (the model
resizes internally and returns depth on the input grid), with
``resolution_level=9``, fp32 and ``apply_mask=False`` so every pixel keeps a
depth (as in the official evaluation); no GT intrinsics are given. MoGe
predicts affine-invariant geometry; MoGe-2/-3 metric.
"""

from __future__ import annotations

import importlib

from spatialarena.models import registry
from spatialarena.models import single_image

# The adapter's command-line description.
DESCRIPTION = __doc__
# Method -> MoGe code version (moge.model.<version>).
VERSIONS = {"MOGE": "v1", "MOGE2": "v2", "MOGE3": "v3"}


@registry.register_model
class MoGe(single_image.SingleImageDepthModel):
    """MoGe-1; MOGE2 and MOGE3 are the later versions."""

    name = "MOGE"
    aliases = ("MOGE1", "MOGE-1")
    description = DESCRIPTION
    code_packages = ("moge",)

    def load(self, device):
        import torch  # pylint: disable=import-outside-toplevel

        del device  # CUDA only.
        torch.manual_seed(self.args.seed)
        model_class = importlib.import_module(
            f"moge.model.{VERSIONS[self.method_name]}").MoGeModel
        return model_class.from_pretrained(self.args.ckpt_path).cuda().eval()

    def infer(self, rgb, image_path):
        import torch  # pylint: disable=import-outside-toplevel

        image = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda() / 255.0
        with torch.inference_mode():
            output = self.model.infer(image,
                                      resolution_level=9,
                                      apply_mask=False,
                                      use_fp16=False)
        return output["depth"].float().cpu().numpy(), None

    def metrics_extra(self, depth_only=True, images_per_run=None):
        del depth_only, images_per_run  # Always one image per call.
        return {
            "moge_version": VERSIONS[self.method_name],
            "resolution_level": 9,
            "apply_mask": False,
            "use_fp16": False,
        }


@registry.register_model
class MoGe2(MoGe):
    """MoGe-2."""

    name = "MOGE2"
    aliases = ("MOGE-2",)


@registry.register_model
class MoGe3(MoGe):
    """MoGe-3."""

    name = "MOGE3"
    aliases = ("MOGE-3",)
