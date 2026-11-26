# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official Marigold depth (prs-eth): v1-0 (MARIGOLD) and v1-1 (MARIGOLD_V1_1),
via diffusers' MarigoldDepthPipeline.

Each image is predicted on its own with the checkpoint's own default denoising
steps and processing resolution, one ensemble member, fp32 and a fixed seed,
and returned at the input resolution (``match_input_resolution``). The
prediction is affine-invariant depth in [0, 1].
"""

from __future__ import annotations

import os

from spatialarena.models import registry
from spatialarena.models import single_image

# The adapter's command-line description.
DESCRIPTION = __doc__


def processing_options() -> dict[str, int]:
    """Returns the pipeline's processing_resolution option.

    It comes from MARIGOLD_PROCESSING_RES (0: the input's own size); unset
    keeps the checkpoint's default (768).

    Returns:
        {} or {"processing_resolution": value}.

    Raises:
        ValueError: The value is negative.
    """
    resolution = os.environ.get("MARIGOLD_PROCESSING_RES")
    kwargs = {} if resolution is None else {
        "processing_resolution": int(resolution)
    }
    if kwargs.get("processing_resolution", 0) < 0:
        raise ValueError("MARIGOLD_PROCESSING_RES must be nonnegative")
    return kwargs


@registry.register_model
class Marigold(single_image.SingleImageDepthModel):
    """Marigold v1-0."""

    # The pipeline comes from the installed diffusers: no code checkout.
    uses_code_path = False

    name = "MARIGOLD"
    description = DESCRIPTION

    def __init__(self, args):
        super().__init__(args)
        self.processing_kwargs = {}

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        import diffusers

        del device  # CUDA only.
        pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
            self.args.ckpt_path, torch_dtype=torch.float32).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self.processing_kwargs = processing_options()
        return pipe

    def infer(self, rgb, image_path):
        # pylint: disable=import-outside-toplevel
        import torch
        from PIL import Image

        generator = torch.Generator("cuda").manual_seed(self.args.seed)
        output = self.model(Image.fromarray(rgb),
                            ensemble_size=1,
                            match_input_resolution=True,
                            generator=generator,
                            **self.processing_kwargs)
        return output.prediction, None

    def metrics_extra(self, depth_only=True, images_per_run=None):
        del depth_only, images_per_run  # Always one image per call.
        return {
            "denoising_steps":
                self.model.default_denoising_steps,
            "processing_resolution":
                self.processing_kwargs.get(
                    "processing_resolution",
                    self.model.default_processing_resolution),
            "ensemble_size":
                1,
        }


@registry.register_model
class MarigoldV11(Marigold):
    """Marigold v1-1."""

    name = "MARIGOLD_V1_1"
    aliases = ("MARIGOLD-V1-1", "MARIGOLD_V1-1")
