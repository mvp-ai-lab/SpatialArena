# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run SenseNova-Vision dense-perception depth output for benchmark evaluation.

The model generates a grayscale relative-depth image per frame
(``SenseNovaVisionModel.generate(mode="dense_perception")``); depth is one
minus its brightness, restored to the source image by a plain resize.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

METHOD = "SENSENOVA_VISION"
DESCRIPTION = ("Run SenseNova-Vision dense-perception depth output for "
               "benchmark evaluation.")
PROMPT = (
    "<image> Estimate relative depth for each pixel in the image, with closer "
    "objects appearing brighter and distant objects appearing darker. Output "
    "is a grayscale image with pixel values ranging from 0-255.")


def relative_depth_from_output(image: Image.Image) -> np.ndarray:
    """Returns the relative depth of a generated grayscale image.

    Args:
        image: The generated image (brighter is closer).

    Returns:
        1 - brightness in [0, 1], as float32.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    depth = 1.0 - gray
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32, copy=False)


@registry.register_model
class SenseNovaVision(base.BaseModel):
    """SenseNova-Vision, dense-perception depth.

    One image per generate() call; the model sizes its input itself, so
    there is no --resolution or dense-point option.
    """

    name = METHOD
    aliases = ("SENSENOVA", "SENSENOVA-VISION", "SENSE_NOVA_VISION")
    description = DESCRIPTION
    code_packages = ("inference", "modeling", "data")
    uses_resolution = False
    uses_image_load_resolution = False
    exports_points = False

    def add_code_path(self):
        """Puts --code_path on sys.path."""
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        utils.add_code_path(self.args.code_path, self.code_packages)

    def select_device(self):
        """Returns "cuda" with visible GPUs, else "auto" (the model's own)."""
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            return "cuda"
        return "auto"

    def load(self, device):
        # pylint: disable-next=import-outside-toplevel
        from inference import sensenova_vision

        return sensenova_vision.SenseNovaVisionModel(
            model_path=self.args.ckpt_path, device=device, dtype="bf16")

    def preprocess(self, image_paths, batch_index=None):
        """Probes the one image of a call; the model loads it itself.

        Args:
            image_paths: The one image of the call.
            batch_index: Unused.

        Returns:
            The `base.ModelInput`, with the source (width, height) as
            extra["source_wh"].

        Raises:
            ValueError: The call has more than one image.
        """
        del batch_index  # One image per call.
        if len(image_paths) != 1:
            raise ValueError(f"{self.name} predicts one image per call")
        with Image.open(image_paths[0]) as probe:
            source_wh = probe.size
        return base.ModelInput(list(image_paths),
                               extra={"source_wh": source_wh})

    def forward(self, inputs):
        """Generates the depth image and restores it to the source image.

        Args:
            inputs: The output of `preprocess`.

        Returns:
            The `base.Prediction`, depth (1, H, W) already on the source
            grid (the input has no transforms).

        Raises:
            RuntimeError: The model did not return an image.
        """
        image_path = inputs.image_paths[0]
        result = self.model.generate(
            question=PROMPT,
            images=[image_path],
            mode="dense_perception",
            return_intermediate_outputs=True,
        )
        output_image = result.get("image") if isinstance(result,
                                                         dict) else result
        if not isinstance(output_image, Image.Image):
            raise RuntimeError(
                f"SenseNova-Vision did not return an image for {image_path}")
        depth = relative_depth_from_output(output_image)
        # ImageTransform's MaxLongEdgeMinShortEdgeResize (data/transforms.py)
        # is a pure resize -- no crop, no pad -- so restoring this generated
        # depth map back to the image's true source resolution is the
        # plain-resize degenerate case restore_prediction already handles
        # (same as depthvlm.py/point_query_depth_adapter.py).
        transform = utils.SpatialTransform(source_wh=inputs.extra["source_wh"],
                                           resized_wh=output_image.size)
        restored = utils.restore_prediction(depth, transform)
        return base.Prediction(depth=restored[None])

    def run(self) -> None:
        """Generates each selected image's depth and writes depths/.

        Raises:
            ValueError: Bad options or no images.
            RuntimeError: The model did not return an image.
        """
        args = self.args
        if args.export_colmap and not args.depth_only:
            raise ValueError(
                "SenseNova-Vision depth adapter only exports dense depth maps.")

        self.images_dir = utils.canonical_image_root(args.data_path)
        self.image_paths = utils.select_interval(
            utils.find_image_paths(self.images_dir), args.interval)
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")
        if args.dry_run:
            utils.print_dry_run(args, METHOD, len(self.image_paths))
            return
        if not args.export_depth:
            raise ValueError(
                "SenseNova-Vision depth adapter requires --export_depth.")

        self.add_code_path()
        self.device = self.select_device()
        self.model = self.load(self.device)

        args.output_path.mkdir(parents=True, exist_ok=True)
        started_at = utils.mark_model_loaded()
        for image_path in self.image_paths:
            prediction, inputs = self.predict([image_path])
            utils.export_depth_predictions(args.output_path, inputs.image_paths,
                                           prediction.depth)

        utils.write_adapter_metrics(
            output_path=args.output_path,
            method=METHOD,
            image_paths=self.image_paths,
            started_at=started_at,
            checkpoint=args.ckpt_path,
            depth_only=True,
            metrics_extra={
                "track":
                    args.track,
                "adapter":
                    "SenseNovaVisionModel.generate(mode=dense_perception)",
                "depth_type":
                    "relative_depth_from_generated_grayscale",
            },
        )
