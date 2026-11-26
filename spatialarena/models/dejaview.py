# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Déjà View (DVLT, nvidia/dvlt).

The images are resized (longest side --image_load_resolution) and padded,
not cropped, to one patch-aligned canvas per chunk, so depth and intrinsics
of a depth-only run are restored to each source image. The scene is
predicted in chunks of --max_images_per_run images (one image per chunk in
single-view mode); --max_images_per_run also caps the number of frames, so
there is one chunk otherwise.
"""

from __future__ import annotations

import os

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

PATCH_SIZE = 14


def to_4x4_camera_to_world(camera_to_worlds):
    """Returns DVLT camera-to-world matrices as a float32 (S, 4, 4) array.

    Raises:
        ValueError: The matrices are neither 3x4 nor 4x4.
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from dvlt.common import pose

    c2w = camera_to_worlds
    if c2w.shape[-2:] == (3, 4):
        c2w = pose.to4x4(c2w)
    if c2w.shape[-2:] != (4, 4):
        raise ValueError("Expected DVLT camera_to_worlds with shape Sx3x4 or "
                         f"Sx4x4, got {tuple(c2w.shape)}")
    if isinstance(c2w, torch.Tensor):
        return c2w.detach().float().cpu().numpy()
    return utils.as_numpy(c2w).astype("float32", copy=False)


def preprocess_dejaview(image_paths, image_load_resolution, device):
    """Builds DVLT's batch dict without its lossy per-image crop.

    DVLT's `dvlt.util.preprocess.preprocess_images` resizes each image's
    longest side to img_size, center-crops both sides down to a multiple of
    the patch size, then pads the batch to one canvas. Here each image is
    resized the same way and padded (not cropped) straight to the batch's
    patch-aligned canvas: the max of each image's size rounded up.

    Args:
        image_paths: Source images of one chunk.
        image_load_resolution: The longest resized side.
        device: Device of the batch.

    Returns:
        A (batch, transforms) tuple: DVLT's batch dict (images and
        "gradio_valid_pixels") and the per-image transforms.
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from torch.nn import functional as F
    from PIL import Image, ImageOps
    from dvlt.common import constants

    resized_images = []
    source_and_resized = []
    ceil_dims = []
    for path in image_paths:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        source_w, source_h = img.size
        scale = image_load_resolution / max(source_h, source_w)
        new_h, new_w = max(1, round(source_h * scale)), max(
            1, round(source_w * scale))
        resized_images.append(
            img.resize((new_w, new_h), Image.Resampling.LANCZOS))
        source_and_resized.append((source_w, source_h, new_w, new_h))
        ceil_dims.append(
            (((new_w + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE,
             ((new_h + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE))

    max_w = max(ceil_w for ceil_w, _ in ceil_dims)
    max_h = max(ceil_h for _, ceil_h in ceil_dims)

    aligned = []
    valid_masks = []
    transforms = []
    for resized, (source_w, source_h, new_w,
                  new_h) in zip(resized_images, source_and_resized):
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0,
                                                             1).float() / 255.0
        pad_w, pad_h = max_w - new_w, max_h - new_h
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        aligned.append(
            F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom),
                  value=0.0))
        valid = F.pad(torch.ones(1, new_h, new_w, dtype=torch.float32),
                      (pad_left, pad_right, pad_top, pad_bottom),
                      value=0.0)
        valid_masks.append(valid.squeeze(0).bool())
        transforms.append(
            utils.SpatialTransform(
                source_wh=(source_w, source_h),
                resized_wh=(new_w, new_h),
                padding=(pad_left, pad_top, pad_right, pad_bottom),
            ))

    images = torch.stack(aligned, dim=0).unsqueeze(0).to(device)
    gradio_valid_pixels = torch.stack(valid_masks,
                                      dim=0).unsqueeze(0).to(device)
    return {
        constants.DataField.IMAGES: images,
        "gradio_valid_pixels": gradio_valid_pixels
    }, transforms


@registry.register_model
class DejaView(base.BaseModel):
    """Déjà View (DVLT)."""

    name = "DEJAVIEW"
    aliases = ("DEJA_VIEW", "DVLT")
    description = (
        "Run Déjà View / DVLT and export benchmark prediction artifacts.")
    code_path_env = "DEJAVIEW_CODE_PATH"
    code_packages = ("dvlt",)
    ckpt_env = "DEJAVIEW_CKPT_PATH"
    ckpt_default = "nvidia/dvlt"
    uses_resolution = False
    default_image_load_resolution = 504

    def __init__(self, args):
        super().__init__(args)
        self.accelerator = None

    def add_code_path(self):
        """Puts the checkout, and its src/ if any, on sys.path."""
        code_path = self.args.code_path
        utils.add_code_path(code_path, self.code_packages)
        source_dir = code_path / "src"
        if source_dir.is_dir():
            utils.add_code_path(source_dir, self.code_packages)

    def load(self, device):
        del device  # Accelerate picks the device.
        import torch  # pylint: disable=import-outside-toplevel

        if os.environ.get("DEJAVIEW_DISABLE_TORCH_COMPILE", "1") == "1":
            torch.compile = lambda fn, *args, **kwargs: fn
        # pylint: disable=import-outside-toplevel
        import accelerate
        from dvlt.model.dvlt import model as dvlt_model

        self.accelerator = accelerate.Accelerator(
            mixed_precision="bf16" if torch.cuda.is_available() else "no")
        model = dvlt_model.DVLT(img_size=self.args.image_load_resolution)
        model.load_pretrained(self.args.ckpt_path, strict=True)
        model.setup_test(self.accelerator)
        self.device = self.accelerator.device
        return model

    def preprocess(self, image_paths, batch_index=None):
        batch, transforms = preprocess_dejaview(image_paths,
                                                self.args.image_load_resolution,
                                                self.device)
        return base.ModelInput(image_paths, batch, transforms, batch_index)

    def forward(self, inputs):
        import torch  # pylint: disable=import-outside-toplevel

        batch = inputs.data
        with torch.no_grad(), self.accelerator.autocast():
            predictions = self.model.predict(batch, self.accelerator)

        prediction = base.Prediction(depth=utils.as_numpy(
            predictions["depths"][0]).astype("float32", copy=False))
        confidence = predictions.get("depth_conf", None)
        if confidence is None:
            confidence = predictions.get("depths_conf", None)
        if isinstance(confidence, (list, tuple)):
            confidence = confidence[0]
        if confidence is not None:
            confidence = utils.as_numpy(confidence).squeeze()
            # A one-image chunk squeezes to (H, W); keep the frame axis so
            # chunks concatenate per frame.
            if confidence.ndim == 2:
                confidence = confidence[None]
            prediction.confidence = confidence

        if self.args.export_colmap:
            cameras = predictions["cameras"][0]
            c2w = to_4x4_camera_to_world(cameras.camera_to_worlds)
            prediction.extrinsics_w2c = utils.c2w_to_w2c(c2w)
            prediction.intrinsics = utils.as_numpy(
                cameras.get_intrinsics_matrices()).astype("float32", copy=False)
        if not self.args.depth_only:
            prediction.points = utils.as_numpy(
                predictions["world_points"][0]).astype("float32", copy=False)
            prediction.colors = utils.colors_from_torch_images(batch["images"])
        return prediction

    def predict(self, image_paths, batch_index=None):
        # Predicts the images in chunks and concatenates the chunks.
        chunk_size = self.args.max_images_per_run
        if chunk_size <= 0:
            chunk_size = len(image_paths)
        if utils.single_view_mode():
            chunk_size = 1  # MonoDepth: one image per forward pass.
        fields = ("depth", "confidence", "extrinsics_w2c", "intrinsics",
                  "points", "colors")
        chunks = {field: [] for field in fields}
        transforms = []
        for start in range(0, len(image_paths), chunk_size):
            inputs = self.preprocess(image_paths[start:start + chunk_size])
            transforms.extend(inputs.transforms)
            prediction = self.forward(inputs)
            for field in fields:
                value = getattr(prediction, field)
                if value is not None:
                    chunks[field].append(value)

        merged = base.Prediction(
            **{
                field: np.concatenate(values, axis=0) if values else None
                for field, values in chunks.items()
            })
        return merged, base.ModelInput(image_paths,
                                       transforms=transforms,
                                       batch_index=batch_index)

    def metrics_extra(self, depth_only, images_per_run=None):
        del depth_only, images_per_run  # Same fields in both modes.
        return {
            "track": self.args.track,
            "adapter": "dvlt.model.dvlt.model.DVLT"
        }

    def run(self):
        """Runs DVLT on the scene and writes its artifacts.

        Raises:
            ValueError: No images were found.
        """
        args = self.args
        self.images_dir = utils.canonical_image_root(args.data_path)
        # --max_images_per_run also caps the frames: chunks are independent
        # predictions in their own coordinate frames, simply concatenated,
        # so capping the frames first makes it one chunk.
        self.image_paths = self.select_images()
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")
        if args.dry_run:
            utils.print_dry_run(args, self.method_label, len(self.image_paths))
            return

        self.add_code_path()
        self.model = self.load(self.device)
        self.started_at = utils.mark_model_loaded()
        if args.depth_only:
            self.run_depth_only()
        else:
            self.run_full()

    def run_depth_only(self):
        """Exports depth and cameras of all chunks, restored."""
        args = self.args
        prediction, inputs = self.predict(self.image_paths)
        if args.export_depth:
            utils.export_spatial_depth_predictions(args.output_path,
                                                   self.image_paths,
                                                   prediction.depth,
                                                   prediction.confidence,
                                                   inputs.transforms)
        cameras = self.cameras_for_export(inputs, prediction)
        if args.export_colmap and cameras is not None:
            extrinsics, intrinsics, (width, height), _ = cameras
            utils.export_camera_pose_priors(
                args.output_path,
                self.images_dir,
                self.image_paths,
                extrinsics,
                intrinsics,
                width=width,
                height=height,
                colmap_model_subdir=args.colmap_model_subdir,
            )
        self.write_metrics(depth_only=True,
                           metrics_extra=self.metrics_extra(depth_only=True))
