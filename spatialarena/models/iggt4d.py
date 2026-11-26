# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""IGGT4D (HorizonRobotics/IGGT4D).

IGGT4D's public API (`iggt4d.api.IGGT4D.inference`) follows Depth Anything
3's: images in, a prediction with per-frame `depth` and `conf` at the
processing resolution (`process_res`, longest side, default 504),
world-to-camera `extrinsics` and `intrinsics` out. The instance head is not
needed by the benchmark and is skipped (`run_instance=False`). With camera
priors (the rgb_cameras track) the input poses condition the model and the
prediction is aligned to them, as in the official `--camera` mode.

The checkpoint is a single `model.safetensors` (Hugging Face
`HorizonRobotics/IGGT4D`); upstream loads it from the relative path
`checkpoints/model.safetensors`, so the adapter points that path at
--ckpt_path.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

PATCH_SIZE = 14


def plan_padded_inputs(image_paths, output_path, resolution):
    """Pads every image to one square canvas on disk (the rgb track).

    This bypasses IGGT4D's own cross-image crop (its
    `InputProcessor._unify_batch_shapes`, a batch-dependent center crop
    needed only because its per-image upper_bound_resize can round images to
    slightly different shapes): with a fixed (width, height) process_res that
    every padded image already has, IGGT4D's own resize is a no-op and
    nothing is cropped.

    Args:
        image_paths: Source images.
        output_path: Method scene output directory (the scratch directory
            is created under it).
        resolution: --resolution, rounded up to the patch size.

    Returns:
        An (inference_paths, process_res, resized_dir, transforms) tuple.
    """
    # pylint: disable-next=import-outside-toplevel
    from PIL import Image, ImageOps

    target = ((resolution + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    source_whs = []
    for image_path in image_paths:
        # Transposed like materialize_resized_images does, so a 90/270 degree
        # EXIF orientation does not swap width and height.
        with Image.open(image_path) as probe:
            source_whs.append(ImageOps.exif_transpose(probe).size)
    resized_dir = output_path / "_iggt4d_resized_input_images"
    inference_paths = utils.materialize_resized_images(image_paths, resized_dir,
                                                       target)
    transforms = []
    for source_w, source_h in source_whs:
        scale = min(target / source_w, target / source_h)
        resized_w = max(1, round(source_w * scale))
        resized_h = max(1, round(source_h * scale))
        pad_l, pad_t = (target - resized_w) // 2, (target - resized_h) // 2
        transforms.append(
            utils.SpatialTransform(
                source_wh=(source_w, source_h),
                resized_wh=(resized_w, resized_h),
                padding=(pad_l, pad_t, target - resized_w - pad_l,
                         target - resized_h - pad_t),
            ))
    return inference_paths, (target, target), resized_dir, transforms


@registry.register_model
class IGGT4D(base.BaseModel):
    """IGGT4D, optionally conditioned on COLMAP camera priors."""

    name = "IGGT4D"
    description = "Run the IGGT4D benchmark adapter."
    code_path_env = "IGGT4D_CODE_PATH"
    code_packages = ("iggt4d",)
    ckpt_env = "IGGT4D_CKPT_PATH"
    default_resolution = 504
    uses_image_load_resolution = False
    tracks = ("rgb", "rgb_cameras")

    @property
    def uses_camera_priors(self):
        """Whether the run is conditioned on COLMAP camera priors."""
        return self.args.track == "rgb_cameras"

    def add_code_path(self):
        utils.add_code_path(self.args.code_path, self.code_packages)

    def select_images(self):
        image_paths = utils.find_image_paths(self.images_dir)
        if self.uses_camera_priors:
            image_paths = utils.resolve_camera_conditioned_image_paths(
                self.args.data_path,
                image_paths,
                colmap_path=self.args.colmap_path)
        return utils.select_interval(image_paths, self.args.interval,
                                     self.args.max_images_per_run)

    def load(self, device):
        # Raises FileNotFoundError when --ckpt_path is not a file.
        # pylint: disable-next=import-outside-toplevel
        from iggt4d.utils import model_loading

        checkpoint = pathlib.Path(self.args.ckpt_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"IGGT4D checkpoint not found: {checkpoint} "
                "(HorizonRobotics/IGGT4D model.safetensors)")
        model_loading.DEFAULT_MODEL_CHECKPOINT_PATH = checkpoint
        from iggt4d import api  # pylint: disable=import-outside-toplevel

        model = api.IGGT4D()
        model.device = device
        return model.to(device).eval()

    def preprocess(self, image_paths, batch_index=None):
        # The rgb track pads the images to one canvas. With camera priors
        # they are passed as they are: IGGT4D rescales the prior intrinsics
        # with each image's true source size, which a padded file would
        # misreport. Their maps are exported on the model grid.
        args = self.args
        extra = {"extrinsics": None, "intrinsics": None}
        if self.uses_camera_priors:
            poses_c2w, prior_intrinsics = utils.load_colmap_camera_priors(
                args.data_path,
                image_paths,
                colmap_path=args.colmap_path,
                colmap_model_subdir=args.colmap_model_subdir)
            extra["extrinsics"] = np.linalg.inv(poses_c2w).astype(np.float32)
            extra["intrinsics"] = prior_intrinsics.astype(np.float32)
            extra["process_res"] = args.resolution
            return base.ModelInput(image_paths,
                                   image_paths,
                                   None,
                                   batch_index,
                                   extra=extra)

        inference_paths, process_res, resized_dir, transforms = (
            plan_padded_inputs(image_paths, args.output_path, args.resolution))
        extra["process_res"] = process_res
        return base.ModelInput(image_paths,
                               inference_paths,
                               transforms,
                               batch_index,
                               scratch_dirs=[resized_dir],
                               extra=extra)

    def forward(self, inputs):
        raw = self.model.inference(
            inputs.data,
            extrinsics=inputs.extra["extrinsics"],
            intrinsics=inputs.extra["intrinsics"],
            process_res=inputs.extra["process_res"],
            run_instance=False,
            cluster_instances=False,
        )
        # The padded inputs are not needed after the call.
        utils.cleanup_intermediate_paths(list(inputs.scratch_dirs),
                                         keep=utils.should_keep_intermediate(
                                             self.args))
        depth = np.asarray(raw.depth, dtype=np.float32)
        utils.record_model_input(depth.shape[-1], depth.shape[-2])
        extrinsics_w2c = np.asarray(raw.extrinsics, dtype=np.float32)
        if extrinsics_w2c.shape[-2:] == (3, 4):
            bottom = np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32),
                                     (len(extrinsics_w2c), 1, 4))
            extrinsics_w2c = np.concatenate([extrinsics_w2c, bottom], axis=1)
        confidence = None
        if raw.conf is not None:
            confidence = np.asarray(raw.conf, dtype=np.float32)
        return base.Prediction(
            depth=depth,
            confidence=confidence,
            extrinsics_w2c=extrinsics_w2c,
            intrinsics=np.asarray(raw.intrinsics, dtype=np.float32),
            colors=raw.processed_images,
        )

    def metrics_extra(self, depth_only, images_per_run=None):
        extra = {
            "track": self.args.track,
            "process_res": self.args.resolution,
            "max_images_per_run": self.args.max_images_per_run,
        }
        if depth_only:
            extra["max_images_per_run"] = images_per_run
        return extra

    def run(self):
        """Runs IGGT4D on the scene and writes its artifacts.

        Raises:
            FileNotFoundError: The scene has no images directory.
            ValueError: No images, no checkpoint, or --depth_only without
                --export_depth.
        """
        args = self.args
        self.images_dir = utils.canonical_image_root(args.data_path)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"images directory not found: {self.images_dir}")
        self.image_paths = self.select_images()
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")
        if args.dry_run:
            utils.print_dry_run(args, self.method_label, len(self.image_paths))
            return
        if not args.ckpt_path:
            raise ValueError("IGGT4D requires --ckpt_path or IGGT4D_CKPT_PATH")

        self.device = self.select_device()
        # "inference time" includes loading the model.
        self.started_at = time.time()
        self.add_code_path()
        self.model = self.load(self.device)
        utils.mark_model_loaded()
        if args.depth_only:
            if not args.export_depth:
                raise ValueError("--depth_only requires --export_depth")
            self.run_depth_only()
        else:
            self.run_full()
