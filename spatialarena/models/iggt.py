# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""IGGT (lifuguan/IGGT_official).

IGGT's own `load_and_preprocess_images(mode="resize")` stretches every image
to one square (or, for fixed-resolution inputs, their own size) without
cropping or padding, so depth and intrinsics are restored to each source
image by a plain resize. The whole scene is one model call, also with
--depth_only: --max_images_per_run only caps the number of frames.
"""

from __future__ import annotations

import importlib.util
import logging
import pathlib
import sys

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

LOGGER = logging.getLogger("benchmark_iggt")
CHECKPOINT_SUFFIXES = (".pth", ".pt", ".bin", ".safetensors")


def extrinsics_3x4_to_4x4(extrinsics):
    """Returns IGGT's (1, N, 3, 4) or (N, 3|4, 4) extrinsics as (N, 4, 4).

    Raises:
        ValueError: The extrinsics have another shape.
    """
    if extrinsics.ndim == 4 and extrinsics.shape[0] == 1:
        extrinsics = extrinsics[0]
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics.astype(np.float32, copy=False)
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(
            f"Expected IGGT extrinsics Nx3x4 or Nx4x4, got {extrinsics.shape}")
    mats = np.repeat(np.eye(4, dtype=np.float32)[None],
                     extrinsics.shape[0],
                     axis=0)
    mats[:, :3, :] = extrinsics.astype(np.float32, copy=False)
    return mats


def resized_colors(image_paths, height, width):
    """Returns the images resized to (width, height), as uint8 (N, H, W, 3)."""
    # pylint: disable-next=import-outside-toplevel
    from PIL import Image, ImageOps

    colors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB").resize(
                (width, height), Image.Resampling.BICUBIC)
            colors.append(np.asarray(image, dtype=np.uint8))
    return np.stack(colors, axis=0)


def candidate_checkpoint_files(root):
    """Yields the checkpoint files under `root`, by suffix, then sorted."""
    for suffix in CHECKPOINT_SUFFIXES:
        yield from sorted(root.rglob(f"*{suffix}"))


def resolve_checkpoint_path(checkpoint):
    """Resolves a local checkpoint file, local directory, or HF repo id.

    Args:
        checkpoint: --ckpt_path.

    Returns:
        The checkpoint file.

    Raises:
        FileNotFoundError: The directory or snapshot has no checkpoint.
        RuntimeError: huggingface_hub is needed but not installed.
    """
    path = pathlib.Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in candidate_checkpoint_files(path):
            return candidate
        raise FileNotFoundError(f"No checkpoint file found under {path}")

    try:
        import huggingface_hub  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            f"IGGT checkpoint {checkpoint!r} is not a local file/directory and "
            "huggingface_hub is unavailable for HF repo resolution.") from exc

    snapshot_dir = pathlib.Path(
        huggingface_hub.snapshot_download(repo_id=checkpoint,
                                          repo_type="model"))
    for candidate in candidate_checkpoint_files(snapshot_dir):
        return candidate
    raise FileNotFoundError(
        f"No checkpoint file found in downloaded HF snapshot: {snapshot_dir}")


def load_checkpoint_state_dict(checkpoint_path, device):
    """Loads a .safetensors or torch checkpoint onto `device`.

    Raises:
        RuntimeError: safetensors is needed but not installed.
    """
    import torch  # pylint: disable=import-outside-toplevel

    if checkpoint_path.suffix == ".safetensors":
        try:
            # pylint: disable-next=import-outside-toplevel
            from safetensors import torch as safetensors_torch
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required to load IGGT safetensors checkpoints"
            ) from exc
        return safetensors_torch.load_file(str(checkpoint_path), device=device)
    return torch.load(str(checkpoint_path),
                      map_location=device,
                      weights_only=False)


def load_iggt_state_dict_aligner(code_path):
    """Returns `align_and_update_state_dicts` of the checkout's utils/model.py.

    Raises:
        FileNotFoundError: The checkout has no utils/model.py.
        ImportError: It cannot be loaded.
    """
    module_path = code_path / "utils" / "model.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"IGGT state-dict helper is missing: {module_path}")
    spec = importlib.util.spec_from_file_location("benchmark_iggt_utils_model",
                                                  module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load IGGT state-dict helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.align_and_update_state_dicts


def _autocast(device):
    """Returns IGGT's mixed-precision context for `device`."""
    import torch  # pylint: disable=import-outside-toplevel

    if device != "cuda":
        return torch.no_grad()
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


@registry.register_model
class IGGT(base.BaseModel):
    """IGGT."""

    name = "IGGT"
    description = "Run IGGT and export benchmark prediction artifacts."
    code_packages = ("iggt",)
    code_path_env = "IGGT_CODE_PATH"
    ckpt_env = "IGGT_CKPT_PATH"
    uses_resolution = False
    default_image_load_resolution = 504

    def __init__(self, args):
        super().__init__(args)
        self.checkpoint_path = None  # Resolved by load().
        self.single_view = False

    @property
    def checkpoint(self):
        return str(self.checkpoint_path)

    def add_code_path(self):
        """Puts the checkout and its nested iggt/ first on sys.path."""
        utils.add_code_path(self.args.code_path, self.code_packages)
        code_path = self.args.code_path.expanduser().resolve()
        for candidate in (code_path / "iggt", code_path):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        from iggt.models import vggt as iggt_vggt

        code_path = self.args.code_path.expanduser().resolve()
        align_and_update_state_dicts = load_iggt_state_dict_aligner(code_path)

        model = iggt_vggt.IGGT()
        self.checkpoint_path = resolve_checkpoint_path(self.args.ckpt_path)
        state_dict = load_checkpoint_state_dict(self.checkpoint_path, device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(align_and_update_state_dicts(
            LOGGER, model.state_dict(), state_dict),
                              strict=False)
        if hasattr(model, "part_adaptor"):
            model.part_adaptor = None
        if hasattr(model, "part_head"):
            model.part_head = None
        return model.to(device).eval()

    def preprocess(self, image_paths, batch_index=None):
        # IGGT's own loader; restored by a plain resize.
        # pylint: disable=import-outside-toplevel
        from iggt.utils import load_fn
        from PIL import Image

        size = self.args.image_load_resolution
        images = load_fn.load_and_preprocess_images(
            image_paths,
            mode="resize",
            resize_target_size=utils.fixed_input_size(image_paths) or
            (size, size),
        ).to(self.device)
        resized_wh = (images.shape[-1], images.shape[-2])
        transforms = []
        for image_path in image_paths:
            with Image.open(image_path) as probe:
                source_wh = probe.size
            transforms.append(
                utils.SpatialTransform(source_wh=source_wh,
                                       resized_wh=resized_wh))
        return base.ModelInput(image_paths, images, transforms, batch_index)

    def forward(self, inputs):
        # pylint: disable=import-outside-toplevel
        import torch
        from iggt.utils import pose_enc as iggt_pose_enc

        images = inputs.data
        self.single_view = utils.single_view_mode()
        with torch.no_grad():
            with _autocast(self.device):
                if self.single_view:
                    # MonoDepth: independent views [B, 1, 3, H, W], folded
                    # back to the sequence layout.
                    predictions = utils.fold_single_view_outputs(
                        self.model(images[:, None]), images.shape[0])
                else:
                    predictions = self.model(images)
        extrinsics, intrinsics = iggt_pose_enc.pose_encoding_to_extri_intri(
            predictions["pose_enc"][-1], images.shape[-2:])
        depths = utils.normalize_depths(
            predictions["depth"].detach().float().cpu().numpy().squeeze(0))
        confidence = predictions.get("depth_conf")
        if confidence is not None:
            confidence = confidence.detach().float().cpu().numpy().squeeze()
        return base.Prediction(
            depth=depths,
            confidence=confidence,
            extrinsics_w2c=extrinsics_3x4_to_4x4(
                extrinsics.detach().float().cpu().numpy()),
            intrinsics=intrinsics.detach().float().cpu().numpy().squeeze(
                0).astype(np.float32),
        )

    def postprocess(self, prediction, inputs):
        # A full run adds the dense points and colors, on the model grid.
        if self.args.depth_only:
            return prediction
        # pylint: disable-next=import-outside-toplevel
        from iggt.utils import geometry as iggt_geometry

        depths = prediction.depth
        prediction.points = utils.normalize_points(
            iggt_geometry.unproject_depth_map_to_point_map(
                depths[..., None], prediction.extrinsics_w2c,
                prediction.intrinsics))
        prediction.colors = resized_colors(inputs.image_paths, depths.shape[1],
                                           depths.shape[2])
        return prediction

    def metrics_extra(self, depth_only, images_per_run=None):
        del depth_only, images_per_run  # Same fields in both modes.
        return {"track": self.args.track, "single_view": self.single_view}

    def run(self):
        """Runs IGGT on the scene and writes its artifacts.

        Raises:
            ValueError: No images, or no checkpoint.
        """
        args = self.args
        self.images_dir = utils.canonical_image_root(args.data_path)
        self.image_paths = self.select_images()
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")
        if args.dry_run:
            utils.print_dry_run(args, self.method_label, len(self.image_paths))
            return
        if not args.ckpt_path:
            raise ValueError("IGGT requires --ckpt_path or IGGT_CKPT_PATH.")

        self.add_code_path()
        self.device = self.select_device()
        self.model = self.load(self.device)
        self.started_at = utils.mark_model_loaded()
        if args.depth_only:
            self.run_depth_only()
        else:
            self.run_full()

    def run_depth_only(self):
        """Exports depth and cameras of the one model call, restored."""
        args = self.args
        prediction, inputs = self.predict(self.image_paths)
        if args.export_depth:
            utils.export_spatial_depth_predictions(args.output_path,
                                                   self.image_paths,
                                                   prediction.depth,
                                                   prediction.confidence,
                                                   inputs.transforms)
        if args.export_colmap:
            extrinsics, intrinsics, (width,
                                     height), _ = (self.cameras_for_export(
                                         inputs, prediction))
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
