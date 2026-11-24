# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""VGGT-Omega: VGGT with 16-pixel patches and an optional text head.

--spatial_policy (see `preprocess_vggt_omega`):
    legacy (default): VGGT-Omega's own loader on images pre-padded to disk.
    legacy-pad: the same, for dense/camera export only.
    shared-pad, shared-rect-pad, shared-resize: the benchmark's resize, with
        depth and intrinsics restored to each source image (depth-only).
"""

from __future__ import annotations

import os
import pathlib

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

SHARED_POLICIES = ("shared-pad", "shared-rect-pad", "shared-resize")
PRETRAINED_DIR = "~/Projects/FoundationModels/pretrained"
TEXT_CKPT_NAME = "vggt_omega_1b_256_text.pt"


def resolve_internal_resolution(args):
    """Returns the model resolution: --internal_resolution, or the nearest
    patch-compatible size at or above --resolution."""
    if args.internal_resolution > 0:
        return args.internal_resolution
    return utils.patch_grid_compatible_resolution(args.resolution,
                                                  patch_size=args.patch_size)


def preprocess_vggt_omega(image_paths,
                          args,
                          device,
                          load_and_preprocess_images=None,
                          internal_resolution=None,
                          batch_suffix=""):
    """Builds VGGT-Omega's input.

    shared-*: a restorable tensor built directly from the source images.
    legacy on fixed-size (SLAM protocol) inputs: only center-cropped to the
        16-pixel patch grid, never rescaled.
    legacy otherwise, and legacy-pad: the images are padded to a square on
        disk, then resized by VGGT-Omega's own loader. For legacy-pad that
        resize is a no-op, since --image_load_resolution must equal the
        internal resolution.

    Args:
        image_paths: Source images.
        args: The parsed options.
        device: Device of the returned tensor.
        load_and_preprocess_images: VGGT-Omega's own loader.
        internal_resolution: The model resolution.
        batch_suffix: Keeps each depth-only batch's scratch directory apart.

    Returns:
        A ({"images": tensor, "resized_dir": path or None}, transforms)
        tuple; transforms is None for the legacy policies.
    """
    if args.spatial_policy not in ("legacy", "legacy-pad"):
        images, transforms = utils.preprocess_images(
            image_paths,
            method="vggt_omega",
            size=args.image_load_resolution,
            mode=args.spatial_policy.removeprefix("shared-"),
        )
        return {"images": images.to(device), "resized_dir": None}, transforms

    fixed = (args.spatial_policy == "legacy" and
             utils.fixed_input_size(image_paths))
    if fixed:
        images = utils.load_patch_grid_tensor(image_paths, args.patch_size)
        return {"images": images.to(device), "resized_dir": None}, None

    resized_paths = utils.materialize_resized_images(
        image_paths,
        args.output_path / f"_resized_input_images{batch_suffix}",
        args.image_load_resolution,
    )
    images = load_and_preprocess_images(
        resized_paths,
        mode=args.preprocess_mode,
        image_resolution=internal_resolution,
        patch_size=args.patch_size,
    )
    resized_dir = pathlib.Path(resized_paths[0]).parent
    return {"images": images.to(device), "resized_dir": resized_dir}, None


@registry.register_model
class VGGTOmega(base.BaseModel):
    """VGGT-Omega."""

    name = "VGGT_OMEGA"
    description = "Run VGGT-Omega benchmark adapter."
    code_path_env = "VGGT_OMEGA_CODE_PATH"
    code_packages = ("vggt_omega",)
    ckpt_env = "VGGT_OMEGA_CKPT_PATH"
    ckpt_default = os.path.expanduser(f"{PRETRAINED_DIR}/vggt_omega_1b_512.pt")
    default_resolution = 512
    default_image_load_resolution = 512
    spatial_policies = ("legacy", "legacy-pad", *SHARED_POLICIES)
    batch_size_from_env = True
    requires_cuda = True

    @classmethod
    def add_arguments(cls, parser):
        # The text-aligned checkpoint sits next to the main one.
        main_ckpt = cls.spec_default("ckpt_default", cls.ckpt_default)
        text_ckpt = str(pathlib.Path(main_ckpt).with_name(TEXT_CKPT_NAME))
        parser.add_argument(
            "--text_ckpt_path",
            type=str,
            default=os.environ.get("VGGT_OMEGA_TEXT_CKPT_PATH") or text_ckpt,
            help=("Checkpoint of the text-aligned model, used with "
                  "--enable_alignment (default: $VGGT_OMEGA_TEXT_CKPT_PATH, "
                  "else next to the default checkpoint)."),
        )
        parser.add_argument(
            "--internal_resolution",
            type=int,
            default=0,
            help="Model resolution (0: nearest patch-compatible size).",
        )
        parser.add_argument("--patch_size",
                            type=int,
                            default=16,
                            help="Patch size of the network.")
        parser.add_argument("--preprocess_mode",
                            choices=("balanced", "max_size"),
                            default="balanced",
                            help="Mode of VGGT-Omega's own image loader.")
        parser.add_argument(
            "--enable_alignment",
            action="store_true",
            help="Run the text-aligned model (--text_ckpt_path).")

    def __init__(self, args):
        super().__init__(args)
        self.internal_resolution = resolve_internal_resolution(args)
        # VGGT-Omega's own loader and pose decoder, imported by load().
        self.load_and_preprocess_images = None
        self.encoding_to_camera = None

    @property
    def checkpoint(self):
        if self.args.enable_alignment:
            path = self.args.text_ckpt_path
        else:
            path = self.args.ckpt_path
        return str(pathlib.Path(path).expanduser())

    def validate_args(self):
        super().validate_args()
        if self.args.spatial_policy != "legacy-pad":
            return
        if self.args.depth_only:
            raise ValueError("legacy-pad is for camera geometry export; use "
                             "shared-pad for depth evaluation")
        if self.args.image_load_resolution != self.internal_resolution:
            raise ValueError("legacy-pad requires matching "
                             "image_load_resolution and internal resolution")

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        from vggt_omega import models as omega_models
        from vggt_omega.utils import load_fn
        from vggt_omega.utils import pose_enc

        self.load_and_preprocess_images = load_fn.load_and_preprocess_images
        self.encoding_to_camera = pose_enc.encoding_to_camera

        ckpt_path = pathlib.Path(self.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"VGGT-Omega checkpoint not found: {ckpt_path}. Set "
                "VGGT_OMEGA_CKPT_PATH/VGGT_OMEGA_TEXT_CKPT_PATH or run the "
                "download script.")
        model = omega_models.VGGTOmega(
            enable_alignment=self.args.enable_alignment)
        model = model.to(device).eval()
        state = torch.load(str(ckpt_path), map_location="cpu")
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"VGGT-Omega checkpoint loaded with "
                  f"{len(result.missing_keys)} missing keys and "
                  f"{len(result.unexpected_keys)} unexpected keys.")
        return model

    def preprocess(self, image_paths, batch_index=None):
        suffix = ""
        if batch_index is not None:
            suffix = f"_batch_{batch_index:04d}"
        data, transforms = preprocess_vggt_omega(
            image_paths,
            args=self.args,
            device=self.device,
            load_and_preprocess_images=self.load_and_preprocess_images,
            internal_resolution=self.internal_resolution,
            batch_suffix=suffix,
        )
        scratch_dirs = []
        if data["resized_dir"] is not None:
            scratch_dirs.append(data["resized_dir"])
        return base.ModelInput(
            image_paths=image_paths,
            data=data["images"],
            transforms=transforms,
            batch_index=batch_index,
            scratch_dirs=scratch_dirs,
        )

    def forward(self, inputs):
        import torch  # pylint: disable=import-outside-toplevel

        with torch.inference_mode():
            predictions = self.model(inputs.data)
        extrinsic, intrinsic = self.encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:])
        extrinsic = extrinsic[0].detach().float().cpu().numpy()
        intrinsic = intrinsic[0].detach().float().cpu().numpy()
        depths = predictions["depth"][0].detach().float().cpu().numpy()
        depth_conf = predictions["depth_conf"][0].detach().float().cpu()
        return base.Prediction(
            depth=depths,
            confidence=depth_conf.numpy(),
            extrinsics_w2c=extrinsic,
            intrinsics=intrinsic,
            points=utils.unproject_depths_to_world(depths, extrinsic,
                                                   intrinsic),
            colors=utils.colors_from_torch_images(predictions["images"]),
        )

    def metrics_extra(self, depth_only, images_per_run=None):
        extra = super().metrics_extra(depth_only, images_per_run)
        extra["enable_alignment"] = self.args.enable_alignment
        extra["preprocess_mode"] = self.args.preprocess_mode
        extra["image_load_resolution"] = self.args.image_load_resolution
        extra["internal_resolution"] = self.internal_resolution
        extra["patch_size"] = self.args.patch_size
        extra["resized_input"] = True
        return extra
