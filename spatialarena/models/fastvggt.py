# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""FastVGGT: VGGT with token merging, from its own fork of `vggt`.

--spatial_policy legacy (default): the images are padded to a square on disk
at --image_load_resolution, then reopened and resized to --resolution; the
depth is exported on that grid. --spatial_policy shared: one in-memory
pad-to-square resize to --resolution; depth and intrinsics are restored to
each source image.
"""

from __future__ import annotations

import contextlib
import inspect
import pathlib

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils


def _load_resized_images(image_paths, resolution):
    """Loads images resized to a `resolution` square, as a [N, 3, R, R]."""
    # pylint: disable=import-outside-toplevel
    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF

    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((resolution, resolution),
                                                Image.Resampling.BICUBIC)
            tensors.append(TF.to_tensor(image))
    if not tensors:
        raise ValueError("FastVGGT did not load any images")
    return torch.stack(tensors, dim=0)


def _autocast(device):
    """Returns FastVGGT's mixed-precision context for `device`."""
    import torch  # pylint: disable=import-outside-toplevel

    if device != "cuda":
        return contextlib.nullcontext()
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def run_fastvggt(model, image_paths, args, device, images=None):
    """Runs FastVGGT.

    Args:
        model: The FastVGGT network.
        image_paths: Images to load when `images` is None (legacy policy).
        args: The parsed options.
        device: "cuda" or "cpu".
        images: A preprocessed [N, 3, H, W] tensor (shared policy), or None.

    Returns:
        An (extrinsic, intrinsic, depth, depth_conf, points, colors) tuple of
        numpy arrays.
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from vggt.utils import pose_enc

    if images is None:
        images = _load_resized_images(image_paths, args.resolution)
    utils.record_model_input(images.shape[-1], images.shape[-2])
    if hasattr(model, "update_patch_dimensions"):
        model.update_patch_dimensions(images.shape[-1] // 14,
                                      images.shape[-2] // 14)
    images = images.to(device)

    forward_kwargs = {}
    if "image_paths" in inspect.signature(model.forward).parameters:
        forward_kwargs["image_paths"] = [
            pathlib.Path(path).name for path in image_paths
        ]
    with torch.no_grad(), _autocast(device):
        predictions = model(images, **forward_kwargs)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    extrinsic, intrinsic = pose_enc.pose_encoding_to_extri_intri(
        predictions["pose_enc"], (images.shape[-2], images.shape[-1]))
    extrinsic = extrinsic[0].detach().float().cpu().numpy()
    intrinsic = intrinsic[0].detach().float().cpu().numpy()
    depths = predictions["depth"][0].detach().float().cpu().numpy()
    depth_conf = predictions["depth_conf"][0].detach().float().cpu().numpy()
    points = utils.unproject_depths_to_world(depths, extrinsic, intrinsic)
    colors = utils.colors_from_torch_images(images)
    return extrinsic, intrinsic, depths, depth_conf, points, colors


@registry.register_model
class FastVGGT(base.BaseModel):
    """FastVGGT."""

    name = "FASTVGGT"
    description = "Run FastVGGT benchmark adapter."
    code_path_env = "FASTVGGT_CODE_PATH"
    code_packages = ("vggt",)
    ckpt_env = "FASTVGGT_CKPT_PATH"
    ckpt_default = "./ckpt/model_tracker_fixed_e20.pt"
    spatial_policies = ("legacy", "shared")

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--merging",
            type=int,
            default=0,
            help="FastVGGT's `merging` (token merging) setting.")
        parser.add_argument("--merge_ratio",
                            type=float,
                            default=0.9,
                            help="FastVGGT's token merge ratio.")

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        from vggt.models import vggt as vggt_net

        # Pass only the options this version of the fork accepts.
        parameters = inspect.signature(vggt_net.VGGT).parameters
        init_kwargs = {}
        if "merging" in parameters:
            init_kwargs["merging"] = self.args.merging
        if "merge_ratio" in parameters:
            init_kwargs["merge_ratio"] = self.args.merge_ratio
        if "vis_attn_map" in parameters:
            init_kwargs["vis_attn_map"] = False
        model = vggt_net.VGGT(**init_kwargs)

        ckpt_path = pathlib.Path(self.args.ckpt_path).expanduser()
        if ckpt_path.is_file():
            state = torch.load(str(ckpt_path), map_location="cpu")
            if isinstance(state, dict):
                state = state.get("model", state.get("state_dict", state))
            model.load_state_dict(state, strict=False)
        else:
            model = vggt_net.VGGT.from_pretrained(self.args.ckpt_path)
        return model.to(device).eval()

    def preprocess(self, image_paths, batch_index=None):
        if self.is_shared_policy:
            images, transforms = utils.preprocess_images(
                image_paths,
                method="fastvggt",
                size=self.args.resolution,
                mode="pad")
            data = {"paths": image_paths, "images": images.to(self.device)}
            return base.ModelInput(image_paths, data, transforms, batch_index)

        suffix = ""
        if batch_index is not None:
            suffix = f"_batch_{batch_index:04d}"
        resized_paths = utils.materialize_resized_images(
            image_paths,
            self.args.output_path / f"_resized_input_images{suffix}",
            self.args.image_load_resolution,
        )
        return base.ModelInput(
            image_paths=image_paths,
            data={
                "paths": resized_paths,
                "images": None
            },
            transforms=None,
            batch_index=batch_index,
            scratch_dirs=[pathlib.Path(resized_paths[0]).parent],
        )

    def forward(self, inputs):
        extrinsic, intrinsic, depth, depth_conf, points, colors = run_fastvggt(
            self.model,
            inputs.data["paths"],
            self.args,
            self.device,
            images=inputs.data["images"],
        )
        return base.Prediction(
            depth=depth,
            confidence=depth_conf,
            extrinsics_w2c=extrinsic,
            intrinsics=intrinsic,
            points=points,
            colors=colors,
        )

    def metrics_extra(self, depth_only, images_per_run=None):
        extra = super().metrics_extra(depth_only, images_per_run)
        extra["merging"] = self.args.merging
        extra["merge_ratio"] = self.args.merge_ratio
        extra["image_load_resolution"] = self.args.image_load_resolution
        extra["resized_input"] = True
        return extra
