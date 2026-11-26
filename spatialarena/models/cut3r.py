# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""CUT3R: a recurrent 3D reconstruction model, through its own demo code.

--spatial_policy legacy (default): the images are padded to a square on disk
at --image_load_resolution, then CUT3R's own `demo.prepare_input`
(dust3r.utils.image.load_images) resizes and center-crops them to its
16-pixel grid, which discards border pixels (and crops square images to
4:3); depth is exported on that grid. --spatial_policy shared: one in-memory
pad-to-square resize to --resolution (VGGT-Omega's plan, also a 16-pixel
patch), with CUT3R's view dicts built from that tensor; depth and
intrinsics are restored to each source image.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils


def _cut3r_module(name):
    """Imports a submodule of CUT3R's code by its full name.

    Not `from package import name`, which would return a same-named
    attribute the package re-exports instead of the submodule.

    Args:
        name: Dotted module name, e.g. "src.dust3r.inference".

    Returns:
        The module.
    """
    return importlib.import_module(name)


def stack_list(values: list) -> np.ndarray:
    """Concatenates CUT3R's per-frame outputs along the first axis.

    Args:
        values: Per-frame tensors or arrays, each (1, ...).

    Returns:
        The (N, ...) array.
    """
    arrays = [utils.as_numpy(value) for value in values]
    return np.concatenate(arrays, axis=0)


def intrinsics_from_cam_dict(cam_dict: dict) -> np.ndarray:
    """Returns (N, 3, 3) intrinsics from CUT3R's focal and principal point.

    Args:
        cam_dict: CUT3R's `prepare_output` cameras ("focal", "pp").
    """
    focal = np.asarray(cam_dict["focal"], dtype=np.float32).reshape(-1)
    pp = np.asarray(cam_dict["pp"], dtype=np.float32).reshape(-1, 2)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None],
                           focal.shape[0],
                           axis=0)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = pp[:, 0]
    intrinsics[:, 1, 2] = pp[:, 1]
    return intrinsics


def extrinsics_from_cam_dict(cam_dict: dict, num_frames: int) -> np.ndarray:
    """Returns (N, 4, 4) world-to-camera matrices from CUT3R's R and t.

    Args:
        cam_dict: CUT3R's `prepare_output` cameras (camera-to-world "R",
            "t").
        num_frames: Number of frames N.
    """
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], num_frames, axis=0)
    c2w[:, :3, :3] = np.asarray(cam_dict["R"], dtype=np.float32)
    c2w[:, :3, 3] = np.asarray(cam_dict["t"], dtype=np.float32)
    return utils.c2w_to_w2c(c2w)


def patch_rope_negative_positions() -> None:
    """Lets CUT3R's RoPE2D take the pose tokens' negative positions.

    Needed when the CUDA RoPE extension is unavailable.
    """
    # pylint: disable=import-outside-toplevel
    import torch
    import torch.nn.functional as F
    from models import pos_embed

    rope_cls = getattr(pos_embed, "RoPE2D", None)
    if rope_cls is None or getattr(rope_cls, "_ffb_negative_position_patch",
                                   False):
        return
    if not hasattr(rope_cls, "get_cos_sin") or not hasattr(
            rope_cls, "rotate_half"):
        return

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        valid = pos1d >= 0
        safe_pos = pos1d.clamp_min(0)
        cos_pos = F.embedding(safe_pos, cos)[:, None, :, :]
        sin_pos = F.embedding(safe_pos, sin)[:, None, :, :]
        rotated = (tokens * cos_pos) + (self.rotate_half(tokens) * sin_pos)
        if bool(valid.all()):
            return rotated
        return torch.where(valid[:, None, :, None], rotated, tokens)

    def forward(self, tokens, positions):
        assert tokens.size(
            3) % 2 == 0, "number of dimensions should be a multiple of two"
        assert positions.ndim == 3 and positions.shape[-1] == 2
        dim = tokens.size(3) // 2
        max_position = int(positions.clamp_min(0).max()) + 1
        cos, sin = self.get_cos_sin(dim, max_position, tokens.device,
                                    tokens.dtype)
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        return torch.cat((y, x), dim=-1)

    rope_cls.apply_rope1d = apply_rope1d
    rope_cls.forward = forward
    rope_cls._ffb_negative_position_patch = True


def build_shared_views(image_paths: list[str], args, device):
    """Builds CUT3R's view dicts from a padded tensor (shared policy).

    The dicts (img/ray_map/true_shape/idx/instance/...) match the image-only
    output of `demo.prepare_input`, whose loader is bypassed, so no source
    pixel is cropped away.

    Args:
        image_paths: Source images.
        args: The parsed options (uses resolution).
        device: Device of the views.

    Returns:
        A (views, transforms) tuple.
    """
    import torch  # pylint: disable=import-outside-toplevel

    # exif_transpose=True as dust3r.utils.image.load_images does.
    images, transforms = utils.preprocess_images(image_paths,
                                                 method="vggt_omega",
                                                 size=args.resolution,
                                                 mode="pad",
                                                 exif_transpose=True)
    # dust3r's ImgNorm is ToTensor ([0, 1]) + Normalize(0.5, 0.5) ([-1, 1]);
    # preprocess_images already returns [0, 1].
    images = (images * 2.0 - 1.0).to(device)
    views = []
    for index in range(images.shape[0]):
        frame = images[index:index + 1]
        height, width = frame.shape[-2:]
        views.append({
            "img":
                frame,
            "ray_map":
                torch.full((1, 6, height, width), float("nan"), device=device),
            "true_shape":
                torch.from_numpy(np.int32([[height, width]])),
            "idx":
                index,
            "instance":
                str(index),
            # As demo.prepare_input's image-only branch: a fixed identity.
            "camera_pose":
                torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0),
            "img_mask":
                torch.tensor(True, device=device).unsqueeze(0),
            "ray_mask":
                torch.tensor(False, device=device).unsqueeze(0),
            "update":
                torch.tensor(True, device=device).unsqueeze(0),
            "reset":
                torch.tensor(False, device=device).unsqueeze(0),
        })
    return views, transforms


@registry.register_model
class CUT3R(base.BaseModel):
    """CUT3R."""

    name = "CUT3R"
    description = "Run CUT3R benchmark adapter."
    code_path_env = "CUT3R_CODE_PATH"
    code_packages = ("src", "dust3r")
    ckpt_env = "CUT3R_CKPT_PATH"
    ckpt_default = "src/cut3r_512_dpt_4_64.pth"
    default_resolution = 512
    default_image_load_resolution = 512
    spatial_policies = ("legacy", "shared")

    def __init__(self, args):
        super().__init__(args)
        self.resolved_checkpoint = None

    @property
    def checkpoint(self):
        """The checkpoint file actually loaded (see `load`)."""
        if self.resolved_checkpoint is not None:
            return self.resolved_checkpoint
        return self.args.ckpt_path

    def add_code_path(self):
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        super().add_code_path()
        code_path = self.args.code_path
        if code_path is not None:
            src_path = code_path.expanduser().resolve() / "src"
            if src_path.is_dir() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

    def load(self, device):
        # A relative --ckpt_path may be under --code_path.
        # pylint: disable=import-outside-toplevel
        import torch
        import add_ckpt_path

        # CUT3R's demo modules load before the weights, as in its demo.py.
        _cut3r_module("demo")
        _cut3r_module("src.dust3r.inference")
        cut3r_model = _cut3r_module("src.dust3r.model")

        patch_rope_negative_positions()

        checkpoint = self.args.ckpt_path
        code_path = self.args.code_path
        if code_path is not None:
            candidate = code_path / self.args.ckpt_path
            if (not pathlib.Path(self.args.ckpt_path).expanduser().is_file() and
                    candidate.is_file()):
                checkpoint = str(candidate)
        self.resolved_checkpoint = checkpoint
        add_ckpt_path.add_path_to_dust3r(checkpoint)

        # CUT3R's checkpoints are trusted pickles.
        original_torch_load = torch.load

        def trusted_torch_load(*load_args, **load_kwargs):
            load_kwargs.setdefault("weights_only", False)
            return original_torch_load(*load_args, **load_kwargs)

        torch.load = trusted_torch_load
        try:
            model = cut3r_model.ARCroco3DStereo.from_pretrained(checkpoint)
            model = model.to(device).eval()
        finally:
            torch.load = original_torch_load
        return model

    def preprocess(self, image_paths, batch_index=None):
        # Fixed-resolution frames (full-slam) go to CUT3R's own loader as
        # they are: it center-crops them to its 16-pixel grid (504x378 ->
        # 496x368); a square pad first would make it crop a 4:3 window off
        # the grid. Otherwise --spatial_policy shared builds the views from
        # a padded tensor (`build_shared_views`) and legacy pads the images
        # on disk for CUT3R's own `demo.prepare_input`.
        import demo  # pylint: disable=import-outside-toplevel

        args = self.args
        if batch_index is None:
            batch_suffix = ""
            raw_dir = args.output_path / "_cut3r_raw"
        else:
            batch_suffix = f"_batch_{batch_index:04d}"
            raw_dir = args.output_path / f"_cut3r_raw{batch_suffix}"

        fixed = utils.fixed_input_size(image_paths)
        if fixed or self.is_shared_policy:
            model_input_paths = image_paths
            resized_dir = None
        else:
            model_input_paths = utils.materialize_resized_images(
                image_paths,
                args.output_path / f"_resized_input_images{batch_suffix}",
                args.image_load_resolution,
            )
            resized_dir = pathlib.Path(model_input_paths[0]).parent

        if self.is_shared_policy and not fixed:
            views, transforms = build_shared_views(model_input_paths, args,
                                                   self.device)
        else:
            views = demo.prepare_input(
                img_paths=model_input_paths,
                img_mask=[True] * len(model_input_paths),
                size=args.resolution,
                revisit=1,
                update=True,
            )
            transforms = None
        return base.ModelInput(
            image_paths=image_paths,
            data=views,
            transforms=transforms,
            batch_index=batch_index,
            scratch_dirs=[resized_dir, raw_dir],
            extra={"raw_dir": raw_dir},
        )

    def forward(self, inputs):
        # CUT3R's prepare_output writes the per-frame depth to raw_dir.
        import demo  # pylint: disable=import-outside-toplevel

        inference = _cut3r_module("src.dust3r.inference")
        views = inputs.data
        raw_dir = inputs.extra["raw_dir"]
        num_frames = len(inputs.image_paths)
        utils.record_model_input(views[0]["img"].shape[-1],
                                 views[0]["img"].shape[-2])
        outputs, _ = inference.inference(views, self.model, self.device)
        points_list, colors_list, conf_list, cam_dict = demo.prepare_output(
            outputs,
            str(raw_dir),
            revisit=1,
            use_pose=True,
        )
        points = stack_list(points_list)
        colors = stack_list(colors_list)
        confidence = stack_list(conf_list)
        depths = np.stack([
            np.load(raw_dir / "depth" / f"{frame_id:06d}.npy")
            for frame_id in range(num_frames)
        ],
                          axis=0)
        return base.Prediction(
            depth=depths,
            confidence=confidence,
            extrinsics_w2c=extrinsics_from_cam_dict(cam_dict, num_frames),
            intrinsics=intrinsics_from_cam_dict(cam_dict),
            points=points,
            colors=colors,
        )

    def metrics_extra(self, depth_only, images_per_run=None):
        if depth_only:
            max_images_per_run = images_per_run
        else:
            max_images_per_run = self.args.max_images_per_run
        extra = {
            "Peak GPU Memory": base.peak_gpu_memory_mib(),
            "resolution": self.args.resolution,
            "image_load_resolution": self.args.image_load_resolution,
            "max_images_per_run": max_images_per_run,
            "resized_input": True,
        }
        extra.update(
            utils.spatial_policy_metrics(self.args.spatial_policy,
                                         is_shared=self.is_shared_policy))
        return extra
