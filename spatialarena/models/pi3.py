# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Pi3 and Pi3X, including Pi3X's depth- and camera-conditioned tracks.

Pi3's own loaders resize every image of a batch to one size without
cropping or padding, so depth and intrinsics are always restored to each
source image.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

TRACKS = (
    "rgb",
    "rgb_depth",
    "rgb_cameras",
    "rgb_masked_depth",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
)
CAMERA_TRACKS = (
    "rgb_cameras",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth_cameras",
)
DEPTH_TRACKS = (
    "rgb_depth",
    "rgb_masked_depth",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
)
MASKED_DEPTH_TRACKS = ("rgb_masked_depth", "rgb_masked_depth_cameras")
SPARSE_DEPTH_TRACKS = ("rgb_sparse_depth", "rgb_sparse_depth_cameras")
SPARSE_MODEL_LONG_SIDE = 518


def _load_track_depth_priors(data_path, image_paths, track):
    """Loads the depth priors of a depth track.

    Masked tracks use masked ground truth (spatialarena/depth_prior_masking.py);
    sparse tracks use real sparse depth (spatialarena/sparse_depth.py).
    """
    if track in SPARSE_DEPTH_TRACKS:
        return utils.load_sparse_depth_priors(
            data_path, image_paths, model_long_side=SPARSE_MODEL_LONG_SIDE)
    if track in MASKED_DEPTH_TRACKS:
        return utils.load_masked_depth_priors(data_path, image_paths)
    return utils.load_optional_depth_priors(data_path, image_paths)


def _zero_depth_like_image(image_path):
    """Returns an all-zero depth map of the image's size (no prior)."""
    # pylint: disable-next=import-outside-toplevel
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return np.zeros((height, width), dtype=np.float32)


def load_multimodal_conditions(args, image_paths):
    """Loads Pi3X's conditioning inputs of the track.

    Args:
        args: The parsed options.
        image_paths: Source images.

    Returns:
        None on the "rgb" track, else a dict of "poses", "depths",
        "intrinsics" and "mask_add_depth" (None where the track has none).

    Raises:
        ValueError: The track needs Pi3X or is unknown.
        FileNotFoundError: A depth track has no depth priors.
    """
    if args.track == "rgb":
        return None
    if args.model_variant != "pi3x":
        raise ValueError("Pi3 multimodal tracks require --model_variant pi3x")
    if args.track not in TRACKS[1:]:
        raise ValueError(f"Unsupported benchmark track: {args.track}")

    conditions = {
        "poses": None,
        "depths": None,
        "intrinsics": None,
        "mask_add_depth": None,
    }
    if args.track in DEPTH_TRACKS:
        depths, has_depth, _ = _load_track_depth_priors(args.data_path,
                                                        image_paths, args.track)
        if not has_depth.any():
            raise FileNotFoundError(
                f"No matched depth priors found for {args.data_path}")
        conditions["depths"] = []
        for image_path, depth in zip(image_paths, depths, strict=True):
            if depth is None:
                depth = _zero_depth_like_image(image_path)
            conditions["depths"].append(depth)
        conditions["mask_add_depth"] = has_depth
    if args.track in CAMERA_TRACKS:
        poses, intrinsics = utils.load_colmap_camera_priors(
            args.data_path,
            image_paths,
            colmap_path=args.colmap_path,
            colmap_model_subdir=args.colmap_model_subdir,
        )
        conditions["poses"] = poses
        conditions["intrinsics"] = intrinsics
    return conditions


def materialize_selected_original_images(image_paths, output_dir):
    """Links the selected images into one flat directory.

    Pi3's multimodal loader reads a directory, not a list of files.

    Args:
        image_paths: Source images.
        output_dir: Directory to fill; its old files are removed.

    Returns:
        The linked (or copied) paths, named by index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.iterdir():
        if existing.is_file() or existing.is_symlink():
            existing.unlink()

    linked_paths = []
    for index, image_path in enumerate(image_paths):
        source = pathlib.Path(image_path).resolve()
        suffix = source.suffix.lower() or ".png"
        destination = output_dir / f"{index:06d}{suffix}"
        try:
            destination.symlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
        linked_paths.append(str(destination))
    return linked_paths


def load_pi3_official_eval_rgb_tensor(image_paths, new_width, device, verbose):
    """Loads RGB images with Pi3's official evaluation resize.

    The first image's width is resized to `new_width` and its aspect ratio
    kept for the whole batch; both axes are then put on the 14-pixel patch
    grid (Pi3's eval `load_and_resize14`, not its pixel-budget loader).

    Args:
        image_paths: Source images.
        new_width: Target width before the patch-grid step.
        device: Device of the returned tensor.
        verbose: Print the target size.

    Returns:
        A [N, 3, H, W] tensor, empty when no image could be read.
    """
    # pylint: disable=import-outside-toplevel
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as tvf
    from PIL import Image

    sources = []
    for image_path in image_paths:
        try:
            sources.append(Image.open(image_path).convert("RGB"))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Could not load image {image_path}: {exc}")
    if not sources:
        return torch.empty(0, device=device)

    original_width, original_height = sources[0].size
    target_width = int(new_width)
    scale = target_width / original_width
    target_height = max(14, round(original_height * scale / 14) * 14)
    if verbose:
        print(f"Pi3 official-eval RGB resize: ({target_width}, "
              f"{target_height}) before patch-grid interpolation")

    to_tensor = tvf.ToTensor()
    tensors = []
    for image in sources:
        resized = image.resize((target_width, target_height),
                               Image.Resampling.LANCZOS)
        tensors.append(to_tensor(resized))
    images = torch.stack(tensors, dim=0).to(device)

    patch_height = images.shape[-2] // 14
    patch_width = images.shape[-1] // 14
    return F.interpolate(
        images,
        (patch_height * 14, patch_width * 14),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def pure_resize_transforms(image_paths, target_wh):
    """Returns the restoration transforms of Pi3's loaders.

    Both loaders stretch every image to one batch size with no crop and no
    padding, so a transform needs only the image's own size.

    Args:
        image_paths: Source images.
        target_wh: (width, height) of the model input.

    Returns:
        One `utils.SpatialTransform` per image.
    """
    # pylint: disable-next=import-outside-toplevel
    from PIL import Image

    transforms = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            transforms.append(
                utils.SpatialTransform(source_wh=image.size,
                                       resized_wh=target_wh))
    return transforms


def _network_class(pi3x):
    """Returns Pi3's Pi3X or Pi3 network class."""
    # pylint: disable=import-outside-toplevel
    if pi3x:
        from pi3.models import pi3x as pi3x_models

        return pi3x_models.Pi3X
    from pi3.models import pi3 as pi3_models

    return pi3_models.Pi3


def _autocast(device):
    """Returns Pi3's mixed-precision context for `device`."""
    import torch  # pylint: disable=import-outside-toplevel

    if device != "cuda":
        return contextlib.nullcontext()
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


@registry.register_model
class Pi3(base.BaseModel):
    """Pi3 and Pi3X."""

    name = "PI3_X"
    aliases = ("PI3",)
    metrics_name = "PI3"
    description = "Run Pi3/Pi3X benchmark adapter."
    code_path_env = "PI3_CODE_PATH"
    code_packages = ("pi3",)
    ckpt_env = "PI3_CKPT_PATH"
    ckpt_default = "yyfz233/Pi3X"
    # Pi3 sizes its input from --image_load_resolution (RGB track) or
    # --pixel_limit (multimodal tracks) only; it has no --resolution.
    uses_resolution = False
    image_load_resolution_help = (
        "Width the RGB track's images are resized to (Pi3's official "
        "evaluation resize).")
    default_image_load_resolution = 512
    default_pixel_limit = 262144
    tracks = TRACKS

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--pixel_limit",
            type=int,
            default=cls.spec_default("pixel_limit", cls.default_pixel_limit),
            help="Pixel budget per image of the multimodal tracks' loader.",
        )
        parser.add_argument(
            "--model_variant",
            choices=("pi3", "pi3x"),
            default=os.environ.get("PI3_MODEL_VARIANT", "pi3x"),
            help="Network: Pi3X (default: $PI3_MODEL_VARIANT, else pi3x) or "
            "the original Pi3 (RGB track only).",
        )

    def __init__(self, args):
        super().__init__(args)
        self.use_multimodal = args.track != "rgb"
        # Pi3's own loader and intrinsics decoder, imported by load().
        self.load_multimodal_data = None
        self.recover_intrinsic_from_rays_d = None

    @property
    def independent_single_view(self):
        """Whether single views run as one batch (only on the rgb track)."""
        return not self.use_multimodal

    def select_images(self):
        image_paths = utils.find_image_paths(self.images_dir)
        if self.args.track in CAMERA_TRACKS:
            if self.args.colmap_path is None:
                raise FileNotFoundError(
                    f"{self.args.track} requires --colmap_path")
            image_paths = utils.resolve_camera_conditioned_image_paths(
                self.args.data_path,
                image_paths,
                colmap_path=self.args.colmap_path)
        return utils.select_interval(image_paths, self.args.interval,
                                     self.args.max_images_per_run)

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        from pi3.utils import basic
        from pi3.utils import geometry

        self.load_multimodal_data = basic.load_multimodal_data
        self.recover_intrinsic_from_rays_d = (
            geometry.recover_intrinsic_from_rays_d)
        fixed = utils.fixed_input_size(self.image_paths)
        if fixed:
            self.args.pixel_limit = fixed[0] * fixed[1]

        pi3x = self.args.model_variant == "pi3x"
        network_cls = _network_class(pi3x)
        ckpt = pathlib.Path(self.args.ckpt_path).expanduser()
        if not ckpt.is_file():
            model = network_cls.from_pretrained(self.args.ckpt_path).eval()
            if (pi3x and not self.use_multimodal and
                    hasattr(model, "disable_multimodal")):
                model.disable_multimodal()
            return model.to(device)

        if pi3x:
            model = network_cls(use_multimodal=self.use_multimodal).eval()
        else:
            model = network_cls().eval()
        if ckpt.suffix == ".safetensors":
            # pylint: disable-next=import-outside-toplevel
            from safetensors import torch as safetensors_torch

            state = safetensors_torch.load_file(str(ckpt))
        else:
            state = torch.load(str(ckpt),
                               map_location="cpu",
                               weights_only=False)
        model.load_state_dict(state, strict=False)
        return model.to(device)

    def preprocess(self, image_paths, batch_index=None):
        if self.use_multimodal:
            images, model_kwargs, scratch_dirs = self._load_multimodal(
                image_paths, batch_index)
        else:
            images = load_pi3_official_eval_rgb_tensor(
                image_paths,
                new_width=self.args.image_load_resolution,
                device=self.device,
                verbose=True,
            )
            model_kwargs, scratch_dirs = {}, []
        if images.numel() == 0:
            raise ValueError(f"Pi3 did not load any images of "
                             f"{self.args.data_path}")

        target_wh = (images.shape[-1], images.shape[-2])
        return base.ModelInput(
            image_paths=image_paths,
            data=images,
            transforms=pure_resize_transforms(image_paths, target_wh),
            batch_index=batch_index,
            scratch_dirs=scratch_dirs,
            extra={"model_kwargs": model_kwargs},
        )

    def _load_multimodal(self, image_paths, batch_index):
        """Loads the images and conditions of a multimodal track.

        Returns:
            An (images, model_kwargs, scratch_dirs) tuple.
        """
        import torch  # pylint: disable=import-outside-toplevel

        suffix = ""
        if batch_index is not None:
            suffix = f"_batch_{batch_index:04d}"
        input_dir = self.args.output_path / f"_selected_input_images{suffix}"
        input_paths = materialize_selected_original_images(
            image_paths, input_dir)
        input_dir = pathlib.Path(input_paths[0]).parent

        conditions = load_multimodal_conditions(self.args, image_paths)
        images, model_kwargs = self.load_multimodal_data(
            str(input_dir),
            conditions,
            interval=1,
            PIXEL_LIMIT=self.args.pixel_limit,
            verbose=True,
            device=self.device,
        )
        if (self.args.track in DEPTH_TRACKS and conditions is not None and
                conditions.get("mask_add_depth") is not None):
            depth_mask = torch.as_tensor(conditions["mask_add_depth"],
                                         dtype=torch.bool,
                                         device=self.device)
            model_kwargs["mask_add_depth"] = depth_mask[None, :images.shape[1]]
        return images, model_kwargs, [input_dir]

    def forward(self, inputs):
        # pylint: disable=import-outside-toplevel
        import torch
        import torch.nn.functional as F

        images = inputs.data
        # Depth-only batches run as independent views in MonoDepth mode.
        single_view = (inputs.batch_index is not None and
                       utils.single_view_mode())
        with torch.no_grad(), _autocast(self.device):
            if self.use_multimodal:
                result = self.model(imgs=images, **inputs.extra["model_kwargs"])
            elif single_view:
                result = utils.fold_single_view_outputs(
                    self.model(images[:, None]), images.shape[0])
            else:
                result = self.model(images[None])

        local_points = result.get("local_points")
        if local_points is None:
            raise KeyError("Pi3 output missing local_points; cannot export "
                           "depth")
        depths = utils.normalize_depths(
            utils.normalize_points(local_points)[..., 2])
        confidence = None
        if "conf" in result:
            confidence = utils.normalize_depths(
                torch.sigmoid(result["conf"][..., 0]))
        rays_d = F.normalize(local_points, dim=-1)
        intrinsics = self.recover_intrinsic_from_rays_d(
            rays_d, force_center_principal_point=True)
        if images.ndim == 5:
            colors = utils.colors_from_torch_images(images[0])
        else:
            colors = utils.colors_from_torch_images(images)
        return base.Prediction(
            depth=depths,
            confidence=confidence,
            extrinsics_w2c=utils.c2w_to_w2c(result["camera_poses"][0]),
            intrinsics=intrinsics[0].detach().float().cpu().numpy(),
            points=utils.normalize_points(result["points"]),
            colors=colors,
        )

    def metrics_extra(self, depth_only, images_per_run=None):
        extra = super().metrics_extra(depth_only, images_per_run)
        extra["model_variant"] = self.args.model_variant
        extra["track"] = self.args.track
        extra["pixel_limit"] = self.args.pixel_limit
        extra["image_load_resolution"] = self.args.image_load_resolution
        extra["pi3_rgb_preprocess"] = "official_width_resize14"
        extra["resized_input"] = not self.use_multimodal
        return extra
