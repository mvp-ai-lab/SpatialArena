# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""AMB3R: VGGT's camera, depth and point heads on AMB3R's front-end.

AMB3R's front-end decodes VGGT's own heads (amb3r/frontend.py's
`decode_heads` uses `vggt.utils.pose_enc.pose_encoding_to_extri_intri` from
its vendored `thirdparty/vggt`), so its predictions follow VGGT's layout
(spatialarena/models/vggt.py).

The images go through AMB3R's own `amb3r.datasets.Demo` loader, the one its
demo and benchmark scripts use, pointed at a scratch directory with this
run's images symlinked in. `disable_crop=True` makes its dust3r crop-resize
a plain anisotropic resize to 518 x 392 (no crop, no padding), so depth and
intrinsics of depth-only runs are restored to each source image. There is
no --spatial_policy option, and no --resolution / --image_load_resolution
(the size is fixed).
"""

from __future__ import annotations

import contextlib
import pathlib

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

# AMB3R's own benchmark harness (benchmark/eval_mvrecon.py etc.) hardcodes
# this non-square resolution for the base model; it is not a knob in its own
# scripts either.
AMB3R_WIDTH = 518
AMB3R_HEIGHT = 392


def materialize_symlinked_images(image_paths: list[str],
                                 dest_dir: pathlib.Path) -> None:
    """Symlinks the run's images into a scratch directory for Demo.

    AMB3R's Demo dataset lists its root with os.listdir (it takes no file
    list), so it is fed only this run's images, symlinked in with
    zero-padded names: Demo's natural sort then keeps `image_paths` order.

    Args:
        image_paths: The images, in model order.
        dest_dir: The scratch directory; stale links are replaced.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(image_paths):
        source = pathlib.Path(image_path).expanduser().resolve()
        target = dest_dir / f"{index:06d}{source.suffix.lower()}"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)


def load_model(ckpt_path: str, device):
    """Loads AMB3R through its model zoo.

    Args:
        ckpt_path: The checkpoint file.
        device: Device of the returned network.

    Returns:
        The network in eval mode on `device`.
    """
    # pylint: disable-next=import-outside-toplevel
    from amb3r import model_zoo

    model = model_zoo.load_model("amb3r", ckpt_path=ckpt_path)
    model.eval()
    return model.to(device)


def preprocess_amb3r(image_paths: list[str],
                     *,
                     args,
                     device=None,
                     run_name: str = ""):
    """Loads images through AMB3R's own Demo dataset.

    The images are symlinked into `<output>/_amb3r_input_images[_run]`,
    which Demo reads with `disable_crop=True` (a plain resize).

    Args:
        image_paths: Source images.
        args: The parsed options (uses output_path).
        device: Device of the views.
        run_name: Suffix of the scratch directory (a depth-only batch).

    Returns:
        A (model_input, transforms) tuple: model_input holds "views",
        "views_all" and "temp_images_dir"; transforms map each source image
        to the model grid.
    """
    # pylint: disable=import-outside-toplevel
    from amb3r import datasets
    from PIL import Image
    from PIL import ImageOps
    from torch.utils import data as torch_data

    # Fixed-resolution inputs (full-slam) keep their own size (an exact
    # no-op resize); otherwise 518 x 392.
    target_w, target_h = utils.fixed_input_size(image_paths) or (AMB3R_WIDTH,
                                                                 AMB3R_HEIGHT)
    transforms = []
    for image_path in image_paths:
        # Demo's loader (thirdparty/dust3r/utils/image.py) applies the EXIF
        # orientation; without it a rotated image's size reads swapped.
        with Image.open(image_path) as image:
            source_w, source_h = ImageOps.exif_transpose(image).size
        transforms.append(
            utils.SpatialTransform(source_wh=(source_w, source_h),
                                   resized_wh=(target_w, target_h)))

    suffix = f"_{run_name}" if run_name else ""
    temp_images_dir = args.output_path / f"_amb3r_input_images{suffix}"
    materialize_symlinked_images(image_paths, temp_images_dir)
    dataset = datasets.Demo(
        ROOT=str(temp_images_dir),
        resolution=(target_w, target_h),
        num_seq=1,
        full_video=True,
        kf_every=1,
        disable_crop=True,
        max_images=len(image_paths),
    )
    dataloader = torch_data.DataLoader(dataset,
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=0)
    views, views_all = next(iter(dataloader))
    for key in list(views_all.keys()):
        views_all[key] = views_all[key].to(device)
    return {
        "views": views,
        "views_all": views_all,
        "temp_images_dir": temp_images_dir
    }, transforms


def _autocast(device):
    """Returns AMB3R's bfloat16 autocast for `device`."""
    import torch  # pylint: disable=import-outside-toplevel

    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device == "cpu":
        # The checkpoint's weights are bf16; without autocast the fp32 input
        # makes conv2d fail on the dtype mismatch.
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def infer_amb3r(model, model_input, device, keep_intermediate=False):
    """Runs AMB3R on the output of `preprocess_amb3r`.

    The scratch image directory is removed afterwards, also on failure
    (unless kept).

    Args:
        model: The AMB3R network.
        model_input: The model input of `preprocess_amb3r`.
        device: "cuda" or "cpu".
        keep_intermediate: Keep the scratch image directory.

    Returns:
        An (extrinsic, intrinsic, pose, depth, world_points, confidence,
        colors) tuple of numpy arrays.
    """
    import torch  # pylint: disable=import-outside-toplevel

    temp_images_dir = model_input["temp_images_dir"]
    try:
        views = model_input["views"]
        views_all = model_input["views_all"]
        utils.record_model_input(views[0]["img"].shape[-1],
                                 views[0]["img"].shape[-2])
        with torch.no_grad(), _autocast(device):
            res_all = model(views_all)
        res = res_all[-1]

        def to_numpy(key):
            return res[key][0].float().detach().cpu().numpy()

        pose = to_numpy("pose")
        extrinsic = to_numpy("extrinsic")
        intrinsic = to_numpy("intrinsic")
        depth = to_numpy("depth")
        world_points = to_numpy("world_points")
        confidence = to_numpy("world_points_conf")
        # decode_heads' docstring: the images are in [0, 1] here.
        images = to_numpy("images")
    finally:
        utils.cleanup_intermediate_paths([temp_images_dir],
                                         keep=keep_intermediate)

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    colors = (images.transpose(0, 2, 3, 1).clip(0.0, 1.0) * 255.0).astype(
        np.uint8)
    return extrinsic, intrinsic, pose, depth, world_points, confidence, colors


def run_amb3r(model,
              image_paths,
              args,
              device,
              run_name="",
              keep_intermediate=False):
    """Preprocesses and runs AMB3R on images in one call.

    Args:
        model: The AMB3R network.
        image_paths: Source images.
        args: The parsed options.
        device: "cuda" or "cpu".
        run_name: Suffix of the scratch directory.
        keep_intermediate: Keep the scratch image directory.

    Returns:
        An (extrinsic, intrinsic, pose, depth, world_points, confidence,
        colors, transforms) tuple.
    """
    model_input, transforms = preprocess_amb3r(image_paths,
                                               args=args,
                                               device=device,
                                               run_name=run_name)
    outputs = infer_amb3r(model,
                          model_input,
                          device,
                          keep_intermediate=keep_intermediate)
    return (*outputs, transforms)


def run_name_of(batch_index):
    """Returns the scratch suffix of a depth-only batch ("" for none)."""
    if batch_index is None:
        return ""
    return f"batch_{batch_index:04d}"


@registry.register_model
class AMB3R(base.BaseModel):
    """AMB3R."""

    name = "AMB3R"
    aliases = ("AMB3R_BASE",)
    description = "Run AMB3R benchmark adapter."
    code_path_env = "AMB3R_CODE_PATH"
    code_packages = ("amb3r",)
    ckpt_env = "AMB3R_CKPT_PATH"
    ckpt_default = ""
    # AMB3R resizes to its own fixed 518 x 392 (see `preprocess_amb3r`).
    uses_resolution = False
    uses_image_load_resolution = False
    default_conf_thresh = 5.0

    @property
    def checkpoint(self):
        return self.args.ckpt_path or None

    def load(self, device):
        return load_model(self.args.ckpt_path, device)

    def preprocess(self, image_paths, batch_index=None):
        model_input, transforms = preprocess_amb3r(
            image_paths,
            args=self.args,
            device=self.device,
            run_name=run_name_of(batch_index))
        return base.ModelInput(image_paths, model_input, transforms,
                               batch_index)

    def forward(self, inputs):
        extrinsic, intrinsic, _, depth, points, confidence, colors = (
            infer_amb3r(self.model,
                        inputs.data,
                        self.device,
                        keep_intermediate=utils.should_keep_intermediate(
                            self.args)))
        return base.Prediction(
            depth=depth,
            confidence=confidence,
            extrinsics_w2c=extrinsic,
            intrinsics=intrinsic,
            points=points,
            colors=colors,
        )

    def metrics_extra(self, depth_only, images_per_run=None):
        extra = {"Peak GPU Memory": base.peak_gpu_memory_mib()}
        if depth_only:
            extra["max_images_per_run"] = images_per_run
        extra["resolution"] = f"{AMB3R_WIDTH}x{AMB3R_HEIGHT}"
        return extra
