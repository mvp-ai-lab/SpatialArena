# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""VGGT and RobustVGGT (VGGT's network from its own checkout and weights).

--spatial_policy legacy (default): VGGT's own two-stage resize; the depth is
exported with the square padding cropped off, not restored to the source
size. --spatial_policy shared: one pad-to-square resize to --resolution;
depth and intrinsics are restored to each source image.
"""

from __future__ import annotations

import contextlib
import pathlib

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils


def legacy_load_vggt_images(image_paths, args, device):
    """Loads images with VGGT's own two-stage resize.

    The images are padded to a square at --image_load_resolution by VGGT's
    loader, then interpolated to --resolution.

    Args:
        image_paths: Source images.
        args: The parsed options.
        device: Device of the returned tensor.

    Returns:
        An (images, original_coords) tuple: the [N, 3, R, R] tensor and the
        loader's per-image coordinates of the unpadded region.
    """
    # pylint: disable=import-outside-toplevel
    import torch.nn.functional as F
    from vggt.utils import load_fn

    images, original_coords = load_fn.load_and_preprocess_images_square(
        image_paths, target_size=args.image_load_resolution)
    images = F.interpolate(
        images.to(device),
        size=(args.resolution, args.resolution),
        mode="bilinear",
        align_corners=False,
    )
    return images, original_coords


def crop_to_original_image_region(depth, original_coord, padded_size):
    """Cuts the square padding off a legacy-policy map.

    Args:
        depth: One map, (H, W[, ...]).
        original_coord: The loader's (x1, y1, x2, y2, w, h) of the image in
            the padded square, or None to return the map unchanged.
        padded_size: Side of the padded square the coordinates refer to.

    Returns:
        The map without the padding.
    """
    if original_coord is None:
        return depth
    x1, y1, x2, y2, _, _ = [float(value) for value in original_coord]
    height, width = depth.shape[:2]
    scale_x = width / float(padded_size)
    scale_y = height / float(padded_size)
    left = max(0, min(int(np.floor(x1 * scale_x)), width - 1))
    top = max(0, min(int(np.floor(y1 * scale_y)), height - 1))
    right = max(left + 1, min(int(np.ceil(x2 * scale_x)), width))
    bottom = max(top + 1, min(int(np.ceil(y2 * scale_y)), height))
    return depth[top:bottom, left:right]


def export_cropped_depths(output_path,
                          image_paths,
                          depths,
                          original_coords=None,
                          padded_size=518,
                          confidence=None):
    """Writes depth (float32) and confidence (float16) without the padding.

    Args:
        output_path: Method scene output directory.
        image_paths: Source images, one per map.
        depths: Depth maps, (N, H, W).
        original_coords: The loader's coordinates, one per image, or None.
        padded_size: Side of the padded square the coordinates refer to.
        confidence: Confidence maps, (N, H, W), or None.

    Raises:
        ValueError: The confidence maps do not match the depth maps.
    """
    depth_dir = output_path / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)
    if (confidence is not None and
            np.shape(confidence)[:3] != np.shape(depths)[:3]):
        raise ValueError(f"confidence {np.shape(confidence)} does not match "
                         f"depth {np.shape(depths)}")

    for index, (image_path, depth) in enumerate(zip(image_paths, depths)):
        original_coord = None
        if original_coords is not None:
            original_coord = original_coords[index]
        depth = crop_to_original_image_region(depth, original_coord,
                                              padded_size)
        depth = depth.astype(np.float32, copy=False)
        utils.save_prediction_map(depth_dir,
                                  pathlib.Path(image_path).stem, depth)
        if confidence is not None:
            cropped = crop_to_original_image_region(confidence[index],
                                                    original_coord, padded_size)
            utils.save_confidence_map(output_path, image_path, cropped)


def _autocast(device):
    """Returns VGGT's mixed-precision context for `device`."""
    import torch  # pylint: disable=import-outside-toplevel

    if device != "cuda":
        return contextlib.nullcontext()
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def _to_numpy(tensor):
    """Returns a [1, N, ...] model output as an [N, ...] numpy array."""
    array = tensor.squeeze(0).detach().cpu().numpy()
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]  # Drop the channel axis of depth maps.
    return array


def run_vggt(model, images, device, single_view=False):
    """Runs VGGT on preprocessed images.

    Args:
        model: The VGGT network.
        images: [N, 3, H, W] tensor on `device`.
        device: "cuda" or "cpu".
        single_view: Run the images as independent single-image samples
            ([N, 1, 3, H, W]) instead of one N-view sequence.

    Returns:
        An (extrinsic, intrinsic, depth, depth_conf, world_points, colors)
        tuple of numpy arrays in the sequence layout.

    Raises:
        TypeError: The model does not return a prediction dict.
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from vggt.utils import pose_enc

    with torch.no_grad(), _autocast(device):
        if single_view:
            batched_images = images[:, None]
        else:
            batched_images = images[None]
        predictions = model(batched_images)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if single_view:
            predictions = utils.fold_single_view_outputs(
                predictions, images.shape[0])
        if not isinstance(predictions, dict):
            raise TypeError("A VGGT model must return a prediction dict, got "
                            f"{type(predictions).__name__}")
        extrinsic, intrinsic = pose_enc.pose_encoding_to_extri_intri(
            predictions["pose_enc"], batched_images.shape[-2:])

    colors = images.detach().cpu().numpy() * 255.0
    colors = colors.clip(0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
    return (
        _to_numpy(extrinsic),
        _to_numpy(intrinsic),
        _to_numpy(predictions["depth"]),
        _to_numpy(predictions["depth_conf"]),
        _to_numpy(predictions["world_points"]),
        colors,
    )


@registry.register_model
class VGGT(base.BaseModel):
    """VGGT."""

    name = "VGGT"
    description = "Run VGGT and export benchmark prediction artifacts."
    code_path_env = "VGGT_CODE_PATH"
    code_packages = ("vggt",)
    ckpt_env = "VGGT_CKPT_PATH"
    ckpt_default = "facebook/VGGT-1B"
    default_resolution = 518
    default_image_load_resolution = 1024
    default_conf_thresh = 5.0
    spatial_policies = ("legacy", "shared")
    independent_single_view = True

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--allow_partial_checkpoint",
            action="store_true",
            help="Load a checkpoint file that lacks some of the weights.")

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        from vggt.models import vggt as vggt_net

        ckpt_path = pathlib.Path(self.args.ckpt_path).expanduser()
        if ckpt_path.is_file():
            model = vggt_net.VGGT()
            state = torch.load(str(ckpt_path), map_location="cpu")
            if isinstance(state, dict):
                state = state.get("model", state.get("state_dict", state))
            strict = not self.args.allow_partial_checkpoint
            model.load_state_dict(state, strict=strict)
        else:
            model = vggt_net.VGGT.from_pretrained(self.args.ckpt_path)
        model.eval()
        return model.to(device)

    def preprocess(self, image_paths, batch_index=None):
        images, original_coords, transforms = utils.legacy_or_shared_tensor(
            image_paths,
            method="vggt",
            args=self.args,
            device=self.device,
            legacy_loader=lambda paths: legacy_load_vggt_images(
                paths, self.args, self.device),
        )
        return base.ModelInput(
            image_paths=image_paths,
            data=images,
            transforms=transforms,
            batch_index=batch_index,
            extra={"original_coords": original_coords},
        )

    def forward(self, inputs):
        # Depth-only batches run as independent views in MonoDepth mode.
        single_view = (inputs.batch_index is not None and
                       utils.single_view_mode())
        extrinsic, intrinsic, depth, depth_conf, points, colors = run_vggt(
            self.model, inputs.data, self.device, single_view=single_view)
        return base.Prediction(
            depth=depth,
            confidence=depth_conf,
            extrinsics_w2c=extrinsic,
            intrinsics=intrinsic,
            points=points,
            colors=colors,
        )

    def export_depth_legacy(self, inputs, prediction):
        export_cropped_depths(
            self.args.output_path,
            inputs.image_paths,
            prediction.depth,
            original_coords=inputs.extra["original_coords"],
            padded_size=self.args.image_load_resolution,
            confidence=prediction.confidence,
        )

    def legacy_camera_size(self, prediction):
        return self.args.resolution, self.args.resolution

    def run_full(self):
        """Exports the dense points, depth and cameras of one call.

        Unlike the default, the points come from VGGT's world-point head
        (not unprojected depth), the depth has the legacy padding cropped
        off, and the cameras refer to the --resolution square.
        """
        # pylint: disable=import-outside-toplevel
        from spatialarena.geometry import utils as geometry_utils
        from spatialarena.utils import point_utils

        args = self.args
        prediction, inputs = self.predict(self.image_paths)
        runtime = utils.runtime_metrics(self.started_at)

        points, colors = utils.sample_points(
            prediction.points,
            prediction.colors,
            prediction.confidence,
            conf_thresh=self.conf_thresh,
            max_points=args.max_points,
        )
        point_utils.export_ply(str(args.output_path / "points3D_dense.ply"),
                               points, colors)
        if args.export_depth:
            self.export_depth(inputs, prediction)
        if args.export_colmap:
            colmap_dir = args.output_path / "sparse" / args.colmap_model_subdir
            colmap_dir.mkdir(parents=True, exist_ok=True)
            geometry_utils.write_colmap_formats(
                str(self.images_dir),
                prediction.extrinsics_w2c,
                prediction.intrinsics,
                self.image_paths,
                points,
                colors,
                width=args.resolution,
                height=args.resolution,
                output_path=str(colmap_dir),
            )

        extra = self.metrics_extra(depth_only=False)
        self.write_metrics(depth_only=False,
                           metrics_extra=extra,
                           runtime=runtime)


@registry.register_model
class RobustVGGT(VGGT):
    """RobustVGGT: VGGT's network and loader with RobustVGGT's weights."""

    name = "ROBUSTVGGT"
    description = "Run RobustVGGT through its VGGT-compatible network."
    code_path_env = "ROBUSTVGGT_CODE_PATH"
    ckpt_env = "ROBUSTVGGT_CKPT_PATH"

    def metrics_extra(self, depth_only, images_per_run=None):
        extra = super().metrics_extra(depth_only, images_per_run)
        extra["adapter"] = "robustvggt.py"
        return extra
