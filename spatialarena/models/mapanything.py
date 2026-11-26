# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""MapAnything, optionally conditioned on depth and camera priors.

The "rgb" track pads each image into MapAnything's 14 px patch grid, so
depth and intrinsics are restored to the source images. The depth and
camera tracks still go through MapAnything's own `preprocess_inputs`,
which crops images, depth priors and intrinsics together to one target
size; their maps are exported on that grid.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

# MapAnything's patch size is 14 (the DUSt3R-family wrappers use 16). Mean and
# std of its default norm_type "dinov2", the ImageNet statistics of every
# DINOv2-backed encoder (uniception's IMAGE_NORMALIZATION_DICT["dinov2"]).
_PATCH_SIZE = 14
_NORM_TYPE = "dinov2"
_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)

TRACKS = (
    "rgb",
    "rgb_depth",
    "rgb_cameras",
    "rgb_masked_depth",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
)
CAMERA_TRACKS = ("rgb_cameras", "rgb_masked_depth_cameras",
                 "rgb_sparse_depth_cameras")
DEPTH_TRACKS = (
    "rgb_depth",
    "rgb_masked_depth",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
)
MASKED_DEPTH_TRACKS = ("rgb_masked_depth", "rgb_masked_depth_cameras")
SPARSE_DEPTH_TRACKS = ("rgb_sparse_depth", "rgb_sparse_depth_cameras")
# The input's long side the sparse points are drawn for
# (utils.load_sparse_depth_priors).
SPARSE_MODEL_LONG_SIDE = 518


def _load_track_depth_priors(data_path, image_paths, track):
    """Loads the depth priors of a depth track.

    Masked ground truth on the masked-depth tracks
    (spatialarena/depth_prior_masking.py), the real sparse depth (LiDAR or
    COLMAP keypoints) on the sparse-depth tracks (spatialarena/sparse_depth.py),
    the full depth prior otherwise.

    Args:
        data_path: Scene root.
        image_paths: The images.
        track: The run's track.

    Returns:
        The loader's (depths, has_depth, missing) tuple.
    """
    if track in SPARSE_DEPTH_TRACKS:
        return utils.load_sparse_depth_priors(
            data_path, image_paths, model_long_side=SPARSE_MODEL_LONG_SIDE)
    if track in MASKED_DEPTH_TRACKS:
        return utils.load_masked_depth_priors(data_path, image_paths)
    return utils.load_optional_depth_priors(data_path, image_paths)


def build_rgb_views(image_paths, resolution_set):
    """Builds MapAnything views of the "rgb" track from padded images.

    This bypasses `mapanything.utils.image.load_images`, which crops every
    image to one target size computed from the batch's average aspect ratio
    and so discards real content. The "rgb" track is not conditioned on any
    prior, so a per-image pad into the patch grid is enough.

    Args:
        image_paths: Source images.
        resolution_set: Long side of the model input, rounded up to the
            patch size.

    Returns:
        A (views, transforms) tuple: the view dicts (img, data_norm_type,
        true_shape, idx, instance) and the per-image
        `utils.SpatialTransform`.
    """
    import torch  # pylint: disable=import-outside-toplevel

    size = resolution_set
    if resolution_set % _PATCH_SIZE != 0:
        size = utils.round_up_to_multiple(resolution_set, _PATCH_SIZE)
    # exif_transpose=True as mapanything.utils.image.load_images, which this
    # replaces.
    images, transforms = utils.preprocess_images(image_paths,
                                                 method="vggt",
                                                 size=size,
                                                 mode="pad",
                                                 exif_transpose=True)
    mean = torch.tensor(_NORM_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_NORM_STD).view(1, 3, 1, 1)
    images = (images - mean) / std
    views = []
    for index, image_path in enumerate(image_paths):
        frame = images[index:index + 1]
        height, width = frame.shape[-2:]
        views.append({
            "img": frame,
            "data_norm_type": [_NORM_TYPE],
            "true_shape": np.int32([[height, width]]),
            "idx": index,
            "instance": [pathlib.Path(image_path).name],
        })
    return views, transforms


def _load_camera_priors(args, image_paths):
    """Returns the COLMAP (camera_poses, intrinsics) priors of the scene."""
    return utils.load_colmap_camera_priors(
        args.data_path,
        image_paths,
        colmap_path=args.colmap_path,
        colmap_model_subdir=args.colmap_model_subdir,
    )


def preprocess_mapanything(image_paths, *, args, device=None, **unused):
    """Builds MapAnything's input views.

    The "rgb" track uses `build_rgb_views`. The depth and camera tracks go
    through MapAnything's own `preprocess_inputs`, which crops image, depth
    prior and intrinsics together to one batch-average target size: a real
    content loss, kept because padding the image without padding its priors
    in lockstep would desynchronize them.

    Args:
        image_paths: Source images.
        args: The parsed options.
        device: Unused; the views stay on the CPU.
        **unused: Further `utils.prepare_model_input` arguments.

    Returns:
        A (views, transforms) tuple; transforms is None on the depth and
        camera tracks.
    """
    del device, unused  # The views are built on the CPU.
    # pylint: disable=import-outside-toplevel
    import torch
    from mapanything.utils import image as mapanything_image
    from PIL import Image

    resolution_set = args.image_load_resolution
    if resolution_set not in {504, 512, 518}:
        resolution_set = 518

    if args.track == "rgb":
        return build_rgb_views(image_paths, resolution_set)

    depths = None
    camera_poses = None
    intrinsics = None
    if args.track in DEPTH_TRACKS:
        depths, _, _ = _load_track_depth_priors(args.data_path, image_paths,
                                                args.track)
    if args.track in CAMERA_TRACKS:
        camera_poses, intrinsics = _load_camera_priors(args, image_paths)

    raw_views = []
    for index, image_path in enumerate(image_paths):
        view = {"img": Image.open(image_path).convert("RGB")}
        metric_scale = torch.ones(1, dtype=torch.bool)
        if depths is not None and depths[index] is not None:
            if intrinsics is None:
                try:
                    _, intrinsics = _load_camera_priors(args, image_paths)
                except FileNotFoundError:
                    _, intrinsics = utils.build_default_camera_priors(
                        image_paths)
            view["depth_z"] = utils.resize_depth_to_image_size(
                depths[index], view["img"].size)
            view["intrinsics"] = intrinsics[index]
            view["is_metric_scale"] = metric_scale
        if camera_poses is not None:
            view["camera_poses"] = camera_poses[index]
            view["intrinsics"] = intrinsics[index]
            view["is_metric_scale"] = metric_scale
        raw_views.append(view)
    # TODO(rgb_depth/rgb_cameras content loss): convert these tracks to the
    # pad-based input too, padding and rescaling the depth prior and the
    # input intrinsics in lockstep with the image.
    views = mapanything_image.preprocess_inputs(
        raw_views,
        resize_mode="fixed_mapping",
        resolution_set=resolution_set,
    )
    return views, None


utils.register_preprocessor("MAPANYTHING", preprocess_mapanything)


@registry.register_model
class MapAnything(base.BaseModel):
    """MapAnything."""

    name = "MAPANYTHING"
    aliases = ("MAP_ANYTHING",)
    description = "Run MapAnything and export benchmark prediction artifacts."
    code_path_env = "MAPANYTHING_CODE_PATH"
    code_packages = ("mapanything",)
    ckpt_env = "MAPANYTHING_CKPT_PATH"
    ckpt_default = "facebook/map-anything"
    tracks = TRACKS
    uses_resolution = False
    image_load_resolution_help = (
        "Long side of the model input: 504, 512 or 518 (any other value: "
        "518).")

    def __init__(self, args):
        super().__init__(args)
        self.load_started_at = None

    @classmethod
    def add_arguments(cls, parser):
        """Adds --image_load_resolution, MapAnything's input long side.

        MapAnything has no separate model input resolution (no
        --resolution): the images are loaded at this long side.

        Args:
            parser: The parser.
        """

    def select_images(self):
        """Returns the frames; camera tracks keep those with a COLMAP pose."""
        image_paths = utils.find_image_paths(self.images_dir)
        if self.args.track in CAMERA_TRACKS:
            image_paths = utils.resolve_camera_conditioned_image_paths(
                self.args.data_path,
                image_paths,
                colmap_path=self.args.colmap_path,
            )
        return utils.select_interval(image_paths, self.args.interval,
                                     self.args.max_images_per_run)

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        from mapanything import models as mapanything_models

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # The legacy "inference time" of metrics.json includes the load.
        self.load_started_at = time.time()
        model = mapanything_models.MapAnything.from_pretrained(
            self.args.ckpt_path).to(device)
        model.eval()
        return model

    def preprocess(self, image_paths, batch_index=None):
        views, transforms = preprocess_mapanything(image_paths,
                                                   args=self.args,
                                                   device=self.device)
        return base.ModelInput(image_paths, views, transforms, batch_index)

    def forward(self, inputs):
        import torch  # pylint: disable=import-outside-toplevel

        depth_only = self.args.depth_only
        with torch.no_grad():
            # Depth-only (MonoDepth / VideoDepth / MVRecon depth) is unmasked:
            # MapAnything's non-ambiguous and edge masks write depth 0 on the
            # pixels they reject (up to ~14% of KITTI's GT pixels), a
            # method-specific coverage change; MVRecon fusion filters with the
            # engine's thresholds as for every other method.
            outputs = self.model.infer(
                inputs.data,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=not depth_only,
                mask_edges=not depth_only,
            )
        depths = utils.normalize_depths(
            [utils.as_numpy(pred["depth_z"])[0, ..., 0] for pred in outputs])
        prediction = base.Prediction(depth=depths)
        if not depth_only:
            prediction.points = utils.normalize_points(
                [utils.as_numpy(pred["pts3d"])[0] for pred in outputs])
            prediction.colors = utils.colors_from_torch_images(
                [utils.as_numpy(pred["img_no_norm"])[0] for pred in outputs])
        prediction.confidence = utils.normalize_confidence(
            [utils.as_numpy(pred["conf"])[0] for pred in outputs], depths.shape)
        if not depth_only or self.args.export_colmap:
            prediction.extrinsics_w2c = utils.c2w_to_w2c(
                [utils.as_numpy(pred["camera_poses"])[0] for pred in outputs])
            prediction.intrinsics = utils.as_numpy([
                utils.as_numpy(pred["intrinsics"])[0] for pred in outputs
            ]).astype("float32")
        return prediction

    def run_depth_only(self):
        self.started_at = self.load_started_at
        super().run_depth_only()

    def run_full(self):
        self.started_at = self.load_started_at
        super().run_full()

    def metrics_extra(self, depth_only, images_per_run=None):
        extra = {
            "Peak GPU Memory": base.peak_gpu_memory_mib(),
            "track": self.args.track,
            "image_load_resolution": self.args.image_load_resolution,
        }
        if depth_only:
            extra["max_images_per_run"] = images_per_run
        return extra
