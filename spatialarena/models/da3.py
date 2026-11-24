# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Depth Anything 3 (DA3-L, DA3-G, DA3-NGL).

One model class serves the three methods; the method name (run_model.py
--model) selects the checkpoint and the other defaults from its MethodSpec.

`DepthAnything3.inference` reads and resizes the images itself (to
--resolution), so preprocessing only plans the matching restoration
transforms. On a camera track (DA3-NGL's rgb_cameras) DA3 is conditioned on
the COLMAP camera priors of --colmap_path. --spatial_policy shared forces
DA3's upper_bound_resize, so depth and intrinsics can be restored to each
source image (depth-only).

--max_images_per_run is the images per model call of depth-only runs and of
full runs with camera priors; a full run without priors predicts the whole
scene in one call.
"""

from __future__ import annotations

import glob
import os
import sys
import types

import numpy as np
from PIL import Image

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

IMAGE_EXTENSIONS = (
    "*.jpg",
    "*.JPG",
    "*.jpeg",
    "*.JPEG",
    "*.png",
    "*.PNG",
    "*.gif",
    "*.GIF",
    "*.bmp",
    "*.BMP",
    "*.tiff",
    "*.TIFF",
    "*.webp",
    "*.WEBP",
)
DENSE_FIELDS = ("processed_images", "depth", "conf")
PREDICTION_FIELDS = DENSE_FIELDS + ("extrinsics", "intrinsics")


def find_image_paths(images_dir):
    """Returns every image under `images_dir`, recursively, sorted."""
    image_paths = []
    for extension in IMAGE_EXTENSIONS:
        pattern = os.path.join(str(images_dir), "**", extension)
        image_paths.extend(glob.glob(pattern, recursive=True))
    return sorted(image_paths)


def load_prior_model(args):
    """Reads the COLMAP model of the camera priors (--colmap_path).

    Args:
        args: The parsed options.

    Returns:
        A namespace of the model's cameras, images and points3D.

    Raises:
        FileNotFoundError: The scene has no prior model.
    """
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.geometry import read_write_model

    if getattr(args, "colmap_path", None) is not None:
        sparse_dir = args.colmap_path
    else:
        sparse_dir = utils.resolve_colmap_prior_path(
            args.data_path, colmap_model_subdir=args.colmap_model_subdir)
    if sparse_dir is None:
        raise FileNotFoundError(f"No COLMAP prior model for {args.data_path}")
    cameras, images, points3d = read_write_model.read_model(str(sparse_dir))
    return types.SimpleNamespace(cameras=cameras,
                                 images=images,
                                 points3D=points3d)


def center_crop_spatial(array, target_h, target_w):
    """Center-crops a batched N, H, W[, C] array to (target_h, target_w)."""
    if array.ndim < 3:
        return array
    height, width = array.shape[1], array.shape[2]
    if height == target_h and width == target_w:
        return array
    if target_h > height or target_w > width:
        raise ValueError(f"Cannot crop spatial shape {(height, width)} to "
                         f"larger target {(target_h, target_w)}")
    y0 = (height - target_h) // 2
    x0 = (width - target_w) // 2
    return array[:, y0:y0 + target_h, x0:x0 + target_w, ...]


def concatenate_prediction_batches(prediction):
    """Concatenates per-batch DA3 outputs whose crop sizes may differ.

    The dense maps are center-cropped to the smallest batch size first.
    """
    dense_batches = [
        array for field in DENSE_FIELDS for array in getattr(prediction, field)
    ]
    target_h = min(a.shape[1] for a in dense_batches if a.ndim >= 3)
    target_w = min(a.shape[2] for a in dense_batches if a.ndim >= 3)
    for field in DENSE_FIELDS:
        cropped = [
            center_crop_spatial(array, target_h, target_w)
            for array in getattr(prediction, field)
        ]
        setattr(prediction, field, np.concatenate(cropped, axis=0))
    prediction.extrinsics = np.concatenate(prediction.extrinsics, axis=0)
    prediction.intrinsics = np.concatenate(prediction.intrinsics, axis=0)


def _is_degenerate_alignment(error):
    message = str(error)
    return ("Degenerate covariance rank" in message or
            "Umeyama alignment is not possible" in message)


def run_depth_inference_with_camera_fallback(model, image_paths, extrinsics,
                                             intrinsics, process_res):
    """Runs DA3, without the camera priors if their alignment degenerates.

    Args:
        model: The DA3 model.
        image_paths: Images of the call.
        extrinsics: Camera priors, or None.
        intrinsics: Intrinsics priors, or None.
        process_res: DA3's processing resolution.

    Returns:
        DA3's prediction.
    """
    try:
        return model.inference(image_paths,
                               extrinsics=extrinsics,
                               intrinsics=intrinsics,
                               process_res=process_res)
    except Exception as error:  # pylint: disable=broad-exception-caught
        if extrinsics is None or not _is_degenerate_alignment(error):
            raise
    print(
        "WARNING: DA3 camera-prior alignment was degenerate; rerunning "
        "this depth batch without camera priors.",
        file=sys.stderr)
    return model.inference(image_paths,
                           extrinsics=None,
                           intrinsics=None,
                           process_res=process_res)


def is_nested_da3(model):
    """Returns whether `model` is the nested (metric) DA3.

    The nested model fits one depth scale and one sky depth over its whole
    input, so a batch of images would not be independent samples.
    """
    network = getattr(model, "model", None)
    return type(network).__name__ == "NestedDepthAnything3Net"


def _merge_single_image_predictions(predictions):
    """Concatenates one-image DA3 predictions into the first one."""
    merged = predictions[0]
    for field in ("depth", "conf", "extrinsics", "intrinsics",
                  "processed_images"):
        values = [
            getattr(prediction, field, None) for prediction in predictions
        ]
        present = [value is not None for value in values]
        if all(present):
            setattr(merged, field, np.concatenate(values, axis=0))
        elif any(present):
            raise ValueError(f"DA3 returned {field} for only some of the "
                             f"{len(values)} images")
    return merged


def run_single_view_inference(model, image_paths, process_res):
    """Predicts each image independently (MonoDepth).

    DA3's own pre- and post-processing are used, with the network run on
    [B, 1, 3, H, W] instead of one B-view sequence. The nested model runs
    one image per call instead.

    Args:
        model: The DA3 model.
        image_paths: Images of the call.
        process_res: DA3's processing resolution.

    Returns:
        DA3's prediction of all images.
    """
    if is_nested_da3(model):
        predictions = [
            model.inference([path], process_res=process_res)
            for path in image_paths
        ]
        return _merge_single_image_predictions(predictions)

    sequence_forward = model._run_model_forward

    def independent_forward(imgs, ex_t, in_t, *args, **kwargs):
        num_samples = imgs.shape[1]
        output = sequence_forward(imgs.transpose(0, 1), ex_t, in_t, *args,
                                  **kwargs)
        return utils.fold_single_view_outputs(output, num_samples)

    model._run_model_forward = independent_forward
    try:
        return model.inference(image_paths, process_res=process_res)
    finally:
        del model._run_model_forward  # Back to the class method.


def _slice_prediction(prediction, count):
    """Keeps the first `count` frames of a DA3 prediction."""
    for field in PREDICTION_FIELDS:
        setattr(prediction, field, getattr(prediction, field)[:count])
    return prediction


@registry.register_model
class DA3(base.BaseModel):
    """Depth Anything 3."""

    name = "DA3"
    aliases = ("DA3-L", "DA3-G", "DA3-NGL")
    description = "Run Depth Anything 3 and export benchmark artifacts."
    code_path_env = "DA3_CODE_PATH"
    ckpt_default = "depth-anything/DA3-LARGE-1.1"
    # DA3 loads and resizes the images itself: --resolution only.
    uses_image_load_resolution = False
    resolution_help = "DA3's processing resolution (process_res)."
    default_resolution = 480
    spatial_policies = ("legacy", "shared")
    tracks = ("rgb", "rgb_cameras")
    batch_size_from_env = True
    independent_single_view = True

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--align_scale_shift",
            action="store_true",
            help="Align each depth map to the COLMAP prior points (full "
            "runs with camera priors).",
        )

    def __init__(self, args):
        super().__init__(args)
        self.extrinsics_prior = None
        self.intrinsics_prior = None
        self.prior_index = {}  # Image path -> row of the priors.

    @property
    def uses_camera_priors(self):
        """Whether the run is conditioned on COLMAP camera priors."""
        return "cameras" in self.args.track

    def select_images(self):
        args = self.args
        if self.uses_camera_priors:
            resolved = utils.resolve_camera_conditioned_inputs(
                args.data_path,
                colmap_path=getattr(args, "colmap_path", None),
                colmap_model_subdir=args.colmap_model_subdir,
            )
            image_paths = resolved.image_paths
            self.extrinsics_prior = np.linalg.inv(resolved.poses)
            self.intrinsics_prior = resolved.intrinsics
        else:
            image_paths = find_image_paths(self.images_dir)
        if not image_paths:
            return []

        # The total-frame cap is BENCHMARK_MAX_INPUT_FRAMES, as for every
        # model (utils.select_interval); --max_images_per_run is a batch
        # size here, not a further cap.
        indices = utils.resolve_capped_image_indices(len(image_paths),
                                                     int(args.interval))
        subset = set(
            utils.restrict_to_image_subset(
                [image_paths[index] for index in indices]))
        indices = [index for index in indices if image_paths[index] in subset]
        image_paths = [image_paths[index] for index in indices]
        if self.extrinsics_prior is not None:
            self.extrinsics_prior = self.extrinsics_prior[indices, :, :]
            self.intrinsics_prior = self.intrinsics_prior[indices, :, :]
        self.prior_index = {path: i for i, path in enumerate(image_paths)}
        return image_paths

    def load(self, device):
        # pylint: disable-next=import-outside-toplevel
        from depth_anything_3 import api

        model = api.DepthAnything3.from_pretrained(self.args.ckpt_path)
        model = model.to(device=device)
        original_inference = model.inference
        shared = self.is_shared_policy

        def inference_recording_input_size(*args, **kwargs):
            if shared:
                kwargs["process_res_method"] = "upper_bound_resize"
            result = original_inference(*args, **kwargs)
            processed = getattr(result, "processed_images", None)
            if processed is not None and np.ndim(processed) == 4:
                utils.record_model_input(
                    np.shape(processed)[2],
                    np.shape(processed)[1])
            return result

        model.inference = inference_recording_input_size
        print(f"Processing {len(self.image_paths)} images...")
        return model

    def _priors(self, image_paths):
        """Returns the (extrinsics, intrinsics) priors of the images."""
        if self.extrinsics_prior is None:
            return None, None
        indices = [self.prior_index[path] for path in image_paths]
        return self.extrinsics_prior[indices], self.intrinsics_prior[indices]

    def preprocess(self, image_paths, batch_index=None):
        transforms = None
        if self.is_shared_policy:
            sizes = [Image.open(path).size for path in image_paths]
            if len(set(sizes)) != 1:
                raise ValueError("Shared DA3 mapping rejects mixed shapes "
                                 "within one depth-only batch")
            transforms = [
                utils.plan_spatial_transform(size,
                                             method="da3",
                                             size=self.args.resolution)
                for size in sizes
            ]
        extrinsics, intrinsics = self._priors(image_paths)
        return base.ModelInput(
            image_paths=image_paths,
            data=image_paths,  # DA3 reads the images itself.
            transforms=transforms,
            batch_index=batch_index,
            extra={
                "extrinsics": extrinsics,
                "intrinsics": intrinsics
            },
        )

    def forward(self, inputs):
        extrinsics = inputs.extra["extrinsics"]
        intrinsics = inputs.extra["intrinsics"]
        if utils.single_view_mode():
            if self.uses_camera_priors:
                raise ValueError("single-view (MonoDepth) inference takes no "
                                 "camera priors")
            raw = run_single_view_inference(self.model, inputs.image_paths,
                                            self.args.resolution)
        else:
            raw = run_depth_inference_with_camera_fallback(
                self.model, inputs.image_paths, extrinsics, intrinsics,
                self.args.resolution)

        if extrinsics is None:
            extrinsics = raw.extrinsics
        if intrinsics is None:
            intrinsics = raw.intrinsics
        return base.Prediction(
            depth=raw.depth,
            confidence=raw.conf,
            extrinsics_w2c=extrinsics,
            intrinsics=intrinsics,
            extra={"raw": raw},
        )

    def metrics_extra(self, depth_only, images_per_run=None):
        extra = super().metrics_extra(depth_only, images_per_run)
        if depth_only:
            extra["nested_one_image_per_call"] = (extra["single_view"] and
                                                  is_nested_da3(self.model))
            extra["camera_priors"] = self.uses_camera_priors
        return extra

    # ------------------------------------------------------------------
    # Full (not depth-only) export: DA3's own dense points and cameras.
    # ------------------------------------------------------------------

    def run_full(self):
        """Exports DA3's dense points, depth and COLMAP cameras."""
        args = self.args
        if self.uses_camera_priors:
            prediction = self._infer_with_priors_in_batches()
        else:
            prediction = self.model.inference(
                self.image_paths,
                extrinsics=self.extrinsics_prior,
                intrinsics=self.intrinsics_prior,
                process_res=args.resolution,
            )
        runtime = utils.runtime_metrics(self.started_at)
        metrics_extra = self.metrics_extra(depth_only=False)

        extrinsics = prediction.extrinsics
        if self.extrinsics_prior is not None:
            extrinsics = self.extrinsics_prior
        points, colors = self._dense_points(prediction, extrinsics)
        self._export_full(prediction, extrinsics, points, colors)
        self.write_metrics(depth_only=False,
                           metrics_extra=metrics_extra,
                           runtime=runtime)

    def _infer_with_priors_in_batches(self):
        """Predicts the whole scene with camera priors.

        Calls take --max_images_per_run images; a short last batch is
        padded with frames from the start so every call has the same size.

        Returns:
            The concatenated prediction.
        """
        image_paths = self.image_paths
        per_batch = self.args.max_images_per_run
        if per_batch <= 0:
            per_batch = len(image_paths)
        prediction = types.SimpleNamespace(
            **{field: [] for field in PREDICTION_FIELDS})

        for start in range(0, len(image_paths), per_batch):
            batch_paths = image_paths[start:start + per_batch]
            extrinsics = self.extrinsics_prior[start:start + per_batch]
            intrinsics = self.intrinsics_prior[start:start + per_batch]
            count = len(batch_paths)
            needed = per_batch - count
            if needed > 0:
                print(f"Last batch has only {count} images, padding with "
                      f"images from beginning to reach {per_batch}")
                batch_paths = batch_paths + image_paths[:needed]
                extrinsics = np.concatenate(
                    [extrinsics, self.extrinsics_prior[:needed]], axis=0)
                intrinsics = np.concatenate(
                    [intrinsics, self.intrinsics_prior[:needed]], axis=0)

            batch_prediction = self.model.inference(
                batch_paths,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                process_res=self.args.resolution,
            )
            if needed > 0:
                batch_prediction = _slice_prediction(batch_prediction, count)
            for field in PREDICTION_FIELDS:
                getattr(prediction,
                        field).append(getattr(batch_prediction, field))

        concatenate_prediction_batches(prediction)
        return prediction

    def _dense_points(self, prediction, extrinsics):
        """Returns the (points, colors) of the dense point cloud.

        With camera priors the points are sampled for 3DGS from (optionally
        aligned) depth; otherwise confident pixels are kept at random.
        """
        # pylint: disable=import-outside-toplevel
        from spatialarena.geometry import geometry
        from spatialarena.geometry import utils as geometry_utils

        args = self.args
        # Drawn in both cases, so the random state matches the original
        # adapter's.
        conf_mask = geometry_utils.randomly_limit_trues(
            prediction.conf >= self.conf_thresh, args.max_points)
        if self.uses_camera_priors:
            return self._sample_points_for_3dgs(prediction, extrinsics)

        points = geometry.unproject_depth_map_to_point_map(
            prediction.depth, extrinsics, prediction.intrinsics)
        colors = prediction.processed_images.astype(np.uint8)
        return points[conf_mask], colors[conf_mask]

    def _sample_points_for_3dgs(self, prediction, extrinsics):
        """Samples the dense points of a camera-conditioned prediction."""
        # pylint: disable=import-outside-toplevel
        from spatialarena.geometry import geometry
        from spatialarena.utils import dense_points_sample_for_gs

        args = self.args
        depths = prediction.depth
        if args.align_scale_shift:
            depths = self._align_depths_to_prior(prediction)
        points_3d = geometry.unproject_depth_map_to_point_map(
            depths, extrinsics, prediction.intrinsics)

        sampler = dense_points_sample_for_gs.PointSamplerFor3DGS(
            max_points=args.max_points,
            edge_weight=3.0,
            corner_weight=5.0,
            use_confidence=True,
            confidence_threshold=self.conf_thresh,
            use_redundancy_filter=True,
            voxel_size=0.01,
        )
        images = prediction.processed_images
        points, colors = sampler.sample_points_from_views(
            points_3d=points_3d,
            colors=images,
            images=images,
            confidence_maps=prediction.conf,
            masks=None,
        )
        print(f"sampled_points shape: {points.shape}")
        return points, colors

    def _align_depths_to_prior(self, prediction):
        """Returns the depth maps aligned in scale and shift to the prior."""
        # pylint: disable-next=import-outside-toplevel
        from spatialarena.utils import depth_scale_shift_estimator as estimator

        prior_model = load_prior_model(self.args)
        names = [os.path.basename(path) for path in self.image_paths]
        print("Estimating global scale and shift...")
        aligned, _ = estimator.estimate_and_apply_local_alignment(
            cameras=prior_model.cameras,
            images=prior_model.images,
            points3d=prior_model.points3D,
            depth_predictions=dict(zip(names, prediction.depth)),
            model_resolution=prediction.processed_images.shape[1:3],
            outlier_rejection=True,
            n_jobs=-1,
        )
        return np.stack([aligned[name] for name in names], axis=0)

    def _export_full(self, prediction, extrinsics, points, colors):
        """Writes the dense points, depth and COLMAP cameras."""
        # pylint: disable=import-outside-toplevel
        from spatialarena.geometry import utils as geometry_utils
        from spatialarena.utils import point_utils

        args = self.args
        point_utils.export_ply(
            os.path.join(args.output_path, "points3D_dense.ply"), points,
            colors)
        print(f"Number of 3D points for 3DGS: {points.shape[0]}")
        if args.export_depth:
            utils.export_depth_predictions(args.output_path, self.image_paths,
                                           prediction.depth, prediction.conf)
        if not args.export_colmap:
            return

        colmap_dir = os.path.join(args.output_path, "sparse",
                                  args.colmap_model_subdir)
        os.makedirs(colmap_dir, exist_ok=True)
        _, grid_h, grid_w, _ = prediction.processed_images.shape
        geometry_utils.write_colmap_formats(
            str(self.images_dir),
            extrinsics,
            prediction.intrinsics,
            self.image_paths,
            points,
            colors,
            width=grid_w,
            height=grid_h,
            output_path=colmap_dir,
        )
