# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Fast3R: all views in one forward pass.

Fast3R's view dicts (img/true_shape/idx/instance, as its vendored
`fast3r.dust3r.utils.image.load_images` returns them) are built from one
in-memory pad-to-square resize to --resolution (VGGT-Omega's plan), not
from that loader, which center-crops to a 16-pixel grid (and crops square
images to 4:3). There is no --spatial_policy option: depth and intrinsics
of depth-only runs are always restored to each source image.

The whole scene is one model call (--max_images_per_run only caps the
frames); in MonoDepth mode (BENCHMARK_SINGLE_VIEW=1) every image is its own
call. --depth_only does not require --export_depth.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils


def stack_spatial_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    """Stacks per-view arrays, center-cropping them to a common size.

    Args:
        arrays: Per-view (H, W[, ...]) arrays.

    Returns:
        The stacked (N, H, W[, ...]) array.

    Raises:
        ValueError: `arrays` is empty.
    """
    if not arrays:
        raise ValueError("No arrays to stack")
    if len({array.shape for array in arrays}) == 1:
        return np.stack(arrays, axis=0)
    if not all(array.ndim >= 2 for array in arrays):
        return np.stack(arrays, axis=0)
    target_h = min(array.shape[0] for array in arrays)
    target_w = min(array.shape[1] for array in arrays)
    cropped = []
    for array in arrays:
        height, width = array.shape[:2]
        top = max(0, (height - target_h) // 2)
        left = max(0, (width - target_w) // 2)
        cropped.append(array[top:top + target_h, left:left + target_w, ...])
    return np.stack(cropped, axis=0)


def stack_prediction_key(preds: list[dict], key: str) -> np.ndarray:
    """Stacks one output of Fast3R's per-view predictions.

    Args:
        preds: Fast3R's per-view prediction dicts.
        key: The output, e.g. "pts3d_in_other_view".

    Returns:
        The (N, H, W[, C]) array; singleton axes are dropped.
    """
    arrays = []
    for pred in preds:
        array = utils.as_numpy(pred[key])
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        arrays.append(array)
    return stack_spatial_arrays(arrays)


def colors_from_views(views: list[dict]) -> np.ndarray:
    """Returns the views' [-1, 1] images as (N, H, W, 3) uint8 colors.

    Args:
        views: Fast3R's view dicts.
    """
    colors = []
    for view in views:
        image = utils.as_numpy(view["img"])
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        colors.append(((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8))
    return stack_spatial_arrays(colors)


def preprocess_fast3r(image_paths: list[str], *, args, device):
    """Builds Fast3R's view dicts from a padded tensor.

    Args:
        image_paths: Source images.
        args: The parsed options (uses resolution).
        device: Device of the views.

    Returns:
        A (views, transforms) tuple.
    """
    # exif_transpose=True as fast3r.dust3r.utils.image.load_images does.
    images, transforms = utils.preprocess_images(image_paths,
                                                 method="vggt_omega",
                                                 size=args.resolution,
                                                 mode="pad",
                                                 exif_transpose=True)
    # Fast3R's ImgNorm is ToTensor ([0, 1]) + Normalize(0.5, 0.5) ([-1, 1]);
    # preprocess_images already returns [0, 1].
    images = (images * 2.0 - 1.0).to(device)
    views = []
    for index in range(images.shape[0]):
        frame = images[index:index + 1]
        height, width = frame.shape[-2:]
        views.append({
            "img": frame,
            "true_shape": np.int32([[height, width]]),
            "idx": index,
            "instance": str(index),
        })
    return views, transforms


def extend_view_position_embeddings(model, num_views: int) -> dict:
    """Extends Fast3R's view-position tables to `num_views` (> 1000) views.

    Args:
        model: The Fast3R network.
        num_views: Views per forward pass.

    Returns:
        The metrics.json fields recording the extension.
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from fast3r.croco.models import pos_embed

    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return {}

    metadata = {"num_views": int(num_views)}
    if hasattr(decoder,
               "image_idx_emb") and decoder.image_idx_emb.shape[0] < num_views:
        current = decoder.image_idx_emb
        extended = torch.from_numpy(
            pos_embed.get_1d_sincos_pos_embed_from_grid(current.shape[1],
                                                        np.arange(num_views)))
        decoder.image_idx_emb = extended.to(device=current.device,
                                            dtype=current.dtype)
        metadata["image_idx_emb_extended_from"] = int(current.shape[0])
        metadata["image_idx_emb_extended_to"] = int(num_views)

    if (hasattr(decoder, "precomputed_freqs_cis") and
            decoder.precomputed_freqs_cis.shape[0] < num_views and
            hasattr(decoder, "_precompute_freqs_cis")):
        current = decoder.precomputed_freqs_cis
        decoder.precomputed_freqs_cis = decoder._precompute_freqs_cis(
            num_views).to(device=current.device)
        metadata["precomputed_freqs_cis_extended_from"] = int(current.shape[0])
        metadata["precomputed_freqs_cis_extended_to"] = int(num_views)

    return metadata


def _concatenate(arrays):
    """Concatenates per-call arrays; None when any call returned None."""
    if any(array is None for array in arrays):
        return None
    return np.concatenate(arrays, axis=0)


@registry.register_model
class Fast3R(base.BaseModel):
    """Fast3R."""

    name = "FAST3R"
    description = "Run Fast3R benchmark adapter."
    code_path_env = "FAST3R_CODE_PATH"
    code_packages = ("fast3r",)
    ckpt_env = "FAST3R_CKPT_PATH"
    ckpt_default = "jedyang97/Fast3R_ViT_Large_512"
    # Only --resolution (below): the images are not loaded at a separate
    # --image_load_resolution.
    uses_image_load_resolution = False
    resolution_help = "Model input resolution (pad-to-square)."
    default_resolution = 512

    def __init__(self, args):
        super().__init__(args)
        self.position_metadata = {}

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--niter_pnp", type=int, default=100)
        parser.add_argument("--is_lightning_checkpoint", action="store_true")

    def select_device(self):
        import torch  # pylint: disable=import-outside-toplevel

        return torch.device(super().select_device())

    def load(self, device):
        # Also extends Fast3R's view tables to the scene's views.
        # pylint: disable-next=import-outside-toplevel
        from fast3r.utils import checkpoint_utils

        model, _ = checkpoint_utils.load_model(
            self.args.ckpt_path,
            device=device,
            is_lightning_checkpoint=self.args.is_lightning_checkpoint,
        )
        num_views = 1 if utils.single_view_mode() else len(self.image_paths)
        self.position_metadata = extend_view_position_embeddings(
            model, num_views)
        return model

    def preprocess(self, image_paths, batch_index=None):
        views, transforms = preprocess_fast3r(image_paths,
                                              args=self.args,
                                              device=self.device)
        utils.record_model_input(views[0]["img"].shape[-1],
                                 views[0]["img"].shape[-2])
        for index, (view, image_path) in enumerate(
                zip(views, image_paths, strict=True)):
            view.setdefault("dataset", self.args.data_path.parent.name)
            view.setdefault("label", self.args.data_path.name)
            view.setdefault("instance", pathlib.Path(image_path).name)
            view.setdefault("idx", index)
        return base.ModelInput(image_paths, views, transforms, batch_index)

    def forward(self, inputs):
        # pylint: disable=import-outside-toplevel
        import torch
        from fast3r.dust3r import inference_multiview
        from fast3r.models import multiview_dust3r_module

        output_dict, _ = inference_multiview.inference(
            inputs.data,
            self.model,
            self.device,
            dtype=torch.float32,
            verbose=True,
            profiling=True,
        )
        preds = output_dict["preds"]
        lit_module = multiview_dust3r_module.MultiViewDUSt3RLitModule
        poses_c2w_batch, estimated_focals = lit_module.estimate_camera_poses(
            preds,
            niter_PnP=self.args.niter_pnp,
            focal_length_estimation_method="first_view_from_global_head",
        )
        c2w = utils.as_numpy(poses_c2w_batch[0]).astype(np.float32, copy=False)
        extrinsics_w2c = utils.c2w_to_w2c(c2w)
        points = stack_prediction_key(preds, "pts3d_in_other_view")
        if "pts3d_local" in preds[0]:
            depths = stack_prediction_key(preds, "pts3d_local")[..., 2]
            confidence = None
            if "conf_local" in preds[0]:
                confidence = stack_prediction_key(preds, "conf_local")
        else:
            camera_points = np.einsum(
                "nij,nhwj->nhwi",
                extrinsics_w2c[:, :3, :3],
                points,
            ) + extrinsics_w2c[:, None, None, :3, 3]
            depths = camera_points[..., 2]
            confidence = None
            if "conf" in preds[0]:
                confidence = stack_prediction_key(preds, "conf")
        height, width = depths.shape[1:]
        intrinsics = utils.intrinsics_from_focals(estimated_focals[0],
                                                  height=height,
                                                  width=width)
        return base.Prediction(
            depth=depths,
            confidence=confidence,
            extrinsics_w2c=extrinsics_w2c,
            intrinsics=intrinsics,
            points=points,
            colors=colors_from_views(output_dict["views"]),
        )

    def predict_scene(self):
        """Runs the whole scene: one call, or one per image (MonoDepth).

        Returns:
            A (Prediction, transforms) tuple over all images.
        """
        if not utils.single_view_mode():
            prediction, inputs = self.predict(self.image_paths)
            return prediction, inputs.transforms
        # MonoDepth: every image is its own one-view input.
        predictions = []
        transforms = []
        for path in self.image_paths:
            prediction, inputs = self.predict([path])
            predictions.append(prediction)
            transforms.extend(inputs.transforms)

        def stacked(field):
            return _concatenate([getattr(p, field) for p in predictions])

        prediction = base.Prediction(
            depth=stacked("depth"),
            confidence=stacked("confidence"),
            extrinsics_w2c=stacked("extrinsics_w2c"),
            intrinsics=stacked("intrinsics"),
            points=stacked("points"),
            colors=stacked("colors"),
        )
        return prediction, transforms

    def run(self):
        """Runs Fast3R on the scene and writes its artifacts.

        Unlike the default, --depth_only does not require --export_depth,
        and the scene is never split into batches.

        Raises:
            FileNotFoundError: The scene has no images directory.
            ValueError: No images were found.
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
            print(json.dumps(self.planned_layout(), indent=2))
            return

        self.add_code_path()
        self.device = self.select_device()
        self.model = self.load(self.device)
        self.started_at = utils.mark_model_loaded()
        if args.depth_only:
            self.run_depth_only()
        else:
            self.run_full()

    def run_depth_only(self):
        """Exports depth and cameras restored to the source images."""
        args = self.args
        prediction, transforms = self.predict_scene()
        extra = self.metrics_extra(depth_only=True)
        if args.export_depth:
            utils.export_spatial_depth_predictions(args.output_path,
                                                   self.image_paths,
                                                   prediction.depth,
                                                   prediction.confidence,
                                                   transforms)
        if args.export_colmap:
            intrinsics, width, height = utils.restore_camera_export(
                prediction.intrinsics, transforms)
            utils.export_camera_pose_priors(
                args.output_path,
                self.images_dir,
                self.image_paths,
                prediction.extrinsics_w2c,
                intrinsics,
                width=width,
                height=height,
                colmap_model_subdir=args.colmap_model_subdir,
            )
        self.write_metrics(depth_only=True, metrics_extra=extra)

    def run_full(self):
        """Exports dense points, depth and cameras on the model grid."""
        args = self.args
        prediction, _ = self.predict_scene()
        utils.export_standard_artifacts(
            method=self.method_label,
            output_path=args.output_path,
            images_dir=self.images_dir,
            image_paths=self.image_paths,
            depths=prediction.depth,
            points=prediction.points,
            colors=prediction.colors,
            confidence=prediction.confidence,
            extrinsics_w2c=prediction.extrinsics_w2c,
            intrinsics=prediction.intrinsics,
            export_depth=args.export_depth,
            export_colmap=args.export_colmap,
            depth_only=args.depth_only,
            colmap_model_subdir=args.colmap_model_subdir,
            max_points=args.max_points,
            conf_thresh=self.conf_thresh,
            started_at=self.started_at,
            checkpoint=self.checkpoint,
            metrics_extra=self.metrics_extra(depth_only=False),
        )

    def metrics_extra(self, depth_only, images_per_run=None):
        del depth_only, images_per_run  # The same fields either way.
        extra = {
            "Peak GPU Memory": base.peak_gpu_memory_mib(),
            "resolution": self.args.resolution,
            "niter_pnp": self.args.niter_pnp,
            "is_lightning_checkpoint": self.args.is_lightning_checkpoint,
            "max_images_per_run": self.args.max_images_per_run,
            "single_view": utils.single_view_mode(),
        }
        extra.update(self.position_metadata)
        return extra
