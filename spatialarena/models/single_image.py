# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Shared loop of the official single-image MonoDepth baselines.

MoGe, Marigold, Lotus, Depth Pro, UniDepth V2 and InfiniDepth each take one
RGB image and predict depth on that image's own pixel grid, so a model only
supplies `SingleImageDepthModel.infer(rgb, image_path) -> (depth,
raw_disparity or None)`. This module selects the frames (`select_interval`,
which honours the engine's BENCHMARK_IMAGE_SUBSET size groups), checks every
prediction is exactly the source image's (H, W) -- a model that changed the
grid (e.g. an EXIF-rotated input) fails the case instead of being silently
resized -- and exports depths/<stem>.npz (plus raw_disparities/<stem>.npz for
disparity models, read by the inverse-depth protocols) and metrics.json.

Each method is its own registered model (MOGE, MOGE2, MARIGOLD_V1_1, ...):
the method name picks the variant and its MethodSpec the defaults of
--ckpt_path and --code_path. The models resize on their own, so they have
no --resolution, --spatial_policy or dense-point options; --seed (default
42) seeds the method's RNG, and every image is its own call (BaseModel's
--max_images_per_run does not batch or cap them).

scripts/inference/single_image_depth.py is an alias of this module, so the
function API (`select_images`, `run_single_image_depth`,
`resolve_checkpoint_file`, `load_rgb`, ...) stays available to the scripts.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Callable

import numpy as np
from PIL import Image

from spatialarena.models import base
from spatialarena.models import utils

Predictor = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray | None]"]


def select_images(args: argparse.Namespace) -> list[str] | None:
    """Returns the frames this call predicts, as absolute paths.

    Args:
        args: The parsed options.

    Returns:
        The image paths, or None after a --dry_run report.

    Raises:
        ValueError: --export_depth is missing, --export_colmap is given, or
            the scene has no images.
    """
    if args.export_colmap or not args.export_depth:
        raise ValueError(f"{args.method} predicts dense depth only: pass "
                         "--export_depth, not --export_colmap")
    images_dir = utils.canonical_image_root(args.data_path)
    image_paths = [
        str(pathlib.Path(path).resolve()) for path in utils.select_interval(
            utils.find_image_paths(images_dir), args.interval)
    ]
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        utils.print_dry_run(args, args.method, len(image_paths))
        return None
    return image_paths


def resolve_checkpoint_file(ckpt_path: str, filename: str) -> str:
    """Returns a local checkpoint file.

    Args:
        ckpt_path: The file itself, a directory holding `filename`, or a
            Hugging Face repo id (downloaded to the cache).
        filename: The file's name in the directory or repo.

    Returns:
        The file's path.
    """
    path = pathlib.Path(ckpt_path).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        return str(path / filename)
    import huggingface_hub  # pylint: disable=import-outside-toplevel

    return huggingface_hub.hf_hub_download(ckpt_path, filename)


def load_rgb(path: str) -> np.ndarray:
    """Loads an image as H x W x 3 uint8.

    The image keeps the file's stored orientation (no EXIF transpose), like
    the GT depth maps.

    Args:
        path: The image file.

    Returns:
        The RGB array.
    """
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def predict_and_export(
    args: argparse.Namespace,
    image_paths: list[str],
    predict: Callable[[str, np.ndarray], tuple[Any, Any]],
    *,
    metrics_extra: dict[str, Any] | None = None,
) -> None:
    """Predicts each image on its own and exports depth on its source grid.

    Args:
        args: The parsed options.
        image_paths: The images.
        predict: (image_path, rgb) -> (depth, raw disparity or None), both
            on the image's own grid.
        metrics_extra: The method's own metrics.json fields.

    Raises:
        ValueError: A prediction is not on the source image's grid.
    """
    args.output_path.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_path / "raw_disparities"
    started_at = utils.mark_model_loaded()
    nonfinite = 0
    for index, path in enumerate(image_paths):
        rgb = load_rgb(path)
        depth, raw = predict(path, rgb)
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        for name, array in (("depth", depth), ("raw disparity", raw)):
            if array is not None and np.asarray(
                    array).squeeze().shape != rgb.shape[:2]:
                raise ValueError(f"{args.method} {name} for {path} has shape "
                                 f"{np.asarray(array).squeeze().shape}, "
                                 f"not the source image's {rgb.shape[:2]}")
        nonfinite += int((~np.isfinite(depth)).sum())
        utils.export_depth_predictions(args.output_path, [path], depth[None])
        if raw is not None:
            raw_dir.mkdir(exist_ok=True)
            utils.save_prediction_map(
                raw_dir,
                pathlib.Path(path).stem,
                np.asarray(raw, dtype=np.float32).squeeze())
        print(
            f"[{args.method}] {index + 1}/{len(image_paths)} "
            f"{pathlib.Path(path).name}",
            flush=True)
    metrics = {
        "track": args.track,
        "seed": args.seed,
        "nonfinite_depth_pixels": nonfinite,
    }
    metrics.update(metrics_extra or {})
    utils.write_adapter_metrics(
        output_path=args.output_path,
        method=args.method,
        image_paths=image_paths,
        started_at=started_at,
        checkpoint=args.ckpt_path,
        depth_only=True,
        metrics_extra=metrics,
    )


def run_single_image_depth(
    args: argparse.Namespace,
    image_paths: list[str],
    predict: Predictor,
    *,
    metrics_extra: dict[str, Any] | None = None,
) -> None:
    """Predicts each image on its own and exports depth on its source grid.

    Args:
        args: The parsed options.
        image_paths: The images.
        predict: rgb -> (depth, raw disparity or None), both on the image's
            own grid.
        metrics_extra: The method's own metrics.json fields.
    """
    predict_and_export(args,
                       image_paths,
                       lambda _path, rgb: predict(rgb),
                       metrics_extra=metrics_extra)


class SingleImageDepthModel(base.BaseModel):
    """An official single-image MonoDepth baseline.

    Subclasses implement `load` and `infer`; `metrics_extra` adds the
    method's own metrics.json fields. `self.method_name` is the method a
    model that serves several (e.g. MoGe-1/-2/-3) runs.
    """

    uses_resolution = False
    uses_image_load_resolution = False
    exports_points = False
    default_seed = 42
    requires_cuda = True

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        if getattr(args, "output_path", None) is not None:
            args.output_path = pathlib.Path(args.output_path).resolve()

    def infer(self, rgb: np.ndarray, image_path: str) -> tuple[Any, Any]:
        """Predicts the depth of one image.

        Args:
            rgb: The image, H x W x 3 uint8 (see `load_rgb`).
            image_path: Its file.

        Returns:
            A (depth, raw disparity or None) tuple on the image's own grid.
        """
        raise NotImplementedError

    def preprocess(self,
                   image_paths: list[str],
                   batch_index: int | None = None) -> base.ModelInput:
        """Loads the one image of a call (see `load_rgb`).

        Args:
            image_paths: The one image of the call.
            batch_index: Unused.

        Returns:
            The `base.ModelInput`, with the RGB array as data.

        Raises:
            ValueError: The call has more than one image.
        """
        del batch_index  # One image per call.
        if len(image_paths) != 1:
            raise ValueError(f"{self.name} predicts one image per call")
        return base.ModelInput(list(image_paths), data=load_rgb(image_paths[0]))

    def forward(self, inputs: base.ModelInput) -> base.Prediction:
        """Runs `infer`.

        Args:
            inputs: The output of `preprocess`.

        Returns:
            The `base.Prediction`: the depth as the method's code returned
            it, and the raw disparity (or None) as extra["raw_disparity"].
        """
        depth, raw = self.infer(inputs.data, inputs.image_paths[0])
        return base.Prediction(depth=depth, extra={"raw_disparity": raw})

    def predict(self, image_paths: list[str], batch_index: int | None = None):
        """Runs one model call, loading the model on first use.

        Unlike `base.BaseModel.predict`, no model input size is recorded:
        the methods resize internally.

        Args:
            image_paths: The one image of the call.
            batch_index: Unused.

        Returns:
            A (Prediction, ModelInput) tuple.
        """
        if self.model is None:
            self.model = self.load(self.device)
        inputs = self.preprocess(image_paths, batch_index=batch_index)
        return self.forward(inputs), inputs

    def predict_image(self, image_path: str,
                      rgb: np.ndarray) -> tuple[Any, Any]:
        """Predicts one loaded image.

        Args:
            image_path: The image file.
            rgb: The image (see `load_rgb`).

        Returns:
            A (depth, raw disparity or None) tuple on the image's own grid.
        """
        inputs = base.ModelInput([image_path], data=rgb)
        prediction = self.forward(inputs)
        return prediction.depth, prediction.extra["raw_disparity"]

    def metrics_extra(self,
                      depth_only: bool = True,
                      images_per_run: int | None = None) -> dict[str, Any]:
        """Returns the method's own metrics.json fields. Default: none.

        Args:
            depth_only: Unused (always a depth-only run).
            images_per_run: Unused (always one image per call).

        Returns:
            The fields.
        """
        del depth_only, images_per_run  # Always one image per call.
        return {}

    def run(self) -> None:
        """Predicts every selected image and writes depths/ and metrics.json.

        Raises:
            ValueError: --export_depth is missing, --export_colmap is given,
                the scene has no images, or a prediction is off the source
                grid.
        """
        image_paths = select_images(self.args)
        if image_paths is None:
            return
        self.image_paths = image_paths
        self.add_code_path()
        self.device = self.select_device()
        self.model = self.load(self.device)
        predict_and_export(self.args,
                           image_paths,
                           self.predict_image,
                           metrics_extra=self.metrics_extra())
