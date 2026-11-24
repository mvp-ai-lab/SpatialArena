# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Co-Me (confidence-guided token merging), run through its Hydra entry point.

Co-Me runs in a subprocess, `python -m src.infer` in its checkout, on images
padded to a square on disk at --image_load_resolution (504); its raw
`sample_0_00.npz` is read back. No network is loaded in this process, and
the runtime metrics count from the adapter's start (runtime_scope
"adapter_started_at").

--spatial_policy legacy (default): depth stays on the padded square Co-Me
sees. --spatial_policy shared (depth-only): depth, confidence and
intrinsics are restored to each source image.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

# Co-Me's src.infer hardcodes this resize (see `run_come`).
COME_INFER_SIZE = 504


def internal_resolution_of(args) -> int:
    """Returns Co-Me's model resolution.

    That is --internal_resolution, or else the size nearest above
    --resolution on the merged (patch_size x token_merge_group) grid.

    Args:
        args: The parsed options.
    """
    if args.internal_resolution > 0:
        return args.internal_resolution
    return utils.patch_grid_compatible_resolution(
        args.resolution,
        patch_size=args.patch_size,
        grid_multiple=args.token_merge_group,
    )


def load_come_archive(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Reads Co-Me's raw output archive.

    Args:
        path: The archive (sample_0_00.npz).

    Returns:
        Its arrays by name.

    Raises:
        FileNotFoundError: The archive does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Co-Me raw output not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def run_come(args, image_paths: list[str], run_name: str = "") -> pathlib.Path:
    """Runs Co-Me's Hydra entry point on images in a subprocess.

    Args:
        args: The parsed options.
        image_paths: Source images.
        run_name: Suffix of the scratch directories (a depth-only batch).

    Returns:
        Path of the raw output archive.

    Raises:
        ValueError: No --code_path, or --image_load_resolution is not
            the 504 Co-Me resizes to.
        FileNotFoundError: The code path does not exist.
        subprocess.CalledProcessError: Co-Me failed.
    """
    if args.code_path is None:
        raise ValueError(
            "--code_path or COME_CODE_PATH is required for real Co-Me runs")
    code_path = args.code_path.expanduser().resolve()
    if not code_path.is_dir():
        raise FileNotFoundError(f"Co-Me code path not found: {code_path}")

    # Co-Me's src.infer hardcodes sample.multiview_input.resize(504, 504)
    # (cover, then center crop) and ignores the +infer_size override below.
    # The images are padded to an --image_load_resolution square first, so
    # that resize crops nothing only while that is 504.
    if args.image_load_resolution != COME_INFER_SIZE:
        raise ValueError(
            "Co-Me's own src.infer hardcodes .resize(504, 504) regardless of "
            "--image_load_resolution; changing --image_load_resolution away "
            "from 504 would make that resize crop real image content instead "
            "of a no-op. Leave it at the default (504) until that upstream "
            "hardcoding is fixed.")
    args.output_path.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_name}" if run_name else ""
    temp_images = args.output_path / f"_come_input_images{suffix}"
    raw_dir = args.output_path / f"_come_raw{suffix}"
    utils.materialize_resized_images(image_paths, temp_images,
                                     args.image_load_resolution)
    internal_resolution = internal_resolution_of(args)

    command = [
        sys.executable,
        "-m",
        "src.infer",
        "dataset=custom_image",
        f"model={args.come_model}",
        "render=save_raw",
        f"dataset.data_root={temp_images}",
        f"dataset.max_length={len(image_paths)}",
        "dataset.step_size=1",
        "dataset.start=0",
        "first_n=1",
        f"+infer_size=[{internal_resolution},{internal_resolution}]",
        f"output_directory={raw_dir}",
        f"hydra.run.dir={raw_dir / 'hydra'}",
    ]
    command.extend(args.come_extra_override)
    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    subprocess.run(command, cwd=code_path, env=env, check=True)
    return raw_dir / "sample_0_00.npz"


def preprocess_come(image_paths: list[str], *, internal_resolution: int):
    """Returns the transforms of Co-Me's input, source -> model grid.

    `run_come` pads every image onto a square on disk before Co-Me resizes
    that square to the internal resolution; a uniform scale of a square
    keeps the padding ratio, so one pad-to-square transform from the source
    size to `internal_resolution` describes both steps.

    Args:
        image_paths: Source images.
        internal_resolution: Co-Me's model resolution.

    Returns:
        The per-image `utils.SpatialTransform`s.
    """
    # pylint: disable=import-outside-toplevel
    from PIL import Image
    from PIL import ImageOps

    transforms = []
    for image_path in image_paths:
        with Image.open(image_path) as probe:
            source_wh = ImageOps.exif_transpose(probe).size
        transforms.append(
            utils.plan_spatial_transform(source_wh,
                                         method="vggt",
                                         size=internal_resolution,
                                         mode="pad"))
    return transforms


def come_intermediate_paths(args, raw_path: pathlib.Path) -> list:
    """Returns the scratch directories of the run that wrote `raw_path`.

    Args:
        args: The parsed options (uses output_path).
        raw_path: The run's raw archive.
    """
    raw_dir = raw_path.parent
    input_name = raw_dir.name.replace("_come_raw", "_come_input_images", 1)
    return [args.output_path / input_name, raw_dir]


def parse_scene_arrays(raw: dict[str, np.ndarray]) -> tuple:
    """Unpacks Co-Me's raw archive.

    Args:
        raw: The archive's arrays.

    Returns:
        A (depths, points, colors, confidence, extrinsics_w2c, intrinsics)
        tuple; an entry is None when the archive lacks it.
    """
    depths = None
    if "scene_depths" in raw:
        depths = raw["scene_depths"]
        if depths.ndim == 4 and depths.shape[1] == 1:
            depths = depths[:, 0]

    points = None
    if "scene_points" in raw:
        points = raw["scene_points"]
        if points.ndim == 4 and points.shape[1] == 3:
            points = np.transpose(points, (0, 2, 3, 1))

    confidence = None
    for key in ("scene_points_conf", "scene_depths_conf"):
        if key in raw:
            confidence = raw[key]
            if confidence.ndim == 4 and confidence.shape[1] == 1:
                confidence = confidence[:, 0]
            break

    colors = None
    if "input_images" in raw:
        colors = raw["input_images"]
        if colors.ndim == 4 and colors.shape[1] == 3:
            colors = np.transpose(colors, (0, 2, 3, 1))
        colors = (colors.clip(0.0, 1.0) * 255.0).astype(np.uint8)

    extrinsics_w2c = None
    if "scene_poses" in raw:
        extrinsics_w2c = utils.c2w_to_w2c(
            utils.se3_xyzw_to_c2w(raw["scene_poses"]))

    intrinsics = raw.get("scene_intrinsics")
    return depths, points, colors, confidence, extrinsics_w2c, intrinsics


@registry.register_model
class Come(base.BaseModel):
    """Co-Me, through its own Hydra entry point in a subprocess."""

    name = "COME"
    description = "Run Co-Me benchmark adapter."
    code_path_env = "COME_CODE_PATH"
    ckpt_env = "COME_CKPT_PATH"
    ckpt_default = ""
    default_resolution = 504
    default_image_load_resolution = 504
    spatial_policies = ("legacy", "shared")

    def __init__(self, args):
        super().__init__(args)
        self.raw_path = None

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--come_model",
            type=str,
            default=os.environ.get("COME_MODEL", "co_me_fused_vggt_3x3"),
            help=("Hydra model config name, e.g. co_me_fused_vggt_3x3 or "
                  "co_me_fused_pi3x_3x3."),
        )
        parser.add_argument(
            "--come_extra_override",
            action="append",
            default=[],
            help=("Additional Hydra override passed through to Co-Me. Can "
                  "be repeated."),
        )
        parser.add_argument(
            "--internal_resolution",
            type=int,
            default=0,
            help=("Co-Me model resolution. Defaults to the nearest larger "
                  "size whose VGGT patch grid is compatible with the "
                  "configured 3x3 merger."),
        )
        parser.add_argument("--patch_size", type=int, default=14)
        parser.add_argument("--token_merge_group", type=int, default=3)

    @property
    def checkpoint(self):
        return self.args.ckpt_path or None

    # Hooks around Co-Me's process and archive (patched in tests).

    def invoke(self, image_paths, run_name):
        """Runs Co-Me on images.

        Args:
            image_paths: Source images.
            run_name: Suffix of the scratch directories.

        Returns:
            The raw archive's path.
        """
        return run_come(self.args, image_paths, run_name=run_name)

    def read_raw(self, raw_path):
        """Returns the arrays of the raw archive.

        Args:
            raw_path: The archive.
        """
        return load_come_archive(raw_path)

    def intermediate_paths(self, raw_path):
        """Returns the scratch directories of one Co-Me run.

        Args:
            raw_path: The run's raw archive.
        """
        return come_intermediate_paths(self.args, raw_path)

    def export_cameras(self, extrinsics, intrinsics, width, height):
        """Writes the depth-only cameras as a COLMAP model.

        Args:
            extrinsics: World-to-camera matrices of all images.
            intrinsics: Their intrinsics.
            width: Image width the intrinsics refer to.
            height: Image height the intrinsics refer to.
        """
        utils.export_camera_pose_priors(
            self.args.output_path,
            self.images_dir,
            self.image_paths,
            extrinsics,
            intrinsics,
            width=width,
            height=height,
            colmap_model_subdir=self.args.colmap_model_subdir,
        )

    # BaseModel.

    def load(self, device):
        # The network loads in Co-Me's own process; the "model" here is
        # its Hydra config name.
        del device  # Unused.
        return self.args.come_model

    def preprocess(self, image_paths, batch_index=None):
        run_name = "" if batch_index is None else f"batch_{batch_index:04d}"
        return base.ModelInput(image_paths,
                               image_paths,
                               batch_index=batch_index,
                               extra={"run_name": run_name})

    def forward(self, inputs):
        raw_path = self.invoke(inputs.image_paths, inputs.extra["run_name"])
        self.raw_path = raw_path
        depths, points, colors, confidence, extrinsics_w2c, intrinsics = (
            parse_scene_arrays(self.read_raw(raw_path)))
        inputs.scratch_dirs.extend(self.intermediate_paths(raw_path))
        if self.is_shared_policy:
            inputs.transforms = preprocess_come(
                inputs.image_paths,
                internal_resolution=internal_resolution_of(self.args))
        return base.Prediction(
            depth=depths,
            confidence=confidence,
            extrinsics_w2c=extrinsics_w2c,
            intrinsics=intrinsics,
            points=points,
            colors=colors,
        )

    def run(self):
        """Runs Co-Me on the scene and writes its artifacts.

        Unlike the default, the dry run comes before the option checks and
        nothing is loaded in this process.

        Raises:
            FileNotFoundError: The scene has no images directory.
            ValueError: No images were found, or the options conflict.
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

        self.started_at = time.time()
        self.validate_args()
        if args.depth_only and not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")
        self.model = self.load(self.device)
        if args.depth_only:
            self.run_depth_only()
        else:
            self.run_full()

    def run_depth_only(self):
        """Exports depth (and cameras) per batch.

        The cameras keep the size of the first batch's cameras.

        Raises:
            ValueError: Co-Me returned no depth.
        """
        args = self.args
        images_per_run = self.images_per_run()
        extrinsics = []
        intrinsics = []
        size = None
        for batch_index, batch_paths in enumerate(
                self.depth_batches(images_per_run)):
            prediction, inputs = self.predict(batch_paths, batch_index)
            depths = prediction.depth
            if depths is None:
                raise ValueError(
                    "Co-Me raw output did not include depth predictions")
            has_cameras = (prediction.extrinsics_w2c is not None and
                           prediction.intrinsics is not None)
            if self.is_shared_policy:
                utils.export_spatial_depth_predictions(
                    args.output_path, batch_paths,
                    utils.normalize_depths(depths), prediction.confidence,
                    inputs.transforms)
                if args.export_colmap and has_cameras:
                    batch_intrinsics, width, height = (
                        utils.restore_camera_export(
                            np.asarray(prediction.intrinsics),
                            inputs.transforms))
                    extrinsics.append(np.asarray(prediction.extrinsics_w2c))
                    intrinsics.append(batch_intrinsics)
                    if size is None:
                        size = (width, height)
            else:
                utils.export_depth_predictions(args.output_path, batch_paths,
                                               depths, prediction.confidence)
                if args.export_colmap and has_cameras:
                    extrinsics.append(np.asarray(prediction.extrinsics_w2c))
                    intrinsics.append(np.asarray(prediction.intrinsics))
                    if size is None:
                        depth_array = np.asarray(depths)
                        size = (depth_array.shape[2], depth_array.shape[1])
            self.cleanup(inputs)
        if extrinsics:
            width, height = size
            self.export_cameras(np.concatenate(extrinsics, axis=0),
                                np.concatenate(intrinsics, axis=0), width,
                                height)
        extra = self.metrics_extra(depth_only=True,
                                   images_per_run=images_per_run)
        self.write_metrics(depth_only=True, metrics_extra=extra)

    def metrics_extra(self, depth_only, images_per_run=None):
        args = self.args
        extra = {"come_model": args.come_model}
        if depth_only:
            extra["max_images_per_run"] = images_per_run
        else:
            extra["raw_output"] = str(self.raw_path)
            extra["max_images_per_run"] = args.max_images_per_run
        extra.update({
            "resolution": args.resolution,
            "image_load_resolution": args.image_load_resolution,
            "internal_resolution": internal_resolution_of(args),
            "patch_size": args.patch_size,
            "token_merge_group": args.token_merge_group,
            "resized_input": True,
        })
        if depth_only:
            extra.update(
                utils.spatial_policy_metrics(args.spatial_policy,
                                             is_shared=self.is_shared_policy))
        return extra
