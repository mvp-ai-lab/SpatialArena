# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Base class of the inference models in the evaluation pipeline.

A model subclasses `BaseModel` and implements three methods:

    load(device)         Builds the network on `device`.
    preprocess(paths)    Turns source images into a `ModelInput`.
    forward(inputs)      Runs the network and returns a `Prediction`.

`BaseModel` does everything else the same way for every method: the
command line of the evaluation engine, frame selection, the dry run, the
timers, the depth-only batch loop, restoring predictions to the source
image size, and writing the artifacts and metrics.json.

Typical usage:

    model = registry.get_model("VGGT").from_argv(argv)
    model.run()
"""

from __future__ import annotations

import abc
import argparse
import dataclasses
import json
import os
import pathlib
import sys
from typing import Any, ClassVar, Sequence

import numpy as np

from spatialarena.models import utils


@dataclasses.dataclass
class ModelInput:
    """The input of one model call.

    Attributes:
        image_paths: Source images the call covers; outputs are named after
            them.
        data: What the method's own code takes (a tensor, file paths, ...).
        transforms: Per-image `utils.SpatialTransform` from the source image
            to the model grid, or None when the mapping cannot be undone
            (the method's own legacy resize).
        batch_index: Number of the depth-only batch, or None for the single
            whole-scene call.
        scratch_dirs: Directories removed after the batch is exported.
        extra: Method-specific state passed from preprocess to forward.
    """

    image_paths: list[str]
    data: Any = None
    transforms: list[utils.SpatialTransform] | None = None
    batch_index: int | None = None
    scratch_dirs: list[pathlib.Path] = dataclasses.field(default_factory=list)
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Prediction:
    """The outputs of one model call, on the model grid.

    Every field is a numpy array with one entry per frame, or None when the
    method does not produce it.

    Attributes:
        depth: Depth maps, (N, H, W).
        confidence: Confidence maps, (N, H, W).
        extrinsics_w2c: World-to-camera matrices, (N, 3, 4) or (N, 4, 4).
        intrinsics: Pinhole intrinsics, (N, 3, 3).
        points: World points, (N, H, W, 3).
        colors: Point colors, (N, H, W, 3), uint8.
        extra: Method-specific outputs.
    """

    depth: Any = None
    confidence: Any = None
    extrinsics_w2c: Any = None
    intrinsics: Any = None
    points: Any = None
    colors: Any = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


class CameraExport:
    """Cameras collected over the depth-only batches.

    They are written once, at the end, as a COLMAP model without points.
    """

    def __init__(self):
        self.image_paths = []
        self.extrinsics = []
        self.intrinsics = []
        self.size = None

    def add(self, image_paths, extrinsics, intrinsics, size, restored):
        """Adds the cameras of one batch.

        Args:
            image_paths: The batch's source images.
            extrinsics: World-to-camera matrices of the batch.
            intrinsics: Intrinsics of the batch.
            size: (width, height) the intrinsics refer to.
            restored: Whether the intrinsics were restored to the source
                images. The size of the last restored batch is kept;
                otherwise the size of the first batch.
        """
        self.image_paths.extend(image_paths)
        self.extrinsics.append(np.asarray(extrinsics))
        self.intrinsics.append(np.asarray(intrinsics))
        if restored or self.size is None:
            self.size = size

    def write(self, output_path, images_dir, colmap_model_subdir):
        """Writes the collected cameras, if any, to sparse/<subdir>."""
        if not self.extrinsics:
            return
        width, height = self.size
        utils.export_camera_pose_priors(
            output_path,
            images_dir,
            self.image_paths,
            np.concatenate(self.extrinsics, axis=0),
            np.concatenate(self.intrinsics, axis=0),
            width=width,
            height=height,
            colmap_model_subdir=colmap_model_subdir,
        )


def expand_code_root(value: str) -> str:
    """Expands $METHOD_CODE_ROOT (and other variables) in a code path.

    METHOD_CODE_ROOT defaults to $SPATIALARENA_ROOT/benchmarks/_model_codes,
    as in the evaluation engine.
    """
    root = os.environ.get("METHOD_CODE_ROOT")
    if not root:
        spatialarena_root = os.environ.get("SPATIALARENA_ROOT",
                                           "~/spatialarena")
        root = os.path.join(os.path.expanduser(spatialarena_root), "benchmarks",
                            "_model_codes")
    value = value.replace("${METHOD_CODE_ROOT}",
                          root).replace("$METHOD_CODE_ROOT", root)
    return os.path.expanduser(os.path.expandvars(value))


def peak_gpu_memory_mib() -> float:
    """Returns the peak GPU memory of this process in MiB (0 without CUDA)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024.0**2)


def empty_gpu_cache() -> None:
    """Releases cached GPU memory between batches."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class BaseModel(abc.ABC):
    """A method of the benchmark, run as one adapter call on one scene.

    Subclasses set the class attributes below and implement `load`,
    `preprocess` and `forward`. Other methods have defaults; override them
    only where the method really differs.

    Every model has the same command line (`add_common_arguments`): the
    options of all models, plus the option groups below that the model
    switches on, plus its own options (`add_arguments`). The defaults come
    from the model's `methods.MethodSpec` (the method registry, also what
    the evaluation engine passes); the `default_*` class attributes are only
    the fallback of a model without one.

    Attributes:
        name: Registry name; also the method run when no other is asked for.
        aliases: Other registry names.
        description: Description of the command line.
        metrics_name: "method" in metrics.json when it differs from the
            method name.
        code_path_env: Environment variable --code_path defaults to, without
            a MethodSpec.
        code_packages: Top-level packages of the method's code.
        ckpt_env: Environment variable --ckpt_path defaults to, without a
            MethodSpec.
        ckpt_default: Default of --ckpt_path, without a MethodSpec.
        ckpt_required: Whether --ckpt_path must be given (no default).
        uses_code_path: Whether the model has --code_path (False for one
            without a code checkout: an API, an installed package).
        uses_resolution: Whether the model has --resolution (the model
            input resolution).
        uses_image_load_resolution: Whether the model has
            --image_load_resolution (the resolution images are loaded at).
        resolution_help: Help of --resolution.
        image_load_resolution_help: Help of --image_load_resolution.
        exports_points: Whether the model has the dense point export options
            --max_points, --conf_thresh and --depth_conf_thresh.
        default_seed: Default of --seed; None when the model has no --seed.
        default_resolution: Default of --resolution, without a MethodSpec.
        default_image_load_resolution: Default of --image_load_resolution,
            without a MethodSpec.
        default_conf_thresh: Default of --conf_thresh, without a MethodSpec.
        default_max_images_per_run: Default of --max_images_per_run.
        default_max_points: Default of --max_points.
        spatial_policies: Choices of --spatial_policy, the default first.
            Empty when the method has no such option.
        tracks: Choices of --track, the default first, without a MethodSpec.
            A track with camera priors adds --colmap_path.
        batch_size_from_env: Whether DEPTH_INFERENCE_BATCH_SIZE overrides
            --max_images_per_run.
        independent_single_view: Whether single-view (MonoDepth) mode runs
            a batch as independent samples rather than one image per call.
        requires_cuda: Whether the method cannot run on the CPU.
    """

    name: ClassVar[str] = ""
    aliases: ClassVar[tuple[str, ...]] = ()
    description: ClassVar[str] = ""
    metrics_name: ClassVar[str | None] = None

    code_path_env: ClassVar[str | None] = None
    code_packages: ClassVar[tuple[str, ...]] = ()
    ckpt_env: ClassVar[str | None] = None
    ckpt_default: ClassVar[str] = ""
    ckpt_required: ClassVar[bool] = False
    uses_code_path: ClassVar[bool] = True

    uses_resolution: ClassVar[bool] = True
    uses_image_load_resolution: ClassVar[bool] = True
    resolution_help: ClassVar[str] = "Model input resolution."
    image_load_resolution_help: ClassVar[str] = (
        "Resolution the images are loaded at.")
    exports_points: ClassVar[bool] = True
    default_seed: ClassVar[int | None] = None
    default_resolution: ClassVar[int] = 518
    default_image_load_resolution: ClassVar[int] = 518
    default_conf_thresh: ClassVar[float] = 0.0
    default_max_images_per_run: ClassVar[int] = 0
    default_max_points: ClassVar[int] = 1_000_000
    spatial_policies: ClassVar[tuple[str, ...]] = ()
    tracks: ClassVar[tuple[str, ...]] = ("rgb",)

    batch_size_from_env: ClassVar[bool] = False
    independent_single_view: ClassVar[bool] = False
    requires_cuda: ClassVar[bool] = False

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.method_name = getattr(args, "method", None) or self.name
        self.device = "cpu"
        self.model = None
        self.images_dir = None
        self.image_paths = []
        self.started_at = 0.0

    # ------------------------------------------------------------------
    # Command line.
    # ------------------------------------------------------------------

    @classmethod
    def method_spec(cls, method: str | None = None):
        """Returns the `methods.MethodSpec` of a method this model runs.

        Args:
            method: Method name or alias; None for the model's own name.

        Returns:
            The spec, or None when the method registry has none (a model
            registered without one).
        """
        # pylint: disable-next=import-outside-toplevel  # avoids an import cycle
        from spatialarena import methods

        try:
            return methods.resolve_method(method or cls.name)
        except KeyError:
            return None

    @classmethod
    def spec_default(cls, field: str, fallback: Any, method: str | None = None):
        """Returns a MethodSpec field, or `fallback` without a spec.

        Args:
            field: The `methods.MethodSpec` field.
            fallback: The value without a spec.
            method: Method name or alias; None for the model's own name.
        """
        spec = cls.method_spec(method)
        if spec is None:
            return fallback
        return getattr(spec, field)

    @classmethod
    def build_parser(cls, method: str | None = None) -> argparse.ArgumentParser:
        """Returns the parser of the common and model-specific options.

        Args:
            method: The method the parser is for (its MethodSpec gives the
                defaults); None for the model's own name.
        """
        description = cls.description or f"Run {cls.name} inference."
        parser = argparse.ArgumentParser(description=description)
        parser.set_defaults(method=cls._method_name(method))
        cls.add_common_arguments(parser, method)
        cls.add_arguments(parser)
        return parser

    @classmethod
    def _method_name(cls, method: str | None) -> str:
        """Returns the canonical name of `method` (default: the model's)."""
        spec = cls.method_spec(method)
        if spec is not None:
            return spec.name
        return cls.name

    @classmethod
    def add_common_arguments(cls,
                             parser: argparse.ArgumentParser,
                             method: str | None = None) -> None:
        """Adds the options every model has, and its option groups.

        Args:
            parser: The parser.
            method: The method whose MethodSpec gives the defaults.
        """
        spec = cls.method_spec(method)
        parser.add_argument(
            "--data_path",
            type=pathlib.Path,
            required=True,
            help="Scene root containing images/.",
        )
        parser.add_argument(
            "--output_path",
            type=pathlib.Path,
            required=True,
            help="Method scene output directory.",
        )
        ckpt_default = cls._default_ckpt_path(spec)
        parser.add_argument(
            "--ckpt_path",
            type=str,
            default=None if cls.ckpt_required else ckpt_default,
            required=cls.ckpt_required,
            help="Checkpoint file or directory, or Hugging Face model id.",
        )
        if cls.uses_code_path:
            parser.add_argument(
                "--code_path",
                type=pathlib.Path,
                default=cls._default_code_path(spec),
                help="Checkout of the method's code.",
            )
        tracks = tuple(spec.tracks) if spec is not None else cls.tracks
        parser.add_argument("--track", choices=tracks, default=tracks[0])
        parser.add_argument("--interval",
                            type=int,
                            default=1,
                            help="Use every n-th frame.")
        parser.add_argument(
            "--max_images_per_run",
            type=int,
            default=cls.default_max_images_per_run,
            help="Maximum images per model call (0: all).",
        )
        parser.add_argument("--export_depth", action="store_true")
        parser.add_argument("--export_colmap", action="store_true")
        parser.add_argument(
            "--depth_only",
            action="store_true",
            help="Only export depth (and cameras); no dense points.",
        )
        parser.add_argument("--colmap_model_subdir", type=str, default="0")
        parser.add_argument(
            "--keep_intermediate",
            action="store_true",
            help="Keep the scratch files of the run.",
        )
        parser.add_argument(
            "--dry_run",
            action="store_true",
            help="Print the planned outputs without loading the model.",
        )

        if cls.uses_resolution:
            parser.add_argument(
                "--resolution",
                type=int,
                default=cls._spec_or(spec, "input_resolution",
                                     cls.default_resolution),
                help=cls.resolution_help,
            )
        if cls.uses_image_load_resolution:
            parser.add_argument(
                "--image_load_resolution",
                type=int,
                default=cls._spec_or(spec, "image_load_resolution",
                                     cls.default_image_load_resolution),
                help=cls.image_load_resolution_help,
            )
        if cls.exports_points:
            parser.add_argument("--max_points",
                                type=int,
                                default=cls.default_max_points,
                                help="Maximum points of the dense export.")
            parser.add_argument(
                "--conf_thresh",
                type=float,
                default=cls._spec_or(spec, "conf_thresh",
                                     cls.default_conf_thresh),
                help="Confidence threshold of the dense export.",
            )
            # Unset by default, so that --conf_thresh alone takes effect; the
            # engine passes both (DEPTH_CONF_THRESH, the MethodSpec's
            # depth_conf_thresh).
            parser.add_argument(
                "--depth_conf_thresh",
                type=float,
                default=None,
                help="Overrides --conf_thresh when set.",
            )
        if cls.default_seed is not None:
            parser.add_argument("--seed", type=int, default=cls.default_seed)
        if any("cameras" in track for track in tracks):
            parser.add_argument(
                "--colmap_path",
                type=pathlib.Path,
                default=None,
                help="COLMAP model of the camera priors (camera tracks).",
            )
        if cls.spatial_policies:
            parser.add_argument(
                "--spatial_policy",
                choices=cls.spatial_policies,
                default=cls.spatial_policies[0],
                help=("'legacy': the method's own resize and export. "
                      "'shared*': the benchmark's resize, with depth and "
                      "intrinsics restored to the source images."),
            )

    @staticmethod
    def _spec_or(spec, field: str, fallback: Any) -> Any:
        """Returns `spec.<field>`, or `fallback` without a spec."""
        if spec is None:
            return fallback
        return getattr(spec, field)

    @classmethod
    def _default_ckpt_path(cls, spec=None) -> str:
        env = spec.ckpt_env if spec is not None else cls.ckpt_env
        default = spec.ckpt_default if spec is not None else cls.ckpt_default
        if env and os.environ.get(env):
            return os.environ[env]
        return expand_code_root(default) if default else ""

    @classmethod
    def _default_code_path(cls, spec=None) -> pathlib.Path | None:
        env = spec.code_env if spec is not None else cls.code_path_env
        if env and os.environ.get(env):
            return pathlib.Path(os.environ[env])
        if spec is not None and spec.code_default:
            return pathlib.Path(expand_code_root(spec.code_default))
        return None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Adds the model-specific options. Default: none."""

    @classmethod
    def from_argv(cls,
                  argv: Sequence[str] | None = None,
                  method: str | None = None) -> BaseModel:
        """Creates the model from command-line arguments.

        Args:
            argv: The arguments; None for sys.argv.
            method: The method to run (a name or alias this model serves);
                None for the model's own name.
        """
        return cls(cls.build_parser(method).parse_args(argv))

    @classmethod
    def main(cls,
             argv: Sequence[str] | None = None,
             method: str | None = None) -> None:
        """Runs the model as a command-line program."""
        cls.from_argv(argv, method).run()

    # ------------------------------------------------------------------
    # Methods every model implements.
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self, device: str) -> Any:
        """Builds the network with its weights.

        Args:
            device: "cuda" or "cpu".

        Returns:
            The network in eval mode on `device`.
        """

    @abc.abstractmethod
    def preprocess(self,
                   image_paths: list[str],
                   batch_index: int | None = None) -> ModelInput:
        """Turns source images into the model's input.

        Args:
            image_paths: Source images of one model call.
            batch_index: Number of the depth-only batch, or None.

        Returns:
            The `ModelInput` of the call.
        """

    @abc.abstractmethod
    def forward(self, inputs: ModelInput) -> Prediction:
        """Runs the network.

        Args:
            inputs: The output of `preprocess`.

        Returns:
            The `Prediction`, as numpy arrays on the model grid.
        """

    # ------------------------------------------------------------------
    # Methods with a default.
    # ------------------------------------------------------------------

    def postprocess(self, prediction: Prediction,
                    inputs: ModelInput) -> Prediction:
        """Fixes up a raw prediction before export. Default: no change."""
        del inputs  # Unused by default.
        return prediction

    def predict(self, image_paths: list[str], batch_index: int | None = None):
        """Runs one model call, loading the model on first use.

        Args:
            image_paths: Source images of the call.
            batch_index: Number of the depth-only batch, or None.

        Returns:
            A (Prediction, ModelInput) tuple.
        """
        if self.model is None:
            self.model = self.load(self.device)
        inputs = self.preprocess(image_paths, batch_index=batch_index)
        # Record the input size for metrics.json; a forward() that resizes
        # internally records its own size afterwards.
        shape = getattr(inputs.data, "shape", None)
        if shape is not None and len(shape) >= 3:
            utils.record_model_input(shape[-1], shape[-2])
        prediction = self.forward(inputs)
        return self.postprocess(prediction, inputs), inputs

    @property
    def is_shared_policy(self) -> bool:
        """Whether --spatial_policy is one of the restorable "shared*"."""
        policy = getattr(self.args, "spatial_policy", "legacy")
        return policy.startswith("shared")

    @property
    def method_label(self) -> str:
        """The "method" written to metrics.json and the dry run."""
        return self.metrics_name or self.method_name

    @property
    def checkpoint(self) -> str | None:
        """The checkpoint written to metrics.json."""
        return self.args.ckpt_path

    @property
    def conf_thresh(self) -> float:
        """The confidence threshold of the dense point export."""
        if getattr(self.args, "depth_conf_thresh", None) is not None:
            return self.args.depth_conf_thresh
        return self.args.conf_thresh

    def validate_args(self) -> None:
        """Rejects inconsistent options before anything is loaded."""
        utils.validate_shared_spatial_policy_depth_only(
            is_shared=self.is_shared_policy,
            depth_only=self.args.depth_only,
        )

    def select_images(self) -> list[str]:
        """Returns the scene's frames after interval and frame-cap sampling."""
        image_paths = utils.find_image_paths(self.images_dir)
        return utils.select_interval(image_paths, self.args.interval,
                                     self.args.max_images_per_run)

    def planned_layout(self) -> dict[str, Any]:
        """Returns the outputs a run would write (the dry run)."""
        return utils.planned_layout(self.args, self.method_label,
                                    len(self.image_paths))

    def add_code_path(self) -> None:
        """Puts the method's code on sys.path."""
        utils.add_code_path(getattr(self.args, "code_path", None),
                            self.code_packages)

    def select_device(self) -> str:
        """Returns "cuda" when available, else "cpu".

        Raises:
            RuntimeError: The method requires CUDA and none is available.
        """
        import torch  # pylint: disable=import-outside-toplevel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.requires_cuda and device != "cuda":
            raise RuntimeError(f"{self.name} requires CUDA.")
        return device

    def images_per_run(self) -> int:
        """Returns the number of images per depth-only model call."""
        images_per_run = self.args.max_images_per_run
        if images_per_run <= 0:
            images_per_run = len(self.image_paths)
        if self.batch_size_from_env:
            images_per_run = int(
                os.environ.get("DEPTH_INFERENCE_BATCH_SIZE", images_per_run))
            if images_per_run < 1:
                raise ValueError("DEPTH_INFERENCE_BATCH_SIZE/"
                                 "--max_images_per_run must be >= 1")
        return images_per_run

    def depth_batches(self, images_per_run: int) -> list[list[str]]:
        """Returns the image paths of each depth-only model call."""
        return utils.batched_image_paths(
            self.image_paths,
            images_per_run,
            independent_batch=self.independent_single_view,
        )

    def export_depth(self, inputs: ModelInput, prediction: Prediction) -> None:
        """Writes the depth and confidence maps of one model call.

        With `inputs.transforms` the maps are restored to the source images;
        otherwise `export_depth_legacy` writes them.
        """
        utils.export_depths_for_policy(
            self.args.output_path,
            inputs.image_paths,
            prediction.depth,
            prediction.confidence,
            inputs.transforms,
            legacy_export=lambda: self.export_depth_legacy(inputs, prediction),
        )

    def export_depth_legacy(self, inputs: ModelInput,
                            prediction: Prediction) -> None:
        """Writes the maps on the model grid, as the method returned them."""
        utils.export_depth_predictions(
            self.args.output_path,
            inputs.image_paths,
            prediction.depth,
            prediction.confidence,
        )

    def restore_depths(self, prediction: Prediction,
                       inputs: ModelInput) -> list[np.ndarray]:
        """Returns one depth map per image at its source size.

        Raises:
            ValueError: The input has no restorable transforms.
        """
        if inputs.transforms is None:
            raise ValueError(f"{self.name}: the input has no restorable "
                             "spatial transform (legacy policy)")
        depths = utils.normalize_depths(prediction.depth)
        return [
            utils.restore_prediction(depth, transform)
            for depth, transform in zip(depths, inputs.transforms)
        ]

    def legacy_camera_size(self, prediction: Prediction) -> tuple[int, int]:
        """Returns (width, height) of unrestored cameras: the depth grid."""
        depth = np.asarray(prediction.depth)
        return depth.shape[2], depth.shape[1]

    def cameras_for_export(self, inputs: ModelInput, prediction: Prediction):
        """Returns the cameras of one model call for the COLMAP export.

        Returns:
            None without cameras, else an (extrinsics, intrinsics, size,
            restored) tuple. The intrinsics are restored to the source
            images when the input has transforms.
        """
        if prediction.extrinsics_w2c is None or prediction.intrinsics is None:
            return None
        if inputs.transforms is None:
            size = self.legacy_camera_size(prediction)
            return (prediction.extrinsics_w2c, prediction.intrinsics, size,
                    False)
        intrinsics, width, height = utils.restore_camera_export(
            prediction.intrinsics, inputs.transforms)
        return prediction.extrinsics_w2c, intrinsics, (width, height), True

    def metrics_extra(self,
                      depth_only: bool,
                      images_per_run: int | None = None) -> dict[str, Any]:
        """Returns the model's own metrics.json fields.

        Args:
            depth_only: Whether this is a depth-only run.
            images_per_run: Images per call of a depth-only run.

        Returns:
            The fields; subclasses extend the dict.
        """
        extra = {
            "Scene": str(self.args.data_path),
            "Peak GPU Memory": peak_gpu_memory_mib(),
        }
        if hasattr(self.args, "spatial_policy"):
            extra.update(
                utils.spatial_policy_metrics(self.args.spatial_policy,
                                             is_shared=self.is_shared_policy))
        if depth_only:
            extra["max_images_per_run"] = images_per_run
            extra["single_view"] = utils.single_view_mode()
        else:
            extra["max_images_per_run"] = self.args.max_images_per_run
        return extra

    def cleanup(self, inputs: ModelInput) -> None:
        """Removes the scratch directories of one model call."""
        keep = utils.should_keep_intermediate(self.args)
        utils.cleanup_intermediate_paths(list(inputs.scratch_dirs), keep=keep)

    # ------------------------------------------------------------------
    # The pipeline.
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Runs the model on the scene and writes its artifacts.

        Raises:
            FileNotFoundError: The scene has no images directory.
            ValueError: No images were found, or the options conflict.
        """
        args = self.args
        self.validate_args()
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
        if args.depth_only and not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")

        self.add_code_path()
        self.device = self.select_device()
        args.output_path.mkdir(parents=True, exist_ok=True)
        self.model = self.load(self.device)
        self.started_at = utils.mark_model_loaded()
        if args.depth_only:
            self.run_depth_only()
        else:
            self.run_full()

    def run_depth_only(self) -> None:
        """Exports depth (and cameras) in batches; no dense points."""
        args = self.args
        images_per_run = self.images_per_run()
        cameras = CameraExport()
        batches = self.depth_batches(images_per_run)
        for batch_index, batch_paths in enumerate(batches):
            print(f"Depth-only batch {batch_index + 1}/{len(batches)}: "
                  f"{len(batch_paths)} images")
            prediction, inputs = self.predict(batch_paths, batch_index)
            self.export_depth(inputs, prediction)
            if args.export_colmap:
                batch_cameras = self.cameras_for_export(inputs, prediction)
                if batch_cameras is not None:
                    extrinsics, intrinsics, size, restored = batch_cameras
                    cameras.add(inputs.image_paths, extrinsics, intrinsics,
                                size, restored)
            self.cleanup(inputs)
            del prediction
            empty_gpu_cache()

        cameras.write(args.output_path, self.images_dir,
                      args.colmap_model_subdir)
        extra = self.metrics_extra(depth_only=True,
                                   images_per_run=images_per_run)
        self.write_metrics(depth_only=True, metrics_extra=extra)

    def run_full(self) -> None:
        """Exports dense points, depth and COLMAP cameras of one call."""
        args = self.args
        prediction, inputs = self.predict(self.image_paths)
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
            depth_only=False,
            colmap_model_subdir=args.colmap_model_subdir,
            max_points=args.max_points,
            conf_thresh=self.conf_thresh,
            started_at=self.started_at,
            checkpoint=self.checkpoint,
            metrics_extra=self.metrics_extra(depth_only=False),
        )
        self.cleanup(inputs)

    def write_metrics(self,
                      depth_only: bool,
                      metrics_extra: dict[str, Any] | None = None,
                      runtime: dict[str, Any] | None = None) -> None:
        """Writes metrics.json.

        Args:
            depth_only: Whether this is a depth-only run.
            metrics_extra: The model's own fields.
            runtime: Runtime metrics taken before the export, or None to
                take them now.
        """
        utils.write_adapter_metrics(
            output_path=self.args.output_path,
            method=self.method_label,
            image_paths=self.image_paths,
            started_at=self.started_at,
            checkpoint=self.checkpoint,
            depth_only=depth_only,
            metrics_extra=metrics_extra,
            runtime=runtime,
        )
