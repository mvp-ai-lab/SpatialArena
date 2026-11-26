# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official DepthCrafter/ChronoDepth pipelines on canonical selected image
sequences."""

from __future__ import annotations

import argparse
import os
import pathlib

import cv2
import numpy as np
from PIL import Image

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

DESCRIPTION = __doc__
REPOS = {
    "DEPTHCRAFTER": "tencent/DepthCrafter",
    "CHRONODEPTH": "jhshao/ChronoDepth-v1"
}


def processing_size(width: int, height: int, max_res: int) -> tuple[int, int]:
    """Returns the official open-video size of a frame.

    Each side is rounded to 64, with a long-edge limit.

    Args:
        width: Source width.
        height: Source height.
        max_res: Long-edge limit.

    Returns:
        The (width, height) the pipeline runs at.

    Raises:
        ValueError: A size is not positive or would round to zero.
    """
    if min(width, height, max_res) <= 0:
        raise ValueError("Image dimensions and max_res must be positive")
    w, h = round(width / 64) * 64, round(height / 64) * 64
    if max(w, h) > max_res:
        scale = max_res / max(width, height)
        w, h = round(width * scale / 64) * 64, round(height * scale / 64) * 64
    if min(w, h) < 64:
        raise ValueError(
            "Official preprocessing would produce a zero-size image")
    return w, h


def restore_output(output, method: str, count: int,
                   source_wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Converts a pipeline output to the source grid.

    The actual pipeline layout is validated, the native representation
    restored to the source size, then converted to depth for the benchmark.

    Args:
        output: The pipeline's frames.
        method: "DEPTHCRAFTER" or "CHRONODEPTH".
        count: The number of input frames.
        source_wh: The (width, height) of the source frames.

    Returns:
        A (raw, depth) tuple of (N, H, W) float32 arrays: the native
        representation and the depth.

    Raises:
        ValueError: The output is malformed, nonfinite or constant.
    """
    raw = np.asarray(output, dtype=np.float32)
    if method == "DEPTHCRAFTER":
        if raw.ndim != 5 or raw.shape[0] != 1 or raw.shape[-1] != 3:
            raise ValueError(f"Unexpected DepthCrafter output {raw.shape}")
        raw = raw[0].mean(axis=-1)
    elif method == "CHRONODEPTH":
        if raw.ndim != 4 or raw.shape[1] != 1:
            raise ValueError(f"Unexpected ChronoDepth output {raw.shape}")
        raw = raw[:, 0]
    else:
        raise ValueError(method)
    if raw.shape[0] != count or not np.isfinite(raw).all():
        raise ValueError("Prediction count mismatch or nonfinite predictions")
    if method == "DEPTHCRAFTER":
        span = float(raw.max() - raw.min())
        if span <= 1e-8:
            raise ValueError("Constant DepthCrafter prediction")
        # The official sequence-wide normalization, not per-frame.
        raw = (raw - raw.min()) / span
    raw = np.stack([
        cv2.resize(frame, source_wh, interpolation=cv2.INTER_LINEAR)
        for frame in raw
    ])
    if method == "DEPTHCRAFTER":
        depth = 1.0 / np.maximum(raw, 1e-3)
    else:
        raw = np.clip(raw, 0.0, 1.0)
        depth = np.maximum(raw, 1e-6)
    return raw, depth.astype(np.float32)


class VideoDiffusionDepth(base.BaseModel):
    """DepthCrafter or ChronoDepth: one depth-only call.

    The methods' own classes, `DepthCrafter` and `ChronoDepth`, are the
    registered models.
    """

    name = "VIDEO_DIFFUSION_DEPTH"
    description = DESCRIPTION
    # The official resize (each side rounded to 64, --resolution caps the
    # long edge) is the model's own, and only depth is exported.
    uses_image_load_resolution = False
    resolution_help = ("Long-edge limit of the official resize (each side "
                       "rounded to 64).")
    default_resolution = 1024
    exports_points = False
    default_seed = 42
    spatial_policies = ("shared",)
    requires_cuda = True

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.source_wh = None
        self.model_wh = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--base_path",
            default=os.environ.get(
                "SVD_BASE_PATH",
                "stabilityai/stable-video-diffusion-img2vid-xt"),
            help=("Stable Video Diffusion base model (directory or HF repo "
                  "id); default $SVD_BASE_PATH."))
        parser.add_argument("--steps",
                            type=int,
                            default=5,
                            help="Denoising steps.")
        parser.add_argument("--cpu_offload",
                            choices=["none", "model", "sequential"],
                            default="none")

    @classmethod
    def argument_error(cls, args: argparse.Namespace) -> str | None:
        """Returns why the options are inconsistent, or None.

        Args:
            args: The parsed options.
        """
        if args.steps < 1:
            return "Invalid steps/window/overlap"
        return None

    @classmethod
    def from_argv(cls, argv=None, method=None) -> VideoDiffusionDepth:
        """Creates the model from command-line arguments.

        Args:
            argv: The arguments; None for sys.argv.
            method: The method to run; None for the model's own name.

        Returns:
            The model; inconsistent options exit with a usage error.
        """
        parser = cls.build_parser(method)
        args = parser.parse_args(argv)
        error = cls.argument_error(args)
        if error:
            parser.error(error)
        return cls(args)

    @property
    def is_crafter(self) -> bool:
        """Whether this is DepthCrafter (disparity) rather than ChronoDepth."""
        return self.method_name == "DEPTHCRAFTER"

    @property
    def checkpoint(self) -> str | None:
        """--ckpt_path, or the method's own Hugging Face repo."""
        return self.args.ckpt_path or REPOS[self.method_name]

    def select_images(self) -> list[str]:
        """Returns the selected frames as absolute (unresolved) paths."""
        return [
            str(pathlib.Path(p).absolute()) for p in utils.select_interval(
                utils.find_image_paths(self.images_dir), self.args.interval,
                self.args.max_images_per_run)
        ]

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        import torch
        import transformers

        args = self.args
        transformers.set_seed(args.seed)
        if self.is_crafter:
            from depthcrafter import depth_crafter_ppl
            from depthcrafter import unet as unet_module
            pipeline_cls = depth_crafter_ppl.DepthCrafterPipeline
            unet_cls = (unet_module.
                        DiffusersUNetSpatioTemporalConditionModelDepthCrafter)
        else:
            from chronodepth import chronodepth_pipeline
            from chronodepth import unet_chronodepth
            pipeline_cls = chronodepth_pipeline.ChronoDepthPipeline
            unet_cls = (unet_chronodepth.
                        DiffusersUNetSpatioTemporalConditionModelChronodepth)
        unet = unet_cls.from_pretrained(self.checkpoint,
                                        low_cpu_mem_usage=True,
                                        torch_dtype=torch.float16)
        pipe = pipeline_cls.from_pretrained(args.base_path,
                                            unet=unet,
                                            torch_dtype=torch.float16,
                                            variant="fp16")
        if args.cpu_offload == "model":
            pipe.enable_model_cpu_offload()
        elif args.cpu_offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)
        if self.is_crafter:
            pipe.enable_attention_slicing()
        else:
            pipe.n_tokens, pipe.chunk_size = args.n_tokens, args.chunk_size
        return pipe

    def plan_size(self, image_paths: list[str]) -> None:
        """Sets the source and processing sizes of a sequence.

        Args:
            image_paths: The sequence's frames.

        Raises:
            ValueError: The frames differ in size.
        """
        sizes = set()
        for path in image_paths:
            with Image.open(path) as image:
                sizes.add(image.size)
        if len(sizes) != 1:
            raise ValueError("Video frames must share a source resolution")
        self.source_wh = sizes.pop()
        self.model_wh = processing_size(*self.source_wh, self.args.resolution)

    def preprocess(self, image_paths, batch_index=None):
        if self.model_wh is None:
            self.plan_size(image_paths)
        frames = []
        for path in image_paths:
            with Image.open(path) as image:
                frames.append(
                    cv2.resize(np.asarray(image.convert("RGB")),
                               self.model_wh,
                               interpolation=cv2.INTER_LINEAR))
        frames = np.stack(frames).astype(np.float32) / 255.0
        return base.ModelInput(image_paths,
                               data={"frames": frames},
                               batch_index=batch_index)

    def forward(self, inputs):
        import torch  # pylint: disable=import-outside-toplevel

        args = self.args
        frames = inputs.data["frames"]
        generator = torch.Generator("cuda").manual_seed(args.seed)
        with torch.inference_mode():
            if self.is_crafter:
                output = self.model(frames,
                                    height=self.model_wh[1],
                                    width=self.model_wh[0],
                                    output_type="np",
                                    guidance_scale=1.0,
                                    num_inference_steps=args.steps,
                                    window_size=args.window_size,
                                    overlap=args.overlap,
                                    generator=generator,
                                    track_time=False).frames
            else:
                output = self.model(frames,
                                    num_inference_steps=args.steps,
                                    decode_chunk_size=8,
                                    motion_bucket_id=127,
                                    fps=7,
                                    noise_aug_strength=0.0,
                                    generator=generator,
                                    infer_mode="ours",
                                    sigma_epsilon=-4.0).frames
        return base.Prediction(extra={"output": output})

    def postprocess(self, prediction, inputs):
        # Restores the output to the source grid and converts it to depth.
        raw, depth = restore_output(prediction.extra["output"],
                                    self.method_name, len(inputs.image_paths),
                                    self.source_wh)
        prediction.depth = depth
        prediction.extra["raw"] = raw
        return prediction

    def run(self) -> None:
        """Runs the pipeline on the scene and writes its artifacts.

        Raises:
            ValueError: The options ask for more than depth, the selection is
                empty or repeats a frame name, or the frames differ in size.
        """
        args = self.args
        if args.export_colmap or not args.export_depth:
            raise ValueError("These methods export depth only")
        self.images_dir = utils.canonical_image_root(args.data_path)
        paths = self.select_images()
        if not paths or len({pathlib.Path(p).stem for p in paths
                            }) != len(paths):
            raise ValueError("Empty selection or duplicate frame identifiers")
        self.image_paths = paths
        if args.dry_run:
            utils.print_dry_run(args, self.method_label, len(paths))
            return
        self.plan_size(paths)
        self.add_code_path()
        self.device = "cuda"
        self.model = self.load(self.device)
        self.started_at = utils.mark_model_loaded()
        inputs = self.preprocess(paths)
        prediction = self.postprocess(self.forward(inputs), inputs)
        self.export(inputs, prediction)

    def export(self, inputs: base.ModelInput,
               prediction: base.Prediction) -> None:
        """Writes depths, DepthCrafter's raw disparities and metrics.json.

        Args:
            inputs: The sequence's input.
            prediction: The postprocessed prediction.
        """
        args = self.args
        paths = inputs.image_paths
        raw = prediction.extra["raw"]
        utils.export_depth_predictions(args.output_path, paths,
                                       prediction.depth)
        if self.is_crafter:
            folder = args.output_path / "raw_disparities"
            folder.mkdir(parents=True, exist_ok=True)
            for path, frame in zip(paths, raw):
                utils.save_prediction_map(folder,
                                          pathlib.Path(path).stem, frame)
        crafter = self.is_crafter
        self.write_metrics(
            depth_only=True,
            metrics_extra={
                "source_wh":
                    self.source_wh,
                "model_wh":
                    self.model_wh,
                "seed":
                    args.seed,
                "steps":
                    args.steps,
                "base_model":
                    args.base_path,
                "raw_representation":
                    "disparity" if crafter else "depth",
                "window_size":
                    args.window_size if crafter else args.n_tokens,
                "overlap": (args.overlap if crafter else args.n_tokens -
                            args.chunk_size),
                "clamped_fraction":
                    float((raw < (1e-3 if crafter else 1e-6)).mean()),
            })


@registry.register_model
class DepthCrafter(VideoDiffusionDepth):
    """DepthCrafter: sequence-normalized disparity, inverted to depth."""

    name = "DEPTHCRAFTER"
    aliases = ("DEPTHCRAFT",)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--window_size",
                            type=int,
                            default=110,
                            help="Frames per denoising window.")
        parser.add_argument("--overlap",
                            type=int,
                            default=25,
                            help="Frames shared by consecutive windows.")

    @classmethod
    def argument_error(cls, args: argparse.Namespace) -> str | None:
        if args.steps < 1 or not 0 <= args.overlap < args.window_size:
            return "Invalid steps/window/overlap"
        return None


@registry.register_model
class ChronoDepth(VideoDiffusionDepth):
    """ChronoDepth: relative depth in [0, 1]."""

    name = "CHRONODEPTH"
    aliases = ("CHRONO_DEPTH",)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--n_tokens",
                            type=int,
                            default=10,
                            help="Frames per denoising window.")
        parser.add_argument("--chunk_size",
                            type=int,
                            default=5,
                            help="Frames the window advances by.")

    @classmethod
    def argument_error(cls, args: argparse.Namespace) -> str | None:
        error = super().argument_error(args)
        if error is None and not 0 < args.chunk_size <= args.n_tokens:
            error = "Require 0 < chunk_size <= n_tokens"
        return error
