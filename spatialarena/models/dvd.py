# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Official DVD v1.1 (EnVision-Research/DVD) VideoDepth, RGB only.

Ported from PR #21's dvd.py (eval_refactor_hsy), depth-only variant: the
official Wan2.1-T2V-1.3B DiT + VAE with the DVD LoRA merged, the official
training-scale resize (at least 640x480, each side rounded up to 16 --
adapter_utils.plan_spatial_transform(method="dvd")), and the official
sliding-window stitching (test_single_video.generate_depth_sliced, window 45,
overlap 9). The stitched normalized disparity is restored to each frame's
source grid before inversion: raw_disparities/ keeps it, depths/ holds the
relative-depth proxy 1 / max(q, 0.001) (not metric depth). No GT is used.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib

import numpy as np
from PIL import Image

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

DESCRIPTION = __doc__
WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
CHECKPOINT_FILENAME = "dvd_1.1.safetensors"


def resolve_checkpoint_file(ckpt_path: str, filename: str) -> str:
    """Returns a checkpoint file.

    Args:
        ckpt_path: The file itself, a directory holding `filename`, or a
            Hugging Face repo id (resolved from the cache).
        filename: The file's name in a directory or repo.

    Returns:
        The path of the file.
    """
    path = pathlib.Path(ckpt_path).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        return str(path / filename)
    import huggingface_hub  # pylint: disable=import-outside-toplevel

    return huggingface_hub.hf_hub_download(ckpt_path, filename)


def wan_directory(wan_path: str) -> pathlib.Path:
    """Returns the Wan2.1 base model directory.

    Args:
        wan_path: The directory, or a Hugging Face repo id (resolved from the
            cache).

    Returns:
        The directory.
    """
    if pathlib.Path(wan_path).is_dir():
        return pathlib.Path(wan_path)
    import huggingface_hub  # pylint: disable=import-outside-toplevel

    return pathlib.Path(
        huggingface_hub.snapshot_download(wan_path,
                                          allow_patterns=[
                                              "diffusion_pytorch_model*"
                                              ".safetensors", "Wan2.1_VAE.pth",
                                              "config.json"
                                          ]))


@registry.register_model
class DVD(base.BaseModel):
    """DVD v1.1: one depth-only call over the whole sequence."""

    name = "DVD"
    aliases = ("DVD_V11", "DVD-V1.1")
    description = DESCRIPTION
    ckpt_default = "FayeHongfeiZhang/DVD"
    code_packages = ("examples", "test_script", "diffsynth")
    # The official resize is fixed (at least 640x480, sides rounded up to
    # 16), and only depth is exported.
    uses_resolution = False
    uses_image_load_resolution = False
    exports_points = False
    # One policy, the benchmark's "shared" one: the official resize, with
    # every prediction restored to its frame's source grid.
    spatial_policies = ("shared",)
    requires_cuda = True

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        args.output_path = args.output_path.resolve()
        self.checkpoint_file = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--wan_path",
            default=os.environ.get("WAN_MODEL_PATH", WAN_REPO),
            help=("Wan2.1-T2V-1.3B directory or HF repo id (DiT safetensors "
                  "+ Wan2.1_VAE.pth); default $WAN_MODEL_PATH."))
        parser.add_argument("--window_size",
                            type=int,
                            default=45,
                            help="Sliding-window length (4n+1 frames).")
        parser.add_argument("--overlap",
                            type=int,
                            default=9,
                            help="Frames shared by consecutive windows.")
        parser.add_argument(
            "--inverse_epsilon",
            type=float,
            default=1e-3,
            help="Disparity floor of the depth proxy 1 / max(q, eps).")

    def validate_args(self) -> None:
        """Rejects inconsistent options before anything is loaded.

        Raises:
            ValueError: --window_size is not 4n+1 or --overlap is outside
                [0, window_size).
        """
        args = self.args
        if not (0 <= args.overlap < args.window_size and
                args.window_size % 4 == 1):
            raise ValueError(
                "--window_size must be 4n+1 and --overlap in [0, window_size)")

    @property
    def checkpoint(self) -> str | None:
        """The resolved checkpoint file once loaded, else --ckpt_path."""
        return self.checkpoint_file or self.args.ckpt_path

    def select_images(self) -> list[str]:
        """Returns the selected frames as resolved paths."""
        return [
            str(pathlib.Path(path).resolve()) for path in utils.select_interval(
                utils.find_image_paths(self.images_dir), self.args.interval,
                self.args.max_images_per_run)
        ]

    @staticmethod
    def plan_transform(image_paths: list[str]) -> utils.SpatialTransform:
        """Plans the official resize of a sequence.

        Args:
            image_paths: The sequence's frames.

        Returns:
            The transform shared by every frame.

        Raises:
            ValueError: The frames differ in size.
        """
        sizes = set()
        for path in image_paths:
            with Image.open(path) as image:
                sizes.add(image.size)
        if len(sizes) != 1:
            raise ValueError(
                f"DVD needs one frame size per sequence, got {sorted(sizes)}")
        return utils.plan_spatial_transform(sizes.pop(), method="dvd")

    def load(self, device):
        # Fetched before the method's code is imported, as upstream.
        self.checkpoint_file = resolve_checkpoint_file(self.args.ckpt_path,
                                                       CHECKPOINT_FILENAME)
        wan = wan_directory(self.args.wan_path)

        code_path = self.args.code_path.resolve()
        utils.add_code_path(code_path, self.code_packages)
        # The official pipeline resolves its bundled tokenizer relative to
        # the repository root.
        os.chdir(code_path)
        # Imported in upstream's order.
        # pylint: disable=import-outside-toplevel
        import torch  # noqa: F401
        import accelerate
        training_module = importlib.import_module(
            "examples.wanvideo.model_training.WanTrainingModule")
        import omegaconf
        from safetensors import torch as safetensors_torch
        from test_script import test_single_video  # noqa: F401

        # pylint: enable=import-outside-toplevel
        cfg = omegaconf.OmegaConf.load(code_path / "ckpt" / "model_config.yaml")
        model = training_module.WanTrainingModule(
            args=cfg,
            accelerator=accelerate.Accelerator(),
            model_paths=json.dumps([
                str(wan / "diffusion_pytorch_model.safetensors"),
                str(wan / "Wan2.1_VAE.pth")
            ]),
            trainable_models=None,
            lora_rank=cfg.lora_rank,
            lora_base_model=cfg.lora_base_model,
            use_gradient_checkpointing=False,
        )
        weights = safetensors_torch.load_file(self.checkpoint_file)
        dit = {
            key.removeprefix("pipe.dit."): value
            for key, value in weights.items()
            if key.startswith("pipe.dit.")
        }
        if not dit:
            raise ValueError(
                f"{self.checkpoint_file} contains no pipe.dit parameters")
        model.pipe.dit.load_state_dict(dit, strict=True)
        del weights, dit
        model.merge_lora_layer()
        return model.to(device).eval()

    def preprocess(self, image_paths, batch_index=None):
        import torch  # pylint: disable=import-outside-toplevel

        transform = self.plan_transform(image_paths)
        model_w, model_h = transform.model_wh
        rgb = np.stack([
            np.asarray(Image.open(path).convert("RGB")) for path in image_paths
        ])
        video = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
        video = utils.apply_spatial_transform_tensor(video, transform)[None]
        if tuple(video.shape[-2:]) != (model_h, model_w):
            raise ValueError(f"Preprocessed frames are "
                             f"{tuple(video.shape[-2:])}, planned "
                             f"{(model_h, model_w)}")
        # The transform is applied here, not restorable by the base class:
        # the disparity is restored before its inversion (postprocess).
        return base.ModelInput(image_paths,
                               data={"video": video},
                               batch_index=batch_index,
                               extra={"transform": transform})

    def forward(self, inputs):
        # pylint: disable=import-outside-toplevel
        import torch
        from test_script import test_single_video

        with torch.inference_mode():
            disparity = test_single_video.generate_depth_sliced(
                self.model,
                inputs.data["video"],
                window_size=self.args.window_size,
                overlap=self.args.overlap)[0]
        disparity = np.asarray(disparity).mean(-1)
        model_w, model_h = inputs.extra["transform"].model_wh
        if disparity.shape != (len(inputs.image_paths), model_h,
                               model_w) or not np.isfinite(disparity).all():
            raise ValueError(
                f"Invalid stitched disparity: shape {disparity.shape}")
        return base.Prediction(extra={"disparity": disparity})

    def postprocess(self, prediction, inputs):
        # Restores the disparity to the source grid, then inverts it.
        transform = inputs.extra["transform"]
        # Interpolate disparity (DVD's representation) to the source grid,
        # then invert.
        disparity = np.stack([
            utils.restore_prediction(frame, transform)
            for frame in prediction.extra["disparity"]
        ]).astype(np.float32)
        prediction.extra["raw_disparity"] = disparity
        prediction.depth = 1.0 / np.maximum(disparity,
                                            self.args.inverse_epsilon)
        return prediction

    def run(self) -> None:
        """Runs DVD on the scene and writes its artifacts.

        Raises:
            ValueError: The options ask for more than dense depth, no images
                were found, or the frames differ in size.
        """
        args = self.args
        self.validate_args()
        if args.export_colmap or not args.export_depth:
            raise ValueError("DVD predicts dense depth only: pass "
                             "--export_depth, not --export_colmap")
        self.images_dir = utils.canonical_image_root(args.data_path)
        self.image_paths = self.select_images()
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")
        if args.dry_run:
            print(json.dumps(self.planned_layout(), indent=2))
            return
        self.plan_transform(self.image_paths)  # Fails early on mixed sizes.
        self.device = "cuda"
        self.model = self.load(self.device)
        self.started_at = utils.mark_model_loaded()
        inputs = self.preprocess(self.image_paths)
        prediction = self.postprocess(self.forward(inputs), inputs)
        self.export(inputs, prediction)

    def export(self, inputs: base.ModelInput,
               prediction: base.Prediction) -> None:
        """Writes raw disparities, depths, transforms and metrics.json.

        Args:
            inputs: The sequence's input.
            prediction: The postprocessed prediction.
        """
        args = self.args
        disparity = prediction.extra["raw_disparity"]
        transform = inputs.extra["transform"]
        raw_dir = args.output_path / "raw_disparities"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for path, frame in zip(inputs.image_paths, disparity):
            utils.save_prediction_map(raw_dir, pathlib.Path(path).stem, frame)
        utils.export_depth_predictions(args.output_path, inputs.image_paths,
                                       prediction.depth)
        (args.output_path / "spatial_transforms.json").write_text(
            json.dumps(
                {
                    pathlib.Path(path).name: transform.metadata()
                    for path in inputs.image_paths
                },
                indent=2),
            encoding="utf-8")
        model_w, model_h = transform.model_wh
        extra = {
            "track":
                args.track,
            "spatial_policy":
                args.spatial_policy,
            "model_wh": [model_w, model_h],
            "window_size":
                args.window_size,
            "overlap":
                args.overlap,
            "raw_representation":
                "normalized_disparity",
            "depth_conversion":
                f"1 / max(q, {args.inverse_epsilon})",
            "clamped_fraction":
                float((disparity < args.inverse_epsilon).mean()),
        }
        self.write_metrics(depth_only=True, metrics_extra=extra)
