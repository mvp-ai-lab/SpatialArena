# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""DepthVLM: dense metric depth from a Qwen3-VL with a depth head.

Depth-only. Each image is fit into --image_load_resolution
(`ImageOps.contain`) and asked for its depth map; the model's
`outputs.depth_pred` is resized back to the source image. Batches of
--max_images_per_run images (one image per call in single-view mode).

DepthVLM's code targets a newer transformers than the benchmark environment
has; `patch_transformers_compat` adds the few names it needs.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
import types

import numpy as np
from PIL import Image
from PIL import ImageOps

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

PROMPT = "Estimate the metric depth map for this image."

# The pre-refactor adapter's module docstring, its --help description.
DESCRIPTION = ("Run DepthVLM dense metric-depth inference for benchmark depth "
               "evaluation.")


def build_messages(processor, image_path: str,
                   image_load_resolution: int) -> tuple[str, list]:
    """Builds DepthVLM's chat for one image.

    Args:
        processor: The model's processor.
        image_path: The source image.
        image_load_resolution: Bound of both image sides (0: source size).

    Returns:
        A (prompt text, image inputs) tuple for the processor.
    """
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    if image_load_resolution > 0:
        image = ImageOps.contain(
            image,
            (image_load_resolution, image_load_resolution),
            method=Image.Resampling.BICUBIC,
        )
    messages = [
        {
            "role":
                "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant."
            }],
        },
        {
            "role":
                "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": PROMPT
                },
            ],
        },
    ]
    text = processor.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    try:
        # Qwen's own image preprocessing, when installed.
        # pylint: disable-next=import-outside-toplevel
        import qwen_vl_utils

        image_inputs, _ = qwen_vl_utils.process_vision_info(messages)
    except ModuleNotFoundError:
        image_inputs = [image]
    return text, image_inputs


def _json_safe_config_value(value):
    """Returns a config value with sets turned into sorted lists."""
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {
            key: _json_safe_config_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_config_value(item) for item in value]
    return value


def _patch_config_to_dict(configuration_utils):
    """Makes `PretrainedConfig.to_dict` JSON-serializable (no sets)."""
    config_cls = getattr(configuration_utils, "PretrainedConfig", None)
    if config_cls is None or getattr(config_cls, "_depthvlm_json_safe_to_dict",
                                     False):
        return
    original_to_dict = config_cls.to_dict

    def json_safe_to_dict(self):
        return _json_safe_config_value(original_to_dict(self))

    config_cls.to_dict = json_safe_to_dict
    config_cls._depthvlm_json_safe_to_dict = True


def _patch_attention_interface(modeling_utils):
    """Adds `ALL_ATTENTION_FUNCTIONS.get_interface` where it is missing."""
    attention_functions = modeling_utils.ALL_ATTENTION_FUNCTIONS
    if hasattr(attention_functions, "get_interface"):
        return

    def get_attention_interface(name, default):
        if name == "eager":
            return default
        return attention_functions.get(name, default)

    attention_functions.get_interface = get_attention_interface


def _patch_special_tokens(tokenization_utils_base):
    """Accepts a list of model-specific special tokens (as an empty dict)."""
    tokenizer_cls = tokenization_utils_base.PreTrainedTokenizerBase
    original = tokenizer_cls._set_model_specific_special_tokens

    def set_model_specific_special_tokens_compat(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {}
        return original(self, special_tokens)

    tokenizer_cls._set_model_specific_special_tokens = (
        set_model_specific_special_tokens_compat)


def _set_missing(module, name, value):
    """Sets `module.name` to `value` unless it exists."""
    if not hasattr(module, name):
        setattr(module, name, value)


def patch_transformers_compat():
    """Adds the transformers names DepthVLM's code needs, where missing."""
    # pylint: disable=import-outside-toplevel
    import transformers
    from transformers import configuration_utils
    from transformers import integrations
    from transformers import modeling_rope_utils
    from transformers import modeling_utils
    from transformers import tokenization_utils_base
    from transformers import utils as transformers_utils
    from transformers.utils import generic

    _patch_config_to_dict(configuration_utils)
    if hasattr(configuration_utils, "PretrainedConfig"):
        _set_missing(configuration_utils, "PreTrainedConfig",
                     configuration_utils.PretrainedConfig)
    _set_missing(modeling_rope_utils, "RopeParameters", dict)
    _set_missing(transformers, "initialization",
                 types.SimpleNamespace(copy_=lambda dst, src: dst.copy_(src)))
    _patch_attention_interface(modeling_utils)
    _set_missing(integrations, "use_kernel_func_from_hub", lambda _name:
                 (lambda fn: fn))
    _set_missing(integrations, "use_kernelized_func", lambda _fn:
                 (lambda cls: cls))
    _set_missing(transformers_utils, "torch_compilable_check",
                 lambda *args, **kwargs: None)
    _set_missing(generic, "torch_compilable_check",
                 lambda *args, **kwargs: None)
    _set_missing(
        generic, "is_flash_attention_requested", lambda config: getattr(
            config, "_attn_implementation", None) == "flash_attention_2")
    _set_missing(generic, "maybe_autocast",
                 lambda *args, **kwargs: contextlib.nullcontext())
    _set_missing(generic, "merge_with_config_defaults", lambda fn: fn)
    _patch_special_tokens(tokenization_utils_base)
    sys.modules.setdefault(
        "transformers.utils.output_capturing",
        types.SimpleNamespace(capture_outputs=lambda fn: fn),
    )


def _patch_create_causal_mask(depthvlm_modeling):
    """Renames inputs_embeds to the input_embeds an older mask builder takes."""
    signature = inspect.signature(depthvlm_modeling.create_causal_mask)
    if ("inputs_embeds" in signature.parameters or
            "input_embeds" not in signature.parameters):
        return
    original_create_causal_mask = depthvlm_modeling.create_causal_mask

    def create_causal_mask_compat(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original_create_causal_mask(*args, **kwargs)

    depthvlm_modeling.create_causal_mask = create_causal_mask_compat


def _depth_map(depth) -> np.ndarray:
    """Returns one `depth_pred` entry as a 2-D float32 map, NaN/inf as 0."""
    depth_np = depth.detach().cpu().float().numpy()
    while depth_np.ndim > 2:
        depth_np = depth_np[0]
    depth_np = depth_np.astype(np.float32, copy=False)
    depth_np[~np.isfinite(depth_np)] = 0.0
    return depth_np


@registry.register_model
class DepthVLM(base.BaseModel):
    """DepthVLM.

    The images are resized by `ImageOps.contain` to --image_load_resolution
    only (no --resolution), and only depth is exported (no point options).
    --max_images_per_run is the batch size, not a frame cap.
    """

    name = "DEPTHVLM"
    description = DESCRIPTION
    code_packages = ("model", "utils")
    uses_resolution = False
    image_load_resolution_help = (
        "Bound of both image sides the VLM sees (0: source size).")
    exports_points = False
    default_image_load_resolution = 224
    default_max_images_per_run = 1

    @classmethod
    def add_arguments(cls, parser):
        """Adds --image_load_resolution and --attn_implementation.

        Args:
            parser: The model's parser.
        """
        parser.add_argument(
            "--attn_implementation",
            default=os.environ.get("DEPTHVLM_ATTN_IMPLEMENTATION", "sdpa"),
            help=("Attention backend passed to DepthVLM/Qwen3-VL. Defaults to "
                  "'sdpa' so the adapter runs in the benchmark dogs "
                  "environment without flash_attn."),
        )

    def __init__(self, args):
        super().__init__(args)
        self.processor = None

    def validate_args(self):
        if self.args.export_colmap and not self.args.depth_only:
            raise ValueError(
                "DepthVLM only exports dense depth maps; use it on "
                "depth-evaluation tracks.")

    def select_images(self):
        """Returns every --interval-th frame (no frame cap)."""
        return utils.select_interval(utils.find_image_paths(self.images_dir),
                                     self.args.interval)

    def load(self, device):
        del device  # transformers places the model (device_map).
        # pylint: disable=import-outside-toplevel
        import torch

        patch_transformers_compat()
        import model as depthvlm_model
        from model import modeling_qwen3_vl
        import transformers

        _patch_create_causal_mask(modeling_qwen3_vl)
        self.processor = transformers.AutoProcessor.from_pretrained(
            self.args.ckpt_path)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        cuda = torch.cuda.is_available()
        network_cls = depthvlm_model.Qwen3VLForConditionalGeneration
        network = network_cls.from_pretrained(
            self.args.ckpt_path,
            torch_dtype=torch.bfloat16 if cuda else torch.float32,
            attn_implementation=self.args.attn_implementation if cuda else None,
            device_map="auto" if cuda else None,
        )
        network.eval()
        return network

    def preprocess(self, image_paths, batch_index=None):
        """Builds the processor inputs of a batch, on the model device.

        Args:
            image_paths: The batch's source images.
            batch_index: Unused.

        Returns:
            The `ModelInput`; no transforms (each output grid is only known
            after the forward pass).
        """
        import torch  # pylint: disable=import-outside-toplevel

        texts = []
        image_inputs = []
        for image_path in image_paths:
            text, images = build_messages(self.processor, image_path,
                                          self.args.image_load_resolution)
            texts.append(text)
            image_inputs.append(images)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        return base.ModelInput(image_paths, inputs, None, batch_index)

    def forward(self, inputs):
        """Runs DepthVLM on a batch.

        Args:
            inputs: The output of `preprocess`.

        Returns:
            The `Prediction`, with the model's per-image `depth_pred`
            tensors (grid sizes may differ) in `extra`.

        Raises:
            RuntimeError: The model returned no `depth_pred`.
        """
        import torch  # pylint: disable=import-outside-toplevel

        data = inputs.data
        with torch.no_grad():
            outputs = self.model(
                input_ids=data.input_ids,
                attention_mask=data.attention_mask,
                pixel_values=data.get("pixel_values"),
                image_grid_thw=data.get("image_grid_thw"),
            )
        if outputs.depth_pred is None:
            raise RuntimeError("DepthVLM did not return outputs.depth_pred for "
                               "the selected batch.")
        return base.Prediction(extra={"depth_pred": outputs.depth_pred})

    def export_depth(self, inputs, prediction):
        """Restores each image's depth to its source size and writes it.

        Qwen3-VL's dynamic-resolution vision tower can give the images of a
        batch different grids, so each map is restored on its own. Nothing
        is cropped or padded (only `ImageOps.contain`, then the tower's
        grid), so the restore is a plain resize.

        Args:
            inputs: The batch's `ModelInput`.
            prediction: The batch's `Prediction`.
        """
        for image_path, depth in zip(inputs.image_paths,
                                     prediction.extra["depth_pred"]):
            depth_np = _depth_map(depth)
            # The EXIF transpose of build_messages: an untransposed size
            # would swap width and height for a 90/270 degree orientation.
            with Image.open(image_path) as probe:
                source_wh = ImageOps.exif_transpose(probe).size
            transform = utils.SpatialTransform(source_wh=source_wh,
                                               resized_wh=(depth_np.shape[1],
                                                           depth_np.shape[0]))
            restored = utils.restore_prediction(depth_np, transform)
            utils.export_depth_predictions(self.args.output_path, [image_path],
                                           restored[None])

    def run(self):
        """Runs DepthVLM on the scene (always depth-only).

        Raises:
            ValueError: The options conflict, no images were found, or
                --export_depth is missing.
        """
        args = self.args
        self.validate_args()
        self.images_dir = utils.canonical_image_root(args.data_path)
        self.image_paths = self.select_images()
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")
        if args.dry_run:
            utils.print_dry_run(args, self.method_label, len(self.image_paths))
            return
        if not args.export_depth:
            raise ValueError("DepthVLM adapter requires --export_depth.")

        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        self.add_code_path()
        self.model = self.load(self.device)
        args.output_path.mkdir(parents=True, exist_ok=True)
        self.started_at = utils.mark_model_loaded()
        self.run_depth_only()

    def run_depth_only(self):
        """Runs the batches and writes depths and metrics.json."""
        # One image per call in single-view mode: a bf16 batch's numerics
        # depend on its size and padding.
        images_per_run = max(1, int(self.args.max_images_per_run))
        for batch_paths in self.depth_batches(images_per_run):
            inputs = self.preprocess(batch_paths)
            prediction = self.forward(inputs)
            self.export_depth(inputs, prediction)
            base.empty_gpu_cache()
        self.write_metrics(depth_only=True,
                           metrics_extra={
                               "track": self.args.track,
                               "adapter": "DepthVLM.outputs.depth_pred"
                           })
