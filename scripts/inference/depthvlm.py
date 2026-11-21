# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run DepthVLM dense metric-depth inference for benchmark depth evaluation."""

from __future__ import annotations

import argparse
import os
import types
from pathlib import Path

import numpy as np

from adapter_utils import (
    SpatialTransform,
    batched_image_paths,
    canonical_image_root,
    mark_model_loaded,
    add_code_path,
    export_depth_predictions,
    find_image_paths,
    print_dry_run,
    restore_prediction,
    select_interval,
    write_adapter_metrics,
)


METHOD = "DEPTHVLM"
PROMPT = "Estimate the metric depth map for this image."


# Shared method checkouts (the engine's METHOD_CODE_ROOT); the engine also passes the *_CODE_PATH below.
# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
METHOD_CODE_ROOT = Path(
    os.environ.get("METHOD_CODE_ROOT", f"{SPATIALARENA_ROOT}/benchmarks/_model_codes")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", default=os.environ.get("DEPTHVLM_CKPT_PATH", "JonnyYu828/DepthVLM-4B"))
    parser.add_argument(
        "--depthvlm_code_path",
        type=Path,
        default=Path(os.environ["DEPTHVLM_CODE_PATH"])
        if "DEPTHVLM_CODE_PATH" in os.environ
        else METHOD_CODE_ROOT / "DepthVLM",
    )
    parser.add_argument("--track", choices=("rgb",), default="rgb")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--image_load_resolution", type=int, default=224)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--max_images_per_run", type=int, default=1)
    parser.add_argument(
        "--attn_implementation",
        default=os.environ.get("DEPTHVLM_ATTN_IMPLEMENTATION", "sdpa"),
        help=(
            "Attention backend passed to DepthVLM/Qwen3-VL. Defaults to 'sdpa' so the "
            "adapter runs in the benchmark dogs environment without flash_attn."
        ),
    )
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=0.0)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", default="0")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_messages(processor, image_path: str, image_load_resolution: int) -> tuple[str, list]:
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    if image_load_resolution > 0:
        image = ImageOps.contain(
            image,
            (image_load_resolution, image_load_resolution),
            method=Image.Resampling.BICUBIC,
        )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, _ = process_vision_info(messages)
    except ModuleNotFoundError:
        image_inputs = [image]
    return text, image_inputs


def run() -> None:
    args = parse_args()
    if args.export_colmap and not args.depth_only:
        raise ValueError("DepthVLM only exports dense depth maps; use it on depth-evaluation tracks.")

    images_dir = canonical_image_root(args.data_path)
    image_paths = select_interval(find_image_paths(images_dir), args.interval)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, METHOD, len(image_paths))
        return
    if not args.export_depth:
        raise ValueError("DepthVLM adapter requires --export_depth.")

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    add_code_path(args.depthvlm_code_path, ("model", "utils"))

    import torch
    import transformers
    import sys
    import transformers.configuration_utils as configuration_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.integrations as transformers_integrations
    import transformers.modeling_rope_utils as modeling_rope_utils
    import transformers.utils as transformers_utils
    import transformers.utils.generic as transformers_generic
    import transformers.tokenization_utils_base as tokenization_utils_base
    from contextlib import nullcontext

    def json_safe_config_value(value):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, dict):
            return {key: json_safe_config_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe_config_value(item) for item in value]
        return value

    config_cls = getattr(configuration_utils, "PretrainedConfig", None)
    if config_cls is not None and not getattr(config_cls, "_depthvlm_json_safe_to_dict", False):
        original_config_to_dict = config_cls.to_dict

        def json_safe_to_dict(self):
            return json_safe_config_value(original_config_to_dict(self))

        config_cls.to_dict = json_safe_to_dict
        config_cls._depthvlm_json_safe_to_dict = True

    if not hasattr(configuration_utils, "PreTrainedConfig") and hasattr(configuration_utils, "PretrainedConfig"):
        configuration_utils.PreTrainedConfig = configuration_utils.PretrainedConfig
    if not hasattr(modeling_rope_utils, "RopeParameters"):
        modeling_rope_utils.RopeParameters = dict
    if not hasattr(transformers, "initialization"):
        transformers.initialization = types.SimpleNamespace(copy_=lambda dst, src: dst.copy_(src))
    if not hasattr(modeling_utils.ALL_ATTENTION_FUNCTIONS, "get_interface"):
        attention_functions = modeling_utils.ALL_ATTENTION_FUNCTIONS

        def get_attention_interface(name, default):
            if name == "eager":
                return default
            return attention_functions.get(name, default)

        attention_functions.get_interface = get_attention_interface
    if not hasattr(transformers_integrations, "use_kernel_func_from_hub"):
        transformers_integrations.use_kernel_func_from_hub = lambda _name: (lambda fn: fn)
    if not hasattr(transformers_integrations, "use_kernelized_func"):
        transformers_integrations.use_kernelized_func = lambda _fn: (lambda cls: cls)
    if not hasattr(transformers_utils, "torch_compilable_check"):
        transformers_utils.torch_compilable_check = lambda *args, **kwargs: None
    if not hasattr(transformers_generic, "torch_compilable_check"):
        transformers_generic.torch_compilable_check = lambda *args, **kwargs: None
    if not hasattr(transformers_generic, "is_flash_attention_requested"):
        transformers_generic.is_flash_attention_requested = (
            lambda config: getattr(config, "_attn_implementation", None) == "flash_attention_2"
        )
    if not hasattr(transformers_generic, "maybe_autocast"):
        transformers_generic.maybe_autocast = lambda *args, **kwargs: nullcontext()
    if not hasattr(transformers_generic, "merge_with_config_defaults"):
        transformers_generic.merge_with_config_defaults = lambda fn: fn
    original_set_special_tokens = tokenization_utils_base.PreTrainedTokenizerBase._set_model_specific_special_tokens

    def set_model_specific_special_tokens_compat(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {}
        return original_set_special_tokens(self, special_tokens)

    tokenization_utils_base.PreTrainedTokenizerBase._set_model_specific_special_tokens = (
        set_model_specific_special_tokens_compat
    )
    sys.modules.setdefault(
        "transformers.utils.output_capturing",
        types.SimpleNamespace(capture_outputs=lambda fn: fn),
    )
    from model import Qwen3VLForConditionalGeneration
    import inspect
    import model.modeling_qwen3_vl as depthvlm_modeling
    from transformers import AutoProcessor

    causal_mask_signature = inspect.signature(depthvlm_modeling.create_causal_mask)
    if (
        "inputs_embeds" not in causal_mask_signature.parameters
        and "input_embeds" in causal_mask_signature.parameters
    ):
        original_create_causal_mask = depthvlm_modeling.create_causal_mask

        def create_causal_mask_compat(*args, **kwargs):
            if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
            return original_create_causal_mask(*args, **kwargs)

        depthvlm_modeling.create_causal_mask = create_causal_mask_compat

    processor = AutoProcessor.from_pretrained(args.ckpt_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    args.output_path.mkdir(parents=True, exist_ok=True)
    started_at = mark_model_loaded()
    # One image per forward in single-view mode: a bf16 batch's numerics depend on its size and padding.
    for batch_paths in batched_image_paths(image_paths, max(1, int(args.max_images_per_run))):
        texts = []
        image_inputs = []
        for image_path in batch_paths:
            text, images = build_messages(processor, image_path, args.image_load_resolution)
            texts.append(text)
            image_inputs.append(images)
        inputs = processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
        if outputs.depth_pred is None:
            raise RuntimeError("DepthVLM did not return outputs.depth_pred for the selected batch.")
        from PIL import Image as PILImage, ImageOps as PILImageOps

        # Qwen3-VL's dynamic-resolution vision tower can give different images in the same batch
        # different depth_pred grid sizes, so each is restored and exported individually rather
        # than stacked into one array. There's no crop/pad anywhere in this pipeline (just
        # ImageOps.contain's resize, then whatever grid the vision tower itself produced), so
        # restore_prediction's plain-resize degenerate case is the correct inverse regardless of
        # depth_pred's exact pixel count not matching the contained image's size one-for-one.
        for image_path, depth in zip(batch_paths, outputs.depth_pred):
            depth_np = depth.detach().cpu().float().numpy()
            while depth_np.ndim > 2:
                depth_np = depth_np[0]
            depth_np = depth_np.astype(np.float32, copy=False)
            depth_np[~np.isfinite(depth_np)] = 0.0
            # Match build_messages's own ImageOps.exif_transpose exactly -- an un-transposed probe
            # would report width/height swapped for any image whose EXIF orientation tag requires
            # a 90/270 degree rotation.
            with PILImage.open(image_path) as probe:
                source_wh = PILImageOps.exif_transpose(probe).size
            resized_wh = (depth_np.shape[1], depth_np.shape[0])
            transform = SpatialTransform(source_wh=source_wh, resized_wh=resized_wh)
            restored = restore_prediction(depth_np, transform)
            export_depth_predictions(args.output_path, [image_path], restored[None])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_adapter_metrics(
        output_path=args.output_path,
        method=METHOD,
        image_paths=image_paths,
        started_at=started_at,
        checkpoint=args.ckpt_path,
        depth_only=True,
        metrics_extra={"track": args.track, "adapter": "DepthVLM.outputs.depth_pred"},
    )


if __name__ == "__main__":
    run()
