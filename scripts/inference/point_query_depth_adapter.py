# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run point-query VLM depth models and export dense benchmark depth maps.

VLM3 and DepthLM expose metric depth as text for queried pixels rather than as a
dense tensor. This adapter defines the explicit benchmark bridge: query a fixed
image grid, parse scalar metric depths from text, and interpolate those samples
into the dense ``depths/<stem>.npz`` artifact layout consumed by the evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Fully-qualified (not a bare `from adapter_utils import ...`): this file is
# also imported as scripts.inference.point_query_depth_adapter by
# test/test_point_query_depth_adapter.py, where only ROOT_DIR (added above)
# is on sys.path, not this script's own directory -- see vggt.py's own copy
# of this comment for the full explanation.
from scripts.inference.adapter_utils import (  # noqa: E402
    SpatialTransform, canonical_image_root, mark_model_loaded,
    export_depth_predictions, find_image_paths, print_dry_run,
    restore_prediction, select_interval, write_adapter_metrics,
)

ANSWER_RE = re.compile(
    r"<answer>\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*</answer>")
METER_RE = re.compile(
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:m|meter|meters)\b", re.IGNORECASE)
SINGLE_NUMBER_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\.?\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method_name",
                        required=True,
                        choices=("VLM3", "DEPTHLM", "QWEN3_VL"))
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--track", choices=("rgb",), default="rgb")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--image_load_resolution", type=int, default=224)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--max_images_per_run", type=int, default=1)
    parser.add_argument(
        "--point_grid_size",
        type=int,
        default=int(os.environ.get("POINT_QUERY_DEPTH_GRID_SIZE", "1")),
        help=
        "Per-axis query grid size. 1 is center-only; larger values are slower but less degenerate.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=int(os.environ.get("POINT_QUERY_DEPTH_BATCH_SIZE", "4")),
        help="Number of point prompts generated per model batch.",
    )
    parser.add_argument(
        "--require_spatial_queries",
        action="store_true",
        default=os.environ.get("POINT_QUERY_REQUIRE_SPATIAL", "0") == "1",
        help=
        "Require a multi-point grid and reject images with no valid answers instead of constant imputation.",
    )
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=int(
                            os.environ.get("POINT_QUERY_MAX_NEW_TOKENS",
                                           "256")))
    parser.add_argument("--attn_implementation",
                        default=os.environ.get(
                            "POINT_QUERY_ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=0.0)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", default="0")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_rgb_image(path: str | Path, max_side: int) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side > 0:
        image = ImageOps.contain(image, (max_side, max_side),
                                 method=Image.Resampling.BICUBIC)
    return image


def grid_points(width: int, height: int,
                grid_size: int) -> list[tuple[int, int]]:
    grid_size = max(1, int(grid_size))
    if grid_size == 1:
        return [(width // 2, height // 2)]
    margin_x = max(1, width // (grid_size * 2))
    margin_y = max(1, height // (grid_size * 2))
    xs = np.linspace(margin_x, max(margin_x, width - 1 - margin_x), grid_size)
    ys = np.linspace(margin_y, max(margin_y, height - 1 - margin_y), grid_size)
    return [(int(round(x)), int(round(y))) for y in ys for x in xs]


def draw_depthlm_arrow(image: Image.Image, x: int, y: int) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    size = max(4, min(marked.size) // 40)
    start_x = max(0, x - size)
    draw.line((start_x, y, x, y), fill=(255, 0, 0), width=2)
    draw.line((x, y, max(0, x - size // 2), max(0, y - size // 2)),
              fill=(255, 0, 0),
              width=2)
    draw.line(
        (x, y, max(0, x - size // 2), min(marked.height - 1, y + size // 2)),
        fill=(255, 0, 0),
        width=2)
    return marked


def draw_point_marker(image: Image.Image, x: int, y: int) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    radius = max(5, min(marked.size) // 35)
    width = max(2, radius // 4)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 outline=(255, 0, 0),
                 width=width)
    draw.line((x - radius, y, x + radius, y), fill=(255, 0, 0), width=width)
    draw.line((x, y - radius, x, y + radius), fill=(255, 0, 0), width=width)
    return marked


def build_message(method: str, image: Image.Image, x: int,
                  y: int) -> list[dict[str, Any]]:
    if method == "DEPTHLM":
        # Matches facebookresearch/DepthLM_Official's own
        # utils/datasets.py::generate_prompt_depth_sft() verbatim -- the
        # model was SFT'd to always produce a <think> block before
        # <answer>, so asking it to skip straight to <answer> (as this
        # adapter used to) is off the distribution it was trained on.
        prompt = (
            "Given this image, how far is the point pointed by the red arrow from the camera? "
            "Output the thinking process in <think> </think> and final answer (the meter number "
            "only, without the unit) in <answer> </answer> tags.")
        query_image = draw_depthlm_arrow(image, x, y)
        # DepthLM is a Pixtral (LLaVA) model: its official eval.py builds the chat with convert_example_pixtral(...,
        # image_before_text=True) -- no system message (the Pixtral chat template only takes a plain-string system
        # prompt and fails on a content list), the image before the question.
        return [
            {
                "role":
                    "user",
                "content": [
                    {
                        "type": "image",
                        "image": query_image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            },
        ]
    elif method == "VLM3":
        norm_x = int(round(2000.0 * x / max(1, image.width - 1)))
        norm_y = int(round(2000.0 * y / max(1, image.height - 1)))
        prompt = (
            f"Given this image, how far is the point at coordinates ({norm_x}, {norm_y}) "
            "from the camera? The coordinates are in normalised [0, 2000] format relative "
            "to image width and height. Output the thinking process in <think> </think> "
            "and final answer, the meter number only without the unit, in <answer> </answer> tags."
        )
        query_image = image
    else:
        prompt = (
            "Given this image, how far is the point marked by the red cross from the camera? "
            "Return only the final metric depth as <answer>NUMBER</answer>, where NUMBER is in meters. "
            "Do not include any other numbers or explanation.")
        query_image = draw_point_marker(image, x, y)
    return [{
        "role":
            "user",
        "content": [
            {
                "type": "image",
                "image": query_image
            },
            {
                "type": "text",
                "text": prompt
            },
        ],
    }]


def parse_depth(text: str) -> float | None:
    match = ANSWER_RE.search(text)
    if match is None:
        match = METER_RE.search(text)
    if match is None:
        match = SINGLE_NUMBER_RE.search(text)
    if match is None:
        return None
    value_text = match.group(1)
    try:
        value = float(value_text)
    except ValueError:
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def load_model_and_processor(args: argparse.Namespace):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.ckpt_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        if args.attn_implementation:
            kwargs["attn_implementation"] = args.attn_implementation
    if args.method_name == "DEPTHLM":
        config = AutoConfig.from_pretrained(args.ckpt_path)
        model_type = str(getattr(config, "model_type", "")).lower()
        architectures = " ".join(
            str(item).lower()
            for item in getattr(config, "architectures", []) or [])
        if "pixtral" in model_type or "pixtral" in architectures or "llava" in model_type or "llava" in architectures:
            try:
                from transformers import AutoModelForMultimodalLM

                model = AutoModelForMultimodalLM.from_pretrained(
                    args.ckpt_path, **kwargs)
            except ImportError:
                from transformers import LlavaForConditionalGeneration

                model = LlavaForConditionalGeneration.from_pretrained(
                    args.ckpt_path, **kwargs)
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.ckpt_path, **kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.ckpt_path, **kwargs)
    model.eval()
    return model, processor


def model_device(model: Any) -> str:
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def model_dtype(model: Any):
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        import torch

        return torch.float32


def move_inputs_to_model(inputs: Any, model: Any) -> Any:
    inputs = inputs.to(model_device(model))
    dtype = model_dtype(model)
    for key, value in list(inputs.items()):
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            inputs[key] = value.to(dtype=dtype)
    return inputs


def generate_batch(
    model: Any,
    processor: Any,
    messages_batch: list[list[dict[str, Any]]],
    max_new_tokens: int,
) -> list[str]:
    import torch

    inputs = processor.apply_chat_template(
        messages_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = move_inputs_to_model(inputs, model)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            top_k=None,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def interpolate_sparse_depth(
    samples: list[tuple[int, int, float]],
    width: int,
    height: int,
    grid_size: int,
) -> np.ndarray:
    valid_values = [
        depth for _x, _y, depth in samples if np.isfinite(depth) and depth > 0
    ]
    if not valid_values:
        raise RuntimeError(
            "No point-query depth outputs could be parsed for this image.")
    fallback = float(np.median(valid_values))
    if grid_size <= 1:
        return np.full((height, width), fallback, dtype=np.float32)

    grid = np.full((grid_size, grid_size), fallback, dtype=np.float32)
    for index, (_x, _y, depth) in enumerate(samples):
        row = index // grid_size
        col = index % grid_size
        if 0 <= row < grid_size and 0 <= col < grid_size and np.isfinite(
                depth) and depth > 0:
            grid[row, col] = depth
    dense = Image.fromarray(grid, mode="F").resize((width, height),
                                                   Image.Resampling.BILINEAR)
    depth = np.asarray(dense, dtype=np.float32).copy()
    depth[~np.isfinite(depth)] = fallback
    depth[depth <= 0] = fallback
    return depth


def run() -> None:
    args = parse_args()
    if args.export_colmap and not args.depth_only:
        raise ValueError(
            f"{args.method_name} point-query adapter only supports depth tracks."
        )
    if not args.export_depth and not args.dry_run:
        raise ValueError(f"{args.method_name} adapter requires --export_depth.")
    if args.point_grid_size < 1:
        raise ValueError("--point_grid_size must be positive")
    if args.require_spatial_queries and args.point_grid_size < 2:
        raise ValueError(
            "Spatial depth evaluation requires --point_grid_size >= 2")
    if args.method_name == "DEPTHLM" and "POINT_QUERY_MAX_NEW_TOKENS" not in os.environ:
        # DepthLM's prompt now asks for a <think> block before <answer>
        # (see build_message()); the old answer-only 256-token default
        # would truncate before <answer> ever appears. Matches the
        # official eval.py's own --max_new_tokens default.
        args.max_new_tokens = max(args.max_new_tokens, 4096)

    images_dir = canonical_image_root(args.data_path)
    image_paths = select_interval(find_image_paths(images_dir), args.interval,
                                  args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, args.method_name, len(image_paths))
        return

    model, processor = load_model_and_processor(args)
    started_at = mark_model_loaded()
    depths: list[np.ndarray] = []
    raw_records: list[dict[str, Any]] = []
    parse_failures = 0
    fallback_imputations = 0
    total_queries = 0
    query_batch_size = max(1, int(args.query_batch_size))
    fallback_depth = float(
        os.environ.get("POINT_QUERY_DEPTH_FALLBACK_METERS", "1.0"))
    last_valid_depth: float | None = None
    args.output_path.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_path / "point_query_outputs.jsonl"
    raw_path.write_text("", encoding="utf-8")
    image_coverage: list[dict[str, Any]] = []

    for image_path in image_paths:
        # Match load_rgb_image's own ImageOps.exif_transpose exactly -- an un-transposed probe
        # would report width/height swapped (relative to what the model actually processes) for
        # any image whose EXIF orientation tag requires a 90/270 degree rotation.
        with Image.open(image_path) as probe:
            source_wh = ImageOps.exif_transpose(probe).size
        image = load_rgb_image(image_path, args.image_load_resolution)
        # load_rgb_image's ImageOps.contain is a pure uniform-scale resize (no crop, no pad) --
        # restore_prediction maps the dense depth map built below back to this image's true
        # source resolution, the same mechanism every other patched adapter uses, degenerate here
        # to a plain resize since there's no padding/crop margin to also undo.
        transform = SpatialTransform(source_wh=source_wh,
                                     resized_wh=(image.width, image.height))
        points = grid_points(image.width, image.height, args.point_grid_size)
        pending_messages: list[list[dict[str, Any]]] = []
        pending_points: list[tuple[int, int]] = []
        samples: list[tuple[int, int, float]] = []
        for x, y in points:
            pending_messages.append(build_message(args.method_name, image, x,
                                                  y))
            pending_points.append((x, y))
            if len(pending_messages) < query_batch_size:
                continue
            outputs = generate_batch(model, processor, pending_messages,
                                     args.max_new_tokens)
            for (px, py), text in zip(pending_points, outputs, strict=True):
                depth = parse_depth(text)
                if depth is None:
                    parse_failures += 1
                    depth = np.nan
                samples.append((px, py, float(depth)))
                raw_records.append({
                    "image": str(image_path),
                    "x": px,
                    "y": py,
                    "depth": depth,
                    "text": text
                })
            total_queries += len(pending_messages)
            pending_messages = []
            pending_points = []
        if pending_messages:
            outputs = generate_batch(model, processor, pending_messages,
                                     args.max_new_tokens)
            for (px, py), text in zip(pending_points, outputs, strict=True):
                depth = parse_depth(text)
                if depth is None:
                    parse_failures += 1
                    depth = np.nan
                samples.append((px, py, float(depth)))
                raw_records.append({
                    "image": str(image_path),
                    "x": px,
                    "y": py,
                    "depth": depth,
                    "text": text
                })
            total_queries += len(pending_messages)
        valid_depths = [
            depth for _px, _py, depth in samples
            if np.isfinite(depth) and depth > 0
        ]
        # Persist query evidence before raising, so unsuccessful images can be audited.
        with raw_path.open("a", encoding="utf-8") as handle:
            for record in raw_records:
                clean = dict(record)
                if not np.isfinite(clean["depth"]):
                    clean["depth"] = None
                handle.write(
                    json.dumps(clean, ensure_ascii=True, allow_nan=False) +
                    "\n")
        raw_records.clear()
        image_coverage.append({
            "image": str(image_path),
            "queries": len(samples),
            "valid": len(valid_depths)
        })
        (args.output_path / "point_query_coverage.json").write_text(
            json.dumps(image_coverage, indent=2),
            encoding="utf-8",
        )
        if args.require_spatial_queries and not valid_depths:
            raise RuntimeError(
                f"No valid point-depth answers for {image_path}; constant imputation disabled"
            )
        if valid_depths:
            last_valid_depth = float(np.median(valid_depths))
            dense = interpolate_sparse_depth(samples, image.width, image.height,
                                             args.point_grid_size)
            depths.append(restore_prediction(dense, transform))
        else:
            imputed_depth = last_valid_depth if last_valid_depth is not None else fallback_depth
            fallback_imputations += 1
            source_w, source_h = source_wh
            depths.append(
                np.full((source_h, source_w), imputed_depth, dtype=np.float32))
            raw_records.append({
                "image": str(image_path),
                "depth": imputed_depth,
                "fallback_imputed": True,
                "text": "No parseable point-query depth response.",
            })

    # Legacy mode can add an imputation record after the final image's query flush.
    if raw_records:
        with raw_path.open("a", encoding="utf-8") as handle:
            for record in raw_records:
                handle.write(
                    json.dumps(record, ensure_ascii=True, allow_nan=False) +
                    "\n")
    export_depth_predictions(args.output_path, image_paths,
                             np.stack(depths, axis=0))
    args.output_path.mkdir(parents=True, exist_ok=True)
    write_adapter_metrics(
        output_path=args.output_path,
        method=args.method_name,
        image_paths=image_paths,
        started_at=started_at,
        checkpoint=args.ckpt_path,
        depth_only=True,
        metrics_extra={
            "track":
                args.track,
            "adapter":
                "point_query_depth_adapter",
            "point_grid_size":
                args.point_grid_size,
            "total_point_queries":
                total_queries,
            "interpolation":
                "bilinear",
            "require_spatial_queries":
                args.require_spatial_queries,
            "image_query_coverage":
                image_coverage,
            "missing_point_policy":
                "per-image valid median; reject all-invalid if spatial queries required",
            "parse_failures":
                parse_failures,
            "fallback_imputations":
                fallback_imputations,
            "protocol_note":
                "Dense maps are interpolated from text point-depth queries.",
        },
    )


if __name__ == "__main__":
    run()
