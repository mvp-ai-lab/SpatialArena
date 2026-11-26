# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Point-query VLMs (VLM3, DepthLM, Qwen3-VL): dense depth from text answers.

These VLMs give metric depth as text for a queried pixel, not as a dense
tensor. The adapter queries a fixed grid of pixels per image, parses one
scalar depth per answer and interpolates the samples bilinearly into the
dense `depths/<stem>.npz` maps the evaluator reads. Depth-only; each image is
queried on its own (`ImageOps.contain` to --image_load_resolution) and the
dense map is resized back to the source image.

scripts/inference/point_query_depth_adapter.py is an alias of this module, so
the module-level helpers (`load_model_and_processor`, `generate_batch`,
`mark_model_loaded`, `write_adapter_metrics`, ...) are the ones the models
call and can be replaced there.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Sequence

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

METHOD_NAMES = ("VLM3", "DEPTHLM", "QWEN3_VL")

# The pre-refactor adapter's module docstring, its --help description.
DESCRIPTION = """Run point-query VLM depth models and export dense benchmark
depth maps.

VLM3 and DepthLM expose metric depth as text for queried pixels rather than as
a dense tensor. This adapter defines the explicit benchmark bridge: query a
fixed image grid, parse scalar metric depths from text, and interpolate those
samples into the dense ``depths/<stem>.npz`` artifact layout consumed by the
evaluator.
"""

ANSWER_RE = re.compile(
    r"<answer>\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*</answer>")
METER_RE = re.compile(
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:m|meter|meters)\b", re.IGNORECASE)
SINGLE_NUMBER_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\.?\s*$")

DEPTHLM_PROMPT = (
    "Given this image, how far is the point pointed by the red arrow from the "
    "camera? Output the thinking process in <think> </think> and final answer "
    "(the meter number only, without the unit) in <answer> </answer> tags.")
QWEN3_VL_PROMPT = (
    "Given this image, how far is the point marked by the red cross from the "
    "camera? Return only the final metric depth as <answer>NUMBER</answer>, "
    "where NUMBER is in meters. Do not include any other numbers or "
    "explanation.")

# Run-time hooks, module-level so that they can be replaced (tests do).
mark_model_loaded = utils.mark_model_loaded
write_adapter_metrics = utils.write_adapter_metrics


def load_rgb_image(path, max_side: int) -> Image.Image:
    """Loads an RGB image, EXIF-transposed and fit into `max_side` (if > 0).

    Args:
        path: The image file.
        max_side: Bound of both sides (`ImageOps.contain`, bicubic); 0 keeps
            the source size.

    Returns:
        The image.
    """
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side > 0:
        image = ImageOps.contain(image, (max_side, max_side),
                                 method=Image.Resampling.BICUBIC)
    return image


def grid_points(width: int, height: int,
                grid_size: int) -> list[tuple[int, int]]:
    """Returns the row-major query pixels of a `grid_size` x `grid_size` grid.

    Args:
        width: Image width.
        height: Image height.
        grid_size: Points per axis; 1 is the image centre only.

    Returns:
        (x, y) pixels.
    """
    grid_size = max(1, int(grid_size))
    if grid_size == 1:
        return [(width // 2, height // 2)]
    margin_x = max(1, width // (grid_size * 2))
    margin_y = max(1, height // (grid_size * 2))
    xs = np.linspace(margin_x, max(margin_x, width - 1 - margin_x), grid_size)
    ys = np.linspace(margin_y, max(margin_y, height - 1 - margin_y), grid_size)
    return [(int(round(x)), int(round(y))) for y in ys for x in xs]


def draw_depthlm_arrow(image: Image.Image, x: int, y: int) -> Image.Image:
    """Returns a copy of the image with DepthLM's red arrow at a pixel.

    Args:
        image: The image.
        x: Pixel column the arrow points at.
        y: Pixel row the arrow points at.

    Returns:
        The marked copy.
    """
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
    """Returns a copy of the image with a red circled cross on a pixel.

    Args:
        image: The image.
        x: Pixel column of the cross.
        y: Pixel row of the cross.

    Returns:
        The marked copy.
    """
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


def _user_turn(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    """Returns a single user turn: the image, then the question."""
    return [{
        "role":
            "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {
                "type": "text",
                "text": prompt
            },
        ],
    }]


def build_message(method: str, image: Image.Image, x: int,
                  y: int) -> list[dict[str, Any]]:
    """Returns the chat asking `method` for the depth of pixel (x, y).

    Args:
        method: "DEPTHLM", "VLM3" or "QWEN3_VL".
        image: The (resized) image.
        x: Query pixel column.
        y: Query pixel row.

    Returns:
        The chat messages: one user turn, the image before the question.
    """
    if method == "DEPTHLM":
        # facebookresearch/DepthLM_Official's
        # utils/datasets.py::generate_prompt_depth_sft() verbatim: the model
        # was SFT'd to produce a <think> block before <answer>. Its eval.py
        # builds the chat with convert_example_pixtral(...,
        # image_before_text=True): no system message (the Pixtral chat
        # template only takes a plain-string system prompt), the image before
        # the question.
        return _user_turn(draw_depthlm_arrow(image, x, y), DEPTHLM_PROMPT)
    if method == "VLM3":
        norm_x = int(round(2000.0 * x / max(1, image.width - 1)))
        norm_y = int(round(2000.0 * y / max(1, image.height - 1)))
        prompt = (
            f"Given this image, how far is the point at coordinates "
            f"({norm_x}, {norm_y}) from the camera? The coordinates are in "
            "normalised [0, 2000] format relative to image width and height. "
            "Output the thinking process in <think> </think> and final "
            "answer, the meter number only without the unit, in <answer> "
            "</answer> tags.")
        return _user_turn(image, prompt)
    return _user_turn(draw_point_marker(image, x, y), QWEN3_VL_PROMPT)


def parse_depth(text: str) -> float | None:
    """Returns the positive depth in an answer, or None.

    Tried in order: an <answer> tag, a number with a meter unit, a bare
    number.

    Args:
        text: The model's answer.

    Returns:
        The depth in meters, or None when none parses or it is not positive.
    """
    match = ANSWER_RE.search(text)
    if match is None:
        match = METER_RE.search(text)
    if match is None:
        match = SINGLE_NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def load_model_and_processor(args: argparse.Namespace):
    """Loads the VLM and its processor with Hugging Face transformers.

    Args:
        args: The parsed options (method, ckpt_path,
            attn_implementation).

    Returns:
        A (model, processor) tuple; the model in eval mode.
    """
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # pylint: disable=import-outside-toplevel
    import torch
    import transformers

    processor = transformers.AutoProcessor.from_pretrained(args.ckpt_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        if args.attn_implementation:
            kwargs["attn_implementation"] = args.attn_implementation
    if args.method == "DEPTHLM":
        model = _load_depthlm(args.ckpt_path, kwargs)
    else:
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            args.ckpt_path, **kwargs)
    model.eval()
    return model, processor


def _load_depthlm(ckpt_path: str, kwargs: dict[str, Any]):
    """Loads DepthLM: a Pixtral (LLaVA) checkpoint, else a Qwen2.5-VL one."""
    # pylint: disable=import-outside-toplevel
    import transformers

    config = transformers.AutoConfig.from_pretrained(ckpt_path)
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(
        str(item).lower()
        for item in getattr(config, "architectures", []) or [])
    if not any(name in model_type or name in architectures
               for name in ("pixtral", "llava")):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt_path, **kwargs)
    try:
        # Older transformers have no AutoModelForMultimodalLM.
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM.from_pretrained(ckpt_path, **kwargs)
    except ImportError:
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration.from_pretrained(
            ckpt_path, **kwargs)


def model_device(model: Any) -> str:
    """Returns the device of a model (its first parameter's), or "cpu".

    Args:
        model: The model.

    Returns:
        The device name.
    """
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def model_dtype(model: Any):
    """Returns the dtype of a model (its first parameter's), or float32.

    Args:
        model: The model.

    Returns:
        The torch dtype.
    """
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        import torch  # pylint: disable=import-outside-toplevel

        return torch.float32


def move_inputs_to_model(inputs: Any, model: Any) -> Any:
    """Moves processor outputs to the model's device, floats to its dtype.

    Args:
        inputs: The processor's batch.
        model: The model.

    Returns:
        The moved batch.
    """
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
    """Answers a batch of chats greedily.

    Args:
        model: The VLM.
        processor: Its processor.
        messages_batch: One chat per query.
        max_new_tokens: Generation length limit.

    Returns:
        The decoded answer of each chat.
    """
    import torch  # pylint: disable=import-outside-toplevel

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
    """Interpolates row-major grid samples bilinearly into a dense map.

    Missing samples (NaN or <= 0) take the median of the valid ones.

    Args:
        samples: (x, y, depth) per grid point, row-major.
        width: Output width.
        height: Output height.
        grid_size: Points per axis; 1 gives a constant map.

    Returns:
        A float32 (height, width) depth map.

    Raises:
        RuntimeError: No sample is valid.
    """
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
        if (0 <= row < grid_size and 0 <= col < grid_size and
                np.isfinite(depth) and depth > 0):
            grid[row, col] = depth
    dense = Image.fromarray(grid, mode="F").resize((width, height),
                                                   Image.Resampling.BILINEAR)
    depth = np.asarray(dense, dtype=np.float32).copy()
    depth[~np.isfinite(depth)] = fallback
    depth[depth <= 0] = fallback
    return depth


def _source_size(image_path) -> tuple[int, int]:
    """Returns (width, height) of an image after its EXIF transpose.

    The same transpose as `load_rgb_image`: an untransposed size would swap
    width and height for a 90/270 degree EXIF orientation.
    """
    with Image.open(image_path) as probe:
        return ImageOps.exif_transpose(probe).size


class PointQueryVLM(base.BaseModel):
    """Point-query VLM depth; each subclass is one method.

    The images are resized by `ImageOps.contain` to --image_load_resolution
    only (no --resolution), and only depth is exported (no point options).
    --max_images_per_run is the frame cap, 1 by default (the adapter's).
    """

    description = DESCRIPTION
    uses_resolution = False
    image_load_resolution_help = (
        "Bound of both image sides the VLM sees (0: source size).")
    exports_points = False
    uses_code_path = False
    default_image_load_resolution = 224
    default_max_images_per_run = 1

    @classmethod
    def add_arguments(cls, parser):
        """Adds --image_load_resolution and the query options.

        --point_grid_size defaults to POINT_QUERY_DEPTH_GRID_SIZE (the
        engine sets it), then to the MethodSpec's point_grid_size, then to 1
        (the image centre only).

        Args:
            parser: The model's parser.
        """
        env = os.environ
        grid_size = cls.spec_default("point_grid_size", 0) or 1
        parser.add_argument(
            "--point_grid_size",
            type=int,
            default=int(env.get("POINT_QUERY_DEPTH_GRID_SIZE", grid_size)),
            help=("Per-axis query grid size. 1 is center-only; larger values "
                  "are slower but less degenerate."),
        )
        parser.add_argument(
            "--query_batch_size",
            type=int,
            default=int(env.get("POINT_QUERY_DEPTH_BATCH_SIZE", "4")),
            help="Number of point prompts generated per model batch.",
        )
        parser.add_argument(
            "--require_spatial_queries",
            action="store_true",
            default=env.get("POINT_QUERY_REQUIRE_SPATIAL", "0") == "1",
            help=("Require a multi-point grid and reject images with no valid "
                  "answers instead of constant imputation."),
        )
        parser.add_argument("--max_new_tokens",
                            type=int,
                            default=int(
                                env.get("POINT_QUERY_MAX_NEW_TOKENS", "256")))
        parser.add_argument("--attn_implementation",
                            default=env.get("POINT_QUERY_ATTN_IMPLEMENTATION",
                                            "sdpa"))

    def __init__(self, args):
        super().__init__(args)
        self.processor = None

    def validate_args(self):
        """Rejects inconsistent options.

        Raises:
            ValueError: COLMAP export outside a depth-only run, no
                --export_depth (outside a dry run), or an invalid grid.
        """
        args = self.args
        if args.export_colmap and not args.depth_only:
            raise ValueError(f"{self.method_name} point-query adapter only "
                             "supports depth tracks.")
        if not args.export_depth and not args.dry_run:
            raise ValueError(
                f"{self.method_name} adapter requires --export_depth.")
        if args.point_grid_size < 1:
            raise ValueError("--point_grid_size must be positive")
        if args.require_spatial_queries and args.point_grid_size < 2:
            raise ValueError(
                "Spatial depth evaluation requires --point_grid_size >= 2")

    def load(self, device):
        del device  # transformers places the model (device_map).
        model, self.processor = load_model_and_processor(self.args)
        return model

    def preprocess(self, image_paths, batch_index=None):
        """Loads one image; the transform maps its resize back.

        Args:
            image_paths: The one source image of the call.
            batch_index: Unused (one image per call).

        Returns:
            The `ModelInput`: the resized PIL image and its transform.
        """
        image_path = image_paths[0]
        source_wh = _source_size(image_path)
        image = load_rgb_image(image_path, self.args.image_load_resolution)
        # ImageOps.contain is a uniform resize (no crop, no pad), so the
        # restore is a plain resize to the source size.
        transform = utils.SpatialTransform(source_wh=source_wh,
                                           resized_wh=(image.width,
                                                       image.height))
        return base.ModelInput(image_paths, image, [transform], batch_index)

    def query(self, image_path, image):
        """Asks the model for the depth of every grid point of one image.

        Args:
            image_path: The source image (for the records).
            image: The resized image.

        Returns:
            A (samples, records, parse_failures) tuple: (x, y, depth) per
            point (NaN when unparsed), one record per answer and the number
            of unparsed answers.
        """
        args = self.args
        query_batch_size = max(1, int(args.query_batch_size))
        points = grid_points(image.width, image.height, args.point_grid_size)
        samples = []
        records = []
        parse_failures = 0
        for start in range(0, len(points), query_batch_size):
            batch_points = points[start:start + query_batch_size]
            messages = [
                build_message(self.method_name, image, x, y)
                for x, y in batch_points
            ]
            outputs = generate_batch(self.model, self.processor, messages,
                                     args.max_new_tokens)
            for (x, y), text in zip(batch_points, outputs, strict=True):
                depth = parse_depth(text)
                if depth is None:
                    parse_failures += 1
                    depth = np.nan
                samples.append((x, y, float(depth)))
                records.append({
                    "image": str(image_path),
                    "x": x,
                    "y": y,
                    "depth": depth,
                    "text": text
                })
        return samples, records, parse_failures

    def forward(self, inputs):
        """Queries one image.

        Args:
            inputs: The output of `preprocess`.

        Returns:
            The `Prediction`: the dense depth on the resized image grid
            (None when no answer parsed); the samples, the answer records
            and the parse failures in `extra`.
        """
        image = inputs.data
        samples, records, parse_failures = self.query(inputs.image_paths[0],
                                                      image)
        valid = [
            depth for _x, _y, depth in samples
            if np.isfinite(depth) and depth > 0
        ]
        dense = None
        if valid:
            dense = interpolate_sparse_depth(samples, image.width, image.height,
                                             self.args.point_grid_size)
        return base.Prediction(depth=dense,
                               extra={
                                   "samples": samples,
                                   "valid": valid,
                                   "records": records,
                                   "parse_failures": parse_failures,
                               })

    def run(self):
        """Runs the adapter on the scene (always depth-only).

        Raises:
            ValueError: The options conflict or no images were found.
            RuntimeError: --require_spatial_queries and an image has no
                valid answer.
        """
        args = self.args
        self.validate_args()
        if (self.method_name == "DEPTHLM" and
                "POINT_QUERY_MAX_NEW_TOKENS" not in os.environ):
            # DepthLM thinks before it answers; 256 tokens would cut the
            # answer off. The official eval.py's --max_new_tokens default.
            args.max_new_tokens = max(args.max_new_tokens, 4096)

        self.images_dir = utils.canonical_image_root(args.data_path)
        self.image_paths = self.select_images()
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")
        if args.dry_run:
            utils.print_dry_run(args, self.method_label, len(self.image_paths))
            return

        self.model = self.load(self.device)
        self.started_at = mark_model_loaded()
        self.run_depth_only()

    def run_depth_only(self):
        """Queries every image and writes depths, query evidence, metrics."""
        args = self.args
        depths = []
        raw_records = []
        parse_failures = 0
        fallback_imputations = 0
        total_queries = 0
        fallback_depth = float(
            os.environ.get("POINT_QUERY_DEPTH_FALLBACK_METERS", "1.0"))
        last_valid_depth = None
        args.output_path.mkdir(parents=True, exist_ok=True)
        raw_path = args.output_path / "point_query_outputs.jsonl"
        raw_path.write_text("", encoding="utf-8")
        image_coverage = []

        for image_path in self.image_paths:
            inputs = self.preprocess([image_path])
            prediction = self.forward(inputs)
            samples = prediction.extra["samples"]
            valid_depths = prediction.extra["valid"]
            raw_records.extend(prediction.extra["records"])
            parse_failures += prediction.extra["parse_failures"]
            total_queries += len(samples)
            # Write the query evidence before raising, so that failed images
            # can be audited.
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
                    f"No valid point-depth answers for {image_path}; "
                    "constant imputation disabled")
            if valid_depths:
                last_valid_depth = float(np.median(valid_depths))
                depths.append(
                    utils.restore_prediction(prediction.depth,
                                             inputs.transforms[0]))
            else:
                if last_valid_depth is not None:
                    imputed_depth = last_valid_depth
                else:
                    imputed_depth = fallback_depth
                fallback_imputations += 1
                source_w, source_h = inputs.transforms[0].source_wh
                depths.append(
                    np.full((source_h, source_w),
                            imputed_depth,
                            dtype=np.float32))
                raw_records.append({
                    "image": str(image_path),
                    "depth": imputed_depth,
                    "fallback_imputed": True,
                    "text": "No parseable point-query depth response.",
                })

        # The last image's imputation record comes after its flush.
        if raw_records:
            with raw_path.open("a", encoding="utf-8") as handle:
                for record in raw_records:
                    handle.write(
                        json.dumps(record, ensure_ascii=True, allow_nan=False) +
                        "\n")
        utils.export_depth_predictions(args.output_path, self.image_paths,
                                       np.stack(depths, axis=0))
        args.output_path.mkdir(parents=True, exist_ok=True)
        write_adapter_metrics(
            output_path=args.output_path,
            method=self.method_label,
            image_paths=self.image_paths,
            started_at=self.started_at,
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
                    ("per-image valid median; reject all-invalid if spatial "
                     "queries required"),
                "parse_failures":
                    parse_failures,
                "fallback_imputations":
                    fallback_imputations,
                "protocol_note":
                    ("Dense maps are interpolated from text point-depth "
                     "queries."),
            },
        )


@registry.register_model
class VLM3(PointQueryVLM):
    """VLM3: the query pixel as normalized [0, 2000] coordinates."""

    name = "VLM3"


@registry.register_model
class DepthLM(PointQueryVLM):
    """DepthLM: a red arrow points at the query pixel."""

    name = "DEPTHLM"


@registry.register_model
class Qwen3VL(PointQueryVLM):
    """Qwen3-VL baseline: a red cross marks the query pixel."""

    name = "QWEN3_VL"


MODELS = {cls.name: cls for cls in (VLM3, DepthLM, Qwen3VL)}


def split_model_option(
    argv: Sequence[str] | None,
    models: dict[str, type[base.BaseModel]],
    description: str = "",
) -> tuple[str, list[str]]:
    """Splits the required --model NAME off the command line of a script.

    The scripts that serve several methods take --model to choose one.

    Args:
        argv: The arguments; None for sys.argv.
        models: Canonical method name -> model class the script serves.
        description: The script's --help description.

    Returns:
        A (canonical method name, remaining arguments) tuple.
    """

    def family_method(name: str) -> str:
        spec = base.BaseModel.method_spec(name)
        canonical = spec.name if spec is not None else name
        if canonical not in models:
            raise argparse.ArgumentTypeError(
                f"invalid choice: {name!r} (choose from "
                f"{', '.join(models)})")
        return canonical

    parser = argparse.ArgumentParser(description=description, add_help=False)
    parser.add_argument("--model", required=True, type=family_method)
    known, rest = parser.parse_known_args(argv)
    return known.model, rest


def parse_args(argv=None) -> argparse.Namespace:
    """Parses the command line of the model --model names."""
    method, rest = split_model_option(argv, MODELS, DESCRIPTION)
    return MODELS[method].build_parser(method).parse_args(rest)


def run(argv=None) -> None:
    """Runs the model --model names (VLM3, DEPTHLM or QWEN3_VL)."""
    method, rest = split_model_option(argv, MODELS, DESCRIPTION)
    MODELS[method].main(rest, method=method)
