# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Query commercial VLMs through OpenRouter for MonoDepth and export dense benchmark depth maps.

API VLMs return depth as text, like the local point-query baselines (``point_query_depth_adapter.py``), but each
request re-sends the whole image, so one call per queried pixel would multiply the cost by the grid size. This
adapter instead draws an ``N x N`` grid of numbered markers on the image, asks for every marker's metric depth in a
single call as a JSON object, and interpolates those samples into the dense ``depths/<stem>.npz`` layout with the same
``interpolate_sparse_depth`` bridge the point-query baselines use.

Responses are cached on disk keyed by model, prompt version, query settings and image bytes, so a rerun of a case
does not pay for the same request again. Every call's token usage and USD cost (as reported by OpenRouter) are
recorded in ``metrics.json``, and ``--max_cost_usd`` stops a run before it exceeds its budget.

The API key is read from ``OPENROUTER_API_KEY`` or from the file named by ``OPENROUTER_API_KEY_FILE``; it is never
written to any output.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.inference.adapter_utils import (  # noqa: E402
    SpatialTransform, canonical_image_root, export_depth_predictions,
    find_image_paths, mark_model_loaded, print_dry_run, restore_prediction,
    select_interval, write_adapter_metrics,
)
from scripts.inference.point_query_depth_adapter import (  # noqa: E402
    grid_points, interpolate_sparse_depth, load_rgb_image,
)

# Method name -> default OpenRouter model slug (the registry's ckpt_default; overridable per method via
# <METHOD>_CKPT_PATH like any other checkpoint).
OPENROUTER_MODELS: dict[str, str] = {
    "CLAUDE_OPUS_5_5": "anthropic/claude-opus-5.5",
    "GPT_6_ASTRA": "openai/gpt-6-astra",
    "GEMINI_4_ARGON": "google/gemini-4-argon",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Bump when the prompt or marker rendering changes, so cached answers to the old query are not reused.
PROMPT_VERSIONS = ("grid-v1", "grid-v2")
INTERPOLATIONS = ("bilinear", "joint_bilateral")
SYSTEM_PROMPTS = {
    "none":
        None,
    "depth-expert": (
        "You are an expert in monocular metric depth estimation. Given a single photo, you estimate the depth of "
        "marked points: the distance in meters from the camera along its optical axis (z-depth, not straight-line "
        "distance). Use cues such as object sizes, perspective, occlusion and the ground plane. Always answer in the "
        "exact JSON format requested."),
}
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>",
                             re.DOTALL | re.IGNORECASE)
PAIR_RE = re.compile(r"\"?(\d+)\"?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method_name",
                        required=True,
                        choices=tuple(OPENROUTER_MODELS))
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--ckpt_path",
        default="",
        help="OpenRouter model slug; defaults to the method's slug.")
    parser.add_argument("--track", choices=("rgb",), default="rgb")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--image_load_resolution",
                        type=int,
                        default=1024,
                        help="Long side of the image sent.")
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=0.0)
    parser.add_argument("--point_grid_size",
                        type=int,
                        default=int(
                            os.environ.get("OPENROUTER_DEPTH_GRID_SIZE", "16")))
    parser.add_argument("--concurrency",
                        type=int,
                        default=int(
                            os.environ.get("OPENROUTER_CONCURRENCY", "8")))
    parser.add_argument("--max_tokens",
                        type=int,
                        default=int(
                            os.environ.get("OPENROUTER_MAX_TOKENS", "8192")))
    parser.add_argument(
        "--reasoning_effort",
        default=os.environ.get("OPENROUTER_REASONING_EFFORT", "low"),
        help=
        "OpenRouter unified reasoning effort (none/minimal/low/medium/high); 'default' sends no setting.",
    )
    parser.add_argument("--max_retries",
                        type=int,
                        default=int(
                            os.environ.get("OPENROUTER_MAX_RETRIES", "6")))
    parser.add_argument("--timeout",
                        type=float,
                        default=float(
                            os.environ.get("OPENROUTER_TIMEOUT", "300")))
    parser.add_argument(
        "--max_cost_usd",
        type=float,
        default=float(os.environ.get("OPENROUTER_MAX_COST_USD", "0")),
        help=
        "Stop issuing requests once this run's billed cost reaches this many USD (0 = no cap).",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path(
            os.environ.get(
                "OPENROUTER_CACHE_DIR",
                "~/.cache/feedforward_benchmark/openrouter")).expanduser(),
    )
    parser.add_argument(
        "--allow_data_collection",
        action="store_true",
        default=os.environ.get("OPENROUTER_ALLOW_DATA_COLLECTION", "0") == "1",
        help=
        "Allow providers that may store or train on prompts (OpenRouter data_collection=allow).",
    )
    parser.add_argument(
        "--prompt_version",
        choices=PROMPT_VERSIONS,
        default=os.environ.get("OPENROUTER_PROMPT_VERSION", "grid-v2"),
        help=
        "grid-v1: marked image only. grid-v2: clean + marked image, and dots on the sky may be answered 'sky'.",
    )
    parser.add_argument(
        "--system_prompt",
        choices=tuple(SYSTEM_PROMPTS),
        default=os.environ.get("OPENROUTER_SYSTEM_PROMPT", "none"),
        help=
        "Optional system turn framing the task (none = a single user turn, like the local VLM baselines).",
    )
    parser.add_argument(
        "--interpolation",
        choices=INTERPOLATIONS,
        default=os.environ.get("OPENROUTER_DEPTH_INTERPOLATION", "bilinear"),
        help=
        "How the sampled depths become a dense map; joint_bilateral follows the RGB image's edges.",
    )
    parser.add_argument(
        "--refine_points",
        type=int,
        default=int(os.environ.get("OPENROUTER_REFINE_POINTS", "0")),
        help=
        "Up to this many extra points around depth edges, asked in a second call (0 = single pass).",
    )
    parser.add_argument(
        "--completion_followups",
        type=int,
        default=int(os.environ.get("OPENROUTER_COMPLETION_FOLLOWUPS", "2")),
        help="Follow-up rounds asking for dots an answer left out (0 = off).",
    )
    parser.add_argument("--refine_ratio",
                        type=float,
                        default=float(
                            os.environ.get("OPENROUTER_REFINE_RATIO", "1.3")))
    parser.add_argument(
        "--no_depth_clamp",
        action="store_true",
        default=os.environ.get("OPENROUTER_DEPTH_CLAMP", "1") == "0",
        help=
        "Do not clamp sampled depths to the scene's metadata.json [min_depth, max_depth] before interpolation.",
    )
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", default="0")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key and os.environ.get("OPENROUTER_API_KEY_FILE"):
        key_file = Path(os.environ["OPENROUTER_API_KEY_FILE"]).expanduser()
        text = key_file.read_text(encoding="utf-8").strip()
        # Accept a bare key or a dotenv-style OPENROUTER_API_KEY=... line.
        match = re.search(r"OPENROUTER_API_KEY\s*=\s*['\"]?([^'\"\s]+)", text)
        key = match.group(1) if match else text.splitlines()[0].strip()
    if not key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY or OPENROUTER_API_KEY_FILE to query OpenRouter."
        )
    return key


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font.
        return ImageFont.load_default()


def draw_numbered_markers(image: Image.Image,
                          points: list[tuple[int, int]]) -> Image.Image:
    """Mark each query pixel with a small red dot and its 1-based number on a white tag beside it."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    side = min(marked.size)
    radius = max(3, side // 170)
    font = _font(max(10, side // 45))
    for index, (x, y) in enumerate(points, start=1):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=(255, 0, 0),
                     outline=(255, 255, 255))
        label = str(index)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tag_w, tag_h = right - left + 4, bottom - top + 2
        tag_x = min(max(0, x + radius + 1), marked.width - tag_w)
        tag_y = min(max(0, y - radius - tag_h), marked.height - tag_h)
        draw.rectangle((tag_x, tag_y, tag_x + tag_w, tag_y + tag_h),
                       fill=(255, 255, 255))
        draw.text((tag_x + 2 - left, tag_y + 1 - top),
                  label,
                  fill=(200, 0, 0),
                  font=font)
    return marked


ANSWER_FORMAT_V2 = (
    "Return the final answer as a single JSON object mapping each dot number (as a string) to its depth in meters "
    '(or "sky"), wrapped in <answer> </answer> tags, e.g. <answer>{"1": 2.4, "2": "sky", "3": 3.1, ...}</answer>.'
)


def build_prompt(num_points: int, version: str = "grid-v1") -> str:
    """The first-pass question. ``grid-v2`` adds the clean photo before the marked one and allows "sky"."""
    if version == "grid-v2":
        return (
            f"The first image is a photo. The second image is the same photo with {num_points} red dots, each "
            f"labelled with a number from 1 to {num_points}; use the first image to see the scene the markers "
            "cover. For every dot, estimate the metric depth of the scene point under the dot centre: its distance "
            "from the camera along the camera's viewing (optical) axis, in meters. If a dot lies on the sky or on "
            'anything effectively infinitely far away, answer "sky" instead of a number. Give a numeric estimate '
            "for every other dot even when unsure.\n" + ANSWER_FORMAT_V2)
    return (
        f"This image has {num_points} red dots, each labelled with a number from 1 to {num_points}. "
        "For every dot, estimate the metric depth of the scene point under the dot centre: its distance from the "
        "camera along the camera's viewing (optical) axis, in meters. Judge the scene behind the marker, not the "
        "marker itself. Give an estimate for every dot even when unsure.\n"
        "Return the final answer as a single JSON object mapping each dot number (as a string) to its depth in "
        'meters, wrapped in <answer> </answer> tags, e.g. <answer>{"1": 2.4, "2": 3.1, ...}</answer>.'
    )


def build_refine_prompt(num_points: int, version: str = "grid-v1") -> str:
    """The second-pass question: depths of extra dots placed on both sides of the depth edges found in pass one."""
    images = ("The first image is the same photo again and the second shows it"
              if version == "grid-v2" else "This image shows the same photo")
    sky = ' Answer "sky" for dots on the sky.' if version == "grid-v2" else ""
    answer = ANSWER_FORMAT_V2 if version == "grid-v2" else (
        "Return the final answer as a single JSON object mapping each new dot number (as a string) to its depth in "
        'meters, wrapped in <answer> </answer> tags, e.g. <answer>{"1": 2.4, "2": 3.1, ...}</answer>.'
    )
    return (
        f"{images} with {num_points} new red dots labelled 1 to {num_points}, placed on either side of depth "
        "edges. Estimate the metric depth of the scene point under each new dot centre in the same way (distance "
        "along the optical axis, in meters), consistent with your previous estimates; dots on different sides of an "
        f"edge may lie on different surfaces.{sky} Give a numeric estimate for every other dot even when unsure.\n"
        + answer)


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()).decode("ascii")


def user_turn(image_urls: list[str] | str, prompt: str) -> dict[str, Any]:
    urls = [image_urls] if isinstance(image_urls, str) else list(image_urls)
    content: list[dict[str, Any]] = [{
        "type": "image_url",
        "image_url": {
            "url": url
        }
    } for url in urls]
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def system_turn(args: argparse.Namespace) -> list[dict[str, Any]]:
    text = SYSTEM_PROMPTS[getattr(args, "system_prompt", "none")]
    return [{"role": "system", "content": text}] if text else []


def build_payload(
    model: str,
    image_urls: list[str] | str,
    prompt: str,
    args: argparse.Namespace,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            *system_turn(args), *(history or []),
            user_turn(image_urls, prompt)
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "usage": {
            "include": True
        },
    }
    if args.reasoning_effort and args.reasoning_effort != "default":
        payload["reasoning"] = {
            "effort": args.reasoning_effort,
            "exclude": True
        }
    if not args.allow_data_collection:
        payload["provider"] = {"data_collection": "deny"}
    return payload


def sky_indices(text: str) -> set[int]:
    """Dot numbers answered "sky" in the final answer (grid-v2 prompts)."""
    blocks = ANSWER_BLOCK_RE.findall(text or "")
    body = blocks[-1] if blocks else (text or "")
    return {
        int(number) for number in re.findall(
            r"\"?(\d+)\"?\s*:\s*\"sky\"", body, flags=re.IGNORECASE)
    }


def format_dot_ranges(numbers: list[int]) -> str:
    """``[1, 2, 3, 7, 9, 10]`` -> ``"1-3, 7, 9-10"``."""
    ranges: list[str] = []
    start = previous = None
    for number in sorted(numbers):
        if previous is not None and number == previous + 1:
            previous = number
            continue
        if start is not None:
            ranges.append(f"{start}-{previous}" if previous !=
                          start else str(start))
        start = previous = number
    if start is not None:
        ranges.append(f"{start}-{previous}" if previous !=
                      start else str(start))
    return ", ".join(ranges)


def build_followup_prompt(missing: list[int], version: str = "grid-v1") -> str:
    """Ask for exactly the dots a previous answer left out, in the same format (no new images)."""
    sky = ' (or "sky")' if version == "grid-v2" else ""
    return (
        f"Your answer did not include dots {format_dot_ranges(missing)}. Give the metric depth{sky} for exactly "
        "these dots, estimated the same way and consistent with your previous estimates. Return one JSON object "
        "mapping each of these dot numbers (as a string) to its depth, wrapped in <answer> </answer> tags."
    )


def count_sky_answers(text: str) -> int:
    """How many entries of the final answer are the string "sky" (grid-v2 prompts)."""
    blocks = ANSWER_BLOCK_RE.findall(text or "")
    body = blocks[-1] if blocks else (text or "")
    return len(re.findall(r":\s*\"sky\"", body, flags=re.IGNORECASE))


def parse_grid_answer(text: str, num_points: int) -> list[float]:
    """Depths in marker order (NaN where missing or invalid) from a JSON-object (or list) answer."""
    blocks = ANSWER_BLOCK_RE.findall(text or "")
    body = blocks[-1] if blocks else (text or "")
    values: dict[int, Any] = {}
    start, end = body.find("{"), body.rfind("}")
    parsed: Any = None
    if start != -1 and end > start:
        try:
            parsed = json.loads(body[start:end + 1])
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        start, end = body.find("["), body.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(body[start:end + 1])
            except json.JSONDecodeError:
                parsed = None
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if str(key).strip().isdigit():
                values[int(str(key).strip())] = value
    elif isinstance(parsed, list):
        values = {index: value for index, value in enumerate(parsed, start=1)}
    else:
        values = {int(key): value for key, value in PAIR_RE.findall(body)}
    depths: list[float] = []
    for index in range(1, num_points + 1):
        try:
            depth = float(values.get(index, np.nan))
        except (TypeError, ValueError):
            depth = np.nan
        depths.append(
            depth if np.isfinite(depth) and depth > 0 else float("nan"))
    return depths


def scene_depth_limits(data_path: Path) -> tuple[float | None, float | None]:
    """The scene's evaluation depth range (``min_depth``, ``max_depth``) from its MonoDepth ``metadata.json``.

    The scene folder's own ``metadata.json`` is checked first, then the dataset's (whose ``scenes`` list has one
    entry per scene). ``None`` where a bound is not declared.
    """
    for meta_path in (data_path / "metadata.json",
                      data_path.parent / "metadata.json"):
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entries = [
            entry for entry in meta.get("scenes", [])
            if entry.get("scene_name") == data_path.name
        ]
        entry = entries[0] if entries else meta
        if "min_depth" in entry or "max_depth" in entry:
            low, high = entry.get("min_depth"), entry.get("max_depth")
            return (None if low is None else float(low),
                    None if high is None else float(high))
    return None, None


def clamp_samples(
        samples: list[tuple[int, int, float]], low: float | None,
        high: float | None) -> tuple[list[tuple[int, int, float]], int]:
    """Clamp each sampled depth into ``[low, high]`` before interpolation; returns the samples and how many moved.

    Answers far outside the evaluated range (e.g. thousands of metres for sky) would otherwise bleed through the
    bilinear interpolation into neighbouring in-range pixels; the evaluator's own post-alignment clip cannot undo
    that.
    """
    clamped: list[tuple[int, int, float]] = []
    moved = 0
    for x, y, depth in samples:
        if np.isfinite(depth):
            new_depth = depth
            if high is not None:
                new_depth = min(new_depth, high)
            if low is not None:
                new_depth = max(new_depth, low)
            moved += int(new_depth != depth)
            depth = new_depth
        clamped.append((x, y, depth))
    return clamped, moved


def joint_bilateral_depth(
    samples: list[tuple[int, int, float]],
    image: Image.Image,
    grid_size: int,
    spatial_sigma_cells: float = 1.0,
    range_sigma: float = 0.1,
    patch_radius: int = 3,
) -> np.ndarray:
    """Densify scattered depth samples with joint bilateral upsampling guided by the RGB image (Kopf et al. 2007).

    Each pixel takes a weighted mean of the sampled log-depths, weighted by spatial distance (sigma in grid cells)
    and by the colour difference between the pixel and the mean colour around each sample, so depth follows the
    image's edges instead of being blended across them. No learned prior is involved: every value comes from the
    model's own answers. A small spatial-only term keeps pixels unlike every sample well defined. Works for
    irregular sample sets (e.g. refinement points), not only full grids.
    """
    guide = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = guide.shape[:2]
    valid = [(x, y, d) for x, y, d in samples if np.isfinite(d) and d > 0]
    if not valid:
        raise RuntimeError("No valid depth samples to densify.")
    cell = float(np.sqrt(width * height)) / max(1, grid_size)
    inv_two_s2 = 1.0 / (2.0 * (spatial_sigma_cells * cell)**2)
    inv_two_r2 = 1.0 / (2.0 * range_sigma**2)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    numerator = np.zeros((height, width), dtype=np.float64)
    denominator = np.zeros((height, width), dtype=np.float64)
    spatial_num = np.zeros((height, width), dtype=np.float64)
    spatial_den = np.zeros((height, width), dtype=np.float64)
    for x, y, depth in valid:
        x0, x1 = max(0, x - patch_radius), min(width, x + patch_radius + 1)
        y0, y1 = max(0, y - patch_radius), min(height, y + patch_radius + 1)
        colour = guide[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        spatial = np.exp(-((xs - x)**2 + (ys - y)**2) * inv_two_s2)
        colour_w = np.exp(-np.sum((guide - colour)**2, axis=2) * inv_two_r2)
        weight = spatial * colour_w
        log_depth = float(np.log(depth))
        numerator += weight * log_depth
        denominator += weight
        spatial_num += spatial * log_depth
        spatial_den += spatial
    epsilon = 1e-3
    log_dense = (numerator + epsilon * spatial_num) / (
        denominator + epsilon * spatial_den + 1e-30)
    return np.exp(log_dense).astype(np.float32)


def fill_missing_grid(grid: np.ndarray) -> np.ndarray:
    """Fill NaN grid cells ("sky" or unparsed answers) from their answered neighbours, in log depth.

    Each pass gives every missing cell with at least one finite 8-neighbour the mean log depth of those neighbours,
    so gaps are filled from the nearest answers inward. (A single scene-median value per missing cell, as the local
    point-query bridge does, turns every gap into a flat block.) Answered cells are never changed.
    """
    log_grid = np.log(np.where(np.isfinite(grid) & (grid > 0), grid, np.nan))
    if not np.isfinite(log_grid).any():
        raise RuntimeError("No valid depth samples to densify.")
    while not np.isfinite(log_grid).all():
        padded = np.pad(log_grid, 1, constant_values=np.nan)
        rows, cols = log_grid.shape
        neighbours = np.stack([
            padded[1 + dy:1 + dy + rows, 1 + dx:1 + dx + cols]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if dy or dx
        ])
        counts = np.isfinite(neighbours).sum(axis=0)
        sums = np.nansum(neighbours, axis=0)
        fillable = ~np.isfinite(log_grid) & (counts > 0)
        log_grid[fillable] = sums[fillable] / counts[fillable]
    return np.exp(log_grid)


def interpolate_grid_depth(samples: list[tuple[int, int, float]], width: int,
                           height: int, grid_size: int) -> np.ndarray:
    """Bilinear dense map from the row-major grid samples, with missing cells filled from their neighbours."""
    grid = np.array(
        [depth for _x, _y, depth in samples[:grid_size * grid_size]],
        dtype=np.float64)
    grid = fill_missing_grid(grid.reshape(grid_size,
                                          grid_size)).astype(np.float32)
    dense = Image.fromarray(grid, mode="F").resize((width, height),
                                                   Image.Resampling.BILINEAR)
    return np.asarray(dense, dtype=np.float32).copy()


def densify(samples: list[tuple[int, int, float]], image: Image.Image,
            grid_size: int, method: str) -> np.ndarray:
    """Dense depth on the resized image grid from the sampled depths."""
    if method == "bilinear":
        return interpolate_grid_depth(samples, image.width, image.height,
                                      grid_size)
    if method == "joint_bilateral":
        return joint_bilateral_depth(samples, image, grid_size)
    raise ValueError(f"Unknown densification method {method!r}")


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class UsageLedger:
    max_cost_usd: float
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self) -> None:
        with self.lock:
            if self.max_cost_usd > 0 and self.cost_usd >= self.max_cost_usd:
                raise BudgetExceeded(
                    f"OpenRouter budget reached: ${self.cost_usd:.4f} >= ${self.max_cost_usd:.4f}"
                )

    def add(self, usage: dict[str, Any] | None, cached: bool) -> None:
        with self.lock:
            if cached:
                self.cache_hits += 1
                return
            self.api_calls += 1
            usage = usage or {}
            self.cost_usd += float(usage.get("cost") or 0.0)
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            details = usage.get("completion_tokens_details") or {}
            self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)


def post_json(payload: dict[str, Any], api_key: str,
              timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "SpatialArena MonoDepth",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_openrouter(payload: dict[str, Any], api_key: str,
                    args: argparse.Namespace) -> dict[str, Any]:
    """POST with exponential backoff on rate limits, transient server errors and network failures."""
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            body = post_json(payload, api_key, args.timeout)
            if "error" in body and not body.get("choices"):
                error = body["error"]
                code = int(error.get("code") or 0) if str(
                    error.get("code") or "").isdigit() else 0
                if code and code not in RETRYABLE_STATUS:
                    raise RuntimeError(
                        f"OpenRouter error {code}: {error.get('message')}")
                raise ConnectionError(
                    f"OpenRouter error {code}: {error.get('message')}")
            return body
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code not in RETRYABLE_STATUS:
                raise RuntimeError(
                    f"OpenRouter HTTP {exc.code}: {detail}") from exc
            last_error = RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < args.max_retries:
            time.sleep(min(60.0, 2.0**attempt) + 0.1 * attempt)
    raise RuntimeError(
        f"OpenRouter request failed after {args.max_retries + 1} attempts: {last_error}"
    )


def refinement_points(
    samples: list[tuple[int, int, float]],
    image: Image.Image,
    grid_size: int,
    max_points: int = 64,
    ratio: float = 1.3,
) -> list[tuple[int, int]]:
    """Extra query pixels on both sides of the depth edges the first pass revealed.

    For every pair of 4-neighbouring grid samples whose depths differ by more than ``ratio``, the strongest image
    edge on the segment between them is located and one point is placed just before and one just after it. Pairs
    are taken in order of decreasing depth ratio until ``max_points`` points are chosen; points too close to a grid
    sample or to an already chosen point are skipped.
    """
    grid = samples[:grid_size * grid_size]
    if len(grid) < grid_size * grid_size or max_points <= 0:
        return []
    gray = np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(2)),
                      dtype=np.float32)
    grad_y, grad_x = np.gradient(gray)
    gradient = np.hypot(grad_x, grad_y)
    height, width = gray.shape
    cell = float(np.sqrt(width * height)) / grid_size
    offset = max(3.0, 0.15 * cell)
    min_dist = 0.25 * cell
    pairs = []
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            for other in ((index + 1) if col + 1 < grid_size else None,
                          (index + grid_size) if row + 1 < grid_size else None):
                if other is None:
                    continue
                (x0, y0, d0), (x1, y1, d1) = grid[index], grid[other]
                if np.isfinite(d0) and np.isfinite(d1) and d0 > 0 and d1 > 0:
                    pair_ratio = max(d0, d1) / min(d0, d1)
                    if pair_ratio > ratio:
                        pairs.append((pair_ratio, (x0, y0), (x1, y1)))
    pairs.sort(key=lambda item: -item[0])
    taken: list[tuple[int, int]] = [(x, y) for x, y, _d in grid]
    chosen: list[tuple[int, int]] = []
    for _ratio, (x0, y0), (x1, y1) in pairs:
        length = float(np.hypot(x1 - x0, y1 - y0))
        steps = np.linspace(0.15, 0.85, max(3, int(length)))
        xs = np.clip(np.round(x0 + steps * (x1 - x0)).astype(int), 0, width - 1)
        ys = np.clip(
            np.round(y0 + steps * (y1 - y0)).astype(int), 0, height - 1)
        best = int(np.argmax(gradient[ys, xs]))
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        for sign in (-1.0, 1.0):
            px = int(
                round(min(max(xs[best] + sign * offset * ux, 0), width - 1)))
            py = int(
                round(min(max(ys[best] + sign * offset * uy, 0), height - 1)))
            if all(np.hypot(px - qx, py - qy) >= min_dist for qx, qy in taken):
                taken.append((px, py))
                chosen.append((px, py))
        if len(chosen) >= max_points:
            break
    return chosen[:max_points]


def cache_key(model: str,
              args: argparse.Namespace,
              image_bytes_sha1: str,
              extra: Any = None) -> str:
    settings = {
        "model": model,
        "prompt_version": getattr(args, "prompt_version", "grid-v1"),
        "grid": args.point_grid_size,
        "image_load_resolution": args.image_load_resolution,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "image_sha1": image_bytes_sha1,
    }
    if getattr(
            args, "system_prompt", "none"
    ) != "none":  # absent for "none", so older cache entries stay valid
        settings["system_prompt"] = args.system_prompt
    if extra is not None:  # the refinement pass: its points and the first-pass answer it follows
        settings["refine"] = extra
    return hashlib.sha1(json.dumps(settings,
                                   sort_keys=True).encode("utf-8")).hexdigest()


def cached_ask(
    cache_path: Path,
    build: Any,
    num_points: int,
    api_key: str,
    args: argparse.Namespace,
    ledger: UsageLedger,
) -> tuple[str, Any, Any, bool]:
    """One cached OpenRouter request: returns (answer text, usage, generation id, whether it came from the cache).

    ``build`` is called only on a cache miss and returns the request payload.
    """
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            ledger.add(cached.get("usage"), cached=True)
            return cached["text"], cached.get("usage"), cached.get("id"), True
        except (json.JSONDecodeError, KeyError):
            pass
    ledger.check()
    body = call_openrouter(build(), api_key, args)
    choice = (body.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    usage, generation_id = body.get("usage"), body.get("id")
    ledger.add(usage, cached=False)
    # Only cache answers that parsed, so a malformed response is retried on the next run instead of replayed.
    if any(np.isfinite(parse_grid_answer(
            text, num_points))) or count_sky_answers(text):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(
            f".tmp{os.getpid()}.{threading.get_ident()}")
        tmp_path.write_text(json.dumps({
            "text": text,
            "usage": usage,
            "id": generation_id
        }),
                            encoding="utf-8")
        tmp_path.replace(cache_path)
    return text, usage, generation_id, False


def query_image(
    image_path: str,
    model: str,
    api_key: str,
    args: argparse.Namespace,
    ledger: UsageLedger,
) -> dict[str, Any]:
    """Query one image (and, with refinement, a second pass); returns samples, transform and the raw records."""
    version = getattr(args, "prompt_version", "grid-v1")
    with Image.open(image_path) as probe:
        source_wh = ImageOps.exif_transpose(probe).size
    image = load_rgb_image(image_path, args.image_load_resolution)
    transform = SpatialTransform(source_wh=source_wh,
                                 resized_wh=(image.width, image.height))
    points = grid_points(image.width, image.height, args.point_grid_size)
    image_sha1 = hashlib.sha1(Path(image_path).read_bytes()).hexdigest()
    model_dir = args.cache_dir / model.replace("/", "__")

    def first_images() -> list[str]:
        marked = encode_image(draw_numbered_markers(image, points))
        return [encode_image(image), marked
               ] if version == "grid-v2" else [marked]

    prompt = build_prompt(len(points), version)
    first_key = cache_key(model, args, image_sha1)
    text, usage, generation_id, cached = cached_ask(
        model_dir / f"{first_key}.json",
        lambda: build_payload(model, first_images(), prompt, args),
        len(points),
        api_key,
        args,
        ledger,
    )
    depths = parse_grid_answer(text, len(points))
    sky = sky_indices(text)
    # Completion follow-ups: an answer that stops partway (e.g. "only dots 1-64 are covered") is asked, in the same
    # conversation, for exactly the dots it left out -- a request for the rest of the answer, not a re-sample.
    followups: list[dict[str, Any]] = []
    conversation_texts = [text]
    for round_index in range(
            1,
            int(getattr(args, "completion_followups", 0) or 0) + 1):
        missing = [
            i + 1
            for i, d in enumerate(depths)
            if not np.isfinite(d) and (i + 1) not in sky
        ]
        if not missing:
            break
        follow_prompt = build_followup_prompt(missing, version)

        def followup_payload(
            texts=tuple(conversation_texts),
            follow_prompt=follow_prompt,
            asked=tuple(followups)) -> dict[str, Any]:
            turns = [
                user_turn(first_images(), prompt), {
                    "role": "assistant",
                    "content": texts[0]
                }
            ]
            for done, later_text in zip(asked, texts[1:]):
                turns += [{
                    "role": "user",
                    "content": done["prompt"]
                }, {
                    "role": "assistant",
                    "content": later_text
                }]
            return build_payload(model, [], follow_prompt, args, history=turns)

        follow_key = cache_key(
            model,
            args,
            image_sha1,
            extra={
                "followup":
                    round_index,
                "missing":
                    missing,
                "conversation":
                    hashlib.sha1("\x00".join(conversation_texts).encode("utf-8")
                                ).hexdigest()
            })
        follow_text, follow_usage, follow_id, follow_cached = cached_ask(
            model_dir / f"{follow_key}.json", followup_payload, len(points),
            api_key, args, ledger)
        follow_depths = parse_grid_answer(follow_text, len(points))
        follow_sky = sky_indices(follow_text) & set(missing)
        filled = [n for n in missing if np.isfinite(follow_depths[n - 1])]
        for number in filled:
            depths[number - 1] = follow_depths[number - 1]
        sky |= follow_sky
        followups.append({
            "round": round_index,
            "prompt": follow_prompt,
            "missing_before": len(missing),
            "filled": len(filled),
            "sky": len(follow_sky),
            "generation_id": follow_id,
            "cached": follow_cached,
            "usage": follow_usage,
            "text": follow_text
        })
        conversation_texts.append(follow_text)
        if not filled and not follow_sky:
            break  # the model will not complete it; stop asking
    samples = [(x, y, d) for (x, y), d in zip(points, depths, strict=True)]
    records = [{
        "image": image_path,
        "model": model,
        "pass": 1,
        "generation_id": generation_id,
        "cached": cached,
        "parsed_points": int(np.isfinite(depths).sum()),
        "sky_points": len(sky),
        "points": points,
        "depths": [None if not np.isfinite(d) else d for d in depths],
        "usage": usage,
        "text": text,
        "followups": followups,
    }]

    refine_budget = int(getattr(args, "refine_points", 0) or 0)
    extra = refinement_points(
        samples, image, args.point_grid_size, refine_budget,
        float(getattr(args, "refine_ratio", 1.3))) if refine_budget else []
    if extra:

        def second_payload() -> dict[str, Any]:
            marked = encode_image(draw_numbered_markers(image, extra))
            urls = [encode_image(image), marked
                   ] if version == "grid-v2" else [marked]
            history = [
                user_turn(first_images(), prompt), {
                    "role": "assistant",
                    "content": text
                }
            ]
            return build_payload(model,
                                 urls,
                                 build_refine_prompt(len(extra), version),
                                 args,
                                 history=history)

        second_key = cache_key(
            model,
            args,
            image_sha1,
            extra={
                "points": extra,
                "first": hashlib.sha1(text.encode("utf-8")).hexdigest()
            })
        text2, usage2, generation_id2, cached2 = cached_ask(
            model_dir / f"{second_key}.json", second_payload, len(extra),
            api_key, args, ledger)
        depths2 = parse_grid_answer(text2, len(extra))
        samples += [(x, y, d) for (x, y), d in zip(extra, depths2, strict=True)]
        records.append({
            "image": image_path,
            "model": model,
            "pass": 2,
            "generation_id": generation_id2,
            "cached": cached2,
            "parsed_points": int(np.isfinite(depths2).sum()),
            "sky_points": count_sky_answers(text2),
            "points": extra,
            "depths": [None if not np.isfinite(d) else d for d in depths2],
            "usage": usage2,
            "text": text2,
        })
    return {
        "image_path":
            image_path,
        "image":
            image,
        "transform":
            transform,
        "samples":
            samples,
        "record":
            records[0] if len(records) == 1 else {
                **records[0], "refinement": records[1]
            },
    }


def run() -> None:
    args = parse_args()
    if args.export_colmap and not args.depth_only:
        raise ValueError(
            f"{args.method_name} OpenRouter adapter only supports depth tracks."
        )
    if not args.export_depth and not args.dry_run:
        raise ValueError(f"{args.method_name} adapter requires --export_depth.")
    if args.point_grid_size < 2:
        raise ValueError(
            "--point_grid_size must be at least 2 (1 gives a constant depth map)."
        )
    if args.refine_points and args.interpolation == "bilinear":
        raise ValueError(
            "--refine_points needs --interpolation joint_bilateral (refinement points are off-grid)."
        )
    model = args.ckpt_path or OPENROUTER_MODELS[args.method_name]

    images_dir = canonical_image_root(args.data_path)
    image_paths = select_interval(find_image_paths(images_dir), args.interval,
                                  args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, args.method_name, len(image_paths))
        return

    api_key = read_api_key()
    started_at = mark_model_loaded()
    ledger = UsageLedger(max_cost_usd=args.max_cost_usd)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        results = list(
            pool.map(
                lambda path: query_image(path, model, api_key, args, ledger),
                image_paths))

    clamp_low, clamp_high = (
        None,
        None) if args.no_depth_clamp else scene_depth_limits(args.data_path)
    clamped_points = 0
    for result in results:
        result["samples"], moved = clamp_samples(result["samples"], clamp_low,
                                                 clamp_high)
        result["record"]["clamped_points"] = moved
        # Sky answers are excluded on purpose, so they are neither failures nor samples.
        result["sky"] = result["record"]["sky_points"] + result["record"].get(
            "refinement", {}).get("sky_points", 0)
        clamped_points += moved
    image_medians = [
        float(np.nanmedian([d
                            for _x, _y, d in result["samples"]]))
        for result in results
        if any(np.isfinite(d) for _x, _y, d in result["samples"])
    ]
    # A frame with no parseable answer gets the scene's median depth (deterministic, unlike "last valid" under
    # concurrency), or OPENROUTER_DEPTH_FALLBACK_METERS when no frame of the scene parsed at all.
    scene_fallback = float(
        np.median(image_medians)) if image_medians else float(
            os.environ.get("OPENROUTER_DEPTH_FALLBACK_METERS", "1.0"))
    depths: list[np.ndarray] = []
    raw_records: list[dict[str, Any]] = []
    parse_failures = 0
    sky_points = 0
    followup_calls = sum(
        len(result["record"].get("followups", [])) for result in results)
    followup_filled = sum(f["filled"]
                          for result in results
                          for f in result["record"].get("followups", []))
    fallback_imputations = 0
    for result in results:
        samples = result["samples"]
        missing = sum(1 for _x, _y, d in samples if not np.isfinite(d))
        parse_failures += missing - result["sky"]
        sky_points += result["sky"]
        record = result["record"]
        if missing == len(samples):
            fallback_imputations += 1
            source_w, source_h = result["transform"].source_wh
            depths.append(
                np.full((source_h, source_w), scene_fallback, dtype=np.float32))
            record["fallback_imputed"] = scene_fallback
        else:
            image = result["image"]
            dense = densify(samples, image, args.point_grid_size,
                            args.interpolation)
            depths.append(restore_prediction(dense, result["transform"]))
        raw_records.append(record)

    export_depth_predictions(args.output_path, image_paths,
                             np.stack(depths, axis=0))
    args.output_path.mkdir(parents=True, exist_ok=True)
    (args.output_path / "openrouter_outputs.jsonl").write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=True) for record in raw_records) +
        "\n",
        encoding="utf-8",
    )
    write_adapter_metrics(
        output_path=args.output_path,
        method=args.method_name,
        image_paths=image_paths,
        started_at=started_at,
        checkpoint=model,
        depth_only=True,
        metrics_extra={
            "track":
                args.track,
            "adapter":
                "openrouter_vlm_depth",
            "openrouter_model":
                model,
            "prompt_version":
                args.prompt_version,
            "system_prompt":
                args.system_prompt,
            "interpolation":
                args.interpolation,
            "refine_points":
                args.refine_points,
            "sky_points":
                sky_points,
            "completion_followups":
                args.completion_followups,
            "followup_calls":
                followup_calls,
            "followup_filled_points":
                followup_filled,
            "point_grid_size":
                args.point_grid_size,
            "image_long_side":
                args.image_load_resolution,
            "reasoning_effort":
                args.reasoning_effort,
            "total_point_queries":
                sum(len(result["samples"]) for result in results),
            "parse_failures":
                parse_failures,
            "fallback_imputations":
                fallback_imputations,
            "depth_clamp":
                None if args.no_depth_clamp else [clamp_low, clamp_high],
            "clamped_points":
                clamped_points,
            "api_calls":
                ledger.api_calls,
            "cache_hits":
                ledger.cache_hits,
            "api_cost_usd":
                round(ledger.cost_usd, 6),
            "api_prompt_tokens":
                ledger.prompt_tokens,
            "api_completion_tokens":
                ledger.completion_tokens,
            "api_reasoning_tokens":
                ledger.reasoning_tokens,
            "protocol_note": (
                "One OpenRouter call per image (two with refinement) returns depths for numbered markers; dense "
                "maps are interpolated from those samples."),
        },
    )
    print(
        f"[{args.method_name}] {len(image_paths)} images, {ledger.api_calls} API calls, {ledger.cache_hits} cached, "
        f"${ledger.cost_usd:.4f}, {parse_failures} points unparsed, {sky_points} sky, "
        f"{followup_calls} follow-ups filled {followup_filled} points, "
        f"{fallback_imputations} frames imputed, {clamped_points} points clamped to [{clamp_low}, {clamp_high}]",
        flush=True,
    )


if __name__ == "__main__":
    run()
