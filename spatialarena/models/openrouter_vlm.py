# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Commercial VLMs queried through OpenRouter for MonoDepth.

API VLMs return depth as text, like the local point-query models
(`spatialarena.models.point_query`), but each request re-sends the whole
image, so one call per queried pixel would multiply the cost by the grid
size. Instead an N x N grid of numbered markers is drawn on the image and one
call asks for every marker's metric depth as a JSON object; the samples are
interpolated into the dense `depths/<stem>.npz` maps.

Responses are cached on disk, keyed by model, prompt version, query settings
and image bytes, so a rerun does not pay for the same request again. Every
call's token usage and USD cost (as OpenRouter reports them) go into
metrics.json, and --max_cost_usd stops a run before it exceeds its budget.

The API key is read from OPENROUTER_API_KEY or from the file named by
OPENROUTER_API_KEY_FILE; it is never written to any output.

scripts/inference/openrouter_vlm_depth.py is an alias of this module, so the
module-level helpers (`post_json`, ...) are the ones the models call and can
be replaced there.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import re
import threading
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont
from PIL import ImageOps

from spatialarena.models import base
from spatialarena.models import point_query
from spatialarena.models import registry
from spatialarena.models import utils

# Names the pre-refactor script took from point_query_depth_adapter.
grid_points = point_query.grid_points
interpolate_sparse_depth = point_query.interpolate_sparse_depth
load_rgb_image = point_query.load_rgb_image

# Method name -> default OpenRouter model slug (the registry's ckpt_default;
# overridable per method via <METHOD>_CKPT_PATH like any other checkpoint).
OPENROUTER_MODELS: dict[str, str] = {
    "CLAUDE_OPUS_5_5": "anthropic/claude-opus-5.5",
    "GPT_6_ASTRA": "openai/gpt-6-astra",
    "GEMINI_4_ARGON": "google/gemini-4-argon",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Bump when the prompt or marker rendering changes, so that cached answers to
# the old query are not reused.
PROMPT_VERSIONS = ("grid-v1", "grid-v2")
INTERPOLATIONS = ("bilinear", "joint_bilateral")
SYSTEM_PROMPTS = {
    "none":
        None,
    "depth-expert":
        ("You are an expert in monocular metric depth estimation. Given a "
         "single photo, you estimate the depth of marked points: the distance "
         "in meters from the camera along its optical axis (z-depth, not "
         "straight-line distance). Use cues such as object sizes, perspective, "
         "occlusion and the ground plane. Always answer in the exact JSON "
         "format requested."),
}
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>",
                             re.DOTALL | re.IGNORECASE)
PAIR_RE = re.compile(r"\"?(\d+)\"?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

# The --help description (the pre-refactor adapter's module docstring).
DESCRIPTION = """Query commercial VLMs through OpenRouter for MonoDepth and
export dense benchmark depth maps.

API VLMs return depth as text, like the local point-query baselines
(``point_query_depth_adapter.py``), but each request re-sends the whole image,
so one call per queried pixel would multiply the cost by the grid size. This
adapter instead draws an ``N x N`` grid of numbered markers on the image, asks
for every marker's metric depth in a single call as a JSON object, and
interpolates those samples into the dense ``depths/<stem>.npz`` layout with
the same ``interpolate_sparse_depth`` bridge the point-query baselines use.

Responses are cached on disk keyed by model, prompt version, query settings and
image bytes, so a rerun of a case does not pay for the same request again.
Every call's token usage and USD cost (as reported by OpenRouter) are recorded
in ``metrics.json``, and ``--max_cost_usd`` stops a run before it exceeds its
budget.

The API key is read from ``OPENROUTER_API_KEY`` or from the file named by
``OPENROUTER_API_KEY_FILE``; it is never written to any output.
"""

ANSWER_FORMAT_V2 = (
    "Return the final answer as a single JSON object mapping each dot number "
    "(as a string) to its depth in meters (or \"sky\"), wrapped in <answer> "
    "</answer> tags, e.g. <answer>{\"1\": 2.4, \"2\": \"sky\", \"3\": 3.1, "
    "...}</answer>.")
ANSWER_FORMAT_V1 = (
    "Return the final answer as a single JSON object mapping each dot number "
    "(as a string) to its depth in meters, wrapped in <answer> </answer> "
    "tags, e.g. <answer>{\"1\": 2.4, \"2\": 3.1, ...}</answer>.")
REFINE_ANSWER_FORMAT_V1 = (
    "Return the final answer as a single JSON object mapping each new dot "
    "number (as a string) to its depth in meters, wrapped in <answer> "
    "</answer> tags, e.g. <answer>{\"1\": 2.4, \"2\": 3.1, ...}</answer>.")


def read_api_key() -> str:
    """Returns the OpenRouter API key.

    From OPENROUTER_API_KEY, else from the file OPENROUTER_API_KEY_FILE names
    (a bare key or a dotenv-style OPENROUTER_API_KEY=... line).

    Raises:
        RuntimeError: Neither is set.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key and os.environ.get("OPENROUTER_API_KEY_FILE"):
        key_file = pathlib.Path(
            os.environ["OPENROUTER_API_KEY_FILE"]).expanduser()
        text = key_file.read_text(encoding="utf-8").strip()
        match = re.search(r"OPENROUTER_API_KEY\s*=\s*['\"]?([^'\"\s]+)", text)
        key = match.group(1) if match else text.splitlines()[0].strip()
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY or OPENROUTER_API_KEY_FILE "
                           "to query OpenRouter.")
    return key


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font.
        return ImageFont.load_default()


def draw_numbered_markers(image: Image.Image,
                          points: list[tuple[int, int]]) -> Image.Image:
    """Marks each point with a red dot and its 1-based number on a white tag.

    Args:
        image: The image.
        points: (x, y) query pixels.

    Returns:
        A marked copy of the image.
    """
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


def build_prompt(num_points: int, version: str = "grid-v1") -> str:
    """Returns the first-pass question.

    Args:
        num_points: Number of markers.
        version: Prompt version; "grid-v2" sends the clean photo before the
            marked one and allows "sky".

    Returns:
        The prompt.
    """
    if version == "grid-v2":
        return (
            f"The first image is a photo. The second image is the same photo "
            f"with {num_points} red dots, each labelled with a number from 1 "
            f"to {num_points}; use the first image to see the scene the "
            "markers cover. For every dot, estimate the metric depth of the "
            "scene point under the dot centre: its distance from the camera "
            "along the camera's viewing (optical) axis, in meters. If a dot "
            "lies on the sky or on anything effectively infinitely far away, "
            "answer \"sky\" instead of a number. Give a numeric estimate for "
            "every other dot even when unsure.\n" + ANSWER_FORMAT_V2)
    return (
        f"This image has {num_points} red dots, each labelled with a "
        f"number from 1 to {num_points}. For every dot, estimate the "
        "metric depth of the scene point under the dot centre: its "
        "distance from the camera along the camera's viewing (optical) "
        "axis, in meters. Judge the scene behind the marker, not the "
        "marker itself. Give an estimate for every dot even when unsure.\n" +
        ANSWER_FORMAT_V1)


def build_refine_prompt(num_points: int, version: str = "grid-v1") -> str:
    """Returns the second-pass question.

    It asks for the depths of extra dots on both sides of the depth edges
    the first pass found.

    Args:
        num_points: Number of new markers.
        version: Prompt version.

    Returns:
        The prompt.
    """
    if version == "grid-v2":
        images = ("The first image is the same photo again and the second "
                  "shows it")
        sky = " Answer \"sky\" for dots on the sky."
        answer = ANSWER_FORMAT_V2
    else:
        images = "This image shows the same photo"
        sky = ""
        answer = REFINE_ANSWER_FORMAT_V1
    return (f"{images} with {num_points} new red dots labelled 1 to "
            f"{num_points}, placed on either side of depth edges. Estimate "
            "the metric depth of the scene point under each new dot centre in "
            "the same way (distance along the optical axis, in meters), "
            "consistent with your previous estimates; dots on different sides "
            f"of an edge may lie on different surfaces.{sky} Give a numeric "
            "estimate for every other dot even when unsure.\n" + answer)


def encode_image(image: Image.Image) -> str:
    """Returns the image as a JPEG data URL.

    Args:
        image: The image.

    Returns:
        The data URL (JPEG quality 90).
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()).decode("ascii")


def user_turn(image_urls: list[str] | str, prompt: str) -> dict[str, Any]:
    """Returns a user turn: the images, then the prompt.

    Args:
        image_urls: One image URL or several.
        prompt: The text.

    Returns:
        The chat message.
    """
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
    """Returns the --system_prompt turn, or no turn for "none".

    Args:
        args: The parsed options.

    Returns:
        A list of zero or one chat messages.
    """
    text = SYSTEM_PROMPTS[getattr(args, "system_prompt", "none")]
    return [{"role": "system", "content": text}] if text else []


def build_payload(
    model: str,
    image_urls: list[str] | str,
    prompt: str,
    args: argparse.Namespace,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Returns the chat-completions request body.

    Args:
        model: OpenRouter model slug.
        image_urls: Images of the new user turn.
        prompt: Text of the new user turn.
        args: The parsed options.
        history: Earlier turns of the conversation.

    Returns:
        The request body.
    """
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


def _answer_body(text: str) -> str:
    """Returns the last <answer> block of a response, or the whole text."""
    blocks = ANSWER_BLOCK_RE.findall(text or "")
    return blocks[-1] if blocks else (text or "")


def sky_indices(text: str) -> set[int]:
    """Returns the dot numbers answered "sky" (grid-v2 prompts).

    Args:
        text: The response.

    Returns:
        The dot numbers.
    """
    return {
        int(number) for number in re.findall(r"\"?(\d+)\"?\s*:\s*\"sky\"",
                                             _answer_body(text),
                                             flags=re.IGNORECASE)
    }


def format_dot_ranges(numbers: list[int]) -> str:
    """Formats dot numbers as ranges: [1, 2, 3, 7, 9, 10] -> "1-3, 7, 9-10".

    Args:
        numbers: Dot numbers, in any order.

    Returns:
        The ranges, comma-separated.
    """
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
    """Returns the question for exactly the dots an answer left out.

    Args:
        missing: The dot numbers left out.
        version: Prompt version.

    Returns:
        The prompt (no new images are sent).
    """
    sky = " (or \"sky\")" if version == "grid-v2" else ""
    return (f"Your answer did not include dots {format_dot_ranges(missing)}. "
            f"Give the metric depth{sky} for exactly these dots, estimated the "
            "same way and consistent with your previous estimates. Return one "
            "JSON object mapping each of these dot numbers (as a string) to "
            "its depth, wrapped in <answer> </answer> tags.")


def count_sky_answers(text: str) -> int:
    """Returns how many answer entries are the string "sky" (grid-v2).

    Args:
        text: The response.

    Returns:
        The count.
    """
    return len(
        re.findall(r":\s*\"sky\"", _answer_body(text), flags=re.IGNORECASE))


def _parse_json_span(body: str, opening: str, closing: str) -> Any:
    """Parses body[first opening:last closing] as JSON, or returns None."""
    start, end = body.find(opening), body.rfind(closing)
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None


def parse_grid_answer(text: str, num_points: int) -> list[float]:
    """Returns the depths in marker order from a JSON object (or list) answer.

    Args:
        text: The response.
        num_points: Number of markers.

    Returns:
        One depth per marker, NaN where missing, invalid or not positive.
    """
    body = _answer_body(text)
    values: dict[int, Any] = {}
    parsed = _parse_json_span(body, "{", "}")
    if parsed is None:
        parsed = _parse_json_span(body, "[", "]")
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if str(key).strip().isdigit():
                values[int(str(key).strip())] = value
    elif isinstance(parsed, list):
        values = dict(enumerate(parsed, start=1))
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


def scene_depth_limits(
        data_path: pathlib.Path) -> tuple[float | None, float | None]:
    """Returns the scene's evaluation depth range from its metadata.json.

    The scene folder's own metadata.json is read first, then the dataset's
    (whose "scenes" list has one entry per scene).

    Args:
        data_path: The scene folder.

    Returns:
        (min_depth, max_depth); None where a bound is not declared.
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
    """Clamps each sampled depth into [low, high] before interpolation.

    Answers far outside the evaluated range (e.g. thousands of meters for
    sky) would otherwise bleed through the interpolation into neighbouring
    in-range pixels; the evaluator's post-alignment clip cannot undo that.

    Args:
        samples: (x, y, depth) samples.
        low: Lower bound, or None.
        high: Upper bound, or None.

    Returns:
        The clamped samples and how many of them changed.
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
    """Densifies scattered depth samples with joint bilateral upsampling.

    Guided by the RGB image (Kopf et al. 2007): each pixel takes a weighted
    mean of the sampled log depths, weighted by spatial distance (sigma in
    grid cells) and by the colour difference between the pixel and the mean
    colour around each sample, so that depth follows the image's edges
    instead of being blended across them. Every value comes from the model's
    own answers. A small spatial-only term keeps pixels unlike every sample
    well defined. Works for irregular sample sets (refinement points), not
    only full grids.

    Args:
        samples: (x, y, depth) samples.
        image: The guide image.
        grid_size: Grid points per axis (sets the spatial scale).
        spatial_sigma_cells: Spatial sigma, in grid cells.
        range_sigma: Colour sigma (RGB in [0, 1]).
        patch_radius: Radius of the colour patch around each sample.

    Returns:
        A float32 depth map of the image's size.

    Raises:
        RuntimeError: No sample is valid.
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
    """Fills NaN grid cells ("sky" or unparsed) from their neighbours.

    Each pass gives every missing cell with at least one finite 8-neighbour
    the mean log depth of those neighbours, so gaps fill from the nearest
    answers inward (one scene-median value per missing cell, as the local
    point-query bridge does, turns every gap into a flat block). Answered
    cells never change.

    Args:
        grid: (rows, cols) depths, NaN or <= 0 where missing.

    Returns:
        The filled grid.

    Raises:
        RuntimeError: No cell is valid.
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
    """Returns the bilinear dense map of row-major grid samples.

    Missing cells are filled from their neighbours (`fill_missing_grid`).

    Args:
        samples: (x, y, depth) samples, the row-major grid first.
        width: Output width.
        height: Output height.
        grid_size: Grid points per axis.

    Returns:
        The float32 depth map.
    """
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
    """Returns the dense depth on the resized image grid.

    Args:
        samples: (x, y, depth) samples, the grid first.
        image: The resized image.
        grid_size: Grid points per axis.
        method: "bilinear" or "joint_bilateral".

    Returns:
        The depth map.

    Raises:
        ValueError: Unknown method.
    """
    if method == "bilinear":
        return interpolate_grid_depth(samples, image.width, image.height,
                                      grid_size)
    if method == "joint_bilateral":
        return joint_bilateral_depth(samples, image, grid_size)
    raise ValueError(f"Unknown densification method {method!r}")


class BudgetExceeded(RuntimeError):
    """The run's billed cost reached --max_cost_usd."""


@dataclasses.dataclass
class UsageLedger:
    """Token usage and cost of a run's requests (thread-safe).

    Attributes:
        max_cost_usd: Budget (0: none).
        cost_usd: Billed cost so far.
        prompt_tokens: Prompt tokens so far.
        completion_tokens: Completion tokens so far.
        reasoning_tokens: Reasoning tokens so far.
        api_calls: Requests sent.
        cache_hits: Requests answered from the cache.
        lock: Guards the counters.
    """

    max_cost_usd: float
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def check(self) -> None:
        """Raises BudgetExceeded once the cost reached the budget."""
        with self.lock:
            if self.max_cost_usd > 0 and self.cost_usd >= self.max_cost_usd:
                raise BudgetExceeded(
                    f"OpenRouter budget reached: ${self.cost_usd:.4f} >= "
                    f"${self.max_cost_usd:.4f}")

    def add(self, usage: dict[str, Any] | None, cached: bool) -> None:
        """Counts one request (a cached one only as a cache hit).

        Args:
            usage: The response's usage, or None.
            cached: Whether the answer came from the cache.
        """
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
    """POSTs one request to OpenRouter.

    Args:
        payload: The request body.
        api_key: The API key.
        timeout: Timeout in seconds.

    Returns:
        The decoded response body.
    """
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


def _check_error_body(body: dict[str, Any]) -> None:
    """Raises on an error response without choices.

    Raises:
        RuntimeError: A non-retryable error.
        ConnectionError: A retryable one.
    """
    if "error" not in body or body.get("choices"):
        return
    error = body["error"]
    code = 0
    if str(error.get("code") or "").isdigit():
        code = int(error.get("code") or 0)
    if code and code not in RETRYABLE_STATUS:
        raise RuntimeError(f"OpenRouter error {code}: {error.get('message')}")
    raise ConnectionError(f"OpenRouter error {code}: {error.get('message')}")


def call_openrouter(payload: dict[str, Any], api_key: str,
                    args: argparse.Namespace) -> dict[str, Any]:
    """POSTs a request, retrying with exponential backoff.

    Rate limits, transient server errors and network failures are retried.

    Args:
        payload: The request body.
        api_key: The API key.
        args: The parsed options (max_retries, timeout).

    Returns:
        The response body.

    Raises:
        RuntimeError: A non-retryable error, or every attempt failed.
    """
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            body = post_json(payload, api_key, args.timeout)
            _check_error_body(body)
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
    raise RuntimeError(f"OpenRouter request failed after "
                       f"{args.max_retries + 1} attempts: {last_error}")


def refinement_points(
    samples: list[tuple[int, int, float]],
    image: Image.Image,
    grid_size: int,
    max_points: int = 64,
    ratio: float = 1.3,
) -> list[tuple[int, int]]:
    """Returns extra query pixels on both sides of the first pass's edges.

    For every pair of 4-neighbouring grid samples whose depths differ by
    more than `ratio`, the strongest image edge on the segment between them
    is located and one point is placed just before and one just after it.
    Pairs are taken in order of decreasing depth ratio until `max_points`
    points are chosen; points too close to a grid sample or to an already
    chosen point are skipped.

    Args:
        samples: (x, y, depth) samples, the row-major grid first.
        image: The resized image.
        grid_size: Grid points per axis.
        max_points: Most points returned.
        ratio: Depth ratio of a pair that marks an edge.

    Returns:
        (x, y) pixels.
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
    pairs = _edge_pairs(grid, grid_size, ratio)
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


def _edge_pairs(grid, grid_size, ratio):
    """Returns 4-neighbour grid pairs whose depth ratio exceeds `ratio`.

    Sorted by decreasing ratio, as (ratio, (x0, y0), (x1, y1)).
    """
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
    return pairs


def cache_key(model: str,
              args: argparse.Namespace,
              image_bytes_sha1: str,
              extra: Any = None) -> str:
    """Returns the cache key of a request.

    Args:
        model: OpenRouter model slug.
        args: The parsed options.
        image_bytes_sha1: SHA-1 of the source image file.
        extra: Further request state (follow-up or refinement pass).

    Returns:
        A SHA-1 hex digest.
    """
    settings = {
        "model": model,
        "prompt_version": getattr(args, "prompt_version", "grid-v1"),
        "grid": args.point_grid_size,
        "image_load_resolution": args.image_load_resolution,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "image_sha1": image_bytes_sha1,
    }
    # Absent for "none", so that older cache entries stay valid.
    if getattr(args, "system_prompt", "none") != "none":
        settings["system_prompt"] = args.system_prompt
    if extra is not None:
        # The follow-up or refinement pass: its points and what it follows.
        settings["refine"] = extra
    return hashlib.sha1(json.dumps(settings,
                                   sort_keys=True).encode("utf-8")).hexdigest()


def _write_cache(cache_path: pathlib.Path, text: str, usage: Any,
                 generation_id: Any) -> None:
    """Writes one response to the cache atomically."""
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


def cached_ask(
    cache_path: pathlib.Path,
    build: Any,
    num_points: int,
    api_key: str,
    args: argparse.Namespace,
    ledger: UsageLedger,
) -> tuple[str, Any, Any, bool]:
    """Sends one request, or answers it from the cache.

    Args:
        cache_path: The request's cache file.
        build: Returns the request body; called only on a cache miss.
        num_points: Markers the answer covers.
        api_key: The API key.
        args: The parsed options.
        ledger: The run's usage ledger.

    Returns:
        (answer text, usage, generation id, whether it came from the cache).
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
    # Only answers that parsed are cached, so that a malformed response is
    # retried on the next run instead of replayed.
    if any(np.isfinite(parse_grid_answer(
            text, num_points))) or count_sky_answers(text):
        _write_cache(cache_path, text, usage, generation_id)
    return text, usage, generation_id, False


def _image_record(image_path, model, pass_index, answer, points, depths):
    """Returns the openrouter_outputs.jsonl record of one pass's answer."""
    text, usage, generation_id, cached = answer
    return {
        "image": image_path,
        "model": model,
        "pass": pass_index,
        "generation_id": generation_id,
        "cached": cached,
        "parsed_points": int(np.isfinite(depths).sum()),
        "sky_points": None,
        "points": points,
        "depths": [None if not np.isfinite(d) else d for d in depths],
        "usage": usage,
        "text": text,
    }


class _ImageQuery:
    """The requests of one image: first pass, follow-ups and refinement."""

    def __init__(self, image_path, model, api_key, args, ledger):
        self.image_path = image_path
        self.model = model
        self.api_key = api_key
        self.args = args
        self.ledger = ledger
        self.version = getattr(args, "prompt_version", "grid-v1")
        with Image.open(image_path) as probe:
            self.source_wh = ImageOps.exif_transpose(probe).size
        self.image = point_query.load_rgb_image(image_path,
                                                args.image_load_resolution)
        self.points = point_query.grid_points(self.image.width,
                                              self.image.height,
                                              args.point_grid_size)
        self.image_sha1 = hashlib.sha1(
            pathlib.Path(image_path).read_bytes()).hexdigest()
        self.model_dir = args.cache_dir / model.replace("/", "__")
        self.prompt = build_prompt(len(self.points), self.version)

    def ask(self, key, build, num_points):
        """Sends (or replays) one request; see `cached_ask`.

        Args:
            key: The request's cache key.
            build: Returns the request body.
            num_points: Markers the answer covers.

        Returns:
            (answer text, usage, generation id, whether it was cached).
        """
        return cached_ask(self.model_dir / f"{key}.json", build, num_points,
                          self.api_key, self.args, self.ledger)

    def first_images(self) -> list[str]:
        """Returns the image URLs of the first pass."""
        marked = encode_image(draw_numbered_markers(self.image, self.points))
        if self.version == "grid-v2":
            return [encode_image(self.image), marked]
        return [marked]

    def first_turn(self) -> dict[str, Any]:
        """Returns the first pass's user turn."""
        return user_turn(self.first_images(), self.prompt)

    def first_pass(self):
        """Returns the first pass's (text, usage, id, cached) answer."""
        return self.ask(
            cache_key(self.model, self.args, self.image_sha1),
            lambda: build_payload(self.model, self.first_images(), self.prompt,
                                  self.args), len(self.points))

    def followups(self, text, depths, sky):
        """Asks, in the same conversation, for the dots an answer left out.

        An answer that stops partway (e.g. "only dots 1-64 are covered") is
        asked for the rest, not re-sampled.

        Args:
            text: The first-pass answer.
            depths: The parsed depths; filled in place.
            sky: The dots answered "sky"; extended in place.

        Returns:
            One record per follow-up round.
        """
        followups: list[dict[str, Any]] = []
        conversation_texts = [text]
        rounds = int(getattr(self.args, "completion_followups", 0) or 0)
        for round_index in range(1, rounds + 1):
            missing = [
                i + 1
                for i, d in enumerate(depths)
                if not np.isfinite(d) and (i + 1) not in sky
            ]
            if not missing:
                break
            follow_prompt = build_followup_prompt(missing, self.version)
            build = self._followup_builder(tuple(conversation_texts),
                                           follow_prompt, tuple(followups))
            conversation_sha1 = hashlib.sha1(
                "\x00".join(conversation_texts).encode("utf-8")).hexdigest()
            follow_key = cache_key(self.model,
                                   self.args,
                                   self.image_sha1,
                                   extra={
                                       "followup": round_index,
                                       "missing": missing,
                                       "conversation": conversation_sha1
                                   })
            follow_text, follow_usage, follow_id, follow_cached = self.ask(
                follow_key, build, len(self.points))
            follow_depths = parse_grid_answer(follow_text, len(self.points))
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
                break  # The model will not complete it; stop asking.
        return followups

    def _followup_builder(self, texts, follow_prompt, asked):
        """Returns the payload builder of one follow-up round."""

        def build() -> dict[str, Any]:
            turns = [
                self.first_turn(), {
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
            return build_payload(self.model, [],
                                 follow_prompt,
                                 self.args,
                                 history=turns)

        return build

    def refinement(self, samples, first_text):
        """Asks for the depths of extra points around the depth edges.

        Args:
            samples: The first pass's (x, y, depth) samples.
            first_text: The first-pass answer.

        Returns:
            (extra samples, the pass's record), or ([], None) without
            refinement points.
        """
        refine_budget = int(getattr(self.args, "refine_points", 0) or 0)
        if not refine_budget:
            return [], None
        extra = refinement_points(
            samples, self.image, self.args.point_grid_size, refine_budget,
            float(getattr(self.args, "refine_ratio", 1.3)))
        if not extra:
            return [], None

        def second_payload() -> dict[str, Any]:
            marked = encode_image(draw_numbered_markers(self.image, extra))
            if self.version == "grid-v2":
                urls = [encode_image(self.image), marked]
            else:
                urls = [marked]
            history = [
                self.first_turn(), {
                    "role": "assistant",
                    "content": first_text
                }
            ]
            return build_payload(self.model,
                                 urls,
                                 build_refine_prompt(len(extra), self.version),
                                 self.args,
                                 history=history)

        second_key = cache_key(
            self.model,
            self.args,
            self.image_sha1,
            extra={
                "points": extra,
                "first": hashlib.sha1(first_text.encode("utf-8")).hexdigest()
            })
        answer = self.ask(second_key, second_payload, len(extra))
        depths2 = parse_grid_answer(answer[0], len(extra))
        record = _image_record(self.image_path, self.model, 2, answer, extra,
                               depths2)
        record["sky_points"] = count_sky_answers(answer[0])
        samples2 = [(x, y, d) for (x, y), d in zip(extra, depths2, strict=True)]
        return samples2, record


def query_image(
    image_path: str,
    model: str,
    api_key: str,
    args: argparse.Namespace,
    ledger: UsageLedger,
) -> dict[str, Any]:
    """Queries one image (with refinement, in two passes).

    Args:
        image_path: The source image.
        model: OpenRouter model slug.
        api_key: The API key.
        args: The parsed options.
        ledger: The run's usage ledger.

    Returns:
        A dict with the image path, the resized image, its transform, the
        (x, y, depth) samples and the jsonl record.
    """
    query = _ImageQuery(image_path, model, api_key, args, ledger)
    answer = query.first_pass()
    text = answer[0]
    depths = parse_grid_answer(text, len(query.points))
    sky = sky_indices(text)
    followups = query.followups(text, depths, sky)
    samples = [(x, y, d) for (x, y), d in zip(query.points, depths, strict=True)
              ]
    record = _image_record(image_path, model, 1, answer, query.points, depths)
    record["sky_points"] = len(sky)
    record["followups"] = followups

    samples2, refinement = query.refinement(samples, text)
    samples += samples2
    if refinement is not None:
        record["refinement"] = refinement
    transform = utils.SpatialTransform(source_wh=query.source_wh,
                                       resized_wh=(query.image.width,
                                                   query.image.height))
    return {
        "image_path": image_path,
        "image": query.image,
        "transform": transform,
        "samples": samples,
        "record": record,
    }


class OpenRouterVLM(base.BaseModel):
    """A VLM queried through OpenRouter; each subclass is one method.

    --ckpt_path is the OpenRouter model slug. The image is fit into
    --image_load_resolution (its long side) only (no --resolution), and only
    depth is exported (no point options). The query options default to
    OPENROUTER_* environment variables.
    """

    description = DESCRIPTION
    uses_resolution = False
    image_load_resolution_help = "Long side of the image sent."
    exports_points = False
    uses_code_path = False
    default_image_load_resolution = 1024

    @classmethod
    def build_parser(cls, method=None):
        """Returns the common parser, keeping the description's layout.

        Args:
            method: The method the parser is for; None for the model's own.
        """
        parser = super().build_parser(method)
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        return parser

    @classmethod
    def add_arguments(cls, parser):
        """Adds --image_load_resolution and the query options.

        Args:
            parser: The model's parser.
        """
        _add_query_arguments(parser)

    def __init__(self, args):
        super().__init__(args)
        self.api_key = None
        self.ledger = None

    @property
    def checkpoint(self):
        """The OpenRouter model slug."""
        return self.args.ckpt_path or OPENROUTER_MODELS[self.method_name]

    def validate_args(self):
        """Rejects inconsistent options.

        Raises:
            ValueError: COLMAP export outside a depth-only run, no
                --export_depth (outside a dry run), a grid below 2, or
                refinement points with bilinear interpolation.
        """
        args = self.args
        if args.export_colmap and not args.depth_only:
            raise ValueError(f"{self.method_name} OpenRouter adapter only "
                             "supports depth tracks.")
        if not args.export_depth and not args.dry_run:
            raise ValueError(
                f"{self.method_name} adapter requires --export_depth.")
        if args.point_grid_size < 2:
            raise ValueError("--point_grid_size must be at least 2 (1 gives "
                             "a constant depth map).")
        if args.refine_points and args.interpolation == "bilinear":
            raise ValueError("--refine_points needs --interpolation "
                             "joint_bilateral (refinement points are "
                             "off-grid).")

    def load(self, device):
        """Reads the API key.

        Args:
            device: Unused; nothing runs locally.

        Returns:
            The OpenRouter model slug.
        """
        del device
        self.api_key = read_api_key()
        return self.checkpoint

    def preprocess(self, image_paths, batch_index=None):
        return base.ModelInput(image_paths, batch_index=batch_index)

    def forward(self, inputs):
        """Queries the images concurrently (--concurrency).

        Args:
            inputs: The images of the scene.

        Returns:
            The `Prediction`, with the `query_image` results in image order
            in `extra["results"]`.
        """
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, self.args.concurrency)) as pool:
            results = list(
                pool.map(
                    lambda path: query_image(path, self.model, self.api_key,
                                             self.args, self.ledger),
                    inputs.image_paths))
        return base.Prediction(extra={"results": results})

    def run(self):
        """Runs the adapter on the scene (always depth-only).

        Raises:
            ValueError: The options conflict or no images were found.
            BudgetExceeded: The run reached --max_cost_usd.
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

        self.model = self.load(self.device)
        self.started_at = utils.mark_model_loaded()
        self.ledger = UsageLedger(max_cost_usd=args.max_cost_usd)
        self.run_depth_only()

    def run_depth_only(self):
        """Queries every image, writes depths, answers and metrics.json."""
        args = self.args
        results = self.forward(self.preprocess(
            self.image_paths)).extra["results"]
        if args.no_depth_clamp:
            clamp_low, clamp_high = None, None
        else:
            clamp_low, clamp_high = scene_depth_limits(args.data_path)
        clamped_points = 0
        for result in results:
            result["samples"], moved = clamp_samples(result["samples"],
                                                     clamp_low, clamp_high)
            result["record"]["clamped_points"] = moved
            # Sky answers are excluded on purpose: neither failures nor
            # samples.
            result[
                "sky"] = result["record"]["sky_points"] + result["record"].get(
                    "refinement", {}).get("sky_points", 0)
            clamped_points += moved
        stats = _densify_results(results, args)

        utils.export_depth_predictions(args.output_path, self.image_paths,
                                       np.stack(stats.pop("depths"), axis=0))
        args.output_path.mkdir(parents=True, exist_ok=True)
        (args.output_path / "openrouter_outputs.jsonl").write_text(
            "\n".join(
                json.dumps(result["record"], ensure_ascii=True)
                for result in results) + "\n",
            encoding="utf-8",
        )
        stats["clamped_points"] = clamped_points
        stats["followup_calls"] = sum(
            len(result["record"].get("followups", [])) for result in results)
        stats["followup_filled"] = sum(
            followup["filled"]
            for result in results
            for followup in result["record"].get("followups", []))
        stats["total_point_queries"] = sum(
            len(result["samples"]) for result in results)
        clamp = None if args.no_depth_clamp else [clamp_low, clamp_high]
        self.write_metrics(depth_only=True,
                           metrics_extra=self._metrics_fields(stats, clamp))
        ledger = self.ledger
        print(
            f"[{self.method_name}] {len(self.image_paths)} images, "
            f"{ledger.api_calls} API calls, {ledger.cache_hits} cached, "
            f"${ledger.cost_usd:.4f}, {stats['parse_failures']} points "
            f"unparsed, {stats['sky_points']} sky, "
            f"{stats['followup_calls']} follow-ups filled "
            f"{stats['followup_filled']} points, "
            f"{stats['fallback_imputations']} frames imputed, "
            f"{clamped_points} points clamped to [{clamp_low}, {clamp_high}]",
            flush=True,
        )

    def _metrics_fields(self, stats, clamp):
        """Returns the adapter's metrics.json fields."""
        args = self.args
        ledger = self.ledger
        return {
            "track":
                args.track,
            "adapter":
                "openrouter_vlm_depth",
            "openrouter_model":
                self.model,
            "prompt_version":
                args.prompt_version,
            "system_prompt":
                args.system_prompt,
            "interpolation":
                args.interpolation,
            "refine_points":
                args.refine_points,
            "sky_points":
                stats["sky_points"],
            "completion_followups":
                args.completion_followups,
            "followup_calls":
                stats["followup_calls"],
            "followup_filled_points":
                stats["followup_filled"],
            "point_grid_size":
                args.point_grid_size,
            "image_long_side":
                args.image_load_resolution,
            "reasoning_effort":
                args.reasoning_effort,
            "total_point_queries":
                stats["total_point_queries"],
            "parse_failures":
                stats["parse_failures"],
            "fallback_imputations":
                stats["fallback_imputations"],
            "depth_clamp":
                clamp,
            "clamped_points":
                stats["clamped_points"],
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
            "protocol_note":
                ("One OpenRouter call per image (two with refinement) returns "
                 "depths for numbered markers; dense maps are interpolated "
                 "from those samples."),
        }


def _add_query_arguments(parser):
    """Adds the OpenRouter query options (OPENROUTER_* defaults)."""
    env = os.environ
    parser.add_argument("--point_grid_size",
                        type=int,
                        default=int(env.get("OPENROUTER_DEPTH_GRID_SIZE",
                                            "16")))
    parser.add_argument("--concurrency",
                        type=int,
                        default=int(env.get("OPENROUTER_CONCURRENCY", "8")))
    parser.add_argument("--max_tokens",
                        type=int,
                        default=int(env.get("OPENROUTER_MAX_TOKENS", "8192")))
    parser.add_argument(
        "--reasoning_effort",
        default=env.get("OPENROUTER_REASONING_EFFORT", "low"),
        help=("OpenRouter unified reasoning effort "
              "(none/minimal/low/medium/high); 'default' sends no setting."),
    )
    parser.add_argument("--max_retries",
                        type=int,
                        default=int(env.get("OPENROUTER_MAX_RETRIES", "6")))
    parser.add_argument("--timeout",
                        type=float,
                        default=float(env.get("OPENROUTER_TIMEOUT", "300")))
    parser.add_argument(
        "--max_cost_usd",
        type=float,
        default=float(env.get("OPENROUTER_MAX_COST_USD", "0")),
        help=("Stop issuing requests once this run's billed cost reaches this "
              "many USD (0 = no cap)."),
    )
    parser.add_argument(
        "--cache_dir",
        type=pathlib.Path,
        default=pathlib.Path(
            env.get("OPENROUTER_CACHE_DIR",
                    "~/.cache/feedforward_benchmark/openrouter")).expanduser(),
    )
    parser.add_argument(
        "--allow_data_collection",
        action="store_true",
        default=env.get("OPENROUTER_ALLOW_DATA_COLLECTION", "0") == "1",
        help=("Allow providers that may store or train on prompts "
              "(OpenRouter data_collection=allow)."),
    )
    parser.add_argument(
        "--prompt_version",
        choices=PROMPT_VERSIONS,
        default=env.get("OPENROUTER_PROMPT_VERSION", "grid-v2"),
        help=("grid-v1: marked image only. grid-v2: clean + marked image, and "
              "dots on the sky may be answered 'sky'."),
    )
    parser.add_argument(
        "--system_prompt",
        choices=tuple(SYSTEM_PROMPTS),
        default=env.get("OPENROUTER_SYSTEM_PROMPT", "none"),
        help=("Optional system turn framing the task (none = a single user "
              "turn, like the local VLM baselines)."),
    )
    parser.add_argument(
        "--interpolation",
        choices=INTERPOLATIONS,
        default=env.get("OPENROUTER_DEPTH_INTERPOLATION", "bilinear"),
        help=("How the sampled depths become a dense map; joint_bilateral "
              "follows the RGB image's edges."),
    )
    parser.add_argument(
        "--refine_points",
        type=int,
        default=int(env.get("OPENROUTER_REFINE_POINTS", "0")),
        help=("Up to this many extra points around depth edges, asked in a "
              "second call (0 = single pass)."),
    )
    parser.add_argument(
        "--completion_followups",
        type=int,
        default=int(env.get("OPENROUTER_COMPLETION_FOLLOWUPS", "2")),
        help="Follow-up rounds asking for dots an answer left out (0 = off).",
    )
    parser.add_argument("--refine_ratio",
                        type=float,
                        default=float(env.get("OPENROUTER_REFINE_RATIO",
                                              "1.3")))
    parser.add_argument(
        "--no_depth_clamp",
        action="store_true",
        default=env.get("OPENROUTER_DEPTH_CLAMP", "1") == "0",
        help=("Do not clamp sampled depths to the scene's metadata.json "
              "[min_depth, max_depth] before interpolation."),
    )


def _densify_results(results, args):
    """Turns each image's samples into its source-size depth map.

    A frame with no parsed answer gets the scene's median depth
    (deterministic, unlike "last valid" under concurrency), or
    OPENROUTER_DEPTH_FALLBACK_METERS when no frame of the scene parsed.

    Returns:
        A dict: "depths" (one map per image), "parse_failures",
        "sky_points" and "fallback_imputations".
    """
    image_medians = [
        float(np.nanmedian([d
                            for _x, _y, d in result["samples"]]))
        for result in results
        if any(np.isfinite(d) for _x, _y, d in result["samples"])
    ]
    if image_medians:
        scene_fallback = float(np.median(image_medians))
    else:
        scene_fallback = float(
            os.environ.get("OPENROUTER_DEPTH_FALLBACK_METERS", "1.0"))
    depths = []
    parse_failures = 0
    sky_points = 0
    fallback_imputations = 0
    for result in results:
        samples = result["samples"]
        missing = sum(1 for _x, _y, d in samples if not np.isfinite(d))
        parse_failures += missing - result["sky"]
        sky_points += result["sky"]
        if missing == len(samples):
            fallback_imputations += 1
            source_w, source_h = result["transform"].source_wh
            depths.append(
                np.full((source_h, source_w), scene_fallback, dtype=np.float32))
            result["record"]["fallback_imputed"] = scene_fallback
        else:
            dense = densify(samples, result["image"], args.point_grid_size,
                            args.interpolation)
            depths.append(utils.restore_prediction(dense, result["transform"]))
    return {
        "depths": depths,
        "parse_failures": parse_failures,
        "sky_points": sky_points,
        "fallback_imputations": fallback_imputations,
    }


@registry.register_model
class ClaudeOpus55(OpenRouterVLM):
    """Claude Opus 5.5 through OpenRouter."""

    name = "CLAUDE_OPUS_5_5"


@registry.register_model
class Gpt6Astra(OpenRouterVLM):
    """GPT-6 Astra through OpenRouter."""

    name = "GPT_6_ASTRA"


@registry.register_model
class Gemini4Argon(OpenRouterVLM):
    """Gemini 4 Argon through OpenRouter."""

    name = "GEMINI_4_ARGON"


MODELS = {cls.name: cls for cls in (ClaudeOpus55, Gpt6Astra, Gemini4Argon)}

# The scripts that serve several methods take --model to choose one.
split_model_option = point_query.split_model_option


def parse_args(argv=None) -> argparse.Namespace:
    """Parses the command line of the model --model names."""
    method, rest = split_model_option(argv, MODELS, DESCRIPTION)
    return MODELS[method].build_parser(method).parse_args(rest)


def run(argv=None) -> None:
    """Runs the OpenRouter VLM --model names."""
    method, rest = split_model_option(argv, MODELS, DESCRIPTION)
    MODELS[method].main(rest, method=method)
