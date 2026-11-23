#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run GenCeption and export benchmark prediction artifacts.

GenCeption (google-deepmind/representations4d) is a single-forward-pass
video foundation model (WAN 2.1 diffusion transformer) steered by a text
prompt. Two tasks are supported here:

- ``--task depth``: uses the officially released "Input: RGB. Output:
  Depth." prompt embedding and ``pipe.decode_depth()`` -- reliable, matches
  the repo's own colab demo exactly.
- ``--task pose``: the official "Input: RGB. Output: Raymap." prompt embedding (``raymap.npy``) and the decode of
  representations4d/genception/evals/eval_camera_pose.py, released with the answer to
  google-deepmind/representations4d#10 (spatialarena/geometry/genception_raymap.py). As that evaluation does,
  frames are resized to cover 480x832 and center-cropped (aspect ratio kept), the unquantized float output is
  decoded when the pipeline keeps it, all 81 output frames are decoded (padding included, so the temporal
  smoothing sees the same window) and only the real frames are kept. The fitted pinhole K is mapped back to the
  original image size.

GenCeption only ever takes RGB video as input (no camera/depth priors), so
this adapter is rgb-track-only, and (unlike most other adapters here)
cannot produce depth and pose from the same forward pass -- each requires
its own prompt/forward pass, so callers must pick one task per invocation
(never registered for MVRecon, which needs both at once).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.inference.adapter_utils import (  # noqa: E402
    canonical_image_root, batched_image_paths, c2w_to_w2c,
    export_camera_pose_priors, export_depth_predictions, find_image_paths,
    mark_model_loaded, select_interval, write_adapter_metrics,
)

METHOD = "GenCeption"

# GenCeption's own VAE/transformer are hardwired to this spatial resolution
# and frame count (representations4d/genception/utils.py::preprocess_video);
# not a tunable knob.
GENCEPTION_HEIGHT = 480
GENCEPTION_WIDTH = 832
GENCEPTION_NUM_FRAMES = 81

DEPTH_PROMPT_FILENAME = "depth.npy"
RAYMAP_PROMPT_FILENAME = "raymap.npy"
POSE_SOLVERS = ("robust2", "svd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GenCeption benchmark adapter.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("depth", "pose"),
        required=True,
        help=
        "depth: official depth prompt and decoder. pose: official raymap prompt and decode.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        default=Path(os.environ["GENCEPTION_CKPT_DIR"])
        if "GENCEPTION_CKPT_DIR" in os.environ else None,
        help=
        "Directory with genception_{model_size}_transformer.npz + vae/ (see scripts/download/genception.sh).",
    )
    parser.add_argument(
        "--prompt_embeddings_dir",
        type=Path,
        default=(Path(os.environ["GENCEPTION_EMBEDDINGS_DIR"])
                 if "GENCEPTION_EMBEDDINGS_DIR" in os.environ else None),
    )
    parser.add_argument("--model_size",
                        type=str,
                        default=os.environ.get("GENCEPTION_MODEL_SIZE", "14b"))
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--colmap_model_subdir", type=str, default="0")
    parser.add_argument(
        "--pose_solver",
        choices=POSE_SOLVERS,
        default=os.environ.get("GENCEPTION_POSE_SOLVER", "robust2"),
        help=
        "Rotation solver of the official raymap decode (both are reported upstream; robust2 first).",
    )
    parser.add_argument(
        "--raw_poses",
        action="store_true",
        default=os.environ.get("GENCEPTION_RAW_POSES", "0") == "1",
        help=
        "Export the per-frame poses instead of the temporally smoothed ones (official headline).",
    )
    parser.add_argument("--dry_run", action="store_true")
    # run_method_case_list.sh appends these shared flags to every adapter command; --task alone decides what
    # GenCeption exports, so they are accepted and ignored.
    for flag in ("--export_colmap", "--export_depth", "--depth_only"):
        parser.add_argument(flag, action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _print_dry_run(args: argparse.Namespace, num_images: int) -> None:
    # GenCeption's CLI shape (a single --task instead of the shared
    # --export_depth/--export_colmap/--depth_only flags) doesn't fit
    # adapter_utils.print_dry_run()'s generic planned_layout(), so this is
    # its own small equivalent rather than forcing an artificial fit.
    plan = {
        "method":
            METHOD,
        "data_path":
            str(args.data_path),
        "output_path":
            str(args.output_path),
        "num_images":
            num_images,
        "task":
            args.task,
        "pose_decode":
            (f"{args.pose_solver}, {'raw' if args.raw_poses else 'smoothed'}"
             if args.task == "pose" else None),
        "model_size":
            args.model_size,
        "ckpt_dir":
            str(args.ckpt_dir) if args.ckpt_dir else None,
        "prompt_embeddings_dir":
            str(args.prompt_embeddings_dir)
            if args.prompt_embeddings_dir else None,
        "metrics":
            str(args.output_path / "metrics.json"),
        "depths":
            str(args.output_path / "depths") if args.task == "depth" else None,
        "sparse":
            (str(args.output_path / "sparse" /
                 args.colmap_model_subdir) if args.task == "pose" else None),
    }
    print(json.dumps(plan, indent=2))


def _load_video(
        image_paths: list[str]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Frames resized on the CPU to GenCeption's input size, and each frame's original (height, width).

    preprocess_video resizes on the GPU in float32 and runs out of memory on full-resolution photos (e.g. MonoDepth
    ETH3D: 81 x 4032 x 6048); frames already at (480, 832) pass through it unchanged. This also allows mixed sizes.
    """
    frames, sizes = [], []
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            sizes.append((image.height, image.width))
            frames.append(
                np.asarray(
                    image.resize((GENCEPTION_WIDTH, GENCEPTION_HEIGHT),
                                 Image.BILINEAR)))
    return np.stack(frames, axis=0).astype(np.uint8), sizes  # [T, 480, 832, 3]


def _cover_crop_frame(
        image: Image.Image
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """The official pose evaluation's resize: scale to cover 480x832 keeping the aspect ratio, then center-crop.

    Returns the (480, 832, 3) uint8 frame and (scale_x, scale_y, offset_x, offset_y): a pixel ``u`` of the original
    lands at ``u * scale_x - offset_x`` of the crop. OpenCV (INTER_AREA down, INTER_LINEAR up) is used when it is
    installed, as upstream; otherwise PIL's box / bilinear filters.
    """
    width, height = image.size
    scale = max(GENCEPTION_WIDTH / width, GENCEPTION_HEIGHT / height)
    new_width, new_height = max(1, int(round(width * scale))), max(
        1, int(round(height * scale)))
    frame = np.asarray(image)
    if (new_width, new_height) != (width, height):
        if importlib.util.find_spec("cv2") is not None:
            import cv2  # pylint: disable=import-outside-toplevel

            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (new_width, new_height),
                               interpolation=interpolation)
        else:
            resample = Image.BOX if scale < 1.0 else Image.BILINEAR
            frame = np.asarray(image.resize((new_width, new_height), resample))
    offset_x, offset_y = (new_width - GENCEPTION_WIDTH) // 2, (
        new_height - GENCEPTION_HEIGHT) // 2
    crop = frame[offset_y:offset_y + GENCEPTION_HEIGHT,
                 offset_x:offset_x + GENCEPTION_WIDTH]
    return np.ascontiguousarray(crop), (new_width / width, new_height / height,
                                        float(offset_x), float(offset_y))


def _load_pose_video(
    image_paths: list[str]
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple]]:
    """Cover-resized, center-cropped (T, 480, 832, 3) frames, each original (height, width) and crop transform."""
    frames, sizes, transforms = [], [], []
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            sizes.append((image.height, image.width))
            frame, transform = _cover_crop_frame(image)
            frames.append(frame)
            transforms.append(transform)
    return np.stack(frames, axis=0).astype(np.uint8), sizes, transforms


def _pad_to_81(video: np.ndarray) -> np.ndarray:
    """Repeat the last frame up to 81 frames, as preprocess_video and the official evaluation do."""
    if len(video) >= GENCEPTION_NUM_FRAMES:
        return video[:GENCEPTION_NUM_FRAMES]
    return np.concatenate([
        video,
        np.repeat(video[-1:], GENCEPTION_NUM_FRAMES - len(video), axis=0)
    ],
                          axis=0)


def _float_output(pipe, result: dict) -> np.ndarray:
    """The decoded video in [0, 1]: the pipeline's unquantized float copy when it keeps one (upstream's
    eval_common.get_last_video_float), else its uint8 output / 255."""
    if hasattr(pipe, "_last_video_float"):
        return np.asarray(pipe._last_video_float, dtype=np.float32)  # pylint: disable=protected-access
    return np.asarray(result["video"], dtype=np.float32) / 255.0


def _intrinsics_on_originals(intrinsics: np.ndarray,
                             transforms: list[tuple]) -> np.ndarray:
    """Per-frame K on each original image from K on the 480x832 crop (``u_crop = u * scale - offset``)."""
    per_frame = np.repeat(intrinsics[None].astype(np.float64),
                          len(transforms),
                          axis=0)
    for index, (scale_x, scale_y, offset_x, offset_y) in enumerate(transforms):
        per_frame[index, 0, 0] /= scale_x
        per_frame[index, 0, 2] = (intrinsics[0, 2] + offset_x) / scale_x
        per_frame[index, 1, 1] /= scale_y
        per_frame[index, 1, 2] = (intrinsics[1, 2] + offset_y) / scale_y
    return per_frame


def _select_input_frame_indices(num_frames: int) -> np.ndarray:
    """The input frames GenCeption predicts: all of them up to 81, else preprocess_video's own temporal sampling
    (``np.linspace(0, T - 1, 81)``). Selecting them here keeps every prediction under its own image name."""
    if num_frames > GENCEPTION_NUM_FRAMES:
        return np.linspace(0, num_frames - 1, GENCEPTION_NUM_FRAMES, dtype=int)
    return np.arange(num_frames)


def _select_output_frame_indices(num_real_frames: int) -> np.ndarray:
    """Which of GenCeption's always-81 output frames correspond to our real
    input frames, replicating representations4d's own preprocess_video
    frame-count handling exactly (see genception/utils.py) so this stays a
    real 1:1 mapping rather than a guess:

    - num_real_frames > 81: preprocess_video linearly subsamples down to 81
      (np.linspace(0, T-1, 81)) -- every output frame maps to a distinct
      real frame, so we return all 81 indices as-is (identity).
    - num_real_frames <= 81: preprocess_video pads by repeating the last
      frame up to 81 -- only the first num_real_frames output frames
      correspond to real input frames; the rest are repeats we discard.
    """
    if num_real_frames > GENCEPTION_NUM_FRAMES:
        return np.arange(GENCEPTION_NUM_FRAMES)
    return np.arange(num_real_frames)


def _depth_pass(pipe, video: np.ndarray, prompt_path: Path,
                sizes: list[tuple[int, int]]) -> list[np.ndarray]:
    """GenCeption's official depth prompt on ``video`` (<= 81 frames): one relative depth map per input frame, at
    that frame's original size."""
    result = pipe(prompt_embeds=str(prompt_path), input_video=video)
    relative_depth = pipe.decode_depth(
        result["video"])[_select_output_frame_indices(
            len(video))]  # [T, 480, 832]
    return [
        _resize_to_original(depth[None], height, width)[0]
        for depth, (height, width) in zip(relative_depth, sizes)
    ]


def _export_clip_depths(pipe, image_paths: list[str], prompt_path: Path,
                        output_path: Path) -> None:
    """The depth task: consecutive clips of at most 81 frames, so every input frame gets its own depth map."""
    for clip_paths in batched_image_paths(image_paths, GENCEPTION_NUM_FRAMES):
        video, sizes = _load_video(clip_paths)
        for image_path, depth in zip(clip_paths,
                                     _depth_pass(pipe, video, prompt_path,
                                                 sizes),
                                     strict=True):
            export_depth_predictions(output_path, [image_path], depth[None])


def _resize_to_original(frames: np.ndarray, target_height: int,
                        target_width: int) -> np.ndarray:
    """GenCeption anisotropically resizes to (480, 832) with no
    aspect-preserving crop (see representations4d/genception/utils.py's
    preprocess_video and google-deepmind/representations4d#10's own note on
    this), so recovering the original resolution is a plain resize back,
    not a crop."""
    resized = np.empty(
        (frames.shape[0], target_height, target_width) + frames.shape[3:],
        dtype=frames.dtype)
    for index, frame in enumerate(frames):
        mode = "F" if frame.dtype == np.float32 else None
        image = Image.fromarray(frame, mode=mode)
        resized[index] = np.asarray(
            image.resize((target_width, target_height), Image.BILINEAR))
    return resized


def run() -> None:
    args = parse_args()
    images_dir = canonical_image_root(args.data_path)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    image_paths = select_interval(find_image_paths(images_dir), args.interval,
                                  args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    if args.dry_run:
        _print_dry_run(args, len(image_paths))
        return

    if args.ckpt_dir is None:
        raise ValueError("--ckpt_dir or GENCEPTION_CKPT_DIR is required")
    if args.prompt_embeddings_dir is None:
        raise ValueError(
            "--prompt_embeddings_dir or GENCEPTION_EMBEDDINGS_DIR is required")

    from representations4d.genception import GenCeptionPipeline

    started_at = time.time()
    pipe = GenCeptionPipeline.from_pretrained(str(args.ckpt_dir),
                                              model_size=args.model_size)
    mark_model_loaded()

    depth_prompt_path = args.prompt_embeddings_dir / DEPTH_PROMPT_FILENAME
    if (args.task == "depth" or
            args.export_depth) and not depth_prompt_path.is_file():
        raise FileNotFoundError(
            f"{depth_prompt_path} not found -- run scripts/download/genception.sh first."
        )

    if args.task == "depth":
        _export_clip_depths(pipe, image_paths, depth_prompt_path,
                            args.output_path)
        write_adapter_metrics(
            output_path=args.output_path,
            method=METHOD,
            image_paths=image_paths,
            started_at=started_at,
            checkpoint=str(args.ckpt_dir),
            depth_only=True,
            metrics_extra={
                "task": "depth",
                "model_size": args.model_size,
                "prompt": "official (Input: RGB. Output: Depth.)",
                "resolution": f"{GENCEPTION_WIDTH}x{GENCEPTION_HEIGHT}",
                "frames_predicted": len(image_paths),
            },
        )
        return

    # --task pose: the official raymap decode (see module docstring). One clip: at most 81 frames get poses.
    selected = _select_input_frame_indices(len(image_paths))
    image_paths = [image_paths[index] for index in selected
                  ]  # the frames GenCeption predicts, by name
    video, sizes, transforms = _load_pose_video(
        image_paths)  # [T <= 81, 480, 832, 3] uint8, cover-cropped
    from spatialarena.geometry.genception_raymap import decode_poses_from_raymap

    prompt_path = args.prompt_embeddings_dir / RAYMAP_PROMPT_FILENAME
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"{prompt_path} not found -- run scripts/download/genception.sh first."
        )
    result = pipe(prompt_embeds=str(prompt_path), input_video=_pad_to_81(video))
    # All 81 output frames are decoded (the clip-wide K fit and the smoothing window see the padding, as upstream);
    # only the real frames are exported.
    decoded = decode_poses_from_raymap(_float_output(pipe, result),
                                       solver=args.pose_solver)
    output_indices = _select_output_frame_indices(len(video))
    camera_to_world = decoded.poses(smoothed=not args.raw_poses)[output_indices]
    extrinsics_w2c = c2w_to_w2c(camera_to_world)
    if len(set(sizes)) == 1:
        height, width = sizes[0]
        intrinsics = _intrinsics_on_originals(decoded.intrinsics,
                                              transforms[:1])
        intrinsics = np.repeat(intrinsics, len(image_paths), axis=0)
    else:  # mixed sizes: one COLMAP camera size only, so keep K on the 480x832 crop
        height, width = GENCEPTION_HEIGHT, GENCEPTION_WIDTH
        intrinsics = np.repeat(decoded.intrinsics[None],
                               len(image_paths),
                               axis=0)

    if args.export_depth:
        # Second pass with the official depth prompt on the same frames (GenCeption predicts one output per prompt):
        # relative depth, not in the pose's scale; no confidence exists.
        depth_video, _ = _load_video(image_paths)
        depths = _depth_pass(pipe, depth_video, depth_prompt_path, sizes)
        for image_path, depth in zip(image_paths, depths, strict=True):
            export_depth_predictions(args.output_path, [image_path],
                                     depth[None])

    export_camera_pose_priors(
        args.output_path,
        images_dir,
        image_paths,
        extrinsics_w2c,
        intrinsics,
        width=width,
        height=height,
        colmap_model_subdir=args.colmap_model_subdir,
    )
    decode_info = {
        "solver":
            decoded.solver,
        "poses":
            "raw" if args.raw_poses else "smoothed",
        "output":
            "float" if hasattr(pipe, "_last_video_float") else "uint8 / 255",
        "preprocess":
            "cover resize + center crop to 832x480",
        "intrinsics_on_crop":
            decoded.intrinsics.tolist(),
        "intrinsics_size": [width, height],
    }
    write_adapter_metrics(
        output_path=args.output_path,
        method=METHOD,
        image_paths=image_paths,
        started_at=started_at,
        checkpoint=str(args.ckpt_dir),
        depth_only=False,
        metrics_extra={
            "task":
                "pose",
            "model_size":
                args.model_size,
            "prompt":
                "official (Input: RGB. Output: Raymap.)",
            "decoder":
                "representations4d genception/evals/eval_camera_pose.py",
            "resolution":
                f"{GENCEPTION_WIDTH}x{GENCEPTION_HEIGHT}",
            "frames_predicted":
                len(image_paths),
            "depth_pass":
                "official depth prompt" if args.export_depth else None,
            "pose_decode":
                decode_info,
        },
    )
    np.savez_compressed(
        args.output_path / "genception_pose_decode.npz",
        camera_to_world=decoded.camera_to_world[output_indices],
        camera_to_world_raw=decoded.camera_to_world_raw[output_indices],
        intrinsics_on_crop=decoded.intrinsics,
    )
    (args.output_path / "genception_pose_decode_diagnostics.json").write_text(
        json.dumps(decode_info, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
