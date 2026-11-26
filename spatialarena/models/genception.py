# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""GenCeption: benchmark prediction artifacts.

GenCeption (google-deepmind/representations4d) is a single-forward-pass
video foundation model (WAN 2.1 diffusion transformer) steered by a text
prompt. Two tasks are supported here:

- GENCEPTION_DEPTH: uses the officially released "Input:
  RGB. Output: Depth." prompt embedding and ``pipe.decode_depth()`` --
  reliable, matches the repo's own colab demo exactly.
- GENCEPTION_POSE: the official "Input: RGB. Output:
  Raymap." prompt embedding (``raymap.npy``) and the decode of
  representations4d/genception/evals/eval_camera_pose.py, released with the
  answer to google-deepmind/representations4d#10
  (spatialarena/geometry/genception_raymap.py). As that evaluation does,
  frames are resized to cover 480x832 and center-cropped (aspect ratio
  kept), the unquantized float output is decoded when the pipeline keeps it,
  all 81 output frames are decoded (padding included, so the temporal
  smoothing sees the same window) and only the real frames are kept. The
  fitted pinhole K is mapped back to the original image size.

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
import pathlib
import time
from typing import Any, Callable, ClassVar

import numpy as np
from PIL import Image

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

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


def load_video(
        image_paths: list[str]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Loads frames resized on the CPU to GenCeption's input size.

    preprocess_video resizes on the GPU in float32 and runs out of memory on
    full-resolution photos (e.g. MonoDepth ETH3D: 81 x 4032 x 6048); frames
    already at (480, 832) pass through it unchanged. This also allows mixed
    sizes.

    Args:
        image_paths: The frames.

    Returns:
        A ([T, 480, 832, 3] uint8 video, per-frame original (height, width))
        tuple.
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
    return np.stack(frames, axis=0).astype(np.uint8), sizes


def cover_crop_frame(
        image: Image.Image
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Resizes a frame as the official pose evaluation does.

    The frame is scaled to cover 480x832 keeping the aspect ratio, then
    center-cropped. OpenCV (INTER_AREA down, INTER_LINEAR up) is used when it
    is installed, as upstream; otherwise PIL's box / bilinear filters.

    Args:
        image: The RGB frame.

    Returns:
        The (480, 832, 3) uint8 frame and (scale_x, scale_y, offset_x,
        offset_y): a pixel ``u`` of the original lands at
        ``u * scale_x - offset_x`` of the crop.
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


def load_pose_video(
    image_paths: list[str]
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple]]:
    """Loads cover-resized, center-cropped frames.

    Args:
        image_paths: The frames.

    Returns:
        A ((T, 480, 832, 3) uint8 video, per-frame original (height, width),
        per-frame `cover_crop_frame` transform) tuple.
    """
    frames, sizes, transforms = [], [], []
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            sizes.append((image.height, image.width))
            frame, transform = cover_crop_frame(image)
            frames.append(frame)
            transforms.append(transform)
    return np.stack(frames, axis=0).astype(np.uint8), sizes, transforms


def pad_to_81(video: np.ndarray) -> np.ndarray:
    """Repeats the last frame up to 81 frames.

    As preprocess_video and the official evaluation do.

    Args:
        video: [T, H, W, 3] frames.

    Returns:
        The first 81 frames, padded with the last one.
    """
    if len(video) >= GENCEPTION_NUM_FRAMES:
        return video[:GENCEPTION_NUM_FRAMES]
    return np.concatenate([
        video,
        np.repeat(video[-1:], GENCEPTION_NUM_FRAMES - len(video), axis=0)
    ],
                          axis=0)


def float_output(pipe: Any, result: dict) -> np.ndarray:
    """Returns the decoded video in [0, 1].

    The pipeline's unquantized float copy when it keeps one (upstream's
    eval_common.get_last_video_float), else its uint8 output / 255.

    Args:
        pipe: The GenCeption pipeline after a call.
        result: The call's result.

    Returns:
        The float32 video.
    """
    if hasattr(pipe, "_last_video_float"):
        return np.asarray(pipe._last_video_float, dtype=np.float32)
    return np.asarray(result["video"], dtype=np.float32) / 255.0


def intrinsics_on_originals(intrinsics: np.ndarray,
                            transforms: list[tuple]) -> np.ndarray:
    """Maps K on the 480x832 crop to each original image.

    Args:
        intrinsics: (3, 3) K on the crop.
        transforms: Per-frame `cover_crop_frame` transforms
            (``u_crop = u * scale - offset``).

    Returns:
        (N, 3, 3) float64 per-frame K.
    """
    per_frame = np.repeat(intrinsics[None].astype(np.float64),
                          len(transforms),
                          axis=0)
    for index, (scale_x, scale_y, offset_x, offset_y) in enumerate(transforms):
        per_frame[index, 0, 0] /= scale_x
        per_frame[index, 0, 2] = (intrinsics[0, 2] + offset_x) / scale_x
        per_frame[index, 1, 1] /= scale_y
        per_frame[index, 1, 2] = (intrinsics[1, 2] + offset_y) / scale_y
    return per_frame


def select_input_frame_indices(num_frames: int) -> np.ndarray:
    """Returns the input frames GenCeption predicts.

    All of them up to 81, else preprocess_video's own temporal sampling
    (``np.linspace(0, T - 1, 81)``). Selecting them here keeps every
    prediction under its own image name.

    Args:
        num_frames: Number of input frames.

    Returns:
        The frame indices.
    """
    if num_frames > GENCEPTION_NUM_FRAMES:
        return np.linspace(0, num_frames - 1, GENCEPTION_NUM_FRAMES, dtype=int)
    return np.arange(num_frames)


def select_output_frame_indices(num_real_frames: int) -> np.ndarray:
    """Returns the output frames that correspond to real input frames.

    GenCeption always returns 81 frames. This replicates representations4d's
    own preprocess_video frame-count handling exactly (see
    genception/utils.py) so this stays a real 1:1 mapping rather than a
    guess:

    - num_real_frames > 81: preprocess_video linearly subsamples down to 81
      (np.linspace(0, T-1, 81)) -- every output frame maps to a distinct
      real frame, so we return all 81 indices as-is (identity).
    - num_real_frames <= 81: preprocess_video pads by repeating the last
      frame up to 81 -- only the first num_real_frames output frames
      correspond to real input frames; the rest are repeats we discard.

    Args:
        num_real_frames: Number of input frames.

    Returns:
        The output frame indices.
    """
    if num_real_frames > GENCEPTION_NUM_FRAMES:
        return np.arange(GENCEPTION_NUM_FRAMES)
    return np.arange(num_real_frames)


def resize_to_original(frames: np.ndarray, target_height: int,
                       target_width: int) -> np.ndarray:
    """Resizes frames back to an original size.

    GenCeption anisotropically resizes to (480, 832) with no
    aspect-preserving crop (see representations4d/genception/utils.py's
    preprocess_video and google-deepmind/representations4d#10's own note on
    this), so recovering the original resolution is a plain resize back,
    not a crop.

    Args:
        frames: [T, H, W, ...] frames.
        target_height: Original height.
        target_width: Original width.

    Returns:
        The resized frames, same dtype.
    """
    resized = np.empty(
        (frames.shape[0], target_height, target_width) + frames.shape[3:],
        dtype=frames.dtype)
    for index, frame in enumerate(frames):
        mode = "F" if frame.dtype == np.float32 else None
        image = Image.fromarray(frame, mode=mode)
        resized[index] = np.asarray(
            image.resize((target_width, target_height), Image.BILINEAR))
    return resized


def depth_pass(pipe: Any, video: np.ndarray, prompt_path: pathlib.Path,
               sizes: list[tuple[int, int]]) -> list[np.ndarray]:
    """Runs GenCeption's official depth prompt on a clip.

    Args:
        pipe: The GenCeption pipeline.
        video: At most 81 frames, from `load_video`.
        prompt_path: The depth prompt embedding.
        sizes: Per-frame original (height, width).

    Returns:
        One relative depth map per input frame, at its original size.
    """
    result = pipe(prompt_embeds=str(prompt_path), input_video=video)
    relative_depth = pipe.decode_depth(
        result["video"])[select_output_frame_indices(
            len(video))]  # [T, 480, 832]
    return [
        resize_to_original(depth[None], height, width)[0]
        for depth, (height, width) in zip(relative_depth, sizes)
    ]


def export_clip_depths(
        pipe: Any,
        image_paths: list[str],
        prompt_path: pathlib.Path,
        output_path: pathlib.Path,
        export_depth: Callable[...,
                               None] = utils.export_depth_predictions) -> None:
    """Runs the depth task over consecutive clips of at most 81 frames.

    Every input frame gets its own depth map.

    Args:
        pipe: The GenCeption pipeline.
        image_paths: The frames.
        prompt_path: The depth prompt embedding.
        output_path: Method scene output directory.
        export_depth: Writer of one frame's depth,
            `utils.export_depth_predictions`'s signature.
    """
    for clip_paths in utils.batched_image_paths(image_paths,
                                                GENCEPTION_NUM_FRAMES):
        video, sizes = load_video(clip_paths)
        for image_path, depth in zip(clip_paths,
                                     depth_pass(pipe, video, prompt_path,
                                                sizes),
                                     strict=True):
            export_depth(output_path, [image_path], depth[None])


def _path_env(name: str) -> pathlib.Path | None:
    """Returns the path in environment variable `name`, or None."""
    if name in os.environ:
        return pathlib.Path(os.environ[name])
    return None


class GenCeption(base.BaseModel):
    """GenCeption, one task per run.

    The tasks' own classes, `GenCeptionDepth` and `GenCeptionPose`, are the
    registered models. --ckpt_path is the directory with
    genception_{model_size}_transformer.npz and vae/ (see
    scripts/download/genception.sh).

    Attributes:
        task: "depth" or "pose".
    """

    name = "GENCEPTION"
    description = "Run GenCeption benchmark adapter."
    metrics_name = METHOD
    # GenCeption's input size is hardwired (480x832) and it exports no dense
    # points.
    uses_resolution = False
    uses_image_load_resolution = False
    exports_points = False
    uses_code_path = False
    task: ClassVar[str] = ""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.depth_prompt_path = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--prompt_embeddings_dir",
            type=pathlib.Path,
            default=_path_env("GENCEPTION_EMBEDDINGS_DIR"),
            help=("Directory of the official prompt embeddings; default "
                  "$GENCEPTION_EMBEDDINGS_DIR, else "
                  "<ckpt_path>/prompt_embeddings."),
        )
        parser.add_argument(
            "--model_size",
            type=str,
            default=os.environ.get("GENCEPTION_MODEL_SIZE", "14b"),
            help="Transformer size; default $GENCEPTION_MODEL_SIZE or 14b.")

    @property
    def prompt_embeddings_dir(self) -> pathlib.Path | None:
        """--prompt_embeddings_dir, else <ckpt_path>/prompt_embeddings."""
        if self.args.prompt_embeddings_dir is not None:
            return self.args.prompt_embeddings_dir
        if self.args.ckpt_path:
            return pathlib.Path(self.args.ckpt_path) / "prompt_embeddings"
        return None

    @property
    def checkpoint(self) -> str | None:
        """--ckpt_path."""
        return str(pathlib.Path(self.args.ckpt_path))

    def select_images(self) -> list[str]:
        """Returns the selected frames, as found."""
        return utils.select_interval(utils.find_image_paths(self.images_dir),
                                     self.args.interval,
                                     self.args.max_images_per_run)

    def planned_layout(self) -> dict[str, Any]:
        """Returns the outputs a run would write (the dry run).

        What GenCeption exports depends on its task, not on the shared
        --export_depth/--export_colmap/--depth_only flags, so this is its own
        small equivalent of `utils.planned_layout`.
        """
        args = self.args
        prompt_dir = self.prompt_embeddings_dir
        pose_decode = None
        if self.task == "pose":
            pose_decode = (f"{args.pose_solver}, "
                           f"{'raw' if args.raw_poses else 'smoothed'}")
        return {
            "method":
                METHOD,
            "data_path":
                str(args.data_path),
            "output_path":
                str(args.output_path),
            "num_images":
                len(self.image_paths),
            "task":
                self.task,
            "pose_decode":
                pose_decode,
            "model_size":
                args.model_size,
            "ckpt_path":
                str(pathlib.Path(args.ckpt_path)) if args.ckpt_path else None,
            "prompt_embeddings_dir":
                str(prompt_dir) if prompt_dir else None,
            "metrics":
                str(args.output_path / "metrics.json"),
            "depths":
                str(args.output_path /
                    "depths") if self.task == "depth" else None,
            "sparse":
                (str(args.output_path / "sparse" /
                     args.colmap_model_subdir) if self.task == "pose" else None
                ),
        }

    def load(self, device):
        # Loads the pipeline; it places itself on its device.
        del device  # GenCeption's pipeline picks its own device.
        # pylint: disable-next=import-outside-toplevel
        from representations4d import genception

        self.started_at = time.time()
        pipe = genception.GenCeptionPipeline.from_pretrained(
            self.checkpoint, model_size=self.args.model_size)
        utils.mark_model_loaded()
        return pipe

    def run(self) -> None:
        """Runs the task on the scene and writes its artifacts.

        Raises:
            FileNotFoundError: The scene has no images directory, or a prompt
                embedding is missing.
            ValueError: No images were found, or a directory option is
                missing.
        """
        args = self.args
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

        if not args.ckpt_path:
            raise ValueError("--ckpt_path or GENCEPTION_CKPT_DIR is required")

        self.model = self.load(self.device)
        self.depth_prompt_path = (self.prompt_embeddings_dir /
                                  DEPTH_PROMPT_FILENAME)
        if (self.task == "depth" or
                args.export_depth) and not self.depth_prompt_path.is_file():
            raise FileNotFoundError(
                f"{self.depth_prompt_path} not found -- run "
                "scripts/download/genception.sh first.")
        self.run_task()

    def run_task(self) -> None:
        """Runs the task with the loaded pipeline and writes its artifacts."""
        raise NotImplementedError


@registry.register_model
class GenCeptionDepth(GenCeption):
    """GenCeption's depth task: relative depth for every frame."""

    name = "GENCEPTION_DEPTH"
    task = "depth"

    def preprocess(self, image_paths, batch_index=None):
        # Loads one clip of at most 81 frames.
        video, sizes = load_video(image_paths)
        return base.ModelInput(image_paths,
                               data={"video": video},
                               batch_index=batch_index,
                               extra={"sizes": sizes})

    def forward(self, inputs):
        # Returns the clip's depth maps, each at its original size.
        depths = depth_pass(self.model, inputs.data["video"],
                            self.depth_prompt_path, inputs.extra["sizes"])
        return base.Prediction(depth=depths)

    def run_task(self):
        args = self.args
        export_clip_depths(self.model, self.image_paths, self.depth_prompt_path,
                           args.output_path)
        self.write_metrics(
            depth_only=True,
            metrics_extra={
                "task": "depth",
                "model_size": args.model_size,
                "prompt": "official (Input: RGB. Output: Depth.)",
                "resolution": f"{GENCEPTION_WIDTH}x{GENCEPTION_HEIGHT}",
                "frames_predicted": len(self.image_paths),
            },
        )


@registry.register_model
class GenCeptionPose(GenCeption):
    """GenCeption's pose task: the official raymap decode, one clip.

    At most 81 frames get poses.
    """

    name = "GENCEPTION_POSE"
    task = "pose"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--pose_solver",
            choices=POSE_SOLVERS,
            default=os.environ.get("GENCEPTION_POSE_SOLVER", "robust2"),
            help=("Rotation solver of the official raymap decode (both are "
                  "reported upstream; robust2 first); default "
                  "$GENCEPTION_POSE_SOLVER."),
        )
        parser.add_argument(
            "--raw_poses",
            action="store_true",
            default=os.environ.get("GENCEPTION_RAW_POSES", "0") == "1",
            help=("Export the per-frame poses instead of the temporally "
                  "smoothed ones (official headline); default "
                  "$GENCEPTION_RAW_POSES=1."),
        )

    def preprocess(self, image_paths, batch_index=None):
        # Loads cover-cropped frames (T <= 81).
        video, sizes, transforms = load_pose_video(image_paths)
        return base.ModelInput(image_paths,
                               data={"video": video},
                               batch_index=batch_index,
                               extra={
                                   "sizes": sizes,
                                   "crop_transforms": transforms
                               })

    def forward(self, inputs):
        # The raymap prompt and its decode: world-to-camera extrinsics; the
        # decode is in `extra`, the intrinsics are set by postprocess.
        # pylint: disable-next=import-outside-toplevel
        from spatialarena.geometry import genception_raymap

        args = self.args
        video = inputs.data["video"]
        prompt_path = self.prompt_embeddings_dir / RAYMAP_PROMPT_FILENAME
        if not prompt_path.is_file():
            raise FileNotFoundError(f"{prompt_path} not found -- run "
                                    "scripts/download/genception.sh first.")
        result = self.model(prompt_embeds=str(prompt_path),
                            input_video=pad_to_81(video))
        # All 81 output frames are decoded (the clip-wide K fit and the
        # smoothing window see the padding, as upstream); only the real
        # frames are exported.
        decoded = genception_raymap.decode_poses_from_raymap(
            float_output(self.model, result), solver=args.pose_solver)
        output_indices = select_output_frame_indices(len(video))
        camera_to_world = decoded.poses(
            smoothed=not args.raw_poses)[output_indices]
        return base.Prediction(extrinsics_w2c=utils.c2w_to_w2c(camera_to_world),
                               extra={
                                   "decoded": decoded,
                                   "output_indices": output_indices
                               })

    def postprocess(self, prediction, inputs):
        # Maps K to the originals (one size) or keeps it on the crop.
        sizes = inputs.extra["sizes"]
        transforms = inputs.extra["crop_transforms"]
        decoded = prediction.extra["decoded"]
        count = len(inputs.image_paths)
        if len(set(sizes)) == 1:
            height, width = sizes[0]
            intrinsics = intrinsics_on_originals(decoded.intrinsics,
                                                 transforms[:1])
            intrinsics = np.repeat(intrinsics, count, axis=0)
        else:
            # Mixed sizes: one COLMAP camera size only, so keep K on the
            # 480x832 crop.
            height, width = GENCEPTION_HEIGHT, GENCEPTION_WIDTH
            intrinsics = np.repeat(decoded.intrinsics[None], count, axis=0)
        prediction.intrinsics = intrinsics
        prediction.extra["intrinsics_size"] = (width, height)
        return prediction

    def run_task(self):
        args = self.args
        # The frames GenCeption predicts, by name.
        image_paths = [
            self.image_paths[index]
            for index in select_input_frame_indices(len(self.image_paths))
        ]
        self.image_paths = image_paths
        inputs = self.preprocess(image_paths)
        prediction = self.postprocess(self.forward(inputs), inputs)
        decoded = prediction.extra["decoded"]
        output_indices = prediction.extra["output_indices"]
        width, height = prediction.extra["intrinsics_size"]

        if args.export_depth:
            # Second pass with the official depth prompt on the same frames
            # (GenCeption predicts one output per prompt): relative depth,
            # not in the pose's scale; no confidence exists.
            depth_video, _ = load_video(image_paths)
            depths = depth_pass(self.model, depth_video, self.depth_prompt_path,
                                inputs.extra["sizes"])
            for image_path, depth in zip(image_paths, depths, strict=True):
                utils.export_depth_predictions(args.output_path, [image_path],
                                               depth[None])

        utils.export_camera_pose_priors(
            args.output_path,
            self.images_dir,
            image_paths,
            prediction.extrinsics_w2c,
            prediction.intrinsics,
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
                "float"
                if hasattr(self.model, "_last_video_float") else "uint8 / 255",
            "preprocess":
                "cover resize + center crop to 832x480",
            "intrinsics_on_crop":
                decoded.intrinsics.tolist(),
            "intrinsics_size": [width, height],
        }
        depth_pass_info = None
        if args.export_depth:
            depth_pass_info = "official depth prompt"
        self.write_metrics(
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
                    depth_pass_info,
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
        (args.output_path /
         "genception_pose_decode_diagnostics.json").write_text(
             json.dumps(decode_info, indent=2) + "\n", encoding="utf-8")
