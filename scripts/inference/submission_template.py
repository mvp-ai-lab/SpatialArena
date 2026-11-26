#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Template of an inference script for a SpatialArena submission.

Copy this file into your model's repository, replace `predict()` with your
model, and give the command that runs it in the submission form, e.g.

    python {code_path}/spatialarena_infer.py --images {images_dir} \
        --output {output_path} --checkpoint {ckpt_path}

The evaluation pipeline runs that command once per scene (or per group of a
scene's images) and scores what it writes to --output:

    depths/<image stem>.npz   one per input image, float32 array "depth" (H, W)
    sparse/0/                 COLMAP model (cameras.txt, images.txt,
                              points3D.txt) of the predicted cameras, whose
                              image names are the input file names

It needs only NumPy and Pillow besides your model. Check the output with
scripts/inference/check_submission_output.py before submitting; see
docs/benchmark/submitting_a_model.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclasses.dataclass
class Prediction:
    """What the model predicts for a set of images, in the input order.

    Attributes:
        depths: One (H, W) depth map per image, in any unit and at any
            resolution: the evaluator resizes it to the ground truth and
            aligns its scale. Depth, not disparity; invalid pixels are 0, NaN
            or inf.
        world_to_camera: (N, 4, 4) world-to-camera transforms (COLMAP
            convention: x right, y down, z forward), or None for a model that
            predicts no cameras.
        intrinsics: (N, 3, 3) pinhole intrinsics in pixels of the input
            images, or None.
        confidences: Optional (H, W) confidence maps, higher is more
            confident.
    """
    depths: list[np.ndarray]
    world_to_camera: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    confidences: list[np.ndarray] | None = None


def predict(images: list[np.ndarray], checkpoint: str | None,
            known_cameras: pathlib.Path | None) -> Prediction:
    """Runs the model on the images. REPLACE THIS with your model.

    This placeholder predicts a constant depth and cameras spaced along a
    line, so the template runs end to end and its output is scored (badly).

    Args:
        images: The input images, (H, W, 3) uint8 RGB, in file-name order.
        checkpoint: The --checkpoint path, or None.
        known_cameras: The ground-truth COLMAP model on the camera-conditioned
            tracks (--cameras), else None.

    Returns:
        The prediction.
    """
    del checkpoint, known_cameras  # Unused by the placeholder.
    print("submission_template.py: placeholder prediction, replace predict()",
          file=sys.stderr)
    count = len(images)
    height, width = images[0].shape[:2]
    focal = 1.2 * max(height, width)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2],
                           [0, 0, 1]])
    world_to_camera = np.repeat(np.eye(4)[None], count, axis=0)
    world_to_camera[:, 0, 3] = -0.1 * np.arange(count)  # Centers x = 0.1 i.
    return Prediction(
        depths=[np.full(image.shape[:2], 2.0, np.float32) for image in images],
        world_to_camera=world_to_camera,
        intrinsics=np.repeat(intrinsics[None], count, axis=0),
    )


def list_images(images_dir: pathlib.Path) -> list[pathlib.Path]:
    """Returns the images of a folder, sorted by name."""
    return sorted(path for path in images_dir.iterdir()
                  if path.suffix.lower() in IMAGE_SUFFIXES)


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Converts a rotation matrix to a COLMAP quaternion (w, x, y, z)."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        quaternion = [
            0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        quaternion = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        quaternion = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        quaternion = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    quaternion = np.asarray(quaternion)
    return quaternion / np.linalg.norm(quaternion)


def write_colmap_text(model_dir: pathlib.Path, names: list[str],
                      sizes: list[tuple[int, int]], world_to_camera: np.ndarray,
                      intrinsics: np.ndarray) -> None:
    """Writes a COLMAP text model: one PINHOLE camera per image, no points.

    Args:
        model_dir: The model folder (sparse/0).
        names: Image file names, as the input folder has them.
        sizes: (width, height) of each image.
        world_to_camera: (N, 4, 4) world-to-camera transforms.
        intrinsics: (N, 3, 3) intrinsics in pixels.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    cameras = ["# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy"]
    images = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, then points"]
    for index, name in enumerate(names):
        width, height = sizes[index]
        k = intrinsics[index]
        cameras.append(f"{index + 1} PINHOLE {width} {height} "
                       f"{k[0, 0]:.6f} {k[1, 1]:.6f} {k[0, 2]:.6f} "
                       f"{k[1, 2]:.6f}")
        quaternion = rotation_to_quaternion(world_to_camera[index, :3, :3])
        translation = world_to_camera[index, :3, 3]
        values = " ".join(
            f"{value:.9f}" for value in (*quaternion, *translation))
        images.append(f"{index + 1} {values} {index + 1} {name}")
        images.append("")  # No 2D points.
    (model_dir / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (model_dir / "images.txt").write_text("\n".join(images) + "\n")
    (model_dir /
     "points3D.txt").write_text("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")


def write_prediction(output: pathlib.Path, image_paths: list[pathlib.Path],
                     sizes: list[tuple[int,
                                       int]], prediction: Prediction) -> None:
    """Writes a prediction in the layout the evaluator reads.

    Args:
        output: The --output folder.
        image_paths: The input images, in the prediction's order.
        sizes: (width, height) of each image.
        prediction: The model's prediction.

    Raises:
        ValueError: The prediction does not have one entry per image.
    """
    if len(prediction.depths) != len(image_paths):
        raise ValueError(f"{len(prediction.depths)} depth maps for "
                         f"{len(image_paths)} images")
    for path, depth in zip(image_paths, prediction.depths):
        np.savez_compressed(output / "depths" / f"{path.stem}.npz",
                            depth=np.asarray(depth, np.float32))
    for path, confidence in zip(image_paths, prediction.confidences or ()):
        np.savez_compressed(output / "confidences" / f"{path.stem}.npz",
                            confidence=np.asarray(confidence, np.float16))
    if prediction.world_to_camera is not None:
        write_colmap_text(output / "sparse" / "0",
                          [path.name for path in image_paths], sizes,
                          prediction.world_to_camera, prediction.intrinsics)


def main() -> int:
    """Runs the model on one scene and writes its prediction."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images",
                        type=pathlib.Path,
                        required=True,
                        help="the input images ({images_dir})")
    parser.add_argument("--output",
                        type=pathlib.Path,
                        required=True,
                        help="where to write the prediction ({output_path})")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="the weights ({ckpt_path})")
    parser.add_argument("--cameras",
                        type=pathlib.Path,
                        default=None,
                        help="ground-truth cameras on conditioned tracks "
                        "({camera_dir})")
    args = parser.parse_args()

    image_paths = list_images(args.images)
    if not image_paths:
        print(f"no images in {args.images}", file=sys.stderr)
        return 1
    images = []
    sizes = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB")))
            sizes.append(image.size)
    prediction = predict(images, args.checkpoint, args.cameras)
    (args.output / "depths").mkdir(parents=True, exist_ok=True)
    if prediction.confidences:
        (args.output / "confidences").mkdir(parents=True, exist_ok=True)
    write_prediction(args.output, image_paths, sizes, prediction)
    print(f"wrote the prediction of {len(image_paths)} images to "
          f"{args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
