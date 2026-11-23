# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Write the SLAM track's fixed-resolution inputs.

Every sequence is written at its dataset's protocol size. The SLAM track feeds
every method the same pixels (`spatialarena.slam_protocol.DATASET_INPUT_SIZE`:
504x378 for the 4:3 datasets, 504x252 for VBR, 504x154 for KITTI Odometry), so
methods are compared at one resolution instead of each one's own resize. This
script produces those frames once, from the native-resolution
`<benchmark_root>/SLAM_source/<dataset>/<scene>` (`slam_protocol.source_root`),
into `<benchmark_root>/SLAM/<dataset>/<scene>` (`slam_protocol.input_root`):

- `images/`: resized to fit within the target aspect ratio preserving the
  source's own aspect ratio (a no-op for 4:3 frames), then padded with zeros to
  the target size (a no-op crop/pad would otherwise discard real content --
  VBR's 1388x700 and KITTI Odometry's wide frames keep every source pixel
  instead of losing rows/columns), saved as PNG under the same names, so
  predictions still match the ground truth by image name;
- `depths/` (rgb_depth track priors, where the dataset has them): the same
  resize-within and pad, nearest-neighbor, dtype and units unchanged; the
  padded border reads as 0, this codebase's established "no depth prior here"
  convention (e.g. scripts/inference/pi3.py's _zero_depth_like_image) -- not
  a real (and definitely not a zero) depth measurement;
- `sparse/0`: the ground-truth model with the camera intrinsics mapped through
  the same resize-within and pad (poses unchanged).

This is the benchmark's SLAM data: methods read their frames and the evaluator
its ground-truth poses from it. Run on a compute node (CPU):
`python scripts/preprocess/materialize_slam_inputs.py --datasets tum_dynamics`.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent import futures
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena import slam_protocol  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402
from spatialarena.geometry import read_write_model  # noqa: E402
# pylint: enable=wrong-import-position

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
# COLMAP camera models whose leading parameters are (f) or (fx, fy) followed by
# (cx, cy); distortion terms after them are unitless and unchanged by a crop
# and uniform scale.
FOCAL_COUNT = {
    "SIMPLE_PINHOLE": 1,
    "SIMPLE_RADIAL": 1,
    "RADIAL": 1,
    "PINHOLE": 2,
    "OPENCV": 2,
    "OPENCV_FISHEYE": 2,
    "FULL_OPENCV": 2
}


def pad_box(width: int, height: int,
            target: tuple[int, int]) -> tuple[float, int, int, int, int]:
    """Computes the resize-within and pad of a frame.

    A `width` x `height` frame is resized to fit within `target` preserving
    its own aspect ratio (never upscaling past it, never cropping), then
    padded, centered, to `target`.

    Args:
        width: Frame width.
        height: Frame height.
        target: Target (width, height).

    Returns:
        (scale, pad_left, pad_top, resized_width, resized_height).
    """
    target_w, target_h = target
    scale = min(target_w / width, target_h / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    return scale, pad_left, pad_top, resized_w, resized_h


def fit_image(image: np.ndarray,
              target: tuple[int, int],
              *,
              nearest: bool = False) -> np.ndarray:
    """Resizes `image` to fit within `target` and pads it to `target`.

    The image keeps its own aspect ratio and is not cropped; the padded border
    is zero (see this module's docstring).

    Args:
        image: An HxW or HxWxC image.
        target: Target (width, height).
        nearest: Whether to resize with nearest-neighbor (for depth maps)
            instead of area interpolation.

    Returns:
        The image at the target size.
    """
    # Only the worker processes need OpenCV.
    import cv2  # pylint: disable=import-outside-toplevel

    _scale, pad_l, pad_t, resized_w, resized_h = pad_box(
        image.shape[1], image.shape[0], target)
    if (resized_w, resized_h) == (image.shape[1], image.shape[0]):
        resized = image
    else:
        if nearest:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_AREA
        resized = cv2.resize(image, (resized_w, resized_h),
                             interpolation=interpolation)
    if (resized_w, resized_h) == target:
        return np.ascontiguousarray(resized)
    canvas = np.zeros((target[1], target[0]) + image.shape[2:],
                      dtype=image.dtype)
    canvas[pad_t:pad_t + resized_h, pad_l:pad_l + resized_w] = resized
    return canvas


def fit_camera(camera: read_write_model.Camera,
               target: tuple[int, int]) -> read_write_model.Camera:
    """Maps a camera's intrinsics through `fit_image`'s resize-within and pad.

    Pixel-center convention: unlike a crop offset, which is subtracted in
    source space before scaling, a pad offset is added in destination space
    after scaling.

    Args:
        camera: A COLMAP camera with a model in `FOCAL_COUNT`.
        target: Target (width, height).

    Returns:
        The camera at the target size.

    Raises:
        ValueError: If the camera model is not supported.
    """
    if camera.model not in FOCAL_COUNT:
        raise ValueError(
            f"Unsupported camera model {camera.model} for camera {camera.id}")
    scale, pad_l, pad_t, _resized_w, _resized_h = pad_box(
        camera.width, camera.height, target)
    params = np.asarray(camera.params, dtype=np.float64).copy()
    focals = FOCAL_COUNT[camera.model]
    params[:focals] *= scale
    params[focals] = (params[focals] + 0.5) * scale - 0.5 + pad_l
    params[focals + 1] = (params[focals + 1] + 0.5) * scale - 0.5 + pad_t
    return read_write_model.Camera(id=camera.id,
                                   model=camera.model,
                                   width=target[0],
                                   height=target[1],
                                   params=params)


def _frames(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir()
                  if path.suffix.lower() in IMAGE_SUFFIXES)


def _write_frames(source: Path, destination: Path,
                  target: tuple[int, int]) -> set[tuple[int, int]]:
    """Writes the resized and padded images and depths of one sequence.

    Returns:
        The distinct (width, height) sizes of the source images.

    Raises:
        RuntimeError: If a frame cannot be read or written.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    sizes = set()
    for subdir, nearest, flags in (("images", False, cv2.IMREAD_COLOR),
                                   ("depths", True, cv2.IMREAD_UNCHANGED)):
        if not (source / subdir).is_dir():
            continue
        (destination / subdir).mkdir(parents=True, exist_ok=True)
        for path in _frames(source / subdir):
            frame = cv2.imread(str(path), flags)
            if frame is None:
                raise RuntimeError(f"Could not read {path}")
            if subdir == "images":
                sizes.add((frame.shape[1], frame.shape[0]))
            # Same names: predictions are matched to the GT by name.
            out = destination / subdir / path.name
            if not cv2.imwrite(str(out),
                               fit_image(frame, target, nearest=nearest)):
                raise RuntimeError(f"Could not write {out}")
    return sizes


def materialize_scene(source: Path, destination: Path, target: tuple[int, int],
                      overwrite: bool) -> dict:
    """Writes one sequence's pre-sized images, depths and camera model.

    An existing scene record with the same image count and input size is
    reused unless `overwrite` is set.

    Args:
        source: The native-resolution sequence folder.
        destination: The output sequence folder.
        target: Target (width, height).
        overwrite: Whether to rewrite an up-to-date sequence.

    Returns:
        The scene record (also written to `scene.json`).
    """
    record_path = destination / "scene.json"
    images = _frames(source / "images")
    if record_path.is_file() and not overwrite:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("num_images") == len(images) and tuple(
                record.get("input_size", ())) == target:
            return record
    sizes = _write_frames(source, destination, target)
    sparse = source / "sparse" / "0"
    if sparse.is_dir():
        cameras, model_images, points = read_write_model.read_model(str(sparse))
        (destination / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        read_write_model.write_model(
            {
                key: fit_camera(camera, target)
                for key, camera in cameras.items()
            },
            model_images,
            points,
            str(destination / "sparse" / "0"),
            ext=".txt")
    record = {
        "scene_name": source.name,
        "source": str(source),
        "num_images": len(images),
        "source_sizes": sorted(sizes),
        "input_size": list(target),
        "has_depths": (source / "depths").is_dir(),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n",
                           encoding="utf-8")
    return record


def scene_dirs(dataset_root: Path) -> list[Path]:
    """Returns every sequence directory (one with `images/`) of a dataset."""
    return sorted(
        path for path in dataset_root.iterdir() if (path / "images").is_dir())


def _parse_args() -> argparse.Namespace:
    """Parses the command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root",
                        type=Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--datasets",
                        nargs="*",
                        default=sorted(slam_protocol.DATASET_INPUT_SIZE))
    parser.add_argument("--scenes",
                        nargs="*",
                        default=[],
                        help="Restrict to these sequences (default: all).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _materialize_dataset(args: argparse.Namespace, dataset: str) -> None:
    """Materializes one dataset's sequences and writes its metadata."""
    target = slam_protocol.input_size(dataset)
    source_root = slam_protocol.source_root(args.benchmark_root, dataset)
    output_root = slam_protocol.input_root(args.benchmark_root, dataset)
    scenes = [
        path for path in scene_dirs(source_root)
        if not args.scenes or path.name in args.scenes
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [
            pool.submit(materialize_scene, scene, output_root / scene.name,
                        target, args.overwrite) for scene in scenes
        ]
        records = [future.result() for future in pending]
    source_metadata = json.loads(
        (source_root / "metadata.json").read_text(encoding="utf-8"))
    payload = {
        "dataset_name": dataset,
        "source_root": str(source_root),
        "input_size": list(target),
        "resize": ("resize to fit the target aspect ratio (cv2.INTER_AREA; "
                   "depths: INTER_NEAREST), then zero-pad"),
        "num_scenes": len(records),
        "total_images": sum(record["num_images"] for record in records),
        "scenes": sorted(records, key=lambda record: record["scene_name"]),
    }
    # adapter_utils.dataset_depth_png_scale reads it from here.
    if source_metadata.get("depth_png_scale"):
        payload["depth_png_scale"] = source_metadata["depth_png_scale"]
    (output_root / "metadata.json").write_text(json.dumps(payload, indent=2) +
                                               "\n",
                                               encoding="utf-8")
    summary_keys = ("dataset_name", "input_size", "num_scenes", "total_images")
    print(json.dumps({key: payload[key] for key in summary_keys}))


def main() -> None:
    """Materializes the selected datasets' sequences in parallel.

    Each dataset's metadata is written after its sequences.
    """
    args = _parse_args()
    for dataset in args.datasets:
        _materialize_dataset(args, dataset)


if __name__ == "__main__":
    main()
