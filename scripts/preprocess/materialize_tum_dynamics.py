# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Materialize the TUM RGB-D "Dynamic Objects" sequences as SLAM cases.

The sequences (Sturm et al., IROS 2012 --
https://cvg.cit.tum.de/data/datasets/rgbd-dataset) are written to the
benchmark_root/SLAM_source/tum_dynamics canonical layout.

Source layout (one directory per sequence, see scripts/download/
tum_dynamics.sh):

    <source_root>/rgbd_dataset_freiburg{2,3}_<name>/
        rgb.txt        # "timestamp path", one row per RGB frame
        depth.txt      # "timestamp path", one row per depth frame
        groundtruth.txt  # "timestamp tx ty tz qx qy qz qw", camera-to-world
                         # pose of the RGB optical center in the world frame
        rgb/*.png
        depth/*.png    # 16-bit, factor 5000 (raw TUM scale, copied through
                        # unchanged -- SLAM evaluation is pose-only, this
                        # dataset does not feed the depth-accuracy metric
                        # table's per-dataset gt_depth_scale lookup)

rgb/depth/groundtruth are associated by nearest timestamp (the standard
TUM `associate.py` greedy-nearest-neighbor algorithm, max_difference=0.02s,
matching the tool this dataset's own benchmark scripts have always used).
Camera intrinsics are the dataset's own published per-camera calibration
(freiburg2 has real lens distortion -- FULL_OPENCV COLMAP model; freiburg3
images are pre-undistorted -- PINHOLE).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.geometry import read_write_model  # noqa: E402

# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
# ("Intrinsic Camera Calibration of the Kinect").
FREIBURG_INTRINSICS = {
    "freiburg2": {
        "fx": 520.9,
        "fy": 521.0,
        "cx": 325.1,
        "cy": 249.7,
        "k1": 0.2312,
        "k2": -0.7849,
        "p1": -0.0033,
        "p2": -0.0001,
        "k3": 0.9172,
    },
    "freiburg3": {
        "fx": 535.4,
        "fy": 539.2,
        "cx": 320.1,
        "cy": 247.6,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
    },
}

SEQUENCES = (
    "rgbd_dataset_freiburg2_desk_with_person",
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_walking_static",
    "rgbd_dataset_freiburg3_walking_xyz",
    "rgbd_dataset_freiburg3_walking_halfsphere",
    "rgbd_dataset_freiburg3_walking_rpy",
)


def freiburg_model(sequence_name: str) -> str:
    """Returns the sequence's camera, "freiburg2" or "freiburg3".

    Args:
        sequence_name: The TUM sequence directory name.

    Returns:
        The key of FREIBURG_INTRINSICS in the name.

    Raises:
        ValueError: If the name contains no known camera.
    """
    for model in FREIBURG_INTRINSICS:
        if model in sequence_name:
            return model
    raise ValueError(
        f"Cannot determine freiburg camera model for {sequence_name!r}")


def read_file_list(path: Path) -> dict[float, str]:
    """Reads a TUM "timestamp path" list as {timestamp: path}."""
    result: dict[float, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        result[float(parts[0])] = parts[1]
    return result


def read_groundtruth(path: Path) -> dict[float, np.ndarray]:
    """Reads groundtruth.txt as {timestamp: 4x4 camera-to-world matrix}.

    TUM pose convention: tx ty tz qx qy qz qw, scalar-last quaternion, already
    in OpenCV/COLMAP camera axis convention -- no axis flip needed, unlike
    NRGBD's OpenGL poses.

    Args:
        path: The sequence's groundtruth.txt.

    Returns:
        The poses keyed by timestamp.
    """
    result: dict[float, np.ndarray] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, tx, ty, tz, qx, qy, qz, qw = (
            float(value) for value in line.split())
        quat = np.array([qx, qy, qz, qw], dtype=np.float64)
        quat = quat / np.linalg.norm(quat)
        x, y, z, w = quat
        rotation = np.array(
            [
                [
                    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 *
                    (x * z + y * w)
                ],
                [
                    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 *
                    (y * z - x * w)
                ],
                [
                    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 *
                    (x * x + y * y)
                ],
            ],
            dtype=np.float64,
        )
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = [tx, ty, tz]
        result[timestamp] = c2w
    return result


def associate(first: dict,
              second: dict,
              max_difference: float = 0.02) -> list[tuple[float, float]]:
    """Matches timestamps as the standard TUM associate.py does.

    Greedy nearest-neighbor matching: the closest remaining pair within
    `max_difference` is matched first.

    Args:
        first: A dict keyed by timestamp.
        second: A dict keyed by timestamp.
        max_difference: The largest accepted timestamp difference (seconds).

    Returns:
        The matched (first, second) timestamps, sorted.
    """
    first_keys = list(first.keys())
    second_keys = list(second.keys())
    potential = sorted((abs(a - b), a, b)
                       for a in first_keys
                       for b in second_keys
                       if abs(a - b) < max_difference)
    matches = []
    used_first: set[float] = set()
    used_second: set[float] = set()
    for _, a, b in potential:
        if a in used_first or b in used_second:
            continue
        used_first.add(a)
        used_second.add(b)
        matches.append((a, b))
    matches.sort()
    return matches


def image_size(path: Path) -> tuple[int, int]:
    """Returns an image's (width, height)."""
    from PIL import Image  # pylint: disable=import-outside-toplevel

    with Image.open(path) as image:
        return image.size


def camera_for_model(model: str, width: int, height: int,
                     camera_id: int) -> read_write_model.Camera:
    """Builds the COLMAP camera of a freiburg calibration.

    Args:
        model: "freiburg2" or "freiburg3".
        width: The image width.
        height: The image height.
        camera_id: The COLMAP camera id.

    Returns:
        A FULL_OPENCV camera when the calibration has distortion, else a
        PINHOLE camera.
    """
    intr = FREIBURG_INTRINSICS[model]
    has_distortion = any(
        intr[key] != 0.0 for key in ("k1", "k2", "p1", "p2", "k3"))
    if has_distortion:
        return read_write_model.Camera(
            id=camera_id,
            model="FULL_OPENCV",
            width=width,
            height=height,
            params=np.array(
                [
                    intr["fx"],
                    intr["fy"],
                    intr["cx"],
                    intr["cy"],
                    intr["k1"],
                    intr["k2"],
                    intr["p1"],
                    intr["p2"],
                    intr["k3"],
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
        )
    return read_write_model.Camera(
        id=camera_id,
        model="PINHOLE",
        width=width,
        height=height,
        params=np.array([intr["fx"], intr["fy"], intr["cx"], intr["cy"]],
                        dtype=np.float64),
    )


def link_path(src: Path, dst: Path, overwrite: bool, copy: bool) -> None:
    """Copies or symlinks `src` to `dst`.

    Args:
        src: The source file.
        dst: The destination path.
        overwrite: Replace an existing `dst`; otherwise it is kept.
        copy: Copy `src`; when False, symlink to it.
    """
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def materialize_sequence(
    source_root: Path,
    sequence_name: str,
    output_root: Path,
    overwrite: bool,
    copy: bool,
    max_difference: float,
) -> dict | None:
    """Writes one sequence's associated frames and GT COLMAP model.

    Args:
        source_root: The directory of extracted sequence folders.
        sequence_name: The sequence folder name.
        output_root: benchmark_root/SLAM_source/tum_dynamics.
        overwrite: Replace existing outputs.
        copy: Copy frames; when False, symlink to them.
        max_difference: The timestamp association tolerance (seconds).

    Returns:
        The scene record, or None when the sequence's lists are missing.
    """
    seq_root = source_root / sequence_name
    rgb_file = seq_root / "rgb.txt"
    depth_file = seq_root / "depth.txt"
    gt_file = seq_root / "groundtruth.txt"
    if not (rgb_file.is_file() and depth_file.is_file() and gt_file.is_file()):
        return None

    rgb = read_file_list(rgb_file)
    depth = read_file_list(depth_file)
    groundtruth = read_groundtruth(gt_file)

    rgb_depth_matches = associate(rgb, depth, max_difference=max_difference)
    rgb_depth = dict(rgb_depth_matches)
    triple_matches = associate({rgb_ts: None for rgb_ts in rgb_depth},
                               groundtruth,
                               max_difference=max_difference)

    out_scene = output_root / sequence_name
    (out_scene / "images").mkdir(parents=True, exist_ok=True)
    (out_scene / "depths").mkdir(parents=True, exist_ok=True)

    model = freiburg_model(sequence_name)
    cameras: dict[int, read_write_model.Camera] = {}
    images: dict[int, read_write_model.BaseImage] = {}

    for image_id, (rgb_ts, gt_ts) in enumerate(triple_matches, start=1):
        depth_ts = rgb_depth[rgb_ts]
        rgb_path = seq_root / rgb[rgb_ts]
        depth_path = seq_root / depth[depth_ts]
        image_name = f"{image_id:06d}.png"
        link_path(rgb_path,
                  out_scene / "images" / image_name,
                  overwrite=overwrite,
                  copy=copy)
        link_path(depth_path,
                  out_scene / "depths" / image_name,
                  overwrite=overwrite,
                  copy=copy)

        width, height = image_size(rgb_path)
        c2w = groundtruth[gt_ts]
        w2c = np.linalg.inv(c2w)
        cameras[image_id] = camera_for_model(model, width, height, image_id)
        images[image_id] = read_write_model.BaseImage(
            id=image_id,
            qvec=read_write_model.rotmat2qvec(w2c[:3, :3]).astype(np.float64),
            tvec=w2c[:3, 3].astype(np.float64),
            camera_id=image_id,
            name=image_name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    sparse_dir = out_scene / "sparse" / "0"
    if overwrite or not sparse_dir.exists():
        sparse_dir.mkdir(parents=True, exist_ok=True)
        read_write_model.write_model(
            cameras,
            images,
            {},
            str(sparse_dir),
            ext=".txt",
        )

    return {
        "dataset_name": "tum_dynamics",
        "scene_name": sequence_name,
        "root_directory": str(out_scene),
        "total_images": len(images),
        "camera_model": model,
        "rgb_frames_available": len(rgb),
        "matched_frames": len(images),
    }


TUM_DEPTH_PNG_SCALE = 5000.0


def write_metadata(output_root: Path, scenes: list[dict]) -> None:
    """Writes the dataset-level metadata.json over the scene records."""
    output_root.mkdir(parents=True, exist_ok=True)
    total_images = sum(int(scene.get("total_images", 0)) for scene in scenes)
    payload = {
        "dataset_name":
            "tum_dynamics",
        "root_directory":
            str(output_root),
        "total_images":
            total_images,
        "average_images_per_scene":
            total_images / len(scenes) if scenes else 0.0,
        # TUM depth PNGs are 16-bit with 5000 units per metre (cvg.cit.tum.de
        # "File Formats"), not millimetres; adapter_utils.load_depth_priors
        # reads this instead of guessing.
        "depth_png_scale":
            TUM_DEPTH_PNG_SCALE,
        "scenes":
            scenes,
    }
    (output_root / "metadata.json").write_text(json.dumps(payload, indent=2) +
                                               "\n",
                                               encoding="utf-8")


def materialize_dataset(
    source_root: Path,
    benchmark_root: Path,
    overwrite: bool,
    copy: bool = True,
    max_difference: float = 0.02,
) -> dict:
    """Materializes every available sequence and writes the metadata.

    Args:
        source_root: The directory of extracted sequence folders.
        benchmark_root: The benchmark root to write into.
        overwrite: Replace existing outputs.
        copy: Copy frames; when False, symlink to them.
        max_difference: The timestamp association tolerance (seconds).

    Returns:
        A summary with the dataset name, scene count and output root.
    """
    output_root = benchmark_root / "SLAM_source" / "tum_dynamics"
    records = []
    for sequence_name in SEQUENCES:
        record = materialize_sequence(source_root, sequence_name, output_root,
                                      overwrite, copy, max_difference)
        if record is not None:
            records.append(record)
    write_metadata(output_root, records)
    return {
        "dataset": "tum_dynamics",
        "num_scenes": len(records),
        "output_root": str(output_root)
    }


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source_root",
        type=Path,
        required=True,
        help="Directory holding the extracted rgbd_dataset_* sequence folders.")
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--symlink",
                        action="store_true",
                        help="Symlink instead of copying (not self-contained).")
    parser.add_argument("--max_difference", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    """Materializes the dataset and prints the summary."""
    args = parse_args()
    result = materialize_dataset(
        args.source_root,
        args.benchmark_root,
        overwrite=args.overwrite,
        copy=not args.symlink,
        max_difference=args.max_difference,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
