# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Materialize KITTI Odometry (Geiger et al., CVPR 2012) for the SLAM track.

The output is benchmark_root/SLAM_source/kitti_odometry.

Source: the official odometry release (scripts/download/kitti_odometry.sh),
the form LoGeR (eval/run_kitti.sh), LingBot-Map (benchmark/datasets/kitti.py)
and VGGT-Long evaluate on: the left color camera of sequences 00-10, the ones
with public ground truth.

    <raw_root>/dataset/sequences/NN/image_2/NNNNNN.png
        left color camera (cam2), rectified, one file per frame
    <raw_root>/dataset/sequences/NN/calib.txt
        P0..P3 (3x4 rectified projections) and Tr
    <raw_root>/dataset/sequences/NN/times.txt
        one timestamp per frame (10 Hz)
    <raw_root>/dataset/poses/NN.txt
        row k: the 3x4 camera-to-world pose of frame k's cam0

The ground truth is cam0's (left grayscale) trajectory. cam2 is rectified to
the same orientation and offset along the baseline: `P2 = K2 [I | t]`, so a
cam0 point maps to cam2 as `x2 = x0 + t` with `t = K2^-1 P2[:, 3]` (about
6 cm). The poses written here are cam2's, `T_world_cam2 = T_world_cam0
[I | -t]`, so the GT is the camera the methods see (LingBot-Map keeps cam0's
and leaves the offset to the Sim(3) alignment). Images are linked (or copied)
unchanged, with a PINHOLE camera from `P2` in `sparse/0`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent import futures
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena.geometry import read_write_model  # noqa: E402
# pylint: enable=wrong-import-position

# The sequences with public ground truth (LoGeR eval/run_kitti.sh, LingBot-Map
# GT_SEQUENCES).
SEQUENCES = tuple(f"{index:02d}" for index in range(11))


def read_projection(calib_path: Path, name: str = "P2") -> np.ndarray:
    """Reads a 3x4 rectified projection matrix from a KITTI calib.txt.

    Args:
        calib_path: The sequence's calib.txt.
        name: The matrix name, e.g. "P2".

    Returns:
        The 3x4 projection matrix.

    Raises:
        ValueError: If `name` is not in the file.
    """
    for line in calib_path.read_text(encoding="utf-8").splitlines():
        key, _, values = line.partition(":")
        if key.strip() == name:
            return np.array(values.split(), dtype=np.float64).reshape(3, 4)
    raise ValueError(f"{name} not found in {calib_path}")


def cam0_to_cam2_offset(projection: np.ndarray) -> np.ndarray:
    """Returns `t` with `x_cam2 = x_cam0 + t` for rectified `P2 = K2 [I|t]`."""
    return np.linalg.solve(projection[:, :3], projection[:, 3])


def read_cam2_poses(poses_path: Path, offset: np.ndarray) -> np.ndarray:
    """Reads cam2's camera-to-world poses from the cam0 rows of `poses/NN.txt`.

    Args:
        poses_path: The sequence's poses file.
        offset: The cam0-to-cam2 offset from `cam0_to_cam2_offset`.

    Returns:
        (N, 4, 4) camera-to-world poses of cam2.

    Raises:
        ValueError: If a row does not have 12 values.
    """
    rows = np.loadtxt(poses_path, ndmin=2)
    if rows.shape[1] != 12:
        raise ValueError(
            f"{poses_path}: expected 12 values per row, got {rows.shape[1]}")
    cam0_to_world = np.tile(np.eye(4), (len(rows), 1, 1))
    cam0_to_world[:, :3, :4] = rows.reshape(-1, 3, 4)
    cam2_to_cam0 = np.eye(4)
    cam2_to_cam0[:3, 3] = -offset
    return cam0_to_world @ cam2_to_cam0


def link_or_copy(source: Path, target: Path) -> None:
    """Hard-links `source` to `target` (same file system), else copies it."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _check_frames(sequence: str, frames: list[Path], poses: np.ndarray,
                  timestamps: np.ndarray) -> None:
    """Checks that frames, poses and timestamps match one to one.

    Raises:
        ValueError: If the counts differ or the frames are not numbered
            consecutively from 000000.png.
    """
    if not len(frames) == len(poses) == len(timestamps):
        raise ValueError(f"{sequence}: {len(frames)} frames, {len(poses)} "
                         f"poses, {len(timestamps)} timestamps")
    expected_names = [f"{index:06d}.png" for index in range(len(frames))]
    if [path.name for path in frames] != expected_names:
        raise ValueError(f"{sequence}: image_2 is not numbered 000000.png .. "
                         f"{len(frames) - 1:06d}.png")


def materialize_sequence(raw_root: Path, sequence: str, output_root: Path,
                         overwrite: bool) -> dict:
    """Links one sequence's frames and writes its cam2 GT COLMAP model.

    An existing sequence is reused unless `overwrite` is set.

    Args:
        raw_root: The downloaded KITTI odometry release.
        sequence: Sequence name, e.g. "00".
        output_root: The output dataset folder.
        overwrite: Whether to rewrite an existing sequence.

    Returns:
        The scene record (also written to `scene.json`).
    """
    from PIL import Image  # pylint: disable=import-outside-toplevel

    source = raw_root / "dataset" / "sequences" / sequence
    out_scene = output_root / sequence
    sparse_dir = out_scene / "sparse" / "0"
    if sparse_dir.is_dir() and not overwrite:
        return json.loads(
            (out_scene / "scene.json").read_text(encoding="utf-8"))
    frames = sorted(path for path in (source / "image_2").iterdir()
                    if path.suffix.lower() == ".png")
    projection = read_projection(source / "calib.txt")
    poses = read_cam2_poses(raw_root / "dataset" / "poses" / f"{sequence}.txt",
                            cam0_to_cam2_offset(projection))
    timestamps = np.loadtxt(source / "times.txt", ndmin=1)
    _check_frames(sequence, frames, poses, timestamps)
    with Image.open(frames[0]) as first:
        width, height = first.size
    camera = read_write_model.Camera(id=1,
                                     model="PINHOLE",
                                     width=width,
                                     height=height,
                                     params=np.array([
                                         projection[0, 0], projection[1, 1],
                                         projection[0, 2], projection[1, 2]
                                     ]))

    if out_scene.exists():
        shutil.rmtree(out_scene)
    (out_scene / "images").mkdir(parents=True)
    images: dict[int, read_write_model.BaseImage] = {}
    for index, path in enumerate(frames):
        link_or_copy(path, out_scene / "images" / path.name)
        world_to_camera = np.linalg.inv(poses[index])
        images[index + 1] = read_write_model.BaseImage(
            id=index + 1,
            qvec=read_write_model.rotmat2qvec(world_to_camera[:3, :3]),
            tvec=world_to_camera[:3, 3],
            camera_id=1,
            name=path.name,
            xys=np.empty((0, 2)),
            point3D_ids=np.empty((0,), int))
    sparse_dir.mkdir(parents=True)
    read_write_model.write_model({1: camera},
                                 images, {},
                                 str(sparse_dir),
                                 ext=".txt")
    trajectory_length = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0),
                                       axis=1).sum()
    record = {
        "dataset_name": "kitti_odometry",
        "scene_name": sequence,
        "root_directory": str(out_scene),
        "total_images": len(images),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "trajectory_length_m": float(trajectory_length),
        "camera_model": "PINHOLE",
        "width": width,
        "height": height,
    }
    (out_scene / "scene.json").write_text(json.dumps(record, indent=2) + "\n",
                                          encoding="utf-8")
    return record


def main() -> None:
    """Materializes the selected sequences in parallel.

    The dataset metadata is written after all sequences.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_root",
                        type=Path,
                        required=True,
                        help="scripts/download/kitti_odometry.sh output.")
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="*", default=list(SEQUENCES))
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_root = args.benchmark_root / "SLAM_source" / "kitti_odometry"
    output_root.mkdir(parents=True, exist_ok=True)
    count = len(args.sequences)
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(
            pool.map(materialize_sequence, [args.raw_root] * count,
                     args.sequences, [output_root] * count,
                     [args.overwrite] * count))
    total = sum(record["total_images"] for record in records)
    payload = {
        "dataset_name": "kitti_odometry",
        "root_directory": str(output_root),
        "total_images": total,
        "average_images_per_scene": total / len(records) if records else 0.0,
        "protocol_reference":
            ("KITTI odometry 00-10, image_2 (LoGeR eval/run_kitti.sh, "
             "lingbot-map datasets/kitti.py); "
             "GT moved from cam0 to cam2 through P2"),
        "license": "CC BY-NC-SA 3.0 (KITTI)",
        "scenes": sorted(records, key=lambda record: record["scene_name"]),
    }
    (output_root / "metadata.json").write_text(json.dumps(payload, indent=2) +
                                               "\n",
                                               encoding="utf-8")
    summary_keys = ("scene_name", "total_images", "width", "height",
                    "trajectory_length_m")
    for record in records:
        print(json.dumps({key: record[key] for key in summary_keys}))


if __name__ == "__main__":
    main()
