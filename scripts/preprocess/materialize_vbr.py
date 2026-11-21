# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Materialize VBR: A Vision Benchmark in Rome (Brizi et al., ICRA 2024) into benchmark_root/SLAM_source/vbr.

Source: LoGeR's processed release (HF dataset Junyi42/vbr_processed, scripts/download/vbr.sh), the form both LoGeR
(eval/demo_run_longeval.sh --mode vbr) and LingBot-Map (benchmark/datasets/vbr.py) evaluate on. As extracted from the
per-sequence tarballs:

    <raw_root>/<seq>/rgb/NNNNNN.png      one LiDAR-synchronized camera frame per file; NNNNNN is the frame's index
                                         among the synchronized frames, with gaps where LoGeR's
                                         eval/datasets_preprocess/vbr_preprocess.py found no LiDAR sweep
    <raw_root>/<seq>/intrinsics.txt      3x3 K
    <raw_root>/<seq>/<seq>_gt.txt        TUM "k tx ty tz qx qy qz qw", camera-to-world (T_world_cam)
    <raw_root>/<seq>/camera_pose/NNNNNN.txt  the same pose per frame as a 4x4 matrix

The GT row index ``k`` is the frame's *position* in the sorted ``rgb/`` listing, not its file number (as LingBot-Map's
loader reads it): checked on colosseo_train0, whose files are numbered 3, 6, ..., 26441 -- row k equals
``camera_pose/<k-th file>.txt`` exactly, while matching by file number is off by up to 159 m. Every frame has a GT
pose. Images are copied unchanged (native 1388x700; each method's adapter resizes as its own evaluation does), with
a PINHOLE camera from ``intrinsics.txt`` in ``sparse/0``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena.geometry.read_write_model import BaseImage, Camera, rotmat2qvec, write_model  # noqa: E402
# pylint: enable=wrong-import-position

# LoGeR eval/run_vbr.sh sequence list (the seven training sequences with public GT).
SEQUENCES = ("campus_train0", "campus_train1", "ciampino_train1", "colosseo_train0", "diag_train0", "pincio_train0",
             "spagna_train0")


def read_gt_by_frame(path: Path) -> dict[int, np.ndarray]:
    """``k tx ty tz qx qy qz qw`` rows -> {k: 4x4 camera-to-world}; k is the position in the sorted rgb/ listing."""
    poses = {}
    for row in np.loadtxt(path, comments="#", ndmin=2):
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        c2w[:3, 3] = row[1:4]
        poses[int(round(row[0]))] = c2w
    return poses


def materialize_sequence(raw_root: Path, sequence: str, output_root: Path, overwrite: bool) -> dict:
    """Copy one sequence's frames and write its GT COLMAP model; returns its scene record."""
    from PIL import Image  # pylint: disable=import-outside-toplevel

    source = raw_root / sequence
    out_scene = output_root / sequence
    sparse_dir = out_scene / "sparse" / "0"
    if sparse_dir.is_dir() and not overwrite:
        return json.loads((out_scene / "scene.json").read_text(encoding="utf-8"))
    frames = sorted(path for path in (source / "rgb").iterdir() if path.suffix.lower() == ".png")
    gt = read_gt_by_frame(source / f"{sequence}_gt.txt")
    intrinsics = np.loadtxt(source / "intrinsics.txt")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"{source / 'intrinsics.txt'}: expected a 3x3 matrix, got {intrinsics.shape}")
    with Image.open(frames[0]) as first:
        width, height = first.size
    camera = Camera(id=1, model="PINHOLE", width=width, height=height,
                    params=np.array([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
                                    dtype=np.float64))

    (out_scene / "images").mkdir(parents=True, exist_ok=True)
    images: dict[int, BaseImage] = {}
    without_gt = []
    for index, path in enumerate(frames):
        if index not in gt:
            without_gt.append(index)
            continue
        shutil.copy2(path, out_scene / "images" / path.name)
        w2c = np.linalg.inv(gt[index])
        images[index + 1] = BaseImage(id=index + 1, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=1,
                                      name=path.name, xys=np.empty((0, 2)), point3D_ids=np.empty((0,), int))
    sparse_dir.mkdir(parents=True, exist_ok=True)
    write_model({1: camera}, images, {}, str(sparse_dir), ext=".txt")
    centers = np.stack([gt[image_id - 1][:3, 3] for image_id in sorted(images)])
    record = {
        "dataset_name": "vbr",
        "scene_name": sequence,
        "root_directory": str(out_scene),
        "total_images": len(images),
        "rgb_frames_available": len(frames),
        "frames_without_gt": len(without_gt),
        "trajectory_length_m": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()),
        "camera_model": "PINHOLE",
        "width": width,
        "height": height,
    }
    (out_scene / "scene.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    """Materialize the selected sequences in parallel and write the dataset metadata."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_root", type=Path, required=True, help="scripts/download/vbr.sh output.")
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="*", default=list(SEQUENCES))
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_root = args.benchmark_root / "SLAM_source" / "vbr"
    output_root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(materialize_sequence, [args.raw_root] * len(args.sequences), args.sequences,
                                [output_root] * len(args.sequences), [args.overwrite] * len(args.sequences)))
    total = sum(record["total_images"] for record in records)
    payload = {
        "dataset_name": "vbr",
        "root_directory": str(output_root),
        "total_images": total,
        "average_images_per_scene": total / len(records) if records else 0.0,
        "protocol_reference": "LoGeR processed VBR (HF Junyi42/vbr_processed), as read by lingbot-map datasets/vbr.py",
        "license": "no data license stated (c) RVP-Group; evaluate only, not redistributed",
        "scenes": sorted(records, key=lambda record: record["scene_name"]),
    }
    (output_root / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(json.dumps({k: record[k] for k in ("scene_name", "total_images", "rgb_frames_available",
                                                  "frames_without_gt", "trajectory_length_m")}))


if __name__ == "__main__":
    main()
