# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Materialize The Oxford Spires Dataset (Tao et al., 2024; HF ori-drs/oxford_spires_dataset, CC BY-NC-SA 4.0) into
benchmark_root/SLAM_source/oxford_spires, following LingBot-Map's evaluation preprocessing exactly.

The reference is ``preprocess/oxford.py`` of github.com/robbyant/lingbot-map (pinned in
spatialarena/download.py::METHOD_CODE_COMMITS), run with ``--images_only`` as its benchmark does for camera
poses. Its functions are imported from that checkout, not re-implemented, so frames and poses match it:

- the ten sequences of its ``PROCESS_SCENE`` (its christ-church-05 entry is dated 2024-03-18, but the dataset's
  sequence is ``2024-03-20-christ-church-05``; that one is used);
- camera 0 (``raw/images.zip`` ``cam0/<timestamp>.jpg``), the first ``--max_frames`` = 3840 frames in timestamp order
  (its default), each matched to the nearest ``processed/trajectory/gt-tum.txt`` pose within 0.1 s (frames without
  one are dropped, as there);
- fisheye undistortion with ``ImageRectifier`` (OPENCV_FISHEYE, balance 0, principal point at the image center) and
  the top-level ``calibration/cam0.yaml``; rectified images saved as PNG;
- camera-to-world ``C2W = T_WB(t) @ T_base_cam0`` with ``T_base_cam0`` from ``calibration/cam-lidar-imu.yaml`` and
  its fixed base-lidar transform.

Output per sequence (scene name = sequence name without the date, as LingBot-Map's): ``images/NNNNNN.png``,
``sparse/0`` (PINHOLE rectified intrinsics, GT poses; text COLMAP model) and ``frames.tsv`` (image name, camera
timestamp, matched GT timestamp). The SLAM range tracks feed these frames at the dataset's protocol strides
(spatialarena/slam_protocol.py).
"""

from __future__ import annotations

import os
import argparse
import json
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena.geometry.read_write_model import BaseImage, Camera, rotmat2qvec, write_model  # noqa: E402
# pylint: enable=wrong-import-position

# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
DEFAULT_LINGBOT_CODE = Path(f"{SPATIALARENA_ROOT}/benchmarks/_model_codes/lingbot-map")
SEQUENCES = (
    "2024-03-12-keble-college-02",
    "2024-03-12-keble-college-03",
    "2024-03-12-keble-college-04",
    "2024-03-12-keble-college-05",
    "2024-03-13-observatory-quarter-01",
    "2024-03-13-observatory-quarter-02",
    "2024-03-18-christ-church-02",
    "2024-03-18-christ-church-03",
    "2024-03-20-christ-church-05",
    "2024-05-20-bodleian-library-02",
)
MAX_FRAMES = 3840  # lingbot-map preprocess/oxford.py --max_frames default
MAX_TIME_GAP = 0.1  # seconds, its MAX_TIME_GAP


def _lingbot_preprocess(code_path: Path):
    """LingBot-Map's ``preprocess/oxford.py`` module from its checkout (importable only once it joins sys.path)."""
    sys.path.insert(0, str(code_path / "preprocess"))
    import oxford  # pylint: disable=import-outside-toplevel  # lingbot-map preprocess/oxford.py

    return oxford


def scene_name(sequence: str) -> str:
    """Sequence name without its date prefix, as LingBot-Map names its scenes."""
    return re.sub(r"^\d{4}-\d{2}-\d{2}-", "", sequence)


def materialize_sequence(raw_root: Path, sequence: str, output_root: Path, code_path: Path, max_frames: int,
                         overwrite: bool) -> dict:
    """Write one sequence's rectified images, GT COLMAP model and frame table; returns its scene record."""
    import cv2  # pylint: disable=import-outside-toplevel  # only the worker processes need OpenCV

    ref = _lingbot_preprocess(code_path)
    seq_dir = raw_root / "sequences" / sequence
    calib_dir = raw_root / "calibration"
    out_scene = output_root / scene_name(sequence)
    sparse_dir = out_scene / "sparse" / "0"
    if sparse_dir.is_dir() and not overwrite:
        return json.loads((out_scene / "scene.json").read_text(encoding="utf-8"))

    intrinsics, dist, width, height = ref.load_cam0_intrinsics(calib_dir / "cam0.yaml")
    t_cam0_lidar = ref.load_T_cam0_lidar(calib_dir / "cam-lidar-imu.yaml")
    t_base_cam0 = np.linalg.inv(t_cam0_lidar @ np.linalg.inv(ref.build_T_base_lidar()))
    rectifier = ref.ImageRectifier(intrinsics, dist, width, height)
    traj_ts, traj_lines = ref.load_tum(seq_dir / "processed" / "trajectory" / "gt-tum.txt")

    (out_scene / "images").mkdir(parents=True, exist_ok=True)
    camera = Camera(id=1, model="PINHOLE", width=width, height=height,
                    params=np.array([rectifier.fx, rectifier.fy, rectifier.cx, rectifier.cy], dtype=np.float64))
    images: dict[int, BaseImage] = {}
    rows = ["name\tcamera_timestamp\tgt_timestamp"]
    skipped_pose = skipped_decode = 0
    with zipfile.ZipFile(seq_dir / "raw" / "images.zip") as archive:
        cam0 = sorted(n for n in archive.namelist() if n.startswith("cam0/") and n.lower().endswith(".jpg"))
        available = len(cam0)
        cam0 = cam0[:max_frames] if max_frames > 0 else cam0
        for name in cam0:
            stem = Path(name).stem
            index, _gap = ref.find_nearest(float(stem), traj_ts, MAX_TIME_GAP)
            if index is None:
                skipped_pose += 1
                continue
            image = cv2.imdecode(np.frombuffer(archive.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                skipped_decode += 1
                continue
            frame = len(images)
            image_name = f"{frame:06d}.png"
            cv2.imwrite(str(out_scene / "images" / image_name), rectifier.rectify(image))
            c2w = ref.tum_line_to_T_WB(traj_lines[index]) @ t_base_cam0
            w2c = np.linalg.inv(c2w)
            images[frame + 1] = BaseImage(id=frame + 1, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=1,
                                          name=image_name, xys=np.empty((0, 2)), point3D_ids=np.empty((0,), int))
            rows.append(f"{image_name}\t{stem}\t{traj_lines[index].split()[0]}")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    write_model({1: camera}, images, {}, str(sparse_dir), ext=".txt")
    (out_scene / "frames.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    record = {
        "dataset_name": "oxford_spires",
        "scene_name": scene_name(sequence),
        "source_sequence": sequence,
        "root_directory": str(out_scene),
        "total_images": len(images),
        "cam0_frames_available": available,
        "cam0_frames_considered": len(cam0),
        "skipped_no_pose": skipped_pose,
        "skipped_decode": skipped_decode,
        "camera_model": "PINHOLE",
        "width": width,
        "height": height,
    }
    (out_scene / "scene.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def write_metadata(output_root: Path, scenes: list[dict]) -> None:
    """Dataset-level ``metadata.json`` over the materialized scenes."""
    total = sum(scene["total_images"] for scene in scenes)
    payload = {
        "dataset_name": "oxford_spires",
        "root_directory": str(output_root),
        "total_images": total,
        "average_images_per_scene": total / len(scenes) if scenes else 0.0,
        "frame_rate_hz": 20.0,
        "protocol_reference": "lingbot-map preprocess/oxford.py --images_only (first 3840 cam0 frames, GT <= 0.1 s)",
        "license": "CC BY-NC-SA 4.0 (ori-drs/oxford_spires_dataset)",
        "scenes": sorted(scenes, key=lambda scene: scene["scene_name"]),
    }
    (output_root / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Materialize the selected sequences in parallel and write the dataset metadata."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_root", type=Path, required=True, help="scripts/download/oxford_spires.sh output.")
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--lingbot_code_path", type=Path, default=DEFAULT_LINGBOT_CODE)
    parser.add_argument("--sequences", nargs="*", default=list(SEQUENCES))
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_root = args.benchmark_root / "SLAM_source" / "oxford_spires"
    output_root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(materialize_sequence, args.raw_root, sequence, output_root, args.lingbot_code_path,
                               args.max_frames, args.overwrite) for sequence in args.sequences]
        records = [future.result() for future in futures]
    for record in records:
        print(json.dumps({k: record[k] for k in ("scene_name", "total_images", "cam0_frames_available",
                                                  "skipped_no_pose")}))
    write_metadata(output_root, records)


if __name__ == "__main__":
    main()
