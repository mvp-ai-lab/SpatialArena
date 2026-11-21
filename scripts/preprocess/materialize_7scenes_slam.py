# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Materialize the 7-Scenes test sequences as per-sequence SLAM cases, as symlinks into MVRecon/7scenes.

The canonical 7-Scenes scene folders (``MVRecon/7scenes/<scene>``) mix every recording of a scene (train and test,
``seq-01`` ... ``seq-NN``) in one ``images/`` folder and one ``sparse/0`` model. A SLAM case must be one continuous
recording, so each sequence listed in the scene's official ``TestSplit.txt`` gets its own case folder
(see docs/benchmark/slam_protocol.md):

    benchmark_root/SLAM_source/7scenes/<scene>_seq-NN/
        images/seq-NN_frame-XXXXXX.color.png       # relative symlink -> ../../../../MVRecon/7scenes/<scene>/images/...
        depths/seq-NN_frame-XXXXXX.depth.proj.png  # relative symlink, same pairing as MVRecon/7scenes
        sparse/0/{cameras,images,points3D}.txt     # the sequence's poses from the scene model (written; a few KB)
    benchmark_root/SLAM_source/7scenes/metadata.json

Images and depths are not duplicated: the links are relative, so they stay valid when ``benchmark_root`` is moved or
re-extracted from the release, as long as ``MVRecon/7scenes`` sits next to ``SLAM/7scenes``. MVRecon/7scenes is only
read. 18 sequences, 17,000 frames.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.geometry.read_write_model import read_model, write_model  # noqa: E402

SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")


def test_sequences(scene_root: Path) -> list[str]:
    """``TestSplit.txt`` lines ("sequence3") as sequence prefixes ("seq-03")."""
    names = []
    for line in (scene_root / "TestSplit.txt").read_text(encoding="utf-8").split():
        match = re.fullmatch(r"sequence(\d+)", line.strip())
        if not match:
            raise ValueError(f"Unexpected TestSplit.txt entry {line!r} in {scene_root}")
        names.append(f"seq-{int(match.group(1)):02d}")
    return names


# Integer depth PNG units per metre (millimetres), read by adapter_utils.load_depth_priors.
DEPTH_PNG_SCALE = 1000.0


def link_file(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.is_symlink() or dst.exists():
        if not overwrite:
            return
        dst.unlink()
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.symlink_to(os.path.relpath(src, dst.parent))


def materialize_sequence(
    scene_root: Path, scene: str, sequence: str, model: tuple, output_root: Path, overwrite: bool, workers: int
) -> dict:
    cameras, images, _points = model
    prefix = f"{sequence}_"
    seq_images = {image_id: image for image_id, image in images.items() if image.name.startswith(prefix)}
    if not seq_images:
        raise ValueError(f"No {sequence} images in {scene_root / 'sparse' / '0'}")
    out_scene = output_root / f"{scene}_{sequence}"
    (out_scene / "images").mkdir(parents=True, exist_ok=True)
    (out_scene / "depths").mkdir(parents=True, exist_ok=True)

    depth_names = sorted(path.name for path in (scene_root / "depths").glob(f"{prefix}*"))
    jobs = [(scene_root / "images" / image.name, out_scene / "images" / image.name) for image in seq_images.values()]
    jobs += [(scene_root / "depths" / name, out_scene / "depths" / name) for name in depth_names]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda job: link_file(*job, overwrite), jobs))

    # Renumber images 1..N in frame order; keep each image's camera (7-Scenes has one shared PINHOLE camera).
    ordered = sorted(seq_images.values(), key=lambda image: image.name)
    out_images = {index: image._replace(id=index) for index, image in enumerate(ordered, start=1)}
    camera_ids = {image.camera_id for image in ordered}
    sparse_dir = out_scene / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    write_model({cid: cameras[cid] for cid in camera_ids}, out_images, {}, str(sparse_dir), ext=".txt")
    return {
        "dataset_name": "7scenes",
        "scene_name": out_scene.name,
        "source_scene": scene,
        "sequence": sequence,
        "root_directory": str(out_scene),
        "total_images": len(out_images),
        "total_depths": len(depth_names),
    }


def materialize_dataset(source_root: Path, benchmark_root: Path, overwrite: bool, workers: int) -> dict:
    output_root = benchmark_root / "SLAM_source" / "7scenes"
    records = []
    for scene in SCENES:
        scene_root = source_root / scene
        model = read_model(str(scene_root / "sparse" / "0"))
        for sequence in test_sequences(scene_root):
            record = materialize_sequence(scene_root, scene, sequence, model, output_root, overwrite, workers)
            print(f"{record['scene_name']}: {record['total_images']} images, {record['total_depths']} depths",
                  flush=True)
            records.append(record)
    total = sum(record["total_images"] for record in records)
    payload = {
        "dataset_name": "7scenes",
        "root_directory": str(output_root),
        "total_images": total,
        "average_images_per_scene": total / len(records) if records else 0.0,
        "split": "official TestSplit.txt sequences, one case per sequence",
        "storage": "images/ and depths/ are relative symlinks into MVRecon/7scenes",
        "depth_png_scale": DEPTH_PNG_SCALE,
        "scenes": records,
    }
    (output_root / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"dataset": "7scenes", "task": "SLAM", "num_scenes": len(records), "total_images": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_root", type=Path, required=True, help="Canonical 7-Scenes root (MVRecon/7scenes).")
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    result = materialize_dataset(args.source_root, args.benchmark_root, args.overwrite, args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
