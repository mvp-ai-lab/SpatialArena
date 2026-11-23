#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Materialize MoGe's HAMMER, iBims-1 and Spring sets as MonoDepth datasets.

Source: HF dataset Ruicheng/monocular-geometry-evaluation
(scripts/download/moge_eval.sh), the exact processed evaluation frames MoGe /
MoGe-2 / MoGe-3 report on. Each MoGe sample folder (`image.jpg`, `depth.png`,
`meta.json`) becomes one frame of the single scene
`MonoDepth/<dataset>/<dataset>`:

- the image is copied unchanged;
- depth.png is decoded (MoGe's log encoding) and MoGe's own evaluation validity
  is baked in: unknown / infinite depth and depth beyond 1000x the frame's 1%
  depth quantile (MoGe's `drop_max_depth`) are stored as 0;
- meta.json's UV-normalized intrinsics are scaled to pixels.

Every frame of MoGe's `.index.txt` is used; the list is committed as
`spatialarena/eval/monodepth_splits/<dataset>/<dataset>.txt`.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess import monodepth_layout  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402

DEFAULT_BENCHMARK_ROOT = dataset_registry.DEFAULT_BENCHMARK_ROOT
DEFAULT_SOURCE_ROOT = DEFAULT_BENCHMARK_ROOT / "MonoDepth_source" / "moge_eval"
MOGE_EVAL_REPO = "Ruicheng/monocular-geometry-evaluation"
MOGE_EVAL_REVISION = "574610d77c0cc14d911a1ff46ec529ba934c8960"
DROP_MAX_DEPTH = 1000.0

# Our dataset name -> MoGe folder name, scene limits and a provenance note.
DATASETS = {
    "hammer": {
        "moge_name":
            "HAMMER",
        "limits":
            monodepth_layout.SceneLimits(min_depth=1e-3, max_depth=None),
        "note": ("HAMMER test scenes (with and without objects) as processed "
                 "by MoGe; metric depth."),
    },
    "ibims1": {
        "moge_name": "iBims-1",
        "limits": monodepth_layout.SceneLimits(min_depth=1e-3, max_depth=None),
        "note": "All 100 iBims-1 images as processed by MoGe; metric depth.",
    },
    "spring": {
        "moge_name":
            "Spring",
        # Spring depth is not metric (median ~0.24 scene units, 1% quantile
        # ~0.003): no absolute near cut-off.
        "limits":
            monodepth_layout.SceneLimits(min_depth=1e-6, max_depth=None),
        "note": ("Every 5th frame of the Spring train split (1000 of 5000) as "
                 "selected by MoGe; depth is not metric."),
    },
}


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--benchmark_root",
                        type=Path,
                        default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--datasets",
                        nargs="+",
                        choices=sorted(DATASETS),
                        default=sorted(DATASETS))
    parser.add_argument("--max_samples",
                        type=int,
                        default=-1,
                        help="Materialize only the first N frames (testing).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write_split",
        action="store_true",
        help=("(Re)write the committed frame list from the source index "
              "instead of requiring it to match."),
    )
    return parser.parse_args()


def read_moge_index(source_dir: Path) -> list[str]:
    """Returns the sorted sample folders in a MoGe dataset's .index.txt.

    Args:
        source_dir: The MoGe dataset folder.

    Returns:
        The sample folder names.

    Raises:
        FileNotFoundError: If the folder has no .index.txt.
    """
    index_path = source_dir / ".index.txt"
    if not index_path.is_file():
        raise FileNotFoundError(f"MoGe index not found: {index_path}")
    return sorted(
        line.strip()
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip())


def materialize_frame(source_dir: Path, scene_root: Path, sample: str,
                      overwrite: bool) -> dict:
    """Copies one MoGe sample into the scene.

    Args:
        source_dir: The MoGe dataset folder.
        scene_root: The output scene directory.
        sample: The sample folder name.
        overwrite: Rewrite an existing image or depth map.

    Returns:
        The frame's manifest record as a dict.

    Raises:
        ValueError: If the depth map and image sizes differ.
    """
    sample_dir = source_dir / sample
    key = monodepth_layout.frame_key(sample)
    image_rel = f"images/{key}.jpg"
    depth_rel = f"depths/{key}.npy"
    image_path = scene_root / image_rel
    depth_path = scene_root / depth_rel

    with Image.open(sample_dir / "image.jpg") as image:
        width, height = image.size
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    intrinsics = monodepth_layout.normalized_to_pixel_intrinsics(
        np.asarray(meta["intrinsics"]), width, height)

    if overwrite or not image_path.is_file():
        monodepth_layout.write_bytes(image_path,
                                     (sample_dir / "image.jpg").read_bytes())
    num_valid = None
    if overwrite or not depth_path.is_file():
        depth = monodepth_layout.decode_moge_depth(
            (sample_dir / "depth.png").read_bytes())
        if depth.shape != (height, width):
            raise ValueError(f"{sample}: depth {depth.shape} does not match "
                             f"image {(height, width)}")
        valid = monodepth_layout.drop_far_depth(depth, DROP_MAX_DEPTH)
        monodepth_layout.save_depth(depth_path, depth, valid)
        num_valid = int(valid.sum())
    extra = {}
    if num_valid is not None:
        extra["num_valid_pixels"] = num_valid
    return monodepth_layout.FrameRecord(
        key=key,
        image=image_rel,
        depth=depth_rel,
        source=sample,
        intrinsics=intrinsics,
        extra=extra,
    ).as_dict()


def materialize_dataset(
    *,
    dataset: str,
    source_root: Path,
    benchmark_root: Path,
    max_samples: int,
    workers: int,
    overwrite: bool,
    update_split: bool,
) -> dict:
    """Materializes one MoGe dataset as a single-scene MonoDepth dataset.

    Args:
        dataset: Our dataset name, a key of DATASETS.
        source_root: The downloaded MoGe evaluation sets.
        benchmark_root: The benchmark root to write into.
        max_samples: Materialize only the first N frames; 0 or less for all.
        workers: The number of worker processes.
        overwrite: Rewrite existing frames.
        update_split: Rewrite the committed frame list from the source index.

    Returns:
        The dataset metadata.

    Raises:
        ValueError: If the source index differs from the committed frame list
            and `update_split` is False.
    """
    config = DATASETS[dataset]
    source_dir = source_root / config["moge_name"]
    samples = read_moge_index(source_dir)
    committed = monodepth_layout.read_split(dataset, dataset)
    if update_split or committed is None:
        monodepth_layout.write_split(dataset, dataset, samples)
    elif committed != samples:
        raise ValueError(
            f"{dataset}: MoGe index ({len(samples)} frames) differs from the "
            f"committed frame list ({len(committed)} frames); rerun with "
            "--write_split to accept the new source")
    if max_samples > 0:
        samples = samples[:max_samples]

    dataset_root = benchmark_root / "MonoDepth" / dataset
    scene_root = dataset_root / dataset
    with futures.ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(
            pool.map(
                materialize_frame,
                [source_dir] * len(samples),
                [scene_root] * len(samples),
                samples,
                [overwrite] * len(samples),
                chunksize=8,
            ))
    frame_records = [
        monodepth_layout.FrameRecord(
            key=record.pop("key"),
            image=record.pop("image"),
            depth=record.pop("depth"),
            source=record.pop("source"),
            intrinsics=record.pop("intrinsics"),
            extra=record,
        ) for record in records
    ]
    provenance = {
        "source": f"https://huggingface.co/datasets/{MOGE_EVAL_REPO}",
        "revision": MOGE_EVAL_REVISION,
        "source_folder": config["moge_name"],
        "frame_list":
            (f"spatialarena/eval/monodepth_splits/{dataset}/{dataset}.txt "
             "(MoGe .index.txt)"),
        "depth_validity":
            (f"finite, > 0 and <= {DROP_MAX_DEPTH:g} x the frame's 1% depth "
             "quantile (MoGe)"),
        "note": config["note"],
    }
    entry = monodepth_layout.write_scene(
        dataset_root=dataset_root,
        dataset=dataset,
        scene=dataset,
        records=frame_records,
        limits=config["limits"],
        provenance=provenance,
    )
    return monodepth_layout.write_dataset_metadata(dataset_root, dataset,
                                                   [entry])


def main() -> None:
    """Materializes the selected datasets."""
    args = parse_args()
    for dataset in args.datasets:
        metadata = materialize_dataset(
            dataset=dataset,
            source_root=args.source_root.expanduser(),
            benchmark_root=args.benchmark_root.expanduser(),
            max_samples=args.max_samples,
            workers=args.workers,
            overwrite=args.overwrite,
            update_split=args.write_split,
        )
        print(f"{dataset}: {metadata['total_images']} frames in "
              f"{len(metadata['scenes'])} scene(s)")


if __name__ == "__main__":
    main()
