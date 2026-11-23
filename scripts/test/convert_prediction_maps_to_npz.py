#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Converts existing per-frame prediction maps from .npy to compressed .npz.

The conversion happens in place. New predictions are written as `<stem>.npz`
(`adapter_utils.save_prediction_map`: depth float32, confidence float16,
`np.savez_compressed`). This script converts the `.npy` maps of older and
imported predictions the same way: every `<stem>.npy` in a `depths/`,
`confidences/`, `raw_disparities/` or `raw_log_depths/` folder below an
`artifacts/` folder (the prediction trees; benchmark ground truth, which also
has `depths/` folders, is not touched). Each new file is read back and compared
with the source before the `.npy` is deleted; hard-linked copies (releases
share identical files between batches) are converted once and the `.npz`
hard-linked in their place. A frame that already has a `.npz` keeps it when it
holds the same values, and both files are kept (and reported) when they
differ. Safe to rerun: converted frames have no `.npy` left.

    python scripts/test/convert_prediction_maps_to_npz.py ROOT [ROOT ...] \\
        [--workers 8] [--dry-run]
"""

from __future__ import annotations

import argparse
import collections
from concurrent import futures
import os
import pathlib
import sys

import numpy as np

MAP_DIRS = {
    "depths": ("depth", np.float32),
    "raw_disparities": ("depth", np.float32),
    "raw_log_depths": ("depth", np.float32),
    "confidences": ("confidence", np.float16)
}
FLOAT16_MAX = float(np.finfo(np.float16).max)


def map_files(roots: list[pathlib.Path]) -> list[pathlib.Path]:
    """Lists every .npy of a prediction-map folder below an artifacts/ folder.

    Args:
        roots: The folders to search.

    Returns:
        The .npy paths.
    """
    found = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            path = pathlib.Path(dirpath)
            if path.name in MAP_DIRS and "artifacts" in path.parts:
                found.extend(
                    path / name for name in filenames if name.endswith(".npy"))
                dirnames[:] = []
    return found


def target_array(source: np.ndarray, dtype) -> np.ndarray:
    """Casts a map to the stored dtype, clipping to the float16 range first."""
    values = np.asarray(source, dtype=np.float32)
    if np.dtype(dtype) == np.float16:
        values = np.clip(values, -FLOAT16_MAX, FLOAT16_MAX)
    return values.astype(dtype)


def read_npz(path: pathlib.Path) -> np.ndarray:
    """Returns the first array of an .npz file."""
    with np.load(path) as data:
        return data[data.files[0]]


def convert_group(paths: list[str]) -> tuple[str, int, int, list[str]]:
    """Converts one set of hard-linked .npy files.

    Args:
        paths: The hard-linked .npy paths of one file.

    Returns:
        The status ("converted", "conflict", "skipped", "unreadable" or
        "failed"), the size in bytes before and after, and the messages to
        print.
    """
    first = pathlib.Path(paths[0])
    key, dtype = MAP_DIRS[first.parent.name]
    try:
        source = np.load(first, allow_pickle=False)
    except (ValueError, OSError) as error:
        return "unreadable", 0, 0, [f"{first}: {error}"]
    if source.dtype.kind != "f" or source.ndim not in (2, 3):
        return "skipped", 0, 0, [
            f"{first}: {source.dtype} {source.shape} is not a map"
        ]
    expected = target_array(source, dtype)
    before = first.stat().st_size
    done = None
    messages = []
    for name in paths:
        npy = pathlib.Path(name)
        npz = npy.with_suffix(".npz")
        if npz.exists():
            if not np.array_equal(read_npz(npz), expected, equal_nan=True):
                messages.append(
                    f"{npy}: differs from the existing {npz.name}; both kept")
                continue
        elif done is None:
            partial = npz.with_name(npz.name + ".partial")
            with open(partial, "wb") as handle:
                np.savez_compressed(handle, **{key: expected})
            written = read_npz(partial)
            if written.dtype != expected.dtype or not np.array_equal(
                    written, expected, equal_nan=True):
                partial.unlink()
                return "failed", 0, 0, [f"{npy}: read-back differs; .npy kept"]
            os.replace(partial, npz)
            done = npz
        else:
            os.link(done, npz)
        npy.unlink()
    if done is not None:
        after = done.stat().st_size
    else:
        after = 0
    if messages:
        status = "conflict"
    else:
        status = "converted"
    return status, before, after, messages


def main(argv: list[str] | None = None) -> int:
    """Converts the maps under the given roots.

    Args:
        argv: The command line arguments; None reads `sys.argv`.

    Returns:
        The exit status: 1 if a file failed or was unreadable, else 0.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="count the files and their size only")
    args = parser.parse_args(argv)

    files = map_files(args.roots)
    groups: dict[tuple[int, int], list[str]] = collections.defaultdict(list)
    for path in files:
        info = path.stat()
        groups[(info.st_dev, info.st_ino)].append(str(path))
    total = sum(
        pathlib.Path(paths[0]).stat().st_size for paths in groups.values())
    print(
        f"{len(files)} .npy maps ({len(groups)} distinct files, "
        f"{total / 1e9:.1f} GB) under {len(args.roots)} roots",
        flush=True)
    if args.dry_run:
        return 0
    counts: dict[str, int] = collections.defaultdict(int)
    before = after = 0
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(convert_group, groups.values(), chunksize=64)
        for index, result in enumerate(results, start=1):
            status, size_before, size_after, messages = result
            counts[status] += 1
            before += size_before
            after += size_after
            for message in messages:
                print(f"  {status}: {message}", flush=True)
            if index % 20000 == 0:
                print(
                    f"  {index}/{len(groups)} files, {before / 1e9:.1f} GB -> "
                    f"{after / 1e9:.1f} GB",
                    flush=True)
    print(
        f"done: {dict(counts)}; {before / 1e9:.1f} GB of .npy -> "
        f"{after / 1e9:.1f} GB of .npz",
        flush=True)
    if counts["failed"] or counts["unreadable"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
