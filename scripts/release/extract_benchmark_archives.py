# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Unpacks a downloaded benchmark release into a ready-to-use `benchmark_root`.

The Hugging Face release (`Bluefish2333/SpatialArena-Dataset`) stores each dataset scene
as one uncompressed `<Task>/<dataset>/<scene>.tar` (the raw tree is ~900k
files, beyond what the Hub handles well). Run this once after downloading; it
extracts every archive next to itself, restoring the exact layout the
evaluator expects (relative symlinks are preserved, within a dataset and from
`SLAM/7scenes` into `MVRecon/7scenes`), and with `--delete` removes each
archive after it extracts.

    hf download Bluefish2333/SpatialArena-Dataset --repo-type dataset \\
        --local-dir /path/to/benchmark_root
    python extract_benchmark_archives.py /path/to/benchmark_root --delete

Standard library only, so it can run from the downloaded folder without this
repo installed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tarfile


def _safe_members(archive: tarfile.TarFile, dest: pathlib.Path,
                  root: pathlib.Path):
    """Yields the members that stay inside `dest`.

    Members may not use absolute paths or `..` escapes. Symlinks may point
    anywhere inside the release `root` (SLAM/7scenes links into
    MVRecon/7scenes), never outside it.

    Raises:
        ValueError: If a member or link would land outside `dest` or `root`.
    """
    dest = dest.resolve()
    root = root.resolve()
    for member in archive.getmembers():
        target = (dest / member.name).resolve()
        if dest != target and dest not in target.parents:
            raise ValueError(
                f"Refusing to extract {member.name!r} outside {dest}")
        if member.issym() or member.islnk():
            if pathlib.Path(member.linkname).is_absolute():
                raise ValueError(f"Refusing absolute link {member.name!r} -> "
                                 f"{member.linkname!r}")
            if member.issym():
                link_target = (target.parent / member.linkname).resolve()
            else:
                link_target = dest / member.linkname
            if root not in pathlib.Path(link_target).resolve().parents:
                raise ValueError(
                    f"Refusing link {member.name!r} -> {member.linkname!r} "
                    f"pointing outside {root}")
        yield member


def extract_all(root: pathlib.Path, delete: bool = False) -> int:
    """Extracts every scene archive under `root` next to itself.

    An archive whose `.<scene>.extracted` marker exists is skipped, so an
    interrupted run can be resumed.

    Args:
        root: The folder the release was downloaded into.
        delete: Whether to delete each archive after it is extracted.

    Returns:
        The number of archives found.
    """
    archives = sorted(
        path for path in root.rglob("*.tar") if ".cache" not in path.parts)
    for index, path in enumerate(archives, start=1):
        dest = path.parent
        done_marker = dest / f".{path.stem}.extracted"
        if done_marker.exists():
            print(f"[{index}/{len(archives)}] skip (already extracted): "
                  f"{path.relative_to(root)}")
        else:
            print(
                f"[{index}/{len(archives)}] extracting "
                f"{path.relative_to(root)}",
                flush=True)
            with tarfile.open(path) as archive:
                archive.extractall(dest,
                                   members=_safe_members(archive, dest, root))
            done_marker.touch()
        if delete:
            path.unlink()
    return len(archives)


def main() -> None:
    """Extracts the release folder given on the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "root",
        type=pathlib.Path,
        help="Folder the release was downloaded into (becomes benchmark_root).")
    parser.add_argument("--delete",
                        action="store_true",
                        help="Delete each .tar after it is extracted.")
    args = parser.parse_args()
    if not args.root.is_dir():
        sys.exit(f"{args.root} is not a directory")
    count = extract_all(args.root, delete=args.delete)
    print(f"Done: {count} archives under {args.root}")


if __name__ == "__main__":
    main()
