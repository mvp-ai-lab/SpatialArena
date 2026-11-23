# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Downloads SpatialArena tracks from the Hugging Face release.

SpatialArena (docs/benchmark/spatialarena.md) is our own capture, so the
release is its only source: each track is `<Task>/spatialarena/` in the dataset
repo, one uncompressed `<scene>.tar` per scene plus its metadata.json. This
script fetches the selected tracks into `<benchmark_root>/<Task>/spatialarena`
and unpacks their scene archives.

    python scripts/download/spatialarena.py \\
        --benchmark_root /path/to/benchmarks --tasks VideoDepth MVRecon
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from scripts.release import extract_benchmark_archives  # noqa: E402
from spatialarena import spatialarena  # noqa: E402

# pylint: enable=wrong-import-position

DEFAULT_REPO_ID = "Bluefish2333/SpatialArena-Dataset"


def download_tracks(benchmark_root: Path,
                    tasks: list[str],
                    repo_id: str = DEFAULT_REPO_ID) -> list[Path]:
    """Fetches `<Task>/spatialarena/` of each task and unpacks it.

    Args:
        benchmark_root: Benchmark root the tracks are downloaded into.
        tasks: Task names, e.g. "VideoDepth".
        repo_id: Hugging Face dataset repo of the release.

    Returns:
        The track folders, one per task.

    Raises:
        FileNotFoundError: If the release has no metadata.json for a track.
    """
    # pylint: disable-next=import-outside-toplevel
    import huggingface_hub

    patterns = [f"{task}/{spatialarena.SPATIALARENA}/*" for task in tasks]
    huggingface_hub.snapshot_download(repo_id=repo_id,
                                      repo_type="dataset",
                                      allow_patterns=patterns,
                                      local_dir=str(benchmark_root))
    folders = [
        benchmark_root / task / spatialarena.SPATIALARENA for task in tasks
    ]
    for folder in folders:
        if not (folder / "metadata.json").is_file():
            relative = folder.relative_to(benchmark_root)
            raise FileNotFoundError(
                f"{repo_id} has no {relative}/metadata.json")
        extract_benchmark_archives.extract_all(folder, delete=True)
    return folders


def main() -> None:
    """Downloads and unpacks the tracks selected on the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--tasks",
                        nargs="+",
                        choices=spatialarena.TASKS,
                        default=list(spatialarena.TASKS))
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    args = parser.parse_args()
    folders = download_tracks(args.benchmark_root.expanduser(), args.tasks,
                              args.repo_id)
    for folder in folders:
        print(f"Ready: {folder}")


if __name__ == "__main__":
    main()
