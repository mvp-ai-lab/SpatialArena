# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Download SpatialArena tracks from the benchmark's Hugging Face release into a benchmark root.

SpatialArena (docs/benchmark/spatialarena.md) is our own capture, so the release is its only source: each track is
``<Task>/spatialarena/`` in the dataset repo, one uncompressed ``<scene>.tar`` per scene plus its metadata.json. This
fetches the selected tracks into ``<benchmark_root>/<Task>/spatialarena`` and unpacks their scene archives.

    python scripts/download/spatialarena.py --benchmark_root /path/to/benchmarks --tasks VideoDepth MVRecon
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena.spatialarena import SPATIALARENA, TASKS  # noqa: E402
from scripts.release.extract_benchmark_archives import extract_all  # noqa: E402

# pylint: enable=wrong-import-position

DEFAULT_REPO_ID = "Bluefish2333/benchmark"


def download_tracks(benchmark_root: Path, tasks: list[str], repo_id: str = DEFAULT_REPO_ID) -> list[Path]:
    """Fetch ``<Task>/spatialarena/`` of each task and unpack it; returns the track folders."""
    from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel

    patterns = [f"{task}/{SPATIALARENA}/*" for task in tasks]
    snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=patterns, local_dir=str(benchmark_root))
    folders = [benchmark_root / task / SPATIALARENA for task in tasks]
    for folder in folders:
        if not (folder / "metadata.json").is_file():
            raise FileNotFoundError(f"{repo_id} has no {folder.relative_to(benchmark_root)}/metadata.json")
        extract_all(folder, delete=True)
    return folders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    args = parser.parse_args()
    for folder in download_tracks(args.benchmark_root.expanduser(), args.tasks, args.repo_id):
        print(f"Ready: {folder}")


if __name__ == "__main__":
    main()
