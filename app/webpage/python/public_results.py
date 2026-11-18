#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Strip a dashboard payload (results.json, results_live.json) down to what the public page shows.

The collector records where it read the results from (``collection``, ``source_report_roots``, ``source_tsv``: report
folders on the machine that collected them) and the registry's coverage matrix (``coverage``: every method's finished,
failed and missing cases). The public dashboard shows neither, and the paths name internal machines, so the files
committed with it are stripped:

    python app/webpage/python/public_results.py app/webpage/data/results.json \\
        app/webpage/data/results_live.json

``collect_benchmark_metrics.py --public`` writes its output stripped the same way. The check at the end refuses to
leave an absolute or home-relative path anywhere in the payload.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

INTERNAL_KEYS = ("collection", "source_report_roots", "source_tsv", "coverage")
PATH_PATTERN = re.compile(r"(^|[\s\"'(=])(/(mnt|home|tmp|data|root|Users|scratch|lustre|gpfs)/|~/)")


def strip_internal(payload: dict) -> dict:
    """``payload`` without the collection record, source paths and coverage matrix."""
    return {key: value for key, value in payload.items() if key not in INTERNAL_KEYS}


def path_fields(value, where: str = "$"):
    """(location, string) of every string in ``value`` that contains a filesystem path."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from path_fields(item, f"{where}.{key}")
    elif isinstance(value, list):
        for item in value:
            yield from path_fields(item, f"{where}[]")
    elif isinstance(value, str) and PATH_PATTERN.search(value):
        yield where, value


def make_public(path: Path) -> None:
    payload = strip_internal(json.loads(path.read_text(encoding="utf-8")))
    leaks = list(path_fields(payload))
    if leaks:
        shown = "; ".join(f"{where} = {value[:80]}" for where, value in leaks[:5])
        raise SystemExit(f"{path}: {len(leaks)} fields still hold paths ({shown})")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{path}: public ({', '.join(key for key in INTERNAL_KEYS)} removed)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    for path in parser.parse_args(argv).files:
        make_public(path)


if __name__ == "__main__":
    main()
