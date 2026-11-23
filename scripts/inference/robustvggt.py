# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run RobustVGGT through its VGGT-compatible model implementation."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

# Shared method checkouts (the engine's METHOD_CODE_ROOT); the engine also passes the *_CODE_PATH below.
# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(
    os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
METHOD_CODE_ROOT = Path(
    os.environ.get("METHOD_CODE_ROOT",
                   f"{SPATIALARENA_ROOT}/benchmarks/_model_codes"))


def parse_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path",
                        default=os.environ.get("ROBUSTVGGT_CKPT_PATH",
                                               "facebook/VGGT-1B"))
    parser.add_argument(
        "--robustvggt_code_path",
        type=Path,
        default=Path(os.environ["ROBUSTVGGT_CODE_PATH"])
        if "ROBUSTVGGT_CODE_PATH" in os.environ else METHOD_CODE_ROOT /
        "RobustVGGT",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_wrapper_args()
    target = Path(__file__).with_name("vggt.py")
    forwarded: list[str] = []
    skip_next = False
    for index, value in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if value == "--robustvggt_code_path":
            skip_next = True
            continue
        if value.startswith("--robustvggt_code_path="):
            continue
        forwarded.append(value)
    forwarded.extend([
        "--ckpt_path", args.ckpt_path, "--vggt_code_path",
        str(args.robustvggt_code_path)
    ])
    sys.argv = [str(target), *forwarded]
    runpy.run_path(str(target), run_name="__main__")
    metrics_path = args.output_path / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["method"] = "ROBUSTVGGT"
        metrics["adapter"] = "robustvggt.py"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                                encoding="utf-8")


if __name__ == "__main__":
    main()
