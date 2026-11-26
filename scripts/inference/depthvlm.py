#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs spatialarena.models.depthvlm.DepthVLM; the engine's adapter for it.

The same as `run_model.py --model DEPTHVLM`; see `--help` for its options.
"""

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable-next=wrong-import-position
from scripts.inference import run_model  # noqa: E402


def main(argv=None):
    """Runs DEPTHVLM on argv (default: sys.argv); see run_model.py."""
    run_model.main(argv, default_model="DEPTHVLM")


if __name__ == "__main__":
    main()
