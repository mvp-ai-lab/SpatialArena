#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs spatialarena.models.mapanything_wrappers.Pow3R.

The engine's adapter for it; see run_model.py.
"""

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable-next=wrong-import-position
from scripts.inference import run_model  # noqa: E402


def main(argv=None):
    """Runs POW3R on argv (default: sys.argv); see run_model.py."""
    run_model.main(argv, default_model="POW3R")


if __name__ == "__main__":
    main()
