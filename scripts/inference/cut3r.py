#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs spatialarena.models.cut3r.CUT3R; the engine's adapter for it.

The script's former helpers stay importable from here.
"""

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable=wrong-import-position
from scripts.inference import run_model  # noqa: E402
from spatialarena.models import cut3r as model_module  # noqa: E402

# pylint: enable=wrong-import-position

build_shared_views = model_module.build_shared_views
intrinsics_from_cam_dict = model_module.intrinsics_from_cam_dict
patch_rope_negative_positions = model_module.patch_rope_negative_positions
stack_list = model_module.stack_list


def main(argv=None):
    """Runs CUT3R on argv (default: sys.argv); see run_model.py."""
    run_model.main(argv, default_model="CUT3R")


run = main

if __name__ == "__main__":
    main()
