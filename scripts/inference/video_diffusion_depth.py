#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs DepthCrafter or ChronoDepth (spatialarena.models.video_diffusion_depth).

    video_diffusion_depth.py --model DEPTHCRAFTER|CHRONODEPTH <options>

The same as run_model.py --model DEPTHCRAFTER|CHRONODEPTH.
"""

import argparse
import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable=wrong-import-position
from scripts.inference import run_model  # noqa: E402
from spatialarena.models import video_diffusion_depth as model_module

# pylint: enable=wrong-import-position

MODELS = ("DEPTHCRAFTER", "CHRONODEPTH")

# Kept for callers of the former standalone script.
REPOS = model_module.REPOS
processing_size = model_module.processing_size
restore_output = model_module.restore_output


def main(argv=None):
    """Runs the --model with the command line."""
    parser = argparse.ArgumentParser(
        add_help=False,
        description=(f"Run {' or '.join(MODELS)}; --model NAME --help lists "
                     "its options."))
    parser.add_argument("--model", help=" or ".join(MODELS))
    parser.add_argument("-h", "--help", action="store_true")
    known, _ = parser.parse_known_args(argv)
    if known.model is None:
        if known.help:
            parser.print_help()
            return
        parser.error("the following arguments are required: --model")
    run_model.main(argv)


if __name__ == "__main__":
    main()
