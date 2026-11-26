#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs GenCeption (spatialarena.models.genception).

    genception.py --model GENCEPTION_DEPTH|GENCEPTION_POSE <options>

The same as run_model.py --model GENCEPTION_DEPTH|GENCEPTION_POSE.
"""

import argparse
import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable=wrong-import-position
from scripts.inference import run_model  # noqa: E402
from spatialarena.models import genception as model_module  # noqa: E402
from spatialarena.models import utils  # noqa: E402

# pylint: enable=wrong-import-position

MODELS = ("GENCEPTION_DEPTH", "GENCEPTION_POSE")

# Kept for callers of the former standalone script.
METHOD = model_module.METHOD
GENCEPTION_HEIGHT = model_module.GENCEPTION_HEIGHT
GENCEPTION_WIDTH = model_module.GENCEPTION_WIDTH
GENCEPTION_NUM_FRAMES = model_module.GENCEPTION_NUM_FRAMES
DEPTH_PROMPT_FILENAME = model_module.DEPTH_PROMPT_FILENAME
RAYMAP_PROMPT_FILENAME = model_module.RAYMAP_PROMPT_FILENAME
POSE_SOLVERS = model_module.POSE_SOLVERS
export_depth_predictions = utils.export_depth_predictions
_load_video = model_module.load_video
_cover_crop_frame = model_module.cover_crop_frame
_load_pose_video = model_module.load_pose_video
_pad_to_81 = model_module.pad_to_81
_float_output = model_module.float_output
_intrinsics_on_originals = model_module.intrinsics_on_originals
_select_input_frame_indices = model_module.select_input_frame_indices
_select_output_frame_indices = model_module.select_output_frame_indices
_depth_pass = model_module.depth_pass
_resize_to_original = model_module.resize_to_original


def _export_clip_depths(pipe, image_paths, prompt_path, output_path):
    """model_module.export_clip_depths through this module's writer."""
    model_module.export_clip_depths(
        pipe,
        image_paths,
        prompt_path,
        output_path,
        export_depth=lambda *args: export_depth_predictions(*args))


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
