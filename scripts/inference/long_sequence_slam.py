#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs the long-sequence SLAM methods (spatialarena.models).

    long_sequence_slam.py --model {VGGT_LONG,PI_LONG,VGGT_SLAM}
        --data_path ... --output_path ... [--phase {all,gpu,align}]

That is `run_model.py --model NAME ...`; --model picks the method. The
module's names stay importable from here (tests and tools use them).
"""

import argparse
import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable=wrong-import-position
from scripts.inference import run_model  # noqa: E402
from spatialarena.models import long_sequence_slam as model_module  # noqa: E402
from spatialarena.models import slam  # noqa: E402

# pylint: enable=wrong-import-position

# Compatibility names of the pre-`spatialarena.models` adapter.
METHODS = model_module.METHODS
CODE_DIRS = model_module.CODE_DIRS
SPATIALARENA_ROOT = slam.SPATIALARENA_ROOT
DEFAULT_CODE_ROOT = slam.DEFAULT_CODE_ROOT
DEFAULT_CKPT_ROOT = slam.DEFAULT_CKPT_ROOT
VGGT_SLAM_ARGS = model_module.VGGT_SLAM_ARGS
VGGT_LOADER_MODULES = model_module.VGGT_LOADER_MODULES
FIXED_INPUT_LAUNCHER = model_module.FIXED_INPUT_LAUNCHER
PHASE_LAUNCHER = model_module.PHASE_LAUNCHER
SPLIT_METHODS = model_module.SPLIT_METHODS
ALIGN_PENDING = model_module.ALIGN_PENDING
weight_paths = model_module.weight_paths
write_long_config = model_module.write_long_config
stage_images = slam.stage_images
GpuMemorySampler = slam.GpuMemorySampler
read_long_poses = model_module.read_long_poses
read_tum_poses = model_module.read_tum_poses
system_command = model_module.system_command
phase_command = model_module.phase_command
run_system = model_module.run_system
export_system_depths = model_module.export_system_depths


def main(argv=None):
    """Runs the --model's method on the scene.

    Args:
        argv: The arguments; None for sys.argv.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=METHODS, required=True)
    parser.parse_known_args(argv)
    run_model.main(argv)


run = main

if __name__ == "__main__":
    main()
