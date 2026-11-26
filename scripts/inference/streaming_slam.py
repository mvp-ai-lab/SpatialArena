#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs the streaming SLAM methods (spatialarena.models).

    streaming_slam.py --model {LOGER,ZIPMAP,LINGBOT_MAP,R3,SCAL3R,ABOT_RECON}
        --data_path ... --output_path ...

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
from spatialarena.models import streaming_slam as model_module  # noqa: E402

# pylint: enable=wrong-import-position

# Compatibility names of the pre-`spatialarena.models` adapter.
LOOP_INFO = model_module.LOOP_INFO
METHODS = model_module.METHODS
CODE_DIRS = model_module.CODE_DIRS
LOGER_SETTINGS = model_module.LOGER_SETTINGS
ZIPMAP_MODEL_CONFIG = model_module.ZIPMAP_MODEL_CONFIG
LINGBOT_SETTINGS = model_module.LINGBOT_SETTINGS
weight_paths = model_module.weight_paths
load_weights = model_module.load_weights
_poses_by_index = model_module._poses_by_index
_lingbot_resize = model_module._lingbot_resize
run_loger = model_module.run_loger
run_zipmap = model_module.run_zipmap
run_lingbot_map = model_module.run_lingbot_map
run_abot_recon = model_module.run_abot_recon
run_r3 = model_module.run_r3
read_r3_poses = model_module.read_r3_poses
run_scal3r = model_module.run_scal3r
read_scal3r_poses = model_module.read_scal3r_poses
IN_PROCESS = model_module.IN_PROCESS
SUBPROCESS = model_module.SUBPROCESS


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
