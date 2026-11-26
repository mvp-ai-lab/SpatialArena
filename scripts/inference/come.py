#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs spatialarena.models.come.Come; the engine's adapter for Co-Me.

Also keeps the script's former module API (`run`, `run_come`,
`load_come_archive`, ...) for callers that patch it: `run()` looks those
names up in this module.
"""

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable=wrong-import-position
from scripts.inference import run_model  # noqa: E402
from spatialarena.models import come as model_module  # noqa: E402
from spatialarena.models import utils  # noqa: E402

# pylint: enable=wrong-import-position

run_come = model_module.run_come
load_come_archive = model_module.load_come_archive
come_intermediate_paths = model_module.come_intermediate_paths
parse_scene_arrays = model_module.parse_scene_arrays
export_camera_pose_priors = utils.export_camera_pose_priors


class _ScriptCome(model_module.Come):
    """Co-Me through this module's (patchable) functions."""

    def invoke(self, image_paths, run_name):
        return run_come(self.args, image_paths, run_name=run_name)

    def read_raw(self, raw_path):
        return load_come_archive(raw_path)

    def intermediate_paths(self, raw_path):
        return come_intermediate_paths(self.args, raw_path)

    def export_cameras(self, extrinsics, intrinsics, width, height):
        export_camera_pose_priors(
            self.args.output_path,
            self.images_dir,
            self.image_paths,
            extrinsics,
            intrinsics,
            width=width,
            height=height,
            colmap_model_subdir=self.args.colmap_model_subdir,
        )


def run():
    """Runs Co-Me on sys.argv through this module's functions."""
    _ScriptCome.main()


def main(argv=None):
    """Runs COME on argv (default: sys.argv); see run_model.py."""
    run_model.main(argv, default_model="COME")


if __name__ == "__main__":
    main()
