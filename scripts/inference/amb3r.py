#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs spatialarena.models.amb3r.AMB3R; the engine's adapter for it.

Also keeps the script's former module API (`run`, `run_amb3r`,
`load_model`, ...) for callers that patch it: `run()` looks those names up
in this module.
"""

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable=wrong-import-position
from scripts.inference import run_model  # noqa: E402
from spatialarena.models import amb3r as model_module  # noqa: E402
from spatialarena.models import base  # noqa: E402
from spatialarena.models import utils  # noqa: E402

# pylint: enable=wrong-import-position

AMB3R_WIDTH = model_module.AMB3R_WIDTH
AMB3R_HEIGHT = model_module.AMB3R_HEIGHT
materialize_symlinked_images = model_module.materialize_symlinked_images
preprocess_amb3r = model_module.preprocess_amb3r
load_model = model_module.load_model
run_amb3r = model_module.run_amb3r
add_code_path = utils.add_code_path


class _ScriptAMB3R(model_module.AMB3R):
    """AMB3R through this module's (patchable) functions."""

    def add_code_path(self):
        add_code_path(self.args.code_path, self.code_packages)

    def load(self, device):
        return load_model(self.args.ckpt_path, device)

    def predict(self, image_paths, batch_index=None):
        (extrinsic, intrinsic, _, depth, points, confidence, colors,
         transforms) = run_amb3r(
             self.model,
             image_paths,
             self.args,
             self.device,
             run_name=model_module.run_name_of(batch_index),
             keep_intermediate=utils.should_keep_intermediate(self.args))
        prediction = base.Prediction(depth=depth,
                                     confidence=confidence,
                                     extrinsics_w2c=extrinsic,
                                     intrinsics=intrinsic,
                                     points=points,
                                     colors=colors)
        inputs = base.ModelInput(image_paths,
                                 transforms=transforms,
                                 batch_index=batch_index)
        return prediction, inputs


def run():
    """Runs AMB3R on sys.argv through this module's functions."""
    _ScriptAMB3R.main()


def main(argv=None):
    """Runs AMB3R on argv (default: sys.argv); see run_model.py."""
    run_model.main(argv, default_model="AMB3R")


if __name__ == "__main__":
    main()
