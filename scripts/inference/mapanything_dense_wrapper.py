# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compatibility shim: moved to spatialarena.models.mapanything_wrappers.

dust3r.py, mast3r.py and pow3r.py now run the registered models directly;
`run_wrapper_method` is kept for callers of the old shared wrapper loop.
"""

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable-next=wrong-import-position
from spatialarena.models import mapanything_wrappers  # noqa: E402

PAIRWISE_WRAPPERS = mapanything_wrappers.PAIRWISE_WRAPPERS
attach_aligned_confidence = mapanything_wrappers.attach_aligned_confidence
capture_aligned_scene = mapanything_wrappers.capture_aligned_scene
default_intrinsics = mapanything_wrappers.default_intrinsics
describe_structure = mapanything_wrappers.describe_structure
import_symbol = mapanything_wrappers.import_symbol
infer_intrinsics = mapanything_wrappers.infer_intrinsics
move_views_to_device = mapanything_wrappers.move_views_to_device
outputs_to_standard = mapanything_wrappers.outputs_to_standard
patch_dust3r_global_aligner = mapanything_wrappers.patch_dust3r_global_aligner
preprocess_dust3r_family = mapanything_wrappers.preprocess_dust3r_family
quats_trans_to_c2w = mapanything_wrappers.quats_trans_to_c2w
resolve_checkpoint_arg = mapanything_wrappers.resolve_checkpoint_arg

_MODELS = {
    model_cls.wrapper_class: model_cls for model_cls in (
        mapanything_wrappers.DUSt3R,
        mapanything_wrappers.MASt3R,
        mapanything_wrappers.Pow3R,
    )
}


def run_wrapper_method(*,
                       method,
                       wrapper_module,
                       wrapper_class,
                       ckpt_env,
                       code_env,
                       code_package_names,
                       wrapper_kwargs=None):
    """Runs a MapAnything external-model wrapper as a command-line program.

    Args:
        method: The method name ("method" in metrics.json).
        wrapper_module: The MapAnything module of the wrapper.
        wrapper_class: The wrapper class in it.
        ckpt_env: Environment variable --ckpt_path defaults to.
        code_env: Environment variable --code_path defaults to, for a
            method without a MethodSpec.
        code_package_names: Top-level packages of the method's code.
        wrapper_kwargs: Keyword arguments of the wrapper.
    """
    parent = _MODELS.get(wrapper_class,
                         mapanything_wrappers.MapAnythingWrapperModel)
    model_cls = type(
        f"{method}Wrapper", (parent,), {
            "name": method,
            "description": f"Run {method} and export benchmark artifacts.",
            "wrapper_module": wrapper_module,
            "wrapper_class": wrapper_class,
            "ckpt_env": ckpt_env,
            "code_path_env": code_env,
            "code_packages": tuple(code_package_names),
            "wrapper_kwargs": dict(wrapper_kwargs or {}),
        })
    # A BaseModel subclass built by type(), which pylint cannot infer.
    model_cls.main()  # pylint: disable=no-member
