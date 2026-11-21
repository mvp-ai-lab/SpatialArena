# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run MASt3R sparse global alignment and export benchmark artifacts."""

from mapanything_dense_wrapper import run_wrapper_method


if __name__ == "__main__":
    run_wrapper_method(
        method="MAST3R",
        wrapper_module="mapanything.models.external.mast3r",
        wrapper_class="MASt3RSGAWrapper",
        ckpt_env="MAST3R_CKPT_PATH",
        code_env="MAST3R_CODE_PATH",
        code_package_names=("mast3r", "dust3r"),
        wrapper_kwargs={
            "scene_graph": "complete",
            "sparse_ga_niter1": 120,
            "sparse_ga_niter2": 120,
        },
    )
