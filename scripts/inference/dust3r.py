# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run DUSt3R and export benchmark prediction artifacts."""

from mapanything_dense_wrapper import run_wrapper_method

if __name__ == "__main__":
    run_wrapper_method(
        method="DUST3R",
        wrapper_module="mapanything.models.external.dust3r",
        wrapper_class="DUSt3RBAWrapper",
        ckpt_env="DUST3R_CKPT_PATH",
        code_env="DUST3R_CODE_PATH",
        code_package_names=("dust3r",),
        wrapper_kwargs={
            "scene_graph": "complete",
            "inference_batch_size": 8,
            "global_optim_niter": 120,
        },
    )
