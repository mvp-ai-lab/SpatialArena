# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run Pow3R and export benchmark prediction artifacts."""

from mapanything_dense_wrapper import run_wrapper_method


if __name__ == "__main__":
    run_wrapper_method(
        method="POW3R",
        wrapper_module="mapanything.models.external.pow3r",
        wrapper_class="Pow3RBAWrapper",
        ckpt_env="POW3R_CKPT_PATH",
        code_env="POW3R_CODE_PATH",
        code_package_names=("pow3r", "dust3r"),
        wrapper_kwargs={
            "geometric_input_config": {
                "overall_prob": 0.0,
                "ray_dirs_prob": 0.0,
                "depth_prob": 0.0,
                "cam_prob": 0.0,
            },
            "scene_graph": "complete",
            "inference_batch_size": 4,
            "global_optim_niter": 120,
        },
    )
