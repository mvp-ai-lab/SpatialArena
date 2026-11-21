#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=build-genception-env
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

ENV_NAME=${ENV_NAME:-genception}
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}  # unset under sbatch --export=NIL
# Default: the submitting user's conda (the one `conda` resolves to, else ~/anaconda3).
CONDA_BASE=${CONDA_BASE:-${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}}
CONDA_BASE=${CONDA_BASE:-$HOME/anaconda3}

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

echo "Host: $(hostname)"
echo "Python: $PYTHON"

# Pure-Python/JAX stack -- no CUDA extension compilation needed, unlike
# AMB3R's torch_scatter/pytorch3d (built from source in scripts/slurm/build_si_bench_env.sh).
"$PYTHON" -m pip install --upgrade pip
# opencv-python-headless: the official camera-pose evaluation's frame resize (scripts/inference/genception.py
# falls back to PIL without it). --upgrade picks up upstream fixes (e.g. the diffusion_single_step one).
"$PYTHON" -m pip install "jax[cuda12]" flax mediapy safetensors ml_dtypes opencv-python-headless
"$PYTHON" -m pip install --upgrade \
    "representations4d[genception] @ git+https://github.com/google-deepmind/representations4d.git"

# representations4d itself (not the genception extra specifically) requires
# kauldron, which pulls in tensorflow -- genception's own code (pipeline.py/
# model.py/utils.py/vae.py) never imports tensorflow, only flax/jax/numpy.
# Verified via a real failed run (2026-09-25): tensorflow's own CUDA 13
# toolkit packages (cuda-toolkit, nvidia-cudnn-cu13, bare nvidia-cuda-*)
# get installed alongside jax's self-contained CUDA 12 plugin and torch's
# pinned CUDA 12.4 packages, and something in that three-way mix breaks
# jaxlib's bundled cuDNN at runtime (JaxRuntimeError: dnn_support !=
# nullptr, reproduced even on a completely idle H200 with a modern driver
# -- ruled out as a driver/hardware issue). Removing tensorflow (kauldron
# itself is left installed; nothing in the genception import path calls
# into its tensorflow-dependent code) fixes this.
"$PYTHON" -m pip uninstall -y tensorflow tensorflow-datasets tensorflow-metadata cuda-toolkit nvidia-cudnn-cu13 || true

# No torch here at all -- genception's own code (pipeline.py/model.py/
# utils.py/vae.py) is pure flax/jax, never imports torch. An earlier
# version of this script pinned torch (for scripts/preprocess/
# generate_genception_raymap_prompt.py's UMT5-XXL step) at 2.5.0, then at
# 2.6.0 -- verified via two separate real failed runs (2026-09-25) that
# *either* version of torch present in this same env breaks jax's own
# self-contained CUDA 12 plugin (JaxRuntimeError: dnn_support != nullptr),
# for reasons not fully root-caused beyond "importing torch here somehow
# disturbs jax-cuda12-pjrt's cuDNN init, regardless of torch's own version
# or cudnn pin". Since the raymap prompt script never imports jax either,
# the real fix is running it in a torch-only env instead (si_bench, which
# already has transformers/sentencepiece/accelerate -- see
# scripts/slurm/build_si_bench_env.sh) rather than sharing this one.

# Workaround for a real upstream typo (verified against the repo's own
# commit history and its colab notebook, which imports the correctly-named
# function directly from .utils, bypassing this broken __init__.py path):
# as of commit a47e551 (2026-09-23), genception/__init__.py does
# `from representations4d.genception.utils import render_pointcloud_view`,
# but utils.py only defines render_pointcloud_video/render_pointcloud_frame
# -- render_pointcloud_view does not exist anywhere in the repo, so the
# package fails to import at all until this is patched. Safe to remove once
# upstream fixes this.
GENCEPTION_INIT_PY=$("$PYTHON" -c "
import representations4d, os
# representations4d has no top-level __init__.py (a namespace package), so
# this import never touches genception/__init__.py's broken import chain --
# safe to use for locating the file we're about to patch.
root = list(representations4d.__path__)[0]
print(os.path.join(root, 'genception', '__init__.py'))
" 2>/dev/null || true)
if [ -n "$GENCEPTION_INIT_PY" ] && grep -q "render_pointcloud_view" "$GENCEPTION_INIT_PY" 2>/dev/null; then
    echo "Patching upstream typo in $GENCEPTION_INIT_PY (render_pointcloud_view -> render_pointcloud_video)"
    sed -i 's/render_pointcloud_view/render_pointcloud_video/' "$GENCEPTION_INIT_PY"
fi

echo "=== Verifying imports and real GPU compute ==="
"$PYTHON" - <<'PY'
import jax
import jax.numpy as jnp
print("JAX devices:", jax.devices())
print("JAX version:", jax.__version__)
# A real op, not just an import check -- this is what actually exercises
# cuDNN/XLA's GPU codepath and would have caught the CUDA 12 vs 13 conflict
# this script now avoids.
result = jnp.sum(jnp.ones((4, 4)) @ jnp.ones((4, 4)))
print("JAX GPU matmul smoke test result:", float(result))
import flax
print("flax:", flax.__version__)
from representations4d.genception import GenCeptionPipeline
print("representations4d.genception import OK")
PY

echo "GenCeption env build complete: $CONDA_BASE/envs/$ENV_NAME/bin/python"
