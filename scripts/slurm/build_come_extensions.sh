#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=build-come-ext
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}  # unset under sbatch --export=NIL
# Defaults come from the submitting user and directory (submit from the repository root).
CONDA_ROOT=${CONDA_ROOT:-${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}}
CONDA_ROOT=${CONDA_ROOT:-$HOME/anaconda3}
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -d "$PROJECT_ROOT/spatialarena" ]; then
    echo "ERROR: $PROJECT_ROOT is not the repository root; submit from there or set PROJECT_ROOT" >&2
    exit 1
fi
COME_ROOT=${COME_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/_model_codes/CoMe}
PYTHON=${PYTHON:-$CONDA_ROOT/envs/si_bench/bin/python}
ENV_BIN=$(dirname "$PYTHON")

export PATH="$ENV_BIN:$PATH"
export MAX_JOBS=${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}
TORCH_LIB_DIR=$("$PYTHON" - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"

if [ -f /etc/profile.d/modules.sh ]; then
    # Some clusters expose CUDA only through environment modules.
    # Loading is best-effort; explicit CUDA_HOME probing below is authoritative.
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
    module load cuda 2>/dev/null || module load cuda/12.8 2>/dev/null || true
fi

if ! command -v nvcc >/dev/null 2>&1; then
    for cuda_dir in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.6; do
        if [ -x "$cuda_dir/bin/nvcc" ]; then
            export CUDA_HOME="$cuda_dir"
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
            break
        fi
    done
fi

echo "Host: $(hostname)"
echo "Python: $PYTHON"
"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found; cannot build CoMe CUDA extensions." >&2
    exit 10
fi

nvcc --version
"$PYTHON" -m pip install pybind11 pybind11-stubgen

cd "$COME_ROOT/src/cuda_extension/co_me"
bash install.sh

cd "$COME_ROOT/src/cuda_extension/flash_attn"
bash install.sh

find "$COME_ROOT/src/cuda_extension" -name '*.so' -print
