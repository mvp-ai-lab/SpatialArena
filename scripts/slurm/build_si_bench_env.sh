#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# Builds "si_bench", the default benchmark env (Python 3.11, torch 2.6.0+cu124), used by spatialarena and every
# method except GenCeption -- including AMB3R, SenseNova-Vision and the long-sequence SLAM methods.
#
# It is rebuilt from a package list rather than cloned (scripts/slurm/si_bench_freeze.txt, the `pip freeze` of the
# earlier Python 3.10 si_bench on 2026-10-07; that env was a conda clone of "dogs" and could not change Python version,
# and its torch tree carried stale files from the cloned torch that broke torch.compile):
#   1. torch 2.6.0+cu124 / torchvision 0.21.0.
#   2. Every other pinned package at the same version, one at a time with --no-deps (a package without a cp311
#      wheel at that version falls back to the newest cp311 release and is logged), skipping what the old env
#      inherited from dogs but the benchmark never imports: DOGE's local CUDA extensions (Gaussian rasterizers,
#      fused_ssim, tetra-nerf, ...), GroundingDINO, LoMa, ppisp, simple-knn.
#   3. Editable spatialarena (this repo) and the git pin of utils3d, which AMB3R's vendored MoGe imports
#      (amb3r/model.py -> moge/train/losses.py). (Adapters put the method code from _model_codes on sys.path at run
#      time.)
#   4. AMB3R's source-built extensions (torch_scatter, pytorch3d, flash-attn; --no-cache-dir so no stale ABI), plus
#      spconv for CUDA 12 (AMB3R's PTv3 backend, thirdparty/ptv3/point_transformer.py).
#   5. The long-sequence SLAM deps: gtsam-develop (cp311+ only, the reason for Python 3.11), faiss-gpu-cu12 (no deps,
#      so it cannot pull numpy 2.x), viser 0.2.23 (VGGT-SLAM's pin) and VGGT-Long's C++ sim3solve.
#   6. The official MonoDepth baselines / DVD (scripts/inference/single_image_depth.py, dvd.py): xformers for torch
#      2.6 (UniDepth V2), MoGe's utils3d_moge pin and FlexGEMM's pin (MoGe-3; Triton kernels only, no CUDA extension).
#      Their PyPI packages (diffusers 0.36.0, e3nn, ftfy, tabulate, ...) are in the freeze list of step 2.
# The build node needs no GPU: CUDA extensions are compiled for the H200 (sm_90) and A100 (sm_80) explicitly.
#
# The env can be made with conda or uv (ENV_TOOL). conda also installs conda-forge's GCC 12 (supported by the CUDA
# 12.x nvcc) into the env, and every extension is built with it. uv cannot install a compiler, so a uv build uses
# CC/CXX if given, else the node's g++ (GCC 9-13), and stops if there is none. Every later step is the same: a uv env
# is made with --seed, so it has pip like a conda env.
#
# Usage, from the repository root (the job log lands there unless --output says otherwise):
#   sbatch --export=NIL scripts/slurm/build_si_bench_env.sh
# Defaults come from the submitting user and directory. --export=NIL passes no environment, so override them as
# KEY=VALUE arguments after the script, e.g. `sbatch --export=NIL scripts/slurm/build_si_bench_env.sh ENV_NAME=x`:
#   ENV_TOOL        conda | uv             (default: conda if found, else uv)
#   ENV_TOOL_BIN    the tool's executable  (default: found on PATH or in its usual install folders)
#   ENV_NAME        env to build           (default: si_bench)
#   ENV_PREFIX      env location           (default: conda <CONDA_ROOT>/envs/<ENV_NAME>, uv ~/.venvs/<ENV_NAME>)
#   CONDA_ROOT      conda installation     (default: the user's ~/anaconda3, else ~/miniconda3, ~/miniforge3)
#   CC, CXX         C/C++ compilers        (default: the env's conda-forge GCC 12; uv: the node's gcc/g++)
#   PROJECT_ROOT    this repository        (default: the sbatch submit directory)
#   BENCHMARK_ROOT  shared benchmark root  (default: the team benchmark root)
#   CODE_ROOT       method code            (default: $BENCHMARK_ROOT/_model_codes)
#   SCRATCH_DIR     build temp + logs      (default: $BENCHMARK_ROOT/_scratch/<user>)
#   MAX_JOBS        parallel compile jobs  (default: 16)
# Other scripts pick the env up through PYTHON=<ENV_PREFIX>/bin/python when it is not the conda default.

#SBATCH --job-name=build-si-bench
#SBATCH --output=%x_%j.log
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
# --export=NIL starts the job without PATH.
export PATH="${PATH:+$PATH:}/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

for arg in "$@"; do
    case "$arg" in
        ENV_TOOL=*|ENV_TOOL_BIN=*|ENV_NAME=*|ENV_PREFIX=*|CONDA_ROOT=*|CC=*|CXX=*|PROJECT_ROOT=*|BENCHMARK_ROOT=*|CODE_ROOT=*\
        |SCRATCH_DIR=*|FREEZE=*|MAX_JOBS=*)
            export "${arg?}" ;;
        *) echo "ERROR: unknown argument '$arg' (expected KEY=VALUE, see the header)" >&2; exit 1 ;;
    esac
done
USER_NAME=$(id -un)
HOME=${HOME:-$(getent passwd "$USER_NAME" | cut -d: -f6)}
export HOME
if [ -z "${CONDA_ROOT:-}" ]; then
    for conda_dir in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3"; do
        if [ -x "$conda_dir/bin/conda" ]; then
            CONDA_ROOT=$conda_dir
            break
        fi
    done
fi
# The tool's executable: on PATH (minimal under --export=NIL) or in its usual install folders.
find_tool() {
    local candidate
    for candidate in "$(command -v "$1" 2>/dev/null)" "${CONDA_ROOT:+$CONDA_ROOT/bin/$1}" "$HOME/.local/bin/$1" \
        "$HOME/.cargo/bin/$1" "$HOME/bin/$1"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}
ENV_TOOL=${ENV_TOOL:-auto}
if [ "$ENV_TOOL" = auto ] && [ -n "${ENV_TOOL_BIN:-}" ]; then
    ENV_TOOL=$(basename "$ENV_TOOL_BIN")
fi
if [ "$ENV_TOOL" = auto ]; then
    for tool in conda uv; do
        if find_tool "$tool" >/dev/null; then
            ENV_TOOL=$tool
            break
        fi
    done
fi
case "$ENV_TOOL" in
    conda|uv) ;;
    *) echo "ERROR: no usable ENV_TOOL (conda or uv) found; got '$ENV_TOOL'" >&2; exit 1 ;;
esac
TOOL_BIN=${ENV_TOOL_BIN:-$(find_tool "$ENV_TOOL")} \
    || { echo "ERROR: $ENV_TOOL not found; pass ENV_TOOL_BIN=/path/to/$ENV_TOOL" >&2; exit 1; }
ENV_NAME=${ENV_NAME:-si_bench}
if [ -z "${ENV_PREFIX:-}" ]; then
    case "$ENV_TOOL" in
        conda)
            CONDA_ROOT=${CONDA_ROOT:-$(dirname "$(dirname "$TOOL_BIN")")}
            ENV_PREFIX=$CONDA_ROOT/envs/$ENV_NAME ;;
        uv) ENV_PREFIX=$HOME/.venvs/$ENV_NAME ;;
    esac
fi
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -d "$PROJECT_ROOT/spatialarena" ]; then
    echo "ERROR: $PROJECT_ROOT is not the repository root; submit from there or set PROJECT_ROOT" >&2
    exit 1
fi
BENCHMARK_ROOT=${BENCHMARK_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks}
CODE_ROOT=${CODE_ROOT:-$BENCHMARK_ROOT/_model_codes}
SCRATCH_DIR=${SCRATCH_DIR:-$BENCHMARK_ROOT/_scratch/$USER_NAME}
FREEZE=${FREEZE:-$PROJECT_ROOT/scripts/slurm/si_bench_freeze.txt}
PYTHON=$ENV_PREFIX/bin/python
LOG_DIR=$SCRATCH_DIR/build-si-bench
export TMPDIR=$SCRATCH_DIR/tmp
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=${MAX_JOBS:-16}
mkdir -p "$LOG_DIR" "$TMPDIR"
echo "tool=$ENV_TOOL ($TOOL_BIN) env=$ENV_PREFIX project=$PROJECT_ROOT code=$CODE_ROOT scratch=$SCRATCH_DIR"

if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
    module load cuda 2>/dev/null || module load cuda/12.8 2>/dev/null || true
fi
if ! command -v nvcc >/dev/null 2>&1; then
    for cuda_dir in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.6; do
        if [ -x "$cuda_dir/bin/nvcc" ]; then
            export CUDA_HOME="$cuda_dir" PATH="$cuda_dir/bin:$PATH"
            break
        fi
    done
fi
nvcc --version | tail -1

ENV_BIN=$ENV_PREFIX/bin
if [ ! -x "$PYTHON" ]; then
    case "$ENV_TOOL" in
        conda)
            "$TOOL_BIN" create -y -p "$ENV_PREFIX" -c conda-forge python=3.11 "gcc_linux-64=12" "gxx_linux-64=12" ;;
        uv)
            # --seed puts pip in the venv; uv fetches a managed Python 3.11 when the node has none.
            UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH_DIR/uv_cache} "$TOOL_BIN" venv --seed --python 3.11 "$ENV_PREFIX" ;;
    esac
fi
if [ -z "${CC:-}" ] && [ -x "$ENV_BIN/x86_64-conda-linux-gnu-gcc" ]; then
    CC=$ENV_BIN/x86_64-conda-linux-gnu-gcc CXX=$ENV_BIN/x86_64-conda-linux-gnu-g++
fi
if [ -z "${CC:-}" ]; then
    CC=$(command -v gcc || true) CXX=$(command -v g++ || true)
fi
if [ -z "${CC:-}" ] || [ -z "${CXX:-}" ] || [ ! -x "$CXX" ]; then
    echo "ERROR: no C/C++ compiler: pass CC=... CXX=... (GCC 9-13), or build with ENV_TOOL=conda" >&2
    exit 1
fi
export CC CXX CUDAHOSTCXX=$CXX
export PATH="$ENV_BIN:$PATH"
"$PYTHON" --version
"$CXX" --version | head -1
"$PYTHON" -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
torch_fingerprint() {
    "$PYTHON" -c "
import torch, os
st = os.stat(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_cpu.so'))
print(f'{torch.__version__} size={st.st_size} mtime={int(st.st_mtime)}')"
}
ORIGINAL_FINGERPRINT=$(torch_fingerprint)
check_torch_fingerprint() {
    local current
    current=$(torch_fingerprint)
    if [ "$current" != "$ORIGINAL_FINGERPRINT" ]; then
        echo "ERROR: torch changed during '$1': $ORIGINAL_FINGERPRINT -> $current" >&2
        exit 1
    fi
    echo "torch unchanged after $1"
}

# --- 2. pinned packages, one at a time -------------------------------------------------------------------------
SKIP_REGEX='^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+|torch[_-]scatter|pytorch3d|flash[_-]attn|spconv[a-z0-9-]*|cumm[a-z0-9-]*|timm|benchmark[_-]suite|mapanything|moge|groundingdino|lomatch|ppisp|simple[_-]knn|utils3d|pipeline|diff[_-]gaussian[a-z0-9_-]*|diff[_-]gggs[a-z0-9_-]*|fastergscudabackend|fused[_-]ssim|tetra[_-]nerf|warp[_-]patch[_-]ncc|viser)$'
: > "$LOG_DIR/fallback.txt"
: > "$LOG_DIR/failed.txt"
grep -vE '^(-e |#)' "$FREEZE" | grep -v ' @ ' | while IFS= read -r spec; do
    name=${spec%%==*}
    if echo "$name" | tr 'A-Z' 'a-z' | grep -qE "$SKIP_REGEX"; then
        continue
    fi
    if "$PYTHON" -m pip install --no-deps --no-cache-dir -q "$spec" >/dev/null 2>&1; then
        continue
    elif "$PYTHON" -m pip install --no-deps --no-cache-dir -q "$name" >/dev/null 2>&1; then
        echo "$spec -> $("$PYTHON" -m pip show "$name" 2>/dev/null | awk '/^Version:/ {print $2}')" >> "$LOG_DIR/fallback.txt"
    else
        echo "$spec" >> "$LOG_DIR/failed.txt"
    fi
done
echo "pinned packages done; fallbacks: $(wc -l < "$LOG_DIR/fallback.txt"), failures: $(wc -l < "$LOG_DIR/failed.txt")"
cat "$LOG_DIR/fallback.txt" "$LOG_DIR/failed.txt"
check_torch_fingerprint "pinned packages"
"$PYTHON" -m pip install --no-deps --no-cache-dir "timm==1.0.28"
check_torch_fingerprint timm

# --- 3. editable / git projects ---------------------------------------------------------------------------------
"$PYTHON" -m pip install --no-deps --no-cache-dir -e "$PROJECT_ROOT"
# The old env also had EasternJournalist/pipeline (MoGe's training data loader); nothing on the benchmark path
# imports it (MoGe's own moge/utils/pipeline.py is vendored), so it is not installed.
"$PYTHON" -m pip install --no-deps --no-cache-dir \
    "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"
check_torch_fingerprint "editable projects"

# --- 4. source-built extensions ------------------------------------------------------
"$PYTHON" -m pip install torch-scatter==2.1.2 --no-binary torch-scatter --no-build-isolation --no-cache-dir
check_torch_fingerprint torch-scatter
"$PYTHON" -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation --no-cache-dir
check_torch_fingerprint pytorch3d
"$PYTHON" -m pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir
check_torch_fingerprint flash-attn
"$PYTHON" -m pip install --no-cache-dir "spconv-cu124==2.3.8"
check_torch_fingerprint spconv

# --- 5. SLAM additions -------------------------------------------------------------------------------------------
# All --no-deps: faiss-gpu-cu12's metadata asks for numpy>=2, which would replace si_bench's numpy 1.26.4 (numba and
# the methods need <2.3); its wheel is verified to work with 1.26.4 (index build + search), and its CUDA runtime
# deps are already satisfied by torch's cu124 libraries.
"$PYTHON" -m pip install --no-deps --no-cache-dir gtsam-develop "viser==0.2.23" faiss-gpu-cu12 "numba==0.61.2" \
    "llvmlite==0.44.0" prettytable pytorch-metric-learning torchmetrics lightning-utilities
check_torch_fingerprint "SLAM additions"
"$PYTHON" -c "import numpy; assert numpy.__version__ == '1.26.4', numpy.__version__"
# si_bench's setuptools (pip freeze omits it): 82+ dropped pkg_resources, which lightning_fabric (SALAD) imports.
"$PYTHON" -m pip install --no-deps --no-cache-dir "setuptools==81.0.0"
(cd "$CODE_ROOT/VGGT-Long" && "$PYTHON" -m pip install --no-cache-dir --no-deps --no-build-isolation .)
check_torch_fingerprint sim3solve

# --- 6. MonoDepth baselines / DVD --------------------------------------------------------------------------------
# --no-deps throughout: each would otherwise pull another torch or numpy 2.x. FLEX_GEMM_BUILD_CUDA stays unset, so
# FlexGEMM installs its Triton kernels only (MoGe-3's sparse refinement runs on them; nvcc is not needed).
"$PYTHON" -m pip install --no-deps --no-cache-dir "xformers==0.0.29.post3" \
    --index-url https://download.pytorch.org/whl/cu124
"$PYTHON" -m pip install --no-deps --no-cache-dir \
    "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
"$PYTHON" -m pip install --no-deps --no-cache-dir --no-build-isolation \
    "git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
check_torch_fingerprint "MonoDepth baselines"

echo "=== import check ==="
"$PYTHON" -m pip check || true
PYTHONPATH="$CODE_ROOT/VGGT-SLAM:$CODE_ROOT/VGGT-SLAM/third_party/vggt:$CODE_ROOT/VGGT-SLAM/third_party/salad" "$PYTHON" - <<'PY'
import sys
import torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
import torch_scatter, pytorch3d, flash_attn, spconv.pytorch, utils3d, timm, transformers
from pytorch3d.ops import knn_points
import spatialarena
import gtsam, faiss, numba, sim3solve
import diffusers, xformers, utils3d_moge  # flex_gemm needs a GPU driver to import (Triton), so it is not checked here
from gtsam import SL4
import vggt_slam.solver, salad
print("torch_scatter", torch_scatter.__version__, "pytorch3d", pytorch3d.__version__, "flash_attn", flash_attn.__version__,
      "timm", timm.__version__, "transformers", transformers.__version__)
print("si_bench imports ok")
PY
