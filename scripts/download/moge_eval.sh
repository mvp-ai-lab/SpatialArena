#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-moge-eval
#SBATCH --output=dl-moge-eval_%j.log
#SBATCH --error=dl-moge-eval_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# MoGe's processed monocular-geometry evaluation sets (HF dataset Ruicheng/monocular-geometry-evaluation, the frames,
# masks and intrinsics MoGe / MoGe-2 / MoGe-3 report on; https://github.com/microsoft/MoGe/blob/main/docs/eval.md).
# Downloads and unzips one zip per dataset: HAMMER (~400 MB, 775 frames), iBims-1 (~41 MB, 100 frames) and Spring
# (~1.6 GB, 1000 frames). Each sample is a folder with image.jpg, depth.png (MoGe's log-encoded 16-bit depth) and
# meta.json (normalized intrinsics). scripts/preprocess/materialize_moge_eval.py turns them into MonoDepth/<name>.
#
# Usage: sbatch --export=NIL scripts/download/moge_eval.sh [HAMMER] [iBims-1] [Spring]
# Idempotent: a dataset whose unzipped folder already has .index.txt is skipped.

DATA_ROOT_PATH=${DATA_ROOT_PATH:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/MonoDepth_source}
MOGE_EVAL_PATH=${MOGE_EVAL_PATH:-$DATA_ROOT_PATH/moge_eval}
# Pinned revision of the HF dataset repo (read 2026-10-16).
MOGE_EVAL_REVISION=${MOGE_EVAL_REVISION:-574610d77c0cc14d911a1ff46ec529ba934c8960}
PYTHON=${PYTHON:-$HOME/anaconda3/envs/si_bench/bin/python}
# Keep the HF cache (and its xet chunk store) off the home quota.
export HF_HOME=${HF_HOME:-$DATA_ROOT_PATH/.hf_home}

if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(HAMMER iBims-1 Spring)
fi

mkdir -p "$MOGE_EVAL_PATH"
for dataset in "${DATASETS[@]}"; do
    if [ -f "$MOGE_EVAL_PATH/$dataset/.index.txt" ]; then
        echo "[skip] $dataset already extracted at $MOGE_EVAL_PATH/$dataset"
        continue
    fi
    echo "[download] $dataset.zip @ $MOGE_EVAL_REVISION"
    MOGE_EVAL_PATH="$MOGE_EVAL_PATH" MOGE_EVAL_REVISION="$MOGE_EVAL_REVISION" DATASET="$dataset" "$PYTHON" - <<'PY'
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Ruicheng/monocular-geometry-evaluation",
    repo_type="dataset",
    filename=f"{os.environ['DATASET']}.zip",
    revision=os.environ["MOGE_EVAL_REVISION"],
    local_dir=os.environ["MOGE_EVAL_PATH"],
)
PY
    echo "[unzip] $dataset.zip"
    (cd "$MOGE_EVAL_PATH" && unzip -q -o "$dataset.zip")
    if [ ! -f "$MOGE_EVAL_PATH/$dataset/.index.txt" ]; then
        echo "Unzipped $dataset.zip but $MOGE_EVAL_PATH/$dataset/.index.txt is missing" >&2
        exit 1
    fi
done
echo "MoGe evaluation sets ready under $MOGE_EVAL_PATH"
