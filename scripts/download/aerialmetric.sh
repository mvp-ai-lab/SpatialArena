#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-aerialmetric
#SBATCH --output=dl-aerialmetric_%j.log
#SBATCH --error=dl-aerialmetric_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# AerialMetric (ECCV 2026, arXiv 2606.29716; HF dataset Kuiee/AerialMetric-ECCV2026, academic research use only, see
# https://github.com/kuieless/AerialMetric/blob/main/DATA_LICENSE.md). Only the Test-Split is downloaded (~15 GB):
#   AerialMetric-Decoupled-norm.tar.xz  4 scenes x pitch/altitude/FOV, per sample image.jpg + depth.npy + meta.json
#   AerialMetric-Oblique-norm.tar.xz    oblique test split in the same norm layout
#   AerialMetric-Wild.tar.xz            100 YouTube scenes, rgbs/ + depth/ + depth_1k/ (no intrinsics)
#   AerialMetric-{Decoupled,Oblique}-masks.tar.xz  manually annotated dynamic-object masks
#   AerialMetric-Decoupled-Full-Metadata.zip       flight parameters (pitch, altitude, FOV) per sample
# The Train-Split (Oblique / Synthetic training data of MoGe2-Aerial) is not downloaded. The Decoupled file names on
# the hub start with a space (kept as is under hf/).
# scripts/preprocess/materialize_aerialmetric.py turns them into MonoDepth/aerialmetric.
#
# Usage: sbatch --export=NIL scripts/download/aerialmetric.sh
# Idempotent: an archive whose extraction marker exists is skipped.

DATA_ROOT_PATH=${DATA_ROOT_PATH:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/MonoDepth_source}
AERIALMETRIC_PATH=${AERIALMETRIC_PATH:-$DATA_ROOT_PATH/aerialmetric}
# Pinned revision of the HF dataset repo (read 2026-10-16).
AERIALMETRIC_REVISION=${AERIALMETRIC_REVISION:-c8fb8ea68fba6a3bace5814df26c248995aef8a1}
PYTHON=${PYTHON:-$HOME/anaconda3/envs/si_bench/bin/python}
# Keep the HF cache (and its xet chunk store) off the home quota.
export HF_HOME=${HF_HOME:-$DATA_ROOT_PATH/.hf_home}

ARCHIVES=(
    "Test-Split/ AerialMetric-Decoupled-norm.tar.xz"
    "Test-Split/ AerialMetric-Decoupled-masks.tar.xz"
    "Test-Split/AerialMetric-Oblique-norm.tar.xz"
    "Test-Split/AerialMetric-Oblique-masks.tar.xz"
    "Test-Split/AerialMetric-Wild.tar.xz"
    "Test-Split/AerialMetric-Decoupled-Full-Metadata.zip"
)

mkdir -p "$AERIALMETRIC_PATH/hf" "$AERIALMETRIC_PATH/extracted"
for remote in "${ARCHIVES[@]}"; do
    name=$(basename "$remote")
    name=${name# }
    archive="$AERIALMETRIC_PATH/hf/$remote"
    marker="$AERIALMETRIC_PATH/extracted/.${name}.done"
    if [ -f "$marker" ]; then
        echo "[skip] $name already extracted"
        continue
    fi
    echo "[download] $remote @ $AERIALMETRIC_REVISION"
    REMOTE="$remote" AERIALMETRIC_PATH="$AERIALMETRIC_PATH" AERIALMETRIC_REVISION="$AERIALMETRIC_REVISION" \
        "$PYTHON" - <<'PY'
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Kuiee/AerialMetric-ECCV2026",
    repo_type="dataset",
    filename=os.environ["REMOTE"],
    revision=os.environ["AERIALMETRIC_REVISION"],
    local_dir=os.path.join(os.environ["AERIALMETRIC_PATH"], "hf"),
)
PY
    echo "[extract] $name"
    case "$name" in
        *.zip) unzip -q -o "$archive" -d "$AERIALMETRIC_PATH/extracted" ;;
        *.tar.xz) tar -xJf "$archive" -C "$AERIALMETRIC_PATH/extracted" ;;
    esac
    touch "$marker"
done
echo "AerialMetric Test-Split ready under $AERIALMETRIC_PATH/extracted"
