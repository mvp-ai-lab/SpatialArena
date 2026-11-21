#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-vbr
#SBATCH --output=/home/%u/tmp/dl-vbr_%j.log
#SBATCH --error=/home/%u/tmp/dl-vbr_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# VBR: A Vision Benchmark in Rome (Brizi et al., ICRA 2024; https://rvp-group.net/slam-dataset.html), in the form
# the long-sequence reconstruction papers evaluate on: LoGeR's processed release (HF dataset Junyi42/vbr_processed,
# eval/datasets_preprocess/vbr_preprocess.py of github.com/Junyi42/LoGeR), which LingBot-Map's benchmark
# (benchmark/datasets/vbr.py) also reads. Seven training sequences, one tarball each (~121 GB in total), each with
# <seq>/{rgb/, intrinsics.txt, <seq>_gt.txt (TUM, row k = k-th sorted frame), camera_pose/}.

DATA_ROOT_PATH=${DATA_ROOT_PATH:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/SLAM_source}
VBR_PATH=${VBR_PATH:-$DATA_ROOT_PATH/vbr_raw}
PYTHON=${PYTHON:-$HOME/anaconda3/envs/si_bench/bin/python}
SEQUENCES=${SEQUENCES:-campus_train0 campus_train1 ciampino_train1 colosseo_train0 diag_train0 pincio_train0 spagna_train0}

mkdir -p "$VBR_PATH/archives"
for seq in $SEQUENCES; do
    if [ -f "$VBR_PATH/.extracted_$seq" ]; then
        echo "$seq already extracted"
        continue
    fi
    echo "Downloading $seq..."
    "$PYTHON" -c "from huggingface_hub import hf_hub_download; hf_hub_download('Junyi42/vbr_processed', '$seq.tar.gz', repo_type='dataset', local_dir='$VBR_PATH/archives')"
    echo "Extracting $seq..."
    tar -xzf "$VBR_PATH/archives/$seq.tar.gz" -C "$VBR_PATH"
    touch "$VBR_PATH/.extracted_$seq"
    rm -f "$VBR_PATH/archives/$seq.tar.gz"
done
echo "Done. Raw data in $VBR_PATH:"
ls "$VBR_PATH"
