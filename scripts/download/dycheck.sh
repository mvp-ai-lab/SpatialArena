#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-dycheck
#SBATCH --output=/home/%u/tmp/dl-dycheck_%j.log
#SBATCH --error=/home/%u/tmp/dl-dycheck_%j.err

#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=12:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# DyCheck (Gao et al., NeurIPS 2022) as distributed for MegaSaM's evaluation: one zip per scene in the Google Drive
# folder linked from https://github.com/mega-sam/mega-sam ("Running MegaSaM on DyCheck"), folder ID
# 1BHzjHo58nGAMvKMo_AS0_SwU2tJagXXx. Each scene holds dense/rgb/1x/0_*.png and depth/2x/0_*.npy; link camera 0 into
# the benchmark with scripts/preprocess/materialize_dycheck.py (docs/benchmark/dycheck_videodepth.md). The materialized
# files are symlinks into this raw folder, so keep it.

DATA_ROOT_PATH=${DATA_ROOT_PATH:-$HOME/datasets/benchmarks/VideoDepth_source}
DYCHECK_PATH=$DATA_ROOT_PATH/dycheck_raw
GDOWN=${GDOWN:-gdown}

# Scene -> Drive file ID of <scene>.zip in that folder.
scenes=(
    "apple 16knB32whW5WuDrH0Cp7J6eZ08wlZsaAd"
    "backpack 1KgX74Af2jRpmA1tSOCgEMudFKriYya05"
    "block 1m9NpuhTp94Ex6I3QIYxbHWZlnmxH_1OE"
    "creeper 1j6WPuEddoO92kaYSfiHXGZSXDNLtVyRk"
    "handwavy 1kK-e3G6YUuwepyF3SUvDkPAKMcK2UsaW"
    "haru-sit 1MFQHexWBHXHhQ2FZ1881zaGxIlEMrmR6"
    "mochi-high-five 10mHzlNbWyK4jii3Y0RbCt87uw6pToXtE"
    "paper-windmill 1MIrCdyK0X6lQ1zNZknjWlQYtC_Ny9Xcq"
    "pillow 1e2Y7d_PuBZxd2pcrHAwjL2F7UZMDmdPo"
    "spin 1y-AZfJf4vLvn8P_N_MWxMKmnEw1k-Bx_"
    "sriracha-tree 1kgyXn3QrhXETpEw4HR--FEFQi9-YpUMF"
    "teddy 1uHizwncYT_NH7GIkcQiO8qy4FHXO58ys"
)

if ! command -v "$GDOWN" >/dev/null 2>&1; then
    echo "gdown is required. Install gdown or set GDOWN to a working downloader." >&2
    exit 2
fi

mkdir -p "$DYCHECK_PATH"

for entry in "${scenes[@]}"; do
    read -r name file_id <<< "$entry"
    if [ -d "$DYCHECK_PATH/$name/dense/rgb/1x" ] && [ -d "$DYCHECK_PATH/$name/depth/2x" ]; then
        echo "Skipping ${name}: already extracted."
        continue
    fi
    echo "Downloading ${name}..."
    "$GDOWN" "$file_id" -O "$DYCHECK_PATH/${name}.zip"

    echo "Extracting ${name}..."
    unzip -q -o "$DYCHECK_PATH/${name}.zip" -d "$DYCHECK_PATH"
    if [ ! -d "$DYCHECK_PATH/$name/dense/rgb/1x" ] || [ ! -d "$DYCHECK_PATH/$name/depth/2x" ]; then
        echo "${name}.zip did not extract to ${name}/dense/rgb/1x and ${name}/depth/2x" >&2
        exit 1
    fi

    echo "Cleaning up ${name}.zip..."
    rm "$DYCHECK_PATH/${name}.zip"
done

echo "Done. Raw data in $DYCHECK_PATH."
