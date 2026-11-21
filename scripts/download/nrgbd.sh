#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-nrgbd
#SBATCH --output=/home/%u/tmp/dl-nrgbd_%j.log
#SBATCH --error=/home/%u/tmp/dl-nrgbd_%j.err

#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# Neural RGB-D Surface Reconstruction dataset (Azinovic et al., CVPR 2022).
# https://github.com/dazinovic/neural-rgbd-surface-reconstruction
# 9 synthetic scenes (~7.25GB) + published ground-truth meshes (~514MB).

DATA_ROOT_PATH=${DATA_ROOT_PATH:-$HOME/datasets/benchmarks/MVRecon}
NRGBD_PATH=$DATA_ROOT_PATH/nrgbd_raw

mkdir -p "$NRGBD_PATH"

echo "Downloading neural_rgbd_data.zip..."
wget -c "https://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip" -P "$NRGBD_PATH"

echo "Extracting neural_rgbd_data.zip..."
unzip -q -o "$NRGBD_PATH/neural_rgbd_data.zip" -d "$NRGBD_PATH"
rm "$NRGBD_PATH/neural_rgbd_data.zip"

echo "Downloading meshes.zip (ground-truth meshes)..."
wget -c "https://kaldir.vc.in.tum.de/neural_rgbd/meshes.zip" -P "$NRGBD_PATH"

echo "Extracting meshes.zip..."
mkdir -p "$NRGBD_PATH/meshes"
unzip -q -o "$NRGBD_PATH/meshes.zip" -d "$NRGBD_PATH/meshes"
rm "$NRGBD_PATH/meshes.zip"

echo "Done. Raw data in $NRGBD_PATH."
