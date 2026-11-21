#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-tum-dynamics
#SBATCH --output=/home/%u/tmp/dl-tum-dynamics_%j.log
#SBATCH --error=/home/%u/tmp/dl-tum-dynamics_%j.err

#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# TUM RGB-D SLAM Dataset (Sturm et al., IROS 2012), "Dynamic Objects"
# category. https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
# 9 sequences (~7GB total): 1 freiburg2 (with real lens distortion) + 8
# freiburg3 (pre-undistorted).

DATA_ROOT_PATH=${DATA_ROOT_PATH:-$HOME/datasets/benchmarks/SLAM}
TUM_PATH=$DATA_ROOT_PATH/tum_dynamics_raw

mkdir -p "$TUM_PATH"

base_url="https://vision.in.tum.de/rgbd/dataset"
sequences=(
    "freiburg2/rgbd_dataset_freiburg2_desk_with_person"
    "freiburg3/rgbd_dataset_freiburg3_sitting_static"
    "freiburg3/rgbd_dataset_freiburg3_sitting_xyz"
    "freiburg3/rgbd_dataset_freiburg3_sitting_halfsphere"
    "freiburg3/rgbd_dataset_freiburg3_sitting_rpy"
    "freiburg3/rgbd_dataset_freiburg3_walking_static"
    "freiburg3/rgbd_dataset_freiburg3_walking_xyz"
    "freiburg3/rgbd_dataset_freiburg3_walking_halfsphere"
    "freiburg3/rgbd_dataset_freiburg3_walking_rpy"
)

for seq in "${sequences[@]}"; do
    name=$(basename "$seq")
    echo "Downloading ${name}..."
    wget -c "${base_url}/${seq}.tgz" -P "$TUM_PATH"

    echo "Extracting ${name}..."
    tar -xzf "$TUM_PATH/${name}.tgz" -C "$TUM_PATH"

    echo "Cleaning up ${name}.tgz..."
    rm "$TUM_PATH/${name}.tgz"
done

echo "Done. Raw data in $TUM_PATH."
