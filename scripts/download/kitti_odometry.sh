#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-kitti-odom
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

set -euo pipefail

# KITTI Odometry (Geiger et al., CVPR 2012; https://www.cvlibs.net/datasets/kitti/eval_odometry.php), as the
# long-sequence reconstruction papers evaluate it (LoGeR eval/eval.md, LingBot-Map benchmark/datasets/kitti.py,
# VGGT-Long): the left color camera (image_2) of the eleven sequences with public ground truth, 00-10. The color
# archive (~65 GB) holds all 22 sequences and both color cameras; only image_2 of 00-10 is extracted. The calib
# archive adds each sequence's calib.txt and times.txt, the poses archive the cam0 ground truth.
#
#   <KITTI_ODOMETRY_PATH>/dataset/sequences/NN/{image_2/NNNNNN.png, calib.txt, times.txt}
#   <KITTI_ODOMETRY_PATH>/dataset/poses/NN.txt

HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}  # sbatch --export=NIL leaves HOME unset
DATA_ROOT_PATH=${DATA_ROOT_PATH:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/SLAM_source}
KITTI_ODOMETRY_PATH=${KITTI_ODOMETRY_PATH:-$DATA_ROOT_PATH/kitti_odometry_raw}
SEQUENCES=${SEQUENCES:-00 01 02 03 04 05 06 07 08 09 10}
BASE_URL=https://s3.eu-central-1.amazonaws.com/avg-kitti

mkdir -p "$KITTI_ODOMETRY_PATH/archives"
cd "$KITTI_ODOMETRY_PATH"
for archive in data_odometry_poses data_odometry_calib data_odometry_color; do
    if [ -f ".extracted_$archive" ]; then
        echo "$archive already extracted"
        continue
    fi
    echo "Downloading $archive..."
    wget -c -q --show-progress --progress=dot:giga "$BASE_URL/$archive.zip" -P archives
    patterns=()
    if [ "$archive" = data_odometry_color ]; then
        for seq in $SEQUENCES; do
            patterns+=("dataset/sequences/$seq/image_2/*")
        done
    fi
    echo "Extracting $archive..."
    unzip -q -o "archives/$archive.zip" "${patterns[@]}"
    touch ".extracted_$archive"
    rm -f "archives/$archive.zip"
done
for seq in $SEQUENCES; do
    frames=$(find "dataset/sequences/$seq/image_2" -name '*.png' | wc -l)
    echo "$seq: $frames frames, $(wc -l < "dataset/poses/$seq.txt") poses"
done
