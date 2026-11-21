#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-kitti
#SBATCH --output=/home/%u/tmp/dl-kitti_%j.log
#SBATCH --error=/home/%u/tmp/dl-kitti_%j.err

#SBATCH --partition=cpu
#SBATCH --qos=high
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# This script is adapted from:
#   https://github.com/HengyiWang/amb3r/blob/main/scripts/download_videodepth.sh

export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT_PATH=${PROJECT_ROOT_PATH:-$HOME/Projects/SpatialArena}

DATA_ROOT_PATH=${DATA_ROOT_PATH:-$HOME/datasets/benchmarks/VideoDepth}
KITTI_DIR=$DATA_ROOT_PATH/kitti
mkdir -p $KITTI_DIR

echo "Downloading KITTI..."

wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0013/2011_09_26_drive_0013_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0020/2011_09_26_drive_0020_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0023/2011_09_26_drive_0023_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0036/2011_09_26_drive_0036_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0079/2011_09_26_drive_0079_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0095/2011_09_26_drive_0095_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0113/2011_09_26_drive_0113_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0037/2011_09_28_drive_0037_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0026/2011_09_29_drive_0026_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip -P $KITTI_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0047/2011_10_03_drive_0047_sync.zip -P $KITTI_DIR

echo "Unzipping KITTI..."
find $KITTI_DIR -name "*.zip" -exec unzip -o -q {} -d $KITTI_DIR \;
find $KITTI_DIR -name "*.zip" -exec rm {} \;

echo "Preprocessing KITTI..."
python "$PROJECT_ROOT_PATH/scripts/preprocess/kitti_preprocessing.py" \
    --data_path $KITTI_DIR

echo "KITTI done!"
