#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-bonn
#SBATCH --output=/home/%u/tmp/dl-bonn_%j.log
#SBATCH --error=/home/%u/tmp/dl-bonn_%j.err

#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# This script is adapted from:
#   https://github.com/HengyiWang/amb3r/blob/main/scripts/download_videodepth.sh

export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT_PATH=${PROJECT_ROOT_PATH:-$HOME/Projects/SpatialArena}

DATA_ROOT_PATH=${DATA_ROOT_PATH:-$HOME/datasets/benchmarks/VideoDepth}
BONN_DIR=$DATA_ROOT_PATH/bonn
mkdir -p $BONN_DIR

echo "Downloading Bonn..."

wget -c https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_dataset.zip -P $BONN_DIR

echo "Unzipping Bonn..."
unzip -o -q ${BONN_DIR}/rgbd_bonn_dataset.zip -d $BONN_DIR
rm ${BONN_DIR}/rgbd_bonn_dataset.zip

echo "Preprocessing Bonn..."
python "$PROJECT_ROOT_PATH/scripts/preprocess/bonn_preprocessing.py" \
    --data_path $BONN_DIR

echo "Bonn done!"
