#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-sintel
#SBATCH --output=/home/%u/tmp/dl-sintel_%j.log
#SBATCH --error=/home/%u/tmp/dl-sintel_%j.err

#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# This script is adapted from:
#   https://github.com/HengyiWang/amb3r/blob/main/scripts/download_videodepth.sh

DATA_ROOT_PATH=${DATA_ROOT_PATH:-$HOME/datasets/benchmarks/VideoDepth}
SINTEL_DIR=$DATA_ROOT_PATH/sintel
mkdir -p $SINTEL_DIR
echo "Downloading Sintel..."

wget -c --no-proxy http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip -P $SINTEL_DIR
wget -c --no-proxy http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip -P $SINTEL_DIR
wget -c --no-proxy http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip -P $SINTEL_DIR

echo "Unzipping Sintel..."
find $SINTEL_DIR -name "*.zip" -exec unzip -o -q {} -d $SINTEL_DIR \;
find $SINTEL_DIR -name "*.zip" -exec rm {} \;

echo "Sintel done!"
