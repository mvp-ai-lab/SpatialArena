#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-marigold
#SBATCH --output=/home/%u/tmp/dl-marigold_%j.log
#SBATCH --error=/home/%u/tmp/dl-marigold_%j.err

#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# This script is adapted from:
#   https://github.com/HengyiWang/amb3r/blob/main/scripts/download_mono.sh

DATA_ROOT_PATH=${DATA_ROOT_PATH:-$HOME/datasets/benchmarks/MonoDepth}
MARIGOLD_DIR=$DATA_ROOT_PATH/marigold

mkdir -p $MARIGOLD_DIR
cd $MARIGOLD_DIR

# Download the dataset using wget
# -r: recursive download
# -np: don't ascend to the parent directory
# -nH: disable generation of host-prefixed directories
# --cut-dirs=4: skip the 4 URL directories (~pf/bingkedata/marigold/evaluation_dataset/)
# -R "index.html*": reject auto-generated index.html files

echo "Starting download for Marigold dataset..."
echo "Target directory: $(pwd)"

wget -r -np -nH --cut-dirs=4 -R "index.html*" \
    https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/

echo "Marigold dataset download completed successfully!"
