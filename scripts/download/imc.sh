#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-imc
#SBATCH --output=/home/%u/tmp/dl-imc_%j.log
#SBATCH --error=/home/%u/tmp/dl-imc_%j.err

#SBATCH --partition=cpu
#SBATCH --qos=high
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# This script is adapted from:
#   https://github.com/HengyiWang/amb3r/blob/main/scripts/download_sfm.sh

export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT_PATH=${PROJECT_ROOT_PATH:-$HOME/Projects/SpatialArena}

DATA_ROOT_PATH=${DATA_ROOT_PATH:-$HOME/datasets/benchmarks/SfM}
IMC_DIR=$DATA_ROOT_PATH/imc
mkdir -p $IMC_DIR

echo "Downloading IMC 2020..."
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/british_museum.tar.gz -P $IMC_DIR
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/florence_cathedral_side.tar.gz -P $IMC_DIR
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/lincoln_memorial_statue.tar.gz -P $IMC_DIR
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/milan_cathedral.tar.gz -P $IMC_DIR
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/mount_rushmore.tar.gz -P $IMC_DIR
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/piazza_san_marco.tar.gz -P $IMC_DIR
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/sagrada_familia.tar.gz -P $IMC_DIR
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/st_pauls_cathedral.tar.gz -P $IMC_DIR

echo "Extracting IMC 2020..."
for archive in "${IMC_DIR}"/*.tar.gz; do
    echo "  Extracting $(basename "$archive")..."
    tar -xzf "$archive" -C $IMC_DIR
    echo "  Cleaning up $(basename "$archive")..."
    rm "$archive"
done

echo "Preprocessing IMC 2020..."
python "$PROJECT_ROOT_PATH/scripts/preprocess/data_conversion/imc_to_colmap.py" \
    --data_path $IMC_DIR
