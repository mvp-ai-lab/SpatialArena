#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-7scenes
#SBATCH --output=/home/%u/tmp/dl-7scenes_%j.log
#SBATCH --error=/home/%u/tmp/dl-7scenes_%j.err

#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# This script is adapted from:
#   https://github.com/HengyiWang/amb3r/blob/main/scripts/download_3d.sh

export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT_PATH=${PROJECT_ROOT_PATH:-$HOME/Projects/SpatialArena}

DATA_ROOT_PATH=${DATA_ROOT_PATH:-$HOME/datasets/benchmarks/MVRecon}
SEVEN_SCENES_PATH=$DATA_ROOT_PATH/7scenes

# Create destination directory
mkdir -p $SEVEN_SCENES_PATH

# List of scenes
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

# Base URL
base_url="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"

# Loop through each scene
for scene in "${scenes[@]}"; do
    echo "Downloading ${scene}..."
    wget -c "${base_url}/${scene}.zip" -P $SEVEN_SCENES_PATH

    echo "Unzipping ${scene}..."
    unzip -q "$SEVEN_SCENES_PATH/${scene}.zip" -d $SEVEN_SCENES_PATH

    echo "Cleaning up ${scene}.zip..."
    rm "$SEVEN_SCENES_PATH/${scene}.zip"

    echo "Unzipping sequence files for ${scene}..."
    for seq_zip in "$SEVEN_SCENES_PATH/${scene}"/*.zip; do
        # Check if the file exists to handle cases where there are no zip files
        if [ -f "$seq_zip" ]; then
            echo "  Extracting $(basename "$seq_zip")..."
            unzip -q "$seq_zip" -d "$SEVEN_SCENES_PATH/${scene}"

            echo "  Cleaning up $(basename "$seq_zip")..."
            rm "$seq_zip"
        fi
    done
done

echo "Extracting SfM poses used in AMB3R..."
POSES_ZIP=$PROJECT_ROOT_PATH/scripts/preprocess/assets/7scenes_sfm_poses.zip

if [ -f "$POSES_ZIP" ]; then
    unzip -q $POSES_ZIP -d $SEVEN_SCENES_PATH
else
    echo "Warning: $POSES_ZIP not found. Skipping sfm_poses extraction."
fi

echo "Preprocessing 7scenes depth maps..."
python "$PROJECT_ROOT_PATH/scripts/preprocess/7scenes_preprocessing.py" \
    --data_path $SEVEN_SCENES_PATH

echo "Done! All datasets are saved in $SEVEN_SCENES_PATH."
