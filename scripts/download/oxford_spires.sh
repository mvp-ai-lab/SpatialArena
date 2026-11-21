#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-oxford-spires
#SBATCH --output=/home/%u/tmp/dl-oxford-spires_%j.log
#SBATCH --error=/home/%u/tmp/dl-oxford-spires_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# The Oxford Spires Dataset (Tao et al., 2024; https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/), HF dataset
# ori-drs/oxford_spires_dataset (CC BY-NC-SA 4.0). Only what the camera-pose evaluation needs, for the ten sequences
# LingBot-Map's benchmark evaluates (lingbot-map preprocess/oxford.py PROCESS_SCENE): each sequence's
# raw/images.zip (three fisheye cameras; cam0 is used) and processed/trajectory/gt-tum.txt, plus the top-level
# calibration (cam0.yaml, cam-lidar-imu.yaml). ~76 GB. The dataset's own layout is kept, so lingbot-map's
# preprocess/oxford.py runs on it unchanged (see scripts/preprocess/materialize_oxford_spires.py).

DATA_ROOT_PATH=${DATA_ROOT_PATH:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/SLAM_source}
OXFORD_PATH=${OXFORD_PATH:-$DATA_ROOT_PATH/oxford_spires_raw}
PYTHON=${PYTHON:-$HOME/anaconda3/envs/si_bench/bin/python}

mkdir -p "$OXFORD_PATH"
OXFORD_PATH="$OXFORD_PATH" "$PYTHON" - <<'PY'
import os
from huggingface_hub import hf_hub_download

root = os.environ["OXFORD_PATH"]
# lingbot-map's list names christ-church-05 "2024-03-18-..."; the dataset's sequence is 2024-03-20-christ-church-05.
sequences = [
    "2024-03-12-keble-college-02", "2024-03-12-keble-college-03", "2024-03-12-keble-college-04",
    "2024-03-12-keble-college-05", "2024-03-13-observatory-quarter-01", "2024-03-13-observatory-quarter-02",
    "2024-03-18-christ-church-02", "2024-03-18-christ-church-03", "2024-03-20-christ-church-05",
    "2024-05-20-bodleian-library-02",
]
files = ["calibration/README.md", "calibration/cam0.yaml", "calibration/cam-lidar-imu.yaml"]
for seq in sequences:
    files += [f"sequences/{seq}/processed/trajectory/gt-tum.txt", f"sequences/{seq}/raw/images.zip"]
for name in files:
    path = hf_hub_download("ori-drs/oxford_spires_dataset", name, repo_type="dataset", local_dir=root)
    print(f"{name} -> {path} ({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)
PY
echo "Done. Raw data in $OXFORD_PATH."
