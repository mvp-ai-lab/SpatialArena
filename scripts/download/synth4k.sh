#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-synth4k
#SBATCH --output=dl-synth4k_%j.log
#SBATCH --error=dl-synth4k_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

# Synth4K, the 4K game-captured benchmark of InfiniDepth (CVPR 2026, arXiv 2601.03252), from HF dataset
# ritianyu/game_4k_data (linked from https://github.com/zju3dv/InfiniDepth DATA.md; no license file published).
# One tar per game, ~87 GB in total: cyberpunk, deadisland, spiderman2, spidermanmm, watchdoglegion. The tars are kept
# as downloaded; scripts/preprocess/materialize_synth4k.py reads the selected frames straight out of them.
#
# Usage: sbatch --export=NIL scripts/download/synth4k.sh [game ...]
# Idempotent: a tar that is already complete is not downloaded again.

DATA_ROOT_PATH=${DATA_ROOT_PATH:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/MonoDepth_source}
SYNTH4K_PATH=${SYNTH4K_PATH:-$DATA_ROOT_PATH/synth4k}
# Pinned revision of the HF dataset repo (read 2026-10-16).
SYNTH4K_REVISION=${SYNTH4K_REVISION:-9d5fc8d41dd73304240eff5eaa15ba002753b990}
PYTHON=${PYTHON:-$HOME/anaconda3/envs/si_bench/bin/python}
# Keep the HF cache (and its xet chunk store) off the home quota.
export HF_HOME=${HF_HOME:-$DATA_ROOT_PATH/.hf_home}

if [ "$#" -gt 0 ]; then
    GAMES=("$@")
else
    GAMES=(cyberpunk deadisland spiderman2 spidermanmm watchdoglegion)
fi

mkdir -p "$SYNTH4K_PATH"
for game in "${GAMES[@]}"; do
    echo "[download] $game.tar @ $SYNTH4K_REVISION"
    SYNTH4K_PATH="$SYNTH4K_PATH" SYNTH4K_REVISION="$SYNTH4K_REVISION" GAME="$game" "$PYTHON" - <<'PY'
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="ritianyu/game_4k_data",
    repo_type="dataset",
    filename=f"{os.environ['GAME']}.tar",
    revision=os.environ["SYNTH4K_REVISION"],
    local_dir=os.environ["SYNTH4K_PATH"],
)
PY
done
echo "Synth4K tars ready under $SYNTH4K_PATH"
