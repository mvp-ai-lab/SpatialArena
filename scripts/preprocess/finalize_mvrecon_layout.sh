#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=finalize-mvrecon
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00

# Finish the canonical MVRecon/eth3d layout after scripts/download/eth3d.sh: copy each scene's laser-scan depth
# (ground_truth_depth/dslr_images) to depths/, then write the metadata of eth3d, dtu and tanks_and_temples.
#
#   sbatch --export=NIL --output=<log> scripts/preprocess/finalize_mvrecon_layout.sh
#
# Every file stays a real copy (never a symlink): the MVRecon/ tree is self-contained for its release.

set -euo pipefail

PROJECT_ROOT_PATH=${PROJECT_ROOT_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}  # sbatch --export=NIL leaves HOME unset
BENCHMARK_AUX_ROOT=${BENCHMARK_AUX_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks}
MVRECON_PATH=${MVRECON_PATH:-$BENCHMARK_AUX_ROOT/MVRecon}
PYTHON=${PYTHON:-python}

cd "$PROJECT_ROOT_PATH"

copy_depths() {
    local dataset_path=$1
    local source_relative=$2

    if [ ! -d "$dataset_path" ]; then
        return
    fi

    find "$dataset_path" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' scene_path; do
        if [ -L "$scene_path/depths" ]; then
            # Convert a pre-existing symlink into a real copy: the materialized layout must be self-contained
            # (e.g. for a Hugging Face release), not depend on a relative alias.
            rm "$scene_path/depths"
        elif [ -e "$scene_path/depths" ]; then
            continue
        fi
        if [ -d "$scene_path/$source_relative" ]; then
            cp -r "$scene_path/$source_relative" "$scene_path/depths"
        fi
    done
}

copy_depths "$MVRECON_PATH/eth3d" "ground_truth_depth/dslr_images"

"$PYTHON" scripts/preprocess/generate_benchmark_metadata.py --dataset_roots \
    "$MVRECON_PATH/eth3d" \
    "$MVRECON_PATH/dtu" \
    "$MVRECON_PATH/tanks_and_temples"
