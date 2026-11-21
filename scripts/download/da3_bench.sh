#!/usr/bin/env bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

DATASET_ID=${DATASET_ID:-depth-anything/DA3-BENCH}
OUTPUT_DIR=${OUTPUT_DIR:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/da3_bench}
SUBSETS=${SUBSETS:-"scannetpp hiroom"}
EXTRACT=${EXTRACT:-1}

mkdir -p "$OUTPUT_DIR"

if ! command -v hf >/dev/null 2>&1; then
    echo "hf (the huggingface_hub CLI) is required. Install huggingface_hub or activate the benchmark env." >&2
    exit 2
fi

args=(download "$DATASET_ID" --repo-type dataset --local-dir "$OUTPUT_DIR")  # interrupted downloads resume by default
for subset in $SUBSETS; do
    args+=(--include "$subset/**")
    args+=(--include "${subset}.tar*")
    args+=(--include "${subset}.zip")
done

hf "${args[@]}"

if [ "$EXTRACT" = "1" ]; then
    if ! command -v unzip >/dev/null 2>&1; then
        echo "unzip is required for EXTRACT=1. Install unzip or rerun with EXTRACT=0." >&2
        exit 2
    fi
    for subset in $SUBSETS; do
        zip_path=$OUTPUT_DIR/$subset.zip
        if [ -f "$zip_path" ]; then
            unzip -q -n "$zip_path" -d "$OUTPUT_DIR"
        fi
    done
fi

cat <<EOF
Downloaded DA3-BENCH subsets to:
  $OUTPUT_DIR

Materialize into the benchmark layout with:
  python scripts/preprocess/materialize_da3_bench.py --source_root "$OUTPUT_DIR" --benchmark_root $HOME/datasets/benchmarks --datasets $SUBSETS
EOF
