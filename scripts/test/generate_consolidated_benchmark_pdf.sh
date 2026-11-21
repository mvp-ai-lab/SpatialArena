#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/Projects/SpatialArena}
PYTHON=${PYTHON:-$HOME/anaconda3/envs/dogs/bin/python}
BENCHMARK_AUX_ROOT=${BENCHMARK_AUX_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks}
BENCHMARK_EVAL_ROOT=${BENCHMARK_EVAL_ROOT:-$BENCHMARK_AUX_ROOT/_evaluation}
OUTPUT=${OUTPUT:-$BENCHMARK_EVAL_ROOT/reports/final_benchmark_report.pdf}
TITLE=${TITLE:-FeedForward Consolidated Benchmark Report}
EXCLUDE_SCANNET=${EXCLUDE_SCANNET:-1}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <report_root> [<report_root> ...]" >&2
    echo "Example: $0 $BENCHMARK_EVAL_ROOT/reports/benchmark_full_reports/*/*" >&2
    exit 2
fi

args=("$PYTHON" "$PROJECT_ROOT/scripts/test/generate_benchmark_pdf_report.py" "$@" --output "$OUTPUT" --title "$TITLE")
if [ "$EXCLUDE_SCANNET" = "1" ]; then
    args+=(--exclude-scannet)
fi
"${args[@]}"
