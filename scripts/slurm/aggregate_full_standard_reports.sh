#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=bench-summary
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00

set -euo pipefail

HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}  # unset under sbatch --export=NIL
# Defaults come from the submitting user and directory (submit from the repository root).
CONDA_ROOT=${CONDA_ROOT:-${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}}
CONDA_ROOT=${CONDA_ROOT:-$HOME/anaconda3}
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -d "$PROJECT_ROOT/spatialarena" ]; then
    echo "ERROR: $PROJECT_ROOT is not the repository root; submit from there or set PROJECT_ROOT" >&2
    exit 1
fi
PYTHON=${PYTHON:-$CONDA_ROOT/envs/si_bench/bin/python}
# Generated results go to the evaluation root, not docs/.
BENCHMARK_EVAL_ROOT=${BENCHMARK_EVAL_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/_evaluation}
SUMMARY_DIR=${SUMMARY_DIR:-$BENCHMARK_EVAL_ROOT/reports/full_standard_summary}

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 REPORT_ROOT [REPORT_ROOT ...]" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p "$SUMMARY_DIR"

"$PYTHON" scripts/test/summarize_benchmark_reports.py \
    "$@" \
    --exclude-scannet \
    --output-markdown "$SUMMARY_DIR/full_standard_benchmark_summary.md" \
    --output-tsv "$SUMMARY_DIR/full_standard_benchmark_summary.tsv"
