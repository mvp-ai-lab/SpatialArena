#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/Projects/SpatialArena}

if [ "$#" -ne 15 ]; then
    echo "Usage: $0 METHOD SUITE TRACK RUN_ID BENCHMARK_EVAL_ROOT CHECKPOINT_ROOT METHOD_CODE_ROOT BENCHMARK_AUX_ROOT EXP_ROOT REPORT_DIR EXP_NAME KEEP_PREDICTIONS KEEP_DEPTH_VISUALIZATIONS GENERATE_PDF_REPORT CASE_FAILURES_TSV" >&2
    exit 2
fi

METHOD=$1
SUITE=$2
TRACK=$3
RUN_ID=$4
BENCHMARK_EVAL_ROOT=$5
CHECKPOINT_ROOT=$6
METHOD_CODE_ROOT=$7
BENCHMARK_AUX_ROOT=$8
EXP_ROOT=$9
REPORT_DIR=${10}
EXP_NAME=${11}
KEEP_PREDICTIONS=${12}
KEEP_DEPTH_VISUALIZATIONS=${13}
GENERATE_PDF_REPORT=${14}
CASE_FAILURES_TSV=${15}

METHOD_ID=$METHOD
if [ "$TRACK" != "rgb" ]; then
    METHOD_ID="${METHOD}_${TRACK}"
fi

if [ ! -f "$CASE_FAILURES_TSV" ]; then
    echo "Case-failure TSV not found: $CASE_FAILURES_TSV" >&2
    exit 2
fi

if [[ "$SUITE" == full-* ]]; then
    if [ "$KEEP_PREDICTIONS" != "1" ] && [ "${ALLOW_FULL_BENCHMARK_PREDICTION_DELETION:-0}" != "1" ]; then
        echo "Refusing to repair $SUITE with KEEP_PREDICTIONS=$KEEP_PREDICTIONS." >&2
        echo "Full benchmark repairs must restore prediction artifacts; set KEEP_PREDICTIONS=1." >&2
        exit 7
    fi
    KEEP_PREDICTIONS=1
    if [ "${USE_CANONICAL_FULL_ARTIFACTS:-1}" = "1" ]; then
        FULL_ARTIFACT_NAMESPACE=${FULL_ARTIFACT_NAMESPACE:-full_standard}
        EXP_ROOT=${FULL_ARTIFACT_ROOT:-$BENCHMARK_EVAL_ROOT/artifacts/$FULL_ARTIFACT_NAMESPACE}
        EXP_NAME=${FULL_ARTIFACT_EXP_NAME:-benchmark_${SUITE}_${METHOD_ID}}
    fi
fi

CASE_LIST_ROOT=$BENCHMARK_EVAL_ROOT/repair_case_lists
mkdir -p "$CASE_LIST_ROOT"
CASE_LIST=$CASE_LIST_ROOT/${RUN_ID}_${METHOD}_${SLURM_JOB_ID:-manual}_failed_cases.txt

awk 'BEGIN {
         print "# Case list generated from case_failures.tsv."
         print "# Format: task|dataset|scene|interval"
     }
     FNR == 1 { next }
     NF >= 5 { print $2 "|" $3 "|" $4 "|" $5 }' "$CASE_FAILURES_TSV" > "$CASE_LIST"

if [ "$(awk 'FNR > 2 && NF { count++ } END { print count + 0 }' "$CASE_LIST")" -eq 0 ]; then
    echo "No failed cases listed in $CASE_FAILURES_TSV; nothing to rerun."
    exit 0
fi

# Inference then evaluation of just the failed cases, in this job (see engine_pair.sh).
source "$PROJECT_ROOT/scripts/test/engine_pair.sh"
run_engine_pair \
    "$METHOD" \
    "$SUITE" \
    "$TRACK" \
    "$RUN_ID" \
    "$BENCHMARK_EVAL_ROOT" \
    "$CHECKPOINT_ROOT" \
    "$METHOD_CODE_ROOT" \
    "$BENCHMARK_AUX_ROOT" \
    "$EXP_ROOT" \
    "$REPORT_DIR" \
    "$EXP_NAME" \
    "$KEEP_PREDICTIONS" \
    "$KEEP_DEPTH_VISUALIZATIONS" \
    "$GENERATE_PDF_REPORT" \
    "$CASE_LIST"
