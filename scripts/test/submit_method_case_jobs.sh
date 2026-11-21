#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

PROJECT_ROOT=$HOME/Projects/SpatialArena
BENCHMARK_JOB_MEM=${BENCHMARK_JOB_MEM:-128G}
BENCHMARK_MAX_IN_FLIGHT=${BENCHMARK_MAX_IN_FLIGHT:-4}
cd "$PROJECT_ROOT"
# shellcheck source=engine_pair.sh
source "$PROJECT_ROOT/scripts/test/engine_pair.sh"

suite=${1:-all-task}
if [ "$#" -gt 0 ]; then
    shift
fi

if [ "$#" -gt 0 ]; then
    methods=("$@")
else
    methods=(
        DA3-L DA3-G DA3-NGL VGGT PI3_X FASTVGGT COME CUT3R FAST3R VGGT_OMEGA
        MAPANYTHING OMNIVGGT DEJAVIEW IGGT IGGT4D POW3R DUST3R MAST3R ROBUSTVGGT
        DEPTHVLM VLM3 DEPTHLM SENSENOVA_VISION QWEN3_VL
    )
fi

validate_suite() {
    local suite_name=$1
    case "$suite_name" in
        sampled|depth|all-task|multiscene|sparse|monodepth|full-depth|full-pose|full-slam) ;;
        *)
            echo "Unsupported suite: $suite_name" >&2
            echo "Use one of: sampled depth all-task multiscene sparse monodepth full-depth full-pose full-slam" >&2
            return 1
            ;;
    esac
}

validate_suite "$suite"

job_ids=()
for method in "${methods[@]}"; do
    dependency_args=()
    if [ "$BENCHMARK_MAX_IN_FLIGHT" -gt 0 ] && [ "${#job_ids[@]}" -ge "$BENCHMARK_MAX_IN_FLIGHT" ]; then
        dependency_args=(--dependency="afterany:${job_ids[$((${#job_ids[@]} - BENCHMARK_MAX_IN_FLIGHT))]}")
    fi
    # Inference job (GPU), then an evaluation job (CPU) that starts after it succeeds; see engine_pair.sh.
    # For full-slam, SLAM_RANGE (short|long, default short) selects the range track.
    pair=$(ENGINE_EXPORT="NIL${SLAM_RANGE:+,SLAM_RANGE=$SLAM_RANGE}" submit_engine_pair \
        --mem="$BENCHMARK_JOB_MEM" \
        "${dependency_args[@]}" \
        -- "$method" "$suite")
    read -r job_id eval_job_id <<<"$pair"
    job_ids+=("$job_id")
    echo "$method $suite jobs: inference $job_id, evaluation $eval_job_id"
done

IFS=,
echo "Monitor with: squeue -j ${job_ids[*]}"
