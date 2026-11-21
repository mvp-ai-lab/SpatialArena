#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# Shared helpers for running a benchmark suite through the engine pair in spatialarena/templates/_engine/:
# run_method_benchmark_infer.sh (GPU inference, CASE_PHASE=inference) then run_method_benchmark_eval.sh (CPU-only
# verify + evaluate, CASE_PHASE=evaluate). Both take the same 15 positional arguments:
#
#   METHOD SUITE [TRACK RUN_ID BENCHMARK_EVAL_ROOT CHECKPOINT_ROOT METHOD_CODE_ROOT BENCHMARK_ROOT EXP_ROOT
#                 REPORT_DIR EXP_NAME KEEP_PREDICTIONS KEEP_DEPTH_VISUALIZATIONS GENERATE_PDF_REPORT CASE_FILE]
#
# (empty strings mean "use the default"). Source this file, then call
#
#   submit_engine_pair [infer sbatch options...] -- METHOD SUITE [...]   # two jobs, eval afterok on inference
#   run_engine_pair METHOD SUITE [...]                                     # both phases in the current job
#
# Both phases must see the same experiment and report directory. The full-standard suites derive them from
# RUN_ID (or a fixed default when it is unset), but the sampled suites name the experiment after
# SLURM_JOB_ID, which differs between the two jobs -- so submit_engine_pair pins EXP_NAME for those.

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

ENGINE_PROJECT_ROOT=${PROJECT_ROOT:-$HOME/Projects/SpatialArena}
ENGINE_DIR=${ENGINE_DIR:-$ENGINE_PROJECT_ROOT/spatialarena/templates/_engine}
# The eval script's own header asks for the cpu partition, which is drained on this cluster; run evaluation
# CPU-only on the gpu partition instead. Override with ENGINE_EVAL_SBATCH_ARGS (a space-separated list).
ENGINE_EVAL_SBATCH_ARGS=${ENGINE_EVAL_SBATCH_ARGS:---partition=gpu --gres=none}
ENGINE_EXPORT=${ENGINE_EXPORT:-NIL}

# Pad the positional arguments to 15 and pin EXP_NAME (11th) for suites whose default depends on SLURM_JOB_ID,
# unless the caller already set it (as an argument or an exported EXP_NAME).
# Prints one argument per line.
engine_pair_args() {
    local args=("$@")
    while [ "${#args[@]}" -lt 15 ]; do
        args+=("")
    done
    local method=${args[0]} suite=${args[1]} track=${args[2]:-rgb}
    if [[ "$suite" != full-* ]] && [ -z "${args[10]}" ] && [ -z "${EXP_NAME:-}" ]; then
        local method_id=$method
        [ "$track" != "rgb" ] && method_id=${method}_${track}
        local tag=${args[3]:-${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}}
        args[10]="benchmark_${suite//-/_}_${method_id}_${tag}"
    fi
    printf '%s\n' "${args[@]}"
}

# submit_engine_pair [infer sbatch options...] -- METHOD SUITE [...]
# Prints "<infer_job_id> <eval_job_id>".
submit_engine_pair() {
    local sbatch_opts=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
        sbatch_opts+=("$1")
        shift
    done
    if [ "${1:-}" != "--" ]; then
        echo "submit_engine_pair: missing '--' before the engine arguments" >&2
        return 2
    fi
    shift
    local args=()
    mapfile -t args < <(engine_pair_args "$@")
    local infer_id eval_id
    # shellcheck disable=SC2206
    local eval_opts=($ENGINE_EVAL_SBATCH_ARGS)
    infer_id=$(sbatch --parsable --export="$ENGINE_EXPORT" "${sbatch_opts[@]}" \
        "$ENGINE_DIR/run_method_benchmark_infer.sh" "${args[@]}") || return
    eval_id=$(sbatch --parsable --export="$ENGINE_EXPORT" "${eval_opts[@]}" --dependency="afterok:${infer_id}" \
        "$ENGINE_DIR/run_method_benchmark_eval.sh" "${args[@]}") || return
    echo "$infer_id $eval_id"
}

# run_engine_pair METHOD SUITE [...]: inference then evaluation inside the current allocation (for wrappers that are
# themselves submitted as one GPU job). Evaluation runs only if inference succeeded.
run_engine_pair() {
    local args=()
    mapfile -t args < <(engine_pair_args "$@")
    bash "$ENGINE_DIR/run_method_benchmark_infer.sh" "${args[@]}" &&
        bash "$ENGINE_DIR/run_method_benchmark_eval.sh" "${args[@]}"
}
