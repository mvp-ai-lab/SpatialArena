#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# CPU-only counterpart to run_method_benchmark.sh: runs CASE_PHASE=evaluate
# (verify_case + evaluate_case only -- no model, no GPU, see
# run_method_case_list.sh's CASE_PHASE docstring) against predictions a
# prior run_method_benchmark.sh (CASE_PHASE=inference) job already wrote.
# Identical positional-argument contract and env var resolution to
# run_method_benchmark.sh; only the SBATCH resource header and CASE_PHASE
# differ, so submit both against the same REPORT_DIR/EXP_ROOT/EXP_NAME with
# the eval job depending (--dependency=afterok) on the inference job.

#SBATCH --job-name=method-benchmark-eval
#SBATCH --output=/home/%u/tmp/method-benchmark-eval_%j.log
#SBATCH --error=/home/%u/tmp/method-benchmark-eval_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=24:00:00
# 320G (was 192G, originally 64G): DTU's evaluate_dtu_mvsnet (point-cloud
# downsample/mask/STL-comparison against up to ~1M-point reference clouds)
# OOM-killed a real run at 64G after 14 cumulative oom_kill events (job
# 131893, 2026-09-14). Replacing the O(N^2)-memory radius_neighbors
# downsample with Open3D's voxel_down_sample (see eval_dtu_mesh.py) cut
# that to 2 events even at 192G (job 132501, 2026-09-16) -- both on the
# first dtu cases run right after a batch of concurrent non-dtu cases
# drained, suggesting a residual page-cache/allocator pressure spike at
# that transition rather than a per-scene sizing problem. 320G adds further
# headroom on a 515G/256-CPU node that has consistently had 100G+ free.

set -euo pipefail

export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH=${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}
export SLURM_EXPORT_ENV=ALL
export CASE_PHASE=evaluate
# verify_case/evaluate_case (TSDF fusion, RANSAC/ICP alignment, metric
# computation) is single-threaded Python/Open3D per case -- confirmed via
# sstat on a real run (job 132011: ~84% of exactly one allocated core busy,
# the rest of --cpus-per-task idle), so serial dispatch wastes almost all of
# the allocated CPUs. Running several cases concurrently uses that idle
# capacity. dtu cases are exempted from concurrency inside
# run_method_case_list.sh (a separate, still-unresolved cumulative memory
# issue -- see the --mem=192G comment below), so this is safe for everything
# else. Matches --cpus-per-task=8 above with headroom, since each case only
# ever uses about one core.
export CASE_PARALLELISM=${CASE_PARALLELISM:-6}

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/Projects/SpatialArena}
# Must match run_method_case_list.sh's DEFAULT_BENCHMARK_PYTHON: its per-method env routing (e.g. GenCeption)
# only applies when PYTHON equals that default. IGGT runs in si_bench too (verified 2026-09-25).
DEFAULT_BENCHMARK_PYTHON=$HOME/anaconda3/envs/si_bench/bin/python
PYTHON=${PYTHON:-$DEFAULT_BENCHMARK_PYTHON}
if [ "${1:-}" != "" ]; then
    export METHOD=$1
fi
if [ "${2:-}" != "" ]; then
    export BENCHMARK_SUITE=$2
fi
if [ "${3:-}" != "" ]; then
    export BENCHMARK_TRACK=$3
fi
if [ "${4:-}" != "" ]; then
    export RUN_ID=$4
fi
if [ "${5:-}" != "" ]; then
    export BENCHMARK_EVAL_ROOT=$5
fi
if [ "${6:-}" != "" ]; then
    export CHECKPOINT_ROOT=$6
fi
if [ "${7:-}" != "" ]; then
    export METHOD_CODE_ROOT=$7
fi
if [ "${8:-}" != "" ]; then
    export BENCHMARK_AUX_ROOT=$8
    export BENCHMARK_ROOT=$8
fi
if [ "${9:-}" != "" ]; then
    export EXP_ROOT=$9
fi
if [ "${10:-}" != "" ]; then
    export REPORT_DIR=${10}
fi
if [ "${11:-}" != "" ]; then
    export EXP_NAME=${11}
fi
if [ "${12:-}" != "" ]; then
    export KEEP_PREDICTIONS=${12}
fi
if [ "${13:-}" != "" ]; then
    export KEEP_DEPTH_VISUALIZATIONS=${13}
fi
if [ "${14:-}" != "" ]; then
    export GENERATE_PDF_REPORT=${14}
fi
if [ "${15:-}" != "" ]; then
    export CASE_FILE=${15}
fi

BENCHMARK_AUX_ROOT=${BENCHMARK_AUX_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks}
BENCHMARK_EVAL_ROOT=${BENCHMARK_EVAL_ROOT:-$BENCHMARK_AUX_ROOT/_evaluation}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/_checkpoints}
export BENCHMARK_AUX_ROOT
export BENCHMARK_ROOT=${BENCHMARK_ROOT:-$BENCHMARK_AUX_ROOT}
export BENCHMARK_EVAL_ROOT
export CHECKPOINT_ROOT
select_benchmark_tmp_root() {
    local tmp_id=${SLURM_JOB_ID:-${RUN_ID:-method_benchmark}}
    local local_root=/tmp/${USER:-$(id -un)}/feedforward_benchmark/$tmp_id
    local fallback_root=${BENCHMARK_AUX_ROOT}/_scratch/${USER:-$(id -un)}/$tmp_id
    local min_kb=${BENCHMARK_TMP_MIN_KB:-104857600}
    local local_parent
    local avail_kb
    local_parent=$(dirname "$local_root")
    if mkdir -p "$local_parent" 2>/dev/null; then
        avail_kb=$(df -Pk "$local_parent" 2>/dev/null | awk 'NR == 2 {print $4}')
        if [ -n "$avail_kb" ] && [ "$avail_kb" -ge "$min_kb" ]; then
            printf '%s\n' "$local_root"
            return
        fi
    fi
    printf '%s\n' "$fallback_root"
}
export BENCHMARK_TMP_ROOT=${BENCHMARK_TMP_ROOT:-$(select_benchmark_tmp_root)}
export BENCHMARK_PY_TMP_ROOT=${BENCHMARK_PY_TMP_ROOT:-$BENCHMARK_TMP_ROOT/tmp}
export TMPDIR=$BENCHMARK_PY_TMP_ROOT
export TMP=$BENCHMARK_PY_TMP_ROOT
export TEMP=$BENCHMARK_PY_TMP_ROOT
export WANDB_DIR=${WANDB_DIR:-$BENCHMARK_TMP_ROOT/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$BENCHMARK_TMP_ROOT/wandb-cache}
export WANDB_ARTIFACT_DIR=${WANDB_ARTIFACT_DIR:-$BENCHMARK_TMP_ROOT/wandb-artifacts}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CHECKPOINT_ROOT/xdg}
# Default HF_HOME to the shared, pre-populated checkpoint store rather than a
# fresh per-RUN_ID cache, and force offline mode -- see run_method_benchmark_infer.sh
# for why (this eval job doesn't load models itself, but keep the two scripts
# consistent so cache/env behavior doesn't silently diverge between phases).
export HF_HOME=${HF_HOME:-$CHECKPOINT_ROOT/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export TORCH_HOME=${TORCH_HOME:-$CHECKPOINT_ROOT/torch}
export PREDICTION_CACHE_ROOT=${PREDICTION_CACHE_ROOT:-$BENCHMARK_EVAL_ROOT/artifacts/_prediction_cache}
mkdir -p "$BENCHMARK_TMP_ROOT" "$BENCHMARK_PY_TMP_ROOT" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$PREDICTION_CACHE_ROOT"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
fi

METHOD=${METHOD:-}
BENCHMARK_SUITE=${BENCHMARK_SUITE:-sampled}
BENCHMARK_TRACK=${BENCHMARK_TRACK:-rgb}
export BENCHMARK_SUITE
if [ "$BENCHMARK_TRACK" = "rgb" ]; then
    METHOD_EXP_ID=$METHOD
else
    # A masked-depth run with a non-default BENCHMARK_DEPTH_MASK_RATIO/_GRID carries it in the method name
    # (PI3_X-mask0.7_rgb_masked_depth), as in run_method_case_list.sh and the generated report folders.
    METHOD_VARIANT=""
    case "$BENCHMARK_TRACK" in
        rgb_masked_depth|rgb_masked_depth_cameras)
            METHOD_VARIANT=$(PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m spatialarena.depth_prior_masking \
                variant "$BENCHMARK_TRACK") || exit 2
            ;;
    esac
    METHOD_EXP_ID=${METHOD}${METHOD_VARIANT}_${BENCHMARK_TRACK}
fi
export BENCHMARK_TRACK

if [ -z "$METHOD" ]; then
    echo "Set METHOD or pass it as the first argument." >&2
    exit 2
fi
export PYTHON

case "$BENCHMARK_SUITE" in
    sampled|depth)
        export CASE_FILE=${CASE_FILE:-$PROJECT_ROOT/scripts/test/sampled_benchmark_cases.txt}
        export EXP_NAME=${EXP_NAME:-benchmark_sampled_${METHOD_EXP_ID}_${SLURM_JOB_ID:-manual}}
        ;;
    all-task)
        export CASE_FILE=${CASE_FILE:-$PROJECT_ROOT/scripts/test/all_task_sampled_cases.txt}
        export EXP_NAME=${EXP_NAME:-benchmark_all_task_${METHOD_EXP_ID}_${SLURM_JOB_ID:-manual}}
        ;;
    multiscene)
        export CASE_FILE=${CASE_FILE:-$PROJECT_ROOT/scripts/test/multiscene_sampled_cases.txt}
        export EXP_NAME=${EXP_NAME:-benchmark_multiscene_${METHOD_EXP_ID}_${SLURM_JOB_ID:-manual}}
        ;;
    monodepth)
        export CASE_FILE=${CASE_FILE:-$PROJECT_ROOT/scripts/test/monodepth_benchmark_cases.txt}
        export EXP_NAME=${EXP_NAME:-benchmark_monodepth_${METHOD_EXP_ID}_${SLURM_JOB_ID:-manual}}
        ;;
    sparse)
        export CASE_FILE=${CASE_FILE:-$PROJECT_ROOT/scripts/test/sparse_benchmark_cases.txt}
        export EXP_NAME=${EXP_NAME:-benchmark_sparse_${METHOD_EXP_ID}_${SLURM_JOB_ID:-manual}}
        ;;
    full-depth)
        export CASE_FILE=${CASE_FILE:-$PROJECT_ROOT/scripts/test/full_standard_depth_cases.txt}
        export EXP_NAME=${EXP_NAME:-benchmark_full_depth_${METHOD_EXP_ID}_${RUN_ID:-${SLURM_JOB_ID:-manual}}}
        export REPORT_DIR=${REPORT_DIR:-$BENCHMARK_EVAL_ROOT/reports/benchmark_full_reports/full-depth/$METHOD_EXP_ID}
        export KEEP_PREDICTIONS=${KEEP_PREDICTIONS:-1}
        export BENCHMARK_MAX_INPUT_FRAMES=${BENCHMARK_MAX_INPUT_FRAMES:-100}
        export MAX_IMAGES_PER_RUN=${MAX_IMAGES_PER_RUN:-100}
        ;;
    full-pose)
        export CASE_FILE=${CASE_FILE:-$PROJECT_ROOT/scripts/test/full_standard_sfm_cases.txt}
        export EXP_NAME=${EXP_NAME:-benchmark_full_pose_${METHOD_EXP_ID}_${RUN_ID:-${SLURM_JOB_ID:-manual}}}
        export REPORT_DIR=${REPORT_DIR:-$BENCHMARK_EVAL_ROOT/reports/benchmark_full_pose_reports/full-pose/$METHOD_EXP_ID}
        export KEEP_PREDICTIONS=${KEEP_PREDICTIONS:-1}
        export BENCHMARK_MAX_INPUT_FRAMES=${BENCHMARK_MAX_INPUT_FRAMES:-100}
        export MAX_IMAGES_PER_RUN=${MAX_IMAGES_PER_RUN:-100}
        export MAX_POINTS=${MAX_POINTS:-10000}
        export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
        ;;
    full-slam)
        # One SLAM range track (SLAM_RANGE: short = stride 10, long = stride 3): whole recordings in temporal order,
        # no frame cap or subsampling; only that track's methods run (see spatialarena/slam_protocol.py).
        export SLAM_RANGE=${SLAM_RANGE:-short}
        export CASE_FILE=${CASE_FILE:-$PROJECT_ROOT/scripts/test/full_standard_slam_cases_${SLAM_RANGE}.txt}
        export EXP_NAME=${EXP_NAME:-benchmark_full_slam_${SLAM_RANGE}_${METHOD_EXP_ID}_${RUN_ID:-${SLURM_JOB_ID:-manual}}}
        export REPORT_DIR=${REPORT_DIR:-$BENCHMARK_EVAL_ROOT/reports/benchmark_full_slam_reports/full-slam-${SLAM_RANGE}/$METHOD_EXP_ID}
        export KEEP_PREDICTIONS=${KEEP_PREDICTIONS:-1}
        export BENCHMARK_MAX_INPUT_FRAMES=0
        export MAX_IMAGES_PER_RUN=0
        export MAX_POINTS=${MAX_POINTS:-10000}
        export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
        ;;
    *)
        echo "Unsupported BENCHMARK_SUITE: $BENCHMARK_SUITE" >&2
        echo "Use one of: sampled all-task multiscene monodepth sparse full-depth full-pose full-slam" >&2
        exit 2
        ;;
esac

if [[ "$BENCHMARK_SUITE" == full-* ]] && [ "${USE_CANONICAL_FULL_ARTIFACTS:-1}" = "1" ]; then
    FULL_ARTIFACT_NAMESPACE=${FULL_ARTIFACT_NAMESPACE:-full_standard}
    export EXP_ROOT=${FULL_ARTIFACT_ROOT:-$BENCHMARK_EVAL_ROOT/artifacts/$FULL_ARTIFACT_NAMESPACE}
    if [ "$BENCHMARK_SUITE" = "full-slam" ]; then
        export EXP_NAME=${FULL_ARTIFACT_EXP_NAME:-benchmark_full-slam-${SLAM_RANGE}_${METHOD_EXP_ID}}
    else
        export EXP_NAME=${FULL_ARTIFACT_EXP_NAME:-benchmark_${BENCHMARK_SUITE}_${METHOD_EXP_ID}}
    fi
fi

# Full suites keep ONE report folder per (suite, method): reruns, retries and fixes overwrite in place, so every
# metric stays in one tree (spatialarena/report_layout.py holds the same layout). RUN_ID never changes it; a
# REPORT_DIR elsewhere is refused unless it is a one-off test folder under reports/_<name>/.
if [[ "$BENCHMARK_SUITE" == full-* ]]; then
    case "$BENCHMARK_SUITE" in
        full-depth) CANONICAL_REPORT_DIR=$BENCHMARK_EVAL_ROOT/reports/benchmark_full_reports/full-depth/$METHOD_EXP_ID ;;
        full-pose) CANONICAL_REPORT_DIR=$BENCHMARK_EVAL_ROOT/reports/benchmark_full_pose_reports/full-pose/$METHOD_EXP_ID ;;
        full-slam) CANONICAL_REPORT_DIR=$BENCHMARK_EVAL_ROOT/reports/benchmark_full_slam_reports/full-slam-${SLAM_RANGE}/$METHOD_EXP_ID ;;
    esac
    report_dir_real=$(realpath -m "$REPORT_DIR")
    case "$report_dir_real" in
        "$(realpath -m "$CANONICAL_REPORT_DIR")") ;;
        "$(realpath -m "$BENCHMARK_EVAL_ROOT/reports")"/_*/*) echo "Test report folder: $REPORT_DIR" ;;
        *)
            echo "Refusing REPORT_DIR=$REPORT_DIR for $BENCHMARK_SUITE." >&2
            echo "Full-suite results go to $CANONICAL_REPORT_DIR (reruns overwrite in place);" >&2
            echo "one-off tests go under $BENCHMARK_EVAL_ROOT/reports/_<name>/." >&2
            exit 8
            ;;
    esac
fi

if [[ "$BENCHMARK_SUITE" == full-* ]] && [ "$KEEP_PREDICTIONS" != "1" ] && [ "${ALLOW_FULL_BENCHMARK_PREDICTION_DELETION:-0}" != "1" ]; then
    echo "Refusing to run $BENCHMARK_SUITE with KEEP_PREDICTIONS=$KEEP_PREDICTIONS." >&2
    echo "Full benchmark runs must preserve prediction artifacts; set KEEP_PREDICTIONS=1 or explicitly set ALLOW_FULL_BENCHMARK_PREDICTION_DELETION=1." >&2
    exit 7
fi

# Self-contained: run_method_case_list.sh lives right next to this script
# (both are copied together into <experiment_root>/<experiment_name>/
# scripts/_engine/ by spatialarena.script_generator, or shipped together
# under spatialarena/templates/_engine/) -- not read from a separate
# repo-relative path, so this keeps working even if the original
# scripts/test/ copies this was cloned from are ever removed.
ENGINE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$ENGINE_DIR/run_method_case_list.sh" ]; then
    # `sbatch <this script>` runs a spooled copy (/.../slurmd/job<ID>/slurm_script), away from its sibling engine;
    # use the repo's copy (scripts/test/engine_pair.sh::submit_engine_pair submits these wrappers that way).
    ENGINE_DIR=$PROJECT_ROOT/spatialarena/templates/_engine
fi
RUNNER_SNAPSHOT="$BENCHMARK_TMP_ROOT/run_method_case_list_${SLURM_JOB_ID:-manual}.sh"
cp "$ENGINE_DIR/run_method_case_list.sh" "$RUNNER_SNAPSHOT"
chmod +x "$RUNNER_SNAPSHOT"
exec bash "$RUNNER_SNAPSHOT"
