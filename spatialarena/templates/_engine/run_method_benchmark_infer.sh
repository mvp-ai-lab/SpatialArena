#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=method-benchmark-infer
#SBATCH --output=/home/%u/tmp/method-benchmark-infer_%j.log
#SBATCH --error=/home/%u/tmp/method-benchmark-infer_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# GPU-only counterpart to run_method_benchmark.sh: runs CASE_PHASE=inference
# (adapter inference only -- produces and caches predictions, then stops;
# no verify/evaluate step, which never touches the GPU anyway -- see
# run_method_case_list.sh's CASE_PHASE docstring). Pair with
# run_method_benchmark_eval.sh (CPU-only, --dependency=afterok on this job)
# against the same REPORT_DIR/EXP_ROOT/EXP_NAME.

set -euo pipefail

export HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}
export PATH=${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}
export SLURM_EXPORT_ENV=ALL
export CASE_PHASE=inference
# Run several cases concurrently on the single allocated GPU instead of one
# at a time. Each case is a fresh adapter subprocess that reloads the whole
# model checkpoint before doing a small amount of compute (especially for
# tasks with few images per scene, e.g. SfM); serialized, the GPU sits mostly
# idle during the repeated load/teardown, which external low-GPU-usage
# monitors can flag and kill (observed on job 132009). Overlapping several
# cases' load+compute keeps the GPU busier. Matches --cpus-per-task=4 above.
# full-slam sets its own default below (one recording must have the GPU to itself), so remember whether the caller
# chose a value. INFER_CASE_PARALLELISM sets it for this phase only: a model item's env (exported into both its
# inference and evaluation scripts) uses it to limit GPU concurrency without also serializing the CPU evaluation.
CASE_PARALLELISM=${INFER_CASE_PARALLELISM:-${CASE_PARALLELISM:-}}
CASE_PARALLELISM_GIVEN=${CASE_PARALLELISM:+1}
export CASE_PARALLELISM=${CASE_PARALLELISM:-4}

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
# fresh per-RUN_ID cache, and force offline mode: from_pretrained() otherwise
# issues a network HEAD request on every single load to check for updates
# even when the file is already fully cached, and this cluster's outbound
# path to huggingface.co is slow/flaky enough that those checks can stall
# for many seconds -- with several cases loading concurrently (see
# CASE_PARALLELISM), that stall was long enough to look like "low GPU usage"
# to an external monitor and get a real job killed (132210, 2026-09-15).
# Override HF_HUB_OFFLINE=0 for a method whose checkpoint isn't cached yet.
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
        # Matches DA3-BENCH's own eval_bench.yaml max_frames: 100, applied
        # uniformly to its pose AND reconstruction eval modes -- must match
        # full-pose's value below and run_method_case_list.sh's own default
        # (this export pre-empts that default, since it runs first).
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
        # A range track admits a method only if its longest recording fits one H200 pass (slam_protocol), so a long or
        # large case gets the GPU to itself (four KITTI Odometry sequences at once ran ZipMap out of memory), and the
        # runtime and peak memory it reports are its own. Short cases (at most a few hundred frames) run two at a time.
        if [ -z "$CASE_PARALLELISM_GIVEN" ]; then
            if [ "$SLAM_RANGE" = short ]; then
                export CASE_PARALLELISM=2
            else
                export CASE_PARALLELISM=1
            fi
        fi
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
