#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=profile-max-frames
#SBATCH --output=/home/%u/tmp/profile-max-frames_%j.log
#SBATCH --error=/home/%u/tmp/profile-max-frames_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=12:00:00

# Find how many input frames one method can process in a single forward pass on one H200 (for the offline SLAM
# protocol, docs/benchmark/slam_protocol.md). For each frame count N it runs the real benchmark inference path
# (run_method_benchmark_infer.sh -> run_method_case_list.sh, CASE_PHASE=inference) on one long SLAM sequence with
# BENCHMARK_MAX_INPUT_FRAMES = MAX_IMAGES_PER_RUN = N (no chunking), one case at a time, no prediction cache, and
# samples GPU memory with nvidia-smi. The sweep stops at the first failing N.
#
# Usage: sbatch --export=NIL --wrap "env METHOD=VGGT [FRAME_COUNTS='100 200 ...'] bash scripts/test/profile_max_input_frames.sh"
# Output: ${PROFILE_ROOT}/<METHOD>_<TRACK>.tsv with, per N: frames predicted, status, nvidia-smi peak (whole process incl.
# CUDA context), wall seconds (incl. model load), and the adapter's own metrics.json "inference time" and
# "Peak GPU Memory" (scope differs per adapter; empty when the adapter does not report it).

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/Projects/SpatialArena}
METHOD=${METHOD:?set METHOD}
TRACK=${TRACK:-rgb}
FRAME_COUNTS=${FRAME_COUNTS:-100 200 300 400 500 750 1000 1500 2000 3000}
PYTHON_CHECK=${PYTHON_CHECK:-$HOME/anaconda3/envs/si_bench/bin/python}
# Must equal run_method_case_list.sh's DEFAULT_BENCHMARK_PYTHON so its per-method env routing applies; the older
# run_method_benchmark_infer.sh would otherwise export its stale default (envs/dogs, lacking e.g. torch_scatter).
BENCHMARK_PYTHON=${BENCHMARK_PYTHON:-$HOME/anaconda3/envs/si_bench/bin/python}
CASE_LINE=${CASE_LINE:-SLAM|tum_dynamics|rgbd_dataset_freiburg2_desk_with_person|1}
PROFILE_ROOT=${PROFILE_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/_evaluation/frame_profile}
WORK=${PROFILE_ROOT}/work/${METHOD}_${TRACK}_${SLURM_JOB_ID:-manual}
OUT=${PROFILE_ROOT}/${METHOD}_${TRACK}.tsv
INFER=${PROJECT_ROOT}/spatialarena/templates/_engine/run_method_benchmark_infer.sh

mkdir -p "$WORK"
[ -f "$OUT" ] || printf 'method\ttrack\tn_frames\tn_predicted\tstatus\tpeak_gpu_mib\tgpu_total_mib\tseconds\tjob\tnote\tadapter_inference_s\tadapter_peak_gpu_mib\n' > "$OUT"
printf '%s\n' "$CASE_LINE" > "$WORK/case.txt"
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "method=$METHOD track=$TRACK gpu_total_mib=$GPU_TOTAL host=$(hostname) case=$CASE_LINE"

for N in $FRAME_COUNTS; do
    RUN_DIR="$WORK/n${N}"
    mkdir -p "$RUN_DIR"
    SAMPLES="$RUN_DIR/gpu_mem.txt"
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 1; done ) > "$SAMPLES" &
    SAMPLER=$!
    START=$(date +%s)
    env -u SLURM_JOB_ID \
        PYTHON="$BENCHMARK_PYTHON" METHOD="$METHOD" FRAME_PROFILE=1 BENCHMARK_SUITE=full-pose BENCHMARK_TRACK="$TRACK" CASE_FILE="$WORK/case.txt" \
        CASE_PARALLELISM=1 BENCHMARK_MAX_INPUT_FRAMES="$N" MAX_IMAGES_PER_RUN="$N" USE_SHARED_PREDICTION_CACHE=0 \
        RUN_ID="profile_${METHOD}_${N}" EXP_ROOT="$RUN_DIR/exp" REPORT_DIR="$RUN_DIR/report" \
        USE_CANONICAL_FULL_ARTIFACTS=0 FULL_ARTIFACT_ROOT="$RUN_DIR/exp" \
        FULL_ARTIFACT_EXP_NAME="profile_${METHOD}_${TRACK}_n${N}" PREDICTION_CACHE_ROOT="$RUN_DIR/pred_cache" \
        EXP_NAME="profile_${METHOD}_${TRACK}_n${N}" KEEP_PREDICTIONS=0 ALLOW_FULL_BENCHMARK_PREDICTION_DELETION=1 KEEP_DEPTH_VISUALIZATIONS=0 \
        GENERATE_PDF_REPORT=0 \
        bash "$INFER" > "$RUN_DIR/infer.log" 2>&1
    CODE=$?
    END=$(date +%s)
    kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
    # Guard: profiling must never write into a real experiment (full-* suites default to the canonical store).
    EXP_USED=$(grep -m1 "^Experiment: " "$RUN_DIR/infer.log" | sed 's/^Experiment: //')
    if [ -n "$EXP_USED" ] && [[ "$EXP_USED" != "$RUN_DIR"/* ]]; then
        echo "ABORT: N=$N wrote to $EXP_USED, outside $RUN_DIR" | tee -a "$OUT"
        exit 3
    fi
    PEAK=$(sort -n "$SAMPLES" | tail -1)
    FAILURES=$(find "$RUN_DIR/report" -name '*failure*.tsv' -exec tail -n +2 {} + 2>/dev/null | wc -l)
    if grep -qiE "CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation" "$RUN_DIR/infer.log"; then
        STATUS=oom
    elif [ "$CODE" -ne 0 ] || [ "$FAILURES" -gt 0 ]; then
        STATUS=failed
    else
        STATUS=ok
    fi
    SPARSE=$(find "$RUN_DIR/exp" -path "*sparse/0/images.*" 2>/dev/null | head -1)
    N_PRED=$(PYTHONPATH="$PROJECT_ROOT" "$PYTHON_CHECK" -c "import sys; from spatialarena.geometry.read_write_model import read_model; print(len(read_model(sys.argv[1])[1]))" "$(dirname "$SPARSE")" 2>/dev/null | tail -1)
    if [ "$STATUS" = ok ] && [ "${N_PRED:-0}" != "$N" ]; then
        STATUS=wrong_count
    fi
    METRICS=$(find "$RUN_DIR/exp" -name metrics.json 2>/dev/null | head -1)
    ADAPTER=$("$PYTHON_CHECK" -c "import json,sys; d=json.load(open(sys.argv[1])); d=d[0] if isinstance(d,list) else d; print(d.get('inference time',''), d.get('Peak GPU Memory','') or '', sep='\t')" "$METRICS" 2>/dev/null || printf '\t')
    NOTE=$(grep -m1 -iE "error|exception" "$RUN_DIR/infer.log" | tr '\t' ' ' | cut -c1-160)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$METHOD" "$TRACK" "$N" "${N_PRED:-0}" "$STATUS" "${PEAK:-}" "$GPU_TOTAL" \
        "$((END - START))" "${SLURM_JOB_ID:-manual}" "${NOTE:-}" "${ADAPTER:-	}" | tee -a "$OUT"
    if [ "$STATUS" != ok ]; then
        echo "Stopping sweep at N=$N ($STATUS)"
        break
    fi
done
