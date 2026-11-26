#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=method-case-list
#SBATCH --output=/home/%u/tmp/method-case-list_%j.log
#SBATCH --error=/home/%u/tmp/method-case-list_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/Projects/SpatialArena}
# si_bench is the default env for spatialarena and every method except GenCeption: Python 3.11, torch
# 2.6.0+cu124, AMB3R's source-built extensions (torch_scatter, pytorch3d, flash-attn) and the long-sequence
# SLAM deps (gtsam-develop, faiss, viser, sim3solve). Built by scripts/slurm/build_si_bench_env.sh.
DEFAULT_BENCHMARK_PYTHON=$HOME/anaconda3/envs/si_bench/bin/python
# IGGT (2026-09-25): verified via a real inference smoke test that si_bench
# already has everything IGGT's actual inference code path needs (torch,
# torchvision, transformers for Mask2FormerConfig, huggingface_hub, scipy) --
# its own requirements.txt is a full training-env freeze (apex, detectron2,
# RAPIDS cudf/cuml, a conflicting pytorch3d commit) but none of that is
# reachable from the benchmark's actual import chain, same pattern as
# AMB3R's dead xformers import. No dedicated env needed; falls through to
# DEFAULT_BENCHMARK_PYTHON below.
# GenCeption is JAX/Flax-based (see scripts/slurm/build_genception_env.sh),
# unlike every other registered method's PyTorch stack -- and needs
# tensorflow actively kept OUT of its env, so it can't share si_bench even
# though nothing else here forces it apart.
GENCEPTION_BENCHMARK_PYTHON=$HOME/anaconda3/envs/genception/bin/python
PYTHON=${PYTHON:-$DEFAULT_BENCHMARK_PYTHON}
BENCHMARK_AUX_ROOT=${BENCHMARK_AUX_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-$BENCHMARK_AUX_ROOT}
BENCHMARK_EVAL_ROOT=${BENCHMARK_EVAL_ROOT:-$BENCHMARK_AUX_ROOT/_evaluation}
EXP_ROOT=${EXP_ROOT:-$BENCHMARK_EVAL_ROOT/artifacts}
select_benchmark_tmp_root() {
    local tmp_id=${SLURM_JOB_ID:-${RUN_ID:-method_case_list}}
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
BENCHMARK_TMP_ROOT=${BENCHMARK_TMP_ROOT:-$(select_benchmark_tmp_root)}
BENCHMARK_PY_TMP_ROOT=${BENCHMARK_PY_TMP_ROOT:-$BENCHMARK_TMP_ROOT/tmp}
CAMERA_PRIOR_ROOT=${CAMERA_PRIOR_ROOT:-$BENCHMARK_EVAL_ROOT/camera_priors}
CASE_FILE=${CASE_FILE:-$PROJECT_ROOT/scripts/test/sampled_benchmark_cases.txt}
BENCHMARK_SUITE=${BENCHMARK_SUITE:-}
METHOD=${METHOD:-}
BENCHMARK_TRACK=${BENCHMARK_TRACK:-rgb}
if { [ "$METHOD" = "GENCEPTION_DEPTH" ] || [ "$METHOD" = "GENCEPTION_POSE" ]; } && [ "$PYTHON" = "$DEFAULT_BENCHMARK_PYTHON" ] && [ -x "$GENCEPTION_BENCHMARK_PYTHON" ]; then
    PYTHON=$GENCEPTION_BENCHMARK_PYTHON
fi
# torch.compile / Triton build kernels with $CC / $CXX (default: gcc / g++ on PATH). The compute nodes' system gcc
# cannot find its linker ("collect2: fatal error: cannot find 'ld'"), so use the conda GCC that si_bench ships
# (scripts/slurm/build_si_bench_env.sh) unless the caller set its own. Needed by LoGeR and ZipMap's TTT layers.
PYTHON_BIN_DIR=$(dirname "$PYTHON")
if [ -z "${CC:-}" ] && [ -x "$PYTHON_BIN_DIR/x86_64-conda-linux-gnu-gcc" ]; then
    export CC=$PYTHON_BIN_DIR/x86_64-conda-linux-gnu-gcc
fi
if [ -z "${CXX:-}" ] && [ -x "$PYTHON_BIN_DIR/x86_64-conda-linux-gnu-g++" ]; then
    export CXX=$PYTHON_BIN_DIR/x86_64-conda-linux-gnu-g++
fi
export BENCHMARK_TMP_ROOT
export BENCHMARK_PY_TMP_ROOT
export PYTHON
export TMPDIR=$BENCHMARK_PY_TMP_ROOT
export TMP=$BENCHMARK_PY_TMP_ROOT
export TEMP=$BENCHMARK_PY_TMP_ROOT
export WANDB_DIR=${WANDB_DIR:-$BENCHMARK_TMP_ROOT/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$BENCHMARK_TMP_ROOT/wandb-cache}
export WANDB_ARTIFACT_DIR=${WANDB_ARTIFACT_DIR:-$BENCHMARK_TMP_ROOT/wandb-artifacts}
mkdir -p "$BENCHMARK_TMP_ROOT" "$BENCHMARK_PY_TMP_ROOT" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

METHOD_REGISTRY=$PROJECT_ROOT/scripts/test/benchmark_methods.py
# Per job, not a fixed /tmp file another user (or a concurrent job) may own.
METHOD_LOOKUP_ERROR=$BENCHMARK_TMP_ROOT/method_lookup_error_$$.txt
if METHOD_CANONICAL=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD" --track "$BENCHMARK_TRACK" --field name 2>"$METHOD_LOOKUP_ERROR"); then
    METHOD_ADAPTER=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field adapter)
    METHOD_CODE_ENV=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field code_env)
    METHOD_CODE_DEFAULT=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field code_default)
    METHOD_CKPT_ENV=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field ckpt_env)
    METHOD_CKPT_DEFAULT=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field ckpt_default)
    METHOD_NATIVE=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field native)
    # Tasks the method takes part in (empty: every task); cases of other tasks are skipped as not applicable.
    METHOD_TASKS=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field tasks)
    METHOD_COMMAND_ENV=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field external_command_env)
    METHOD_RESOLUTION_DEFAULT=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field input_resolution)
    METHOD_IMAGE_LOAD_RESOLUTION_DEFAULT=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field image_load_resolution)
    METHOD_PIXEL_LIMIT_DEFAULT=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field pixel_limit)
    METHOD_CONF_THRESH_DEFAULT=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field conf_thresh)
    METHOD_DEPTH_CONF_THRESH_DEFAULT=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field depth_conf_thresh)
    METHOD_RESOLUTION_SOURCE=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field resolution_source)
    METHOD_DEPTH_PROTOCOL=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field default_depth_protocol)
    METHOD_POINT_GRID_SIZE_DEFAULT=$("$PYTHON" "$METHOD_REGISTRY" --method "$METHOD_CANONICAL" --track "$BENCHMARK_TRACK" --field point_grid_size)
elif [ -n "${METHOD_COMMAND:-}" ]; then
    # Fully custom/unregistered method: the caller supplies its own inference
    # command via METHOD_COMMAND (see external_command_adapter.py and
    # BenchmarkEvaluator.select_models()'s custom-adapter item -- the
    # {"name", "track", "command"} extension point the API reference
    # (docs/benchmark/evaluator_quickstart.md, select_models) describes). Registry resolution isn't required
    # in that case, so don't hard-fail just because the name isn't registered.
    # If METHOD_COMMAND is unset, an unresolved name still hard-fails exactly
    # as before -- this branch only ever fires when the caller has explicitly
    # opted into custom-command mode.
    echo "[$METHOD] not found in the method registry; treating as a custom adapter (METHOD_COMMAND is set)." >&2
    METHOD_CANONICAL=$METHOD
    METHOD_ADAPTER=external_command
    # METHOD_CODE_ENV/METHOD_CKPT_ENV may already be exported by the caller
    # (e.g. a generated wrapper script setting METHOD_CODE_ENV=_CUSTOM_
    # METHOD_CODE_PATH plus that indirect variable itself, for a custom
    # item's code_path/ckpt_path override -- see BenchmarkEvaluator.
    # _custom_adapter_env()); only default to empty if unset.
    METHOD_CODE_ENV=${METHOD_CODE_ENV:-}
    METHOD_CODE_DEFAULT=""
    METHOD_CKPT_ENV=${METHOD_CKPT_ENV:-}
    METHOD_CKPT_DEFAULT=""
    METHOD_NATIVE=0
    METHOD_COMMAND_ENV=""
    METHOD_RESOLUTION_DEFAULT=${RESOLUTION:-224}
    METHOD_IMAGE_LOAD_RESOLUTION_DEFAULT=${IMAGE_LOAD_RESOLUTION:-224}
    METHOD_PIXEL_LIMIT_DEFAULT=${PIXEL_LIMIT:-0}
    METHOD_CONF_THRESH_DEFAULT=${CONF_THRESH:-0.0}
    METHOD_DEPTH_CONF_THRESH_DEFAULT=${DEPTH_CONF_THRESH:-0.0}
    METHOD_RESOLUTION_SOURCE="custom"
else
    cat "$METHOD_LOOKUP_ERROR" >&2
    echo "Supported methods:" >&2
    "$PYTHON" "$METHOD_REGISTRY" --list >&2
    exit 2
fi
METHOD=$METHOD_CANONICAL
if [ "$BENCHMARK_TRACK" = "rgb" ]; then
    METHOD_ID=${METHOD_ID:-$METHOD}
else
    # A masked-depth run with a non-default BENCHMARK_DEPTH_MASK_RATIO/_GRID carries it in the method name
    # (PI3_X-mask0.7_rgb_masked_depth; spatialarena/depth_prior_masking.py, script_generator._method_id).
    METHOD_VARIANT=""
    case "$BENCHMARK_TRACK" in
        rgb_masked_depth|rgb_masked_depth_cameras)
            METHOD_VARIANT=$(PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m spatialarena.depth_prior_masking \
                variant "$BENCHMARK_TRACK") || exit 2
            ;;
    esac
    METHOD_ID=${METHOD_ID:-${METHOD}${METHOD_VARIANT}_${BENCHMARK_TRACK}}
fi

if [ ! -f "$CASE_FILE" ]; then
    echo "Case file not found: $CASE_FILE" >&2
    exit 2
fi

    if [ -z "$BENCHMARK_SUITE" ]; then
        case "$(basename "$CASE_FILE")" in
        full_standard_depth_cases.txt|full_standard_depth_cases_no_scannet.txt)
            BENCHMARK_SUITE=full-depth
            ;;
        full_standard_sfm_cases.txt)
            BENCHMARK_SUITE=full-pose
            ;;
        full_standard_slam_cases_short.txt|full_standard_slam_cases_long.txt|full_standard_slam_cases_large.txt)
            BENCHMARK_SUITE=full-slam
            ;;
        *)
            BENCHMARK_SUITE=sampled
            ;;
    esac
fi
export BENCHMARK_SUITE

# full-slam (docs/benchmark/slam_protocol.md): every case is one continuous recording fed in order, whole, at the
# stride its dataset has in the SLAM range track SLAM_RANGE (short / long / large; e.g. TUM stride 10 in short;
# each case line's interval must equal it), with no frame cap and no subsampling. Only the methods of that range
# track run (spatialarena/slam_protocol.py); the range is part of the experiment name, so the tracks never share
# predictions.
SLAM_RANGE=${SLAM_RANGE:-short}
export SLAM_RANGE
# The stride of one dataset in SLAM_RANGE (fails for a dataset that is not part of the range).
slam_case_stride() {
    PYTHONPATH="$PROJECT_ROOT" "$PYTHON" -m spatialarena.slam_protocol stride "$SLAM_RANGE" "$1"
}
if [ "$BENCHMARK_SUITE" = "full-slam" ]; then
    PYTHONPATH="$PROJECT_ROOT" "$PYTHON" -m spatialarena.slam_protocol datasets "$SLAM_RANGE" > /dev/null || exit 2
    BENCHMARK_MAX_INPUT_FRAMES=${BENCHMARK_MAX_INPUT_FRAMES:-0}
    MAX_IMAGES_PER_RUN=${MAX_IMAGES_PER_RUN:-0}
    if [ "$BENCHMARK_MAX_INPUT_FRAMES" != "0" ] || [ "$MAX_IMAGES_PER_RUN" != "0" ]; then
        echo "full-slam feeds whole strided sequences: BENCHMARK_MAX_INPUT_FRAMES and MAX_IMAGES_PER_RUN must be 0" >&2
        exit 2
    fi
fi
if [ -z "${BENCHMARK_MAX_INPUT_FRAMES+x}" ]; then
    case "$BENCHMARK_SUITE" in
        full-pose)
            BENCHMARK_MAX_INPUT_FRAMES=100
            ;;
        *)
            # Was 0 (disabled) for full-depth: da3.py's own --max_input_frames
            # cap (matching DA3-BENCH's eval_bench.yaml max_frames sampling)
            # was only ever applied for full-pose, leaving MVRecon uncapped
            # and letting large scenes fall through to --max_images_per_run
            # batching across multiple independent, coordinate-inconsistent
            # forward passes.
            BENCHMARK_MAX_INPUT_FRAMES=100
            ;;
    esac
fi
export BENCHMARK_MAX_INPUT_FRAMES
BENCHMARK_IMAGE_ORDER=${BENCHMARK_IMAGE_ORDER:-}
export BENCHMARK_IMAGE_ORDER

if [[ "$BENCHMARK_SUITE" == full-* ]] && [ "${USE_CANONICAL_FULL_ARTIFACTS:-1}" = "1" ]; then
    FULL_ARTIFACT_NAMESPACE=${FULL_ARTIFACT_NAMESPACE:-full_standard}
    EXP_ROOT=${FULL_ARTIFACT_ROOT:-$BENCHMARK_EVAL_ROOT/artifacts/$FULL_ARTIFACT_NAMESPACE}
    if [ "$BENCHMARK_SUITE" = "full-slam" ]; then
        EXP_NAME=${FULL_ARTIFACT_EXP_NAME:-benchmark_full-slam-${SLAM_RANGE}_${METHOD_ID}}
    else
        EXP_NAME=${FULL_ARTIFACT_EXP_NAME:-benchmark_${BENCHMARK_SUITE}_${METHOD_ID}}
    fi
fi

PREDICTION_CACHE_ROOT=${PREDICTION_CACHE_ROOT:-$BENCHMARK_EVAL_ROOT/artifacts/_prediction_cache}
if [ -z "${USE_SHARED_PREDICTION_CACHE+x}" ]; then
    case "$BENCHMARK_SUITE" in
        full-*)
            USE_SHARED_PREDICTION_CACHE=1
            ;;
        *)
            USE_SHARED_PREDICTION_CACHE=0
            ;;
    esac
fi
export PREDICTION_CACHE_ROOT
export USE_SHARED_PREDICTION_CACHE

# CASE_PHASE lets a case be split across two separate SLURM jobs -- a
# GPU-allocated "inference" job (runs the adapter, produces predictions,
# stops) and a CPU-only "evaluate" job (verify + evaluate, which never touch
# a model or the GPU -- see prediction_output_ready/verify_case/
# evaluate_case, all pure filesystem/numpy/Open3D work). Default "both"
# preserves the original single-job behavior exactly.
CASE_PHASE=${CASE_PHASE:-both}
case "$CASE_PHASE" in
    inference|evaluate|both) ;;
    *)
        echo "Invalid CASE_PHASE: $CASE_PHASE (expected inference, evaluate, or both)" >&2
        exit 2
        ;;
esac
export CASE_PHASE

# VGGT-Long/Pi-Long spend most of a long sequence aligning chunks on the CPU after a short GPU inference; in a GPU
# job that reads as "low GPU usage" and gets the job killed. With SLAM_DEFER_ALIGN=1 (default) a CASE_PHASE=inference
# job runs only their GPU phase and leaves slam_align_pending.json in the case's prediction directory; the
# CASE_PHASE=evaluate (or both) job then runs the alignment on its CPU node before verify/evaluate
# (scripts/inference/long_sequence_phase_launcher.py). SLAM_DEFER_ALIGN=0 runs both phases in the GPU job.
SLAM_DEFER_ALIGN=${SLAM_DEFER_ALIGN:-1}
export SLAM_DEFER_ALIGN

# CASE_PARALLELISM bounds how many cases run concurrently. Default 1
# preserves the original strictly-sequential behavior. A GPU-bound
# CASE_PHASE=inference job benefits from running several small cases at once
# (e.g. SfM scenes with only a handful of images each): each case is a fresh
# subprocess that reloads the whole model checkpoint onto the GPU before
# doing a small amount of compute, so serial execution leaves the GPU mostly
# idle during the repeated load/teardown and can read as "low GPU usage" to
# an external monitor. Running N cases concurrently overlaps those load
# phases and keeps the GPU busier. Left at 1 for CASE_PHASE=evaluate (CPU
# TSDF/ICP work, already memory constrained -- see run_method_benchmark_eval.sh).
CASE_PARALLELISM=${CASE_PARALLELISM:-1}
if ! [[ "$CASE_PARALLELISM" =~ ^[0-9]+$ ]] || [ "$CASE_PARALLELISM" -lt 1 ]; then
    echo "Invalid CASE_PARALLELISM: $CASE_PARALLELISM (expected a positive integer)" >&2
    exit 2
fi
export CASE_PARALLELISM

EXP_NAME=${EXP_NAME:-benchmark_sampled_${METHOD_ID}_${SLURM_JOB_ID:-manual}}
REPORT_DIR=${REPORT_DIR:-$BENCHMARK_EVAL_ROOT/reports/benchmark_sampled_reports/$EXP_NAME/$METHOD_ID}
# All checkpoints (HF-hub-style and previously-local-file-style alike) are
# read from this one shared, pre-populated store rather than a private path
# or a fresh per-RUN_ID download -- see HF_HOME in run_method_benchmark*.sh
# for the HF-hub half of this; SHARED_CHECKPOINT_ROOT/local holds the ones
# that were plain local files (VGGT, FastVGGT, VGGT-Omega, DA3-Giant, CUT3R,
# IGGT), migrated here from $HOME/Projects/FoundationModels/pretrained
# and per-RUN_ID CHECKPOINT_ROOT copies on 2026-09-15.
SHARED_CHECKPOINT_ROOT=${SHARED_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/_checkpoints}}
DA3_CKPT_PATH=${DA3_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/da3_giant}
DA3_L_CKPT_PATH=${DA3_L_CKPT_PATH:-depth-anything/DA3-LARGE-1.1}
DA3_G_CKPT_PATH=${DA3_G_CKPT_PATH:-$DA3_CKPT_PATH}
DA3_NGL_CKPT_PATH=${DA3_NGL_CKPT_PATH:-depth-anything/DA3NESTED-GIANT-LARGE-1.1}
VGGT_CKPT_PATH=${VGGT_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/vggt/vggt.pt}
AMB3R_CKPT_PATH=${AMB3R_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/amb3r/amb3r.pt}
METHOD_CODE_ROOT=${METHOD_CODE_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/_model_codes}
DA3_CODE_PATH=${DA3_CODE_PATH:-$METHOD_CODE_ROOT/depth-anything-3/src}
VGGT_CODE_PATH=${VGGT_CODE_PATH:-$METHOD_CODE_ROOT/RobustVGGT}
AMB3R_CODE_PATH=${AMB3R_CODE_PATH:-$METHOD_CODE_ROOT/amb3r}
PI3_CKPT_PATH=${PI3_CKPT_PATH:-yyfz233/Pi3X}
FASTVGGT_CKPT_PATH=${FASTVGGT_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/fastvggt/model_tracker_fixed_e20.pt}
COME_CKPT_PATH=${COME_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/come/checkpoint.pth}
CUT3R_CKPT_PATH=${CUT3R_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/cut3r/cut3r_512_dpt_4_64.pth}
FAST3R_CKPT_PATH=${FAST3R_CKPT_PATH:-jedyang97/Fast3R_ViT_Large_512}
VGGT_OMEGA_CKPT_PATH=${VGGT_OMEGA_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/vggt_omega/vggt_omega_1b_512.pt}
VGGT_OMEGA_TEXT_CKPT_PATH=${VGGT_OMEGA_TEXT_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/vggt_omega/vggt_omega_1b_256_text.pt}
MAPANYTHING_CKPT_PATH=${MAPANYTHING_CKPT_PATH:-facebook/map-anything}
OMNIVGGT_CKPT_PATH=${OMNIVGGT_CKPT_PATH:-Livioni/OmniVGGT}
DEJAVIEW_CKPT_PATH=${DEJAVIEW_CKPT_PATH:-nvidia/dvlt}
IGGT_LOCAL_CKPT_PATH=$SHARED_CHECKPOINT_ROOT/local/iggt/iggt_checkpoint.pth
if [ -z "${IGGT_CKPT_PATH:-}" ] && [ -f "$IGGT_LOCAL_CKPT_PATH" ]; then
    IGGT_CKPT_PATH=$IGGT_LOCAL_CKPT_PATH
else
    IGGT_CKPT_PATH=${IGGT_CKPT_PATH:-lifuguan/IGGT_official}
fi
IGGT4D_CKPT_PATH=${IGGT4D_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/iggt4d/model.safetensors}
# Streaming SLAM systems (spatialarena/models/streaming_slam.py); their shared weights (SALAD, DINOv2) are read
# from SHARED_CHECKPOINT_ROOT directly.
LOGER_CKPT_PATH=${LOGER_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/loger/LoGeR/latest.pt}
ZIPMAP_CKPT_PATH=${ZIPMAP_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/zipmap/checkpoint_aff_inv.pt}
LINGBOT_MAP_CKPT_PATH=${LINGBOT_MAP_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/lingbot_map/lingbot-map.pt}
R3_CKPT_PATH=${R3_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/r3/r3.safetensors}
SCAL3R_CKPT_PATH=${SCAL3R_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/scal3r/scal3r.pt}
ABOT_RECON_CKPT_PATH=${ABOT_RECON_CKPT_PATH:-$SHARED_CHECKPOINT_ROOT/local/abot_recon/abot_recon.safetensors}
POW3R_CKPT_PATH=${POW3R_CKPT_PATH:-https://download.europe.naverlabs.com/ComputerVision/Pow3R/Pow3R_ViTLarge_BaseDecoder_512_linear.pth}
DUST3R_CKPT_PATH=${DUST3R_CKPT_PATH:-https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}
MAST3R_CKPT_PATH=${MAST3R_CKPT_PATH:-https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}
ROBUSTVGGT_CKPT_PATH=${ROBUSTVGGT_CKPT_PATH:-facebook/VGGT-1B}
DEPTHVLM_CKPT_PATH=${DEPTHVLM_CKPT_PATH:-JonnyYu828/DepthVLM-4B}
VLM3_CKPT_PATH=${VLM3_CKPT_PATH:-facebook/VLM3-depth}
DEPTHLM_CKPT_PATH=${DEPTHLM_CKPT_PATH:-facebook/DepthLM}
SENSENOVA_VISION_CKPT_PATH=${SENSENOVA_VISION_CKPT_PATH:-sensenova/SenseNova-Vision-7B-MoT}
QWEN3_VL_CKPT_PATH=${QWEN3_VL_CKPT_PATH:-Qwen/Qwen3-VL-8B-Instruct}
GENCEPTION_CKPT_DIR=${GENCEPTION_CKPT_DIR:-$SHARED_CHECKPOINT_ROOT/local/genception}
GENCEPTION_EMBEDDINGS_DIR=${GENCEPTION_EMBEDDINGS_DIR:-$GENCEPTION_CKPT_DIR/prompt_embeddings}
GENCEPTION_MODEL_SIZE=${GENCEPTION_MODEL_SIZE:-14b}
PI3_CODE_PATH=${PI3_CODE_PATH:-$METHOD_CODE_ROOT/Pi3}
FASTVGGT_CODE_PATH=${FASTVGGT_CODE_PATH:-$METHOD_CODE_ROOT/RobustVGGT}
COME_CODE_PATH=${COME_CODE_PATH:-$METHOD_CODE_ROOT/CoMe}
CUT3R_CODE_PATH=${CUT3R_CODE_PATH:-$METHOD_CODE_ROOT/CUT3R}
FAST3R_CODE_PATH=${FAST3R_CODE_PATH:-$METHOD_CODE_ROOT/fast3r}
VGGT_OMEGA_CODE_PATH=${VGGT_OMEGA_CODE_PATH:-$METHOD_CODE_ROOT/vggt-omega}
MAPANYTHING_CODE_PATH=${MAPANYTHING_CODE_PATH:-$METHOD_CODE_ROOT/map-anything}
OMNIVGGT_CODE_PATH=${OMNIVGGT_CODE_PATH:-$METHOD_CODE_ROOT/OmniVGGT-official}
DEJAVIEW_CODE_PATH=${DEJAVIEW_CODE_PATH:-$METHOD_CODE_ROOT/dvlt}
IGGT_CODE_PATH=${IGGT_CODE_PATH:-$METHOD_CODE_ROOT/IGGT_official}
IGGT4D_CODE_PATH=${IGGT4D_CODE_PATH:-$METHOD_CODE_ROOT/IGGT4D}
POW3R_CODE_PATH=${POW3R_CODE_PATH:-$METHOD_CODE_ROOT/pow3r}
DUST3R_CODE_PATH=${DUST3R_CODE_PATH:-$METHOD_CODE_ROOT/dust3r}
MAST3R_CODE_PATH=${MAST3R_CODE_PATH:-$METHOD_CODE_ROOT/mast3r}
ROBUSTVGGT_CODE_PATH=${ROBUSTVGGT_CODE_PATH:-$METHOD_CODE_ROOT/RobustVGGT}
DEPTHVLM_CODE_PATH=${DEPTHVLM_CODE_PATH:-$METHOD_CODE_ROOT/DepthVLM}
VLM3_CODE_PATH=${VLM3_CODE_PATH:-$METHOD_CODE_ROOT/VLM3}
DEPTHLM_CODE_PATH=${DEPTHLM_CODE_PATH:-$METHOD_CODE_ROOT/DepthLM_Official}
SENSENOVA_VISION_CODE_PATH=${SENSENOVA_VISION_CODE_PATH:-$METHOD_CODE_ROOT/SenseNova-Vision}
QWEN3_VL_CODE_PATH=${QWEN3_VL_CODE_PATH:-$METHOD_CODE_ROOT/qwen3-vl}
GENCEPTION_CODE_PATH=${GENCEPTION_CODE_PATH:-$METHOD_CODE_ROOT/representations4d}
COME_MODEL=${COME_MODEL:-co_me_fused_vggt_3x3}
# The models read their model-specific settings (GENCEPTION_*, COME_MODEL, VGGT_OMEGA_TEXT_CKPT_PATH,
# MAPANYTHING_CODE_PATH for the DUSt3R family, ...) and these defaults from the environment.
export IGGT_CKPT_PATH
export SHARED_CHECKPOINT_ROOT DA3_CKPT_PATH DA3_L_CKPT_PATH DA3_G_CKPT_PATH DA3_NGL_CKPT_PATH VGGT_CKPT_PATH
export AMB3R_CKPT_PATH METHOD_CODE_ROOT DA3_CODE_PATH VGGT_CODE_PATH AMB3R_CODE_PATH PI3_CKPT_PATH
export FASTVGGT_CKPT_PATH COME_CKPT_PATH CUT3R_CKPT_PATH FAST3R_CKPT_PATH VGGT_OMEGA_CKPT_PATH
export VGGT_OMEGA_TEXT_CKPT_PATH MAPANYTHING_CKPT_PATH OMNIVGGT_CKPT_PATH DEJAVIEW_CKPT_PATH IGGT4D_CKPT_PATH
export POW3R_CKPT_PATH DUST3R_CKPT_PATH MAST3R_CKPT_PATH ROBUSTVGGT_CKPT_PATH DEPTHVLM_CKPT_PATH VLM3_CKPT_PATH
export DEPTHLM_CKPT_PATH SENSENOVA_VISION_CKPT_PATH QWEN3_VL_CKPT_PATH GENCEPTION_CKPT_DIR GENCEPTION_EMBEDDINGS_DIR
export GENCEPTION_MODEL_SIZE PI3_CODE_PATH FASTVGGT_CODE_PATH COME_CODE_PATH CUT3R_CODE_PATH FAST3R_CODE_PATH
export VGGT_OMEGA_CODE_PATH MAPANYTHING_CODE_PATH OMNIVGGT_CODE_PATH DEJAVIEW_CODE_PATH IGGT_CODE_PATH
export IGGT4D_CODE_PATH POW3R_CODE_PATH DUST3R_CODE_PATH MAST3R_CODE_PATH ROBUSTVGGT_CODE_PATH DEPTHVLM_CODE_PATH
export VLM3_CODE_PATH DEPTHLM_CODE_PATH SENSENOVA_VISION_CODE_PATH QWEN3_VL_CODE_PATH GENCEPTION_CODE_PATH
export COME_MODEL LOGER_CKPT_PATH ZIPMAP_CKPT_PATH LINGBOT_MAP_CKPT_PATH R3_CKPT_PATH SCAL3R_CKPT_PATH
export ABOT_RECON_CKPT_PATH
if [ -n "$METHOD_CODE_ENV" ]; then
    eval "$METHOD_CODE_ENV=\${$METHOD_CODE_ENV:-$METHOD_CODE_DEFAULT}"
    export "${METHOD_CODE_ENV?}"
fi
if [ -n "$METHOD_CKPT_ENV" ]; then
    eval "$METHOD_CKPT_ENV=\${$METHOD_CKPT_ENV:-$METHOD_CKPT_DEFAULT}"
    export "${METHOD_CKPT_ENV?}"
fi
RESOLUTION=${RESOLUTION:-$METHOD_RESOLUTION_DEFAULT}
IMAGE_LOAD_RESOLUTION=${IMAGE_LOAD_RESOLUTION:-$METHOD_IMAGE_LOAD_RESOLUTION_DEFAULT}
# full-slam feeds every method the same pre-sized frames (slam_protocol.DATASET_INPUT_SIZE, written by
# scripts/preprocess/materialize_slam_inputs.py): adapters get the protocol width as both resolutions and
# FIXED_INPUT_RESOLUTION=1, which tells them to use the frames as they are (no rescale; 16 px-patch models only
# center-crop to their patch grid). Frame-budget profiles (FRAME_PROFILE=1, SLAM recordings only) use the same
# inputs, so budgets are measured at the resolution the track runs at.
SLAM_FIXED_INPUTS=0
if [ "$BENCHMARK_SUITE" = "full-slam" ] || [ "${FRAME_PROFILE:-0}" = 1 ]; then
    SLAM_FIXED_INPUTS=1
fi
FIXED_INPUT_RESOLUTION=0
if [ "$SLAM_FIXED_INPUTS" = 1 ]; then
    SLAM_INPUT_WIDTH=$(PYTHONPATH="$PROJECT_ROOT" "$PYTHON" -c \
        "from spatialarena.slam_protocol import SLAM_INPUT_WIDTH; print(SLAM_INPUT_WIDTH)")
    if [ "$RESOLUTION" != "$SLAM_INPUT_WIDTH" ] && [ "$RESOLUTION" != "$METHOD_RESOLUTION_DEFAULT" ]; then
        echo "full-slam fixes the input resolution at $SLAM_INPUT_WIDTH px wide; RESOLUTION=$RESOLUTION is ignored" >&2
    fi
    RESOLUTION=$SLAM_INPUT_WIDTH
    IMAGE_LOAD_RESOLUTION=$SLAM_INPUT_WIDTH
    FIXED_INPUT_RESOLUTION=1
fi
export FIXED_INPUT_RESOLUTION
if [ -z "${PIXEL_LIMIT:-}" ]; then
    if [ "${METHOD_PIXEL_LIMIT_DEFAULT:-0}" -gt 0 ]; then
        PIXEL_LIMIT=$METHOD_PIXEL_LIMIT_DEFAULT
    else
        PIXEL_LIMIT=$((RESOLUTION * RESOLUTION))
    fi
fi
CONF_THRESH=${CONF_THRESH:-$METHOD_CONF_THRESH_DEFAULT}
DEPTH_CONF_THRESH=${DEPTH_CONF_THRESH:-$METHOD_DEPTH_CONF_THRESH_DEFAULT}
export RESOLUTION IMAGE_LOAD_RESOLUTION PIXEL_LIMIT CONF_THRESH DEPTH_CONF_THRESH METHOD_RESOLUTION_SOURCE
# Point-query VLMs: query points per image axis (MethodSpec.point_grid_size), unless the caller set its own. It keys the
# prediction cache (prediction_group_key), so a new grid never reuses another grid's predictions.
if [ "${METHOD_POINT_GRID_SIZE_DEFAULT:-0}" != 0 ]; then
    export POINT_QUERY_DEPTH_GRID_SIZE=${POINT_QUERY_DEPTH_GRID_SIZE:-$METHOD_POINT_GRID_SIZE_DEFAULT}
fi
MAX_POINTS=${MAX_POINTS:-10000}
if [ -z "${MAX_IMAGES_PER_RUN+x}" ]; then
    case "$BENCHMARK_SUITE" in
        full-pose)
            MAX_IMAGES_PER_RUN=100
            ;;
        *)
            # Matches DA3-BENCH's own eval_bench.yaml max_frames: 100,
            # applied uniformly to its pose AND reconstruction eval modes --
            # so full-depth (MVRecon) caps the same as full-pose (SfM) for
            # every method, IGGT included, rather than giving IGGT a
            # narrower comparison basis. IGGT has no internal batching (see
            # iggt.py -- one forward pass over every requested frame, unlike
            # every other adapter), so this raises its actual peak per-call
            # frame count from 12 to 100. Verified on an H200: job 131387
            # (7scenes/heads capped to 100 frames, --depth_only --export_colmap)
            # ran in 27.2 s of inference without CUDA OOM.
            MAX_IMAGES_PER_RUN=100
            ;;
    esac
fi
if [[ "$BENCHMARK_SUITE" == full-* ]]; then
    KEEP_PREDICTIONS=${KEEP_PREDICTIONS:-1}
else
    KEEP_PREDICTIONS=${KEEP_PREDICTIONS:-0}
fi
KEEP_DEPTH_VISUALIZATIONS=${KEEP_DEPTH_VISUALIZATIONS:-1}
# MonoDepth/VideoDepth: save per-frame error maps after alignment (depth_prediction_evaluator --error_artifact_dir).
DEPTH_ERROR_ARTIFACTS=${DEPTH_ERROR_ARTIFACTS:-1}
# Dashboard viewer assets written by the evaluators (spatialarena/eval/viewer_assets.py): per case into
# <case report dir>/viewer/, and the files every method of a scene shares (GT/RGB frames, GT points) into
# VIEWER_SCENE_ROOT. The dashboard collector indexes them; nothing else runs after evaluation.
VIEWER_ASSETS=${VIEWER_ASSETS:-1}
# SfM / SLAM: also export each frame's predicted depth (depths/<stem>.npz, float32, compressed) and the method's confidence
# (confidences/<stem>.npz, float16, compressed) next to the COLMAP model, for the viewer and later depth analysis of these tracks.
EXPORT_POSE_DEPTH=${EXPORT_POSE_DEPTH:-1}
VIEWER_SCENE_ROOT=${VIEWER_SCENE_ROOT:-$BENCHMARK_EVAL_ROOT/viewer_assets/scenes}
GENERATE_PDF_REPORT=${GENERATE_PDF_REPORT:-1}
ALLOW_BENCHMARK_ROOT_OUTPUT=${ALLOW_BENCHMARK_ROOT_OUTPUT:-0}
ALLOW_BENCHMARK_EVAL_OUTPUT=${ALLOW_BENCHMARK_EVAL_OUTPUT:-1}

if [[ "$BENCHMARK_SUITE" == full-* ]] && [ "$KEEP_PREDICTIONS" != "1" ] && [ "${ALLOW_FULL_BENCHMARK_PREDICTION_DELETION:-0}" != "1" ]; then
    echo "Refusing to run $BENCHMARK_SUITE with KEEP_PREDICTIONS=$KEEP_PREDICTIONS." >&2
    echo "Full benchmark runs must preserve prediction artifacts; set KEEP_PREDICTIONS=1 or explicitly set ALLOW_FULL_BENCHMARK_PREDICTION_DELETION=1." >&2
    exit 7
fi

cd "$PROJECT_ROOT"

resolve_path() {
    "$PYTHON" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))' "$1"
}

dataset_root_for_case() {
    local task=$1
    local dataset=$2
    if [ "$SLAM_FIXED_INPUTS" = 1 ] && [ "$task" = "SLAM" ]; then
        # The pre-sized sequences (SLAM/<dataset>): method inputs and the evaluator's ground truth alike.
        local input_root
        input_root=$("$PYTHON" -m spatialarena.slam_protocol input_root "$BENCHMARK_ROOT" "$dataset") || return 1
        if [ ! -f "$input_root/metadata.json" ]; then
            echo "Missing SLAM inputs $input_root: run scripts/preprocess/materialize_slam_inputs.py --datasets $dataset" >&2
            return 1
        fi
        echo "$input_root"
        return 0
    fi
    "$PYTHON" - "$BENCHMARK_ROOT" "$task" "$dataset" <<'PY'
from pathlib import Path
import sys

from spatialarena.eval.dataset_registry import resolve_dataset_root

root = resolve_dataset_root(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
print(root)
PY
}

# A dataset's own frame cap per case (dataset_registry.DatasetSpec.max_input_frames: 0 = every frame), or nothing
# for the suite's.
dataset_frame_cap() {
    PYTHONPATH="$PROJECT_ROOT" "$PYTHON" - "$1" "$2" <<'PY'
import sys

from spatialarena.eval.dataset_registry import dataset_max_input_frames

cap = dataset_max_input_frames(sys.argv[1], sys.argv[2])
print("" if cap is None else cap)
PY
}

assert_outside_benchmark_root() {
    local label=$1
    local candidate=$2
    local resolved_candidate
    local resolved_benchmark
    local resolved_eval_root
    resolved_candidate=$(resolve_path "$candidate")
    resolved_benchmark=$(resolve_path "$BENCHMARK_ROOT")
    resolved_eval_root=$(resolve_path "$BENCHMARK_EVAL_ROOT")
    if [ "$ALLOW_BENCHMARK_EVAL_OUTPUT" = "1" ]; then
        if [ "$resolved_candidate" = "$resolved_eval_root" ] || [[ "$resolved_candidate" == "$resolved_eval_root"/* ]]; then
            return 0
        fi
    fi
    if [ "$resolved_candidate" = "$resolved_benchmark" ] || [[ "$resolved_candidate" == "$resolved_benchmark"/* ]]; then
        if [ "$ALLOW_BENCHMARK_ROOT_OUTPUT" != "1" ]; then
            echo "$label must be outside BENCHMARK_ROOT to keep benchmark inputs read-only." >&2
            echo "$label: $resolved_candidate" >&2
            echo "BENCHMARK_ROOT: $resolved_benchmark" >&2
            echo "Allowed persistent evaluation root: $resolved_eval_root" >&2
            echo "Set ALLOW_BENCHMARK_ROOT_OUTPUT=1 only for intentional debugging." >&2
            exit 6
        fi
    fi
}

assert_outside_benchmark_root EXP_ROOT "$EXP_ROOT"
assert_outside_benchmark_root REPORT_DIR "$REPORT_DIR"

# Inside the shared reports tree, results go only to its fixed folders, whatever script started the run: one folder
# per full suite and method (reruns overwrite in place; spatialarena/report_layout.py), sampled smoke runs, and
# one-off tests under _<name>/. A new reports/<RUN_ID>/ tree would scatter the metrics of one benchmark.
assert_report_dir_in_layout() {
    local reports resolved
    reports=$(resolve_path "$BENCHMARK_EVAL_ROOT/reports")
    resolved=$(resolve_path "$REPORT_DIR")
    [[ "$resolved" == "$reports"/* ]] || return 0
    case "${resolved#"$reports"/}" in
        benchmark_full_reports/full-depth/*/* | benchmark_full_pose_reports/full-pose/*/* \
            | benchmark_full_slam_reports/full-slam-*/*/*) ;;
        benchmark_full_reports/full-depth/* | benchmark_full_pose_reports/full-pose/* \
            | benchmark_full_slam_reports/full-slam-*/* | benchmark_sampled_reports/* | _*/*) return 0 ;;
    esac
    echo "Refusing REPORT_DIR=$REPORT_DIR: inside $reports, results go only to" >&2
    echo "  benchmark_full_reports/full-depth/<METHOD_ID>, benchmark_full_pose_reports/full-pose/<METHOD_ID>," >&2
    echo "  benchmark_full_slam_reports/full-slam-<range>/<METHOD_ID> (reruns overwrite in place)," >&2
    echo "  benchmark_sampled_reports/..., or a one-off test folder _<name>/..." >&2
    exit 8
}
assert_report_dir_in_layout

env_value() {
    local name=${1:-}
    if [ -z "$name" ]; then
        echo ""
        return
    fi
    echo "${!name:-}"
}

hf_config_accessible() {
    local repo_id=$1
    "$PYTHON" - "$repo_id" <<'PY' >/dev/null 2>&1
import sys
from huggingface_hub import hf_hub_download

hf_hub_download(sys.argv[1], "config.json", local_files_only=False)
PY
}

cached_hf_snapshot() {
    local repo_id=$1
    local hf_home=${HF_HOME:-}
    if [ -z "$hf_home" ] && [ -n "${CHECKPOINT_ROOT:-}" ]; then
        hf_home=$CHECKPOINT_ROOT/huggingface
    fi
    if [ -z "$hf_home" ]; then
        return 1
    fi

    local cache_name=${repo_id//\//--}
    local snapshots_dir=$hf_home/hub/models--$cache_name/snapshots
    if [ ! -d "$snapshots_dir" ]; then
        return 1
    fi
    find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -nr \
        | awk 'NR == 1 { print $2 }'
}

if [ "$METHOD" = "SENSENOVA_VISION" ] && [[ "$SENSENOVA_VISION_CKPT_PATH" != /* ]] && [ ! -e "$SENSENOVA_VISION_CKPT_PATH" ]; then
    SENSENOVA_CACHED_SNAPSHOT=$(cached_hf_snapshot "$SENSENOVA_VISION_CKPT_PATH" || true)
    if [ -n "$SENSENOVA_CACHED_SNAPSHOT" ]; then
        SENSENOVA_VISION_CKPT_PATH=$SENSENOVA_CACHED_SNAPSHOT
        export SENSENOVA_VISION_CKPT_PATH
        echo "[SENSENOVA_VISION] using cached checkpoint snapshot: $SENSENOVA_VISION_CKPT_PATH"
    fi
fi

mkdir -p "$REPORT_DIR"
# Name summary/case_failures TSVs after CASE_FILE (unless it's the suite's
# own default full case list) rather than always "summary.tsv": splitting
# one suite's cases across several smaller per-task jobs (e.g. MVRecon vs
# MonoDepth vs VideoDepth within full-depth) means several separate
# run_method_case_list.sh invocations now target the SAME REPORT_DIR, and
# each invocation truncates its summary file on start -- sharing one
# filename would make a later task's job silently wipe an earlier task's
# rows. A default (unsplit) run still gets the original "summary.tsv"/
# "case_failures.tsv" names, unchanged.
SUMMARY_SUFFIX=""
case "$BENCHMARK_SUITE" in
    full-depth) DEFAULT_CASE_FILE_BASENAME=full_standard_depth_cases.txt ;;
    full-pose) DEFAULT_CASE_FILE_BASENAME=full_standard_sfm_cases.txt ;;
    full-slam) DEFAULT_CASE_FILE_BASENAME=full_standard_slam_cases_${SLAM_RANGE}.txt ;;
    *) DEFAULT_CASE_FILE_BASENAME="" ;;
esac
if [ -n "$DEFAULT_CASE_FILE_BASENAME" ] && [ "$(basename "$CASE_FILE")" != "$DEFAULT_CASE_FILE_BASENAME" ]; then
    SUMMARY_SUFFIX="_$(basename "$CASE_FILE" .txt)"
fi
SUMMARY_TSV=$REPORT_DIR/summary${SUMMARY_SUFFIX}.tsv
printf "method\ttask\tdataset\tscene\tinterval\torder\toutput_path\treport_dir\tprediction_retention\n" > "$SUMMARY_TSV"
CASE_FAILURES_TSV=$REPORT_DIR/case_failures${SUMMARY_SUFFIX}.tsv
# reason: gpu_oom / host_oom / error when the inference log was classified (classify_case_failure), else empty.
printf "method\ttask\tdataset\tscene\tinterval\torder\tstatus\treason\n" > "$CASE_FAILURES_TSV"

UNSUPPORTED_STATUS=""
UNSUPPORTED_REASON=""
case "$METHOD" in
    COME)
        if [ ! -d "$COME_CODE_PATH" ]; then
            UNSUPPORTED_STATUS="code_path_missing"
            UNSUPPORTED_REASON="CoME source checkout is missing at COME_CODE_PATH=$COME_CODE_PATH."
        elif ! find "$COME_CODE_PATH/src/cuda_extension/co_me" -maxdepth 1 -name 'co_me_cuext*.so' -print -quit | grep -q . \
            || ! find "$COME_CODE_PATH/src/cuda_extension/flash_attn" -maxdepth 1 -name 'flash_attn_fwd_cuda*.so' -print -quit | grep -q .; then
            UNSUPPORTED_STATUS="dependency_blocked"
            UNSUPPORTED_REASON="CoME CUDA extensions are not built under COME_CODE_PATH=$COME_CODE_PATH; run scripts/slurm/build_come_extensions.sh before evaluation."
        fi
        ;;
    CUT3R)
        if [ ! -f "$CUT3R_CODE_PATH/add_ckpt_path.py" ] && [ ! -f "$CUT3R_CODE_PATH/src/add_ckpt_path.py" ]; then
            UNSUPPORTED_STATUS="code_path_missing"
            UNSUPPORTED_REASON="CUT3R source checkout is missing or incomplete at CUT3R_CODE_PATH=$CUT3R_CODE_PATH."
        else
            cut3r_ckpt_candidate=$CUT3R_CKPT_PATH
            if [[ "$cut3r_ckpt_candidate" != /* ]]; then
                cut3r_ckpt_candidate=$CUT3R_CODE_PATH/$cut3r_ckpt_candidate
            fi
            if [ ! -f "$cut3r_ckpt_candidate" ]; then
                UNSUPPORTED_STATUS="checkpoint_missing"
                UNSUPPORTED_REASON="CUT3R checkpoint is missing at CUT3R_CKPT_PATH=$CUT3R_CKPT_PATH (resolved to $cut3r_ckpt_candidate)."
            fi
        fi
        ;;
    FAST3R)
        if [ ! -d "$FAST3R_CODE_PATH/fast3r" ]; then
            UNSUPPORTED_STATUS="code_path_missing"
            UNSUPPORTED_REASON="Fast3R Python package is missing at FAST3R_CODE_PATH=$FAST3R_CODE_PATH."
        fi
        ;;
    VLM3)
        if [[ "$VLM3_CKPT_PATH" != /* ]] && [ ! -e "$VLM3_CKPT_PATH" ] && ! hf_config_accessible "$VLM3_CKPT_PATH"; then
            UNSUPPORTED_STATUS="upstream_unavailable"
            UNSUPPORTED_REASON="VLM3 public code is available, but checkpoint $VLM3_CKPT_PATH is not accessible from Hugging Face; set VLM3_CKPT_PATH to an accessible local path or model id to run it."
        fi
        ;;
    DEPTHLM)
        if [[ "$DEPTHLM_CKPT_PATH" != /* ]] && [ ! -e "$DEPTHLM_CKPT_PATH" ] && ! hf_config_accessible "$DEPTHLM_CKPT_PATH"; then
            UNSUPPORTED_STATUS="dependency_blocked"
            UNSUPPORTED_REASON="DepthLM public code is available, but checkpoint $DEPTHLM_CKPT_PATH is gated or otherwise inaccessible from Hugging Face in this environment; set DEPTHLM_CKPT_PATH to an accessible local path or model id to run it."
        fi
        ;;
    CLAUDE_OPUS_5_5|GPT_6_ASTRA|GEMINI_4_ARGON)
        openrouter_key_file=${OPENROUTER_API_KEY_FILE:-}
        openrouter_key_file=${openrouter_key_file/#\~/$HOME}
        if [ -z "${OPENROUTER_API_KEY:-}" ] && { [ -z "$openrouter_key_file" ] || [ ! -r "$openrouter_key_file" ]; }; then
            UNSUPPORTED_STATUS="credentials_missing"
            UNSUPPORTED_REASON="$METHOD is queried through OpenRouter; set OPENROUTER_API_KEY or OPENROUTER_API_KEY_FILE (a readable key file) in the job environment."
        fi
        ;;
    SENSENOVA_VISION)
        if ! "$PYTHON" -c "import flash_attn" >/dev/null 2>&1; then
            UNSUPPORTED_STATUS="dependency_blocked"
            UNSUPPORTED_REASON="SenseNova-Vision imports flash_attn unconditionally, but flash_attn is not installed in the configured Python environment."
        fi
        ;;
esac

if [ -n "$UNSUPPORTED_STATUS" ]; then
    "$PYTHON" - "$REPORT_DIR/summary${SUMMARY_SUFFIX}.json" "$METHOD_ID" "$METHOD" "$BENCHMARK_TRACK" "$EXP_ROOT" "$EXP_NAME" "$CASE_FILE" "$REPORT_DIR" "$SUMMARY_TSV" "$CASE_FAILURES_TSV" "$UNSUPPORTED_STATUS" "$UNSUPPORTED_REASON" "$KEEP_PREDICTIONS" "$KEEP_DEPTH_VISUALIZATIONS" "$GENERATE_PDF_REPORT" "$ALLOW_BENCHMARK_ROOT_OUTPUT" "$ALLOW_BENCHMARK_EVAL_OUTPUT" "$BENCHMARK_AUX_ROOT" "$BENCHMARK_EVAL_ROOT" "$PREDICTION_CACHE_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    summary_path,
    method_id,
    method,
    track,
    exp_root,
    exp_name,
    case_file,
    report_dir,
    summary_tsv,
    case_failures_tsv,
    status,
    reason,
    keep_predictions,
    keep_depth_visualizations,
    generate_pdf_report,
    allow_benchmark_root_output,
    allow_benchmark_eval_output,
    benchmark_aux_root,
    benchmark_eval_root,
    prediction_cache_root,
) = sys.argv[1:]
payload = {
    "method": method_id,
    "base_method": method,
    "benchmark_track": track,
    "experiment_root": exp_root,
    "experiment_name": exp_name,
    "case_file": case_file,
    "report_dir": report_dir,
    "summary_tsv": summary_tsv,
    "case_failures_tsv": case_failures_tsv,
    "unsupported": True,
    "unsupported_status": status,
    "unsupported_reason": reason,
    "keep_predictions": keep_predictions,
    "keep_depth_visualizations": keep_depth_visualizations,
    "generate_pdf_report": generate_pdf_report,
    "allow_benchmark_root_output": allow_benchmark_root_output,
    "allow_benchmark_eval_output": allow_benchmark_eval_output,
    "benchmark_aux_root": benchmark_aux_root,
    "benchmark_eval_root": benchmark_eval_root,
    "prediction_cache_root": prediction_cache_root,
    "resolution_profile": {
        "resolution": int(os.environ["RESOLUTION"]),
        "image_load_resolution": int(os.environ["IMAGE_LOAD_RESOLUTION"]),
        "pixel_limit": int(os.environ["PIXEL_LIMIT"]),
        "conf_thresh": float(os.environ["CONF_THRESH"]),
        "depth_conf_thresh": float(os.environ["DEPTH_CONF_THRESH"]),
        "resolution_source": os.environ.get("METHOD_RESOLUTION_SOURCE", ""),
        "dense_depth_confidence_filtering": False,
    },
    # Point-query VLMs only: query points per image axis (dense maps interpolate point_query_grid_size**2 answers).
    **({"point_query_grid_size": int(os.environ["POINT_QUERY_DEPTH_GRID_SIZE"])}
       if os.environ.get("POINT_QUERY_DEPTH_GRID_SIZE") else {}),
}
Path(summary_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
# Every run of the method shares this folder: summary<suffix>.json is this run's; summary.json (how the dashboard
# finds report folders) is replaced atomically so concurrent runs never leave it half-written.
shared_summary = Path(summary_path).with_name("summary.json")
if Path(summary_path) != shared_summary:
    partial = shared_summary.with_name(f".summary.json.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, shared_summary)
(Path(report_dir) / "unsupported_status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
    if [ "$GENERATE_PDF_REPORT" = "1" ]; then
        "$PYTHON" scripts/test/generate_benchmark_pdf_report.py \
            "$REPORT_DIR" \
            --output "$REPORT_DIR/.report_$$.pdf" \
            --title "FeedForward Benchmark: $METHOD_ID ($BENCHMARK_TRACK)" \
            && mv -f "$REPORT_DIR/.report_$$.pdf" "$REPORT_DIR/report.pdf"  # atomic: runs share the folder
    fi
    echo "Benchmark skipped for $METHOD_ID: $UNSUPPORTED_STATUS - $UNSUPPORTED_REASON"
    echo "Reports: $REPORT_DIR"
    exit 0
fi

safe_name() {
    local value=$1
    value=${value//\//__}
    value=${value// /_}
    echo "$value"
}

prediction_group_key() {
    local method=$1
    local track=$2
    local resolution=$3
    local image_load_resolution=$4
    local pixel_limit=$5
    local max_images_per_run=$6
    local max_points=$7
    local conf_thresh=$8
    local depth_conf_thresh=$9
    local max_input_frames=${10}
    local interval=${11}
    local order_name=${12}
    local data_path=${13}
    local code_path=${14:-}
    local ckpt_path=${15:-}
    local single_view=${16:-0}
    "$PYTHON" - \
        "$method" \
        "$track" \
        "$resolution" \
        "$image_load_resolution" \
        "$pixel_limit" \
        "$max_images_per_run" \
        "$max_points" \
        "$conf_thresh" \
        "$depth_conf_thresh" \
        "$max_input_frames" \
    "$interval" \
    "$order_name" \
    "$data_path" \
    "$code_path" \
    "$ckpt_path" \
    "$single_view" <<'PY'
import hashlib
import json
import os
import sys

payload = {
    "method": sys.argv[1],
    "track": sys.argv[2],
    "resolution": int(sys.argv[3]),
    "image_load_resolution": int(sys.argv[4]),
    "pixel_limit": int(sys.argv[5]),
    "max_images_per_run": int(sys.argv[6]),
    "max_points": int(sys.argv[7]),
    "conf_thresh": float(sys.argv[8]),
    "depth_conf_thresh": float(sys.argv[9]),
    "max_input_frames": int(sys.argv[10]),
    "interval": int(sys.argv[11]),
    "order_name": sys.argv[12],
    "data_path": sys.argv[13],
    "code_path": sys.argv[14],
    "ckpt_path": sys.argv[15],
}
# Only present when set, so the keys (and cached predictions) of every non-single-view case stay unchanged.
if sys.argv[16] == "1":
    payload["single_view"] = True
# Masked-depth tracks: the mask setting (BENCHMARK_DEPTH_MASK_RATIO/_GRID) changes the input, so it keys the cache.
if sys.argv[2] in {"rgb_masked_depth", "rgb_masked_depth_cameras"}:
    sys.path.insert(0, os.getcwd())
    from spatialarena.depth_prior_masking import protocol

    payload["depth_mask_protocol"] = protocol()
if sys.argv[2] in {"rgb_sparse_depth", "rgb_sparse_depth_cameras"}:
    sys.path.insert(0, os.getcwd())
    from spatialarena.sparse_depth import PROTOCOL as SPARSE_PROTOCOL

    payload["sparse_depth_protocol"] = SPARSE_PROTOCOL
# OpenRouter API VLMs (scripts/inference/openrouter_vlm_depth.py): the query/fill settings change the predictions,
# so they key the cache; "revision" changes when the adapter's output changes (2: neighbour gap fill, 3: completion
# follow-ups for answers that stop partway).
# Point-query VLMs: the query grid changes the predictions. Keyed only above the adapter's old centre-only default (1),
# so those cached predictions keep their keys.
if sys.argv[1] in {"VLM3", "DEPTHLM", "QWEN3_VL"} and os.environ.get("POINT_QUERY_DEPTH_GRID_SIZE", "1") not in {"", "1"}:
    payload["point_query_grid_size"] = int(os.environ["POINT_QUERY_DEPTH_GRID_SIZE"])
if sys.argv[1] in {"CLAUDE_OPUS_5_5", "GPT_6_ASTRA", "GEMINI_4_ARGON"}:
    payload["openrouter"] = {
        "revision": 3,
        **{name: os.environ.get(name, "") for name in (
            "OPENROUTER_DEPTH_GRID_SIZE", "OPENROUTER_PROMPT_VERSION", "OPENROUTER_SYSTEM_PROMPT",
            "OPENROUTER_DEPTH_INTERPOLATION", "OPENROUTER_REFINE_POINTS", "OPENROUTER_REFINE_RATIO",
            "OPENROUTER_REASONING_EFFORT", "OPENROUTER_DEPTH_CLAMP", "OPENROUTER_ALLOW_DATA_COLLECTION",
            "OPENROUTER_COMPLETION_FOLLOWUPS",
        )},
    }
digest = hashlib.sha1(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
print(digest[:20])
PY
}

# Whether a depth-only (MonoDepth/VideoDepth/MVRecon) case also exports the predicted cameras as a (lightweight,
# points-free) COLMAP model. MVRecon evaluation fuses each frame's depth with its camera (see
# spatialarena/eval/mvrecon_evaluator.py::prepare_fusion_inputs); VideoDepth keeps them for the dashboard's demo
# replay (trajectory, frustums, accumulated points). Only methods whose adapter accumulates poses in its depth-only
# loop instead of discarding them are listed (DEPTHVLM, SENSENOVA_VISION, POINT_QUERY_DEPTH, VLM3/DEPTHLM/QWEN3_VL
# and EXTERNAL_COMMAND methods either predict no camera at all or have not been extended the same way -- add them only
# after fixing and verifying their adapter). AMB3R (amb3r.py) and ROBUSTVGGT (robustvggt.py -> vggt.py) export them
# in their depth-only loops; leaving them out made every MVRecon case of theirs unscorable (no sparse/ to fuse with).
# The MVRecon datasets whose GT depth is sparse or incomplete: the only ones MVRecon rgb_depth runs on
# (TRACK_DATASETS in spatialarena/task_track_matrix.py; test_task_track_matrix.py keeps the two in step).
MVRECON_RGB_DEPTH_DATASETS="7scenes dtu eth3d"
# The SfM datasets whose released GT has depth maps: the only ones SfM rgb_depth runs on (TRACK_DATASETS as above; DTU,
# Tanks and Temples and Weak-Texture SfM have none).
SFM_RGB_DEPTH_DATASETS="eth3d hiroom imc scannetpp spatialarena"
# The SLAM datasets whose inputs have depth maps, and the sequences of those without any (no LiDAR depth): SLAM
# rgb_depth runs only on the first and never on the second (TRACK_DATASETS / TRACK_EXCLUDED_SCENES as above).
SLAM_RGB_DEPTH_DATASETS="7scenes spatialarena tum_dynamics"
SLAM_RGB_DEPTH_EXCLUDED_SCENES="CambridgeWesthub3 CoalDropsYard2 EdinburghCastle1 NationalHistoryMuseumIndoor1 StPancrasOutside"
# The only VideoDepth dataset whose released GT has camera poses: the only one VideoDepth rgb_cameras runs on
# (TRACK_DATASETS as above; KITTI has only the OXTS vehicle trajectory, the released GT no Sintel or DyCheck cameras).
VIDEODEPTH_RGB_CAMERAS_DATASETS="spatialarena"

depth_only_exports_poses() {
    case "$(canonical_task_name "$1")" in
        MVRecon|VideoDepth) ;;
        *) return 1 ;;
    esac
    case "$METHOD" in
        PI3_X|MAPANYTHING|OMNIVGGT|DA3-L|DA3-G|DA3-NGL|COME|CUT3R|FAST3R|FASTVGGT|VGGT|VGGT_OMEGA|DEJAVIEW|IGGT|IGGT4D|POW3R|DUST3R|MAST3R|AMB3R|ROBUSTVGGT)
            return 0
            ;;
    esac
    return 1
}

prediction_output_ready() {
    local task=$1
    local dataset=$2
    local scene=$3
    local output_path=$4
    local exports_poses=0
    if depth_only_exports_poses "$task"; then
        exports_poses=1
    fi
    local data_path
    data_path=$(dataset_root_for_case "$(canonical_task_name "$task")" "$dataset")/$scene
    "$PYTHON" - "$task" "$dataset" "$scene" "$output_path" "$depth_only" "$EXPORT_POSE_DEPTH" "$CASE_PHASE" \
        "$exports_poses" "$data_path" "${interval:-1}" <<'PY'
from pathlib import Path
import sys

from spatialarena.eval.prediction_artifacts import validate_prediction_output

requirements = None
if sys.argv[1] == "MVRecon" and sys.argv[5] in {"1", "true"}:
    # MVRecon fusion needs each frame's camera, and verify_case requires the sparse model: a depth-only prediction
    # (e.g. VGGT_OMEGA's Sep 5 cache entries) is not reused, so its case re-runs inference with --export_colmap.
    from spatialarena.eval.prediction_artifacts import PredictionRequirements
    requirements = PredictionRequirements(require_sparse=True, require_depths=True)
elif sys.argv[1] == "VideoDepth" and sys.argv[8] == "1" and sys.argv[7] != "evaluate":
    # VideoDepth exports the predicted cameras (depth_only_exports_poses): a depth-only prediction from before is
    # not reused, its case re-runs inference. An evaluate-only phase still scores it (scoring needs only depths).
    from spatialarena.eval.prediction_artifacts import PredictionRequirements
    requirements = PredictionRequirements(require_sparse=True, require_depths=True)
elif sys.argv[6] == "1" and sys.argv[1] in {"SLAM", "SfM"} and sys.argv[7] != "evaluate":
    # EXPORT_POSE_DEPTH: a pose-only SLAM / SfM prediction from before depth export is not reused; its case re-runs
    # inference. An evaluate-only phase still scores such a prediction: both evaluators read only the poses, and it
    # only lacks the extra depth maps (re-scoring the existing results after an evaluator change needs this).
    from spatialarena.eval.prediction_artifacts import PredictionRequirements
    requirements = PredictionRequirements(require_sparse=True, require_depths=True)
status = validate_prediction_output(
    task=sys.argv[1],
    dataset=sys.argv[2],
    method="",
    scene=sys.argv[3],
    scene_output_path=Path(sys.argv[4]),
    requirements=requirements,
)
if status.ok and sys.argv[1] == "MonoDepth" and Path(sys.argv[9]).is_dir():
    # MonoDepth scores every frame (no frame cap in single-view mode): a prediction of fewer frames, e.g. one made
    # under the former 100-frame cap, is not reused; its case re-runs inference.
    from scripts.inference.adapter_utils import canonical_image_root, find_image_paths

    frames = find_image_paths(canonical_image_root(Path(sys.argv[9])))[::max(1, int(sys.argv[10]))]
    expected = {Path(path).stem for path in frames}
    predicted = {path.stem for path in status.depth_files}
    if not expected <= predicted:
        print(f"{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}: prediction covers {len(expected & predicted)} of "
              f"{len(expected)} frames; not reused", file=sys.stderr)
        raise SystemExit(1)
raise SystemExit(0 if status.ok else 1)
PY
}

ensure_task_output_alias() {
    local task_output_path=$1
    local shared_output_path=$2
    if [ "$task_output_path" = "$shared_output_path" ]; then
        return 0
    fi
    mkdir -p "$(dirname "$task_output_path")"
    rm -rf "$task_output_path"
    ln -s "$shared_output_path" "$task_output_path"
}

artifact_dir_has_files() {
    local path=$1
    [ -d "$path" ] || return 1
    [ -n "$(find "$path" -type f -print -quit 2>/dev/null)" ]
}

has_colmap_model() {
    local path=$1
    [ -d "$path" ] || return 1
    { [ -f "$path/cameras.bin" ] || [ -f "$path/cameras.txt" ]; } || return 1
    { [ -f "$path/images.bin" ] || [ -f "$path/images.txt" ]; } || return 1
    { [ -f "$path/points3D.bin" ] || [ -f "$path/points3D.txt" ]; } || return 1
    return 0
}

camera_prior_path() {
    local data_path=$1
    local candidate
    for candidate in \
        "$data_path/sparse/0" \
        "$data_path/sparse" \
        "$data_path/dslr_calibration_undistorted" \
        "$data_path/dslr_calibration_jpg"; do
        if has_colmap_model "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

materialized_camera_prior_path() {
    local data_path=$1
    local existing
    if existing=$(camera_prior_path "$data_path"); then
        echo "$existing"
        return 0
    fi
    local rel_path=${data_path#"$BENCHMARK_ROOT"/}
    if [ "$rel_path" = "$data_path" ]; then
        rel_path=$(safe_name "$data_path")
    fi
    local output_path=$CAMERA_PRIOR_ROOT/$rel_path/sparse/0
    local log_path=$CAMERA_PRIOR_ROOT/$rel_path/materialize_camera_priors.log
    mkdir -p "$(dirname "$log_path")"
    if "$PYTHON" scripts/preprocess/materialize_benchmark_camera_priors.py \
        --data_path "$data_path" \
        --output_path "$output_path" \
        >"$log_path" 2>&1; then
        echo "$output_path"
        return 0
    fi
    cat "$log_path" >&2
    return 1
}

validate_camera_prior_mapping() {
    local data_path=$1
    local colmap_path
    if ! colmap_path=$(materialized_camera_prior_path "$data_path"); then
        return 1
    fi
    "$PYTHON" -c '
import sys
from pathlib import Path

sys.path.insert(0, "scripts/inference")
from adapter_utils import canonical_image_root, find_image_paths
from spatialarena.eval.colmap_view_resolver import resolve_colmap_image_mappings

data_path = Path(sys.argv[1])
colmap_path = Path(sys.argv[2])
image_paths = find_image_paths(canonical_image_root(data_path))
resolved, diagnostics = resolve_colmap_image_mappings(
    data_path,
    image_paths,
    colmap_path=colmap_path,
    strict=False,
)
print(
    f"camera-prior mapping: mapped={len(resolved)} "
    f"missing={len(diagnostics.missing)} ambiguous={len(diagnostics.ambiguous)}"
)
if diagnostics.missing:
    print("camera-prior missing records:", ", ".join(diagnostics.missing[:5]), file=sys.stderr)
if diagnostics.ambiguous:
    print("camera-prior ambiguous records:", diagnostics.ambiguous[:3], file=sys.stderr)
if not resolved:
    raise SystemExit(1)
if diagnostics.missing or diagnostics.ambiguous:
    raise SystemExit(1)
' "$data_path" "$colmap_path"
}

# Input tracks that hand the method GT depth priors (rgb_masked_depth*: with half of every map dropped, see
# spatialarena/depth_prior_masking.py) or GT camera priors.
track_uses_depth() {
    case "${BENCHMARK_TRACK:-rgb}" in
        rgb_depth|rgb_masked_depth|rgb_masked_depth_cameras|rgb_sparse_depth|rgb_sparse_depth_cameras) return 0 ;;
    esac
    return 1
}

# The sparse-depth tracks read their own priors (<benchmark_root>/_sparse_depth, spatialarena/sparse_depth.py), not
# the scene's dense GT depth.
track_uses_sparse_depth() {
    case "${BENCHMARK_TRACK:-rgb}" in
        rgb_sparse_depth|rgb_sparse_depth_cameras) return 0 ;;
    esac
    return 1
}

# Whether at least one image of the case has a sparse depth prior.
has_sparse_depth_priors() {
    local data_path=$1
    "$PYTHON" - "$data_path" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts/inference")
from adapter_utils import canonical_image_root, find_image_paths
from spatialarena.sparse_depth import prior_path

data_path = Path(sys.argv[1])
images = find_image_paths(canonical_image_root(data_path))
raise SystemExit(0 if any(prior_path(data_path, image).is_file() for image in images) else 1)
PY
}

track_uses_cameras() {
    case "${BENCHMARK_TRACK:-rgb}" in
        rgb_cameras|rgb_masked_depth_cameras|rgb_sparse_depth_cameras) return 0 ;;
    esac
    return 1
}

supports_partial_depth_priors() {
    case "$METHOD" in
        MAPANYTHING|OMNIVGGT|PI3_X)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

canonical_task_name() {
    case "$1" in
        CameraPose)
            echo "SfM"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

# MonoDepth (single-view) scenes run as one adapter call per image size and at most MAX_IMAGES_PER_RUN frames
# (scripts/inference/single_view_groups.py), so no frame is resized to fit, or stacked with, a frame of another size,
# and every frame is predicted without loading the whole scene at once; the per-frame outputs are then merged into one
# prediction. Uniform scenes within the frame budget and other tasks run unchanged.
run_inference() {
    local output_path=$6
    local data_path
    data_path=$(dataset_root_for_case "$(canonical_task_name "$1")" "$2")/$3
    if [ "${BENCHMARK_SINGLE_VIEW:-0}" != "1" ] || [ ! -d "$data_path" ]; then
        run_inference_once "$@"
        return
    fi
    local groups_dir=$output_path/_size_groups
    rm -rf "$groups_dir"
    local num_groups
    num_groups=$("$PYTHON" scripts/inference/single_view_groups.py plan --data_path "$data_path" --interval "$4" \
        --max_images_per_run "$MAX_IMAGES_PER_RUN" --groups_dir "$groups_dir")
    if [ "$num_groups" -le 1 ]; then
        run_inference_once "$@"
        return
    fi
    echo "[$METHOD_ID] $1/$2/$3: $num_groups single-view runs (one per image size and MAX_IMAGES_PER_RUN frames)"
    local group_list group_status
    for group_list in "$groups_dir"/group_*.txt; do
        (
            set -e
            export BENCHMARK_IMAGE_SUBSET=$group_list
            run_inference_once "$1" "$2" "$3" "$4" "$5" "${group_list%.txt}"
        )
        group_status=$?  # reached only under set +e (the DUSt3R-family caller); set -e has already exited otherwise
        [ "$group_status" -eq 0 ] || return "$group_status"
    done
    "$PYTHON" scripts/inference/single_view_groups.py merge --groups_dir "$groups_dir" --output_path "$output_path"
}

run_inference_once() {
    local task=$1
    local dataset=$2
    local scene=$3
    local interval=$4
    local order_name=$5
    local output_path=$6
    local canonical_task
    canonical_task=$(canonical_task_name "$task")
    local dataset_root
    dataset_root=$(dataset_root_for_case "$canonical_task" "$dataset")
    local data_path=$dataset_root/$scene
    if [ ! -d "$data_path" ]; then
        echo "Dataset scene not found: $data_path" >&2
        exit 3
    fi

    echo "[$METHOD_ID] infer $canonical_task/$dataset/$scene interval=$interval order=${order_name:-default} track=$BENCHMARK_TRACK base_method=$METHOD"
    mkdir -p "$output_path"
    BENCHMARK_IMAGE_ORDER=$order_name
    export BENCHMARK_IMAGE_ORDER

    local depth_only=false
    case "$canonical_task" in
        MonoDepth|VideoDepth|MVRecon)
            depth_only=true
            ;;
    esac

    local -a cmd
    local effective_native=$METHOD_NATIVE
    local method_code_path
    local method_ckpt_path
    method_code_path=$(env_value "$METHOD_CODE_ENV")
    method_ckpt_path=$(env_value "$METHOD_CKPT_ENV")
    # Every registered method runs through one entry point with one option set (spatialarena.models.BaseModel;
    # docs/models.md): the method name selects the model and its variant, its MethodSpec gives the defaults, and
    # --engine drops the options below that a model does not have (e.g. --resolution for a method that resizes
    # on its own). Model-specific settings come from the environment (GENCEPTION_*, COME_MODEL, ...; exported
    # above).
    cmd=(
        "$PYTHON" scripts/inference/run_model.py --engine
        --model "$METHOD"
        --data_path "$data_path"
        --output_path "$output_path"
        --track "$BENCHMARK_TRACK"
        --interval "$interval"
        --resolution "$RESOLUTION"
        --image_load_resolution "$IMAGE_LOAD_RESOLUTION"
        --max_points "$MAX_POINTS"
        --max_images_per_run "$MAX_IMAGES_PER_RUN"
        --conf_thresh "$CONF_THRESH"
        --depth_conf_thresh "$DEPTH_CONF_THRESH"
        --pixel_limit "$PIXEL_LIMIT"
    )
    # Unset paths keep the model's own defaults.
    if [ -n "$method_code_path" ]; then
        cmd+=(--code_path "$method_code_path")
    fi
    if [ -n "$method_ckpt_path" ]; then
        cmd+=(--ckpt_path "$method_ckpt_path")
    fi
    if track_uses_cameras; then
        local camera_prior_path
        if ! camera_prior_path=$(materialized_camera_prior_path "$data_path"); then
            echo "No camera priors found for $METHOD $BENCHMARK_TRACK case: $data_path" >&2
            exit 3
        fi
        cmd+=(--colmap_path "$camera_prior_path")
    fi
    case "$METHOD" in
        VGGT_LONG|PI_LONG|VGGT_SLAM)
            # Long-sequence SLAM: GPU phase and CPU alignment as separate jobs (docs/benchmark/slam_protocol.md).
            if [ "${SLAM_ALIGN_ONLY:-0}" = 1 ]; then
                cmd+=(--phase align)
            elif [ "$CASE_PHASE" = "inference" ] && [ "$SLAM_DEFER_ALIGN" = 1 ]; then
                cmd+=(--phase gpu)
            fi
            ;;
    esac

    if [ -n "${METHOD_COMMAND:-}" ]; then
        cmd=(
            "$PYTHON" "scripts/inference/external_command_adapter.py"
            --data_path "$data_path"
            --output_path "$output_path"
            --method_name "$METHOD"
            --track "$BENCHMARK_TRACK"
            --code_path "$method_code_path"
            --ckpt_path "$method_ckpt_path"
            --command "$METHOD_COMMAND"
            --interval "$interval"
            --resolution "$RESOLUTION"
            --image_load_resolution "$IMAGE_LOAD_RESOLUTION"
            --max_points "$MAX_POINTS"
            --max_images_per_run "$MAX_IMAGES_PER_RUN"
            --conf_thresh "$CONF_THRESH"
            --depth_conf_thresh "$DEPTH_CONF_THRESH"
        )
    elif [ "$effective_native" != "1" ]; then
        cmd=(
            "$PYTHON" "scripts/inference/external_command_adapter.py"
            --data_path "$data_path"
            --output_path "$output_path"
            --method_name "$METHOD"
            --track "$BENCHMARK_TRACK"
            --code_path "$method_code_path"
            --ckpt_path "$method_ckpt_path"
            --interval "$interval"
            --resolution "$RESOLUTION"
            --image_load_resolution "$IMAGE_LOAD_RESOLUTION"
            --max_points "$MAX_POINTS"
            --max_images_per_run "$MAX_IMAGES_PER_RUN"
            --conf_thresh "$CONF_THRESH"
            --depth_conf_thresh "$DEPTH_CONF_THRESH"
        )
    fi

    if [ "$depth_only" = true ]; then
        cmd+=(--max_images_per_run "$MAX_IMAGES_PER_RUN" --export_depth --depth_only)
        if depth_only_exports_poses "$task"; then
            cmd+=(--export_colmap)
        fi
    else
        cmd+=(--export_colmap)
        if [ "$task" = "MVRecon" ]; then
            cmd+=(--export_depth)
        elif [ "$EXPORT_POSE_DEPTH" = 1 ] && { [ "$task" = "SfM" ] || [ "$task" = "SLAM" ]; }; then
            cmd+=(--export_depth)
        fi
    fi

    # Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4): opt-in
    # per-method spatial_policy (vggt_omega.py/da3.py/vggt.py/fastvggt.py's
    # --spatial_policy), set via the method's config `env: SPATIAL_POLICY:
    # ...`. Every adapter's "shared" policy supports depth-only export,
    # restoring depth (and, when --export_colmap is also set, camera
    # intrinsics via adapter_utils.restore_camera_export) to each image's
    # true source resolution -- each adapter enforces depth-only itself and
    # errors out otherwise, so only forward the flag when this case is
    # depth-only (silently falling back to that adapter's own "legacy"
    # default otherwise, not failing the case).
    if [ -n "${SPATIAL_POLICY:-}" ] && [ "$depth_only" = true ]; then
        cmd+=(--spatial_policy "$SPATIAL_POLICY")
    fi

    MPLCONFIGDIR=$BENCHMARK_TMP_ROOT/matplotlib-cache "${cmd[@]}"
}

verify_case() {
    local task=$1
    local dataset=$2
    local scene=$3
    local case_report_dir=$4
    local order_name=${5:-}
    local require_depths=${6:-0}

    local -a requirements=()
    if [ "$require_depths" = "1" ]; then
        requirements+=(--require_depths)
    fi
    "$PYTHON" scripts/preprocess/verify_prediction_outputs.py \
        --benchmark_root "$BENCHMARK_ROOT" \
        --experiment_root "$EXP_ROOT" \
        --experiment_name "$EXP_NAME" \
        --methods "$METHOD_ID" \
        --tasks "$task" \
        --datasets "$dataset" \
        --scenes "$scene" \
        --order_name "$order_name" \
        "${requirements[@]}" \
        --strict \
        --output_path "$case_report_dir/verify.json"
}

evaluate_case() {
    local task=$1
    local dataset=$2
    local scene=$3
    local case_report_dir=$4
    local order_name=${5:-}

    # Per-frame error maps after alignment (error_maps/, error_visualizations/ next to eval_depth.json).
    local -a depth_error_args=()
    if [ "$DEPTH_ERROR_ARTIFACTS" = 1 ]; then
        depth_error_args=(--error_artifact_dir "$case_report_dir")
    fi
    local -a viewer_args=() viewer_scene_args=()
    if [ "$VIEWER_ASSETS" = 1 ]; then
        viewer_args=(--viewer_dir "$case_report_dir/viewer")
        viewer_scene_args=(--viewer_scene_root "$VIEWER_SCENE_ROOT")
    fi
    case "$task" in
        MVRecon)
            "$PYTHON" -m spatialarena.eval.mvrecon_evaluator \
                --spatialarena_backend "${SPATIALARENA_MVRECON_BACKEND:-tsdf}" \
                --metric_workers "${MVRECON_METRIC_WORKERS:-1}" \
                --metric_chunk_points "${MVRECON_METRIC_CHUNK_POINTS:-262144}" \
                --benchmark_root "$BENCHMARK_ROOT" \
                --experiment_root "$EXP_ROOT" \
                --experiment_name "$EXP_NAME" \
                --methods "$METHOD_ID" \
                --datasets "$dataset" \
                --scenes "$scene" \
                --output_path "$case_report_dir/eval_mvrecon.json" \
                "${viewer_args[@]}" "${viewer_scene_args[@]}" \
                --strict
            ;;
        MonoDepth)
            # Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4):
            # MONO_DEPTH_PROTOCOL/MONO_INVERSE_FLOOR_POLICY select the
            # depth_prediction_evaluator protocol; default to PR #21's
            # full_monodepth.yaml settings (GT-only inverse-depth Huber-IRLS,
            # gt_max inverse floor) for every method, as VIDEO_DEPTH_PROTOCOL
            # below does; override per method via its config `env:` block.
            "$PYTHON" -m spatialarena.eval.eval_mono_depth \
                --benchmark_root "$BENCHMARK_ROOT" \
                --experiment_root "$EXP_ROOT" \
                --experiment_name "$EXP_NAME" \
                --methods "$METHOD_ID" \
                --datasets "$dataset" \
                --scenes "$scene" \
                --protocol "${MONO_DEPTH_PROTOCOL:-${METHOD_DEPTH_PROTOCOL:-inverse_huber_irls_gt}}" \
                --inverse_floor_policy "${MONO_INVERSE_FLOOR_POLICY:-gt_max}" \
                --output_path "$case_report_dir/eval_depth.json" \
                "${depth_error_args[@]}" \
                "${viewer_args[@]}" "${viewer_scene_args[@]}" \
                --strict
            ;;
        VideoDepth)
            # Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4):
            # VIDEO_DEPTH_PROTOCOL selects the depth_prediction_evaluator
            # protocol; defaults to the ported GT-only inverse-depth
            # Huber-IRLS alignment rather than "standard".
            "$PYTHON" -m spatialarena.eval.eval_video_depth \
                --benchmark_root "$BENCHMARK_ROOT" \
                --experiment_root "$EXP_ROOT" \
                --experiment_name "$EXP_NAME" \
                --methods "$METHOD_ID" \
                --datasets "$dataset" \
                --scenes "$scene" \
                --protocol "${VIDEO_DEPTH_PROTOCOL:-${METHOD_DEPTH_PROTOCOL:-inverse_huber_irls_gt}}" \
                --output_path "$case_report_dir/eval_depth.json" \
                "${depth_error_args[@]}" \
                "${viewer_args[@]}" "${viewer_scene_args[@]}" \
                --strict
            ;;
        SLAM)
            if [ "$BENCHMARK_SUITE" = "full-slam" ]; then
                local slam_alignment slam_stride
                slam_alignment=$("$PYTHON" -m spatialarena.slam_protocol alignment "$BENCHMARK_TRACK")
                slam_stride=$(slam_case_stride "$dataset") || exit 4
                "$PYTHON" -m spatialarena.eval.slam_trajectory_evaluator \
                    --benchmark_root "$BENCHMARK_ROOT" \
                    --experiment_root "$EXP_ROOT" \
                    --experiment_name "$EXP_NAME" \
                    --methods "$METHOD_ID" \
                    --datasets "$dataset" \
                    --scenes "$scene" \
                    --stride "$slam_stride" \
                    --slam_range "$SLAM_RANGE" \
                    --alignment "$slam_alignment" \
                    --output_path "$case_report_dir/eval_slam.json" \
                    "${viewer_args[@]}" "${viewer_scene_args[@]}" \
                    --strict
            else
                echo "SLAM cases are evaluated only in the full-slam suite (docs/benchmark/slam_protocol.md)" >&2
                exit 4
            fi
            ;;
        SfM)
            "$PYTHON" -m spatialarena.eval.sparse_reconstruction_evaluator \
                --benchmark_root "$BENCHMARK_ROOT" \
                --experiment_root "$EXP_ROOT" \
                --experiment_name "$EXP_NAME" \
                --methods "$METHOD_ID" \
                --tasks "$task" \
                --datasets "$dataset" \
                --scenes "$scene" \
                --order_name "$order_name" \
                --output_path "$case_report_dir/eval_sparse.json" \
                "${viewer_args[@]}" "${viewer_scene_args[@]}" \
                --strict
            ;;
        *)
            echo "Unsupported sampled benchmark task: $task" >&2
            exit 4
            ;;
    esac
}

case_eval_file() {
    local task=$1
    local case_report_dir=$2
    case "$task" in
        MonoDepth|VideoDepth|MVRecon)
            if [ "$task" = "MVRecon" ]; then
                echo "$case_report_dir/eval_mvrecon.json"
            else
                echo "$case_report_dir/eval_depth.json"
            fi
            ;;
        SLAM)
            echo "$case_report_dir/eval_slam.json"
            ;;
        SfM)
            echo "$case_report_dir/eval_sparse.json"
            ;;
        *)
            echo ""
            ;;
    esac
}

has_matched_depth_priors() {
    local data_path=$1
    local interval=$2
    local order_name=${3:-}
    "$PYTHON" -c '
import sys
from pathlib import Path

sys.path.insert(0, "scripts/inference")
from adapter_utils import find_image_paths, resolve_depth_prior_paths, select_interval

data_path = Path(sys.argv[1])
interval = int(sys.argv[2])
order_name = sys.argv[3]
if order_name:
    import os
    os.environ["BENCHMARK_IMAGE_ORDER"] = order_name
image_paths = select_interval(find_image_paths(data_path / "images"), interval)
if not image_paths:
    raise SystemExit(1)
depth_paths, missing = resolve_depth_prior_paths(data_path, image_paths)
raise SystemExit(0 if len(depth_paths) == len(image_paths) and not missing else 1)
' "$data_path" "$interval" "$order_name"
}

has_any_depth_priors() {
    local data_path=$1
    local interval=$2
    local order_name=${3:-}
    "$PYTHON" -c '
import sys
from pathlib import Path

sys.path.insert(0, "scripts/inference")
from adapter_utils import find_image_paths, resolve_optional_depth_prior_paths, select_interval

data_path = Path(sys.argv[1])
interval = int(sys.argv[2])
order_name = sys.argv[3]
if order_name:
    import os
    os.environ["BENCHMARK_IMAGE_ORDER"] = order_name
image_paths = select_interval(find_image_paths(data_path / "images"), interval)
if not image_paths:
    raise SystemExit(1)
depth_paths, _missing = resolve_optional_depth_prior_paths(data_path, image_paths)
raise SystemExit(0 if any(path is not None for path in depth_paths) else 1)
' "$data_path" "$interval" "$order_name"
}

selected_image_count() {
    local data_path=$1
    local interval=$2
    local order_name=${3:-}
    local colmap_path=""
    if track_uses_cameras; then
        colmap_path=$(materialized_camera_prior_path "$data_path")
    fi
    "$PYTHON" -c '
import sys
from pathlib import Path

sys.path.insert(0, "scripts/inference")
from adapter_utils import (
    canonical_image_root,
    find_image_paths,
    resolve_camera_conditioned_image_paths,
    select_interval,
)

data_path = Path(sys.argv[1])
interval = int(sys.argv[2])
order_name = sys.argv[3]
colmap_path = Path(sys.argv[4]) if sys.argv[4] else None
if order_name:
    import os
    os.environ["BENCHMARK_IMAGE_ORDER"] = order_name
image_paths = find_image_paths(canonical_image_root(data_path))
if colmap_path is not None:
    image_paths = resolve_camera_conditioned_image_paths(
        data_path,
        image_paths,
        colmap_path=colmap_path,
    )
image_paths = select_interval(image_paths, interval)
print(len(image_paths))
' "$data_path" "$interval" "$order_name" "$colmap_path"
}

is_dust3r_family_method() {
    case "$METHOD" in
        POW3R|DUST3R|MAST3R)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_dust3r_degenerate_log() {
    local log_path=$1
    [ -s "$log_path" ] || return 1
    grep -Eiq \
        "DUSt3R-family produced zero image pairs|DUSt3R-family global_aligner failed|list\(len=0, first=empty\)|point_coordinate_variance|minimum_spanning_tree|solvePnPRansac|not enough valid image pairs|not enough values to unpack|zero image pairs|local variable 'confs11' referenced before assignment" \
        "$log_path"
}

is_evaluator_degenerate_log() {
    local log_path=$1
    [ -s "$log_path" ] || return 1
    grep -Eiq \
        "Degenerate covariance rank, Umeyama alignment is not possible|evo\.core\.geometry\.GeometryException|Need at least two matched images for relative pose metrics" \
        "$log_path"
}

# Whether an evaluation JSON holds at least one scene result. A file that only records an error (an evaluator crash,
# a missing GT) is not a result: the case is evaluated again rather than reused.
eval_file_has_results() {
    local json_path=$1
    [ -s "$json_path" ] && grep -Eq '"num_scene_results": *[1-9]' "$json_path"
}

is_evaluator_degenerate_json() {
    local json_path=$1
    [ -s "$json_path" ] || return 1
    "$PYTHON" - "$json_path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception:
    raise SystemExit(1)

patterns = (
    "Degenerate covariance rank, Umeyama alignment is not possible",
    "evo.core.geometry.GeometryException",
    "Need at least two matched images for relative pose metrics",
)
for item in payload.get("errors", []):
    message = str(item.get("error", ""))
    if any(pattern in message for pattern in patterns):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

# Cause of a failed inference from its exit status and log: gpu_oom, host_oom (killed by the kernel / cgroup OOM
# killer, exit 137, or a Python MemoryError) or error.
classify_case_failure() {
    local status=$1
    local log_file=$2
    if grep -qiE "CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation" "$log_file" 2>/dev/null; then
        echo gpu_oom
    elif [ "$status" -eq 137 ] || grep -qE "MemoryError|Cannot allocate memory|oom[-_]kill" "$log_file" 2>/dev/null; then
        echo host_oom
    else
        echo error
    fi
}

# Hand a failure cause to run_one_case_line (which writes case_failures*.tsv) through CASE_REASON_FILE.
report_case_reason() {
    if [ -n "${CASE_REASON_FILE:-}" ]; then
        printf "%s\n" "$1" > "$CASE_REASON_FILE"
    fi
}

run_case() {
    local task=$1
    local dataset=$2
    local scene=$3
    local interval=$4
    local order_name=${5:-}
    local safe_scene
    safe_scene=$(safe_name "$scene")
    local safe_order
    safe_order=$(safe_name "${order_name:-default}")
    local case_report_dir
    local output_path
    local shared_output_path
    local method_code_path
    local method_ckpt_path
    local dataset_root
    local canonical_task
    canonical_task=$(canonical_task_name "$task")
    # A dataset with its own frame cap (SpatialArena: whole VideoDepth clips, 300 SfM images) replaces the suite's
    # 100 for this case only; both limits follow it, so a capped case still runs as one forward pass.
    local BENCHMARK_MAX_INPUT_FRAMES=$BENCHMARK_MAX_INPUT_FRAMES
    local MAX_IMAGES_PER_RUN=$MAX_IMAGES_PER_RUN
    if [[ "$BENCHMARK_SUITE" == full-* ]] && [ "$BENCHMARK_SUITE" != full-slam ] \
        && [ "$canonical_task" != MonoDepth ]; then
        local dataset_cap
        dataset_cap=$(dataset_frame_cap "$canonical_task" "$dataset") || return 1
        if [ -n "$dataset_cap" ]; then
            BENCHMARK_MAX_INPUT_FRAMES=$dataset_cap
            MAX_IMAGES_PER_RUN=$dataset_cap
        fi
    fi
    export BENCHMARK_MAX_INPUT_FRAMES MAX_IMAGES_PER_RUN
    local depth_only=false
    case "$canonical_task" in
        MonoDepth|VideoDepth|MVRecon)
            depth_only=true
            ;;
    esac
    # MonoDepth predicts every image on its own (scripts/inference/adapter_utils.py SINGLE_VIEW_ENV): adapters run
    # independent single-view samples ([B, 1, 3, H, W]) or one image per forward pass, never a multi-view batch.
    if [ "$canonical_task" = "MonoDepth" ]; then
        BENCHMARK_SINGLE_VIEW=1
    else
        BENCHMARK_SINGLE_VIEW=0
    fi
    export BENCHMARK_SINGLE_VIEW
    # Single-view cases predict every frame (adapter_utils.resolve_capped_image_indices skips the frame cap), so their
    # cache key records no cap: predictions made under the former 100-frame cap are not reused.
    local case_max_input_frames=$BENCHMARK_MAX_INPUT_FRAMES
    if [ "$BENCHMARK_SINGLE_VIEW" = 1 ]; then
        case_max_input_frames=0
    fi
    if [ -n "$order_name" ]; then
        case_report_dir=$REPORT_DIR/$canonical_task/$dataset/$safe_scene/$safe_order
        output_path=$EXP_ROOT/$EXP_NAME/$canonical_task/$dataset/$METHOD_ID/$scene/$safe_order
    else
        case_report_dir=$REPORT_DIR/$canonical_task/$dataset/$safe_scene
        output_path=$EXP_ROOT/$EXP_NAME/$canonical_task/$dataset/$METHOD_ID/$scene
    fi
    method_code_path=$(env_value "$METHOD_CODE_ENV")
    method_ckpt_path=$(env_value "$METHOD_CKPT_ENV")
    dataset_root=$(dataset_root_for_case "$canonical_task" "$dataset")
    local data_path=$dataset_root/$scene
    if [ "$USE_SHARED_PREDICTION_CACHE" = "1" ]; then
        mkdir -p "$PREDICTION_CACHE_ROOT"
        shared_output_path=$PREDICTION_CACHE_ROOT/$METHOD/$(
            prediction_group_key \
                "$METHOD" \
                "$BENCHMARK_TRACK" \
                "$RESOLUTION" \
                "$IMAGE_LOAD_RESOLUTION" \
                "$PIXEL_LIMIT" \
                "$MAX_IMAGES_PER_RUN" \
                "$MAX_POINTS" \
                "$CONF_THRESH" \
                "$DEPTH_CONF_THRESH" \
                "$case_max_input_frames" \
                "$interval" \
                "$order_name" \
                "$data_path" \
                "$method_code_path" \
                "$method_ckpt_path" \
                "$BENCHMARK_SINGLE_VIEW"
        )/$dataset/$scene
    else
        shared_output_path=$output_path
    fi

    mkdir -p "$case_report_dir"
    if [ -n "${METHOD_TASKS:-}" ] && [[ " $METHOD_TASKS " != *" $canonical_task "* ]]; then
        echo "[$METHOD_ID] skip $task/$dataset/$scene: $METHOD only takes part in $METHOD_TASKS (MethodSpec.tasks)"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_not_applicable" >> "$SUMMARY_TSV"
        return
    fi
    if [ "$BENCHMARK_TRACK" = "rgb_cameras" ] && [[ " MonoDepth SfM SLAM " == *" $canonical_task "* ]]; then
        # Not a data-availability gap (unlike the rgb_depth/rgb_cameras
        # checks below). rgb_cameras is only a VideoDepth/MVRecon track
        # (spatialarena/task_track_matrix.py): MonoDepth is single-frame,
        # and SfM/SLAM would be handed the very poses they are scored on.
        # Excluded unconditionally, even for a scene that happens to have
        # camera priors, so a hand-run case list cannot produce these reports.
        echo "[$METHOD_ID] skip $task/$dataset/$scene: $canonical_task has no rgb_cameras track"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_not_applicable" >> "$SUMMARY_TSV"
        return
    fi
    if [ "$BENCHMARK_TRACK" = "rgb_masked_depth_cameras" ] && [[ " MVRecon SfM " != *" $canonical_task "* ]]; then
        # RGB + masked depth + cameras is an MVRecon/SfM track only (spatialarena/task_track_matrix.py).
        echo "[$METHOD_ID] skip $task/$dataset/$scene: $canonical_task has no rgb_masked_depth_cameras track"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_not_applicable" >> "$SUMMARY_TSV"
        return
    fi
    if [ "$BENCHMARK_TRACK" = "rgb_depth" ] && [ "$canonical_task" = "MVRecon" ] \
        && [[ " $MVRECON_RGB_DEPTH_DATASETS " != *" ${dataset,,} "* ]]; then
        # rgb_depth hands the method the GT depth: MVRecon runs it only where that depth is sparse or incomplete
        # (TRACK_DATASETS in spatialarena/task_track_matrix.py); dense GT depth leaves only the poses to predict.
        echo "[$METHOD_ID] skip $task/$dataset/$scene: MVRecon rgb_depth runs only on $MVRECON_RGB_DEPTH_DATASETS"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_not_applicable" >> "$SUMMARY_TSV"
        return
    fi
    if [ "$BENCHMARK_TRACK" = "rgb_depth" ] && [ "$canonical_task" = "SfM" ] \
        && [[ " $SFM_RGB_DEPTH_DATASETS " != *" ${dataset,,} "* ]]; then
        # rgb_depth hands the method the GT depth: SfM runs it only where the released GT has depth maps.
        echo "[$METHOD_ID] skip $task/$dataset/$scene: SfM rgb_depth runs only on $SFM_RGB_DEPTH_DATASETS"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_not_applicable" >> "$SUMMARY_TSV"
        return
    fi
    if [ "$BENCHMARK_TRACK" = "rgb_depth" ] && [ "$canonical_task" = "SLAM" ] \
        && { [[ " $SLAM_RGB_DEPTH_DATASETS " != *" ${dataset,,} "* ]] \
             || [[ " $SLAM_RGB_DEPTH_EXCLUDED_SCENES " == *" $scene "* ]]; }; then
        # rgb_depth hands the method the input depth maps: SLAM runs it only on the sequences that have them.
        echo "[$METHOD_ID] skip $task/$dataset/$scene: SLAM rgb_depth runs only on sequences with depth maps"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_not_applicable" >> "$SUMMARY_TSV"
        return
    fi
    if [ "$BENCHMARK_TRACK" = "rgb_cameras" ] && [ "$canonical_task" = "VideoDepth" ] \
        && [[ " $VIDEODEPTH_RGB_CAMERAS_DATASETS " != *" ${dataset,,} "* ]]; then
        # rgb_cameras hands the method the GT camera poses: VideoDepth runs it only where the released GT has them
        # (TRACK_DATASETS in spatialarena/task_track_matrix.py).
        echo "[$METHOD_ID] skip $task/$dataset/$scene: VideoDepth rgb_cameras runs only on $VIDEODEPTH_RGB_CAMERAS_DATASETS"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_not_applicable" >> "$SUMMARY_TSV"
        return
    fi
    if track_uses_sparse_depth; then
        if ! has_sparse_depth_priors "$data_path"; then
            echo "[$METHOD_ID] skip $task/$dataset/$scene: $BENCHMARK_TRACK track requires sparse depth priors" \
                "(scripts/preprocess/materialize_sparse_depth_priors.py)"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_missing_depths" >> "$SUMMARY_TSV"
            return
        fi
    elif track_uses_depth; then
        if supports_partial_depth_priors; then
            if ! has_any_depth_priors "$data_path" "$interval" "$order_name"; then
                echo "[$METHOD_ID] skip $task/$dataset/$scene: $BENCHMARK_TRACK track requires at least one matched depth prior"
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                    "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_missing_depths" >> "$SUMMARY_TSV"
                return
            fi
        elif ! has_matched_depth_priors "$data_path" "$interval" "$order_name"; then
            echo "[$METHOD_ID] skip $task/$dataset/$scene: $BENCHMARK_TRACK track requires matched depth priors"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_missing_depths" >> "$SUMMARY_TSV"
            return
        fi
    fi
    if track_uses_cameras && ! validate_camera_prior_mapping "$data_path"; then
        echo "[$METHOD_ID] skip $task/$dataset/$scene: $BENCHMARK_TRACK camera-prior mapping failed"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_missing_sparse" >> "$SUMMARY_TSV"
        return
    fi
    # FRAME_PROFILE=1 is set only by scripts/test/profile_max_input_frames.sh, which measures single-pass budgets on a
    # SLAM recording through the inference phase (no evaluation).
    if [ "$canonical_task" = "SLAM" ] && [ "$BENCHMARK_SUITE" != "full-slam" ] \
        && ! { [ "${FRAME_PROFILE:-0}" = 1 ] && [ "${CASE_PHASE:-}" = inference ]; }; then
        echo "SLAM case $dataset/$scene: SLAM runs only in the full-slam suite (docs/benchmark/slam_protocol.md)" >&2
        return 5
    fi
    if [ "$BENCHMARK_SUITE" = "full-slam" ]; then
        local slam_stride
        slam_stride=$(slam_case_stride "$dataset" 2>/dev/null) || slam_stride="(not in range)"
        if [ "$canonical_task" != "SLAM" ] || [ "$interval" != "$slam_stride" ] || [ -n "$order_name" ]; then
            echo "full-slam $SLAM_RANGE-range case $task/$dataset/$scene must be SLAM with interval=$slam_stride and no order (got interval=$interval order=${order_name:-})" >&2
            return 5
        fi
        local slam_frames slam_eligibility
        slam_frames=$(selected_image_count "$data_path" "$interval" "")
        slam_eligibility=$("$PYTHON" -m spatialarena.slam_protocol check "$METHOD" "$SLAM_RANGE" "$BENCHMARK_TRACK" "$slam_frames")
        if [ "$slam_eligibility" = "not_in_track" ]; then
            echo "[$METHOD_ID] skip $task/$dataset/$scene: not part of the $SLAM_RANGE-range SLAM track"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_not_in_slam_track" >> "$SUMMARY_TSV"
            return
        elif [ "$slam_eligibility" != "ok" ]; then
            # A track method must fit every sequence of the track (slam_protocol.SLAM_RANGES); a sequence longer
            # than the track's max_frames means the range definition is stale, not that this case may be skipped.
            echo "full-slam $SLAM_RANGE-range: $METHOD on $dataset/$scene has $slam_frames frames ($slam_eligibility); update slam_protocol.SLAM_RANGES" >&2
            return 6
        fi
    fi
    if is_dust3r_family_method; then
        local image_count
        image_count=$(selected_image_count "$data_path" "$interval" "$order_name")
        if [ "$image_count" -lt 2 ]; then
            echo "[$METHOD_ID] skip $task/$dataset/$scene: DUSt3R-family adapters require at least two selected images"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_insufficient_images" >> "$SUMMARY_TSV"
            return
        fi
    fi

    local existing_eval_file
    existing_eval_file=$(case_eval_file "$task" "$case_report_dir")
    # A pending chunk alignment (SLAM_DEFER_ALIGN) is the newest run of the case: it goes first, even when the cache
    # entry still holds an earlier pose-only prediction that the evaluate phase would otherwise score.
    local align_pending=0
    if [ -f "$shared_output_path/slam_align_pending.json" ] && [ "$EXPORT_POSE_DEPTH" = 1 ] \
        && ! grep -q '"export_depth": true' "$shared_output_path/slam_align_pending.json"; then
        # A GPU phase from before depth export kept no chunk depth arrays: drop it so the inference phase reruns it.
        echo "[$METHOD_ID] stale GPU phase for $task/$dataset/$scene (no chunk depth); dropping it"
        rm -f "$shared_output_path/slam_align_pending.json"
    fi
    [ -f "$shared_output_path/slam_align_pending.json" ] && align_pending=1
    if [ "$CASE_PHASE" = "evaluate" ] && [ -L "$output_path" ] && [ "$align_pending" = 0 ] \
        && ! prediction_output_ready "$task" "$dataset" "$scene" "$shared_output_path" "$depth_only"; then
        # The cache key hashes the method's code and checkpoint paths, so moving them orphans earlier predictions.
        # Evaluate-only scores the prediction this experiment made, which its task alias still points to.
        local aliased_output_path
        aliased_output_path=$(readlink -f "$output_path")
        if prediction_output_ready "$task" "$dataset" "$scene" "$aliased_output_path" "$depth_only"; then
            echo "[$METHOD_ID] $task/$dataset/$scene: evaluating the aliased prediction $aliased_output_path"
            shared_output_path=$aliased_output_path
        fi
    fi
    # FORCE_REINFER=1 (config key `force_reinfer`): re-run inference although the content-addressed prediction
    # cache already has a matching entry, for testing a pipeline/adapter change the cache key doesn't capture
    # (e.g. --spatial_policy). Only forces the inference phase; an evaluate-only phase still scores whatever
    # prediction is already there.
    if [ "$align_pending" = 0 ] \
        && { [ "${FORCE_REINFER:-0}" != "1" ] || [ "$CASE_PHASE" != "inference" ]; } \
        && prediction_output_ready "$task" "$dataset" "$scene" "$shared_output_path" "$depth_only"; then
        ensure_task_output_alias "$output_path" "$shared_output_path"
        if [ "$CASE_PHASE" = "inference" ]; then
            echo "[$METHOD_ID] inference already ready for $task/$dataset/$scene"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "inference_ready" >> "$SUMMARY_TSV"
            return
        fi
        # REEVALUATE=1 (config key `reevaluate`): score the case again although it has results, after an
        # evaluator fix.
        # An evaluation older than the prediction it would stand for (inference re-ran since) is not reused.
        if [ "${REEVALUATE:-0}" != "1" ] && [ -n "$existing_eval_file" ] && eval_file_has_results "$existing_eval_file" \
            && [ -s "$case_report_dir/verify.json" ] \
            && ! [ "$shared_output_path/metrics.json" -nt "$existing_eval_file" ]; then
            echo "[$METHOD_ID] reuse existing evaluation for $task/$dataset/$scene"
            if [ "$KEEP_PREDICTIONS" != "1" ] && [ -d "$output_path" ]; then
                if [ "$KEEP_DEPTH_VISUALIZATIONS" = "1" ]; then
                    "$PYTHON" scripts/preprocess/materialize_depth_visualizations.py \
                        --prediction_root "$output_path" \
                        --output_dir "$case_report_dir/depth_visualizations" \
                        --remove_numeric_depths
                fi
                rm -rf "$output_path"
            fi
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "reused_existing_eval" >> "$SUMMARY_TSV"
            return
        fi
        echo "[$METHOD_ID] reuse shared predictions for $task/$dataset/$scene"
    elif [ -f "$shared_output_path/slam_align_pending.json" ]; then
        # GPU phase done by an inference job (SLAM_DEFER_ALIGN); the chunk alignment runs in an evaluating job.
        if [ "$CASE_PHASE" = "inference" ]; then
            echo "[$METHOD_ID] GPU phase already done for $task/$dataset/$scene; alignment pending for the evaluate phase"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "inference_align_pending" >> "$SUMMARY_TSV"
            return
        fi
        echo "[$METHOD_ID] align $task/$dataset/$scene (GPU phase done by the inference job)"
        SLAM_ALIGN_ONLY=1 run_inference "$task" "$dataset" "$scene" "$interval" "$order_name" "$shared_output_path"
        ensure_task_output_alias "$output_path" "$shared_output_path"
    else
        if [ "$CASE_PHASE" = "evaluate" ] && [ -s "$case_report_dir/inference_failure.tsv" ]; then
            # The inference job failed on this case (status and cause recorded there); report it as failed again so
            # this phase's case_failures*.tsv keeps the cause instead of a bare "missing prediction".
            local failed_status failed_reason
            IFS=$'\t' read -r failed_status failed_reason < "$case_report_dir/inference_failure.tsv"
            echo "[$METHOD_ID] $task/$dataset/$scene: inference failed earlier ($failed_reason, status $failed_status); see $case_report_dir/inference.log" >&2
            report_case_reason "$failed_reason"
            return "$failed_status"
        fi
        if [ "$CASE_PHASE" = "evaluate" ]; then
            echo "[$METHOD_ID] skip $task/$dataset/$scene: no prediction found for evaluate-only phase" >&2
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "missing_prediction_for_eval_phase" >> "$SUMMARY_TSV"
            return
        fi
        rm -rf "$output_path"
        if [ "${FORCE_REINFER:-0}" = "1" ] && [ "$CASE_PHASE" = "inference" ]; then
            # Clear the shared cache entry itself (not just this task's alias), so the rerun starts clean and
            # doesn't mix stale files from the old run with the new one.
            rm -rf "$shared_output_path"
        fi
        if is_dust3r_family_method; then
            local inference_stdout=$case_report_dir/inference.log
            local inference_stderr=$case_report_dir/inference.err
            set +e
            run_inference "$task" "$dataset" "$scene" "$interval" "$order_name" "$shared_output_path" \
                >"$inference_stdout" 2>"$inference_stderr"
            local inference_status=$?
            set -e
            if [ "$inference_status" -ne 0 ]; then
                if is_dust3r_degenerate_log "$inference_stderr"; then
                    echo "[$METHOD_ID] skip $task/$dataset/$scene: DUSt3R-family alignment degenerated; see $inference_stderr"
                    rm -rf "$output_path"
                    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                        "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_degenerate_alignment" >> "$SUMMARY_TSV"
                    return
                fi
                cat "$inference_stdout"
                cat "$inference_stderr" >&2
                return "$inference_status"
            fi
        else
            # Keep the inference log with the case, so a failure can be classified (OOM or not) and looked up later.
            mkdir -p "$case_report_dir"
            local inference_log=$case_report_dir/inference.log
            set +e
            (
                set -e
                run_inference "$task" "$dataset" "$scene" "$interval" "$order_name" "$shared_output_path"
            ) 2>&1 | tee "$inference_log"
            local inference_status=${PIPESTATUS[0]}
            set -e
            if [ "$inference_status" -ne 0 ]; then
                local failure_reason
                failure_reason=$(classify_case_failure "$inference_status" "$inference_log")
                printf "%s\t%s\n" "$inference_status" "$failure_reason" > "$case_report_dir/inference_failure.tsv"
                report_case_reason "$failure_reason"
                return "$inference_status"
            fi
            rm -f "$case_report_dir/inference_failure.tsv"
        fi
        ensure_task_output_alias "$output_path" "$shared_output_path"
        if [ "$CASE_PHASE" = "inference" ]; then
            echo "[$METHOD_ID] inference complete for $task/$dataset/$scene"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "inference_complete" >> "$SUMMARY_TSV"
            return
        fi
    fi
    verify_case "$task" "$dataset" "$scene" "$case_report_dir" "$order_name" "$depth_only"
    local eval_stdout=$case_report_dir/evaluation.log
    local eval_stderr=$case_report_dir/evaluation.err
    local eval_output_json
    eval_output_json=$(case_eval_file "$task" "$case_report_dir")
    set +e
    evaluate_case "$task" "$dataset" "$scene" "$case_report_dir" "$order_name" >"$eval_stdout" 2>"$eval_stderr"
    local eval_status=$?
    set -e
    if [ "$eval_status" -ne 0 ]; then
        if is_evaluator_degenerate_log "$eval_stderr" || is_evaluator_degenerate_log "$eval_stdout" || is_evaluator_degenerate_json "$eval_output_json"; then
            echo "[$METHOD_ID] skip $task/$dataset/$scene: evaluator alignment degenerated; see $eval_stderr"
            # An alias into the prediction cache is the case's only link to its prediction: keep it, so a fixed
            # evaluator can re-evaluate without inference. Only the case's own prediction folder is dropped.
            [ -L "$output_path" ] || rm -rf "$output_path"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "skipped_degenerate_alignment" >> "$SUMMARY_TSV"
            return
        fi
        cat "$eval_stdout"
        cat "$eval_stderr" >&2
        return "$eval_status"
    fi

    if [ -z "$case_report_dir" ] || [ -z "$output_path" ]; then
        echo "Internal benchmark runner error: missing report/output path for $task/$dataset/$scene" >&2
        return 2
    fi

    if [ "$KEEP_DEPTH_VISUALIZATIONS" = "1" ]; then
        local -a visualization_cmd
        visualization_cmd=(
            "$PYTHON" scripts/preprocess/materialize_depth_visualizations.py
            --prediction_root "$output_path"
            --output_dir "$case_report_dir/depth_visualizations"
        )
        if [ "$KEEP_PREDICTIONS" != "1" ]; then
            visualization_cmd+=(--remove_numeric_depths)
        fi
        "${visualization_cmd[@]}"
    fi

    local prediction_retention=removed_after_eval
    if [ "$KEEP_PREDICTIONS" = "1" ]; then
        prediction_retention=kept
    else
        rm -rf "$output_path"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$output_path" "$case_report_dir" "$prediction_retention" >> "$SUMMARY_TSV"
}

run_one_case_line() {
    local case_line=$1
    local task dataset scene interval order_name extra
    IFS='|' read -r task dataset scene interval order_name extra <<< "$case_line"
    if [ -n "${extra:-}" ] || [ -z "${dataset:-}" ] || [ -z "${scene:-}" ] || [ -z "${interval:-}" ]; then
        echo "Invalid case line in $CASE_FILE: $case_line" >&2
        exit 5
    fi
    local reason_file
    reason_file=$(mktemp "${TMPDIR:-/tmp}/case_reason.XXXXXX")
    set +e
    (
        set -e
        CASE_REASON_FILE=$reason_file run_case "$task" "$dataset" "$scene" "$interval" "$order_name"
    )
    local status=$?
    set -e
    local reason=""
    [ -s "$reason_file" ] && reason=$(head -n 1 "$reason_file")
    rm -f "$reason_file"
    if [ "$status" -ne 0 ]; then
        # printf here is a single sub-PIPE_BUF write to an O_APPEND file, so
        # concurrent cases appending to the same TSV don't interleave.
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$METHOD_ID" "$task" "$dataset" "$scene" "$interval" "$order_name" "$status" "$reason" >> "$CASE_FAILURES_TSV"
        echo "Case failed for $METHOD_ID: $task/$dataset/$scene interval=$interval order=${order_name:-default} status=$status${reason:+ reason=$reason}" >&2
    fi
}

running_cases=0
while IFS= read -r case_line; do
    case_line=${case_line%$'\r'}
    if [ -z "${case_line//[[:space:]|]/}" ] || [[ "$case_line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    # dtu's evaluate_dtu_mvsnet (point-cloud downsample/mask/STL-comparison
    # against up to ~1M-point reference clouds) has shown unexplained
    # cumulative memory growth across cases -- it OOM-killed repeatedly even
    # at 192G running strictly sequentially (job 132008, 2026-09-14). Until
    # that leak is root-caused, never run a dtu case concurrently with
    # anything else: drain any in-flight parallel cases first, run it alone,
    # then resume parallel dispatch for whatever follows.
    peek_dataset=""
    IFS='|' read -r _peek_task peek_dataset _peek_rest <<< "$case_line"
    if [ "$CASE_PARALLELISM" -le 1 ] || [ "${peek_dataset,,}" = "dtu" ]; then
        if [ "$running_cases" -gt 0 ]; then
            wait
            running_cases=0
        fi
        run_one_case_line "$case_line"
        continue
    fi
    run_one_case_line "$case_line" &
    running_cases=$((running_cases + 1))
    if [ "$running_cases" -ge "$CASE_PARALLELISM" ]; then
        wait -n
        running_cases=$((running_cases - 1))
    fi
done < "$CASE_FILE"
if [ "$running_cases" -gt 0 ]; then
    wait
fi

"$PYTHON" - \
    "$REPORT_DIR/summary${SUMMARY_SUFFIX}.json" \
    "$METHOD_ID" \
    "$METHOD" \
    "$BENCHMARK_TRACK" \
    "$EXP_ROOT" \
    "$EXP_NAME" \
    "$CASE_FILE" \
    "$REPORT_DIR" \
    "$SUMMARY_TSV" \
    "$CASE_FAILURES_TSV" \
    "$KEEP_PREDICTIONS" \
    "$KEEP_DEPTH_VISUALIZATIONS" \
    "$GENERATE_PDF_REPORT" \
    "$ALLOW_BENCHMARK_ROOT_OUTPUT" \
    "$ALLOW_BENCHMARK_EVAL_OUTPUT" \
    "$BENCHMARK_AUX_ROOT" \
    "$BENCHMARK_EVAL_ROOT" \
    "$PREDICTION_CACHE_ROOT" \
    "$RESOLUTION" \
    "$IMAGE_LOAD_RESOLUTION" \
    "$PIXEL_LIMIT" \
    "$CONF_THRESH" \
    "$DEPTH_CONF_THRESH" \
    "$METHOD_RESOLUTION_SOURCE" <<'PY'
import json
import os
import sys
from pathlib import Path


def as_int(value: str) -> int:
    return int(float(value)) if value else 0


def as_float(value: str) -> float:
    return float(value) if value else 0.0


(
    output_path,
    method_id,
    method,
    benchmark_track,
    exp_root,
    exp_name,
    case_file,
    report_dir,
    summary_tsv,
    case_failures_tsv,
    keep_predictions,
    keep_depth_visualizations,
    generate_pdf_report,
    allow_benchmark_root_output,
    allow_benchmark_eval_output,
    benchmark_aux_root,
    benchmark_eval_root,
    prediction_cache_root,
    resolution,
    image_load_resolution,
    pixel_limit,
    conf_thresh,
    depth_conf_thresh,
    method_resolution_source,
) = sys.argv[1:]

summary = {
    "method": method_id,
    "base_method": method,
    "benchmark_track": benchmark_track,
    "experiment_root": exp_root,
    "experiment_name": exp_name,
    "case_file": case_file,
    "report_dir": report_dir,
    "summary_tsv": summary_tsv,
    "case_failures_tsv": case_failures_tsv,
    "keep_predictions": keep_predictions,
    "keep_depth_visualizations": keep_depth_visualizations,
    "generate_pdf_report": generate_pdf_report,
    "allow_benchmark_root_output": allow_benchmark_root_output,
    "allow_benchmark_eval_output": allow_benchmark_eval_output,
    "benchmark_aux_root": benchmark_aux_root,
    "benchmark_eval_root": benchmark_eval_root,
    "prediction_cache_root": prediction_cache_root,
    "resolution_profile": {
        "resolution": as_int(resolution),
        "image_load_resolution": as_int(image_load_resolution),
        "pixel_limit": as_int(pixel_limit),
        "conf_thresh": as_float(conf_thresh),
        "depth_conf_thresh": as_float(depth_conf_thresh),
        "resolution_source": method_resolution_source,
        "dense_depth_confidence_filtering": False,
    },
}

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")
# Every run of the method shares this folder: summary<suffix>.json is this run's; summary.json (how the dashboard
# finds report folders) is replaced atomically so concurrent runs never leave it half-written.
shared_summary = Path(output_path).with_name("summary.json")
if Path(output_path) != shared_summary:
    partial = shared_summary.with_name(f".summary.json.{os.getpid()}")
    partial.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, shared_summary)
PY

if [ "$GENERATE_PDF_REPORT" = "1" ]; then
    "$PYTHON" scripts/test/generate_benchmark_pdf_report.py \
        "$REPORT_DIR" \
        --output "$REPORT_DIR/.report_$$.pdf" \
        --title "FeedForward Benchmark: $METHOD_ID ($BENCHMARK_TRACK)" \
        && mv -f "$REPORT_DIR/.report_$$.pdf" "$REPORT_DIR/report.pdf"  # atomic: runs share the folder
fi

echo "Sampled benchmark completed for $METHOD_ID"
echo "Experiment: $EXP_ROOT/$EXP_NAME"
echo "Prediction cache: $PREDICTION_CACHE_ROOT"
echo "Reports: $REPORT_DIR"
