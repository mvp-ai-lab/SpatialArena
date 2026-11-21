#!/usr/bin/env bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=hf-benchmark-release
#SBATCH --output=/home/%u/tmp/hf-benchmark-release_%j.log
#SBATCH --error=/home/%u/tmp/hf-benchmark-release_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

# Stage and upload the benchmark release to the Hugging Face Hub.
#   1. build methods/METHODS.md + methods/methods_manifest.json (code/checkpoint sources, nothing re-hosted)
#   2. stage every license-cleared dataset as one .tar per scene (ARCHIVE=1, default; the raw ~900k-file tree
#      hits Hub API rate limits) or as hard links (ARCHIVE=0), plus LICENSE_NOTICE.md + README card
#   3. HfApi.upload_folder (resumable: rerunning this script continues where it stopped)
# Usage: sbatch --export=NIL scripts/release/upload_benchmark_release.sh
#        (overrides: sbatch --export=NIL --wrap "env UPLOAD=0 REPO_ID=... bash .../upload_benchmark_release.sh")

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

PYTHON=${PYTHON:-$HOME/anaconda3/envs/si_bench/bin/python}
REPO_DIR=${REPO_DIR:-$HOME/Projects/SpatialArena}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks}
# Must share a filesystem with BENCHMARK_ROOT for hard links to cost no space.
ARCHIVE=${ARCHIVE:-1}
if [[ "${ARCHIVE}" == "1" ]]; then
  STAGING_ROOT=${STAGING_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/_hf_release_archives}
  LAYOUT_ARGS=(--archive_scenes --archive_workers "${ARCHIVE_WORKERS:-16}")
else
  STAGING_ROOT=${STAGING_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/_hf_release_staging}
  LAYOUT_ARGS=(--hardlink)
fi
REPO_ID=${REPO_ID:-Bluefish2333/benchmark}
UPLOAD=${UPLOAD:-1}  # 0: stage and write the card only, for inspection

# Keep the Xet chunk/shard cache on node-local disk (the home cache is NFS on compute nodes and produced
# "MerkleDB Shard error: File I/O error" under load) and skip per-file progress bars (they bloat the log to GBs).
export HF_XET_CACHE=${HF_XET_CACHE:-${TMPDIR:-/tmp}/$(id -un)_hf_xet_cache}
export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}
cd "${REPO_DIR}"
echo "repo=${REPO_ID} staging=${STAGING_ROOT} commit=$(git rev-parse --short HEAD)"
"${PYTHON}" scripts/release/build_method_manifests.py --output_dir "${STAGING_ROOT}/methods"
"${PYTHON}" scripts/release/pack_and_upload_benchmark_datasets.py \
  --benchmark_root "${BENCHMARK_ROOT}" --staging_root "${STAGING_ROOT}" \
  --stage "${LAYOUT_ARGS[@]}" --upload_workers "${UPLOAD_WORKERS:-8}" --repo_id "${REPO_ID}" $([[ "${UPLOAD}" == "1" ]] && echo --upload)
