#!/usr/bin/env bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

GDOWN=${GDOWN:-gdown}
OUTPUT_DIR=${OUTPUT_DIR:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/texturepoorsfm}
ARCHIVE_NAME=${ARCHIVE_NAME:-TexturePoorSfM_dataset.tar}
DATASET_ID=${DATASET_ID:-1UDo8K0uYCi-YtpwLviE6hT-6Mxm5jHYH}
EXTRACT=${EXTRACT:-1}

mkdir -p "$OUTPUT_DIR"

if ! command -v "$GDOWN" >/dev/null 2>&1; then
    echo "gdown is required. Install gdown or set GDOWN to a working downloader." >&2
    exit 2
fi

"$GDOWN" "$DATASET_ID" -O "$OUTPUT_DIR/$ARCHIVE_NAME"

if [ "$EXTRACT" = "1" ]; then
    if ! command -v tar >/dev/null 2>&1; then
        echo "tar is required for EXTRACT=1." >&2
        exit 2
    fi
    tar -xf "$OUTPUT_DIR/$ARCHIVE_NAME" -C "$OUTPUT_DIR"
fi

cat <<EOF
Downloaded TexturePoorSfM dataset to:
  $OUTPUT_DIR

Materialize it into the benchmark layout with:
  python scripts/preprocess/materialize_texturepoorsfm.py \
    --source_root "$OUTPUT_DIR" \
    --benchmark_root $HOME/datasets/benchmarks
EOF
