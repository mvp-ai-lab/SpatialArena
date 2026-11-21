#!/usr/bin/env bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# Download the ETH3D high-resolution multi-view training scenes into the canonical MVRecon/eth3d/<scene>/ layout
# (SfM/eth3d links to it). https://www.eth3d.net/datasets
#
# Each scene's per-scene archives extract in place:
#   <scene>_dslr_jpg.7z          -> images/dslr_images/, dslr_calibration_jpg/
#   <scene>_dslr_undistorted.7z  -> images/dslr_images_undistorted/, dslr_calibration_undistorted/
#   <scene>_dslr_depth.7z        -> ground_truth_depth/dslr_images/ (laser-scan depth of the undistorted images)
# Then run scripts/preprocess/finalize_mvrecon_layout.sh (depths/ and metadata) and
# scripts/preprocess/materialize_da3_bench.py (the GT meshes of the 11 scenes DA3-BENCH ships).
#
#   bash scripts/download/eth3d.sh [output_dir]   # default: $BENCHMARK_AUX_ROOT/MVRecon/eth3d

set -euo pipefail

HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}  # sbatch --export=NIL leaves HOME unset
BENCHMARK_AUX_ROOT=${BENCHMARK_AUX_ROOT:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks}
TARGET=${1:-$BENCHMARK_AUX_ROOT/MVRecon/eth3d}
SEVEN_Z=${SEVEN_Z:-$(command -v 7z || command -v 7za || true)}
if [ -z "$SEVEN_Z" ]; then
    echo "ERROR: 7z/7za is required to extract the ETH3D .7z archives (set SEVEN_Z to its path)." >&2
    exit 1
fi

SCENES=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)
PARTS=(dslr_jpg dslr_undistorted dslr_depth)

mkdir -p "$TARGET"
cd "$TARGET"
for scene in "${SCENES[@]}"; do
    for part in "${PARTS[@]}"; do
        archive=${scene}_${part}.7z
        wget -c --tries=20 --waitretry=30 --retry-connrefused --timeout=60 "https://www.eth3d.net/data/$archive"
        "$SEVEN_Z" x -y "$archive"
        rm -f "$archive"
    done
done
echo "ETH3D downloaded to $TARGET"
