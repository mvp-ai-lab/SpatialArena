#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# HOME is unset under sbatch --export=NIL.
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}

set -euo pipefail

OUT_DIR=${VGGT_OMEGA_CKPT_DIR:-$HOME/Projects/FoundationModels/pretrained}
GDOWN=${GDOWN:-gdown}

mkdir -p "$OUT_DIR"

# gdown 6.0.0 dropped --fuzzy but auto-detects a full share URL without it
# (verified 2026-09-25); plain "$GDOWN <url> -O <dest>" works.
"$GDOWN" "https://drive.google.com/file/d/1GIM_3OTk9iCkunD5-1jgkYJM1hcgkwDV/view" \
    -O "$OUT_DIR/vggt_omega_1b_512.pt"
"$GDOWN" "https://drive.google.com/file/d/1778LS0SYy30L0CfLDmzlRsEy6xFUqBdR/view" \
    -O "$OUT_DIR/vggt_omega_1b_256_text.pt"

echo "VGGT-Omega checkpoints:"
echo "$OUT_DIR/vggt_omega_1b_512.pt"
echo "$OUT_DIR/vggt_omega_1b_256_text.pt"
