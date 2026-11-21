#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#SBATCH --job-name=dl-genception
#SBATCH --output=/home/%u/tmp/dl-genception_%j.log
#SBATCH --error=/home/%u/tmp/dl-genception_%j.err

#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00

# Downloads the GenCeption 14B checkpoint (transformer + VAE) and the
# officially-released precomputed prompt embeddings (depth, surface normal,
# camera-pose raymap, negative) into the canonical shared checkpoint store.
# All URLs are public, unauthenticated GCS links from
# google-deepmind/representations4d's README, colab notebook and (raymap.npy)
# the answer to google-deepmind/representations4d#10 -- verified reachable.

set -euo pipefail

MODEL_SIZE=${GENCEPTION_MODEL_SIZE:-14b}
HOME=${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}  # sbatch --export=NIL leaves HOME unset
OUT_DIR=${GENCEPTION_CKPT_DIR:-${SPATIALARENA_ROOT:-$HOME/spatialarena}/benchmarks/_checkpoints/local/genception}
EMBEDDINGS_DIR=${GENCEPTION_EMBEDDINGS_DIR:-$OUT_DIR/prompt_embeddings}

BASE_CKPT_URL=https://storage.googleapis.com/representations4d/checkpoints/genception
BASE_EMBEDS_URL=https://storage.googleapis.com/representations4d/assets/genception_prompt_embeddings

mkdir -p "$OUT_DIR/vae" "$EMBEDDINGS_DIR"

echo "Downloading GenCeption ${MODEL_SIZE} transformer + config..."
wget -q --show-progress -nc "$BASE_CKPT_URL/genception_${MODEL_SIZE}_transformer.npz" -P "$OUT_DIR"
wget -q -nc "$BASE_CKPT_URL/genception_${MODEL_SIZE}_config.json" -P "$OUT_DIR"

echo "Downloading VAE..."
wget -q -nc "$BASE_CKPT_URL/vae/config.json" -P "$OUT_DIR/vae"
wget -q --show-progress -nc "$BASE_CKPT_URL/vae/diffusion_pytorch_model.safetensors" -P "$OUT_DIR/vae"

echo "Downloading officially-released prompt embeddings (depth, normals, raymap, negative)..."
for name in depth.npy normals.npy raymap.npy negative.npy; do
    wget -q -nc "$BASE_EMBEDS_URL/$name" -P "$EMBEDDINGS_DIR"
done

echo "GenCeption ${MODEL_SIZE} checkpoint ready at $OUT_DIR"
find "$OUT_DIR" -maxdepth 2 -type f -exec ls -lh {} \;
