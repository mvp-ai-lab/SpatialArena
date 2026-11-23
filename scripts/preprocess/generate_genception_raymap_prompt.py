# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Regenerate GenCeption's pose/raymap prompt embedding locally.

This is a cross-check; the official embedding is released. Superseded: the
GenCeption authors released `assets/genception_prompt_embeddings/raymap.npy`
with their answer to
https://github.com/google-deepmind/representations4d/issues/10 (closed
2026-11-10), and scripts/download/genception.sh downloads it; the benchmark
uses that file. This script remains for checking a locally encoded embedding
against it. It reconstructs the embedding following the issue reporter's
write-up:

  1. Prompt template: "Input: RGB. Output: <task>." -- inferred by matching
     UMT5 token counts against the released embeddings (depth.npy has 11
     non-zero token rows, matching "Input: RGB. Output: Depth."; normals.npy
     has 10, matching "Input: RGB. Output: Surface Normal."). For pose, the
     reporter used "Input: RGB. Output: Raymap.".
  2. Encoder: UMT5-XXL (google/umt5-xxl on the HF hub) in fp32 -- the
     reporter reports this reproduces the released depth.npy to 1.2e-5
     relative error, which is the whole basis for trusting this approach.
  3. Padding: real (non-zero) token embeddings followed by zero rows out to
     max_seq_len=226 (matching genception/pipeline.py's text_dim=4096,
     and run_inference.py's zero-embedding sentinel for "no text").

Before trusting the generated raymap.npy, this script ALWAYS regenerates
depth.npy with the same pipeline and compares it against the officially
released file at the same relative-error tolerance the issue reporter
verified (1.2e-5) -- if that check fails, something about this
reconstruction (model revision, tokenization, precision) has drifted from
what actually reproduces GenCeption's real released assets, and the pose
embedding this script produces should not be trusted either.

This script only needs torch + transformers -- it never imports jax. Run it
with the si_bench env's python (scripts/slurm/build_si_bench_env.sh), not
genception's: verified via two separate real failed runs (2026-09-25) that
having torch present at all in the genception (JAX/Flax) env breaks jax's
own CUDA 12 plugin, for reasons that held regardless of which torch version
was installed -- see scripts/slurm/build_genception_env.sh's comments.

Usage:

    /path/to/si_bench/bin/python \\
        scripts/preprocess/generate_genception_raymap_prompt.py \\
        --released_depth_embedding /path/to/prompt_embeddings/depth.npy \\
        --output_dir /path/to/prompt_embeddings
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MAX_SEQ_LEN = 226
TEXT_DIM = 4096
UMT5_MODEL_ID = "google/umt5-xxl"

DEPTH_PROMPT = "Input: RGB. Output: Depth."
RAYMAP_PROMPT = "Input: RGB. Output: Raymap."

# The issue reporter's own reproduction fidelity against the officially
# released depth.npy -- our own regeneration must land within this order of
# magnitude before the raymap embedding is trusted.
VERIFICATION_RELATIVE_ERROR_TOLERANCE = 1e-4


def encode_prompt(tokenizer, model, text: str) -> np.ndarray:
    """Encodes a prompt with UMT5 and zero-pads it to MAX_SEQ_LEN tokens.

    Args:
        tokenizer: The UMT5 tokenizer.
        model: The UMT5 encoder model.
        text: The prompt.

    Returns:
        A float32 array of shape [1, MAX_SEQ_LEN, TEXT_DIM].

    Raises:
        ValueError: If the prompt is too long or the hidden size is not
            TEXT_DIM.
    """
    import torch  # pylint: disable=import-outside-toplevel

    inputs = tokenizer(text,
                       return_tensors="pt",
                       padding=False,
                       truncation=False)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    with torch.no_grad():
        hidden_states = model(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
    # Shape [real_len, text_dim].
    hidden_states = hidden_states[0].float().cpu().numpy()

    real_len = hidden_states.shape[0]
    if real_len > MAX_SEQ_LEN:
        raise ValueError(f"prompt {text!r} tokenizes to {real_len} tokens, "
                         f"exceeding max_seq_len={MAX_SEQ_LEN}")
    if hidden_states.shape[1] != TEXT_DIM:
        raise ValueError(
            f"unexpected UMT5 hidden size {hidden_states.shape[1]}, "
            f"expected {TEXT_DIM}")

    padded = np.zeros((1, MAX_SEQ_LEN, TEXT_DIM), dtype=np.float32)
    padded[0, :real_len] = hidden_states
    return padded


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--released_depth_embedding",
        type=Path,
        required=True,
        help=("Path to the officially released depth.npy "
              "(see scripts/download/genception.sh)."),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--umt5_model_id", type=str, default=UMT5_MODEL_ID)
    return parser.parse_args()


def _verify_depth_reproduction(tokenizer, model,
                               released_depth_path: Path) -> None:
    """Checks that this pipeline reproduces the released depth.npy.

    Args:
        tokenizer: The UMT5 tokenizer.
        model: The UMT5 encoder model.
        released_depth_path: The officially released depth.npy.

    Raises:
        ValueError: If the regenerated embedding has a different shape.
        RuntimeError: If the relative error exceeds the tolerance.
    """
    print(f"Regenerating depth.npy for verification "
          f"(prompt: {DEPTH_PROMPT!r})...")
    regenerated_depth = encode_prompt(tokenizer, model, DEPTH_PROMPT)
    released_depth = np.load(released_depth_path)
    if regenerated_depth.shape != released_depth.shape:
        raise ValueError(
            f"regenerated depth embedding shape {regenerated_depth.shape} "
            f"!= released {released_depth.shape} -- something about this "
            "reconstruction has drifted.")
    relative_error = np.linalg.norm(
        regenerated_depth - released_depth) / np.linalg.norm(released_depth)
    print(f"Relative error vs. released depth.npy: {relative_error:.3e}")
    if relative_error > VERIFICATION_RELATIVE_ERROR_TOLERANCE:
        raise RuntimeError(
            f"Verification failed: relative error {relative_error:.3e} "
            f"exceeds tolerance {VERIFICATION_RELATIVE_ERROR_TOLERANCE:.0e}. "
            "Refusing to write a pose/raymap embedding whose generation "
            "pipeline doesn't reproduce a known-good release, since there's "
            "no other way to check its correctness.")
    print("Verification passed -- this UMT5 pipeline reproduces the released "
          "depth.npy.")


def main() -> None:
    """Verifies the pipeline on depth.npy, then writes raymap.npy."""
    args = parse_args()
    if not args.released_depth_embedding.is_file():
        raise FileNotFoundError(f"{args.released_depth_embedding} not found "
                                "-- run scripts/download/genception.sh first.")

    # pylint: disable=import-outside-toplevel
    import torch
    import transformers
    # pylint: enable=import-outside-toplevel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.umt5_model_id} in fp32 on {device}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.umt5_model_id)
    model = transformers.UMT5EncoderModel.from_pretrained(
        args.umt5_model_id, torch_dtype=torch.float32).to(device).eval()

    _verify_depth_reproduction(tokenizer, model, args.released_depth_embedding)

    print(f"Generating raymap.npy (prompt: {RAYMAP_PROMPT!r})...")
    raymap_embedding = encode_prompt(tokenizer, model, RAYMAP_PROMPT)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "raymap.npy"
    np.save(output_path, raymap_embedding)
    print(f"Wrote {output_path} (shape={raymap_embedding.shape}, "
          f"dtype={raymap_embedding.dtype})")
    print("NOTE: a local re-encoding; the benchmark uses the officially "
          "released raymap.npy (scripts/download/genception.sh). Compare "
          "against it rather than replacing it.")


if __name__ == "__main__":
    main()
