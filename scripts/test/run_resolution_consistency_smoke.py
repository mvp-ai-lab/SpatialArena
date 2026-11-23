#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Smoke-tests output resolution and intrinsics consistency, end to end.

This runs for real on CPU (or GPU). Unlike the adapter unit tests (mocked model
internals) or --dry_run (no inference at all), this drives each method's
*real* adapter CLI entry point, with its real checkpoint and real vendored
model code, on a tiny synthetic scene, then checks that the exported depth map
and restored COLMAP intrinsics both land back at the scene's true source image
resolution. That is the concrete, falsifiable claim this script verifies for
"methods in the same track produce depth at the same (correct) resolution":
not by inspecting code, but by actually running it.

Scope: every registered method whose adapter takes the standard
--data_path/--output_path/--export_depth/--depth_only CLI contract (see
spatialarena/methods.py's METHOD_SPECS). Excluded, with reasons (not silently
skipped):
  - The 9 streaming/long-sequence SLAM adapters (long_sequence_slam.py /
    streaming_slam.py): a different CLI contract (multi-phase subprocess
    orchestration over a real frame sequence with poses), already confirmed by
    code review to need no resize fix (fixed-resolution pre-sized SLAM frames,
    no adapter-side resize); see docs/benchmark/slam_protocol.md.
  - The official single-image MonoDepth baselines (MOGE*, MARIGOLD*, LOTUS_D,
    DEPTH_PRO, UNIDEPTH_V2, INFINIDEPTH) and DVD: their upstream code is
    CUDA-only and exports dense depth only; their real outputs are checked
    with scripts/test/check_prediction_resolution.py instead.

Usage:
    python scripts/test/run_resolution_consistency_smoke.py \\
        [--methods VGGT,DUST3R,...] [--device cpu|cuda] [--timeout_s 1200] \\
        [--output_root /path] [--keep_existing]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena import methods  # noqa: E402

# pylint: enable=wrong-import-position

INFERENCE_DIR = REPO_ROOT / "scripts" / "inference"
# See spatialarena/paths.py.
SPATIALARENA_ROOT = os.path.expanduser(
    os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
METHOD_CODE_ROOT = pathlib.Path(
    os.environ.get(
        "METHOD_CODE_ROOT",
        f"{SPATIALARENA_ROOT}/benchmarks/_model_codes",
    ))
DEFAULT_OUTPUT_ROOT = pathlib.Path(
    f"{SPATIALARENA_ROOT}/benchmarks/_scratch/resolution_smoke")

_STREAMING_SLAM_REASON = ("streaming SLAM adapter, different CLI contract, "
                          "out of scope (see docstring)")
# Adapters with a different CLI contract than the standard
# --data_path/--output_path/--track depth_only/export_colmap pattern, or that
# are out of scope for this check (see the module docstring).
EXCLUDED_METHODS = {
    name: _STREAMING_SLAM_REASON
    for name in ("VGGT_LONG", "PI_LONG", "VGGT_SLAM", "LOGER", "ZIPMAP",
                 "LINGBOT_MAP", "R3", "SCAL3R", "ABOT_RECON")
}
# The official single-image MonoDepth baselines and DVD run their upstream code
# on CUDA only and export dense depth only (no --export_colmap); their adapters
# fail any off-grid output themselves, and real runs are checked with
# scripts/test/check_prediction_resolution.py.
EXCLUDED_METHODS.update({
    name: ("CUDA-only official depth adapter (single_image_depth.py / "
           "dvd.py); see check_prediction_resolution.py")
    for name in ("MOGE", "MOGE2", "MOGE3", "MARIGOLD", "MARIGOLD_V1_1",
                 "LOTUS_D", "DEPTH_PRO", "UNIDEPTH_V2", "INFINIDEPTH", "DVD")
})
EXCLUDED_METHODS.update({
    name: ("CUDA-only official video diffusion adapter "
           "(video_diffusion_depth.py); see check_prediction_resolution.py"
          ) for name in ("DEPTHCRAFTER", "CHRONODEPTH")
})

# Methods whose adapter exposes --spatial_policy (default "legacy", unchanged
# behavior); pass "shared" to exercise the shared restoration path.
SUPPORTS_SPATIAL_POLICY = {
    "VGGT", "ROBUSTVGGT", "DA3-L", "DA3-G", "DA3-NGL", "VGGT_OMEGA", "FASTVGGT",
    "CUT3R"
}

# Methods whose adapter exposes --colmap_path explicitly (others auto-discover
# data_path/sparse/0 via --colmap_model_subdir, which the synthetic scene
# already provides at that exact path).
SUPPORTS_COLMAP_PATH_FLAG = {
    "DA3-L", "DA3-G", "DA3-NGL", "OMNIVGGT", "PI3_X", "IGGT4D", "MAPANYTHING"
}

# Methods whose adapter exposes --track.
SUPPORTS_TRACK = {
    "PI3_X",
    "OMNIVGGT",
    "MAPANYTHING",
    "IGGT",
    "IGGT4D",
    "DEJAVIEW",
    "DEPTHVLM",
    "VLM3",
    "DEPTHLM",
    "QWEN3_VL",
    "SENSENOVA_VISION",
}

# Methods that always export poses/intrinsics on the depth_only path too
# (export_colmap valid even on the "rgb" track, not just rgb_cameras). Harmless
# to request; adapters that cannot will just not write sparse/.
ALWAYS_EXPORT_COLMAP = True

# Generous per-method timeouts (seconds). Transformer-decoder VLMs and the
# diffusion-based GenCeption are expected to be slow (or impractical) on CPU;
# report a timeout honestly rather than silently drop the method.
DEFAULT_TIMEOUT_S = 1200
SLOW_METHOD_TIMEOUT_S = {
    "SENSENOVA_VISION": 2400,
    "QWEN3_VL": 2400,
    "GENCEPTION_DEPTH": 3600,
    "GENCEPTION_POSE": 3600,
    "DUST3R": 1800,
    "MAST3R": 1800,
    "POW3R": 1800,
}

# 4:3, deliberately not a multiple of 14 or 16.
SOURCE_W, SOURCE_H = 168, 126
NUM_IMAGES = 3


def build_synthetic_scene(root: pathlib.Path) -> list[str]:
    """Builds the synthetic scene under `root`.

    The scene has images/, plus depths/ and sparse/0 so every track has what
    it needs.

    Args:
        root: The scene folder.

    Returns:
        The image names (without extension).
    """

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    depths_dir = root / "depths"
    depths_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    names = []
    for index in range(NUM_IMAGES):
        name = f"{index:06d}"
        names.append(name)
        rgb = rng.randint(0, 256, size=(SOURCE_H, SOURCE_W, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(images_dir / f"{name}.jpg", quality=95)
        depth = np.linspace(1.0, 3.0, SOURCE_H * SOURCE_W,
                            dtype=np.float32).reshape(SOURCE_H, SOURCE_W)
        depth = depth + 0.1 * index
        np.save(depths_dir / f"{name}.npy", depth)

    sparse_dir = root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    fx = fy = float(max(SOURCE_W, SOURCE_H))
    cx, cy = (SOURCE_W - 1) / 2.0, (SOURCE_H - 1) / 2.0
    (sparse_dir / "cameras.txt").write_text(
        "\n".join([
            "# Camera list with one line of data per camera:",
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
            "# Number of cameras: 1",
            f"1 PINHOLE {SOURCE_W} {SOURCE_H} {fx} {fy} {cx} {cy}",
            "",
        ]),
        encoding="utf-8",
    )
    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {NUM_IMAGES}",
    ]
    for index, name in enumerate(names, start=1):
        image_lines.append(
            f"{index} 1 0 0 0 {0.1 * (index - 1)} 0.0 0.0 1 {name}.jpg")
        image_lines.append("")
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n",
                                           encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text(
        "\n".join([
            "# 3D point list with one line of data per point:",
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
            "# Number of points: 0, mean track length: 0",
            "",
        ]),
        encoding="utf-8",
    )
    return names


# The si_bench env's interpreter (scripts/slurm/build_si_bench_env.sh);
# defaults to the one running this script.
PYTHON_BIN = os.environ.get("SI_BENCH_PYTHON", sys.executable)

# The engine (run_method_case_list.sh) passes checkpoint/code paths as explicit
# CLI flags, not by relying on each adapter's own (sometimes differently-named,
# legacy) os.environ.get default. --ckpt_path is universal except GenCeption's
# --ckpt_dir/--prompt_embeddings_dir. The code-path flag name doesn't always
# match the adapter filename (confirmed per adapter via argparse
# introspection, not guessed): None means the adapter takes no such flag
# (HF-only, no vendored checkout needed).
CODE_PATH_FLAG = {
    "vggt.py": "--vggt_code_path",
    "da3.py": "--da3_code_path",
    "vggt_omega.py": "--vggt_omega_code_path",
    "fastvggt.py": "--fastvggt_code_path",
    "omnivggt.py": "--omnivggt_code_path",
    "amb3r.py": "--amb3r_code_path",
    "pi3.py": "--pi3_code_path",
    "cut3r.py": "--cut3r_code_path",
    "fast3r.py": "--fast3r_code_path",
    "mapanything.py": "--mapanything_code_path",
    "mapanything_dense_wrapper.py": "--method_code_path",
    "iggt.py": "--iggt_code_path",
    "iggt4d.py": "--iggt4d_code_path",
    "dejaview.py": "--dejaview_code_path",
    "come.py": "--come_code_path",
    "depthvlm.py": "--depthvlm_code_path",
    "sensenova_vision_depth.py": "--sensenova_vision_code_path",
    "robustvggt.py": "--robustvggt_code_path",
    "genception.py": None,
    "point_query_depth_adapter.py": None,
}
# These adapter files forward --data_path/.../* through to another module
# (mapanything_dense_wrapper for dust3r/mast3r/pow3r), so their own
# parse_args() introspection is not representative; their real adapter is one
# of the keys above.
ADAPTER_IS_WRAPPER_STUB = {"dust3r.py", "mast3r.py", "pow3r.py"}


def resolve_path_template(template: str) -> str:
    """Substitutes `$METHOD_CODE_ROOT` in a registry path."""
    return template.replace("$METHOD_CODE_ROOT", str(METHOD_CODE_ROOT))


def build_command(spec, track: str, scene_dir: pathlib.Path,
                  out_dir: pathlib.Path) -> list[str] | None:
    """Builds the adapter command line the engine would run.

    Args:
        spec: The method's `MethodSpec`.
        track: The modality track.
        scene_dir: The synthetic scene.
        out_dir: The adapter's output folder.

    Returns:
        The command.
    """
    adapter_path = INFERENCE_DIR / spec.adapter
    cmd = [
        PYTHON_BIN,
        str(adapter_path),
        "--data_path",
        str(scene_dir),
        "--output_path",
        str(out_dir),
        "--export_depth",
        "--depth_only",
        "--interval",
        "1",
        "--max_images_per_run",
        str(NUM_IMAGES),
    ]
    if spec.name == "GENCEPTION_DEPTH" or spec.name == "GENCEPTION_POSE":
        cmd += ["--ckpt_dir", resolve_path_template(spec.ckpt_default)]
        prompt_dir = pathlib.Path(resolve_path_template(
            spec.ckpt_default)) / "prompt_embeddings"
        cmd += ["--prompt_embeddings_dir", str(prompt_dir)]
        if spec.name == "GENCEPTION_DEPTH":
            cmd += ["--task", "depth"]
        else:
            cmd += ["--task", "pose"]
    else:
        cmd += ["--ckpt_path", resolve_path_template(spec.ckpt_default)]
    code_flag = CODE_PATH_FLAG.get(spec.adapter)
    if code_flag is not None and spec.code_default:
        cmd += [code_flag, resolve_path_template(spec.code_default)]
    if spec.name in SUPPORTS_TRACK:
        cmd += ["--track", track]
    if track == "rgb_cameras" and spec.name in SUPPORTS_COLMAP_PATH_FLAG:
        cmd += ["--colmap_path", str(scene_dir / "sparse" / "0")]
    if ALWAYS_EXPORT_COLMAP:
        cmd += ["--export_colmap"]
    if spec.name == "VGGT_OMEGA":
        cmd += ["--spatial_policy", "shared-pad"]
    elif spec.name in SUPPORTS_SPATIAL_POLICY:
        cmd += ["--spatial_policy", "shared"]
    return cmd


CHECKPOINT_ROOT = pathlib.Path(
    os.environ.get("CHECKPOINT_ROOT",
                   f"{SPATIALARENA_ROOT}/benchmarks/_checkpoints"))


def build_env(spec) -> dict:
    """Builds the adapter's environment (CPU only unless `--device cuda`)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["METHOD_CODE_ROOT"] = str(METHOD_CODE_ROOT)
    # Match run_method_benchmark_infer.sh's cache setup exactly. Without this,
    # torch.hub/HF fall back to ~/.cache (empty in a fresh sbatch env) and try
    # to download checkpoints that are already sitting in these shared caches,
    # which fails outright on a compute node with no outbound internet access.
    env.setdefault("HF_HOME", str(CHECKPOINT_ROOT / "huggingface"))
    env.setdefault("TORCH_HOME", str(CHECKPOINT_ROOT / "torch"))
    env.setdefault("XDG_CACHE_HOME", str(CHECKPOINT_ROOT / "xdg"))
    if spec.ckpt_env:
        env[spec.ckpt_env] = resolve_path_template(spec.ckpt_default)
    if spec.code_env:
        env[spec.code_env] = resolve_path_template(spec.code_default)
    return env


def inspect_outputs(out_dir: pathlib.Path) -> dict:
    """Reads the shapes of the exported maps and the exported camera.

    Args:
        out_dir: The adapter's output folder.

    Returns:
        The depth and confidence map shapes (name, [H, W]) and the first
        COLMAP camera's width/height and intrinsics (None if absent).
    """
    result = {
        "depth_shapes": [],
        "confidence_shapes": [],
        "colmap_wh": None,
        "colmap_intrinsics": None
    }
    depths_dir = out_dir / "depths"
    if depths_dir.is_dir():
        for path in sorted(depths_dir.glob("*")):
            if path.suffix.lower() == ".npy":
                array = np.load(path)
                result["depth_shapes"].append([path.name, list(array.shape)])
            elif path.suffix.lower() in {".exr", ".png", ".tif", ".tiff"}:
                with Image.open(path) as image:
                    result["depth_shapes"].append(
                        [path.name, [image.height, image.width]])
    confidences_dir = out_dir / "confidences"
    if confidences_dir.is_dir():
        confidence_paths = sorted(path for path in confidences_dir.glob("*")
                                  if path.suffix in (".npz", ".npy"))
        for path in confidence_paths:
            if path.suffix == ".npz":
                with np.load(path) as data:
                    array = data[data.files[0]]
            else:
                array = np.load(path)
            result["confidence_shapes"].append([path.name, list(array.shape)])
    cameras_txt = out_dir / "sparse" / "0" / "cameras.txt"
    if cameras_txt.is_file():
        for line in cameras_txt.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
            width, height = int(parts[2]), int(parts[3])
            result["colmap_wh"] = [width, height]
            result["colmap_intrinsics"] = [float(value) for value in parts[4:]]
            break
    return result


def run_one(spec, track: str, device: str, output_root: pathlib.Path,
            timeout_s: int) -> dict:
    """Runs one method's adapter on one track and checks its outputs.

    Args:
        spec: The method's `MethodSpec`.
        track: The modality track.
        device: "cpu" or "cuda".
        output_root: The smoke test's output root (holds the scene).
        timeout_s: The adapter's timeout in seconds.

    Returns:
        The run record: the command, status, timing, log path and, for a
        successful run, the outputs and whether their resolution is
        consistent.
    """
    scene_dir = output_root / "scene"
    out_dir = output_root / "runs" / f"{spec.name}_{track}"
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        import shutil  # pylint: disable=import-outside-toplevel

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(spec, track, scene_dir, out_dir)
    env = build_env(spec)
    if device == "cuda":
        env.pop("CUDA_VISIBLE_DEVICES", None)

    record = {
        "method": spec.name,
        "track": track,
        "cmd": cmd,
        "started_at": time.time(),
    }
    log_path = log_dir / f"{spec.name}_{track}.log"
    try:
        result = subprocess.run(
            cmd,
            cwd=INFERENCE_DIR,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            # A failing adapter is a reported result, not an error here.
            check=False,
        )
        log_path.write_text(result.stdout, encoding="utf-8")
        record["returncode"] = result.returncode
        if result.returncode == 0:
            record["status"] = "ok"
        else:
            record["status"] = "failed"
        if result.returncode != 0:
            record["log_tail"] = "\n".join(result.stdout.splitlines()[-40:])
    except subprocess.TimeoutExpired as exc:
        output = exc.output
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log_path.write_text(output or "", encoding="utf-8")
        record["status"] = "timeout"
        record["timeout_s"] = timeout_s
    record["elapsed_s"] = time.time() - record["started_at"]
    record["log_path"] = str(log_path)

    if record["status"] == "ok":
        record["outputs"] = inspect_outputs(out_dir)
        depth_ok = all(shape == [SOURCE_H, SOURCE_W]
                       for _name, shape in record["outputs"]["depth_shapes"])
        confidence_ok = all(
            shape == [SOURCE_H, SOURCE_W]
            for _name, shape in record["outputs"]["confidence_shapes"])
        colmap_ok = record["outputs"]["colmap_wh"] in (None,
                                                       [SOURCE_W, SOURCE_H])
        record["resolution_consistent"] = bool(
            record["outputs"]["depth_shapes"] and depth_ok and confidence_ok and
            colmap_ok)
        if not record["outputs"]["depth_shapes"]:
            record["resolution_consistent"] = False
            record["note"] = "no depth files exported"
    else:
        record["resolution_consistent"] = None
    return record


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated method names; default: all eligible")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout_s",
                        type=int,
                        default=0,
                        help="Override timeout for every method")
    parser.add_argument("--output_root",
                        type=pathlib.Path,
                        default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--keep_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Runs every eligible method once per track and prints the records."""
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scene_dir = output_root / "scene"
    build_synthetic_scene(scene_dir)

    if args.methods:
        wanted = set(args.methods.split(","))
    else:
        wanted = None
    results = []
    summary_path = output_root / "summary.json"

    seen_pairs = set()
    for spec in methods.METHOD_SPECS:
        if wanted is not None and spec.name not in wanted:
            continue
        if spec.name in EXCLUDED_METHODS:
            results.append({
                "method": spec.name,
                "track": None,
                "status": "excluded",
                "reason": EXCLUDED_METHODS[spec.name]
            })
            continue
        for track in spec.tracks:
            key = (spec.adapter, track)
            # Many *_DEPTH/_POSE GENCEPTION-style specs and MonoDepth/VideoDepth
            # duplicates share one adapter+track; the adapter code path is
            # identical regardless of which task/spec invoked it, so run it
            # once.
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            timeout_s = args.timeout_s or SLOW_METHOD_TIMEOUT_S.get(
                spec.name, DEFAULT_TIMEOUT_S)
            print(
                f"[{time.strftime('%H:%M:%S')}] running {spec.name} "
                f"track={track} (timeout={timeout_s}s)",
                flush=True)
            record = run_one(spec, track, args.device, output_root, timeout_s)
            print(
                f"  -> {record['status']} in {record['elapsed_s']:.1f}s;"
                f" consistent={record.get('resolution_consistent')}",
                flush=True,
            )
            results.append(record)
            summary_path.write_text(json.dumps(results, indent=2, default=str),
                                    encoding="utf-8")

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
