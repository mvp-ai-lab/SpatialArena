# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Streaming / long-context reconstruction methods on a SLAM sequence.

Each method runs its own released code, unmodified, with the settings of its
own evaluation (or, where none is released, its documented default), in the
si_bench env: LoGeR, ZipMap, LingBot-Map and ABot-Recon in-process, R3 and
Scal3R as a subprocess (see `DESCRIPTION` for the settings). Poses are
exported as in `slam.SlamModel`. Runtime and peak GPU memory use the
standard adapter metrics for in-process methods, and the nvidia-smi sampler
of the child process tree for the subprocess ones.

scripts/inference/streaming_slam.py runs them (--model picks the method).
--ckpt_path is the method's checkpoint (its MethodSpec's <X>_CKPT_PATH);
LoGeR's original_config.yaml is read next to it, and the SALAD / DINOv2
weights of the loop closures under the shared checkpoint root
(`slam.checkpoint_root`).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml
from PIL import Image

from spatialarena.models import registry
from spatialarena.models import slam
from spatialarena.models import slam_depth
from spatialarena.models import utils

# pylint: disable=line-too-long
DESCRIPTION = """Run a streaming / long-context reconstruction method on a SLAM sequence and export its camera poses.

Each method runs its own released code, unmodified, with the settings of its own evaluation (or, where none is
released, its documented default), in the ``si_bench`` env (``--model`` picks it; ``--ckpt_path`` is its checkpoint,
by default ``$<X>_CKPT_PATH`` or the one under the shared checkpoint root ``$SHARED_CHECKPOINT_ROOT``, which also holds
the SALAD / DINOv2 loop-closure weights):

- ``LOGER`` (github.com/Junyi42/LoGeR, a reimplementation; no license): checkpoint ``LoGeR``, the loader and
  forward pass of its TUM relative-pose evaluation (``eval/relpose``: 512 px, crop, window 64 / overlap 3).
- ``ZIPMAP`` (github.com/Haian-Jin/ZipMap): main offline checkpoint ``checkpoint_aff_inv.pt`` (affine invariant),
  the 518 px VGGT-style loader; one bidirectional pass over the whole sequence.
- ``LINGBOT_MAP`` (github.com/robbyant/lingbot-map): ``lingbot-map.pt`` in streaming mode with the settings of its
  own benchmark (``benchmark/configs/methods/lingbot_map.yaml``: area budget 255,000 px aligned to 14, 8 scale
  frames, KV window 64, keyframe interval auto above 320 frames), SDPA attention.
- ``R3`` (github.com/KevinXu02/R3): ``demo.py --mode local`` (the indoor preset) with the paper checkpoint
  ``r3.safetensors``, run as a subprocess.
- ``SCAL3R`` (github.com/zju3dv/Scal3R): ``python -m scal3r.run`` with its release config (block 60, overlap 30,
  loop closure with DINO-SALAD), run as a subprocess; it runs on one GPU (no process group -> world size 1).
- ``ABOT_RECON`` (github.com/amap-cvlab/ABot-Recon): the Python API at its release settings (504x280, 12-frame
  local window) with its loop-closure backend on, as by default (DINO-SALAD retrieval, ABot-Recon relative-pose
  edges, sparse pose-graph optimization); SDPA attention. The poses are the loop-refined trajectory.

The selected frames (``--interval`` = SLAM stride) are linked into a staging folder as ``000000.png``... so every
method sees the same frames in the same order and its per-index output maps back to the original names. Poses are
written as ``sparse/0`` with nominal intrinsics (focal = max(width, height), principal point at the center): the
SLAM protocol scores camera poses only. With ``--export_depth`` each returned frame's depth and the method's
confidence are written too (``depths/``, ``confidences/``; slam_depth_export.py). Runtime and peak GPU memory use
the standard adapter metrics for in-process methods, and the nvidia-smi sampler of the child process tree for the
subprocess ones (R3, Scal3R).
"""
# pylint: enable=line-too-long

# Loop-closure summary of the last in-process run that has one (ABot-Recon),
# reported in metrics.json.
LOOP_INFO: dict = {}
METHODS = ("LOGER", "ZIPMAP", "LINGBOT_MAP", "R3", "SCAL3R", "ABOT_RECON")
CODE_DIRS = {
    "LOGER": "LoGeR",
    "ZIPMAP": "ZipMap",
    "LINGBOT_MAP": "lingbot-map",
    "R3": "R3",
    "SCAL3R": "Scal3R",
    "ABOT_RECON": "ABot-Recon",
}

# LoGeR eval/eval.md, TUM-dynamics relative pose: --window-size 64
# --overlap-size 3 (input size 512, crop).
LOGER_SETTINGS = {"size": 512, "window_size": 64, "overlap_size": 3}
# ZipMap demo_gradio_zipmap.py model_config (the main, affine-invariant
# checkpoint).
ZIPMAP_MODEL_CONFIG = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": True,
    "enable_local_point": True,
    "enable_depth": True,
    "ttt_config": {
        "ttt_mode": True,
        "params": {
            "bias": True,
            "head_dim": 1024,
            "inter_multi": 2,
            "base_lr": 0.01,
            "muon_update_steps": 5,
            "use_gate_fn": True
        },
    },
    "other_config": {
        "use_gradient_checkpointing_local_point": False,
        "use_gradient_checkpointing_depth": False,
        "affine_invariant": True,
    },
}
# lingbot-map benchmark/configs/methods/lingbot_map.yaml (streaming mode).
LINGBOT_SETTINGS = {
    "image_size": 518,
    "patch_size": 14,
    "enable_3d_rope": True,
    "num_scale_frames": 8,
    "max_frame_num": 1024,
    "kv_cache_sliding_window": 64,
    "kv_cache_scale_frames": 8,
    "auto_keyframe_threshold": 320,
    "area_budget": 255_000,
    "align": 14,
}


def weight_paths(ckpt_root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Returns the weight files of each method under `ckpt_root`.

    That is where spatialarena/download.py puts them.

    Args:
        ckpt_root: The shared checkpoint root.

    Returns:
        Weight file by method (and SALAD / DINO, shared with VGGT-Long).
    """
    local = ckpt_root / "local"
    return {
        "LOGER": local / "loger" / "LoGeR" / "latest.pt",
        "LOGER_CONFIG": local / "loger" / "LoGeR" / "original_config.yaml",
        "ZIPMAP": local / "zipmap" / "checkpoint_aff_inv.pt",
        "LINGBOT_MAP": local / "lingbot_map" / "lingbot-map.pt",
        "R3": local / "r3" / "r3.safetensors",
        "SCAL3R": local / "scal3r" / "scal3r.pt",
        # Shared with VGGT-Long / Pi-Long.
        "SALAD": local / "slam_long" / "dino_salad.ckpt",
        # ABot-Recon's SALAD backbone, also shared.
        "DINO": local / "slam_long" / "dinov2_vitb14_pretrain.pth",
        "ABOT_RECON": local / "abot_recon" / "abot_recon.safetensors",
    }


def load_weights(model, state: dict, method: str) -> None:
    """Loads `state` into `model`, failing on any missing or unexpected key.

    Loading with strict=False and ignoring the result would run a mismatched
    checkpoint with random weights and score its output.

    Args:
        model: The torch module.
        state: The state dict.
        method: Method name for the error message.

    Raises:
        RuntimeError: The checkpoint does not match the model.
    """
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{method} checkpoint does not match the model: "
                           f"{len(result.missing_keys)} missing keys "
                           f"{list(result.missing_keys)[:5]}, "
                           f"{len(result.unexpected_keys)} unexpected keys "
                           f"{list(result.unexpected_keys)[:5]}")


def _cuda_dtype():
    import torch  # pylint: disable=import-outside-toplevel

    return torch.bfloat16 if torch.cuda.get_device_capability(
    )[0] >= 8 else torch.float16


def _poses_by_index(c2w) -> dict[int, np.ndarray]:
    array = np.asarray(c2w, dtype=np.float64)
    if array.shape[-2:] == (3, 4):
        bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]),
                                 (*array.shape[:-2], 1, 4))
        array = np.concatenate([array, bottom], axis=-2)
    return {index: pose for index, pose in enumerate(array)}


# The runners import each method's code inside, as its checkout joins
# sys.path only then; by name, as the method's own code does (a package may
# export a class under its submodule's name, e.g. zipmap.models.ZipMap).
# pylint: disable=import-outside-toplevel,line-too-long


def run_loger(
        staged: list[pathlib.Path],
        code_path: pathlib.Path,
        weights: dict[str, pathlib.Path],
        fixed: tuple[int, int] | None = None,
        sink: slam_depth.DepthSink | None = None) -> dict[int, np.ndarray]:
    """Runs LoGeR through its eval/pi3_adapter.

    Window 64, overlap 3; 512 px, or exactly `fixed` = (width, height).

    Args:
        staged: The staged frames.
        code_path: The LoGeR checkout.
        weights: `weight_paths`.
        fixed: The fixed input size, or None.
        sink: Depth export, or None.

    Returns:
        {input index: c2w}.
    """
    sys.path.insert(0, str(code_path))
    import torch
    from eval.pi3_adapter import _coerce_view_image, load_pi3_model, merge_forward_kwargs
    from eval.relpose.launch import _load_images_basic

    model, forward_kwargs = load_pi3_model(str(weights["LOGER"]),
                                           config_path=str(
                                               weights["LOGER_CONFIG"]),
                                           device=torch.device("cuda"))
    forward_kwargs = merge_forward_kwargs(
        forward_kwargs,
        {
            "window_size": LOGER_SETTINGS["window_size"],
            "overlap_size": LOGER_SETTINGS["overlap_size"]
        },
    )
    utils.mark_model_loaded()
    # A (width, height) size equal to the frames' own is a no-op resize in
    # _load_images_basic.
    images = _load_images_basic([str(path) for path in staged],
                                size=fixed or LOGER_SETTINGS["size"],
                                crop=True)
    # eval/pi3_adapter.run_pi3_inference_on_views with only the poses kept:
    # it also re-expresses every frame's point map in float32 copies on the
    # host (~6x the 28 GB of point maps at 18,846 frames of 504x252, beyond
    # the protocol's 180 GB). Same input tensor, model call and autocast;
    # poses relative to the first frame as there.
    batch = torch.stack([_coerce_view_image(image["img"]) for image in images
                        ]).unsqueeze(0)
    del images
    utils.record_model_input(batch.shape[-1], batch.shape[-2])
    with torch.no_grad(), torch.autocast(device_type="cuda",
                                         dtype=_cuda_dtype()):
        output = model(batch, **forward_kwargs)
    camera_poses = output["camera_poses"].squeeze(0).detach().float().cpu()
    if sink is not None:
        # Depth = z of the local point map; confidence = sigmoid of its logit
        # (Pi3 convention), streamed from GPU.
        confidence = output.get("conf")
        sink.write_tensors(
            output["local_points"][0, ..., 2],
            torch.sigmoid(confidence[0, ..., 0].float())
            if confidence is not None else None)
    del output
    camera_poses = torch.linalg.inv(camera_poses[0]).unsqueeze(0) @ camera_poses
    return _poses_by_index(camera_poses.numpy())  # camera-to-world


def run_zipmap(
        staged: list[pathlib.Path],
        code_path: pathlib.Path,
        weights: dict[str, pathlib.Path],
        fixed: tuple[int, int] | None = None,
        sink: slam_depth.DepthSink | None = None) -> dict[int, np.ndarray]:
    """Runs ZipMap in one forward pass over all frames.

    518 px wide, or `fixed`'s width.

    Args:
        staged: The staged frames.
        code_path: The ZipMap checkout.
        weights: `weight_paths`.
        fixed: The fixed input size, or None.
        sink: Depth export, or None.

    Returns:
        {input index: c2w}.
    """
    sys.path[:0] = [str(code_path), str(code_path / "zipmap")]
    import torch
    from zipmap.models.ZipMap import ZipMap
    from zipmap.utils.load_fn import load_and_preprocess_images
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri

    model = ZipMap(**ZIPMAP_MODEL_CONFIG)
    checkpoint = torch.load(weights["ZIPMAP"],
                            map_location="cpu",
                            weights_only=True)
    # All 1,499 tensors match (2026-10-21).
    load_weights(model, checkpoint.get("model", checkpoint), "ZipMap")
    model = model.eval().to("cuda")
    utils.mark_model_loaded()
    # "crop" mode sets the width to target_size and rounds the height to
    # 14 px: the frames' own size when fixed.
    images = load_and_preprocess_images(
        [str(path) for path in staged],
        target_size=fixed[0] if fixed else 518).to("cuda")
    utils.record_model_input(images.shape[-1], images.shape[-2])
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=_cuda_dtype()):
        predictions = model(images)
    extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"],
                                                images.shape[-2:])
    if sink is not None:
        confidence = predictions.get("depth_conf")
        sink.write_tensors(predictions["depth"][0, ..., 0],
                           confidence[0] if confidence is not None else None)
    # VGGT convention: world-to-camera.
    w2c = _poses_by_index(extrinsic[0].float().cpu().numpy())
    return {index: np.linalg.inv(pose) for index, pose in w2c.items()}


def _lingbot_resize(image: np.ndarray) -> np.ndarray:
    """lingbot-map benchmark/geometry/resize.py, mode 'area_budget'.

    Uniform downscale to the area budget, both sides floored to the
    alignment.
    """
    import cv2

    height, width = image.shape[:2]
    align = LINGBOT_SETTINGS["align"]
    scale = min(np.sqrt(LINGBOT_SETTINGS["area_budget"] / (width * height)),
                1.0)
    new_w = max((int(width * scale) // align) * align, align)
    new_h = max((int(height * scale) // align) * align, align)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def run_lingbot_map(
        staged: list[pathlib.Path],
        code_path: pathlib.Path,
        weights: dict[str, pathlib.Path],
        fixed: tuple[int, int] | None = None,
        sink: slam_depth.DepthSink | None = None) -> dict[int, np.ndarray]:
    """Runs LingBot-Map streaming inference with its benchmark settings.

    Its area-budget resize is skipped for `fixed` inputs.

    Args:
        staged: The staged frames.
        code_path: The lingbot-map checkout.
        weights: `weight_paths`.
        fixed: The fixed input size, or None.
        sink: Depth export, or None.

    Returns:
        {input index: c2w}.
    """
    sys.path.insert(0, str(code_path))
    import torch
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    from torchvision import transforms

    settings = LINGBOT_SETTINGS
    model = GCTStream(
        img_size=settings["image_size"],
        patch_size=settings["patch_size"],
        enable_3d_rope=settings["enable_3d_rope"],
        max_frame_num=settings["max_frame_num"],
        kv_cache_sliding_window=settings["kv_cache_sliding_window"],
        kv_cache_scale_frames=settings["kv_cache_scale_frames"],
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
    )
    checkpoint = torch.load(weights["LINGBOT_MAP"],
                            map_location="cpu",
                            weights_only=False)
    # All 1,342 tensors match (2026-10-21).
    load_weights(model, checkpoint.get("model", checkpoint), "LingBot-Map")
    model = model.to("cuda").eval()
    utils.mark_model_loaded()
    to_tensor = transforms.ToTensor()
    resize = (lambda image: image) if fixed else _lingbot_resize
    images = torch.stack([
        to_tensor(resize(np.asarray(Image.open(path).convert("RGB"))))
        for path in staged
    ]).to("cuda")
    utils.record_model_input(images.shape[-1], images.shape[-2])
    # benchmark/methods/lingbot_map.py::_resolve_keyframe_interval ("auto").
    threshold = settings["auto_keyframe_threshold"]
    keyframe_interval = 1 if len(staged) <= threshold else -(-len(staged) //
                                                             threshold)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=_cuda_dtype()):
        predictions = model.inference_streaming(
            images,
            num_scale_frames=settings["num_scale_frames"],
            keyframe_interval=keyframe_interval,
            output_device=torch.device("cpu"),
        )
    extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"],
                                                images.shape[-2:])
    if sink is not None:
        sink.write_tensors(
            predictions["depth"][0, ..., 0], predictions["depth_conf"][0]
            if predictions.get("depth_conf") is not None else None)
    # lingbot-map's pose encoding decodes to c2w.
    return _poses_by_index(extrinsic[0].float().cpu().numpy())


def run_abot_recon(
        staged: list[pathlib.Path],
        code_path: pathlib.Path,
        weights: dict[str, pathlib.Path],
        fixed: tuple[int, int] | None = None,
        sink: slam_depth.DepthSink | None = None) -> dict[int, np.ndarray]:
    """Runs ABot-Recon (SDPA attention, loop closure on).

    Its 504x280 canvas, or `fixed`. The loop backend's summary
    (loop_info.json) goes into `LOOP_INFO` for metrics.json.

    Args:
        staged: The staged frames.
        code_path: The ABot-Recon checkout.
        weights: `weight_paths`.
        fixed: The fixed input size, or None.
        sink: Depth export, or None.

    Returns:
        {input index: c2w}, the loop-refined trajectory.
    """
    sys.path.insert(0, str(code_path))
    from abot_recon import ABotRecon

    # Its preprocessing locks the width to `width` and crops/pads to
    # `height`: a no-op at the frames' own size.
    size = {"width": fixed[0], "height": fixed[1]} if fixed else {}
    # In the case's work directory (staging is <work>/images).
    loop_dir = staged[0].parent.parent / "abot_loop"
    model = ABotRecon.from_pretrained(str(weights["ABOT_RECON"]),
                                      device="cuda",
                                      attention_backend="sdpa",
                                      loop_closure=True,
                                      loop_salad_checkpoint=weights["SALAD"],
                                      loop_dino_checkpoint=weights["DINO"],
                                      loop_output_dir=loop_dir,
                                      **size)
    utils.record_model_input(model.config.width, model.config.height)
    utils.mark_model_loaded()
    # Local points / confidence only when depth is exported; threshold 0
    # keeps every pixel (no NaN masking).
    with_maps = sink is not None
    result = model.infer(staged,
                         output_local_points=with_maps,
                         output_world_points=False,
                         output_confidence=with_maps,
                         confidence_threshold=0.0,
                         loop_closure=True)
    info = loop_dir / "loop_info.json"
    if info.is_file():
        LOOP_INFO.update(json.loads(info.read_text(encoding="utf-8")))
        print(
            f"ABot-Recon loop closure: {LOOP_INFO.get('num_loop_edges')} "
            f"loop edges from {LOOP_INFO.get('num_candidates')} candidates",
            flush=True)
    if with_maps:
        sink.write_tensors(result.local_points[..., 2], result.confidence)
    # Camera-to-world.
    return _poses_by_index(result.camera_poses.float().cpu().numpy())


# pylint: enable=import-outside-toplevel,line-too-long


def _run_subprocess(command: list[str], cwd: pathlib.Path,
                    env: dict[str, str]) -> dict:
    print("Running:", " ".join(command), flush=True)
    started_at = time.time()
    with subprocess.Popen(command, cwd=cwd, env=env) as process:
        sampler = slam.GpuMemorySampler(process.pid)
        sampler.start()
        status = process.wait()
        peak = sampler.stop()
    if status != 0:
        raise SystemExit(f"{command[1]} exited with status {status}")
    # The method runs in a child process: wall time includes its model
    # loading, and peak memory is the child's process-tree peak from
    # nvidia-smi (sampled at 1 s), not torch.cuda.max_memory_allocated.
    return {
        "inference_seconds": time.time() - started_at,
        "model_load_seconds": None,
        "peak_gpu_memory_mib": peak,
        "runtime_scope": "subprocess_wall"
    }


def run_r3(staging: pathlib.Path,
           work: pathlib.Path,
           code_path: pathlib.Path,
           weights: dict[str, pathlib.Path],
           fixed: tuple[int, int] | None = None,
           sink: slam_depth.DepthSink | None = None) -> tuple[dict, dict]:
    """Runs R3's demo.py (local mode, no viewer) in a subprocess.

    --size is the long edge it resizes to before cropping to 14 px: its
    default 504, or `fixed`'s long edge (a no-op).

    Args:
        staging: The staging folder.
        work: The case's work directory.
        code_path: The R3 checkout.
        weights: `weight_paths`.
        fixed: The fixed input size, or None.
        sink: Depth export, or None.

    Returns:
        ({input index: c2w}, runtime metrics).
    """
    output = work / "r3"
    env = dict(os.environ,
               PYTHONPATH=os.pathsep.join(
                   [str(code_path),
                    os.environ.get("PYTHONPATH", "")]))
    runtime = _run_subprocess(
        [
            sys.executable,
            str(code_path / "demo.py"), "--mode", "local", "--no_viewer",
            "--seq_path",
            str(staging), "--output_dir",
            str(output), "--ckpt",
            str(weights["R3"]), "--size",
            str(max(fixed) if fixed else 504)
        ],
        code_path,
        env,
    )
    # Predicted at the model's input size.
    depth = np.load(output / "depth" / "000000.npy", mmap_mode="r")
    runtime["model_input_size"] = [int(depth.shape[1]), int(depth.shape[0])]
    if sink is not None:
        # infer.py always writes depth/ and conf/ for the k-th returned frame.
        frame_ids = json.loads(
            (output /
             "run_params.json").read_text(encoding="utf-8"))["output_frame_ids"]
        slam_depth.export_numbered_maps(
            sink, output / "depth", [int(frame_id) for frame_id in frame_ids],
            output / "conf")
    return read_r3_poses(output), runtime


def read_r3_poses(output: pathlib.Path) -> dict[int, np.ndarray]:
    """Reads the poses of R3's infer.py output.

    camera/<k>.npz (`pose` = 3x4 or 4x4 camera-to-world) for the k-th
    returned frame, whose input index is run_params.json
    output_frame_ids[k].

    Args:
        output: R3's output directory.

    Returns:
        {input index: c2w}.
    """
    frame_ids = json.loads(
        (output /
         "run_params.json").read_text(encoding="utf-8"))["output_frame_ids"]
    poses = {}
    for index, frame_id in enumerate(frame_ids):
        pose = np.eye(4)
        pose[:3] = np.load(output / "camera" / f"{index:06d}.npz")["pose"][:3]
        poses[int(frame_id)] = pose
    return poses


def run_scal3r(staging: pathlib.Path,
               work: pathlib.Path,
               code_path: pathlib.Path,
               weights: dict[str, pathlib.Path],
               fixed: tuple[int, int] | None = None,
               sink: slam_depth.DepthSink | None = None) -> tuple[dict, dict]:
    """Runs `python -m scal3r.run` with the release config in a subprocess.

    Its frames are resized to proc_max_size px wide (518), or kept at
    `fixed`'s width.

    Args:
        staging: The staging folder.
        work: The case's work directory.
        code_path: The Scal3R checkout.
        weights: `weight_paths`.
        fixed: The fixed input size, or None.
        sink: Depth export, or None.

    Returns:
        ({input index: c2w}, runtime metrics).
    """
    output = work / "scal3r"
    # The release config, with absolute weight paths (its own are relative
    # to the checkout).
    config = work / "scal3r.yaml"
    config.write_text(yaml.safe_dump({
        "configs": [str(code_path / "configs" / "models" / "scal3r.yaml")],
        "model": {
            "checkpoint": str(weights["SCAL3R"])
        },
        "data": {
            "loop_ckpt": str(weights["SALAD"])
        },
        **({
            "val_dataloader_cfg": {
                "dataset_cfg": {
                    "proc_max_size": fixed[0]
                }
            }
        } if fixed else {}),
    }),
                      encoding="utf-8")
    env = dict(os.environ,
               PYTHONPATH=os.pathsep.join(
                   [str(code_path),
                    os.environ.get("PYTHONPATH", "")]))
    runtime = _run_subprocess(
        [
            sys.executable, "-m", "scal3r.run", "--config",
            str(config), "--input_dir",
            str(staging), "--output_dir",
            str(output), "--checkpoint",
            str(weights["SCAL3R"]), "--save_dpt",
            "1" if sink is not None else "0", "--save_xyz", "0"
        ],
        code_path,
        env,
    )
    # Without --export_depth Scal3R saves no per-frame maps (--save_dpt 0).
    # At proc_max_size = the fixed width its resize is a no-op
    # (scripts/test/probe_slam_input_sizes.py), so the model sees the fixed
    # frames as they are.
    runtime["model_input_size"] = list(fixed) if fixed else None
    if sink is not None:
        # depths/<k>.exr, already rescaled to its trajectory; Scal3R saves no
        # confidence.
        for path in sorted((output / "depths").glob("*.exr")):
            sink.write([int(path.stem)], [slam_depth.read_exr_depth(path)])
    return read_scal3r_poses(output), runtime


def read_scal3r_poses(output: pathlib.Path) -> dict[int, np.ndarray]:
    """Reads Scal3R's mat.txt.

    One raveled 4x4 camera-to-world per input image, in the (sorted) input
    order.

    Args:
        output: Scal3R's output directory.

    Returns:
        {input index: c2w}.

    Raises:
        RuntimeError: mat.txt does not have 16 columns.
    """
    rows = np.loadtxt(output / "mat.txt", ndmin=2)
    if rows.shape[1] != 16:
        raise RuntimeError(
            f"{output / 'mat.txt'} has {rows.shape[1]} columns, expected 16")
    return {index: row.reshape(4, 4) for index, row in enumerate(rows)}


IN_PROCESS = {
    "LOGER": run_loger,
    "ZIPMAP": run_zipmap,
    "LINGBOT_MAP": run_lingbot_map,
    "ABOT_RECON": run_abot_recon
}
SUBPROCESS = {"R3": run_r3, "SCAL3R": run_scal3r}


class StreamingSlam(slam.SlamModel):
    """A streaming SLAM method (streaming_slam.py).

    --ckpt_path is the method's own entry of `weight_paths`.
    """

    description = DESCRIPTION

    @classmethod
    def weight_paths(cls, ckpt_root):
        return weight_paths(ckpt_root)

    def run_method(self, staging, staged, work, weights, fixed, sink):
        """Runs the method on the staged frames.

        Args:
            staging: The staging folder.
            staged: The staged frames.
            work: The case's work directory.
            weights: `weight_paths`.
            fixed: The fixed input size, or None.
            sink: Depth export, or None.

        Returns:
            ({input index: c2w}, runtime metrics).
        """
        started_at = time.time()
        if self.name in IN_PROCESS:
            poses = IN_PROCESS[self.name](staged, self.code_path, weights,
                                          fixed, sink)
            runtime = utils.runtime_metrics(started_at)
            if sink is not None:
                # The maps were written while the method ran; time inference
                # without that export.
                runtime["inference_seconds"] -= sink.export_seconds
            return poses, runtime, started_at
        poses, runtime = SUBPROCESS[self.name](staging, work, self.code_path,
                                               weights, fixed, sink)
        return poses, runtime, started_at

    def run_case(self, weights):
        args = self.args
        image_paths = self.image_paths
        args.output_path.mkdir(parents=True, exist_ok=True)
        work = utils.slam_work_dir(self.method_name, args.output_path)
        try:
            staging = work / "images"
            staged = [
                staging / name
                for name in slam.stage_images(image_paths, staging)
            ]
            fixed = utils.fixed_input_size(image_paths)
            sink = slam_depth.DepthSink(
                args.output_path, image_paths) if args.export_depth else None
            poses, runtime, started_at = self.run_method(
                staging, staged, work, weights, fixed, sink)
            returned = self.export_poses(poses, self.pose_indices(poses))
            utils.write_adapter_metrics(
                output_path=args.output_path,
                method=self.method_label,
                image_paths=image_paths,
                started_at=started_at,
                checkpoint=str(weights.get(self.name, self.code_path)),
                depth_only=False,
                metrics_extra={
                    "track":
                        args.track,
                    "frames_with_pose":
                        len(returned),
                    "intrinsics":
                        "nominal",
                    **(sink.metrics() | {
                           "depth_frame": "trajectory"
                       } if sink is not None else {}),
                    **({
                        "loop_closure": dict(LOOP_INFO)
                    } if LOOP_INFO else {})
                },
                runtime=runtime,
            )
            if not args.keep_intermediate:
                shutil.rmtree(work, ignore_errors=True)
        except BaseException:
            # A failed case leaves no work directory (staged frames, method
            # outputs) in the prediction cache.
            if not args.keep_intermediate:
                shutil.rmtree(work, ignore_errors=True)
            raise


@registry.register_model
class LoGeR(StreamingSlam):
    """LoGeR (in-process, eval/relpose settings)."""

    name = "LOGER"
    code_dir = CODE_DIRS["LOGER"]
    ckpt_key = "LOGER"

    def weights(self):
        # Its config ships next to the checkpoint.
        weights = super().weights()
        weights["LOGER_CONFIG"] = (weights["LOGER"].parent /
                                   "original_config.yaml")
        return weights


@registry.register_model
class ZipMap(StreamingSlam):
    """ZipMap (in-process, one bidirectional pass)."""

    name = "ZIPMAP"
    code_dir = CODE_DIRS["ZIPMAP"]
    ckpt_key = "ZIPMAP"


@registry.register_model
class LingBotMap(StreamingSlam):
    """LingBot-Map (in-process, streaming mode)."""

    name = "LINGBOT_MAP"
    code_dir = CODE_DIRS["LINGBOT_MAP"]
    ckpt_key = "LINGBOT_MAP"


@registry.register_model
class R3(StreamingSlam):
    """R3 (demo.py --mode local, subprocess)."""

    name = "R3"
    code_dir = CODE_DIRS["R3"]
    ckpt_key = "R3"


@registry.register_model
class Scal3R(StreamingSlam):
    """Scal3R (python -m scal3r.run, subprocess)."""

    name = "SCAL3R"
    code_dir = CODE_DIRS["SCAL3R"]
    ckpt_key = "SCAL3R"


@registry.register_model
class ABotRecon(StreamingSlam):
    """ABot-Recon (in-process, loop closure on)."""

    name = "ABOT_RECON"
    code_dir = CODE_DIRS["ABOT_RECON"]
    ckpt_key = "ABOT_RECON"
