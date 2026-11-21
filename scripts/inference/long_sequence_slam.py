# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run a long-sequence SLAM system (VGGT-Long, Pi-Long, VGGT-SLAM) and export benchmark camera poses.

These systems process an arbitrarily long recording by chunking it internally (overlapping chunks or submaps,
aligned and loop-closed), so unlike the feed-forward methods they have no single-pass frame budget. Each is run
unmodified through its own entry point, with its published default settings, in the default ``si_bench`` env:

- ``VGGT_LONG``: ``vggt_long.py`` (github.com/DengKaiCQ/VGGT-Long), VGGT-1B backbone, SALAD loop closure.
- ``PI_LONG``: ``pi_long.py`` (github.com/DengKaiCQ/Pi-Long), Pi3 backbone, SALAD loop closure.
- ``VGGT_SLAM``: ``main.py`` (github.com/MIT-SPARK/VGGT-SLAM) with the arguments of its own TUM evaluation script
  (``evals/eval_tum.sh``: submap 16, one loop, min disparity 50, conf 25 %, loop threshold 0.95). VGGT-SLAM keeps
  only keyframes, so it returns poses for a subset of the input frames; the SLAM evaluator reports that as
  completeness and scores the returned frames, as VGGT-SLAM's own evaluation does.

The selected frames (``--interval`` = SLAM stride) are linked into a staging folder as ``000000.png``... so every
system sees the same frames in the same order and its per-index output maps back to the original names. Poses are
written as ``sparse/0`` with nominal intrinsics (focal = max(width, height), principal point at the center): these
systems' intrinsics are not part of the SLAM protocol, which scores camera poses only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.inference.adapter_utils import (  # noqa: E402
    canonical_image_root,
    export_camera_pose_priors,
    find_image_paths,
    fixed_input_size,
    slam_work_dir,
    select_interval,
    write_adapter_metrics,
)
from scripts.inference.slam_depth_export import (  # noqa: E402
    DepthSink,
    chunk_ranges,
    export_long_chunks,
    export_vggt_slam_dumps,
)

METHODS = ("VGGT_LONG", "PI_LONG", "VGGT_SLAM")
# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
DEFAULT_CODE_ROOT = Path(f"{SPATIALARENA_ROOT}/benchmarks/_model_codes")
DEFAULT_CKPT_ROOT = Path(f"{SPATIALARENA_ROOT}/benchmarks/_checkpoints")
CODE_DIRS = {"VGGT_LONG": "VGGT-Long", "PI_LONG": "Pi-Long", "VGGT_SLAM": "VGGT-SLAM"}
# VGGT-SLAM's evals/eval_tum.sh settings.
VGGT_SLAM_ARGS = ["--max_loops", "1", "--min_disparity", "50", "--conf_threshold", "25", "--lc_thres", "0.95",
                  "--submap_size", "16"]
# The module each VGGT-based system imports load_and_preprocess_images from (it hardcodes a 518 px width); for
# fixed-resolution inputs (full-slam) fixed_input_launcher.py swaps it for a loader that keeps the frames as they are.
VGGT_LOADER_MODULES = {"VGGT_LONG": "base_models.vggt.utils.load_fn", "VGGT_SLAM": "vggt.utils.load_fn"}
FIXED_INPUT_LAUNCHER = Path(__file__).with_name("fixed_input_launcher.py")
# VGGT-Long and Pi-Long can run as two processes (long_sequence_phase_launcher.py): --phase gpu does the chunk
# inference and leaves its work directory plus ALIGN_PENDING in --output_path; --phase align, run later on a CPU-only
# node, does the chunk alignment and exports the poses. A benchmark GPU job thus never idles through the alignment.
PHASE_LAUNCHER = Path(__file__).with_name("long_sequence_phase_launcher.py")
SPLIT_METHODS = ("VGGT_LONG", "PI_LONG")
ALIGN_PENDING = "slam_align_pending.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--code_path", type=Path, default=None, help="Method checkout (default: _model_codes/<repo>).")
    parser.add_argument("--ckpt_root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--track", default="rgb")
    parser.add_argument("--colmap_model_subdir", default="0")
    parser.add_argument("--keep_work_dir", action="store_true")
    parser.add_argument("--phase", choices=("all", "gpu", "align"), default="all",
                        help="VGGT-Long/Pi-Long: 'gpu' = chunk inference only, 'align' = the CPU alignment a 'gpu' "
                             "run left pending, 'all' = both in one process. VGGT-SLAM runs 'all' for 'gpu'.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--export_depth", action="store_true",
                        help="Also write each returned frame's depth and confidence (depths/, confidences/).")
    # Shared flags run_method_case_list.sh appends to every adapter; poses are always exported here.
    for flag in ("--export_colmap", "--depth_only"):
        parser.add_argument(flag, action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def weight_paths(ckpt_root: Path) -> dict[str, Path]:
    slam = ckpt_root / "local" / "slam_long"
    return {
        "VGGT": ckpt_root / "local" / "vggt" / "vggt.pt",  # facebook/VGGT-1B model.pt
        "Pi3": slam / "pi3.safetensors",  # yyfz233/Pi3 model.safetensors
        "SALAD": slam / "dino_salad.ckpt",
        "DINO": slam / "dinov2_vitb14_pretrain.pth",
    }


def write_long_config(method: str, code_path: Path, weights: dict[str, Path], destination: Path,
                      fixed: tuple[int, int] | None = None, keep_chunks: bool = False) -> Path:
    """The repo's own base_config.yaml with absolute weight paths (everything else left at the published values);
    for ``fixed`` inputs Pi-Long's ``image_width`` is their width, which its loader then keeps as is. ``keep_chunks``
    keeps the per-chunk predictions the system otherwise deletes at the end, for the depth export."""
    config = yaml.safe_load((code_path / "configs" / "base_config.yaml").read_text(encoding="utf-8"))
    config["Weights"]["SALAD"] = str(weights["SALAD"])
    config["Weights"]["DNIO"] = str(weights["DINO"])  # sic: the repos' key name
    if method == "VGGT_LONG":
        config["Weights"]["model"] = "VGGT"
        config["Weights"]["VGGT"] = str(weights["VGGT"])
    else:
        config["Model"]["model_type"] = "Pi3"
        config["Weights"]["Pi3"] = str(weights["Pi3"])
        if fixed:
            config["Model"]["image_width"] = fixed[0]
    if keep_chunks:
        config["Model"]["delete_temp_files"] = False
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def stage_images(image_paths: list[str], staging: Path) -> list[str]:
    staging.mkdir(parents=True, exist_ok=True)
    names = []
    for index, path in enumerate(image_paths):
        staged = staging / f"{index:06d}{Path(path).suffix.lower()}"
        staged.symlink_to(Path(path).resolve())
        names.append(staged.name)
    return names


def _descendants(pid: int) -> set[int]:
    pids, frontier = {pid}, [pid]
    while frontier:
        parent = frontier.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except OSError:
            continue
        for child in map(int, children):
            if child not in pids:
                pids.add(child)
                frontier.append(child)
    return pids


class GpuMemorySampler(threading.Thread):
    """Peak GPU memory (MiB) used by a child process tree, sampled from nvidia-smi once per second."""

    def __init__(self, pid: int):
        super().__init__(daemon=True)
        # Not ``self._stop``: that name is threading.Thread's own method, which join() calls.
        self.pid, self.peak, self._stop_event = pid, None, threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(1.0):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, check=True,
                ).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            tree = _descendants(self.pid)
            used = sum(int(mem) for pid, mem in (line.split(",") for line in out.splitlines() if "," in line)
                       if int(pid) in tree)
            if used:
                self.peak = max(self.peak or 0, used)

    def stop(self) -> float | None:
        self._stop_event.set()
        self.join(timeout=15)
        return float(self.peak) if self.peak is not None else None


def read_long_poses(exp_root: Path, count: int) -> dict[int, np.ndarray]:
    """camera_poses.txt of VGGT-Long/Pi-Long: one flattened 4x4 camera-to-world per staged image, in order."""
    files = sorted(exp_root.rglob("camera_poses.txt"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one camera_poses.txt under {exp_root}, found {len(files)}")
    rows = np.loadtxt(files[0], ndmin=2)
    if rows.shape != (count, 16):
        raise RuntimeError(f"{files[0]} has shape {rows.shape}, expected ({count}, 16)")
    return {index: row.reshape(4, 4) for index, row in enumerate(rows)}


def read_tum_poses(path: Path) -> dict[int, np.ndarray]:
    """VGGT-SLAM log: ``frame_id x y z qx qy qz qw`` (camera-to-world) per keyframe; frame_id = staged index."""
    from scipy.spatial.transform import Rotation

    poses = {}
    for row in np.loadtxt(path, ndmin=2):
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        c2w[:3, 3] = row[1:4]
        poses[int(round(row[0]))] = c2w
    return poses


def system_command(method: str, code_path: Path, work: Path, fixed: bool,
                   depth_dump: Path | None = None) -> tuple[list[str], dict[str, str]]:
    """The system's own command line on ``work``'s staged frames (and config.yaml), and its environment.
    ``depth_dump`` (VGGT-SLAM) runs it through fixed_input_launcher.py to save its keyframe depths there."""
    staging = work / "images"
    env = dict(os.environ)
    if method == "VGGT_SLAM":
        command = [sys.executable, str(code_path / "main.py"), "--image_folder", str(staging), *VGGT_SLAM_ARGS,
                   "--log_results", "--skip_dense_log", "--log_path", str(work / "poses.txt")]
        env["PYTHONPATH"] = os.pathsep.join(
            [str(code_path), str(code_path / "third_party" / "vggt"), str(code_path / "third_party" / "salad"),
             env.get("PYTHONPATH", "")]
        )
    else:
        script = "vggt_long.py" if method == "VGGT_LONG" else "pi_long.py"
        command = [sys.executable, str(code_path / script), "--image_dir", str(staging),
                   "--config", str(work / "config.yaml")]
        if method == "PI_LONG":
            command += ["--exp_folder_name", str(work / "exps")]
        env["PYTHONPATH"] = os.pathsep.join([str(code_path), env.get("PYTHONPATH", "")])
    launcher_args = []
    if fixed and method in VGGT_LOADER_MODULES:
        launcher_args += ["--patch", VGGT_LOADER_MODULES[method], "--record", str(work / "model_input_size.json")]
    if depth_dump is not None and method == "VGGT_SLAM":
        launcher_args += ["--dump_vggt_slam_depth", str(depth_dump)]
    if launcher_args:
        command = [command[0], str(FIXED_INPUT_LAUNCHER), *launcher_args, "--", *command[1:]]
    return command, env


def phase_command(command: list[str], phase: str, work: Path, export_depth: bool = False) -> list[str]:
    """``command`` (optionally through fixed_input_launcher.py) run through long_sequence_phase_launcher.py;
    ``export_depth`` keeps the chunk arrays the depth export reads (the GPU phase otherwise drops them)."""
    launcher_args = ["--phase", phase, "--state", str(work / "phase_state.pkl")]
    if export_depth:
        launcher_args.append("--export_depth")
    if command[1] == str(FIXED_INPUT_LAUNCHER):
        separator = command.index("--")
        launcher_args += command[2:separator]  # its --patch/--record, which the phase launcher applies itself
        command = [command[0], *command[separator + 1:]]
    return [command[0], str(PHASE_LAUNCHER), *launcher_args, "--", *command[1:]]


def run_system(command: list[str], env: dict[str, str], work: Path, sample_gpu: bool) -> tuple[float, float | None]:
    """Run ``command`` in ``work``; wall seconds and (when sampled) peak GPU MiB of its process tree."""
    print("Running:", " ".join(command), flush=True)
    started_at = time.time()
    process = subprocess.Popen(command, cwd=work, env=env)
    sampler = GpuMemorySampler(process.pid) if sample_gpu else None
    if sampler:
        sampler.start()
    status = process.wait()
    peak = sampler.stop() if sampler else None
    if status != 0:
        raise SystemExit(f"{command[1]} exited with status {status}")
    return time.time() - started_at, peak


def export_system_depths(method: str, work: Path, sink: DepthSink, poses: dict[int, np.ndarray],
                         frame_count: int) -> None:
    """Write the run's per-frame depth and confidence (slam_depth_export.py)."""
    if method == "VGGT_SLAM":
        export_vggt_slam_dumps(sink, work / "depth_dump")
        return
    chunk_dirs = sorted(work.rglob("_tmp_results_unaligned"))
    if len(chunk_dirs) != 1:
        raise RuntimeError(f"Expected one _tmp_results_unaligned under {work}, found {len(chunk_dirs)}")
    model = yaml.safe_load((work / "config.yaml").read_text(encoding="utf-8"))["Model"]
    ranges = chunk_ranges(frame_count, int(model["chunk_size"]), int(model["overlap"]))
    export_long_chunks(sink, chunk_dirs[0], ranges, poses, method)


def run() -> None:
    """Run one long-sequence SLAM method on a scene and export its benchmark artifacts."""
    args = parse_args()
    if args.track != "rgb":
        raise SystemExit(f"{args.method} takes RGB only (track {args.track!r} is not supported)")
    images_dir = canonical_image_root(args.data_path)
    image_paths = select_interval(find_image_paths(images_dir), args.interval, args.max_images_per_run)
    if len(image_paths) < 3:
        raise ValueError(f"Need at least 3 frames, found {len(image_paths)} in {images_dir}")
    code_path = args.code_path or DEFAULT_CODE_ROOT / CODE_DIRS[args.method]
    weights = weight_paths(args.ckpt_root)
    if args.dry_run:
        print({"method": args.method, "frames": len(image_paths), "code_path": str(code_path),
               "weights": {k: str(v) for k, v in weights.items()}, "output": str(args.output_path)})
        return
    if args.phase == "align" and args.method not in SPLIT_METHODS:
        raise SystemExit(f"{args.method} has no separate alignment phase")
    phase = args.phase if args.method in SPLIT_METHODS else "all"
    pending_path = args.output_path / ALIGN_PENDING
    fixed = fixed_input_size(image_paths)
    sink = DepthSink(args.output_path, image_paths) if args.export_depth else None

    work: Path | None = None
    try:
        if phase == "align":
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
            work = Path(pending["work"])
            if sink is not None and not pending.get("export_depth"):
                # A GPU phase run without --export_depth dropped the chunk arrays the depth export reads.
                pending_path.unlink()
                raise RuntimeError(
                    f"{args.method}: the pending GPU phase kept no chunk depth; rerun the inference phase")
            depth_dump = work / "depth_dump" if sink is not None else None
            command, env = system_command(args.method, code_path, work, bool(fixed), depth_dump)
            align_seconds, _ = run_system(phase_command(command, "align", work), env, work, sample_gpu=False)
            peak = pending["peak_gpu_memory_mib"]
            timing = {"inference_seconds": pending["gpu_seconds"] + align_seconds,
                      "gpu_phase_seconds": pending["gpu_seconds"], "align_phase_seconds": align_seconds}
            # write_adapter_metrics' "inference time" counts from started_at: the two phases, not the wait between jobs.
            started_at = time.time() - timing["inference_seconds"]
        else:
            args.output_path.mkdir(parents=True, exist_ok=True)
            # A split run's work directory must outlive this job and be readable from the align job's node.
            work = (Path(tempfile.mkdtemp(prefix=f"{args.method.lower()}_", dir=args.output_path)) if phase == "gpu"
                    else slam_work_dir(args.method, args.output_path))
            stage_images(image_paths, work / "images")
            depth_dump = work / "depth_dump" if sink is not None else None
            if args.method != "VGGT_SLAM":
                write_long_config(args.method, code_path, weights, work / "config.yaml", fixed,
                                  keep_chunks=sink is not None)
            if args.method in ("VGGT_LONG", "PI_LONG"):
                # Their loop detector loads DINOv2 from the cwd-relative ./LoopModels/dinov2; the run's cwd is the work
                # directory (outputs go to ./exps), so link the repo's LoopModels there.
                (work / "LoopModels").symlink_to(code_path / "LoopModels", target_is_directory=True)
            command, env = system_command(args.method, code_path, work, bool(fixed), depth_dump)
            if phase == "gpu":
                command = phase_command(command, "gpu", work, export_depth=sink is not None)
            started_at = time.time()
            elapsed, peak = run_system(command, env, work, sample_gpu=True)
            timing = {"inference_seconds": elapsed}
            if phase == "gpu" and (work / "phase_state.pkl").is_file():
                pending_path.write_text(json.dumps({"method": args.method, "work": str(work), "started_at": started_at,
                                                    "gpu_seconds": elapsed, "peak_gpu_memory_mib": peak,
                                                    "export_depth": sink is not None}) + "\n",
                                        encoding="utf-8")
                print(f"{args.method} GPU phase done in {elapsed:.0f} s; alignment pending ({pending_path})")
                return

        if args.method == "VGGT_SLAM":
            poses = read_tum_poses(work / "poses.txt")
        else:
            poses = read_long_poses(work / "exps", len(image_paths))
        indices = sorted(index for index in poses if 0 <= index < len(image_paths))
        if not indices:
            raise RuntimeError(f"{args.method} returned no poses for the {len(image_paths)} input frames")
        if sink is not None:
            export_system_depths(args.method, work, sink, poses, len(image_paths))
        returned = [image_paths[index] for index in indices]
        w2c = np.stack([np.linalg.inv(poses[index]) for index in indices])[:, :3, :4]
        width, height = Image.open(returned[0]).size
        focal = float(max(width, height))
        intrinsics = np.tile(np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]),
                             (len(returned), 1, 1))
        export_camera_pose_priors(args.output_path, images_dir, returned, w2c, intrinsics, width=width, height=height,
                                  colmap_model_subdir=args.colmap_model_subdir)
        input_record = work / "model_input_size.json"
        write_adapter_metrics(
            output_path=args.output_path,
            method=args.method,
            image_paths=image_paths,
            started_at=started_at,
            checkpoint=str(code_path),
            depth_only=False,
            metrics_extra={"track": args.track, "frames_with_pose": len(returned), "intrinsics": "nominal",
                           **(sink.metrics() | {"depth_frame": "submap" if args.method == "VGGT_SLAM" else "trajectory"}
                              if sink is not None else {})},
            # The system runs in a child process: wall time includes its model loading, and peak memory is the child's
            # process-tree peak from nvidia-smi (sampled at 1 s), not torch.cuda.max_memory_allocated. A split run's
            # inference_seconds is its GPU phase plus its alignment phase (each also reported).
            runtime={**timing, "model_load_seconds": None, "peak_gpu_memory_mib": peak,
                     "runtime_scope": "subprocess_wall",
                     "model_input_size": json.loads(input_record.read_text(encoding="utf-8"))["model_input_size"]
                     if input_record.is_file() else None},
        )
        pending_path.unlink(missing_ok=True)
        if not args.keep_work_dir:
            shutil.rmtree(work, ignore_errors=True)
    except BaseException:
        # A failed case leaves no work directory (staged frames, chunk arrays: often many GB) in the prediction
        # cache. A failed alignment phase keeps it: it holds the finished GPU phase that a retry needs.
        if work is not None and phase != "align" and not args.keep_work_dir:
            shutil.rmtree(work, ignore_errors=True)
        raise

if __name__ == "__main__":
    run()
