# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Bridge adapter for methods whose native integration is not in this repo yet.

The external command must write standardized benchmark artifacts into
``--output_path``: depth predictions under ``depths/`` for depth tasks, COLMAP
files under ``sparse/0`` for pose/SfM tasks, and optionally
``points3D_dense.ply`` for reconstruction visualization.

When the engine passes a subset of the scene's frames (``BENCHMARK_IMAGE_SUBSET``, e.g. one image-size group of
at most MAX_IMAGES_PER_RUN MonoDepth frames), the command sees a temporary scene folder whose ``images/`` holds
only those frames (symlinks) beside links to the scene's other entries, and ``{interval}`` is 1.

Command template placeholders:
  {data_path} {output_path} {images_dir} {depth_dir} {sparse_dir}
  {camera_dir} {method} {track} {code_path}
  {ckpt_path} {interval} {resolution} {image_load_resolution}
  {max_points} {max_images_per_run} {conf_thresh} {depth_conf_thresh}
  {export_depth} {export_colmap} {depth_only}
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.inference.adapter_utils import (  # noqa: E402
    IMAGE_SUBSET_ENV,
    canonical_image_root,
    find_image_paths,
    print_dry_run,
    restrict_to_image_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--track", default="rgb")
    parser.add_argument("--code_path", type=Path, default=None)
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--command", default="", help="External command template. Overrides METHOD_COMMAND env.")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--image_load_resolution", type=int, default=224)
    parser.add_argument("--max_points", type=int, default=10000)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=0.0)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", default="0")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def command_env_name(method_name: str) -> str:
    return f"{method_name.replace('-', '_')}_COMMAND".upper()


def render_command(args: argparse.Namespace, template: str) -> str:
    images_dir = canonical_image_root(args.data_path)
    depth_dir = args.data_path / "depths"
    sparse_dir = args.data_path / "sparse" / args.colmap_model_subdir
    values = {
        "data_path": shlex.quote(str(args.data_path)),
        "output_path": shlex.quote(str(args.output_path)),
        "images_dir": shlex.quote(str(images_dir)),
        "depth_dir": shlex.quote(str(depth_dir)),
        "sparse_dir": shlex.quote(str(sparse_dir)),
        "camera_dir": shlex.quote(str(sparse_dir)),
        "method": shlex.quote(args.method_name),
        "track": shlex.quote(args.track),
        "code_path": shlex.quote(str(args.code_path)) if args.code_path else "",
        "ckpt_path": shlex.quote(args.ckpt_path),
        "interval": str(args.interval),
        "resolution": str(args.resolution),
        "image_load_resolution": str(args.image_load_resolution),
        "max_points": str(args.max_points),
        "max_images_per_run": str(args.max_images_per_run),
        "conf_thresh": str(args.conf_thresh),
        "depth_conf_thresh": str(args.depth_conf_thresh),
        "export_depth": "1" if args.export_depth else "0",
        "export_colmap": "1" if args.export_colmap else "0",
        "depth_only": "1" if args.depth_only else "0",
        "colmap_model_subdir": shlex.quote(args.colmap_model_subdir),
    }
    return template.format(**values)


@contextlib.contextmanager
def subset_scene_view(data_path: Path, image_paths: list[str]):
    """A temporary scene folder holding only ``image_paths`` (as ``images/`` symlinks) and links to the scene's other
    entries, so a command that reads the whole image directory sees just this subset."""
    image_root = canonical_image_root(data_path).resolve()
    with tempfile.TemporaryDirectory(prefix="external_subset_") as tmp:
        view = Path(tmp) / data_path.name
        (view / "images").mkdir(parents=True)
        for image_path in image_paths:
            (view / "images" / Path(image_path).name).symlink_to(Path(image_path).resolve())
        for entry in data_path.iterdir():
            if entry.name != "images" and entry.resolve() != image_root:
                (view / entry.name).symlink_to(entry.resolve())
        yield view


def main() -> None:
    args = parse_args()
    image_paths = find_image_paths(canonical_image_root(args.data_path))
    if args.interval > 1:
        image_paths = image_paths[::args.interval]
    image_paths = restrict_to_image_subset(image_paths)
    if args.dry_run:
        print_dry_run(args, args.method_name, len(image_paths))
        return
    if args.track == "rgb_depth" and not (args.data_path / "depths").is_dir():
        raise FileNotFoundError(f"RGB+depth track requires canonical depth directory: {args.data_path / 'depths'}")
    if args.track == "rgb_cameras" and not (args.data_path / "sparse" / args.colmap_model_subdir).is_dir():
        raise FileNotFoundError(
            f"RGB+cameras track requires canonical COLMAP camera directory: "
            f"{args.data_path / 'sparse' / args.colmap_model_subdir}"
        )
    args.output_path.mkdir(parents=True, exist_ok=True)
    env_name = command_env_name(args.method_name)
    template = args.command or os.environ.get(env_name, "")
    if not template:
        raise RuntimeError(
            f"No native adapter is available for {args.method_name}. "
            f"Set {env_name} to an external command template that writes standardized artifacts."
        )
    log_path = args.output_path / "log.txt"
    with contextlib.ExitStack() as stack:
        render_args = args
        if os.environ.get(IMAGE_SUBSET_ENV, "").strip():
            view = stack.enter_context(subset_scene_view(args.data_path, image_paths))
            render_args = argparse.Namespace(**{**vars(args), "data_path": view, "interval": 1})
        command = render_command(render_args, template)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[external_command_adapter] {command}\n")
            log_file.flush()
            subprocess.run(command, shell=True, check=True, cwd=ROOT_DIR, stdout=log_file, stderr=subprocess.STDOUT)
    metrics = {
        "method": args.method_name,
        "track": args.track,
        "adapter": "external_command_adapter.py",
        "command_env": env_name,
        "data_path": str(args.data_path),
        "output_path": str(args.output_path),
        "num_images": len(image_paths),
    }
    (args.output_path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
