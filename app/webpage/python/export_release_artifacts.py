#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Export the Artifact Viewer's release subset into ``app/webpage/artifacts/``.

The public dashboard is committed to GitHub, so it cannot read the eval root (tens of GB of viewer assets). This copies
a small, fixed set of scenes per task (``release_artifacts.json``) from the eval root's viewer assets into
``artifacts/``, which the comparison modes read instead (``js/compare/compare_common.js``):

    artifacts/index/{depth,recon,trajectories}.json      the viewer index, same schema as viewer_assets/index/
    artifacts/scenes/<task>/<dataset_key>/<scene>/       assets every method of a scene shares (GT, RGB, thumbnails)
    artifacts/cases/<task>/<dataset_key>/<scene>/<method>/  one method's assets
    artifacts/README.md                                  what is included, and each source dataset's license

Per scene it keeps the task's ``top_methods`` best-ranked methods (the leaderboard's task ranks, from
``data/results_live.json`` if present, else ``data/results.json``) that have viewer assets for the scene, plus the
task's ``extra_methods``. Sizes are capped: depth scenes keep ``frames`` frames (``spread`` evenly over the scene, or
the ``first`` ones, so a video still plays), MVRecon clouds a random subset of ``points`` (GT: ``gt_points``) points,
and trajectory thumbnails are re-encoded at ``thumbnail_width``. The export fails if the folder exceeds
``max_total_mb``.

    python app/webpage/python/export_release_artifacts.py --eval-root /path/to/_evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.dataset_licenses import SOURCE_LICENSES  # noqa: E402

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = APP_ROOT / "release_artifacts.json"
DEFAULT_OUTPUT = APP_ROOT / "artifacts"
KIND_OF_TASK = {"MonoDepth": "depth", "VideoDepth": "depth", "MVRecon": "recon", "SfM": "trajectories",
                "WeakTextureSfM": "trajectories", "SLAM": "trajectories"}
RECON_POINT_FILES = (("points.f32", np.float32, 3), ("colors.u8", np.uint8, 3), ("error.f32", np.float32, 1),
                     ("confidence.f32", np.float32, 1))
SAMPLE_SEED = 0


def task_rankings(data_dir: Path) -> dict[str, list[str]]:
    """Method IDs of each task, best task rank first, from the payload the page shows."""
    for name in ("results_live.json", "results.json"):
        path = data_dir / name
        if path.is_file():
            ranking = json.loads(path.read_text(encoding="utf-8")).get("task_ranking") or {}
            if ranking.get("rows"):
                break
    else:
        raise SystemExit(f"No task ranking in {data_dir}/results_live.json or results.json; collect results first")
    order = {}
    for task in ranking.get("tasks", []):
        ranked = [row for row in ranking["rows"] if (row.get("task_scores", {}).get(task) or {}).get("rank")]
        ranked.sort(key=lambda row, task=task: row["task_scores"][task]["rank"])
        order[task] = [row["method"] for row in ranked]
    return order


def pick_methods(entry: dict, task: str, order: dict[str, list[str]], config: dict, eval_root: Path) -> list[dict]:
    """The scene's method entries to release: the best-ranked ones with assets, then the task's extras."""
    available = {item["method"]: item for item in entry["methods"] if (eval_root / item["path"]).exists()}
    picked = [available[method] for method in order.get(task, []) if method in available][: config["top_methods"]]
    for method in config.get("extra_methods", {}).get(task, []):
        if method in available and available[method] not in picked:
            picked.append(available[method])
    return picked


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_json(path: Path, payload, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent) + "\n", encoding="utf-8")


def pick_frames(common: list[str], cap: dict) -> list[str]:
    count = int(cap.get("frames") or len(common))
    if len(common) <= count:
        return common
    if cap.get("frame_pick") == "first":
        return common[:count]
    positions = np.linspace(0, len(common) - 1, count).round().astype(int)
    return [common[position] for position in sorted(set(positions.tolist()))]


def export_depth(entry: dict, methods: list[dict], eval_root: Path, out: Path, rel: str, cap: dict) -> dict:
    manifests = {item["method"]: json.loads((eval_root / item["path"] / "frames.json").read_text()) for item in methods}
    first = manifests[methods[0]["method"]]
    common = [frame["frame"] for frame in first["frames"]
              if all(any(other["frame"] == frame["frame"] for other in manifest["frames"])
                     for manifest in manifests.values())]
    keep = pick_frames(common, cap)
    keep_set = set(keep)
    shared_src = eval_root / entry["scene_path"]
    shared_rel = f"scenes/{rel}"
    for name in ("scene.json",):
        if (shared_src / name).is_file():
            copy_file(shared_src / name, out / shared_rel / name)
    for layer in ("gt", "rgb"):
        for frame in keep:
            src = shared_src / layer / f"{frame}.webp"
            if src.is_file():
                copy_file(src, out / shared_rel / layer / f"{frame}.webp")
    released = []
    for item in methods:
        manifest = dict(manifests[item["method"]])
        manifest["frames"] = [frame for frame in manifest["frames"] if frame["frame"] in keep_set]
        case_rel = f"cases/{rel}/{item['method']}"
        write_json(out / case_rel / "frames.json", manifest, indent=1)
        for layer in ("depth", "error"):
            for frame in keep:
                src = eval_root / item["path"] / layer / f"{frame}.webp"
                if src.is_file():
                    copy_file(src, out / case_rel / layer / f"{frame}.webp")
        released.append({**item, "path": case_rel, "frames": len(manifest["frames"])})
    return {**entry, "scene_path": shared_rel, "methods": released}


def subsample(path: Path, dtype, width: int, index: np.ndarray | None, dst: Path) -> None:
    values = np.fromfile(path, dtype=dtype).reshape(-1, width)
    (values if index is None else values[index]).tofile(dst)


def sample_index(count: int, cap: int) -> np.ndarray | None:
    if count <= cap:
        return None
    return np.sort(np.random.default_rng(SAMPLE_SEED).choice(count, size=cap, replace=False))


def export_recon(entry: dict, methods: list[dict], eval_root: Path, out: Path, rel: str, cap: dict) -> dict:
    shared_src = eval_root / entry["scene_path"]
    shared_rel = f"scenes/{rel}"
    (out / shared_rel).mkdir(parents=True, exist_ok=True)
    gt_count = (shared_src / "gt_points.f32").stat().st_size // 12
    gt_index = sample_index(gt_count, int(cap["gt_points"]))
    subsample(shared_src / "gt_points.f32", np.float32, 3, gt_index, out / shared_rel / "gt_points.f32")
    if (shared_src / "gt_colors.u8").is_file():
        subsample(shared_src / "gt_colors.u8", np.uint8, 3, gt_index, out / shared_rel / "gt_colors.u8")
    gt_kept = gt_count if gt_index is None else len(gt_index)
    scene_json = json.loads((shared_src / "scene.json").read_text())
    write_json(out / shared_rel / "scene.json", {**scene_json, "gt_points": gt_kept}, indent=1)
    released = []
    for item in methods:
        src = eval_root / item["path"]
        case_rel = f"cases/{rel}/{item['method']}"
        (out / case_rel).mkdir(parents=True, exist_ok=True)
        count = (src / "points.f32").stat().st_size // 12
        index = sample_index(count, int(cap["points"]))
        for name, dtype, width in RECON_POINT_FILES:
            if (src / name).is_file():
                subsample(src / name, dtype, width, index, out / case_rel / name)
        kept = count if index is None else len(index)
        cloud = json.loads((src / "cloud.json").read_text())
        write_json(out / case_rel / "cloud.json", {**cloud, "points": kept}, indent=1)
        has_cameras = (src / "cameras.json").is_file()
        if has_cameras:
            copy_file(src / "cameras.json", out / case_rel / "cameras.json")
        # Which optional files exist, so the viewer does not request missing ones.
        released.append({**item, "path": case_rel, "points": kept, "cameras": has_cameras,
                         "confidence": (src / "confidence.f32").is_file()})
    return {**entry, "scene_path": shared_rel, "gt_points": gt_kept, "methods": released}


def export_trajectories(entry: dict, methods: list[dict], eval_root: Path, out: Path, rel: str, config: dict) -> dict:
    names = set()
    released = []
    for item in methods:
        case_rel = f"cases/{rel}/{item['method']}/trajectory.json"
        trajectory = json.loads((eval_root / item["path"]).read_text())
        names.update(trajectory.get("names") or [])
        write_json(out / case_rel, trajectory)
        released.append({**item, "path": case_rel})
    result = {**entry, "methods": released}
    if entry.get("thumbnail_dir"):
        thumbs_rel = f"scenes/{rel}/thumbs"
        width, quality = int(config["caps"]["thumbnail_width"]), int(config["caps"]["thumbnail_quality"])
        for name in sorted(names):
            src = eval_root / entry["thumbnail_dir"] / f"{name}.webp"
            if not src.is_file():
                continue
            dst = out / thumbs_rel / f"{name}.webp"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as image:
                if image.width > width:
                    image = image.resize((width, max(1, round(image.height * width / image.width))), Image.LANCZOS)
                image.save(dst, "WEBP", quality=quality, method=6)
        result["thumbnail_dir"] = thumbs_rel
    return result


def folder_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def write_readme(out: Path, exported: list[dict], sizes: dict[str, int]) -> None:
    lines = [
        "# Artifact Viewer release subset",
        "",
        "Generated by `app/webpage/python/export_release_artifacts.py` from "
        "`app/webpage/release_artifacts.json`; do not edit by hand. The Artifact Viewer reads these files "
        "instead of the full evaluation outputs: a few scenes per task, each with its best-ranked methods, with frames "
        "and points capped to keep the repository small.",
        "",
        "| Task | Dataset | Scene | Methods | Size |",
        "|---|---|---|---|---|",
    ]
    for item in exported:
        lines.append(f"| {item['task']} | {item['dataset_key']} | {item['scene']} | {len(item['methods'])} | "
                     f"{sizes[item['key']] / 1e6:.1f} MB |")
    lines += ["", "## Source data and licenses", "",
              "The GT depth, RGB frames, thumbnails and GT point clouds here are derived from these datasets; each "
              "keeps its own license and terms.", ""]
    for key in sorted({item["license"] for item in exported}):
        source = SOURCE_LICENSES[key]
        lines.append(f"- **{source.name}** ({source.homepage}): {source.license_name}"
                     + (f", {source.license_url}" if source.license_url else "") + ".")
        if source.citation:
            lines.append(f"  Cite: {source.citation}")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path,
                        default=Path(os.environ.get("BENCHMARK_EVAL_ROOT", APP_ROOT / "benchmark_eval")),
                        help="Eval root with viewer_assets/ (default: $BENCHMARK_EVAL_ROOT or ./benchmark_eval).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=APP_ROOT / "data", help="Where results(_live).json are.")
    args = parser.parse_args(argv)

    eval_root = args.eval_root.expanduser().resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    order = task_rankings(args.data_dir)
    indexes = {kind: json.loads((eval_root / "viewer_assets" / "index" / f"{kind}.json").read_text())
               for kind in ("depth", "recon", "trajectories")}
    for key in {scene["license"] for scene in config["scenes"]}:
        if key not in SOURCE_LICENSES:
            raise SystemExit(f"Unknown license key {key!r} (spatialarena.dataset_licenses.SOURCE_LICENSES)")

    staging = args.output.with_name(args.output.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    released = {kind: [] for kind in indexes}
    exported, sizes = [], {}
    for scene in config["scenes"]:
        task, kind = scene["task"], KIND_OF_TASK[scene["task"]]
        entry = next((item for item in indexes[kind]["scenes"] if item["task"] == task
                      and item["dataset_key"] == scene["dataset_key"] and item["scene"] == scene["scene"]), None)
        if entry is None:
            raise SystemExit(f"{task} {scene['dataset_key']}/{scene['scene']} is not in {kind}.json")
        methods = pick_methods(entry, task, order, config, eval_root)
        if not methods:
            raise SystemExit(f"{task} {scene['dataset_key']}/{scene['scene']}: no method has viewer assets")
        rel = f"{task}/{scene['dataset_key']}/{scene['scene']}"
        before = folder_size(staging) if staging.exists() else 0
        if kind == "depth":
            result = export_depth(entry, methods, eval_root, staging, rel, config["caps"][task])
        elif kind == "recon":
            result = export_recon(entry, methods, eval_root, staging, rel, config["caps"][task])
        else:
            result = export_trajectories(entry, methods, eval_root, staging, rel, config)
        released[kind].append(result)
        key = f"{rel}"
        sizes[key] = folder_size(staging) - before
        exported.append({**scene, "key": key, "methods": [item["method"] for item in methods]})
        names = ", ".join(item["method"] for item in methods)
        print(f"{rel}: {len(methods)} methods, {sizes[key] / 1e6:.1f} MB ({names})")

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for kind, scenes in released.items():
        write_json(staging / "index" / f"{kind}.json",
                   {"generated_at": generated_at, "release": True, "scenes": scenes})
    write_readme(staging, exported, sizes)
    total = folder_size(staging)
    limit = float(config.get("max_total_mb") or 0)
    if limit and total > limit * 1e6:
        raise SystemExit(f"Release artifacts are {total / 1e6:.1f} MB, over max_total_mb={limit:g}; lower the caps or "
                         f"drop scenes (left in {staging})")
    shutil.rmtree(args.output, ignore_errors=True)
    staging.rename(args.output)
    print(f"Wrote {args.output}: {total / 1e6:.1f} MB in {sum(1 for f in args.output.rglob('*') if f.is_file())} files")


if __name__ == "__main__":
    main()
