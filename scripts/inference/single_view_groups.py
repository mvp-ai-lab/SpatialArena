# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run a single-view (MonoDepth) scene as one adapter call per image size and frame budget, then merge.

A batched single-view adapter resizes its [B, 1, 3, H, W] batch to one shape, and a one-image-per-call adapter stacks
its per-frame outputs into one (N, H, W) array. A scene whose frames come from different cameras (AerialMetric mixes
twelve image sizes) would therefore make a frame's prediction depend on which frames shared its batch, or fail.
MonoDepth also scores every frame of a scene (adapter_utils.resolve_capped_image_indices applies no frame cap in
single-view mode), and some adapters load all their frames at once, so each call gets at most ``max_images_per_run``
of them. run_method_case_list.sh plans these groups (same size, at most ``max_images_per_run`` frames), runs the
adapter once per group (``BENCHMARK_IMAGE_SUBSET``, read by adapter_utils.select_interval) into
``<groups_dir>/group_<k>/``, and merges the per-frame files back into the case's output:

    python single_view_groups.py plan --data_path SCENE --interval 1 --max_images_per_run 100 --groups_dir DIR
    python single_view_groups.py merge --groups_dir DIR --output_path OUT

``plan`` prints the number of groups and writes ``DIR/group_<k>.txt`` (one image path per line) only when there are
two or more; a uniform scene within the frame budget runs as one call.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter_utils import (  # noqa: E402  # pylint: disable=wrong-import-position
    canonical_image_root, find_image_paths, resolve_capped_image_indices,
)

# Per-frame outputs moved as is (raw_disparities: disparity models' unconverted output, read by the inverse-depth
# protocols; raw_log_depths: Marigold V2's normalized log depth, read by log_huber_irls_gt); point_query_outputs.jsonl
# (VLM point queries) is concatenated, and so is point_query_coverage.json (a JSON array, one entry per image);
# spatial_transforms.json (the shared spatial policy's per-image restore record, keyed by image name) is united.
FRAME_DIRS = ("depths", "confidences", "raw_disparities", "raw_log_depths")
JSONL_FILES = ("point_query_outputs.jsonl", "openrouter_outputs.jsonl")
LIST_FILES = ("point_query_coverage.json",)
# Per-image lists in the adapters' metrics.json (point-query coverage): a scene's list is the groups' concatenated.
CONCATENATED_LISTS = ("image_query_coverage",)
# Per-group counters the adapters record (point-query and OpenRouter VLMs): a scene's total is their sum.
SUMMED_COUNTERS = (
    "total_point_queries",
    "parse_failures",
    "fallback_imputations",
    "sky_points",
    "clamped_points",
    "api_calls",
    "cache_hits",
    "api_cost_usd",
    "api_prompt_tokens",
    "api_completion_tokens",
    "api_reasoning_tokens",
    "followup_calls",
    "followup_filled_points",
)
RECORD_FILES = ("spatial_transforms.json",)
METRICS_FILE = "metrics.json"


def size_groups(image_paths: list[str]) -> list[list[str]]:
    """The image paths grouped by (width, height), each group in input order, groups by first appearance."""
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for path in image_paths:
        with Image.open(path) as image:
            groups[image.size].append(path)
    return list(groups.values())


def plan(data_path: Path, interval: int, max_images_per_run: int,
         groups_dir: Path) -> int:
    """Write one image list per adapter call (one size, at most ``max_images_per_run`` frames) over the frames
    select_interval keeps; returns the number of groups."""
    image_paths = find_image_paths(canonical_image_root(data_path))
    selected = [
        image_paths[index] for index in resolve_capped_image_indices(
            len(image_paths), interval, max_images_per_run)
    ]
    budget = max_images_per_run if max_images_per_run > 0 else len(
        selected) or 1
    groups = [
        group[start:start + budget]
        for group in size_groups(selected)
        for start in range(0, len(group), budget)
    ]
    if len(groups) > 1:
        groups_dir.mkdir(parents=True, exist_ok=True)
        for index, group in enumerate(groups):
            (groups_dir / f"group_{index:02d}.txt").write_text(
                "\n".join(group) + "\n", encoding="utf-8")
    return len(groups)


def merge_metrics(payloads: list[dict]) -> dict:
    """One metrics.json for the scene: times, frame counts and the adapters' per-group counters (API calls, cost,
    tokens, parse failures, ...) add up, the peak memory is the largest, and the model load time is the mean (an
    ungrouped run loads the model once)."""
    merged = dict(payloads[0])
    for key in ("inference time", "inference_seconds", "Preprocessed Images",
                *SUMMED_COUNTERS):
        values = [payload.get(key) for payload in payloads]
        if all(isinstance(value, (int, float)) for value in values):
            merged[key] = sum(values)
    for key in CONCATENATED_LISTS:
        if all(isinstance(payload.get(key), list) for payload in payloads):
            merged[key] = [
                entry for payload in payloads for entry in payload[key]
            ]
    loads = [payload.get("model_load_seconds") for payload in payloads]
    if all(isinstance(value, (int, float)) for value in loads):
        merged["model_load_seconds"] = sum(loads) / len(loads)
    for key in ("peak_gpu_memory_mib", "Peak GPU Memory"):
        values = [
            payload[key]
            for payload in payloads
            if isinstance(payload.get(key), (int, float))
        ]
        if values:
            merged[key] = max(values)
    largest = max(payloads,
                  key=lambda payload: payload.get("Preprocessed Images") or 0)
    merged["model_input_size"] = largest.get("model_input_size")
    merged["size_groups"] = [{
        "frames": payload.get("Preprocessed Images"),
        "model_input_size": payload.get("model_input_size")
    } for payload in payloads]
    return merged


def merge(groups_dir: Path, output_path: Path) -> None:
    """Move every group's per-frame outputs into ``output_path`` and write the combined metrics.json.

    Everything is checked before anything moves, so a failed merge never leaves a partial prediction that the engine
    would take for a finished one: each group must have predicted exactly the frames of its list (an adapter that
    ignored ``BENCHMARK_IMAGE_SUBSET`` would not), and only per-frame files, JSONL, per-image records
    (spatial_transforms.json) and metrics.json are expected.
    """
    group_lists = sorted(groups_dir.glob("group_*.txt"))
    if not group_lists:
        raise FileNotFoundError(f"No group lists under {groups_dir}")
    moves: list[tuple[Path, Path]] = []
    jsonl: list[tuple[Path, Path]] = []
    records: dict[str, dict] = {name: {} for name in RECORD_FILES}
    lists: dict[str, list] = {name: [] for name in LIST_FILES}
    record_clashes: set[str] = set()
    payloads = []
    for group_list in group_lists:
        group_output = group_list.with_suffix("")
        expected = {
            Path(line).stem
            for line in group_list.read_text(encoding="utf-8").split("\n")
            if line
        }
        predicted = {
            path.stem
            for path in (group_output / "depths").glob("*")
            if path.suffix in (".npz", ".npy")
        }
        if predicted != expected:
            raise ValueError(
                f"{group_output} predicted {len(predicted)} frame(s), its list has {len(expected)} "
                f"({len(predicted - expected)} unexpected, {len(expected - predicted)} missing)"
            )
        if not (group_output / METRICS_FILE).is_file():
            raise FileNotFoundError(f"{group_output} wrote no {METRICS_FILE}")
        payload = json.loads(
            (group_output / METRICS_FILE).read_text(encoding="utf-8"))
        payloads.append(payload)
        if payload.get("representation") == "raw_normalized_log_depth":
            # log_huber_irls_gt scores only raw_log_depths/: every frame of the group must have one.
            raw_keys = {
                path.stem
                for path in (group_output / "raw_log_depths").glob("*")
                if path.suffix in (".npz", ".npy")
            }
            if raw_keys != expected:
                raise ValueError(f"Incomplete raw_log_depths in {group_output}")
        for entry in group_output.iterdir():
            if entry.name in FRAME_DIRS:
                moves.extend(
                    (frame_file, output_path / entry.name / frame_file.name)
                    for frame_file in entry.iterdir())
            elif entry.name in JSONL_FILES:
                jsonl.append((entry, output_path / entry.name))
            elif entry.name in LIST_FILES:
                lists[entry.name].extend(
                    json.loads(entry.read_text(encoding="utf-8")))
            elif entry.name in RECORD_FILES:
                group_records = json.loads(entry.read_text(encoding="utf-8"))
                record_clashes |= set(group_records) & set(records[entry.name])
                records[entry.name].update(group_records)
            elif entry.name != METRICS_FILE:
                raise ValueError(
                    f"Unexpected adapter output {entry} (single-view groups merge only per-frame files)"
                )
    targets = [target for _, target in moves]
    clashes = {
        target for target in targets
        if targets.count(target) > 1 or target.exists()
    }
    for name, merged_records in records.items():
        if merged_records and (output_path / name).is_file():
            record_clashes |= set(merged_records) & set(
                json.loads((output_path / name).read_text(encoding="utf-8")))
    clashes |= record_clashes
    if clashes:
        raise FileExistsError(
            f"Size groups would overwrite {sorted(map(str, clashes))[:3]}")
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    for source, target in jsonl:
        with target.open("a", encoding="utf-8") as merged_file:
            merged_file.write(source.read_text(encoding="utf-8"))
    for name, merged_list in lists.items():
        if merged_list:
            target = output_path / name
            existing = json.loads(target.read_text(
                encoding="utf-8")) if target.is_file() else []
            target.write_text(json.dumps(existing + merged_list, indent=2),
                              encoding="utf-8")
    for name, merged_records in records.items():
        if merged_records:
            target = output_path / name
            existing = json.loads(target.read_text(
                encoding="utf-8")) if target.is_file() else {}
            target.write_text(json.dumps({
                **existing,
                **merged_records
            },
                                         indent=2),
                              encoding="utf-8")
    (output_path / METRICS_FILE).write_text(
        json.dumps(merge_metrics(payloads), indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(groups_dir)


def main() -> None:
    """Command-line entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--data_path", type=Path, required=True)
    plan_parser.add_argument("--interval", type=int, default=1)
    plan_parser.add_argument("--max_images_per_run", type=int, default=0)
    plan_parser.add_argument("--groups_dir", type=Path, required=True)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--groups_dir", type=Path, required=True)
    merge_parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(
            plan(args.data_path, args.interval, args.max_images_per_run,
                 args.groups_dir))
    else:
        merge(args.groups_dir, args.output_path)


if __name__ == "__main__":
    main()
