# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Apply the manual benchmark review to the materialized MindCloud benchmark and to its scene selection.

The labels come from the MindCloud monitor's Benchmark Review tab (FeedForward-Benchmark's
``app/mindcloud_monitor/benchmark_review.py``),
saved as ``scripts/preprocess/spatialarena_benchmark/assets/benchmark_review.json``. ``materialize_spatialarena.py``
applies them on every run as well; this script applies them at once, without the released recordings, and records
the scenes left out in ``benchmark_selection.yaml``. Without ``--apply`` it only prints what it would change.

- A scene left out is deleted whole, with its entry in the track's ``metadata.json`` (and VideoDepth's
  ``clips.json``), and removed from the selection: from its track's list, or, for a VideoDepth clip whose scene keeps
  other clips, added to ``videodepth.exclude_clips``, and for a scene MonoDepth only samples, added to
  ``monodepth.exclude``. A VideoDepth or MVRecon scene removed from the selection while MonoDepth keeps its sample
  becomes one of MonoDepth's own scenes, so that sample stays as it is.
- A MonoDepth frame left out is deleted from every folder of the scene and from ``materialized_full.json``.
- A VideoDepth frame left out keeps its image, so the clip stays continuous; its GT depth (dense and LiDAR, and their
  heatmaps) is deleted and its ``frame_manifest.json`` entry gets ``depth: null`` and ``gt_excluded: true``. The
  evaluator pairs GT and predictions by file name, so the frame is just not scored.
- MonoDepth samples the VideoDepth scenes, so a MonoDepth frame left out also loses its GT depth, as above, in every
  VideoDepth clip of the scene that holds the same image (the image stays as model input).

``--apply`` also records the labels and the frames left out in ``<benchmark root>/benchmark_review.json``.

Each changed scene records the frames in ``scene.json`` (``manual_review``), and the track's ``metadata.json`` is
rewritten to match. Running it again is harmless: what is already gone is skipped. It refuses to run while
``materialize_spatialarena.py`` is writing a scene of the benchmark.

Example::

    python scripts/preprocess/spatialarena_benchmark/apply_benchmark_review.py            # what would change
    python scripts/preprocess/spatialarena_benchmark/apply_benchmark_review.py --apply
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from scripts.preprocess.materialize_spatialarena import (  # noqa: E402
    REVIEW_PATH,
    SCENE_RECORD,
    SELECTION_PATH,
    STAGING_DIRNAME,
    exclude_frames,
    load_review,
    record_review,
    review_drops,
)

# pylint: enable=wrong-import-position

RELEASED_ROOT = Path(os.environ.get("MINDCLOUD_RELEASED_ROOT", "~/datasets/released"))
CLIP_NAME = re.compile(r"^(?P<scene>.+)_(?P<kind>clip\d+|full)_cam\d+$")
SELECTION_KEYS = {"SfM": "sfm", "SLAM": "slam", "MVRecon": "mvrecon", "VideoDepth": "videodepth"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def default_benchmark_root(released_root: Path) -> Path:
    """Where the MindCloud monitor shows the benchmark: ``MINDCLOUD_BENCHMARK_ROOT``, else ``benchmark/`` beside the
    released recordings (the ``datasets_root`` of FeedForward-Benchmark's MindCloud pipeline config)."""
    configured = os.environ.get("MINDCLOUD_BENCHMARK_ROOT", "")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(released_root).expanduser().resolve().parent / "benchmark"


class Selection:
    """Edits of benchmark_selection.yaml that change only the lines they must (the comments and layout stay).

    The selection is read as data, changed there, and written back as text edits: a changed flow list
    (``key: [a, b]``) is rewritten in place, wrapped under its bracket at 120 characters with its trailing comment
    kept; a removed MVRecon scene or SLAM sequence loses its line. The result is checked to read back as the data."""

    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text(encoding="utf-8")
        self.data = yaml.safe_load(self.text)
        self.original = copy.deepcopy(self.data)
        self.changes: list[str] = []

    def _remove(self, values, name: str) -> bool:
        if name in values:
            values.remove(name)
            return True
        return False

    def _add(self, section: str, key: str, name: str, after: str) -> None:
        mapping = self.data[section]
        if key not in mapping:
            # Placed after ``after`` (dicts keep their order), as the text edit will place it.
            items = list(mapping.items())
            index = [item[0] for item in items].index(after) + 1 if after in mapping else len(items)
            mapping.clear()
            mapping.update(items[:index] + [(key, [])] + items[index:])
        if name not in mapping[key]:
            mapping[key].append(name)
            self.changes.append(f"{section}.{key}: add {name}")

    def keep_monodepth_sample(self, name: str, review: dict) -> None:
        """Make a scene MonoDepth samples one of its own, so dropping it elsewhere keeps the MonoDepth sample."""
        mono = self.data["monodepth"]
        if review.get("MonoDepth", {}).get(name, {}).get("keep", True) is False:
            return
        if name in mono.get("exclude", []) or name in mono["scenes"]:
            return
        mono["scenes"].append(name)
        self.changes.append(f"monodepth.scenes: add {name} (its MonoDepth sample stays)")

    def drop(self, track: str, name: str, review: dict, clips: dict[str, list[str]]) -> None:
        if track == "SfM" and self._remove(self.data["sfm"]["scenes"], name):
            self.changes.append(f"sfm.scenes: remove {name}")
        elif track == "SLAM":
            sequences = self.data["slam"]["sequences"]
            for sequence in [sequence for sequence in sequences if sequence["scene"] == name]:
                sequences.remove(sequence)
                self.changes.append(f"slam.sequences: remove {name}")
        elif track == "MVRecon" and name in self.data["mvrecon"]["scenes"]:
            del self.data["mvrecon"]["scenes"][name]
            self.changes.append(f"mvrecon.scenes: remove {name}")
            if "mvrecon" in self.data["monodepth"].get("sample_from", []):
                self.keep_monodepth_sample(name, review)
        elif track == "MonoDepth":
            if self._remove(self.data["monodepth"]["scenes"], name):
                self.changes.append(f"monodepth.scenes: remove {name}")
            elif any(name in self.data[track]["scenes"] for track in self.data["monodepth"].get("sample_from", [])):
                self._add("monodepth", "exclude", name, after="sample_from")
        elif track == "VideoDepth":
            self.drop_clip(name, review, clips)

    def drop_clip(self, clip: str, review: dict, clips: dict[str, list[str]]) -> None:
        """Leave out one clip; when none of its scene's clips is left, remove the scene instead."""
        video = self.data["videodepth"]
        match = CLIP_NAME.match(clip)
        scene = next((source for source, names in clips.items() if clip in names), match and match["scene"])
        if not scene or scene not in video["scenes"]:
            return
        left_out = set(video.get("exclude_clips", [])) | {
            name for name, label in review.get("VideoDepth", {}).items() if label.get("keep", True) is False}
        # A whole-sequence clip (``_full_``) is its scene's only one.
        scene_clips = clips.get(scene) or ([clip] if match and match["kind"] == "full" else [])
        if scene_clips and set(scene_clips) <= left_out:
            video["scenes"].remove(scene)
            self.changes.append(f"videodepth.scenes: remove {scene} (all its clips left out)")
            if scene in video.get("whole", []):
                video["whole"].remove(scene)
            for name in scene_clips:
                self._remove(video.get("exclude_clips", []), name)
            if "exclude_clips" in video and not video["exclude_clips"]:
                del video["exclude_clips"]
            if "videodepth" in self.data["monodepth"].get("sample_from", []):
                self.keep_monodepth_sample(scene, review)
        else:
            self._add("videodepth", "exclude_clips", clip, after="whole")

    @staticmethod
    def _section(lines: list[str], section: str) -> tuple[int, int]:
        start = next(index for index, line in enumerate(lines) if line.startswith(f"{section}:"))
        end = next((index for index in range(start + 1, len(lines))
                    if lines[index] and not lines[index][0].isspace() and not lines[index].startswith("#")),
                   len(lines))
        return start, end

    @staticmethod
    def _flow_span(lines: list[str], start: int, end: int, key: str) -> tuple[int, int] | None:
        """The lines of ``  key: [...]`` within a section, end exclusive."""
        pattern = re.compile(rf"^  {re.escape(key)}:\s*\[")
        first = next((index for index in range(start, end) if pattern.match(lines[index])), None)
        if first is None:
            return None
        last = next(index for index in range(first, end) if "]" in lines[index].split("#", 1)[0])
        return first, last + 1

    @staticmethod
    def _flow_lines(key: str, values: list[str], suffix: str) -> list[str]:
        prefix = f"  {key}: ["
        lines, line = [], prefix
        for index, value in enumerate(values):
            item = value + ("]" if index == len(values) - 1 else ",")
            if line not in (prefix, " " * len(prefix)) and len(line) + 1 + len(item) > 120:
                lines.append(line)
                line = " " * len(prefix)
            line += ("" if line in (prefix, " " * len(prefix)) else " ") + item
        lines.append((line if values else prefix + "]") + suffix)
        return lines

    def render(self) -> str:
        lines = self.text.split("\n")
        for section, mapping in self.data.items():
            if not isinstance(mapping, dict):
                continue
            before = self.original.get(section, {})
            for key in list(before) + [key for key in mapping if key not in before]:
                old, new = before.get(key), mapping.get(key)
                if old == new or not (isinstance(old, list) or isinstance(new, list)):
                    continue
                if section == "slam" and key == "sequences":
                    kept = {sequence["scene"] for sequence in new}
                    start, end = self._section(lines, section)
                    lines[start:end] = [line for line in lines[start:end]
                                        if not (match := re.match(r"^\s+- \{scene: ([^,}]+)", line))
                                        or match.group(1).strip() in kept]
                    if not new:  # no sequence left: an empty list, not a null
                        start, end = self._section(lines, section)
                        index = next(index for index in range(start, end) if lines[index].startswith("  sequences:"))
                        lines[index] = lines[index].replace("sequences:", "sequences: []", 1)
                    continue
                start, end = self._section(lines, section)
                span = self._flow_span(lines, start, end, key)
                if new is None:  # the key went away
                    if span:
                        del lines[span[0]:span[1]]
                    continue
                if span:
                    suffix = lines[span[1] - 1].split("]", 1)[1]
                    lines[span[0]:span[1]] = self._flow_lines(key, new, suffix)
                else:  # a new key: after the one it follows in the data
                    keys = list(mapping)
                    previous = keys[keys.index(key) - 1]
                    after = self._flow_span(lines, start, end, previous)
                    if after is None:  # a scalar key: its single line
                        after = next((index, index + 1) for index in range(start, end)
                                     if lines[index].startswith(f"  {previous}:"))
                    lines[after[1]:after[1]] = self._flow_lines(key, new, "")
            if section == "mvrecon":
                removed = set(before.get("scenes", {})) - set(mapping.get("scenes", {}))
                start, end = self._section(lines, section)
                lines[start:end] = [line for line in lines[start:end]
                                    if not (match := re.match(r"^\s{4}(\w+):", line)) or match.group(1) not in removed]
        return "\n".join(lines)

    def save(self) -> None:
        text = self.render()
        if yaml.safe_load(text) != self.data:
            raise RuntimeError(f"Could not update {self.path} in place; edit it by hand: {'; '.join(self.changes)}")
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)


class Review:
    """Applies the review labels to a benchmark root; with ``apply=False`` it only reports."""

    def __init__(self, root: Path, review: dict, apply: bool):
        self.root = root
        self.review = review
        self.apply = apply
        self.removed_files = 0
        self.removed_bytes = 0
        self.changed: dict[str, set[str]] = {}  # track -> scenes whose scene.json changed
        self.dropped: dict[str, set[str]] = {}  # track -> scenes removed whole

    def count(self, paths: list[Path]) -> None:
        self.removed_files += len(paths)
        self.removed_bytes += sum(path.stat().st_size for path in paths)

    def drop_scene(self, track: str, scene: Path) -> None:
        print(f"  {track}/{scene.name}: remove the scene")
        self.count([path for path in scene.rglob("*") if path.is_file()])
        if self.apply:
            shutil.rmtree(scene)
        self.dropped.setdefault(track, set()).add(scene.name)

    def drop_frames(self, track: str, scene: Path, frames: list[str]) -> None:
        changed, removed = exclude_frames(scene, track, frames, apply=False)
        if not changed:
            return
        what = "the GT depth of " if track == "VideoDepth" else ""
        print(f"  {track}/{scene.name}: remove {what}{len(changed)} frames")
        self.count(removed)
        if self.apply:
            exclude_frames(scene, track, frames)
            self.changed.setdefault(track, set()).add(scene.name)

    def update_track_files(self, track: str) -> None:
        """Rewrite the track's metadata.json (and VideoDepth's clips.json) for the scenes changed or removed."""
        changed = self.changed.get(track, set())
        dropped = self.dropped.get(track, set())
        if not self.apply or not (changed or dropped):
            return
        track_root = self.root / track
        path = track_root / "metadata.json"
        if path.is_file():
            metadata = read_json(path)
            scenes = []
            for entry in metadata.get("scenes", []):
                name = entry["scene_name"]
                if name in dropped:
                    continue
                if name in changed:
                    entry = {**entry, **read_json(track_root / name / SCENE_RECORD)}
                scenes.append(entry)
            total = sum(int(entry["total_images"]) for entry in scenes)
            metadata.update(total_images=total, average_images_per_scene=total / len(scenes) if scenes else 0.0,
                            scenes=scenes)
            write_json(path, metadata)
        clips_path = track_root / "clips.json"
        if track == "VideoDepth" and dropped and clips_path.is_file():
            clips = read_json(clips_path)
            clips["clips"] = {source: [clip for clip in scene_clips if clip["scene_name"] not in dropped]
                              for source, scene_clips in clips.get("clips", {}).items()}
            clips["clips"] = {source: scene_clips for source, scene_clips in clips["clips"].items() if scene_clips}
            write_json(clips_path, clips)

    def run(self) -> None:
        for track, scenes in sorted(self.review.items()):
            for name, label in sorted(scenes.items()):
                scene = self.root / track / name
                if not (scene / SCENE_RECORD).is_file():
                    continue
                if label.get("keep", True) is False:
                    self.drop_scene(track, scene)
        for track in ("MonoDepth", "VideoDepth"):
            for record in sorted((self.root / track).glob(f"*/{SCENE_RECORD}")):
                scene = record.parent
                frames = [] if scene.name in self.dropped.get(track, set()) else review_drops(self.review, track, scene)
                if frames:
                    self.drop_frames(track, scene, frames)
        for track in sorted(set(self.changed) | set(self.dropped)):
            self.update_track_files(track)


def video_clips(root: Path) -> dict[str, list[str]]:
    """Each VideoDepth scene's clips, from clips.json (read before any clip is removed from it)."""
    path = root / "VideoDepth" / "clips.json"
    if not path.is_file():
        return {}
    return {scene: [clip["scene_name"] for clip in clips] for scene, clips in read_json(path).get("clips", {}).items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--labels", type=Path, default=REVIEW_PATH, help="the review labels")
    parser.add_argument("--benchmark_root", type=Path, default=None,
                        help="the materialized benchmark (default: MINDCLOUD_BENCHMARK_ROOT, else benchmark/ beside "
                             "the released recordings)")
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH, help="the scene selection to update")
    parser.add_argument("--released_root", type=Path, default=RELEASED_ROOT,
                        help="the released MindCloud recordings "
                             "(default: MINDCLOUD_RELEASED_ROOT, else ~/datasets/released)")
    parser.add_argument("--apply", action="store_true", help="make the changes; without it only report them")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.labels.is_file():
        print(f"No labels at {args.labels}: review the benchmark in the monitor's Benchmark Review tab first.")
        return 1
    review = load_review(args.labels)
    root = (args.benchmark_root or default_benchmark_root(args.released_root)).expanduser().resolve()
    busy = sorted(str(path.relative_to(root)) for path in root.glob(f"*/{STAGING_DIRNAME}/*"))
    if busy and args.apply:
        print(f"The materializer is writing {', '.join(busy)}; run this once it has finished.")
        return 1

    selection = Selection(args.selection)
    clips = video_clips(root)
    for track, scenes in sorted(review.items()):
        for name, label in sorted(scenes.items()):
            if label.get("keep", True) is False:
                selection.drop(track, name, review, clips)

    print(f"{'Removing' if args.apply else 'Would remove'} from {root} what {args.labels.name} leaves out:")
    applied = Review(root, review, args.apply)
    applied.run()
    size = applied.removed_bytes / 2**20
    if not applied.removed_files:
        print("  nothing")
    elif args.apply:
        print(f"Removed {applied.removed_files:,} files ({size:,.0f} MB).")
    else:
        print(f"{applied.removed_files:,} files ({size:,.0f} MB) would be removed.")
    print(f"{'Updated' if args.apply else 'Would update'} {args.selection.name}:")
    print("\n".join(f"  {change}" for change in selection.changes) or "  nothing")
    if args.apply and selection.changes:
        selection.save()
    if args.apply:
        print(f"Recorded the review in {record_review(root, args.labels)}.")
    if not args.apply and (applied.removed_files or selection.changes):
        print("Re-run with --apply to make these changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
