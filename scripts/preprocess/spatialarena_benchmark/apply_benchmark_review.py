# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Applies the manual benchmark review to the MindCloud benchmark.

The review is applied to the materialized benchmark and to its scene
selection. The labels come from the MindCloud monitor's Benchmark Review tab
(FeedForward-Benchmark's `app/mindcloud_monitor/benchmark_review.py`), saved
as `scripts/preprocess/spatialarena_benchmark/assets/benchmark_review.json`.
`materialize_spatialarena.py` applies them on every run as well; this script
applies them at once, without the released recordings, and records the scenes
left out in `benchmark_selection.yaml`. Without `--apply` it only prints what
it would change.

- A scene left out is deleted whole, with its entry in the track's
  `metadata.json` (and VideoDepth's `clips.json`), and removed from the
  selection: from its track's list, or, for a VideoDepth clip whose scene
  keeps other clips, added to `videodepth.exclude_clips`, and for a scene
  MonoDepth only samples, added to `monodepth.exclude`. A VideoDepth or
  MVRecon scene removed from the selection while MonoDepth keeps its sample
  becomes one of MonoDepth's own scenes, so that sample stays as it is.
- A MonoDepth frame left out is deleted from every folder of the scene and
  from `materialized_full.json`.
- A VideoDepth frame left out keeps its image, so the clip stays continuous;
  its GT depth (dense and LiDAR, and their heatmaps) is deleted and its
  `frame_manifest.json` entry gets `depth: null` and `gt_excluded: true`. The
  evaluator pairs GT and predictions by file name, so the frame is just not
  scored.
- MonoDepth samples the VideoDepth scenes, so a MonoDepth frame left out also
  loses its GT depth, as above, in every VideoDepth clip of the scene that
  holds the same image (the image stays as model input).

`--apply` also records the labels and the frames left out in
`<benchmark root>/benchmark_review.json`.

Each changed scene records the frames in `scene.json` (`manual_review`), and
the track's `metadata.json` is rewritten to match. Running it again is
harmless: what is already gone is skipped. It refuses to run while
`materialize_spatialarena.py` is writing a scene of the benchmark.

Example (prints what would change; add `--apply` to make the changes):

    python scripts/preprocess/spatialarena_benchmark/apply_benchmark_review.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import re
import shutil
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from scripts.preprocess import materialize_spatialarena  # noqa: E402

# pylint: enable=wrong-import-position

# The --help description.
_DESCRIPTION = ("Apply the manual benchmark review to the materialized "
                "MindCloud benchmark and to its scene selection.")
RELEASED_ROOT = pathlib.Path(
    os.environ.get("MINDCLOUD_RELEASED_ROOT", "~/datasets/released"))
CLIP_NAME = re.compile(r"^(?P<scene>.+)_(?P<kind>clip\d+|full)_cam\d+$")
SELECTION_KEYS = {
    "SfM": "sfm",
    "SLAM": "slam",
    "MVRecon": "mvrecon",
    "VideoDepth": "videodepth"
}


def read_json(path: pathlib.Path):
    """Reads a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload) -> None:
    """Writes `payload` as indented JSON, replacing `path` atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def default_benchmark_root(released_root: pathlib.Path) -> pathlib.Path:
    """Returns where the MindCloud monitor shows the benchmark.

    That is `MINDCLOUD_BENCHMARK_ROOT`, else `benchmark/` beside the released
    recordings (the `datasets_root` of FeedForward-Benchmark's MindCloud
    pipeline config).
    """
    configured = os.environ.get("MINDCLOUD_BENCHMARK_ROOT", "")
    if configured:
        return pathlib.Path(configured).expanduser().resolve()
    return (pathlib.Path(released_root).expanduser().resolve().parent /
            "benchmark")


def _find_section(lines: list[str], section: str) -> tuple[int, int]:
    """The lines of the top-level YAML `section`, end exclusive."""
    start = next(index for index, line in enumerate(lines)
                 if line.startswith(f"{section}:"))
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index] and
         not lines[index][0].isspace() and not lines[index].startswith("#")),
        len(lines))
    return start, end


def _flow_span(lines: list[str], start: int, end: int,
               key: str) -> tuple[int, int] | None:
    """The lines of `  key: [...]` within a section, end exclusive."""
    pattern = re.compile(rf"^  {re.escape(key)}:\s*\[")
    first = next(
        (index for index in range(start, end) if pattern.match(lines[index])),
        None)
    if first is None:
        return None
    last = next(index for index in range(first, end)
                if "]" in lines[index].split("#", 1)[0])
    return first, last + 1


def _flow_lines(key: str, values: list[str], suffix: str) -> list[str]:
    """Renders `  key: [...]`, wrapped under its bracket at 120 characters.

    Args:
        key: The mapping key.
        values: The list items.
        suffix: Text after the closing bracket (a trailing comment).

    Returns:
        The lines.
    """
    prefix = f"  {key}: ["
    indent = " " * len(prefix)
    lines, line = [], prefix
    for index, value in enumerate(values):
        if index == len(values) - 1:
            item = value + "]"
        else:
            item = value + ","
        if line not in (prefix, indent) and len(line) + 1 + len(item) > 120:
            lines.append(line)
            line = indent
        if line in (prefix, indent):
            line += item
        else:
            line += " " + item
    if values:
        lines.append(line + suffix)
    else:
        lines.append(prefix + "]" + suffix)
    return lines


def _rewrite_slam_sequences(lines: list[str], new: list[dict]) -> None:
    """Removes the SLAM sequence lines of scenes no longer in `new`."""
    kept = {sequence["scene"] for sequence in new}
    start, end = _find_section(lines, "slam")
    kept_lines = []
    for line in lines[start:end]:
        match = re.match(r"^\s+- \{scene: ([^,}]+)", line)
        if not match or match.group(1).strip() in kept:
            kept_lines.append(line)
    lines[start:end] = kept_lines
    if not new:  # No sequence left: an empty list, not a null.
        start, end = _find_section(lines, "slam")
        index = next(index for index in range(start, end)
                     if lines[index].startswith("  sequences:"))
        lines[index] = lines[index].replace("sequences:", "sequences: []", 1)


def _rewrite_flow_key(lines: list[str], section: str, mapping: dict, key: str,
                      new: list[str] | None) -> None:
    """Rewrites, removes or adds the flow list `key` of a section in place."""
    start, end = _find_section(lines, section)
    span = _flow_span(lines, start, end, key)
    if new is None:  # The key went away.
        if span:
            del lines[span[0]:span[1]]
        return
    if span:
        suffix = lines[span[1] - 1].split("]", 1)[1]
        lines[span[0]:span[1]] = _flow_lines(key, new, suffix)
        return
    # A new key: after the one it follows in the data.
    keys = list(mapping)
    previous = keys[keys.index(key) - 1]
    after = _flow_span(lines, start, end, previous)
    if after is None:  # A scalar key: its single line.
        after = next((index, index + 1)
                     for index in range(start, end)
                     if lines[index].startswith(f"  {previous}:"))
    lines[after[1]:after[1]] = _flow_lines(key, new, "")


def _remove_mvrecon_scenes(lines: list[str], removed: set[str]) -> None:
    """Removes the lines of the MVRecon scenes in `removed`."""
    start, end = _find_section(lines, "mvrecon")
    kept_lines = []
    for line in lines[start:end]:
        match = re.match(r"^\s{4}(\w+):", line)
        if not match or match.group(1) not in removed:
            kept_lines.append(line)
    lines[start:end] = kept_lines


class Selection:
    """Edits benchmark_selection.yaml, changing only the lines they must.

    The comments and layout stay. The selection is read as data, changed
    there, and written back as text edits: a changed flow list (`key: [a, b]`)
    is rewritten in place, wrapped under its bracket at 120 characters with
    its trailing comment kept; a removed MVRecon scene or SLAM sequence loses
    its line. The result is checked to read back as the data.

    Attributes:
        path: The selection file.
        text: Its original text.
        data: The selection, as changed so far.
        original: The selection as read.
        changes: A description of each change made.
    """

    def __init__(self, path: pathlib.Path):
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
            # Placed after `after` (dicts keep their order), as the text edit
            # will place it.
            items = list(mapping.items())
            if after in mapping:
                index = [item[0] for item in items].index(after) + 1
            else:
                index = len(items)
            mapping.clear()
            mapping.update(items[:index] + [(key, [])] + items[index:])
        if name not in mapping[key]:
            mapping[key].append(name)
            self.changes.append(f"{section}.{key}: add {name}")

    def keep_monodepth_sample(self, name: str, review: dict) -> None:
        """Makes a scene MonoDepth samples one of its own.

        So dropping it elsewhere keeps the MonoDepth sample.

        Args:
            name: The scene.
            review: The review labels.
        """
        mono = self.data["monodepth"]
        if review.get("MonoDepth", {}).get(name, {}).get("keep", True) is False:
            return
        if name in mono.get("exclude", []) or name in mono["scenes"]:
            return
        mono["scenes"].append(name)
        self.changes.append(
            f"monodepth.scenes: add {name} (its MonoDepth sample stays)")

    def drop(self, track: str, name: str, review: dict,
             clips: dict[str, list[str]]) -> None:
        """Removes a scene the review left out from the selection.

        Args:
            track: The track folder name.
            name: The scene (for VideoDepth, the clip).
            review: The review labels.
            clips: Each VideoDepth scene's clips.
        """
        if track == "SfM" and self._remove(self.data["sfm"]["scenes"], name):
            self.changes.append(f"sfm.scenes: remove {name}")
        elif track == "SLAM":
            sequences = self.data["slam"]["sequences"]
            matching = [
                sequence for sequence in sequences if sequence["scene"] == name
            ]
            for sequence in matching:
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
            elif any(name in self.data[sampled]["scenes"]
                     for sampled in self.data["monodepth"].get(
                         "sample_from", [])):
                self._add("monodepth", "exclude", name, after="sample_from")
        elif track == "VideoDepth":
            self.drop_clip(name, review, clips)

    def drop_clip(self, clip: str, review: dict,
                  clips: dict[str, list[str]]) -> None:
        """Leaves out one clip.

        When none of its scene's clips is left, the scene is removed instead.

        Args:
            clip: The clip's scene name.
            review: The review labels.
            clips: Each VideoDepth scene's clips.
        """
        video = self.data["videodepth"]
        match = CLIP_NAME.match(clip)
        scene = next(
            (source for source, names in clips.items() if clip in names),
            match and match["scene"])
        if not scene or scene not in video["scenes"]:
            return
        reviewed_out = {
            name for name, label in review.get("VideoDepth", {}).items()
            if label.get("keep", True) is False
        }
        left_out = set(video.get("exclude_clips", [])) | reviewed_out
        scene_clips = clips.get(scene)
        if not scene_clips:
            # A whole-sequence clip (`_full_`) is its scene's only one.
            if match and match["kind"] == "full":
                scene_clips = [clip]
            else:
                scene_clips = []
        if scene_clips and set(scene_clips) <= left_out:
            video["scenes"].remove(scene)
            self.changes.append(
                f"videodepth.scenes: remove {scene} (all its clips left out)")
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

    def render(self) -> str:
        """Returns the selection file's text with the changes applied."""
        lines = self.text.split("\n")
        for section, mapping in self.data.items():
            if not isinstance(mapping, dict):
                continue
            before = self.original.get(section, {})
            added = [key for key in mapping if key not in before]
            for key in list(before) + added:
                old, new = before.get(key), mapping.get(key)
                if old == new:
                    continue
                if not (isinstance(old, list) or isinstance(new, list)):
                    continue
                if section == "slam" and key == "sequences":
                    _rewrite_slam_sequences(lines, new)
                else:
                    _rewrite_flow_key(lines, section, mapping, key, new)
            if section == "mvrecon":
                removed = set(before.get("scenes", {})) - set(
                    mapping.get("scenes", {}))
                _remove_mvrecon_scenes(lines, removed)
        return "\n".join(lines)

    def save(self) -> None:
        """Writes the changed selection back.

        Raises:
            RuntimeError: If the edited text does not read back as the data.
        """
        text = self.render()
        if yaml.safe_load(text) != self.data:
            raise RuntimeError(
                f"Could not update {self.path} in place; edit it by hand: "
                f"{'; '.join(self.changes)}")
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)


class Review:
    """Applies the review labels to a benchmark root.

    With `apply=False` it only reports.

    Attributes:
        root: The materialized benchmark.
        review: The review labels.
        apply: Whether changes are made.
        removed_files: The number of files removed (or to remove).
        removed_bytes: Their total size.
        changed: Track -> scenes whose scene.json changed.
        dropped: Track -> scenes removed whole.
    """

    def __init__(self, root: pathlib.Path, review: dict, apply: bool):
        self.root = root
        self.review = review
        self.apply = apply
        self.removed_files = 0
        self.removed_bytes = 0
        self.changed: dict[str, set[str]] = {}
        self.dropped: dict[str, set[str]] = {}

    def count(self, paths: list[pathlib.Path]) -> None:
        """Adds `paths` to the files removed."""
        self.removed_files += len(paths)
        self.removed_bytes += sum(path.stat().st_size for path in paths)

    def drop_scene(self, track: str, scene: pathlib.Path) -> None:
        """Removes a whole scene."""
        print(f"  {track}/{scene.name}: remove the scene")
        self.count([path for path in scene.rglob("*") if path.is_file()])
        if self.apply:
            shutil.rmtree(scene)
        self.dropped.setdefault(track, set()).add(scene.name)

    def drop_frames(self, track: str, scene: pathlib.Path,
                    frames: list[str]) -> None:
        """Removes the frames left out of a MonoDepth or VideoDepth scene."""
        changed, removed = materialize_spatialarena.exclude_frames(scene,
                                                                   track,
                                                                   frames,
                                                                   apply=False)
        if not changed:
            return
        if track == "VideoDepth":
            what = "the GT depth of "
        else:
            what = ""
        print(f"  {track}/{scene.name}: remove {what}{len(changed)} frames")
        self.count(removed)
        if self.apply:
            materialize_spatialarena.exclude_frames(scene, track, frames)
            self.changed.setdefault(track, set()).add(scene.name)

    def update_track_files(self, track: str) -> None:
        """Rewrites the track's metadata.json for the scenes changed or removed.

        VideoDepth's clips.json too.
        """
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
                    entry = dict(entry)
                    entry.update(
                        read_json(track_root / name /
                                  materialize_spatialarena.SCENE_RECORD))
                scenes.append(entry)
            total = sum(int(entry["total_images"]) for entry in scenes)
            average = total / len(scenes) if scenes else 0.0
            metadata.update(total_images=total,
                            average_images_per_scene=average,
                            scenes=scenes)
            write_json(path, metadata)
        clips_path = track_root / "clips.json"
        if track == "VideoDepth" and dropped and clips_path.is_file():
            clips = read_json(clips_path)
            kept_clips = {}
            for source, scene_clips in clips.get("clips", {}).items():
                remaining = [
                    clip for clip in scene_clips
                    if clip["scene_name"] not in dropped
                ]
                if remaining:
                    kept_clips[source] = remaining
            clips["clips"] = kept_clips
            write_json(clips_path, clips)

    def run(self) -> None:
        """Removes the scenes, then the frames, the review leaves out."""
        scene_record = materialize_spatialarena.SCENE_RECORD
        for track, scenes in sorted(self.review.items()):
            for name, label in sorted(scenes.items()):
                scene = self.root / track / name
                if not (scene / scene_record).is_file():
                    continue
                if label.get("keep", True) is False:
                    self.drop_scene(track, scene)
        for track in ("MonoDepth", "VideoDepth"):
            for record in sorted((self.root / track).glob(f"*/{scene_record}")):
                scene = record.parent
                if scene.name in self.dropped.get(track, set()):
                    frames = []
                else:
                    frames = materialize_spatialarena.review_drops(
                        self.review, track, scene)
                if frames:
                    self.drop_frames(track, scene, frames)
        for track in sorted(set(self.changed) | set(self.dropped)):
            self.update_track_files(track)


def video_clips(root: pathlib.Path) -> dict[str, list[str]]:
    """Returns each VideoDepth scene's clips, from clips.json.

    Read before any clip is removed from it.
    """
    path = root / "VideoDepth" / "clips.json"
    if not path.is_file():
        return {}
    return {
        scene: [clip["scene_name"] for clip in clips]
        for scene, clips in read_json(path).get("clips", {}).items()
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line (`argv`, default sys.argv)."""
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("--labels",
                        type=pathlib.Path,
                        default=materialize_spatialarena.REVIEW_PATH,
                        help="the review labels")
    parser.add_argument(
        "--benchmark_root",
        type=pathlib.Path,
        default=None,
        help=("the materialized benchmark (default: MINDCLOUD_BENCHMARK_ROOT, "
              "else benchmark/ beside "
              "the released recordings)"))
    parser.add_argument("--selection",
                        type=pathlib.Path,
                        default=materialize_spatialarena.SELECTION_PATH,
                        help="the scene selection to update")
    parser.add_argument(
        "--released_root",
        type=pathlib.Path,
        default=RELEASED_ROOT,
        help=("the released MindCloud recordings "
              "(default: MINDCLOUD_RELEASED_ROOT, else ~/datasets/released)"))
    parser.add_argument("--apply",
                        action="store_true",
                        help="make the changes; without it only report them")
    return parser.parse_args(argv)


def _print_removed(applied: Review, apply: bool) -> None:
    """Prints how many files were (or would be) removed."""
    size = applied.removed_bytes / 2**20
    if not applied.removed_files:
        print("  nothing")
    elif apply:
        print(f"Removed {applied.removed_files:,} files ({size:,.0f} MB).")
    else:
        print(f"{applied.removed_files:,} files ({size:,.0f} MB) "
              "would be removed.")


def main(argv: list[str] | None = None) -> int:
    """Applies (or reports) the review; returns the exit status."""
    args = parse_args(argv)
    if not args.labels.is_file():
        print(f"No labels at {args.labels}: review the benchmark in the "
              "monitor's Benchmark Review tab first.")
        return 1
    review = materialize_spatialarena.load_review(args.labels)
    root = (args.benchmark_root or
            default_benchmark_root(args.released_root)).expanduser().resolve()
    staging = materialize_spatialarena.STAGING_DIRNAME
    busy = sorted(
        str(path.relative_to(root)) for path in root.glob(f"*/{staging}/*"))
    if busy and args.apply:
        print(f"The materializer is writing {', '.join(busy)}; "
              "run this once it has finished.")
        return 1

    selection = Selection(args.selection)
    clips = video_clips(root)
    for track, scenes in sorted(review.items()):
        for name, label in sorted(scenes.items()):
            if label.get("keep", True) is False:
                selection.drop(track, name, review, clips)

    if args.apply:
        removing, updating = "Removing", "Updated"
    else:
        removing, updating = "Would remove", "Would update"
    print(f"{removing} from {root} what {args.labels.name} leaves out:")
    applied = Review(root, review, args.apply)
    applied.run()
    _print_removed(applied, args.apply)
    print(f"{updating} {args.selection.name}:")
    print("\n".join(f"  {change}" for change in selection.changes) or
          "  nothing")
    if args.apply and selection.changes:
        selection.save()
    if args.apply:
        recorded = materialize_spatialarena.record_review(root, args.labels)
        print(f"Recorded the review in {recorded}.")
    if not args.apply and (applied.removed_files or selection.changes):
        print("Re-run with --apply to make these changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
