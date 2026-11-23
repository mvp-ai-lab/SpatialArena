#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Materialize Synth4K (InfiniDepth's 4K game benchmark) as MonoDepth `synth4k`.

Source: one tar per game from HF dataset ritianyu/game_4k_data
(scripts/download/synth4k.sh), laid out as
`./<game>/<sequence>/{image/<frame>.png, depth/<frame>.exr,
highfreq_mask/<frame>_highfreq_mask.png, gt_pose.csv}`. Each game becomes one
scene (cyberpunk, deadisland, spiderman2, spidermanmm, watchdoglegion).

InfiniDepth publishes no evaluation frame list, so each scene keeps
`--frame_fraction` (default 0.5, half) of the `<game>/<sequence>/<frame>` ids
that have an image, a depth map, a high-frequency mask and some GT in the
evaluated range, spread uniformly over them. `--frames_per_scene N` (N > 0)
instead spreads N frames uniformly over the sorted complete ids, a frame
without any GT in range being replaced by the next such id (the 100-frame
selection of earlier releases). The selection is committed as
`spatialarena/eval/monodepth_splits/synth4k/<game>.txt` and reused on later
runs.

Per frame: the 3840x2160 RGBA PNG is stored as RGB PNG (lossless); the EXR
depth (metres, game far plane for sky) as float32 .npy with non-finite values
and the sky (depth at or beyond the game's far plane, `SKY_DEPTH`) set to 0;
the high-frequency mask (InfiniDepth's fine-detail region, not used by the
default affine protocol) under `highfreq_masks/`. The scene limits follow
InfiniDepth's loader (`training/dataset/depth_estimation/synth4k.py`): depth
in (3, 150) m is evaluated. No intrinsics are written: InfiniDepth evaluates
Synth4K without them and the capture's `fov` column is not documented.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tarfile
from pathlib import Path

# OpenCV reads this flag on import, so it is set before cv2 is imported.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# pylint: disable=wrong-import-position
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess import monodepth_layout  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402
# pylint: enable=wrong-import-position

DATASET = "synth4k"
GAMES = ("cyberpunk", "deadisland", "spiderman2", "spidermanmm",
         "watchdoglegion")
DEFAULT_SOURCE_ROOT = (dataset_registry.DEFAULT_BENCHMARK_ROOT /
                       "MonoDepth_source" / "synth4k")
SYNTH4K_REPO = "ritianyu/game_4k_data"
SYNTH4K_REVISION = "9d5fc8d41dd73304240eff5eaa15ba002753b990"
LIMITS = monodepth_layout.SceneLimits(min_depth=3.0, max_depth=150.0)
# Sky: depth at or beyond the game's far plane is stored as 0 (invalid). Four
# games write one constant far-plane value; Dead Island's sky spans ~2e5-6e5
# with nothing between 5e4 and 2e5, so its cut-off is 1e5.
SKY_DEPTH = {
    "cyberpunk": 17152.0,
    "deadisland": 1e5,
    "spiderman2": 17723.0,
    "spidermanmm": 10039.2,
    "watchdoglegion": 9000.0,
}
MEMBER_PATTERN = re.compile(
    r"^(?:\./)?(?P<game>[^/]+)/(?P<sequence>[^/]+)/"
    r"(?P<kind>image|depth|highfreq_mask)/"
    r"(?P<frame>\d+)(?:_highfreq_mask)?\.(?P<ext>png|exr)$")


def parse_args() -> argparse.Namespace:
    """Parses the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--benchmark_root",
                        type=Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--games",
                        nargs="+",
                        choices=GAMES,
                        default=list(GAMES))
    parser.add_argument(
        "--frames_per_scene",
        type=int,
        default=0,
        help="Frames per game, uniformly spaced (0: use --frame_fraction).")
    parser.add_argument(
        "--frame_fraction",
        type=float,
        default=0.5,
        help="Share of a game's scorable frames kept, uniformly spaced.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write_split",
        action="store_true",
        help=("Reselect the frames and rewrite the committed frame list "
              "instead of reusing it."),
    )
    return parser.parse_args()


def index_members(
        tar: tarfile.TarFile) -> dict[str, dict[str, tarfile.TarInfo]]:
    """Indexes the frame files of a game tar.

    Args:
        tar: The open game tar.

    Returns:
        `<game>/<sequence>/<frame>` -> {"image"|"depth"|"highfreq_mask":
        member}.
    """
    frames: dict[str, dict[str, tarfile.TarInfo]] = {}
    for member in tar:
        if not member.isfile():
            continue
        match = MEMBER_PATTERN.match(member.name)
        if match is None:
            continue
        frame_id = f"{match['game']}/{match['sequence']}/{match['frame']}"
        frames.setdefault(frame_id, {})[match["kind"]] = member
    return frames


def read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    """Returns the bytes of one tar member.

    Raises:
        ValueError: If the member cannot be extracted.
    """
    extracted = tar.extractfile(member)
    if extracted is None:
        raise ValueError(f"Could not extract {member.name}")
    return extracted.read()


def decode_exr_depth(content: bytes) -> np.ndarray:
    """Decodes a single-channel EXR depth map (metres) to float32.

    Raises:
        ValueError: If OpenCV cannot decode `content`.
    """
    depth = cv2.imdecode(np.frombuffer(content, dtype=np.uint8),
                         cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError("Could not decode EXR depth")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def is_scorable(depth: np.ndarray) -> bool:
    """Returns whether any GT pixel lies inside the evaluated depth range."""
    return bool((np.isfinite(depth) & (depth > LIMITS.min_depth) &
                 (depth < LIMITS.max_depth)).any())


def select_scorable(
    tar: tarfile.TarFile,
    frames: dict[str, dict[str, tarfile.TarInfo]],
    complete: list[str],
    count: int,
    fraction: float = 1.0,
) -> list[str]:
    """Selects the frames of a game that can be scored.

    With `count` > 0, `count` frames are spaced uniformly over `complete`,
    each moved forward to the next unselected frame that has GT inside the
    evaluated depth range (frames entirely closer than 3 m or beyond 150 m
    cannot be scored). When `count` is 0, `fraction` of the frames with such
    GT are kept, uniformly spaced over them.

    Args:
        tar: The open game tar.
        frames: The tar index from `index_members`.
        complete: Sorted ids of the frames that have every file.
        count: Frames to select, or 0 to use `fraction`.
        fraction: Share of the scorable frames kept when `count` is 0.

    Returns:
        The selected frame ids, sorted.
    """
    if count <= 0:
        scorable = [
            frame_id for frame_id in complete if is_scorable(
                decode_exr_depth(read_member(tar, frames[frame_id]["depth"])))
        ]
        return monodepth_layout.uniform_subsample(
            scorable, monodepth_layout.fraction_count(len(scorable), fraction))
    selected: list[str] = []
    taken: set[str] = set()
    for frame_id in monodepth_layout.uniform_subsample(complete, count):
        position = complete.index(frame_id)
        while position < len(complete):
            candidate = complete[position]
            if candidate not in taken and is_scorable(
                    decode_exr_depth(
                        read_member(tar, frames[candidate]["depth"]))):
                selected.append(candidate)
                taken.add(candidate)
                break
            position += 1
    return sorted(selected)


def remove_stale_frames(scene_root: Path, keys: set[str]) -> None:
    """Deletes the files of frames no longer in the scene's frame list."""
    for folder in ("images", "depths", "highfreq_masks"):
        if not (scene_root / folder).is_dir():
            continue
        for path in sorted((scene_root / folder).glob("*")):
            key = path.stem
            if key not in keys:
                path.unlink()


def _write_frame(tar: tarfile.TarFile, kinds: dict[str, tarfile.TarInfo],
                 scene_root: Path, frame_id: str, game: str,
                 overwrite: bool) -> monodepth_layout.FrameRecord:
    """Writes the image, depth and high-frequency mask of one frame.

    Args:
        tar: The open game tar.
        kinds: The frame's members, keyed by kind.
        scene_root: The game's scene folder.
        frame_id: The `<game>/<sequence>/<frame>` id.
        game: Game name, for its sky depth.
        overwrite: Whether existing files are rewritten.

    Returns:
        The frame's record.
    """
    key = monodepth_layout.frame_key(frame_id)
    image_rel = f"images/{key}.png"
    depth_rel = f"depths/{key}.npy"
    highfreq_rel = f"highfreq_masks/{key}.png"
    if overwrite or not (scene_root / image_rel).is_file():
        image_bytes = read_member(tar, kinds["image"])
        with Image.open(io.BytesIO(image_bytes)) as image:
            (scene_root / image_rel).parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(scene_root / image_rel, compress_level=6)
    if overwrite or not (scene_root / depth_rel).is_file():
        depth = decode_exr_depth(read_member(tar, kinds["depth"]))
        monodepth_layout.save_depth(scene_root / depth_rel,
                                    depth,
                                    valid=depth < SKY_DEPTH[game] * 0.999)
    if overwrite or not (scene_root / highfreq_rel).is_file():
        (scene_root / highfreq_rel).parent.mkdir(parents=True, exist_ok=True)
        (scene_root / highfreq_rel).write_bytes(
            read_member(tar, kinds["highfreq_mask"]))
    return monodepth_layout.FrameRecord(
        key=key,
        image=image_rel,
        depth=depth_rel,
        source=frame_id,
        extra={"highfreq_mask": highfreq_rel},
    )


def _game_provenance(game: str, tar_path: Path, num_records: int,
                     num_complete: int, frames_per_scene: int,
                     frame_fraction: float) -> dict:
    """Returns the provenance recorded for one game."""
    frame_list = (f"spatialarena/eval/monodepth_splits/{DATASET}/{game}.txt "
                  f"({num_records} of the {num_complete} "
                  "sorted complete frames: ")
    if frames_per_scene <= 0:
        frame_list += (f"{frame_fraction:g} of the frames with GT in range, "
                       "uniformly spaced)")
    else:
        frame_list += ("uniformly spaced, each moved to the next frame with "
                       "GT in range)")
    return {
        "source":
            f"https://huggingface.co/datasets/{SYNTH4K_REPO}",
        "revision":
            SYNTH4K_REVISION,
        "source_archive":
            tar_path.name,
        "frame_list":
            frame_list,
        "depth_validity":
            "finite and in (3, 150) m, as InfiniDepth's synth4k loader",
        "note": ("Game-captured RGB-D (ReShade depth buffer, metres); sky "
                 f"(depth >= {SKY_DEPTH[game]:g}) stored as 0. "
                 "No intrinsics."),
    }


def materialize_game(
    *,
    game: str,
    source_root: Path,
    dataset_root: Path,
    frames_per_scene: int,
    overwrite: bool,
    update_split: bool,
    frame_fraction: float = 1.0,
) -> dict:
    """Materializes one game as a scene.

    Args:
        game: Game name, one of `GAMES`.
        source_root: Folder holding the `<game>.tar` archives.
        dataset_root: The output `MonoDepth/synth4k` folder.
        frames_per_scene: Frames to select, or 0 to use `frame_fraction`.
        overwrite: Whether existing output files are rewritten.
        update_split: Whether to reselect and rewrite the committed frame list.
        frame_fraction: Share of the scorable frames kept.

    Returns:
        The scene's metadata.json entry.

    Raises:
        FileNotFoundError: If the game's tar is missing.
        ValueError: If committed frames are missing from the tar.
    """
    tar_path = source_root / f"{game}.tar"
    if not tar_path.is_file():
        raise FileNotFoundError(f"Synth4K tar not found: {tar_path}")
    scene_root = dataset_root / game
    with tarfile.open(tar_path, "r") as tar:
        frames = index_members(tar)
        complete = sorted(frame_id for frame_id, kinds in frames.items()
                          if {"image", "depth", "highfreq_mask"} <= set(kinds))
        selected = monodepth_layout.read_split(DATASET, game)
        if update_split or selected is None:
            selected = select_scorable(tar, frames, complete, frames_per_scene,
                                       frame_fraction)
            monodepth_layout.write_split(DATASET, game, selected)
            remove_stale_frames(
                scene_root,
                {monodepth_layout.frame_key(frame_id) for frame_id in selected})
        missing = sorted(set(selected) - set(complete))
        if missing:
            raise ValueError(
                f"{game}: {len(missing)} committed frames are not in "
                f"{tar_path}, e.g. {missing[:3]}")

        records = []
        for frame_id in selected:
            records.append(
                _write_frame(tar, frames[frame_id], scene_root, frame_id, game,
                             overwrite))
    provenance = _game_provenance(game, tar_path, len(records), len(complete),
                                  frames_per_scene, frame_fraction)
    return monodepth_layout.write_scene(
        dataset_root=dataset_root,
        dataset=DATASET,
        scene=game,
        records=records,
        limits=LIMITS,
        provenance=provenance,
    )


def main() -> None:
    """Materializes the selected games."""
    args = parse_args()
    dataset_root = args.benchmark_root.expanduser() / "MonoDepth" / DATASET
    entries = []
    for game in args.games:
        entry = materialize_game(
            game=game,
            source_root=args.source_root.expanduser(),
            dataset_root=dataset_root,
            frames_per_scene=args.frames_per_scene,
            overwrite=args.overwrite,
            update_split=args.write_split,
            frame_fraction=args.frame_fraction,
        )
        print(f"{DATASET}/{game}: {entry['total_images']} frames", flush=True)
        entries.append(entry)
    metadata = monodepth_layout.write_dataset_metadata(dataset_root, DATASET,
                                                       entries)
    print(f"{DATASET}: {metadata['total_images']} frames in "
          f"{len(metadata['scenes'])} scene(s)")


if __name__ == "__main__":
    main()
