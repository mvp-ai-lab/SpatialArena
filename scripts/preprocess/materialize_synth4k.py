#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Materialize Synth4K (InfiniDepth's 4K game benchmark) as the MonoDepth dataset ``synth4k``.

Source: one tar per game from HF dataset ritianyu/game_4k_data (scripts/download/synth4k.sh), laid out as
``./<game>/<sequence>/{image/<frame>.png, depth/<frame>.exr, highfreq_mask/<frame>_highfreq_mask.png, gt_pose.csv}``.
Each game becomes one scene (cyberpunk, deadisland, spiderman2, spidermanmm, watchdoglegion).

InfiniDepth publishes no evaluation frame list, so each scene keeps ``--frame_fraction`` (default 0.5, half) of the
``<game>/<sequence>/<frame>`` ids that have an image, a depth map, a high-frequency mask and some GT in the evaluated
range, spread uniformly over them. ``--frames_per_scene N`` (N > 0) instead spreads N frames uniformly over the sorted
complete ids, a frame without any GT in range being replaced by the next such id (the 100-frame selection of earlier
releases). The selection is committed as
``spatialarena/eval/monodepth_splits/synth4k/<game>.txt`` and reused on later runs.

Per frame: the 3840x2160 RGBA PNG is stored as RGB PNG (lossless); the EXR depth (metres, game far plane for sky)
as float32 .npy with non-finite values and the sky (depth at or beyond the game's far plane, ``SKY_DEPTH``) set
to 0; the high-frequency mask (InfiniDepth's fine-detail region, not
used by the default affine protocol) under ``highfreq_masks/``. The scene limits follow InfiniDepth's loader
(``training/dataset/depth_estimation/synth4k.py``): depth in (3, 150) m is evaluated. No intrinsics are written:
InfiniDepth evaluates Synth4K without them and the capture's ``fov`` column is not documented.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tarfile
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.eval.dataset_registry import DEFAULT_BENCHMARK_ROOT  # noqa: E402
from scripts.preprocess.monodepth_layout import (  # noqa: E402
    FrameRecord,
    SceneLimits,
    frame_key,
    read_split,
    save_depth,
    fraction_count,
    uniform_subsample,
    write_dataset_metadata,
    write_scene,
    write_split,
)

DATASET = "synth4k"
GAMES = ("cyberpunk", "deadisland", "spiderman2", "spidermanmm", "watchdoglegion")
DEFAULT_SOURCE_ROOT = DEFAULT_BENCHMARK_ROOT / "MonoDepth_source" / "synth4k"
SYNTH4K_REPO = "ritianyu/game_4k_data"
SYNTH4K_REVISION = "9d5fc8d41dd73304240eff5eaa15ba002753b990"
LIMITS = SceneLimits(min_depth=3.0, max_depth=150.0)
# Sky: depth at or beyond the game's far plane is stored as 0 (invalid). Four games write one constant far-plane
# value; Dead Island's sky spans ~2e5-6e5 with nothing between 5e4 and 2e5, so its cut-off is 1e5.
SKY_DEPTH = {
    "cyberpunk": 17152.0,
    "deadisland": 1e5,
    "spiderman2": 17723.0,
    "spidermanmm": 10039.2,
    "watchdoglegion": 9000.0,
}
MEMBER_PATTERN = re.compile(
    r"^(?:\./)?(?P<game>[^/]+)/(?P<sequence>[^/]+)/(?P<kind>image|depth|highfreq_mask)/"
    r"(?P<frame>\d+)(?:_highfreq_mask)?\.(?P<ext>png|exr)$"
)


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES))
    parser.add_argument(
        "--frames_per_scene", type=int, default=0, help="Frames per game, uniformly spaced (0: use --frame_fraction)."
    )
    parser.add_argument(
        "--frame_fraction", type=float, default=0.5, help="Share of a game's scorable frames kept, uniformly spaced."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write_split",
        action="store_true",
        help="Reselect the frames and rewrite the committed frame list instead of reusing it.",
    )
    return parser.parse_args()


def index_members(tar: tarfile.TarFile) -> dict[str, dict[str, tarfile.TarInfo]]:
    """``<game>/<sequence>/<frame>`` -> {"image"|"depth"|"highfreq_mask": member}."""
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
    """Bytes of one tar member."""
    extracted = tar.extractfile(member)
    if extracted is None:
        raise ValueError(f"Could not extract {member.name}")
    return extracted.read()


def decode_exr_depth(content: bytes) -> np.ndarray:
    """Decode a single-channel EXR depth map (metres) to float32."""
    depth = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError("Could not decode EXR depth")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def is_scorable(depth: np.ndarray) -> bool:
    """Whether any GT pixel lies inside the evaluated depth range."""
    return bool((np.isfinite(depth) & (depth > LIMITS.min_depth) & (depth < LIMITS.max_depth)).any())


def select_scorable(
    tar: tarfile.TarFile,
    frames: dict[str, dict[str, tarfile.TarInfo]],
    complete: list[str],
    count: int,
    fraction: float = 1.0,
) -> list[str]:
    """``count`` frames uniformly spaced over ``complete``, each moved forward to the next unselected frame that has
    GT inside the evaluated depth range (frames entirely closer than 3 m or beyond 150 m cannot be scored); when
    ``count`` is 0, ``fraction`` of the frames with such GT, uniformly spaced over them."""
    if count <= 0:
        scorable = [frame_id for frame_id in complete
                    if is_scorable(decode_exr_depth(read_member(tar, frames[frame_id]["depth"])))]
        return uniform_subsample(scorable, fraction_count(len(scorable), fraction))
    selected: list[str] = []
    taken: set[str] = set()
    for frame_id in uniform_subsample(complete, count):
        position = complete.index(frame_id)
        while position < len(complete):
            candidate = complete[position]
            if candidate not in taken and is_scorable(decode_exr_depth(read_member(tar, frames[candidate]["depth"]))):
                selected.append(candidate)
                taken.add(candidate)
                break
            position += 1
    return sorted(selected)


def remove_stale_frames(scene_root: Path, keys: set[str]) -> None:
    """Delete materialized files of frames that are no longer in the scene's frame list."""
    for folder in ("images", "depths", "highfreq_masks"):
        for path in sorted((scene_root / folder).glob("*")) if (scene_root / folder).is_dir() else []:
            key = path.stem
            if key not in keys:
                path.unlink()


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
    """Materialize one game as a scene and return its metadata entry."""
    tar_path = source_root / f"{game}.tar"
    if not tar_path.is_file():
        raise FileNotFoundError(f"Synth4K tar not found: {tar_path}")
    scene_root = dataset_root / game
    with tarfile.open(tar_path, "r") as tar:
        frames = index_members(tar)
        complete = sorted(
            frame_id for frame_id, kinds in frames.items() if {"image", "depth", "highfreq_mask"} <= set(kinds)
        )
        selected = read_split(DATASET, game)
        if update_split or selected is None:
            selected = select_scorable(tar, frames, complete, frames_per_scene, frame_fraction)
            write_split(DATASET, game, selected)
            remove_stale_frames(scene_root, {frame_key(frame_id) for frame_id in selected})
        missing = sorted(set(selected) - set(complete))
        if missing:
            raise ValueError(f"{game}: {len(missing)} committed frames are not in {tar_path}, e.g. {missing[:3]}")

        records = []
        for frame_id in selected:
            kinds = frames[frame_id]
            key = frame_key(frame_id)
            image_rel, depth_rel = f"images/{key}.png", f"depths/{key}.npy"
            highfreq_rel = f"highfreq_masks/{key}.png"
            if overwrite or not (scene_root / image_rel).is_file():
                with Image.open(io.BytesIO(read_member(tar, kinds["image"]))) as image:
                    (scene_root / image_rel).parent.mkdir(parents=True, exist_ok=True)
                    image.convert("RGB").save(scene_root / image_rel, compress_level=6)
            if overwrite or not (scene_root / depth_rel).is_file():
                depth = decode_exr_depth(read_member(tar, kinds["depth"]))
                save_depth(scene_root / depth_rel, depth, valid=depth < SKY_DEPTH[game] * 0.999)
            if overwrite or not (scene_root / highfreq_rel).is_file():
                (scene_root / highfreq_rel).parent.mkdir(parents=True, exist_ok=True)
                (scene_root / highfreq_rel).write_bytes(read_member(tar, kinds["highfreq_mask"]))
            records.append(
                FrameRecord(
                    key=key,
                    image=image_rel,
                    depth=depth_rel,
                    source=frame_id,
                    extra={"highfreq_mask": highfreq_rel},
                )
            )
    provenance = {
        "source": f"https://huggingface.co/datasets/{SYNTH4K_REPO}",
        "revision": SYNTH4K_REVISION,
        "source_archive": tar_path.name,
        "frame_list": (
            f"spatialarena/eval/monodepth_splits/{DATASET}/{game}.txt ({len(records)} of the {len(complete)} "
            "sorted complete frames: "
            + (f"{frame_fraction:g} of the frames with GT in range, uniformly spaced)" if frames_per_scene <= 0 else
               "uniformly spaced, each moved to the next frame with GT in range)")
        ),
        "depth_validity": "finite and in (3, 150) m, as InfiniDepth's synth4k loader",
        "note": (
            f"Game-captured RGB-D (ReShade depth buffer, metres); sky (depth >= {SKY_DEPTH[game]:g}) stored as 0. "
            "No intrinsics."
        ),
    }
    return write_scene(
        dataset_root=dataset_root,
        dataset=DATASET,
        scene=game,
        records=records,
        limits=LIMITS,
        provenance=provenance,
    )


def main() -> None:
    """Materialize the selected games."""
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
    metadata = write_dataset_metadata(dataset_root, DATASET, entries)
    print(f"{DATASET}: {metadata['total_images']} frames in {len(metadata['scenes'])} scene(s)")


if __name__ == "__main__":
    main()
