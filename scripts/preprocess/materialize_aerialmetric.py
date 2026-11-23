#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Materialize the AerialMetric Test-Split as MonoDepth dataset `aerialmetric`.

Source: the extracted Test-Split archives of HF dataset
Kuiee/AerialMetric-ECCV2026 (scripts/download/aerialmetric.sh). Three scenes,
each keeping `--frame_fraction` (default 0.5, half) of its Test-Split frames
with GT in the evaluated depth range, uniformly spaced over the sorted sample
ids (per site for `decoupled`). `--frames_per_scene N` > 0 instead keeps N
uniformly spaced frames per scene (the 100-frame selection of earlier
releases):

- `decoupled`: the four Decoupled sites
  (`decoupled-norm/Cleaned_Dataset_{Campus,Factory,Farm,Gress}/<id>/`,
  image.jpg + depth.npy + meta.json with UV-normalized intrinsics; with
  `--frames_per_scene` an equal share of frames per site);
- `oblique`: the samples of `Oblique-norm/val_index.txt` (same norm layout;
  the Train-Split Oblique data is training data of MoGe2-Aerial and is not
  used);
- `wild`: frames `Wild/<video scene>/image/<stem>.jpg` +
  `depth/<stem>.npy` (low-resolution depth, the prediction is resized to it),
  with per-frame COLMAP intrinsics from `metadata_full.csv`.

A frame without any GT in the evaluated depth range is never kept (two Wild
videos lie entirely beyond 400 m); with `--frames_per_scene` frames are spread
uniformly over the sorted sample ids (per site for `decoupled`) and such a
frame is replaced by the next sorted id that has some.

Depth is metric (metres) and stored as float32 .npy with non-finite /
non-positive values set to 0. The manually annotated dynamic-object masks
(`*-masks`, white = invalid) are stored inverted under `masks/`
(255 = evaluate) for the frames that have one. The scene limits follow
AerialMetric's evaluation code (`MoGe/moge/scripts/code-final/c-eval-*.py`:
GT in (1e-3, 400) m). Frame lists are committed under
`spatialarena/eval/monodepth_splits/aerialmetric/<scene>.txt`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from concurrent import futures
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from scripts.preprocess import monodepth_layout  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402
# pylint: enable=wrong-import-position

DATASET = "aerialmetric"
DEFAULT_SOURCE_ROOT = (dataset_registry.DEFAULT_BENCHMARK_ROOT /
                       "MonoDepth_source" / "aerialmetric" / "extracted")
AERIALMETRIC_REPO = "Kuiee/AerialMetric-ECCV2026"
AERIALMETRIC_REVISION = "c8fb8ea68fba6a3bace5814df26c248995aef8a1"
LIMITS = monodepth_layout.SceneLimits(min_depth=1e-3, max_depth=400.0)
# The source folder is spelled "Gress"; AerialMetric's flight-parameter CSV
# calls it "grass".
DECOUPLED_SITES = ("Cleaned_Dataset_Campus", "Cleaned_Dataset_Factory",
                   "Cleaned_Dataset_Farm", "Cleaned_Dataset_Gress")
SCENES = ("decoupled", "oblique", "wild")


def parse_args() -> argparse.Namespace:
    """Parses the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--benchmark_root",
                        type=Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--scenes",
                        nargs="+",
                        choices=SCENES,
                        default=list(SCENES))
    parser.add_argument(
        "--frames_per_scene",
        type=int,
        default=0,
        help="Frames per scene, uniformly spaced (0: use --frame_fraction).")
    parser.add_argument(
        "--frame_fraction",
        type=float,
        default=0.5,
        help="Share of a scene's scorable frames kept, uniformly spaced.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write_split",
        action="store_true",
        help=("Reselect the frames and rewrite the committed frame lists "
              "instead of reusing them."),
    )
    return parser.parse_args()


def norm_samples(root: Path) -> list[str]:
    """Returns the sample folders directly under `root`, sorted.

    A sample folder is an `<id>` folder with image.jpg and depth.npy.
    """
    return sorted(
        path.name
        for path in root.iterdir()
        if (path / "image.jpg").is_file() and (path / "depth.npy").is_file())


def scene_groups(source_root: Path, scene: str) -> list[list[str]]:
    """Returns the candidate source ids of a scene in groups.

    Each group gets an equal share of frames; only `decoupled` has more than
    one group (its sites).
    """
    if scene == "decoupled":
        groups = []
        for site in DECOUPLED_SITES:
            samples = norm_samples(source_root / "decoupled-norm" / site)
            groups.append([f"{site}/{sample}" for sample in samples])
        return groups
    return [scene_sources(source_root, scene)]


def scene_sources(source_root: Path, scene: str) -> list[str]:
    """Returns the source sample ids of a scene.

    The ids are relative to the scene's source folder.
    """
    if scene == "decoupled":
        return [
            source for group in scene_groups(source_root, scene)
            for source in group
        ]
    if scene == "oblique":
        index_path = source_root / "Oblique-norm" / "val_index.txt"
        return sorted(
            line.strip()
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip())
    sources = []
    for scene_dir in sorted(
            path for path in (source_root / "Wild").iterdir() if path.is_dir()):
        for image_path in sorted((scene_dir / "image").glob("*.jpg")):
            if (scene_dir / "depth" / f"{image_path.stem}.npy").is_file():
                sources.append(f"{scene_dir.name}/{image_path.stem}")
    return sources


def wild_intrinsics(scene_dir: Path) -> dict[str, list[list[float]]]:
    """Returns the per-image pixel intrinsics of one Wild scene.

    They are read from metadata_full.csv and keyed by image stem.
    """
    result = {}
    csv_path = scene_dir / "metadata_full.csv"
    if not csv_path.is_file():
        return result
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            fx, fy, cx, cy = (
                float(row[key]) for key in ("fx", "fy", "cx", "cy"))
            result[Path(row["filename_img"]).stem] = [[fx, 0.0, cx],
                                                      [0.0, fy, cy],
                                                      [0.0, 0.0, 1.0]]
    return result


def _norm_folder(scene: str) -> str:
    """Returns the norm-layout source folder of a Decoupled / Oblique scene."""
    if scene == "decoupled":
        return "decoupled-norm"
    return "Oblique-norm"


def source_depth(source_root: Path, scene: str, source: str) -> Path:
    """Returns the GT depth file of a source sample."""
    if scene == "wild":
        folder, sample = source.split("/", 1)
        return source_root / "Wild" / folder / "depth" / f"{sample}.npy"
    return source_root / _norm_folder(scene) / source / "depth.npy"


def has_gt_in_range(depth_path: Path) -> bool:
    """Returns whether any GT pixel lies inside the evaluated depth range."""
    depth = np.load(depth_path)
    return bool((np.isfinite(depth) & (depth > LIMITS.min_depth) &
                 (depth < LIMITS.max_depth)).any())


def select_frames(source_root: Path,
                  scene: str,
                  count: int,
                  fraction: float = 1.0) -> list[str]:
    """Selects the source ids of a scene.

    With `count` > 0, `count` ids are spread uniformly over each group's
    sorted ids (an equal share per group), each moved forward to the next
    unselected id with GT in range. When `count` is 0, `fraction` of each
    group's ids with GT in range are kept, uniformly spaced over them.

    Args:
        source_root: The extracted AerialMetric Test-Split.
        scene: Scene name, one of `SCENES`.
        count: Frames per scene, or 0 to use `fraction`.
        fraction: Share of the scorable ids kept when `count` is 0.

    Returns:
        The selected source ids, sorted.
    """
    groups = scene_groups(source_root, scene)
    if count <= 0:
        selected = []
        for group in groups:
            scorable = [
                source for source in group
                if has_gt_in_range(source_depth(source_root, scene, source))
            ]
            selected.extend(
                monodepth_layout.uniform_subsample(
                    scorable,
                    monodepth_layout.fraction_count(len(scorable), fraction)))
        return sorted(selected)
    selected: list[str] = []
    for index, group in enumerate(groups):
        share = count // len(groups) + (1 if index < count % len(groups) else 0)
        taken: set[str] = set()
        for source in monodepth_layout.uniform_subsample(group, share):
            position = group.index(source)
            while position < len(group):
                candidate = group[position]
                if candidate not in taken and has_gt_in_range(
                        source_depth(source_root, scene, candidate)):
                    taken.add(candidate)
                    break
                position += 1
        selected.extend(sorted(taken))
    return sorted(selected)


def mask_source(source_root: Path, scene: str, source: str) -> Path | None:
    """Returns the dynamic-object mask of a Decoupled / Oblique sample.

    The mask is white where invalid. Returns None if the sample has no
    annotated mask.
    """
    folder, sample = source.split("/", 1)
    if scene == "decoupled":
        path = (source_root / "decoupled-masks" / f"{folder}-mask" /
                f"{sample}.png")
    elif scene == "oblique":
        path = (source_root / "Oblique-masks" / f"{folder}-mask" /
                f"{sample}.png")
    else:
        return None
    return path if path.is_file() else None


def materialize_frame(job: dict) -> dict:
    """Writes one frame (image, depth, optional inverted mask).

    Args:
        job: A job built by `frame_jobs`.

    Returns:
        The frame's manifest record.
    """
    scene_root = Path(job["scene_root"])
    key = job["key"]
    image_src = Path(job["image"])
    depth_src = Path(job["depth"])
    image_rel = f"images/{key}.jpg"
    depth_rel = f"depths/{key}.npy"
    with Image.open(image_src) as image:
        width, height = image.size
    if job["overwrite"] or not (scene_root / image_rel).is_file():
        monodepth_layout.write_bytes(scene_root / image_rel,
                                     image_src.read_bytes())
    depth = np.load(depth_src).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if job["overwrite"] or not (scene_root / depth_rel).is_file():
        monodepth_layout.save_depth(scene_root / depth_rel, depth)
    in_range = int((np.isfinite(depth) & (depth > LIMITS.min_depth) &
                    (depth < LIMITS.max_depth)).sum())
    mask_rel = None
    if job["mask"] is not None:
        mask_rel = f"masks/{key}.png"
        if job["overwrite"] or not (scene_root / mask_rel).is_file():
            with Image.open(job["mask"]) as mask_image:
                invalid = mask_image.convert("L").resize(
                    (depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST)
            monodepth_layout.save_mask(scene_root / mask_rel,
                                       np.asarray(invalid) <= 127)
    intrinsics = job["intrinsics"]
    if job["normalized_intrinsics"]:
        intrinsics = monodepth_layout.normalized_to_pixel_intrinsics(
            np.asarray(intrinsics), width, height)
    return monodepth_layout.FrameRecord(
        key=key,
        image=image_rel,
        depth=depth_rel,
        source=job["source"],
        intrinsics=intrinsics,
        mask=mask_rel,
        extra={
            "depth_size": [int(depth.shape[1]),
                           int(depth.shape[0])],
            "num_in_range_pixels": in_range
        },
    ).as_dict()


def _frame_inputs(source_root: Path, scene: str, source: str,
                  wild_cache: dict[str, dict]) -> tuple:
    """Returns the key, image, depth and intrinsics of one source sample.

    Args:
        source_root: The extracted AerialMetric Test-Split.
        scene: Scene name.
        source: Source sample id.
        wild_cache: Per-folder Wild intrinsics, filled as folders are seen.

    Returns:
        A (key, image path, depth path, intrinsics, normalized) tuple, where
        `normalized` tells whether the intrinsics are UV-normalized.
    """
    folder, sample = source.split("/", 1)
    if scene == "wild":
        scene_dir = source_root / "Wild" / folder
        if folder not in wild_cache:
            wild_cache[folder] = wild_intrinsics(scene_dir)
        # Wild stems already embed the video and scene name, so they are
        # unique on their own.
        key = sample
        image = scene_dir / "image" / f"{sample}.jpg"
        depth = scene_dir / "depth" / f"{sample}.npy"
        intrinsics = wild_cache[folder].get(sample)
        normalized = False
    else:
        sample_dir = source_root / _norm_folder(scene) / source
        key = monodepth_layout.frame_key(source)
        image = sample_dir / "image.jpg"
        depth = sample_dir / "depth.npy"
        meta = json.loads(
            (sample_dir / "meta.json").read_text(encoding="utf-8"))
        intrinsics = meta["intrinsics"]
        normalized = True
    return key, image, depth, intrinsics, normalized


def frame_jobs(source_root: Path, scene: str, sources: list[str],
               scene_root: Path, overwrite: bool) -> list[dict]:
    """Builds one picklable job per source sample of a scene.

    Raises:
        ValueError: If two samples map to the same frame key.
    """
    jobs = []
    wild_cache: dict[str, dict] = {}
    for source in sources:
        key, image, depth, intrinsics, normalized = _frame_inputs(
            source_root, scene, source, wild_cache)
        mask = mask_source(source_root, scene, source)
        jobs.append({
            "scene_root": str(scene_root),
            "key": key,
            "source": source,
            "image": str(image),
            "depth": str(depth),
            "mask": str(mask) if mask is not None else None,
            "intrinsics": intrinsics,
            "normalized_intrinsics": normalized,
            "overwrite": overwrite,
        })
    keys = [job["key"] for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{scene}: duplicate frame keys")
    return jobs


def _frame_list_note(scene: str, num_frames: int, frames_per_scene: int,
                     frame_fraction: float) -> str:
    """Returns the provenance description of a scene's frame list."""
    note = (f"spatialarena/eval/monodepth_splits/{DATASET}/{scene}.txt "
            f"({num_frames} frames: ")
    if frames_per_scene <= 0:
        note += (f"{frame_fraction:g} of the sample ids with GT in range, "
                 "uniformly spaced")
        if scene == "decoupled":
            note += " per site"
        return note + ")"
    note += "uniformly spaced over the sorted sample ids"
    if scene == "decoupled":
        note += ", an equal share per site"
    return note + ", each moved to the next id with GT in range)"


def _scene_provenance(scene: str, num_frames: int, frames_per_scene: int,
                      frame_fraction: float) -> dict:
    """Returns the provenance recorded for one scene."""
    source_folder = {
        "decoupled": "Test-Split/AerialMetric-Decoupled-norm (all four sites)",
        "oblique": "Test-Split/AerialMetric-Oblique-norm (val_index.txt)",
        "wild": "Test-Split/AerialMetric-Wild",
    }[scene]
    return {
        "source":
            f"https://huggingface.co/datasets/{AERIALMETRIC_REPO}",
        "revision":
            AERIALMETRIC_REVISION,
        "source_folder":
            source_folder,
        "frame_list":
            _frame_list_note(scene, num_frames, frames_per_scene,
                             frame_fraction),
        "depth_validity":
            ("finite and in (1e-3, 400) m, as AerialMetric's evaluation; "
             "dynamic-object masks applied"),
    }


def materialize_scene(
    *,
    scene: str,
    source_root: Path,
    dataset_root: Path,
    frames_per_scene: int,
    workers: int,
    overwrite: bool,
    update_split: bool,
    frame_fraction: float = 1.0,
) -> dict:
    """Materializes one scene from its committed (or newly selected) frames.

    Args:
        scene: Scene name, one of `SCENES`.
        source_root: The extracted AerialMetric Test-Split.
        dataset_root: The output `MonoDepth/aerialmetric` folder.
        frames_per_scene: Frames to select, or 0 to use `frame_fraction`.
        workers: Number of worker processes.
        overwrite: Whether existing output files are rewritten.
        update_split: Whether to reselect and rewrite the committed frame list.
        frame_fraction: Share of the scorable frames kept.

    Returns:
        The scene's metadata.json entry.

    Raises:
        ValueError: If committed frames are missing from the source or a
            frame has no GT in range.
    """
    sources = monodepth_layout.read_split(DATASET, scene)
    if update_split or sources is None:
        sources = select_frames(source_root, scene, frames_per_scene,
                                frame_fraction)
        monodepth_layout.write_split(DATASET, scene, sources)
    missing = sorted(set(sources) - set(scene_sources(source_root, scene)))
    if missing:
        raise ValueError(
            f"{scene}: {len(missing)} committed frames are missing from the "
            f"source, e.g. {missing[:3]}; "
            "rerun with --write_split to reselect")
    scene_root = dataset_root / scene
    jobs = frame_jobs(source_root, scene, sources, scene_root, overwrite)
    remove_stale_frames(scene_root, {job["key"] for job in jobs})
    with futures.ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(materialize_frame, jobs, chunksize=8))
    unscorable = [
        record["source"]
        for record in records
        if record["num_in_range_pixels"] == 0
    ]
    if unscorable:
        raise ValueError(
            f"{scene}: frames without GT in range: {unscorable[:3]}")
    frame_records = [
        monodepth_layout.FrameRecord(
            key=record.pop("key"),
            image=record.pop("image"),
            depth=record.pop("depth"),
            source=record.pop("source"),
            intrinsics=record.pop("intrinsics", None),
            mask=record.pop("mask", None),
            extra=record,
        ) for record in records
    ]
    provenance = _scene_provenance(scene, len(frame_records), frames_per_scene,
                                   frame_fraction)
    return monodepth_layout.write_scene(
        dataset_root=dataset_root,
        dataset=DATASET,
        scene=scene,
        records=frame_records,
        limits=LIMITS,
        provenance=provenance,
    )


def remove_stale_frames(scene_root: Path, keys: set[str]) -> None:
    """Deletes the files of frames no longer in the scene's frame list."""
    for folder in ("images", "depths", "masks"):
        if (scene_root / folder).is_dir():
            for path in (scene_root / folder).iterdir():
                if path.stem not in keys:
                    path.unlink()


def remove_stale_scenes(dataset_root: Path) -> None:
    """Deletes scene folders of an older scene layout.

    For example the per-site `decoupled_<site>` scenes.
    """
    if not dataset_root.is_dir():
        return
    for path in dataset_root.iterdir():
        if path.is_dir() and path.name not in SCENES and (
                path / "materialized_full.json").is_file():
            shutil.rmtree(path)


def main() -> None:
    """Materializes the selected scenes."""
    args = parse_args()
    dataset_root = args.benchmark_root.expanduser() / "MonoDepth" / DATASET
    remove_stale_scenes(dataset_root)
    entries = []
    for scene in args.scenes:
        entry = materialize_scene(
            scene=scene,
            source_root=args.source_root.expanduser(),
            dataset_root=dataset_root,
            frames_per_scene=args.frames_per_scene,
            workers=args.workers,
            overwrite=args.overwrite,
            update_split=args.write_split,
            frame_fraction=args.frame_fraction,
        )
        print(f"{DATASET}/{scene}: {entry['total_images']} frames", flush=True)
        entries.append(entry)
    metadata = monodepth_layout.write_dataset_metadata(dataset_root,
                                                       DATASET,
                                                       entries,
                                                       keep_scenes=SCENES)
    print(f"{DATASET}: {metadata['total_images']} frames in "
          f"{len(metadata['scenes'])} scene(s)")


if __name__ == "__main__":
    main()
