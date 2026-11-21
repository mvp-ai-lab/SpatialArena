# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Consolidate TexturePoorSfM bags into per-object pools and re-split train/test by object.

TexturePoorSfM ships as ``{object}__{5|10|20}bag{NNN}`` bags: small COLMAP models whose frames are
re-sampled from one capture per object. Bags of the same object share frames (byte-identical images,
identical poses in a shared per-object world frame), so a split by bag leaks test frames into training.

This script:

1. ``consolidate``: reads every bag from one or more bag roots, checks that each (object, frame) has the
   same image bytes, pose and intrinsics in every bag it appears in, and writes one pool per object::

       <pool_root>/<object>/images/<frame>.png
       <pool_root>/<object>/sparse/0/{cameras,images,points3D}.txt   # merged model, one camera per image
       <pool_root>/<object>/bags.json                               # bag name -> frame names
       <pool_root>/manifest.json

2. ``benchmark``: materializes the held-out benchmark bags listed in the split manifest from the pool into
   ``<benchmark_root>/SfM/texturepoorsfm`` (same per-bag layout as before, ``images/`` + ``sparse/0``).

The split manifest (``scripts/preprocess/assets/texturepoorsfm_split_v2.json``) assigns whole objects to
train / val / test, so no frame of a test object can reach training.

Pose convention: TexturePoorSfM's raw pose files are world-to-camera, but the bags in ``texturepoorsfm_train``
and the v1 benchmark were materialized by a version of ``materialize_texturepoorsfm.py`` that inverted them, so
their COLMAP models hold camera-to-world matrices in the world-to-camera slot. ``--source_pose_convention
inverted_w2c`` undoes that on read (verified by epipolar consistency with LoFTR matches); bags materialized by
the fixed script use ``w2c``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.geometry.read_write_model import (  # noqa: E402
    BaseImage,
    Camera,
    qvec2rotmat,
    read_model,
    rotmat2qvec,
    write_model,
)

BAG_PATTERN = re.compile(r"^(?P<object>\d+)__(?P<size>\d+)bag(?P<index>\d+)$")
DEFAULT_SPLIT_PATH = REPO_ROOT / "scripts" / "preprocess" / "assets" / "texturepoorsfm_split_v2.json"
POSE_ATOL = 1e-6
INTRINSIC_ATOL = 1e-4
POSE_CONVENTIONS = ("w2c", "inverted_w2c")


@dataclass(frozen=True)
class FrameRecord:
    """One frame as observed in one bag."""

    object_id: str
    name: str
    image_path: Path
    md5: str
    qvec: tuple[float, ...]
    tvec: tuple[float, ...]
    width: int
    height: int
    params: tuple[float, ...]  # PINHOLE fx, fy, cx, cy
    bag: str


def parse_bag_name(name: str) -> tuple[str, int, int] | None:
    """Return ``(object_id, bag_size, bag_index)`` for a bag directory name, or None."""
    match = BAG_PATTERN.match(name)
    if match is None:
        return None
    return match["object"], int(match["size"]), int(match["index"])


def frame_sort_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    return (int(stem), name) if stem.isdigit() else (sys.maxsize, name)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_bags(bag_roots: list[Path]) -> dict[str, Path]:
    """Map bag name -> bag directory across all roots; a bag name present in two roots is an error."""
    bags: dict[str, Path] = {}
    for root in bag_roots:
        for path in sorted(root.iterdir()):
            if not path.is_dir() or parse_bag_name(path.name) is None:
                continue
            if path.name in bags:
                raise ValueError(f"Bag {path.name} found in both {bags[path.name]} and {path}")
            bags[path.name] = path
    return bags


def to_world_to_camera(qvec, tvec, convention: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the true COLMAP (qvec, tvec) for a stored pose written under ``convention``."""
    qvec, tvec = np.asarray(qvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64)
    if convention == "w2c":
        return qvec, tvec
    if convention != "inverted_w2c":
        raise ValueError(f"Unknown pose convention {convention!r}; expected one of {POSE_CONVENTIONS}")
    rotation = qvec2rotmat(qvec)
    return rotmat2qvec(rotation.T), -rotation.T @ tvec


def read_bag_frames(bag_dir: Path, pose_convention: str = "w2c") -> list[FrameRecord]:
    parsed = parse_bag_name(bag_dir.name)
    if parsed is None:
        raise ValueError(f"Not a TexturePoorSfM bag directory: {bag_dir}")
    object_id = parsed[0]
    cameras, images, _points = read_model(str(bag_dir / "sparse" / "0"), ext=".txt")
    records = []
    for image in images.values():
        camera = cameras[image.camera_id]
        if camera.model != "PINHOLE":
            raise ValueError(f"{bag_dir}: unsupported camera model {camera.model} for {image.name}")
        image_path = bag_dir / "images" / image.name
        if not image_path.is_file():
            raise FileNotFoundError(f"{bag_dir}: missing image file for {image.name}")
        qvec, tvec = to_world_to_camera(image.qvec, image.tvec, pose_convention)
        records.append(
            FrameRecord(
                object_id=object_id,
                name=image.name,
                image_path=image_path,
                md5=file_md5(image_path),
                qvec=tuple(float(v) for v in qvec),
                tvec=tuple(float(v) for v in tvec),
                width=int(camera.width),
                height=int(camera.height),
                params=tuple(float(v) for v in camera.params),
                bag=bag_dir.name,
            )
        )
    image_files = {path.name for path in (bag_dir / "images").iterdir() if path.is_file()}
    extra = image_files - {record.name for record in records}
    if extra:
        raise ValueError(f"{bag_dir}: images without poses: {sorted(extra)}")
    return records


def _same_rotation(qvec_a: tuple[float, ...], qvec_b: tuple[float, ...]) -> bool:
    a, b = np.asarray(qvec_a), np.asarray(qvec_b)
    return bool(np.allclose(a, b, atol=POSE_ATOL) or np.allclose(a, -b, atol=POSE_ATOL))


def frame_conflicts(reference: FrameRecord, other: FrameRecord) -> list[str]:
    """Return human-readable differences between two observations of the same (object, frame)."""
    problems = []
    if reference.md5 != other.md5:
        problems.append("image bytes differ")
    if not _same_rotation(reference.qvec, other.qvec):
        problems.append("rotation differs")
    if not np.allclose(reference.tvec, other.tvec, atol=POSE_ATOL):
        problems.append("translation differs")
    if (reference.width, reference.height) != (other.width, other.height):
        problems.append("image size differs")
    if not np.allclose(reference.params, other.params, atol=INTRINSIC_ATOL):
        problems.append("intrinsics differ")
    return problems


def merge_frames(records: list[FrameRecord]) -> tuple[dict[str, dict[str, FrameRecord]], list[str]]:
    """Group observations into ``object -> frame name -> record`` and collect cross-bag conflicts."""
    merged: dict[str, dict[str, FrameRecord]] = {}
    conflicts = []
    for record in records:
        frames = merged.setdefault(record.object_id, {})
        reference = frames.get(record.name)
        if reference is None:
            frames[record.name] = record
            continue
        problems = frame_conflicts(reference, record)
        if problems:
            conflicts.append(
                f"{record.object_id}/{record.name}: {', '.join(problems)} ({reference.bag} vs {record.bag})"
            )
    return merged, conflicts


def build_colmap_model(frames: list[FrameRecord]) -> tuple[dict, dict]:
    """Merged COLMAP model with one PINHOLE camera per image, image ids in frame order."""
    cameras, images = {}, {}
    for image_id, record in enumerate(sorted(frames, key=lambda r: frame_sort_key(r.name)), start=1):
        cameras[image_id] = Camera(
            id=image_id,
            model="PINHOLE",
            width=record.width,
            height=record.height,
            params=np.asarray(record.params, dtype=np.float64),
        )
        images[image_id] = BaseImage(
            id=image_id,
            qvec=np.asarray(record.qvec, dtype=np.float64),
            tvec=np.asarray(record.tvec, dtype=np.float64),
            camera_id=image_id,
            name=record.name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )
    return cameras, images


def load_split(path: Path) -> dict:
    split = json.loads(path.read_text(encoding="utf-8"))
    groups = {name: set(split[f"{name}_objects"]) for name in ("train", "val", "test")}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[first] & groups[second]
        if overlap:
            raise ValueError(f"{path}: objects {sorted(overlap)} are in both {first} and {second}")
    for bag in split.get("benchmark_scenes", []):
        parsed = parse_bag_name(bag)
        if parsed is None or parsed[0] not in groups["test"]:
            raise ValueError(f"{path}: benchmark scene {bag} does not belong to a test object")
    return split


def check_split_covers(split: dict, object_ids: set[str]) -> None:
    assigned = set(split["train_objects"]) | set(split["val_objects"]) | set(split["test_objects"])
    missing, unknown = object_ids - assigned, assigned - object_ids
    if missing or unknown:
        raise ValueError(f"Split does not match pooled objects: unassigned={sorted(missing)} unknown={sorted(unknown)}")


def select_benchmark_bags(
    bags_by_object: dict[str, list[str]],
    test_objects: list[str],
    per_size: dict[int, int],
    seed: int,
) -> list[str]:
    """Deterministically sample ``per_size[size]`` bags of each size from every test object."""
    rng = random.Random(seed)
    selected = []
    for object_id in sorted(test_objects):
        by_size: dict[int, list[str]] = {}
        for bag in sorted(bags_by_object[object_id]):
            by_size.setdefault(parse_bag_name(bag)[1], []).append(bag)
        for size, count in sorted(per_size.items()):
            candidates = by_size.get(size, [])
            if len(candidates) < count:
                raise ValueError(f"Object {object_id} has only {len(candidates)} bags of size {size}, need {count}")
            selected.extend(sorted(rng.sample(candidates, count)))
    return selected


def consolidate(
    bag_roots: list[Path], pool_root: Path, split_path: Path, overwrite: bool, source_pose_convention: str
) -> dict:
    split = load_split(split_path)
    bags = discover_bags(bag_roots)
    records: list[FrameRecord] = []
    bag_frames: dict[str, dict[str, list[str]]] = {}
    for bag_name, bag_dir in bags.items():
        bag_records = read_bag_frames(bag_dir, source_pose_convention)
        records.extend(bag_records)
        object_id = parse_bag_name(bag_name)[0]
        names = sorted((record.name for record in bag_records), key=frame_sort_key)
        bag_frames.setdefault(object_id, {})[bag_name] = names

    merged, conflicts = merge_frames(records)
    if conflicts:
        preview = "\n".join(conflicts[:20])
        raise ValueError(f"{len(conflicts)} cross-bag frame conflicts, refusing to consolidate:\n{preview}")
    check_split_covers(split, set(merged))

    if pool_root.exists():
        if not overwrite:
            raise FileExistsError(f"{pool_root} exists; pass --overwrite to rebuild it")
        shutil.rmtree(pool_root)
    staging = pool_root.with_name(pool_root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)

    objects_summary = {}
    for object_id in sorted(merged):
        frames = list(merged[object_id].values())
        object_root = staging / object_id
        (object_root / "images").mkdir(parents=True)
        for record in frames:
            shutil.copy2(record.image_path, object_root / "images" / record.name)
        sparse_dir = object_root / "sparse" / "0"
        sparse_dir.mkdir(parents=True)
        cameras, images = build_colmap_model(frames)
        write_model(cameras, images, {}, str(sparse_dir), ext=".txt")
        (object_root / "bags.json").write_text(
            json.dumps(dict(sorted(bag_frames[object_id].items())), indent=2) + "\n", encoding="utf-8"
        )
        objects_summary[object_id] = {
            "num_frames": len(frames),
            "num_bags": len(bag_frames[object_id]),
            "frames": {record.name: record.md5 for record in sorted(frames, key=lambda r: frame_sort_key(r.name))},
        }

    manifest = {
        "dataset_name": "texturepoorsfm",
        "source_bag_roots": [str(root) for root in bag_roots],
        "split_manifest": str(split_path),
        "split_version": split.get("version"),
        "source_pose_convention": source_pose_convention,
        "pose_convention": "colmap_world_to_camera",
        "num_bags": len(bags),
        "num_observations": len(records),
        "num_unique_frames": sum(item["num_frames"] for item in objects_summary.values()),
        "objects": objects_summary,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.rename(pool_root)
    return {key: manifest[key] for key in ("num_bags", "num_observations", "num_unique_frames")} | {
        "num_objects": len(objects_summary),
        "pool_root": str(pool_root),
    }


def subset_points(
    kept: list, tri_by_name: dict, points3d: dict, min_track_length: int = 2
) -> tuple[dict, dict]:
    """Restrict a triangulated object model to one bag.

    ``kept`` are the bag's GT images (already renumbered 1..N). Returns per-new-id ``(xys, point3D_ids)`` and the
    bag's points: tracks keep only the bag's images, and points seen by fewer than ``min_track_length`` of them
    are dropped (their observations become -1).
    """
    tracks: dict[int, list[tuple[int, int]]] = {}
    observations = {}
    for new_id, image in enumerate(kept, start=1):
        tri = tri_by_name[image.name]
        observations[new_id] = (np.asarray(tri.xys, dtype=np.float64), np.asarray(tri.point3D_ids, dtype=np.int64))
        for idx, pid in enumerate(observations[new_id][1]):
            if pid >= 0 and int(pid) in points3d:
                tracks.setdefault(int(pid), []).append((new_id, idx))
    kept_points = {pid: track for pid, track in tracks.items() if len(track) >= min_track_length}
    out_obs = {}
    for new_id, (xys, pids) in observations.items():
        pids = np.array([pid if int(pid) in kept_points else -1 for pid in pids], dtype=np.int64)
        out_obs[new_id] = (xys, pids)
    out_points = {
        pid: points3d[pid]._replace(
            image_ids=np.array([t[0] for t in track], dtype=np.int64),
            point2D_idxs=np.array([t[1] for t in track], dtype=np.int64),
        )
        for pid, track in kept_points.items()
    }
    return out_obs, out_points


def materialize_benchmark(
    pool_root: Path, benchmark_root: Path, split_path: Path, overwrite: bool, sparse_model: str = ""
) -> dict:
    split = load_split(split_path)
    output_root = benchmark_root / "SfM" / "texturepoorsfm"
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; archive it or pass --overwrite")
        shutil.rmtree(output_root)
    staging = output_root.with_name(output_root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)

    scenes = []
    for bag in split["benchmark_scenes"]:
        object_id = parse_bag_name(bag)[0]
        object_root = pool_root / object_id
        bag_frames = json.loads((object_root / "bags.json").read_text(encoding="utf-8"))
        if bag not in bag_frames:
            raise KeyError(f"Benchmark scene {bag} not found in {object_root / 'bags.json'}")
        frame_names = set(bag_frames[bag])
        cameras, images, _points = read_model(str(object_root / "sparse" / "0"), ext=".txt")
        kept = sorted((image for image in images.values() if image.name in frame_names), key=lambda i: i.id)
        if len(kept) != len(frame_names):
            raise ValueError(f"{bag}: pool model is missing frames {sorted(frame_names - {i.name for i in kept})}")
        observations, bag_points = {}, {}
        if sparse_model:
            _c, tri_images, tri_points = read_model(str(object_root / "sparse" / sparse_model))
            tri_by_name = {Path(image.name).name: image for image in tri_images.values()}
            for image in kept:
                tri = tri_by_name[image.name]
                if not (np.allclose(tri.qvec, image.qvec, atol=1e-6) and np.allclose(tri.tvec, image.tvec, atol=1e-6)):
                    raise ValueError(f"{bag}/{image.name}: {sparse_model} pose differs from GT")
            observations, bag_points = subset_points(kept, tri_by_name, tri_points)

        scene_root = staging / bag
        (scene_root / "images").mkdir(parents=True)
        out_cameras, out_images = {}, {}
        for new_id, image in enumerate(kept, start=1):
            shutil.copy2(object_root / "images" / image.name, scene_root / "images" / image.name)
            out_cameras[new_id] = cameras[image.camera_id]._replace(id=new_id)
            out_images[new_id] = image._replace(id=new_id, camera_id=new_id)
            if new_id in observations:
                xys, pids = observations[new_id]
                out_images[new_id] = out_images[new_id]._replace(xys=xys, point3D_ids=pids)
        sparse_dir = scene_root / "sparse" / "0"
        sparse_dir.mkdir(parents=True)
        write_model(out_cameras, out_images, bag_points, str(sparse_dir), ext=".txt")
        scenes.append(
            {
                "dataset_name": "texturepoorsfm",
                "scene_name": bag,
                "object_id": object_id,
                "root_directory": str(output_root / bag),
                "total_images": len(kept),
                "num_points3d": len(bag_points),
            }
        )

    total_images = sum(scene["total_images"] for scene in scenes)
    metadata = {
        "dataset_name": "texturepoorsfm",
        "root_directory": str(output_root),
        "split_version": split.get("version"),
        "test_objects": sorted(split["test_objects"]),
        "sparse_points_source": f"DetectorFreeSfM triangulation with fixed GT poses (pool sparse/{sparse_model})"
        if sparse_model else None,
        "total_images": total_images,
        "average_images_per_scene": total_images / len(scenes) if scenes else 0.0,
        "scenes": scenes,
    }
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    staging.rename(output_root)
    return {"num_scenes": len(scenes), "total_images": total_images, "output_root": str(output_root)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    cons = sub.add_parser("consolidate", help="Merge bags into per-object pools.")
    cons.add_argument("--bag_roots", type=Path, nargs="+", required=True)
    cons.add_argument("--pool_root", type=Path, required=True)
    cons.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    cons.add_argument("--overwrite", action="store_true")
    cons.add_argument(
        "--source_pose_convention",
        choices=POSE_CONVENTIONS,
        required=True,
        help="How the bags' COLMAP poses were written: w2c, or inverted_w2c for bags from the pre-fix materializer.",
    )

    bench = sub.add_parser("benchmark", help="Materialize held-out benchmark bags from the pool.")
    bench.add_argument("--pool_root", type=Path, required=True)
    bench.add_argument("--benchmark_root", type=Path, required=True)
    bench.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    bench.add_argument("--overwrite", action="store_true")
    bench.add_argument(
        "--sparse_model",
        default="dfsfm_tri",
        help="Pool model whose points are copied into each bag (tracks restricted to the bag); '' for no points.",
    )

    select = sub.add_parser("select", help="Print a deterministic benchmark bag selection for the split.")
    select.add_argument("--pool_root", type=Path, required=True)
    select.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    select.add_argument("--per_size", type=str, default="5:2,10:2,20:2", help="bag_size:count pairs")
    select.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "consolidate":
        summary = consolidate(args.bag_roots, args.pool_root, args.split, args.overwrite, args.source_pose_convention)
    elif args.command == "benchmark":
        summary = materialize_benchmark(
            args.pool_root, args.benchmark_root, args.split, args.overwrite, args.sparse_model
        )
    else:
        split = load_split(args.split)
        bags_by_object = {
            object_id: list(json.loads((args.pool_root / object_id / "bags.json").read_text(encoding="utf-8")))
            for object_id in split["test_objects"]
        }
        per_size = {int(size): int(count) for size, count in (item.split(":") for item in args.per_size.split(","))}
        summary = {"benchmark_scenes": select_benchmark_bags(bags_by_object, split["test_objects"], per_size,
                                                             args.seed)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
