# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Materialize the MindCloud benchmark scenes into the five track layouts.

The scenes, protocols and released modalities are those of
FeedForward-Benchmark's docs/mindcloud_benchmark_scene_selection.md, encoded in
`scripts/preprocess/spatialarena_benchmark/assets/benchmark_selection.yaml`.
The input is the released recordings' processed output
(`<released root>/<Scene>-<YYYYMMDD>-<HHMMSS>/convert_final`): images, the
refined and Sim(3)-aligned COLMAP model `colmap/sparse/0` (the GT poses), the
sparse LiDAR depth `lidar_depth/` (uint16 PNG, cm), the fused LiDAR map
`dense_map/points3D_dense.ply`, the privacy masks and, for the depth tracks,
the GGGS full-generation output (`training/<run>/evaluation`: dense depth in
metres, renders for the PSNR filter).

Output, under `--output_root`: one folder per track holding the scenes
directly, and its `metadata.json` (`dataset_name: spatialarena`;
spatialarena/spatialarena.py):

    SfM/<Scene>/
        images/img<k>_<n>.jpg (all 3 cameras), depths/<same stem>.png
        (LiDAR, cm), sparse/0 (text COLMAP, PINHOLE, world-to-camera)
    SLAM/<Scene>/
        images/<i>.jpg (one camera, time order, resized and padded to the
        SLAM input size slam_protocol.DATASET_INPUT_SIZE["spatialarena"]),
        depths/<i>.png (RGB+D sequences only), sparse/0 (intrinsics at that
        size), frames.tsv; the range, camera and input are in scene.json
    MonoDepth/<Scene>/
        images/, depths/<key>.npy (GGGS, m), lidar_depths/<key>.png,
        materialized_full.json (intrinsics; monodepth_layout records)
    VideoDepth/<Scene>_<clip>/
        images/<i>.jpg, depths/<i>.npy (GGGS, m), lidar_depths/<i>.png,
        sparse/0, frame_manifest.json (one folder per clip or long sequence)
    MVRecon/<Scene>/
        images/ (every view), depths/ (GGGS, m), masks/ (depth mask),
        lidar_depths/, sparse/0, gt_points.ply (the LiDAR map, cropped to
        what the views see)

Every scene with LiDAR depth also gets `lidar_depth_heatmaps/<same stem>.png`,
the colour-coded rendering of it for viewing (the 16-bit centimetre PNGs look
black in an image viewer); every scene with dense depth gets
`depth_heatmaps/<same stem>.png` in the same colours and, per frame, the same
range as its LiDAR heatmap. Every name is zero-padded so that sorted order is
time order. Integer depth PNGs are centimetres (`depth_png_scale: 100` in each
track's metadata.json); .npy depths are float32 metres, 0 = invalid. People
are already zeroed in the GGGS depth (it is masked by `masks/`); vehicles are
removed here. In the depth tracks the people and vehicles
(`privacy/masks/dynamic_combined`) are also blacked out of the images and
their LiDAR heatmaps and zeroed in the LiDAR depth, so every input and every
depth leaves out the same objects (`dynamic_objects_masked` in scene.json; the
sky is kept in the images). Each scene is written to `<track>/.staging/` and
moved into place when complete, so an interrupted run never leaves a partial
scene; a finished scene is skipped unless `--overwrite`.

SfM and SLAM need no GGGS. VideoDepth and MVRecon are written in two stages:
first everything but the GGGS output (images, LiDAR depth, poses, the clips
and views), with `dense_depth: false` in scene.json; a later run, once the
scene's GGGS full generation is done, copies its dense depth in (for MVRecon
also the depth mask and the LiDAR map cropped by it) and sets
`dense_depth: true`. MonoDepth's sample is chosen by GGGS render PSNR, so a
MonoDepth scene is written whole once its GGGS output exists. Each run ends by
listing the scenes still waiting for GGGS; re-run the same command after GGGS
finishes. The GGGS output is the compacted model's (`training/<run>_compact`)
when it has one. When a scene's GGGS run is retrained, its VideoDepth and
MVRecon scenes are rebuilt, but a MonoDepth scene keeps its frames, and so
their manual review, and only their depth is replaced;
`--restore_review_drops` puts the frames the review left out back, with the
new depth, to review them again.

The manual review (the MindCloud monitor's Benchmark Review tab, saved as
`assets/benchmark_review.json`; see `--review`) is applied on every run: a
scene or VideoDepth clip it leaves out is skipped and pruned like a deselected
one, a MonoDepth frame it leaves out is removed, and a VideoDepth frame it
leaves out keeps its image but loses its GT depth (`gt_excluded` in
frame_manifest.json), so the clip stays continuous. MonoDepth samples the
VideoDepth scenes too: a MonoDepth frame left out also takes the GT depth off
that image in every VideoDepth clip of the scene (the clip keeps the image as
input). A rebuilt scene comes out reviewed too. Each run records the labels
and the frames left out in `<output_root>/benchmark_review.json`.
`scripts/preprocess/spatialarena_benchmark/apply_benchmark_review.py` applies
it at once, without the released recordings, and records the scenes left out
in the selection (`monodepth.exclude` and `videodepth.exclude_clips` for a
MonoDepth sample or a single clip, which the selection has no entry of their
own for).

Examples:

    python scripts/preprocess/materialize_spatialarena.py \\
        --released_root <released> --output_root <root> --tracks sfm slam
    python scripts/preprocess/materialize_spatialarena.py \\
        --released_root <released> --output_root <root> \\
        --tracks monodepth videodepth mvrecon
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import functools
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

import numpy as np
from PIL import Image
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from scripts.preprocess import materialize_slam_inputs  # noqa: E402
from scripts.preprocess import monodepth_layout  # noqa: E402
# Tests call `materialize_spatialarena.fit_image`, so the name stays bound here.
from scripts.preprocess.materialize_slam_inputs import fit_image  # noqa: E402
from spatialarena import slam_protocol  # noqa: E402
from spatialarena import spatialarena  # noqa: E402
from spatialarena.geometry import read_write_model  # noqa: E402

# pylint: enable=wrong-import-position

_ASSETS_DIR = (REPO_ROOT / "scripts" / "preprocess" / "spatialarena_benchmark" /
               "assets")
SELECTION_PATH = _ASSETS_DIR / "benchmark_selection.yaml"
REVIEW_PATH = _ASSETS_DIR / "benchmark_review.json"
TRACKS = ("sfm", "slam", "monodepth", "videodepth", "mvrecon")
STAGING_DIRNAME = ".staging"
# The tracks with people and vehicles masked out of every input.
MASKED_TRACKS = ("MonoDepth", "VideoDepth", "MVRecon")
SCENE_RECORD = "scene.json"
FRAME_PATTERN = re.compile(r"^img(\d+)_(\d+)$")
RELEASE_SUFFIX = r"-\d{8}-\d{6}"
# training/<release><suffix>: the benchmark's GGGS run.
GGGS_RUN_SUFFIX = "_gggs_lidar0.5"
# training/<release><suffix>_compact: the same run after compaction.
GGGS_COMPACT_SUFFIX = "_compact"
# <run>/evaluation/<name>: the full-generation manifest; DOGE writes the first, older runs have the second.
GGGS_GENERATION_MANIFESTS = ("full_generation_manifest.json",
                             "mindcloud_full_generation_manifest.json")

# ---- Source scenes -----------------------------------------------------------


@dataclasses.dataclass(frozen=True, order=True)
class Frame:
    """One posed image of a released scene.

    Attributes:
        cam: The rig camera.
        number: The rig frame number.
        name: The path relative to images/, e.g. "cam1/img1_42.JPEG".
    """

    cam: int
    number: int
    name: str

    @property
    def stem(self) -> str:
        """The source image's file name without its suffix."""
        return pathlib.Path(self.name).stem

    @property
    def padded(self) -> str:
        """The zero-padded stem used in every output.

        Sorted order is camera, then time.
        """
        return f"img{self.cam}_{self.number:06d}"


class SourceScene:
    """One released scene's `convert_final` output.

    Attributes:
        name: The scene name.
        root: The scene's `convert_final` folder.
        lidar_depth_scale: Units of the LiDAR depth PNGs per metre.
        allow_stale_lidar_depth: Whether LiDAR depth older than the GT poses is
            accepted.
    """

    def __init__(self,
                 name: str,
                 root: pathlib.Path,
                 lidar_depth_scale: float,
                 allow_stale_lidar_depth: bool = False):
        self.name = name
        self.root = root
        self.lidar_depth_scale = lidar_depth_scale
        self.allow_stale_lidar_depth = allow_stale_lidar_depth

    @functools.cached_property
    def model(self) -> tuple[dict, dict]:
        """The GT model `colmap/sparse/0`.

        Returns:
            (cameras by id, COLMAP images by name).

        Raises:
            ValueError: If a camera is not PINHOLE.
        """
        cameras, images, _ = read_write_model.read_model(
            str(self.root / "colmap" / "sparse" / "0"))
        for camera in cameras.values():
            if camera.model != "PINHOLE":
                raise ValueError(f"{self.name}: expected PINHOLE cameras, "
                                 f"got {camera.model}")
        return cameras, {image.name: image for image in images.values()}

    @functools.cached_property
    def frames(self) -> list[Frame]:
        """Every posed image, by camera then rig frame number."""
        frames = []
        for name in self.model[1]:
            match = FRAME_PATTERN.match(pathlib.Path(name).stem)
            if not match:
                raise ValueError(f"{self.name}: unexpected image name {name}")
            frames.append(Frame(int(match.group(1)), int(match.group(2)), name))
        return sorted(frames)

    def camera_frames(self, cam: int) -> list[Frame]:
        """The frames of rig camera `cam`, in time order."""
        return [frame for frame in self.frames if frame.cam == cam]

    @property
    def rig_cameras(self) -> list[int]:
        """The rig cameras that have posed images, sorted."""
        return sorted({frame.cam for frame in self.frames})

    def image_path(self, frame: Frame) -> pathlib.Path:
        """The source photo of `frame`."""
        return self.root / "images" / frame.name

    def camera(self, frame: Frame) -> read_write_model.Camera:
        """The COLMAP camera of `frame`."""
        cameras, images = self.model
        return cameras[images[frame.name].camera_id]

    def intrinsics(self, frame: Frame) -> list[list[float]]:
        """The 3x3 intrinsic matrix of `frame`."""
        fx, fy, cx, cy = (
            float(value) for value in self.camera(frame).params[:4])
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    def world_to_camera(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        """The world-to-camera rotation and translation of `frame`."""
        image = self.model[1][frame.name]
        rotation = read_write_model.qvec2rotmat(image.qvec)
        return rotation, np.asarray(image.tvec, dtype=np.float64)

    # Sparse LiDAR depth.

    def check_lidar_depth(self) -> None:
        """Checks that the LiDAR depth is usable.

        It must be finished and regenerated after the poses were last refined
        or aligned.

        Raises:
            FileNotFoundError: If there is no refined_depth_manifest.json.
            RuntimeError: If the generation is not done, or predates the GT
                poses (unless `allow_stale_lidar_depth`).
            ValueError: If its depth scale is not `lidar_depth_scale`.
        """
        manifest_path = self.root / "refined_depth_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{self.name}: no refined_depth_manifest.json "
                "(lidar_depth not generated)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "done":
            raise RuntimeError(f"{self.name}: lidar_depth generation is not "
                               f"done ({manifest.get('status')})")
        scale = float(manifest.get("depth_scale", 0.0))
        if not np.isclose(scale, self.lidar_depth_scale):
            raise ValueError(f"{self.name}: lidar_depth scale {scale}, "
                             f"expected {self.lidar_depth_scale}")
        poses = self.root / "colmap" / "sparse" / "0" / "images.bin"
        stale = manifest_path.stat().st_mtime < poses.stat().st_mtime
        if stale and not self.allow_stale_lidar_depth:
            raise RuntimeError(
                f"{self.name}: lidar_depth predates the GT poses; "
                "regenerate it (or pass --allow_stale_lidar_depth)")

    def lidar_depth_path(self, frame: Frame) -> pathlib.Path:
        """The LiDAR depth PNG of `frame` (centimetres)."""
        return (self.root / "lidar_depth" / f"cam{frame.cam}" /
                f"{frame.stem}.png")

    def lidar_heatmap(self, frame: Frame) -> np.ndarray:
        """Returns the colour-coded rendering of the LiDAR depth for viewing.

        lidar_depth itself is 16-bit centimetres and looks black in an image
        viewer. The heatmap is turbo, nearer warmer, over its 1st to 99th
        percentile, invalid black.
        """
        with Image.open(self.lidar_depth_path(frame)) as image:
            depth = np.asarray(image, dtype=np.float32) / self.lidar_depth_scale
        values = depth[depth > 0]
        if values.size:
            lower, upper = np.percentile(values, [1.0, 99.0])
        else:
            lower, upper = (0.0, 1.0)
        return depth_heatmap(depth, float(lower), float(upper))

    # Masks and GGGS.

    def dynamic_mask(self, frame: Frame) -> np.ndarray | None:
        """Returns the pixels masked as people or vehicles.

        None if the scene has no such masks.
        """
        masks = self.root / "privacy" / "masks"
        if (masks / "dynamic_combined").is_dir():
            kinds = ["dynamic_combined"]
        else:
            kinds = ["persons", "vehicles"]
        union = None
        for kind in kinds:
            path = masks / kind / f"cam{frame.cam}" / f"{frame.stem}.png"
            if path.is_file():
                with Image.open(path) as image:
                    mask = np.asarray(image.convert("L")) > 0
                union = mask if union is None else union | mask
        return union

    @functools.cached_property
    def generation(self) -> dict | None:
        """The finished GGGS full-generation manifest.

        Only the benchmark's run counts: its compacted model (`<run>_compact`,
        the one released and shown in the viewer) when that has a finished
        generation, else the run itself; older runs kept beside it under other
        names are ignored. `stamp` (the manifest's modification time) tells
        one generation from the next when the scene is retrained.

        Returns:
            The manifest with its frames keyed by image name (`by_image`), or
            None until it is done.
        """
        run_name = f"{self.root.parent.name}{GGGS_RUN_SUFFIX}"
        run = self.root / "training" / run_name
        for experiment in (run.with_name(run.name + GGGS_COMPACT_SUFFIX), run):
            path = next((experiment / "evaluation" / name
                         for name in GGGS_GENERATION_MANIFESTS
                         if (experiment / "evaluation" / name).is_file()), None)
            if path is None:
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") == "done":
                break
        else:
            return None
        manifest["path"] = str(path)
        manifest["stamp"] = generation_stamp(path)
        manifest["by_image"] = {
            frame["image"]: frame for frame in manifest.get("frames", [])
        }
        return manifest

    def gggs_frame(self, frame: Frame) -> dict:
        """The GGGS full-generation record of `frame`.

        Raises:
            KeyError: If the generation has no output for `frame`.
        """
        record = (self.generation or {}).get("by_image", {}).get(frame.name)
        if record is None:
            raise KeyError(f"{self.name}: no GGGS output for {frame.name}")
        return record

    def dense_depth(self, frame: Frame) -> np.ndarray:
        """Returns the GGGS depth in metres.

        Every excluded pixel (sky, people, vehicles, no render) is set to 0.
        """
        depth = np.asarray(np.load(self.gggs_frame(frame)["depth"]),
                           dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        dynamic = self.dynamic_mask(frame)
        if dynamic is not None:
            valid &= ~dynamic
        return np.where(valid, depth, 0.0).astype(np.float32)

    def render_psnr(self, frame: Frame) -> float:
        """Returns the PSNR of the GGGS render against the photo.

        It is computed over the valid (unmasked) pixels.

        Raises:
            KeyError: If the GGGS output of `frame` has no render.
        """
        record = self.gggs_frame(frame)
        if not record.get("rendered_rgb") or not record.get("rgb_gt"):
            raise KeyError(f"{self.name}: GGGS output of {frame.name} "
                           "has no render to score")
        with Image.open(record["rendered_rgb"]) as rendered:
            with Image.open(record["rgb_gt"]) as reference:
                pred = np.asarray(rendered.convert("RGB"),
                                  dtype=np.float32) / 255.0
                target = np.asarray(reference.convert("RGB"),
                                    dtype=np.float32) / 255.0
        # Both are zeroed outside the mask.
        valid = target.any(axis=2) | pred.any(axis=2)
        if record.get("mask"):
            with Image.open(record["mask"]) as mask:
                valid = np.asarray(mask.convert("L")) > 0
        if not valid.any():
            return float("-inf")
        mse = float(np.mean((pred[valid] - target[valid])**2))
        if mse == 0:
            return float("inf")
        return float(10.0 * np.log10(1.0 / mse))


def generation_stamp(path: pathlib.Path) -> str:
    """The modification time of `path` as an ISO 8601 UTC timestamp."""
    modified = datetime.datetime.fromtimestamp(path.stat().st_mtime,
                                               datetime.timezone.utc)
    return modified.isoformat(timespec="seconds")


def find_scene_root(released_root: pathlib.Path, name: str) -> pathlib.Path:
    """Returns `convert_final` of the release `<name>-YYYYMMDD-HHMMSS`.

    Raises:
        FileNotFoundError: If there is not exactly one such release.
    """
    pattern = re.compile(re.escape(name) + RELEASE_SUFFIX + "$")
    matches = sorted(path for path in released_root.iterdir()
                     if path.is_dir() and pattern.match(path.name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one release of {name} under "
                                f"{released_root}, found {len(matches)}")
    return matches[0] / "convert_final"


def scene_rng(seed: int, track: str, scene: str) -> np.random.Generator:
    """Returns a generator fixed by the seed, the track and the scene.

    It does not depend on which other scenes run.
    """
    digest = hashlib.sha256(f"{seed}:{track}:{scene}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


# ---- Writing helpers ---------------------------------------------------------


def copy_file(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Copies `source` to `destination`, creating its folder."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_masked(source: pathlib.Path, destination: pathlib.Path,
                mask: np.ndarray | None) -> None:
    """Copies an image with the `mask` pixels set to 0.

    The image is a photo, depth PNG or heatmap; unmasked files are copied as
    is. `destination` may be `source` (masked in place); it is replaced only
    once fully written.

    Args:
        source: The image to copy.
        destination: Where to write it.
        mask: The pixels to set to 0; None for none.

    Raises:
        ValueError: If the image and the mask differ in size.
    """
    if mask is None or not mask.any():
        if destination != source:
            copy_file(source, destination)
        return
    with Image.open(source) as image:
        pixels = np.array(image)
    if pixels.shape[:2] != mask.shape:
        raise ValueError(f"{source}: size {pixels.shape[:2]} differs from "
                         f"its mask {mask.shape}")
    pixels[mask] = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(
        f"{destination.stem}.partial{destination.suffix}")
    if destination.suffix.lower() in (".jpg", ".jpeg"):
        Image.fromarray(pixels).save(partial, quality=95)
    else:
        Image.fromarray(pixels).save(partial)
    partial.replace(destination)


def copy_frame(scene: SourceScene, frame: Frame, image: pathlib.Path,
               lidar_depth: pathlib.Path) -> None:
    """Copies a depth-track frame's photo and LiDAR depth.

    Its people and vehicles are masked out of both.
    """
    mask = scene.dynamic_mask(frame)
    copy_masked(scene.image_path(frame), image, mask)
    copy_masked(scene.lidar_depth_path(frame), lidar_depth, mask)


def write_sparse_model(directory: pathlib.Path,
                       scene: SourceScene,
                       entries: list[tuple[str, Frame]],
                       single_camera: bool = False,
                       fit_to: tuple[int, int] | None = None) -> None:
    """Writes a text COLMAP model (world-to-camera, PINHOLE, no points).

    Args:
        directory: The folder to write the model into.
        scene: The source scene.
        entries: (output image name, source frame) of each image.
        single_camera: Whether every image shares camera 1.
        fit_to: (width, height); when given, the intrinsics are those of the
            frames resized and padded to it.

    Raises:
        ValueError: If a single-camera sequence mixes intrinsics.
    """
    source_cameras, source_images = scene.model
    cameras: dict[int, read_write_model.Camera] = {}
    images: dict[int, read_write_model.Image] = {}
    for image_id, (name, frame) in enumerate(sorted(entries), start=1):
        source = source_images[frame.name]
        camera = source_cameras[source.camera_id]
        camera_id = 1 if single_camera else camera.id
        if camera_id not in cameras:
            cameras[camera_id] = read_write_model.Camera(id=camera_id,
                                                         model=camera.model,
                                                         width=camera.width,
                                                         height=camera.height,
                                                         params=np.asarray(
                                                             camera.params,
                                                             dtype=np.float64))
        elif single_camera and not np.allclose(cameras[camera_id].params,
                                               camera.params):
            raise ValueError(
                f"{scene.name}: a single-camera sequence mixes intrinsics")
        images[image_id] = read_write_model.Image(id=image_id,
                                                  qvec=np.asarray(source.qvec),
                                                  tvec=np.asarray(source.tvec),
                                                  camera_id=camera_id,
                                                  name=name,
                                                  xys=np.zeros((0, 2)),
                                                  point3D_ids=np.zeros(
                                                      0, dtype=np.int64))
    if fit_to is not None:
        cameras = {
            key: materialize_slam_inputs.fit_camera(camera, fit_to)
            for key, camera in cameras.items()
        }
    directory.mkdir(parents=True, exist_ok=True)
    read_write_model.write_model(cameras,
                                 images, {},
                                 str(directory),
                                 ext=".txt")


class SceneWriter:
    """Stages one output scene and moves it into place once complete.

    Attributes:
        track_root: The track folder.
        scene: The output scene name.
        final: The scene's folder.
        staging: The folder the scene is written to before it is moved.
        overwrite: Whether a complete scene is rebuilt.
    """

    def __init__(self, track_root: pathlib.Path, scene: str, overwrite: bool):
        self.track_root = track_root
        self.scene = scene
        self.final = track_root / scene
        self.staging = track_root / STAGING_DIRNAME / scene
        self.overwrite = overwrite

    def existing_record(self) -> dict | None:
        """The complete scene's scene.json; None if absent or `overwrite`."""
        record = self.final / SCENE_RECORD
        if record.is_file() and not self.overwrite:
            return json.loads(record.read_text(encoding="utf-8"))
        return None

    def begin(self) -> pathlib.Path:
        """Starts an empty staging folder and returns it."""
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.mkdir(parents=True)
        return self.staging

    def commit(self, record: dict) -> dict:
        """Writes `record` as scene.json and moves the scene into place."""
        text = json.dumps(record, indent=2) + "\n"
        (self.staging / SCENE_RECORD).write_text(text, encoding="utf-8")
        if self.final.exists():
            shutil.rmtree(self.final)
        self.staging.rename(self.final)
        staging_root = self.track_root / STAGING_DIRNAME
        if staging_root.is_dir() and not any(staging_root.iterdir()):
            staging_root.rmdir()
        return record


@contextlib.contextmanager
def staged_directory(final: pathlib.Path):
    """Writes a folder under a temporary name.

    The folder is moved into place only when the block succeeds.

    Args:
        final: The folder to write.

    Yields:
        The temporary folder to write into.
    """
    staging = final.with_name(final.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        yield staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)


def update_scene_record(scene_dir: pathlib.Path, **changes) -> dict:
    """Merges `changes` into a scene's scene.json.

    It is written last, so it marks the change complete.
    """
    path = scene_dir / SCENE_RECORD
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(changes)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def gggs_run_name(scene: SourceScene) -> str:
    """The GGGS run (training/<run>) the dense depth comes from."""
    return pathlib.Path(scene.generation["path"]).parent.parent.name


def gggs_fields(scene: SourceScene) -> dict:
    """scene.json fields naming the GGGS run and generation of the depth."""
    return {
        "gggs": gggs_run_name(scene),
        "gggs_generated": scene.generation["stamp"]
    }


HEATMAP_DIRNAME = "lidar_depth_heatmaps"


def _lidar_sources_from_images(scene_dir: pathlib.Path) -> dict[str, str]:
    """Maps the img<k>_<n> image stems of an SfM or MVRecon scene to sources."""
    sources = {}
    for path in (scene_dir / "images").iterdir():
        match = FRAME_PATTERN.match(path.stem)
        if match:
            cam = match.group(1)
            sources[path.stem] = f"cam{cam}/img{cam}_{int(match.group(2))}"
    return sources


def lidar_depth_sources(scene_dir: pathlib.Path) -> dict[str, str]:
    """Maps each LiDAR depth file's output stem to its source image name."""
    if (scene_dir / "frames.tsv").is_file():  # SLAM.
        rows = (scene_dir /
                "frames.tsv").read_text(encoding="utf-8").splitlines()[1:]
        sources = {
            pathlib.Path(row.split("\t")[0]).stem: row.split("\t")[1]
            for row in rows
            if row
        }
    elif (scene_dir / "frame_manifest.json").is_file():
        # VideoDepth (a frame left out by the review has none).
        manifest = json.loads(
            (scene_dir / "frame_manifest.json").read_text(encoding="utf-8"))
        sources = {
            pathlib.Path(entry["lidar_depth"]).stem: entry["source_rgb"]
            for entry in manifest
            if entry.get("lidar_depth")
        }
    elif (scene_dir / monodepth_layout.MANIFEST_NAME).is_file():  # MonoDepth.
        manifest = json.loads(
            (scene_dir /
             monodepth_layout.MANIFEST_NAME).read_text(encoding="utf-8"))
        sources = {
            pathlib.Path(record["lidar_depth"]).stem: record["source"]
            for record in manifest["records"]
        }
    else:  # SfM and MVRecon: img<k>_<n> stems.
        sources = _lidar_sources_from_images(scene_dir)
    if (scene_dir / "lidar_depths").is_dir():
        folder = scene_dir / "lidar_depths"
    else:
        folder = scene_dir / "depths"
    if folder.is_dir():
        stems = {path.stem for path in folder.glob("*.png")}
    else:
        stems = set()
    return {stem: source for stem, source in sources.items() if stem in stems}


def _has_pngs(folder: pathlib.Path, count: int) -> bool:
    """Whether `folder` exists and holds exactly `count` PNG files."""
    return folder.is_dir() and len(list(folder.glob("*.png"))) == count


def add_lidar_heatmaps(scene: SourceScene,
                       scene_dir: pathlib.Path,
                       mask_dynamic: bool = False,
                       fit_to: tuple[int, int] | None = None) -> int:
    """Draws the heatmap of every LiDAR depth file into lidar_depth_heatmaps/.

    The heatmaps keep the depth files' stems.

    Args:
        scene: The source scene.
        scene_dir: The materialized scene.
        mask_dynamic: Whether people and vehicles are blacked out.
        fit_to: (width, height); when given, the heatmaps are resized and
            padded to it like the frames.

    Returns:
        The number of heatmaps drawn (0 when they are all there already).
    """
    sources = lidar_depth_sources(scene_dir)
    final = scene_dir / HEATMAP_DIRNAME
    if not sources or _has_pngs(final, len(sources)):
        return 0
    by_stem = {frame.stem: frame for frame in scene.frames}
    with staged_directory(final) as heatmaps:
        for stem, source in sources.items():
            frame = by_stem[pathlib.Path(source).stem]
            heatmap = scene.lidar_heatmap(frame)
            if fit_to is not None:
                heatmap = fit_image(heatmap, fit_to)
            elif mask_dynamic:
                mask = scene.dynamic_mask(frame)
                if mask is not None:
                    heatmap[mask] = 0
            Image.fromarray(heatmap).save(heatmaps / f"{stem}.png",
                                          compress_level=1)
    return len(sources)


def mask_dynamic_objects(scene: SourceScene, scene_dir: pathlib.Path) -> dict:
    """Blacks out people and vehicles in a depth-track scene, in place.

    For a scene written before they were masked: its images, LiDAR depth and
    LiDAR heatmaps are masked; the sample, clips and views stay as they are.

    Returns:
        The updated scene.json record.
    """
    sources = lidar_depth_sources(scene_dir)
    by_stem = {frame.stem: frame for frame in scene.frames}
    folders = {
        "images": ".jpg",
        "lidar_depths": ".png",
        HEATMAP_DIRNAME: ".png"
    }
    for stem, source in sources.items():
        mask = scene.dynamic_mask(by_stem[pathlib.Path(source).stem])
        for folder, suffix in folders.items():
            path = scene_dir / folder / f"{stem}{suffix}"
            if path.is_file():
                copy_masked(path, path, mask)
    return update_scene_record(scene_dir, dynamic_objects_masked=True)


DENSE_HEATMAP_DIRNAME = "depth_heatmaps"


def depth_heatmap(depth: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Returns an RGB turbo heatmap of metric depth.

    Drawn as the pipeline draws the LiDAR heatmaps: nearer is warmer, the range
    [lower, upper] spans the colours, and invalid (zero) pixels are black.

    Args:
        depth: Depth in metres, 0 = invalid.
        lower: The depth drawn in the warmest colour.
        upper: The depth drawn in the coolest colour.

    Returns:
        A uint8 (H, W, 3) image.
    """
    import matplotlib  # pylint: disable=import-outside-toplevel

    colors = matplotlib.colormaps["turbo"](np.linspace(0.0, 1.0, 256))[:, :3]
    lut = np.rint(colors * 255.0).astype(np.uint8)
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.clip((depth - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
    levels = np.rint((1.0 - np.where(valid, normalized, 1.0)) * 255.0)
    image = lut[levels.astype(np.uint8)]
    image[~valid] = 0
    return image


def add_dense_heatmaps(scene_dir: pathlib.Path,
                       lidar_depth_scale: float) -> int:
    """Draws a heatmap of every dense depth into depth_heatmaps/.

    The dense depths are depths/*.npy (m) and the heatmaps keep their stems.
    Each uses the colour range of the frame's LiDAR heatmap (1st to 99th
    percentile of its LiDAR depth), so the two heatmaps of a frame can be
    compared colour for colour; a frame without LiDAR depth uses its own dense
    depth.

    Args:
        scene_dir: The materialized scene.
        lidar_depth_scale: Units of the LiDAR depth PNGs per metre.

    Returns:
        The number of heatmaps drawn (0 when they are all there already).
    """
    if (scene_dir / "depths").is_dir():
        depths = sorted((scene_dir / "depths").glob("*.npy"))
    else:
        depths = []
    final = scene_dir / DENSE_HEATMAP_DIRNAME
    if not depths or _has_pngs(final, len(depths)):
        return 0
    with staged_directory(final) as heatmaps:
        for path in depths:
            depth = np.load(path).astype(np.float32)
            lidar_path = scene_dir / "lidar_depths" / f"{path.stem}.png"
            reference = depth
            if lidar_path.is_file():
                with Image.open(lidar_path) as lidar:
                    reference = np.asarray(lidar,
                                           dtype=np.float32) / lidar_depth_scale
            values = reference[np.isfinite(reference) & (reference > 0)]
            if not values.size:
                values = depth[np.isfinite(depth) & (depth > 0)]
            if values.size:
                lower, upper = np.percentile(values, [1.0, 99.0])
            else:
                lower, upper = (0.0, 1.0)
            heatmap = depth_heatmap(depth, float(lower), float(upper))
            Image.fromarray(heatmap).save(heatmaps / f"{path.stem}.png",
                                          compress_level=1)
    return len(depths)


def write_track_metadata(track_root: pathlib.Path, task: str,
                         records: list[dict], **extra) -> dict:
    """Writes the track's `metadata.json`.

    Entries of scenes that this run did not touch are kept.

    Args:
        track_root: The track folder.
        task: The track's task name.
        records: The scene.json records of the scenes this run wrote.
        **extra: Further top-level fields.

    Returns:
        The metadata written.
    """
    path = track_root / "metadata.json"
    entries = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for entry in existing.get("scenes", []):
            if (track_root / entry["scene_name"]).is_dir():
                entries[entry["scene_name"]] = entry
    for record in records:
        entries[record["scene_name"]] = record
    scenes = [entries[name] for name in sorted(entries)]
    total = sum(int(entry["total_images"]) for entry in scenes)
    metadata = {
        "dataset_name": spatialarena.SPATIALARENA,
        "task": task,
        "root_directory": ".",
        "total_images": total,
        "average_images_per_scene": total / len(scenes) if scenes else 0.0,
    }
    metadata.update(extra)
    metadata["scenes"] = scenes
    track_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def image_size(scene: SourceScene, frame: Frame) -> list[int]:
    """The [width, height] of `frame`."""
    camera = scene.camera(frame)
    return [int(camera.width), int(camera.height)]


# ---- SfM ---------------------------------------------------------------------


def materialize_sfm_scene(scene: SourceScene, writer: SceneWriter) -> dict:
    """Writes every posed image of the 3 cameras, their LiDAR depth and GT."""
    scene.check_lidar_depth()
    root = writer.begin()
    entries = []
    depths = 0
    for frame in scene.frames:
        name = f"{frame.padded}.jpg"
        copy_file(scene.image_path(frame), root / "images" / name)
        lidar = scene.lidar_depth_path(frame)
        if lidar.is_file():
            copy_file(lidar, root / "depths" / f"{frame.padded}.png")
            depths += 1
        entries.append((name, frame))
    write_sparse_model(root / "sparse" / "0", scene, entries)
    return writer.commit({
        "scene_name": writer.scene,
        "source_scene": scene.root.parent.name,
        "root_directory": writer.scene,
        "total_images": len(entries),
        "num_sparse_images": len(entries),
        "num_depth_maps": depths,
        "cameras": scene.rig_cameras,
    })


# ---- SLAM --------------------------------------------------------------------


def write_fitted(source: pathlib.Path,
                 destination: pathlib.Path,
                 target: tuple[int, int],
                 nearest: bool = False) -> None:
    """Writes `source` resized to fit within `target` and zero-padded to it.

    As every SLAM dataset's inputs are (scripts/preprocess/
    materialize_slam_inputs.py); depth PNGs (`nearest`) keep their dtype and
    units.

    Raises:
        RuntimeError: If `source` cannot be read or `destination` written.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    if nearest:
        flags = cv2.IMREAD_UNCHANGED
    else:
        flags = cv2.IMREAD_COLOR
    frame = cv2.imread(str(source), flags)
    if frame is None:
        raise RuntimeError(f"Could not read {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        params = []
    fitted = fit_image(frame, target, nearest=nearest)
    if not cv2.imwrite(str(destination), fitted, params):
        raise RuntimeError(f"Could not write {destination}")


def materialize_slam_sequence(scene: SourceScene, writer: SceneWriter,
                              sequence: dict, frame_rate_hz: float) -> dict:
    """Writes one camera's frames in time order as 000000.jpg...

    With LiDAR depth for RGB+D sequences, at the SLAM protocol's input size
    (`slam_protocol.DATASET_INPUT_SIZE`), with the intrinsics mapped to it.

    Args:
        scene: The source scene.
        writer: The output scene.
        sequence: The selection's sequence entry (camera, input, range).
        frame_rate_hz: The sequence's frame rate.

    Returns:
        The scene.json record.

    Raises:
        ValueError: If the camera has no posed images.
        FileNotFoundError: If an RGB+D frame has no LiDAR depth.
    """
    scene.check_lidar_depth()
    target = slam_protocol.input_size(spatialarena.SPATIALARENA)
    cam = int(sequence["camera"])
    with_depth = sequence["input"] == "rgb_depth"
    frames = scene.camera_frames(cam)
    if not frames:
        raise ValueError(f"{scene.name}: no posed cam{cam} images")
    root = writer.begin()
    entries = []
    rows = ["name\tsource_image\trig_frame"]
    for index, frame in enumerate(frames):
        name = f"{index:06d}.jpg"
        write_fitted(scene.image_path(frame), root / "images" / name, target)
        if with_depth:
            lidar = scene.lidar_depth_path(frame)
            if not lidar.is_file():
                raise FileNotFoundError(
                    f"{scene.name}: missing LiDAR depth {lidar}")
            write_fitted(lidar,
                         root / "depths" / f"{index:06d}.png",
                         target,
                         nearest=True)
        entries.append((name, frame))
        rows.append(f"{name}\t{frame.name}\t{frame.number}")
    write_sparse_model(root / "sparse" / "0",
                       scene,
                       entries,
                       single_camera=True,
                       fit_to=target)
    (root / "frames.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    width, height = image_size(scene, frames[0])
    return writer.commit({
        "scene_name": writer.scene,
        "source_scene": scene.root.parent.name,
        "source_sequence": f"cam{cam}",
        "root_directory": writer.scene,
        "total_images": len(frames),
        "range": sequence["range"],
        "camera": cam,
        "input": sequence["input"],
        "has_depths": with_depth,
        "camera_model": "PINHOLE",
        "width": target[0],
        "height": target[1],
        "input_size": list(target),
        "source_size": [width, height],
        "frame_rate_hz": frame_rate_hz,
        "first_rig_frame": frames[0].number,
        "last_rig_frame": frames[-1].number,
    })


# ---- MonoDepth ---------------------------------------------------------------


def _camera_sample_counts(scene: SourceScene, total: int) -> dict[int, int]:
    """Each camera's share of a sample of `total` images.

    Shares are rounded by largest remainder so they add up to the total.
    """
    shares = {
        cam: total * len(scene.camera_frames(cam)) / len(scene.frames)
        for cam in scene.rig_cameras
    }
    counts = {cam: int(share) for cam, share in shares.items()}
    by_remainder = sorted(shares, key=lambda cam: counts[cam] - shares[cam])
    for cam in by_remainder[:total - sum(counts.values())]:
        counts[cam] += 1
    return counts


def monodepth_selection(scene: SourceScene, fraction: float,
                        seed: int) -> tuple[list[Frame], dict[str, float]]:
    """Draws 10% of the scene's images, from those above the median PSNR.

    The PSNR is the GGGS render PSNR. The sample is spread over time so it
    does not hold near-duplicate neighbouring frames: each camera's share is
    split into equal stretches of its timeline and one eligible image is drawn
    per stretch, at least half a stretch after the camera's previous pick. A
    stretch with no such image is left out, so a scene can fall a little
    short.

    Args:
        scene: The source scene.
        fraction: The fraction of the scene's images to draw.
        seed: The selection's seed.

    Returns:
        The chosen frames, sorted, and the render PSNR of every image by name.
    """
    psnr = {frame.name: scene.render_psnr(frame) for frame in scene.frames}
    median = float(np.median(list(psnr.values())))
    total = monodepth_layout.fraction_count(len(scene.frames), fraction)
    rng = scene_rng(seed, "monodepth", scene.name)
    chosen = []
    counts = _camera_sample_counts(scene, total)
    for cam in scene.rig_cameras:
        frames = sorted(scene.camera_frames(cam),
                        key=lambda frame: frame.number)
        count = counts[cam]
        if not count:
            continue
        gap = len(frames) / count / 2
        previous = None
        for stretch in np.array_split(np.arange(len(frames)), count):
            spaced = [
                index for index in stretch
                if psnr[frames[index].name] > median and
                (previous is None or index - previous >= gap)
            ]
            if not spaced:
                continue
            pick = int(rng.choice(spaced))
            chosen.append(frames[pick])
            previous = pick
    return sorted(chosen), psnr


def monodepth_key(frame: Frame) -> str:
    """The MonoDepth sample key of `frame`."""
    return monodepth_layout.frame_key(f"cam{frame.cam}/{frame.padded}")


def write_monodepth_frame(scene: SourceScene, root: pathlib.Path, frame: Frame,
                          psnr: float) -> monodepth_layout.FrameRecord:
    """Writes one MonoDepth frame under `root`.

    Its photo, GGGS depth (m) and LiDAR depth.

    Args:
        scene: The source scene.
        root: The MonoDepth scene folder.
        frame: The frame to write.
        psnr: The frame's GGGS render PSNR.

    Returns:
        Its manifest record.
    """
    key = monodepth_key(frame)
    monodepth_layout.save_depth(root / "depths" / f"{key}.npy",
                                scene.dense_depth(frame))
    extra = {
        "lidar_depth": f"lidar_depths/{key}.png",
        "lidar_depth_scale": scene.lidar_depth_scale,
        "render_psnr": round(psnr, 3)
    }
    copy_frame(scene, frame, root / "images" / f"{key}.jpg",
               root / extra["lidar_depth"])
    return monodepth_layout.FrameRecord(key=key,
                                        image=f"images/{key}.jpg",
                                        depth=f"depths/{key}.npy",
                                        source=frame.name,
                                        intrinsics=scene.intrinsics(frame),
                                        extra=extra)


def materialize_monodepth_scene(scene: SourceScene, writer: SceneWriter,
                                config: dict, seed: int) -> dict:
    """Writes the PSNR-filtered 10% sample.

    Its images, GGGS depth (m), LiDAR depth and intrinsics.

    Args:
        scene: The source scene.
        writer: The output scene.
        config: The selection's `monodepth` section.
        seed: The selection's seed.

    Returns:
        The scene.json record.
    """
    scene.check_lidar_depth()
    chosen, psnr = monodepth_selection(scene, float(config["fraction"]), seed)
    root = writer.begin()
    records = [
        write_monodepth_frame(scene, root, frame, psnr[frame.name])
        for frame in chosen
    ]
    limits = monodepth_layout.SceneLimits(min_depth=float(config["min_depth"]),
                                          max_depth=config.get("max_depth"))
    provenance = {
        "gggs": gggs_run_name(scene),
        "selection":
            f"{config['fraction']:.0%} of images, from those above the "
            "median GGGS render PSNR, "
            "spread evenly over each camera's timeline",
        "median_render_psnr": round(float(np.median(list(psnr.values()))), 3),
        "seed": seed
    }
    manifest = {
        "scene": writer.scene,
        "num_samples": len(records),
        "images_dir": "images",
        "depths_dir": "depths",
        "masks_dir": None,
    }
    manifest.update(limits.as_dict())
    manifest["provenance"] = provenance
    manifest["records"] = [
        record.as_dict()
        for record in sorted(records, key=lambda record: record.key)
    ]
    (root / monodepth_layout.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    scene_record = {
        "scene_name": writer.scene,
        "source_scene": scene.root.parent.name,
        "root_directory": writer.scene,
        "total_images": len(records),
        "total_depth_maps": len(records),
        "layout": "monodepth_materialized",
        "dynamic_objects_masked": True,
    }
    scene_record.update(limits.as_dict())
    scene_record.update(gggs_fields(scene))
    scene_record["provenance"] = provenance
    return writer.commit(scene_record)


def refresh_monodepth_depth(scene: SourceScene,
                            scene_dir: pathlib.Path) -> dict:
    """Replaces a MonoDepth scene's GGGS depth with the current generation's.

    Each frame's render PSNR is replaced too; the frames are kept. The sample
    was drawn by the render PSNR of the GGGS run it was first written from.
    Drawing it again from a retrained run would pick other frames and void
    their manual review, so the frames stay and only their depth changes.
    `provenance.sampled_with` names the run the sample was drawn with.

    Returns:
        The updated scene.json record.
    """
    manifest_path = scene_dir / monodepth_layout.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {frame.name: frame for frame in scene.frames}
    for record in manifest["records"]:
        frame = by_name[record["source"]]
        monodepth_layout.save_depth(scene_dir / record["depth"],
                                    scene.dense_depth(frame))
        record["render_psnr"] = round(scene.render_psnr(frame), 3)
    # Redrawn by the caller.
    shutil.rmtree(scene_dir / DENSE_HEATMAP_DIRNAME, ignore_errors=True)
    provenance = dict(manifest.get("provenance", {}))
    provenance.setdefault("sampled_with", provenance.get("gggs"))
    provenance["gggs"] = gggs_run_name(scene)
    manifest["provenance"] = provenance
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    return update_scene_record(scene_dir,
                               **gggs_fields(scene),
                               provenance=provenance)


def restore_monodepth_frames(scene: SourceScene,
                             scene_dir: pathlib.Path) -> list[str]:
    """Puts the frames the review left out of a MonoDepth scene back.

    They come back with the current GGGS depth, so they can be reviewed again
    (after a retrain, say). Their labels are not changed: the next run without
    `--restore_review_drops` removes the frames still left out.

    Returns:
        The keys of the frames put back.
    """
    record = json.loads((scene_dir / SCENE_RECORD).read_text(encoding="utf-8"))
    manifest_path = scene_dir / monodepth_layout.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    present = {entry["key"] for entry in manifest["records"]}
    by_key = {monodepth_key(frame): frame for frame in scene.frames}
    excluded = record.get("manual_review", {}).get("excluded_frames", [])
    restored = [key for key in excluded if key not in present and key in by_key]
    for key in restored:
        frame = by_key[key]
        manifest["records"].append(
            write_monodepth_frame(scene, scene_dir, frame,
                                  scene.render_psnr(frame)).as_dict())
    manifest["records"].sort(key=lambda entry: entry["key"])
    manifest["num_samples"] = len(manifest["records"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    update_scene_record(scene_dir,
                        total_images=len(manifest["records"]),
                        total_depth_maps=len(manifest["records"]),
                        manual_review={"excluded_frames": []})
    return restored


# ---- VideoDepth --------------------------------------------------------------


def _clip_spans(length: int, low: int, high: int,
                rng: np.random.Generator) -> list[tuple[int, int]]:
    """Splits a timeline of `length` frames into clips of `low`-`high` frames.

    Returns:
        (start, end) of each clip, end exclusive.
    """
    spans = []
    start = 0
    while start < length:
        end = min(start + int(rng.integers(low, high + 1)), length)
        spans.append((start, end))
        start = end
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < low:
        tail = spans.pop()
        # Merge a short tail when the clip stays within the maximum.
        if tail[1] - spans[-1][0] <= high:
            spans[-1] = (spans[-1][0], tail[1])
    return spans


def videodepth_clips(scene: SourceScene, config: dict, seed: int) -> list[dict]:
    """Plans a scene's VideoDepth clips.

    Consecutive, non-overlapping clips of 100-500 rig frames, each on one
    random camera; or the whole sequence.

    Args:
        scene: The source scene.
        config: The selection's `videodepth` section.
        seed: The selection's seed.

    Returns:
        One record per clip: its scene name, camera and rig frame range.
    """
    low, high = (int(value) for value in config["clip_frames"])
    timeline = sorted({frame.number for frame in scene.frames})
    rng = scene_rng(seed, "videodepth", scene.name)
    if scene.name in set(config.get("whole", [])):
        spans = [(0, len(timeline))]
    else:
        spans = _clip_spans(len(timeline), low, high, rng)
    cameras = scene.rig_cameras
    clips = []
    for index, (start, end) in enumerate(spans):
        cam = int(cameras[int(rng.integers(len(cameras)))])
        first, last = timeline[start], timeline[end - 1]
        whole = len(spans) == 1 and scene.name in set(config.get("whole", []))
        if whole:
            name = f"{scene.name}_full_cam{cam}"
        else:
            name = f"{scene.name}_clip{index:02d}_cam{cam}"
        clips.append({
            "scene_name": name,
            "camera": cam,
            "first_rig_frame": first,
            "last_rig_frame": last,
            "rig_frames": end - start,
            "whole_sequence": whole
        })
    return clips


def materialize_videodepth_clip(scene: SourceScene, writer: SceneWriter,
                                clip: dict) -> dict:
    """Writes one clip's frames as 000000.jpg...

    With LiDAR depth and its camera model; the GGGS depth is added by
    add_videodepth_dense_depth once it exists.

    Args:
        scene: The source scene.
        writer: The output scene.
        clip: The clip's record from videodepth_clips.

    Returns:
        The scene.json record.
    """
    scene.check_lidar_depth()
    frames = [
        frame for frame in scene.camera_frames(clip["camera"])
        if clip["first_rig_frame"] <= frame.number <= clip["last_rig_frame"]
    ]
    root = writer.begin()
    manifest = []
    entries = []
    for index, frame in enumerate(frames):
        stem = f"{index:06d}"
        copy_frame(scene, frame, root / "images" / f"{stem}.jpg",
                   root / "lidar_depths" / f"{stem}.png")
        entries.append((f"{stem}.jpg", frame))
        manifest.append({
            "frame": index,
            "source_frame": frame.number,
            "rgb": f"images/{stem}.jpg",
            "depth": None,
            "lidar_depth": f"lidar_depths/{stem}.png",
            "rgb_wh": image_size(scene, frame),
            "depth_wh": None,
            "source_rgb": frame.name
        })
    write_sparse_model(root / "sparse" / "0",
                       scene,
                       entries,
                       single_camera=True)
    (root / "frame_manifest.json").write_text(json.dumps(manifest, indent=2) +
                                              "\n",
                                              encoding="utf-8")
    record = {
        "scene_name": writer.scene,
        "source_scene": scene.root.parent.name,
        "root_directory": writer.scene,
        "total_images": len(frames),
        "dynamic_objects_masked": True,
        "dense_depth": False,
        "gt_depth_scale": 1.0,
        "lidar_depth_scale": scene.lidar_depth_scale,
        "rgb_wh": image_size(scene, frames[0]),
        "pairing": "same frame index",
        "has_masks": False,
    }
    for key in ("camera", "first_rig_frame", "last_rig_frame",
                "whole_sequence"):
        record[key] = clip[key]
    return writer.commit(record)


def add_videodepth_dense_depth(scene: SourceScene,
                               scene_dir: pathlib.Path) -> dict:
    """Copies the GGGS depth of a materialized clip's frames into its depths/.

    The depth is in metres, with vehicles removed.

    Returns:
        The updated scene.json record.
    """
    manifest_path = scene_dir / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {frame.name: frame for frame in scene.frames}
    with staged_directory(scene_dir / "depths") as depths:
        for entry in manifest:
            if entry.get("gt_excluded"):
                # Left out by the manual review: no GT depth.
                continue
            depth = scene.dense_depth(by_name[entry["source_rgb"]])
            monodepth_layout.save_depth(depths / f"{entry['frame']:06d}.npy",
                                        depth)
            entry["depth"] = f"depths/{entry['frame']:06d}.npy"
            entry["depth_wh"] = [int(depth.shape[1]), int(depth.shape[0])]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    depth_wh = next(
        (entry["depth_wh"] for entry in manifest if entry.get("depth_wh")),
        None)
    return update_scene_record(scene_dir,
                               dense_depth=True,
                               depth_wh=depth_wh,
                               **gggs_fields(scene))


# ---- MVRecon -----------------------------------------------------------------

_PLY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "ushort": "<u2",
    "uint16": "<u2",
    "short": "<i2",
    "int16": "<i2",
    "uint": "<u4",
    "uint32": "<u4",
    "int": "<i4",
    "int32": "<i4"
}


def _read_ply_header(
        path: pathlib.Path) -> tuple[list[tuple[str, str]], int, int]:
    """Reads the header of a binary little-endian vertex PLY.

    Returns:
        The vertex fields as (name, numpy dtype), the vertex count and the
        byte offset of the data.

    Raises:
        ValueError: If the file is not a supported PLY.
    """
    fields: list[tuple[str, str]] = []
    count = 0
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            words = line.decode("ascii", "replace").split()
            if not words:
                continue
            if words[0] == "format" and words[1] != "binary_little_endian":
                raise ValueError(
                    f"{path}: only binary little-endian PLY is supported")
            if words[0] == "element":
                in_vertex = words[1] == "vertex"
                if in_vertex:
                    count = int(words[2])
            elif words[0] == "property" and in_vertex:
                if words[1] == "list":
                    raise ValueError(f"{path}: list properties in the vertex "
                                     "element are not supported")
                fields.append((words[2], _PLY_TYPES[words[1]]))
            elif words[0] == "end_header":
                offset = handle.tell()
                break
    return fields, count, offset


def read_ply_points(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads xyz (float32) and rgb (uint8) of a binary little-endian PLY.

    Points without colour get grey (200, 200, 200).
    """
    fields, count, offset = _read_ply_header(path)
    vertices = np.memmap(path,
                         dtype=np.dtype(fields),
                         mode="r",
                         offset=offset,
                         shape=(count,))
    xyz = np.stack(
        [
            np.asarray(vertices[axis], dtype=np.float32)
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )
    names = {name for name, _ in fields}
    if {"red", "green", "blue"} <= names:
        rgb = np.stack(
            [
                np.asarray(vertices[channel])
                for channel in ("red", "green", "blue")
            ],
            axis=1,
        ).astype(np.uint8)
    else:
        rgb = np.full((count, 3), 200, dtype=np.uint8)
    return xyz, rgb


def write_ply_points(path: pathlib.Path, xyz: np.ndarray,
                     rgb: np.ndarray) -> None:
    """Writes coloured points as a binary little-endian PLY."""
    vertices = np.empty(len(xyz),
                        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices["x"], vertices["y"], vertices["z"] = np.asarray(xyz,
                                                             dtype=np.float32).T
    vertices["red"], vertices["green"], vertices["blue"] = np.asarray(
        rgb, dtype=np.uint8).T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\n"
              "property uchar blue\nend_header\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


# (R_w2c, t_w2c, K, depth) of one view; depth 0 = no depth.
_View = tuple[np.ndarray, np.ndarray, list[list[float]], np.ndarray]


def visible_points(xyz: np.ndarray,
                   views: list[_View],
                   tolerance: float,
                   chunk: int = 4_000_000) -> np.ndarray:
    """Returns which points some view sees.

    A view sees a point inside its image, in front of it, and not behind its
    dense depth by more than `tolerance` (+ 2% of the depth).

    Args:
        xyz: The points, (N, 3).
        views: (R_w2c, t_w2c, K, depth) of each view, with 0 = no depth.
        tolerance: The allowed distance behind the depth, in metres.
        chunk: The number of points projected at once.

    Returns:
        A boolean mask over the points.
    """
    keep = np.zeros(len(xyz), dtype=bool)
    for rotation, translation, intrinsics, depth in views:
        height, width = depth.shape
        fx, fy = intrinsics[0][0], intrinsics[1][1]
        cx, cy = intrinsics[0][2], intrinsics[1][2]
        for start in range(0, len(xyz), chunk):
            todo = ~keep[start:start + chunk]
            if not todo.any():
                continue
            index = np.nonzero(todo)[0] + start
            camera = xyz[index].astype(np.float64) @ rotation.T + translation
            z = camera[:, 2]
            front = z > 1e-3
            u = np.full(len(z), -1, dtype=np.int64)
            v = np.full(len(z), -1, dtype=np.int64)
            u[front] = np.floor(fx * camera[front, 0] / z[front] + cx).astype(
                np.int64)
            v[front] = np.floor(fy * camera[front, 1] / z[front] + cy).astype(
                np.int64)
            inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            surface = np.zeros(len(z), dtype=np.float32)
            surface[inside] = depth[v[inside], u[inside]]
            near_surface = z <= surface + tolerance + 0.02 * surface
            seen = inside & (surface > 0) & near_surface
            keep[index[seen]] = True
    return keep


def mvrecon_view_count(scene: SourceScene, views: int) -> int:
    """The number of views of an MVRecon scene.

    `views` drawn from all 3 cameras, or every posed image when `views` is 0.
    """
    return len(scene.frames) if views <= 0 else min(views, len(scene.frames))


def materialize_mvrecon_scene(scene: SourceScene, writer: SceneWriter,
                              settings: dict, views: int, seed: int) -> dict:
    """Writes the views with LiDAR depth and the GT model.

    Every image of the 3 cameras when `views` is 0. The GGGS depth, its mask
    and the cropped LiDAR map are added by add_mvrecon_dense_depth once the
    GGGS depth exists.

    Args:
        scene: The source scene.
        writer: The output scene.
        settings: The scene's selection entry (setting, eval_threshold).
        views: The number of views (0 = all).
        seed: The selection's seed.

    Returns:
        The scene.json record.
    """
    scene.check_lidar_depth()
    frames = scene.frames
    if mvrecon_view_count(scene, views) == len(frames):
        chosen = list(frames)
    else:
        rng = scene_rng(seed, "mvrecon", scene.name)
        chosen = sorted(
            frames[index]
            for index in rng.choice(len(frames), size=views, replace=False))
    root = writer.begin()
    entries = []
    for frame in chosen:
        name = f"{frame.padded}.jpg"
        copy_frame(scene, frame, root / "images" / name,
                   root / "lidar_depths" / f"{frame.padded}.png")
        entries.append((name, frame))
    write_sparse_model(root / "sparse" / "0", scene, entries)
    return writer.commit({
        "scene_name": writer.scene,
        "source_scene": scene.root.parent.name,
        "root_directory": writer.scene,
        "total_images": len(chosen),
        "num_sparse_images": len(chosen),
        "dynamic_objects_masked": True,
        "dense_depth": False,
        "setting": settings["setting"],
        "eval_threshold": float(settings["eval_threshold"]),
        "depth_scale": 1.0,
        "lidar_depth_scale": scene.lidar_depth_scale,
        "source_images": [frame.name for frame in chosen],
    })


def add_mvrecon_dense_depth(scene: SourceScene,
                            scene_dir: pathlib.Path) -> dict:
    """Adds the views' GGGS depth to an MVRecon scene.

    Copies the depth (m) and its mask (255 = valid), and crops the LiDAR map
    to what the views see.

    Returns:
        The updated scene.json record.
    """
    record = json.loads((scene_dir / SCENE_RECORD).read_text(encoding="utf-8"))
    by_name = {frame.name: frame for frame in scene.frames}
    chosen = [by_name[name] for name in record["source_images"]]
    view_depths = []
    with staged_directory(scene_dir / "depths") as depths:
        with staged_directory(scene_dir / "masks") as masks:
            for frame in chosen:
                depth = scene.dense_depth(frame)
                monodepth_layout.save_depth(depths / f"{frame.padded}.npy",
                                            depth)
                monodepth_layout.save_mask(masks / f"{frame.padded}.png", depth
                                           > 0)
                rotation, translation = scene.world_to_camera(frame)
                view_depths.append(
                    (rotation, translation, scene.intrinsics(frame), depth))
    xyz, rgb = read_ply_points(scene.root / "dense_map" / "points3D_dense.ply")
    keep = visible_points(xyz,
                          view_depths,
                          tolerance=4.0 * float(record["eval_threshold"]))
    partial = scene_dir / "gt_points.ply.partial"
    write_ply_points(partial, xyz[keep], rgb[keep])
    partial.replace(scene_dir / "gt_points.ply")
    return update_scene_record(scene_dir,
                               dense_depth=True,
                               gt_point_path="gt_points.ply",
                               gt_points=int(keep.sum()),
                               source_points=len(xyz),
                               **gggs_fields(scene))


# ---- Manual review -----------------------------------------------------------

VIDEO_GT_DIRS = ("depths", "lidar_depths", DENSE_HEATMAP_DIRNAME,
                 HEATMAP_DIRNAME)
CLIP_NAME = re.compile(r"^(?P<scene>.+)_(?:clip\d+|full)_cam\d+$")
SOURCE_IMAGE = re.compile(r"^cam(?P<cam>\d+)/img(?P=cam)_(?P<number>\d+)\.\w+$")
REVIEW_RECORD = "benchmark_review.json"


def load_review(path: pathlib.Path | None) -> dict:
    """Loads the manual review's labels.

    The labels come from the MindCloud monitor's Benchmark Review tab.

    Returns:
        {track folder: {scene: {"keep": bool, "reviewed": bool,
        "drop": [frame ids]}}}; empty when there is no file.
    """
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("tracks", {})


def review_drops(review: dict, track: str,
                 scene_dir: pathlib.Path) -> list[str]:
    """Returns the frames the review leaves out of a MonoDepth/VideoDepth scene.

    A VideoDepth clip adds the images its scene's MonoDepth review leaves out
    (MonoDepth samples the VideoDepth scenes), as clip frame stems.

    Args:
        review: The labels from load_review.
        track: The track folder name.
        scene_dir: The materialized scene.

    Returns:
        The frame ids left out.
    """
    frames = list(review.get(track, {}).get(scene_dir.name, {}).get("drop", []))
    match = CLIP_NAME.match(scene_dir.name)
    if match:
        mono_review = review.get("MonoDepth", {}).get(match["scene"], {})
        mono_drops = set(mono_review.get("drop", []))
    else:
        mono_drops = set()
    manifest_path = scene_dir / "frame_manifest.json"
    if track != "VideoDepth" or not mono_drops or not manifest_path.is_file():
        return frames
    for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
        source = SOURCE_IMAGE.match(entry.get("source_rgb", ""))
        if source is None:
            continue
        cam, number = int(source["cam"]), int(source["number"])
        stem = pathlib.Path(entry["rgb"]).stem
        key = monodepth_layout.frame_key(
            f"cam{cam}/{Frame(cam, number, '').padded}")
        if key in mono_drops and stem not in frames:
            frames.append(stem)
    return frames


def record_review(output_root: pathlib.Path,
                  labels: pathlib.Path | None) -> pathlib.Path | None:
    """Writes `<output_root>/benchmark_review.json`.

    It holds the review labels the benchmark was filtered by and, per
    MonoDepth or VideoDepth scene, the frames left out (VideoDepth: their GT
    only), as the scenes' scene.json record.

    Args:
        output_root: The materialized benchmark.
        labels: The review labels file.

    Returns:
        The file's path, or None when there are no labels or no benchmark.
    """
    if labels is None or not labels.is_file() or not output_root.is_dir():
        return None
    raw = json.loads(labels.read_text(encoding="utf-8"))
    excluded = {}
    for track in ("MonoDepth", "VideoDepth"):
        for record_path in sorted(
            (output_root / track).glob(f"*/{SCENE_RECORD}")):
            scene_record = json.loads(record_path.read_text(encoding="utf-8"))
            frames = scene_record.get("manual_review",
                                      {}).get("excluded_frames")
            if frames:
                excluded.setdefault(track, {})[record_path.parent.name] = frames
    record = {
        "labels_updated": raw.get("updated"),
        "labels": raw.get("tracks", {}),
        "excluded_frames": excluded,
        "note": "MonoDepth frames are removed; VideoDepth frames keep their "
                "image (model input) and lose "
                "their GT depth, so they are not scored."
    }
    path = output_root / REVIEW_RECORD
    text = json.dumps(record, indent=2) + "\n"
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return path


def review_frame_files(
        scene_dir: pathlib.Path,
        frame_id: str,
        dirnames: tuple[str, ...] | None = None) -> list[pathlib.Path]:
    """Returns the files named `<frame_id>.*` directly inside scene folders.

    Args:
        scene_dir: The materialized scene.
        frame_id: The frame's file stem.
        dirnames: The folders to look in; all of them when None.

    Returns:
        The files, sorted.
    """
    if dirnames:
        folders = [scene_dir / name for name in dirnames]
    else:
        folders = [path for path in scene_dir.iterdir() if path.is_dir()]
    files = []
    for folder in folders:
        if folder.is_dir():
            files.extend(
                path for path in folder.glob(f"{frame_id}.*") if path.is_file())
    return sorted(files)


def _drop_monodepth_records(manifest: dict, todo: list[str]) -> dict:
    """Removes frames from a MonoDepth manifest; returns scene.json changes."""
    manifest["records"] = [
        record for record in manifest["records"]
        if record["key"] not in set(todo)
    ]
    manifest["num_samples"] = len(manifest["records"])
    return {
        "total_images": len(manifest["records"]),
        "total_depth_maps": len(manifest["records"])
    }


def _drop_videodepth_gt(manifest: list[dict], by_id: dict[str, dict],
                        todo: list[str]) -> dict:
    """Takes the GT depth off VideoDepth frames; returns scene.json changes."""
    for frame in todo:
        by_id[frame].update(depth=None,
                            depth_wh=None,
                            lidar_depth=None,
                            gt_excluded=True)
    return {
        "gt_frames":
            sum(1 for entry in manifest if not entry.get("gt_excluded"))
    }


def exclude_frames(scene_dir: pathlib.Path,
                   track: str,
                   frame_ids: list[str],
                   apply: bool = True) -> tuple[list[str], list[pathlib.Path]]:
    """Removes the frames the review left out of a MonoDepth/VideoDepth scene.

    A MonoDepth frame is removed from every folder and from
    materialized_full.json. A VideoDepth frame keeps its image, so the clip
    stays continuous, and loses its GT depth (dense and LiDAR, and their
    heatmaps): its frame_manifest.json entry gets `depth: null` and
    `gt_excluded: true`, and since GT and predictions are paired by file name
    it is just not scored. scene.json records the frames (`manual_review`).
    Frames already removed are skipped, so this can run on every
    materialization.

    Args:
        scene_dir: The materialized scene.
        track: "MonoDepth" or "VideoDepth"; other tracks are left alone.
        frame_ids: The frames left out, as file stems.
        apply: Whether to make the changes, or only report them.

    Returns:
        The frames changed and the files removed (with `apply=False`, those it
        would change and remove).
    """
    if track == "MonoDepth":
        manifest_path = scene_dir / monodepth_layout.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        present = {record["key"] for record in manifest.get("records", [])}
        files = {
            frame: review_frame_files(scene_dir, frame) for frame in frame_ids
        }
        todo = [
            frame for frame in frame_ids if frame in present or files[frame]
        ]
    elif track == "VideoDepth":
        manifest_path = scene_dir / "frame_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_id = {pathlib.Path(entry["rgb"]).stem: entry for entry in manifest}
        files = {
            frame: review_frame_files(scene_dir, frame, VIDEO_GT_DIRS)
            for frame in frame_ids
        }
        todo = [
            frame for frame in frame_ids if frame in by_id and
            (files[frame] or not by_id[frame].get("gt_excluded"))
        ]
    else:
        return [], []
    removed = [path for frame in todo for path in files[frame]]
    if not todo or not apply:
        return todo, removed
    for path in removed:
        path.unlink()
    if track == "MonoDepth":
        changes = _drop_monodepth_records(manifest, todo)
    else:
        changes = _drop_videodepth_gt(manifest, by_id, todo)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    record = json.loads((scene_dir / SCENE_RECORD).read_text(encoding="utf-8"))
    excluded = sorted(
        set(record.get("manual_review", {}).get("excluded_frames", [])) |
        set(todo))
    update_scene_record(scene_dir,
                        **changes,
                        manual_review={"excluded_frames": excluded})
    return todo, removed


# ---- Driver ------------------------------------------------------------------


def load_selection(path: pathlib.Path) -> dict:
    """Loads the scene selection (benchmark_selection.yaml)."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def monodepth_scenes(selection: dict) -> list[str]:
    """Returns the MonoDepth scenes.

    MonoDepth's own scenes followed by those of the tracks it samples from,
    less `monodepth.exclude`.
    """
    names = list(selection["monodepth"]["scenes"])
    for track in selection["monodepth"].get("sample_from", []):
        scenes = selection[track]["scenes"]
        names += [name for name in scenes if name not in names]
    excluded = set(selection["monodepth"].get("exclude", []))
    return [name for name in names if name not in excluded]


class Run:
    """Shared state of one materialization run.

    Attributes:
        args: The command-line arguments.
        selection: The scene selection.
        seed: The selection's seed.
        scale: Units of the LiDAR depth PNGs per metre.
        only: The scenes this run is restricted to (empty: all).
        pending: The scenes still waiting for GGGS, as "<track>/<scene>".
        review: The manual review's labels.
    """

    def __init__(self, args: argparse.Namespace, selection: dict):
        self.args = args
        self.selection = selection
        self.seed = int(selection["seed"])
        self.scale = float(selection["lidar_depth_scale"])
        self.only = set(args.scenes)
        self.pending: list[str] = []
        self.review = load_review(args.review)

    def wanted(self, name: str) -> bool:
        """Whether this run builds scene `name`."""
        return not self.only or name in self.only

    def source(self, name: str) -> SourceScene:
        """The released scene `name`."""
        return SourceScene(name, find_scene_root(self.args.released_root, name),
                           self.scale, self.args.allow_stale_lidar_depth)

    def log(self, message: str) -> None:
        """Prints a progress message."""
        print(message, flush=True)

    def track_root(self, task: str) -> pathlib.Path:
        """The output folder of track `task`."""
        return self.args.output_root / task

    def current_record(self, writer: SceneWriter,
                       scene: SourceScene) -> dict | None:
        """Returns the scene's record if it is complete and current.

        Current means masked as its track needs and its dense depth (if any)
        from the current GGGS generation. A depth-track scene written before
        people and vehicles were masked is masked in place; a SLAM sequence
        written at full resolution, or a scene whose GGGS run was retrained
        (or is being retrained), is deleted, to be built again. A MonoDepth
        scene is the exception to the last: it keeps its frames, and so their
        manual review (refresh_monodepth_depth); while the new full generation
        is not done it keeps its current depth.
        """
        record = writer.existing_record()
        track = writer.track_root.name
        if (record is not None and track == "SLAM" and
                not record.get("input_size")):
            self.log(f"  {writer.scene}: written before SLAM frames were "
                     "pre-sized; removed to be rebuilt")
            shutil.rmtree(writer.final)
            return None
        if (record is not None and track in MASKED_TRACKS and
                not record.get("dynamic_objects_masked")):
            record = mask_dynamic_objects(scene, writer.final)
            self.log(f"  {writer.scene}: people and vehicles masked out of "
                     "its images and LiDAR depth")
        if record is None:
            return record
        if not (record.get("gggs") or record.get("provenance", {}).get("gggs")):
            return record
        return self._current_gggs_record(writer, scene, record)

    def _current_gggs_record(self, writer: SceneWriter, scene: SourceScene,
                             record: dict) -> dict | None:
        """The part of current_record for a scene with GGGS depth."""
        stamp = record.get("gggs_generated")
        if scene.generation is not None:
            if stamp == scene.generation["stamp"]:
                return record
            built = (writer.final / SCENE_RECORD).stat().st_mtime
            if stamp is None:
                generation_path = pathlib.Path(scene.generation["path"])
                if built >= generation_path.stat().st_mtime:
                    # The scene predates the stamps.
                    return update_scene_record(
                        writer.final, gggs_generated=scene.generation["stamp"])
        if writer.track_root.name == "MonoDepth" and not self.args.overwrite:
            if scene.generation is None:
                if f"MonoDepth/{writer.scene}" not in self.pending:
                    self.pending.append(f"MonoDepth/{writer.scene}")
                    self.log(f"  {writer.scene}: GGGS is being retrained; "
                             "kept with its current depth until the new "
                             "full generation is done")
                return record
            record = refresh_monodepth_depth(scene, writer.final)
            self.log(f"  {writer.scene}: GGGS output changed; the depth of "
                     f"its {record.get('total_images')} frames "
                     f"replaced with {record['gggs']}'s, the frames kept")
            return record
        self.log(f"  {writer.scene}: GGGS output changed since it was "
                 "written; removed to be rebuilt")
        shutil.rmtree(writer.final)
        return None

    def left_out(self, track: str, scene: str) -> bool:
        """Whether the manual review left the whole scene out.

        It is then skipped, and pruned like a deselected one.
        """
        label = self.review.get(track, {}).get(scene, {})
        return label.get("keep", True) is False

    def reviewed(self, track: str, scene_dir: pathlib.Path,
                 record: dict) -> dict:
        """Removes the frames the manual review left out of a scene.

        This happens once; later runs find nothing to do.

        Returns:
            The scene's current scene.json record.
        """
        frames = review_drops(self.review, track, scene_dir)
        if frames:
            changed, removed = exclude_frames(scene_dir, track, frames)
        else:
            changed, removed = [], []
        if not changed:
            return record
        self.log(f"  {scene_dir.name}: {len(changed)} frames left out by the "
                 f"manual review ({len(removed)} files removed)")
        return json.loads(
            (scene_dir / SCENE_RECORD).read_text(encoding="utf-8"))

    def prune(self, root: pathlib.Path, kept: set[str]) -> None:
        """Deletes scenes no longer in the selection.

        Only on a full run, not one restricted by --scenes.
        """
        if self.only or not root.is_dir():
            return
        for scene_dir in sorted(root.iterdir()):
            if ((scene_dir / SCENE_RECORD).is_file() and
                    scene_dir.name not in kept):
                shutil.rmtree(scene_dir)
                self.log(f"  {scene_dir.name}: removed, no longer selected")

    def run_scene(self, writer: SceneWriter, scene: SourceScene, build) -> dict:
        """Builds a scene unless it is complete.

        Then makes sure its LiDAR depth heatmaps are there.

        Args:
            writer: The output scene.
            scene: The source scene.
            build: Called with no arguments to build the scene; returns its
                scene.json record.

        Returns:
            The scene's scene.json record.
        """
        record = self.current_record(writer, scene)
        if record is not None:
            self.log(f"  {writer.scene}: done already "
                     f"({record.get('total_images')} images)")
        else:
            record = build()
            self.log(f"  {writer.scene}: {record.get('total_images')} images")
        if record.get("input_size"):
            fit_to = tuple(record["input_size"])
        else:
            fit_to = None
        mask_dynamic = writer.track_root.name in MASKED_TRACKS
        added = add_lidar_heatmaps(scene,
                                   writer.final,
                                   mask_dynamic=mask_dynamic,
                                   fit_to=fit_to)
        if added:
            self.log(f"  {writer.scene}: {added} LiDAR depth heatmaps added")
        if ((writer.final / HEATMAP_DIRNAME).is_dir() and
                record.get("lidar_depth_heatmaps") != HEATMAP_DIRNAME):
            record = update_scene_record(writer.final,
                                         lidar_depth_heatmaps=HEATMAP_DIRNAME)
        return record

    def dense_heatmaps(self, scene_dir: pathlib.Path, record: dict) -> dict:
        """Makes sure a scene with dense depth has its depth heatmaps."""
        added = add_dense_heatmaps(scene_dir, self.scale)
        if added:
            self.log(f"  {scene_dir.name}: {added} dense depth heatmaps added")
        if ((scene_dir / DENSE_HEATMAP_DIRNAME).is_dir() and
                record.get("depth_heatmaps") != DENSE_HEATMAP_DIRNAME):
            record = update_scene_record(scene_dir,
                                         depth_heatmaps=DENSE_HEATMAP_DIRNAME)
        return record

    def needs_gggs(self, track: str, scene: SourceScene) -> bool:
        """Whether the scene's GGGS output exists; records it as pending if not.

        Returns:
            True when the GGGS full generation is done.
        """
        if scene.generation is None:
            self.pending.append(f"{track}/{scene.name}")
            self.log(f"  {scene.name}: skipped, GGGS full generation not done")
            return False
        return True

    def sfm(self) -> None:
        """Materializes the SfM track."""
        root = self.track_root("SfM")
        self.log(f"SfM -> {root}")
        records = []
        names = [
            name for name in self.selection["sfm"]["scenes"]
            if not self.left_out("SfM", name)
        ]
        for name in names:
            if self.wanted(name):
                scene = self.source(name)
                writer = SceneWriter(root, name, self.args.overwrite)
                records.append(
                    self.run_scene(
                        writer,
                        scene,
                        lambda scene=scene, writer=writer:
                        materialize_sfm_scene(scene, writer),
                    ))
        self.prune(root, set(names))
        write_track_metadata(root, "SfM", records, depth_png_scale=self.scale)

    def slam(self) -> None:
        """Materializes the SLAM track."""
        config = self.selection["slam"]
        fps = float(config["frame_rate_hz"])
        root = self.track_root("SLAM")
        self.log(f"SLAM -> {root}")
        records = []
        sequences = [
            sequence for sequence in config["sequences"]
            if not self.left_out("SLAM", sequence["scene"])
        ]
        for sequence in sequences:
            if not self.wanted(sequence["scene"]):
                continue
            scene = self.source(sequence["scene"])
            writer = SceneWriter(root, sequence["scene"], self.args.overwrite)
            self.log(f"  {sequence['scene']}: {sequence['range']} range, "
                     f"cam{sequence['camera']}, {sequence['input']}")
            records.append(
                self.run_scene(
                    writer,
                    scene,
                    lambda scene=scene, writer=writer, sequence=sequence:
                    materialize_slam_sequence(scene, writer, sequence, fps),
                ))
        self.prune(root, {sequence["scene"] for sequence in sequences})
        write_track_metadata(root,
                             "SLAM",
                             records,
                             frame_rate_hz=fps,
                             depth_png_scale=self.scale)

    def monodepth(self) -> None:
        """Materializes the MonoDepth track."""
        config = self.selection["monodepth"]
        root = self.track_root("MonoDepth")
        self.log(f"MonoDepth -> {root}")
        records = []
        names = [
            name for name in monodepth_scenes(self.selection)
            if not self.left_out("MonoDepth", name)
        ]
        for name in names:
            if not self.wanted(name):
                continue
            scene = self.source(name)
            writer = SceneWriter(root, name, self.args.overwrite)
            existing = self.current_record(writer, scene)
            if existing is None and not self.needs_gggs("MonoDepth", scene):
                continue
            record = self.run_scene(
                writer,
                scene,
                lambda scene=scene, writer=writer: materialize_monodepth_scene(
                    scene, writer, config, self.seed),
            )
            if self.args.restore_review_drops:
                restored = restore_monodepth_frames(scene, writer.final)
                if restored:
                    self.log(f"  {writer.scene}: {len(restored)} frames left "
                             "out by the manual review put back for "
                             "another review")
                    add_lidar_heatmaps(scene, writer.final, mask_dynamic=True)
                record = json.loads(
                    (writer.final / SCENE_RECORD).read_text(encoding="utf-8"))
                records.append(self.dense_heatmaps(writer.final, record))
                continue
            records.append(
                self.reviewed("MonoDepth", writer.final,
                              self.dense_heatmaps(writer.final, record)))
        self.prune(root, set(names))
        if records or root.is_dir():
            write_track_metadata(root, "MonoDepth", records)

    def add_dense_depth(self, track: str, scene: SourceScene, record: dict,
                        add) -> dict:
        """Completes a materialized scene with its GGGS depth when that exists.

        Otherwise the scene is reported as waiting.

        Args:
            track: The track folder name.
            scene: The source scene.
            record: The scene's scene.json record.
            add: Called as `add(scene, scene_dir)` to add the dense depth;
                returns the updated record.

        Returns:
            The scene's scene.json record.
        """
        scene_dir = self.track_root(track) / record["scene_name"]
        if not record.get("dense_depth"):
            if scene.generation is None:
                self.pending.append(f"{track}/{record['scene_name']}")
                return record
            record = add(scene, scene_dir)
            self.log(f"  {record['scene_name']}: GGGS depth added")
        return self.dense_heatmaps(scene_dir, record)

    def videodepth(self) -> None:
        """Materializes the VideoDepth track."""
        config = self.selection["videodepth"]
        root = self.track_root("VideoDepth")
        self.log(f"VideoDepth -> {root}")
        records = []
        plan = {}
        excluded = set(config.get("exclude_clips", []))
        for name in config["scenes"]:
            if not self.wanted(name):
                continue
            scene = self.source(name)
            # A clip left out (here or by the review) keeps its place, so the
            # other clips do not change.
            clips = [
                clip for clip in videodepth_clips(scene, config, self.seed)
                if clip["scene_name"] not in excluded and
                not self.left_out("VideoDepth", clip["scene_name"])
            ]
            if clips:
                plan[name] = clips
            for clip in clips:
                writer = SceneWriter(root, clip["scene_name"],
                                     self.args.overwrite)
                record = self.run_scene(
                    writer,
                    scene,
                    lambda scene=scene, writer=writer, clip=clip:
                    materialize_videodepth_clip(scene, writer, clip),
                )
                record = self.add_dense_depth("VideoDepth", scene, record,
                                              add_videodepth_dense_depth)
                records.append(self.reviewed("VideoDepth", writer.final,
                                             record))
        self.prune(
            root,
            {clip["scene_name"] for clips in plan.values() for clip in clips})
        root.mkdir(parents=True, exist_ok=True)
        clips_record = {"seed": self.seed, "clips": plan}
        (root / "clips.json").write_text(json.dumps(clips_record, indent=2) +
                                         "\n",
                                         encoding="utf-8")
        write_track_metadata(root,
                             "VideoDepth",
                             records,
                             depth_png_scale=self.scale)

    def mvrecon(self) -> None:
        """Materializes the MVRecon track."""
        config = self.selection["mvrecon"]
        root = self.track_root("MVRecon")
        self.log(f"MVRecon -> {root}")
        records = []
        scenes = {
            name: settings
            for name, settings in config["scenes"].items()
            if not self.left_out("MVRecon", name)
        }
        for name, settings in scenes.items():
            if not self.wanted(name):
                continue
            scene = self.source(name)
            writer = SceneWriter(root, name, self.args.overwrite)
            existing = writer.existing_record()
            if existing is not None:
                expected = mvrecon_view_count(scene, int(config["views"]))
                if existing.get("total_images") != expected:
                    self.log(f"  {name}: written with another number of "
                             "views; removed to be rebuilt")
                    shutil.rmtree(writer.final)
            record = self.run_scene(
                writer,
                scene,
                lambda scene=scene, writer=writer, settings=settings:
                materialize_mvrecon_scene(scene, writer, settings,
                                          int(config["views"]), self.seed),
            )
            records.append(
                self.add_dense_depth("MVRecon", scene, record,
                                     add_mvrecon_dense_depth))
        self.prune(root, set(scenes))
        write_track_metadata(root,
                             "MVRecon",
                             records,
                             depth_png_scale=self.scale)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line (`argv`, default sys.argv)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--released_root",
                        type=pathlib.Path,
                        default=os.environ.get("MINDCLOUD_RELEASED_ROOT"),
                        help=("Folder of the released recordings "
                              "(<Scene>-<date>-<time>/convert_final); "
                              "default $MINDCLOUD_RELEASED_ROOT."))
    parser.add_argument(
        "--output_root",
        type=pathlib.Path,
        required=True,
        help=("Folder to write the tracks into "
              "(one subfolder per track, scenes directly inside)."))
    parser.add_argument("--tracks",
                        nargs="+",
                        choices=TRACKS,
                        default=list(TRACKS))
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=[],
        help="Restrict to these scenes (default: all selected).")
    parser.add_argument("--selection",
                        type=pathlib.Path,
                        default=SELECTION_PATH)
    parser.add_argument(
        "--review",
        type=pathlib.Path,
        default=REVIEW_PATH,
        help=("The manual review's labels (the MindCloud monitor's "
              "Benchmark Review tab): scenes, clips "
              "and frames left out are skipped or removed. "
              "Default: assets/benchmark_review.json, if "
              "it exists. They are recorded in "
              "<output_root>/benchmark_review.json."))
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Rebuild scenes that are already complete.")
    parser.add_argument(
        "--restore_review_drops",
        action="store_true",
        help=("Put the MonoDepth frames the manual review left out back, "
              "with the current GGGS depth, "
              "to review them again; the next run without this flag "
              "removes those still left out."))
    parser.add_argument("--allow_stale_lidar_depth",
                        action="store_true",
                        help="Accept lidar_depth older than the GT poses.")
    args = parser.parse_args(argv)
    if args.released_root is None:
        parser.error(
            "--released_root (or $MINDCLOUD_RELEASED_ROOT) is required")
    return args


def main(argv: list[str] | None = None) -> int:
    """Materializes the selected tracks; returns the exit status."""
    args = parse_args(argv)
    run = Run(args, load_selection(args.selection))
    for track in TRACKS:
        if track in args.tracks:
            getattr(run, track)()
    record_review(args.output_root, args.review)
    if run.pending:
        run.log(f"Waiting for GGGS: {', '.join(run.pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
