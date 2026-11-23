# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Shared COLMAP-to-real-image resolution for camera-conditioned workflows.

`resolve_colmap_image_mappings` maps each record of a COLMAP model to one real
image file of a scene (exact path, nested suffix, then basename or stem), and
reports the records that are missing or ambiguous instead of guessing.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import numpy as np

from spatialarena.geometry import read_write_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Glob patterns of the adapters' frame list
# (scripts/inference/adapter_utils.find_image_paths); the SLAM evaluator lists a
# scene's frames with the same patterns.
IMAGE_PATTERNS = tuple(f"*{ext}" for ext in (".jpg", ".jpeg", ".png", ".bmp",
                                             ".tif", ".tiff", ".webp")
                       for ext in (ext, ext.upper()))


class ColmapImageResolutionError(FileNotFoundError):
    """Raised when a COLMAP image record cannot be resolved unambiguously."""


@dataclasses.dataclass(frozen=True)
class ColmapImageRecord:
    """One COLMAP image with its camera.

    Attributes:
        image_id: COLMAP image id.
        name: Image name as stored in the model.
        camera_id: COLMAP camera id.
        model: COLMAP camera model name.
        width: Image width in pixels.
        height: Image height in pixels.
        params: COLMAP camera parameters.
        intrinsic: 3x3 pinhole intrinsic matrix.
        w2c: 4x4 world-to-camera matrix.
    """
    image_id: int
    name: str
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray
    intrinsic: np.ndarray
    w2c: np.ndarray

    @property
    def c2w(self) -> np.ndarray:
        """The 4x4 camera-to-world matrix (float32)."""
        return np.linalg.inv(self.w2c).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class ColmapImageMapping:
    """A COLMAP record matched to a real image file.

    Attributes:
        record: The COLMAP record.
        image_path: The matched real image file.
        match_kind: How it was matched, e.g. "exact_relative".
    """
    record: ColmapImageRecord
    image_path: pathlib.Path
    match_kind: str


@dataclasses.dataclass(frozen=True)
class ColmapMappingDiagnostics:
    """What a mapping run matched, missed and could not decide.

    Attributes:
        mapped: One {colmap_name, image_path, match_kind} dict per match.
        missing: Record names without any candidate file.
        ambiguous: Records with several candidates, or a file already used.
    """
    mapped: tuple[dict[str, str], ...]
    missing: tuple[str, ...]
    ambiguous: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns the diagnostics as a JSON-ready dict with counts."""
        return {
            "mapped": [dict(item) for item in self.mapped],
            "missing": list(self.missing),
            "ambiguous": [dict(item) for item in self.ambiguous],
            "num_mapped": len(self.mapped),
            "num_missing": len(self.missing),
            "num_ambiguous": len(self.ambiguous),
        }


def canonical_image_root(data_path: pathlib.Path) -> pathlib.Path:
    """Returns the canonical RGB root, including ETH3D undistorted DSLR RGB."""
    data_path = pathlib.Path(data_path)
    undistorted = data_path / "images" / "dslr_images_undistorted"
    if undistorted.is_dir():
        return undistorted
    return data_path / "images"


def list_real_image_paths(data_path: pathlib.Path) -> list[pathlib.Path]:
    """Lists the image files under the canonical image root, sorted."""
    root = canonical_image_root(data_path)
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _normalize(value: str | pathlib.Path) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _parts(value: str | pathlib.Path) -> tuple[str, ...]:
    return tuple(part for part in _normalize(value).split("/") if part)


def _stem_key(value: str | pathlib.Path) -> str:
    stem = pathlib.Path(_normalize(value)).stem.casefold()
    for suffix in (".color", ".depth.proj", ".depth", "_depth"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    for prefix in ("depth_map_", "depth_", "disp_", "frame_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def _camera_matrix(camera: Any) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    model = str(camera.model).upper()
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "FOV"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {camera.model}")
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def load_colmap_image_records(
        colmap_path: pathlib.Path) -> list[ColmapImageRecord]:
    """Reads a COLMAP model into records sorted by image name.

    Raises:
        ColmapImageResolutionError: If the model cannot be read.
    """
    model = read_write_model.read_model(str(colmap_path))
    if model is None:
        raise ColmapImageResolutionError(
            f"Unable to read COLMAP model: {colmap_path}")
    cameras, images, _points = model
    records = []
    for image in sorted(images.values(), key=lambda item: item.name):
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = read_write_model.qvec2rotmat(image.qvec).astype(
            np.float32)
        w2c[:3, 3] = np.asarray(image.tvec, dtype=np.float32)
        records.append(
            ColmapImageRecord(
                image_id=int(image.id),
                name=str(image.name),
                camera_id=int(image.camera_id),
                model=str(cameras[image.camera_id].model),
                width=int(cameras[image.camera_id].width),
                height=int(cameras[image.camera_id].height),
                params=np.asarray(cameras[image.camera_id].params,
                                  dtype=np.float32),
                intrinsic=_camera_matrix(cameras[image.camera_id]),
                w2c=w2c,
            ))
    return records


def _real_path_keys(path: pathlib.Path,
                    image_root: pathlib.Path) -> tuple[str, ...]:
    relative = path.relative_to(image_root).as_posix()
    return tuple(dict.fromkeys((_normalize(relative), relative.casefold())))


def _record_exact_keys(name: str) -> tuple[str, ...]:
    normalized = _normalize(name)
    return tuple(dict.fromkeys((normalized, normalized.casefold())))


def _suffix_candidates(record_name: str, image_paths: list[pathlib.Path],
                       image_root: pathlib.Path) -> list[pathlib.Path]:
    record_parts = tuple(part.casefold() for part in _parts(record_name))
    if not record_parts:
        return []
    candidates = []
    for path in image_paths:
        real_parts = tuple(
            part.casefold() for part in _parts(path.relative_to(image_root)))
        if len(real_parts) >= len(record_parts) and real_parts[
                -len(record_parts):] == record_parts:
            candidates.append(path)
    return candidates


# pylint: disable-next=unused-argument
def _basename_candidates(record_name: str, image_paths: list[pathlib.Path],
                         image_root: pathlib.Path) -> list[pathlib.Path]:
    record_path = pathlib.Path(_normalize(record_name))
    basename = record_path.name.casefold()
    stem = _stem_key(record_name)
    candidates = []
    for path in image_paths:
        if path.name.casefold() == basename or _stem_key(path) == stem:
            candidates.append(path)
    return candidates


def resolve_colmap_image_mappings(
    data_path: pathlib.Path,
    image_paths: list[str | pathlib.Path] | None,
    *,
    colmap_path: pathlib.Path,
    strict: bool = True,
) -> tuple[list[ColmapImageMapping], ColmapMappingDiagnostics]:
    """Maps COLMAP records to real files in authoritative COLMAP order.

    Matching priority is exact relative path, nested suffix, basename, then
    stem. Ambiguous candidates are never guessed.

    Args:
        data_path: Scene root.
        image_paths: Candidate image files; defaults to all images under the
            canonical image root.
        colmap_path: COLMAP model folder.
        strict: Raise if any record is missing or ambiguous.

    Returns:
        (mappings, diagnostics).

    Raises:
        ColmapImageResolutionError: If no record maps, or in strict mode if
            any record is missing or ambiguous.
    """
    image_root = canonical_image_root(data_path)
    real_paths = [
        pathlib.Path(path)
        for path in (image_paths or list_real_image_paths(data_path))
    ]
    real_paths = list(dict.fromkeys(real_paths))
    exact_index: dict[str, list[pathlib.Path]] = {}
    for path in real_paths:
        for key in _real_path_keys(path, image_root):
            exact_index.setdefault(key, []).append(path)

    mappings: list[ColmapImageMapping] = []
    mapped_records: list[dict[str, str]] = []
    missing: list[str] = []
    ambiguous: list[dict[str, Any]] = []
    used_paths: set[pathlib.Path] = set()

    for record in load_colmap_image_records(colmap_path):
        exact = []
        for key in _record_exact_keys(record.name):
            exact.extend(exact_index.get(key, []))
        exact = list(dict.fromkeys(exact))
        if len(exact) == 1:
            candidates = exact
            match_kind = "exact_relative"
        elif len(exact) > 1:
            candidates = exact
            match_kind = "ambiguous_exact_relative"
        else:
            candidates = _suffix_candidates(record.name, real_paths, image_root)
            match_kind = "nested_prefix"
            if len(candidates) == 0:
                candidates = _basename_candidates(record.name, real_paths,
                                                  image_root)
                match_kind = "basename_or_stem"

        if len(candidates) != 1:
            if len(candidates) == 0:
                missing.append(record.name)
            else:
                ambiguous.append({
                    "colmap_name": record.name,
                    "candidates": [str(path) for path in candidates],
                    "match_kind": match_kind,
                })
            continue

        image_path = candidates[0]
        if image_path in used_paths:
            ambiguous.append({
                "colmap_name": record.name,
                "candidates": [str(image_path)],
                "match_kind": "duplicate_real_path",
            })
            continue
        used_paths.add(image_path)
        mappings.append(ColmapImageMapping(record, image_path, match_kind))
        mapped_records.append({
            "colmap_name": record.name,
            "image_path": str(image_path),
            "match_kind": match_kind,
        })

    diagnostics = ColmapMappingDiagnostics(
        mapped=tuple(mapped_records),
        missing=tuple(missing),
        ambiguous=tuple(ambiguous),
    )
    if strict and (missing or ambiguous):
        details = diagnostics.to_dict()
        raise ColmapImageResolutionError(
            "No COLMAP camera-prior image mapping: "
            f"mapped={details['num_mapped']} missing={details['num_missing']} "
            f"ambiguous={details['num_ambiguous']}; "
            f"missing={list(missing[:5])}; ambiguous={list(ambiguous[:3])}")
    if not mappings:
        raise ColmapImageResolutionError(
            "No COLMAP camera-prior records map to real images under "
            f"{image_root}")
    return mappings, diagnostics
