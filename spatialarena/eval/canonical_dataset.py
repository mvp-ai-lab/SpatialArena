# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Dataset adapters for canonical benchmark layouts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class BenchmarkSceneInfo:
    # Dataset name.
    dataset: str
    # Benchmark task name, e.g. MonoDepth, VideoDepth, SfM, or MVRecon.
    task: str
    # Scene or subset name.
    scene: str
    # Scene root directory.
    root_path: Path
    # Canonical image directory when materialized.
    image_path: Path | None = None
    # Canonical ground-truth depth directory when available.
    depth_gt_path: Path | None = None
    # Canonical mask directory when available.
    mask_path: Path | None = None
    # Canonical ground-truth sparse COLMAP path when available.
    sparse_gt_path: Path | None = None
    # Raw metadata entry from metadata.json.
    metadata: dict | None = None


class CanonicalDataset:
    """Read scenes from a dataset root with a metadata.json file."""

    dataset_name: str = ""
    task_name: str = ""

    def __init__(
        self,
        data_path: Path,
        scenes: list[str] | None = None,
        task_name: str | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.scenes = set(scenes or [])
        self.task_name = task_name or self.task_name

    @property
    def metadata_path(self) -> Path:
        return self.data_path / "metadata.json"

    def load_metadata(self) -> dict:
        with self.metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def list_scenes(self) -> list[BenchmarkSceneInfo]:
        metadata = self.load_metadata()
        dataset = self.dataset_name or metadata["dataset_name"]
        scene_infos: list[BenchmarkSceneInfo] = []

        for scene_meta in metadata.get("scenes", []):
            scene_name = scene_meta["scene_name"]
            if self.scenes and scene_name not in self.scenes:
                continue
            scene_root = self._scene_root(scene_meta)
            scene_infos.append(
                BenchmarkSceneInfo(
                    dataset=dataset,
                    task=self.task_name,
                    scene=scene_name,
                    root_path=scene_root,
                    image_path=self._optional_path(scene_root / "images"),
                    depth_gt_path=self._optional_path(scene_root / "depths"),
                    mask_path=self._optional_path(scene_root / "masks"),
                    sparse_gt_path=self._optional_path(scene_root / "sparse" / "0"),
                    metadata=scene_meta,
                )
            )

        return scene_infos

    @staticmethod
    def _optional_path(path: Path) -> Path | None:
        return path if path.exists() else None

    def _scene_root(self, scene_meta: dict) -> Path:
        canonical_root = self.data_path / scene_meta["scene_name"]
        if canonical_root.exists():
            return canonical_root
        return Path(scene_meta["root_directory"])
