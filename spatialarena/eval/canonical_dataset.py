# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Dataset adapters for canonical benchmark layouts.

A canonical dataset root holds a metadata.json file that lists its scenes and,
per scene, a directory with images/, depths/, masks/ and sparse/0/. The
per-dataset modules (e.g. `sintel_dataset`) subclass `CanonicalDataset` and
only set the dataset and task names.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib


@dataclasses.dataclass(kw_only=True)
class BenchmarkSceneInfo:
    """One benchmark scene and its canonical directories.

    Attributes:
        dataset: Dataset name.
        task: Benchmark task name, e.g. MonoDepth, VideoDepth, SfM, or MVRecon.
        scene: Scene or subset name.
        root_path: Scene root directory.
        image_path: Canonical image directory when materialized.
        depth_gt_path: Canonical ground-truth depth directory when available.
        mask_path: Canonical mask directory when available.
        sparse_gt_path: Canonical ground-truth sparse COLMAP path when
            available.
        metadata: Raw metadata entry from metadata.json.
    """

    dataset: str
    task: str
    scene: str
    root_path: pathlib.Path
    image_path: pathlib.Path | None = None
    depth_gt_path: pathlib.Path | None = None
    mask_path: pathlib.Path | None = None
    sparse_gt_path: pathlib.Path | None = None
    metadata: dict | None = None


class CanonicalDataset:
    """Reads scenes from a dataset root with a metadata.json file.

    Attributes:
        dataset_name: Dataset name; falls back to metadata.json's
            "dataset_name" when empty.
        task_name: Benchmark task the scenes are listed for.
        data_path: Dataset root.
        scenes: Scene names to keep; empty keeps every scene.
    """

    dataset_name: str = ""
    task_name: str = ""

    def __init__(
        self,
        data_path: pathlib.Path,
        scenes: list[str] | None = None,
        task_name: str | None = None,
    ) -> None:
        self.data_path = pathlib.Path(data_path)
        self.scenes = set(scenes or [])
        self.task_name = task_name or self.task_name

    @property
    def metadata_path(self) -> pathlib.Path:
        """The dataset's metadata.json."""
        return self.data_path / "metadata.json"

    def load_metadata(self) -> dict:
        """Returns the parsed metadata.json."""
        with self.metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def list_scenes(self) -> list[BenchmarkSceneInfo]:
        """Returns the selected scenes, in metadata.json order."""
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
                    sparse_gt_path=self._optional_path(scene_root / "sparse" /
                                                       "0"),
                    metadata=scene_meta,
                ))

        return scene_infos

    # Kept as a static method: subclasses in pose_datasets.py call
    # self._optional_path.
    @staticmethod
    def _optional_path(path: pathlib.Path) -> pathlib.Path | None:
        return path if path.exists() else None

    def _scene_root(self, scene_meta: dict) -> pathlib.Path:
        canonical_root = self.data_path / scene_meta["scene_name"]
        if canonical_root.exists():
            return canonical_root
        return pathlib.Path(scene_meta["root_directory"])
