# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Dataset registry for the benchmark task layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from spatialarena.paths import SPATIALARENA_ROOT  # noqa: E402
from spatialarena.eval.bonn_dataset import DatasetBonn
from spatialarena.eval.canonical_dataset import CanonicalDataset
from spatialarena.eval.imc_dataset import DatasetIMC
from spatialarena.eval.kitti_dataset import DatasetKitti
from spatialarena.eval.marigold_dataset import DatasetMarigold
from spatialarena.eval.monodepth_datasets import (
    DatasetAerialMetric,
    DatasetHammer,
    DatasetIBims1,
    DatasetSpring,
    DatasetSynth4K,
)
from spatialarena.eval.nrgbd_dataset import DatasetNRGBD
from spatialarena.eval.pose_datasets import (
    DatasetDTUPose,
    DatasetEth3DMVRecon,
    DatasetEth3DPose,
    DatasetSevenScenesSLAM,
    DatasetTanksAndTemplesMVRecon,
    DatasetTanksAndTemplesPose,
    DatasetOxfordSpiresSLAM,
    DatasetTUMDynamicsSLAM,
    DatasetVBRSLAM,
    DatasetKITTIOdometrySLAM,
)
from spatialarena.eval.seven_scenes_dataset import DatasetSevenScenes
from spatialarena.eval.sintel_dataset import DatasetSintel
from spatialarena.spatialarena import SPATIALARENA, TASKS as SPATIALARENA_TASKS, track_root_override


DEFAULT_BENCHMARK_ROOT = Path(f"{SPATIALARENA_ROOT}/benchmarks")
TASK_ALIASES = {
    "CameraPose": "SfM",
    "TexturePoorSfM": "WeakTextureSfM",
    "WeakTextureSfM": "SfM",
}
TASK_DISPLAY_NAMES = {
    "MonoDepth": "MonoDepth",
    "VideoDepth": "VideoDepth",
    "MVRecon": "MVRecon",
    "SfM": "SfM",
    "SLAM": "SLAM",
}


@dataclass(frozen=True, kw_only=True)
class DatasetSpec:
    task: str
    dataset: str
    relative_path: Path
    adapter_cls: type[CanonicalDataset] = CanonicalDataset
    metadata_required: bool = True
    legacy_relative_paths: tuple[Path, ...] = ()
    # Frames a full-depth/full-pose case of this dataset feeds a method: None = the suite's cap (100, DA3-BENCH's
    # max_frames), 0 = every frame. run_method_case_list.sh applies it per case (BENCHMARK_MAX_INPUT_FRAMES and
    # MAX_IMAGES_PER_RUN).
    max_input_frames: int | None = None

    def data_path(self, benchmark_root: Path) -> Path:
        primary = benchmark_root / self.relative_path
        if primary.exists():
            return primary
        for legacy_path in self.legacy_relative_paths:
            candidate = benchmark_root / legacy_path
            if candidate.exists():
                return candidate
        return primary

    def create_dataset(
        self,
        benchmark_root: Path,
        scenes: list[str] | None = None,
    ) -> CanonicalDataset:
        data_path = self.data_path(benchmark_root)
        task_name = canonical_task_name(self.task)
        if self.adapter_cls is CanonicalDataset:
            return self.adapter_cls(data_path, scenes=scenes, task_name=task_name)
        return self.adapter_cls(data_path, scenes=scenes)


@dataclass(frozen=True, kw_only=True)
class SpatialArenaSpec(DatasetSpec):
    """SpatialArena (spatialarena/spatialarena.py): ``$SPATIALARENA_ROOT/<Task>`` when set, else the usual
    ``<benchmark_root>/<Task>/spatialarena``."""

    def data_path(self, benchmark_root: Path) -> Path:
        override = track_root_override(self.task)
        return override if override is not None else super().data_path(benchmark_root)


# SpatialArena's VideoDepth clips (132-1,021 frames) are fed whole, its MVRecon scenes (every image, 363-1,806) and
# SfM scenes (about 1,100 images over the 3 cameras) on 300 images; every other dataset keeps the suites' 100-frame
# cap.
SPATIALARENA_MAX_INPUT_FRAMES = {"VideoDepth": 0, "MVRecon": 300, "SfM": 300}
SPATIALARENA_SPECS = tuple(
    SpatialArenaSpec(task=task, dataset=SPATIALARENA, relative_path=Path(task) / SPATIALARENA,
                     max_input_frames=SPATIALARENA_MAX_INPUT_FRAMES.get(task))
    for task in SPATIALARENA_TASKS
)


DEFAULT_DATASET_SPECS = (
    DatasetSpec(
        task="MonoDepth",
        dataset="marigold",
        relative_path=Path("MonoDepth/marigold"),
        adapter_cls=DatasetMarigold,
    ),
    DatasetSpec(
        task="MonoDepth",
        dataset="aerialmetric",
        relative_path=Path("MonoDepth/aerialmetric"),
        adapter_cls=DatasetAerialMetric,
    ),
    DatasetSpec(
        task="MonoDepth",
        dataset="hammer",
        relative_path=Path("MonoDepth/hammer"),
        adapter_cls=DatasetHammer,
    ),
    DatasetSpec(
        task="MonoDepth",
        dataset="ibims1",
        relative_path=Path("MonoDepth/ibims1"),
        adapter_cls=DatasetIBims1,
    ),
    DatasetSpec(
        task="MonoDepth",
        dataset="spring",
        relative_path=Path("MonoDepth/spring"),
        adapter_cls=DatasetSpring,
    ),
    DatasetSpec(
        task="MonoDepth",
        dataset="synth4k",
        relative_path=Path("MonoDepth/synth4k"),
        adapter_cls=DatasetSynth4K,
    ),
    DatasetSpec(
        task="VideoDepth",
        dataset="sintel",
        relative_path=Path("VideoDepth/sintel"),
        adapter_cls=DatasetSintel,
    ),
    DatasetSpec(
        task="VideoDepth",
        dataset="bonn",
        relative_path=Path("VideoDepth/bonn"),
        adapter_cls=DatasetBonn,
    ),
    DatasetSpec(task="VideoDepth", dataset="dycheck", relative_path=Path("VideoDepth/dycheck")),
    DatasetSpec(
        task="VideoDepth",
        dataset="kitti",
        relative_path=Path("VideoDepth/kitti"),
        adapter_cls=DatasetKitti,
    ),
    DatasetSpec(
        task="SfM",
        dataset="eth3d",
        relative_path=Path("SfM/eth3d"),
        adapter_cls=DatasetEth3DPose,
    ),
    DatasetSpec(
        task="SfM",
        dataset="tanks_and_temples",
        relative_path=Path("3DGS/tanks_and_temples"),
        adapter_cls=DatasetTanksAndTemplesPose,
        metadata_required=False,
        # The Hugging Face release has no 3DGS/ folder: it ships this dataset as SfM/tanks_and_temples. The
        # primary path stays first so existing roots keep their resolved path (it is part of the prediction cache key).
        legacy_relative_paths=(Path("SfM/tanks_and_temples"),),
    ),
    DatasetSpec(
        task="SfM",
        dataset="imc",
        relative_path=Path("SfM/imc"),
        adapter_cls=DatasetIMC,
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="7scenes",
        relative_path=Path("MVRecon/7scenes"),
        adapter_cls=DatasetSevenScenes,
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="eth3d",
        relative_path=Path("MVRecon/eth3d"),
        adapter_cls=DatasetEth3DMVRecon,
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="tanks_and_temples",
        relative_path=Path("MVRecon/tanks_and_temples"),
        adapter_cls=DatasetTanksAndTemplesMVRecon,
        metadata_required=False,
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="dtu",
        relative_path=Path("MVRecon/dtu"),
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="scannetpp",
        relative_path=Path("MVRecon/scannetpp"),
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="hiroom",
        relative_path=Path("MVRecon/hiroom"),
    ),
    DatasetSpec(
        task="SLAM",
        dataset="7scenes",
        relative_path=Path("SLAM/7scenes"),
        adapter_cls=DatasetSevenScenesSLAM,
    ),
    DatasetSpec(
        task="SfM",
        dataset="dtu",
        relative_path=Path("SfM/dtu"),
        adapter_cls=DatasetDTUPose,
        metadata_required=False,
        legacy_relative_paths=(Path("3DGS/dtu"),),
    ),
    DatasetSpec(
        task="SfM",
        dataset="scannetpp",
        relative_path=Path("SfM/scannetpp"),
        legacy_relative_paths=(Path("MVRecon/scannetpp"),),
    ),
    DatasetSpec(
        task="SfM",
        dataset="hiroom",
        relative_path=Path("SfM/hiroom"),
        legacy_relative_paths=(Path("MVRecon/hiroom"),),
    ),
    DatasetSpec(
        task="SfM",
        dataset="texturepoorsfm",
        relative_path=Path("SfM/texturepoorsfm"),
        legacy_relative_paths=(Path("WeakTextureSfM/texturepoorsfm"),),
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="nrgbd",
        relative_path=Path("MVRecon/nrgbd"),
        adapter_cls=DatasetNRGBD,
    ),
    DatasetSpec(
        task="SLAM",
        dataset="tum_dynamics",
        relative_path=Path("SLAM/tum_dynamics"),
        adapter_cls=DatasetTUMDynamicsSLAM,
    ),
    DatasetSpec(
        task="SLAM",
        dataset="oxford_spires",
        relative_path=Path("SLAM/oxford_spires"),
        adapter_cls=DatasetOxfordSpiresSLAM,
    ),
    DatasetSpec(
        task="SLAM",
        dataset="vbr",
        relative_path=Path("SLAM/vbr"),
        adapter_cls=DatasetVBRSLAM,
    ),
    DatasetSpec(
        task="SLAM",
        dataset="kitti_odometry",
        relative_path=Path("SLAM/kitti_odometry"),
        adapter_cls=DatasetKITTIOdometrySLAM,
    ),
) + SPATIALARENA_SPECS


SCANNET_DATASET_SPECS = (
    DatasetSpec(
        task="MVRecon",
        dataset="scannet",
        relative_path=Path("MVRecon/scannet"),
    ),
)


def _normalize_filter(values: Iterable[str] | None) -> set[str]:
    return {canonical_task_name(value).lower() for value in values or []}


def canonical_task_name(task: str) -> str:
    return TASK_ALIASES.get(str(task or ""), str(task or ""))


def select_dataset_specs(
    tasks: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    include_scannet: bool = False,
) -> list[DatasetSpec]:
    task_filter = _normalize_filter(tasks)
    dataset_filter = _normalize_filter(datasets)
    specs = list(DEFAULT_DATASET_SPECS)
    if include_scannet:
        specs.extend(SCANNET_DATASET_SPECS)

    selected = []
    for spec in specs:
        if task_filter and canonical_task_name(spec.task).lower() not in task_filter:
            continue
        if dataset_filter and spec.dataset.lower() not in dataset_filter:
            continue
        selected.append(spec)
    return selected


def find_dataset_spec(
    task: str,
    dataset: str,
    *,
    include_scannet: bool = False,
) -> DatasetSpec | None:
    canonical_task = canonical_task_name(task).lower()
    canonical_dataset = str(dataset or "").lower()
    for spec in select_dataset_specs(include_scannet=include_scannet):
        if canonical_task_name(spec.task).lower() != canonical_task:
            continue
        if spec.dataset.lower() != canonical_dataset:
            continue
        return spec
    return None


def resolve_dataset_root(
    benchmark_root: Path,
    task: str,
    dataset: str,
    *,
    include_scannet: bool = False,
) -> Path:
    benchmark_root = Path(benchmark_root).expanduser()
    spec = find_dataset_spec(task, dataset, include_scannet=include_scannet)
    if spec is not None:
        return spec.data_path(benchmark_root)
    return benchmark_root / canonical_task_name(task) / dataset


def dataset_max_input_frames(task: str, dataset: str) -> int | None:
    """The dataset's own frame cap per case (``DatasetSpec.max_input_frames``), or None for the suite's."""
    spec = find_dataset_spec(task, dataset)
    return None if spec is None else spec.max_input_frames


def scene_image_count(scene_meta: dict | None) -> int:
    """A scene's frame count from its metadata.json entry: ``total_images``, or ``num_images`` as the SLAM inputs'
    metadata (scripts/preprocess/materialize_slam_inputs.py) records it."""
    if not scene_meta:
        return 0
    return int(scene_meta.get("total_images", scene_meta.get("num_images", 0)))


def _has_gt_geometry(scene) -> bool:
    # Deferred: mvrecon_evaluator imports this module.
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.eval.mvrecon_evaluator import resolve_gt_geometry_path

    return resolve_gt_geometry_path(scene) is not None


def describe_benchmark_scenes(
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    tasks: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    scenes: Iterable[str] | None = None,
    include_scannet: bool = False,
) -> list[dict]:
    benchmark_root = Path(benchmark_root).expanduser()
    scene_filter = list(scenes or [])
    records = []

    for spec in select_dataset_specs(
        tasks=tasks,
        datasets=datasets,
        include_scannet=include_scannet,
    ):
        data_path = spec.data_path(benchmark_root)
        metadata_path = data_path / "metadata.json"
        canonical_task = canonical_task_name(spec.task)
        record = {
            "task": canonical_task,
            "task_source": spec.task,
            "dataset": spec.dataset,
            "root_directory": str(data_path),
            "exists": data_path.exists(),
            "metadata_path": str(metadata_path),
            "metadata_required": spec.metadata_required,
            "metadata_exists": metadata_path.exists(),
            "scene_count": 0,
            "total_images": 0,
            "scenes": [],
        }

        if spec.metadata_required and not metadata_path.exists():
            records.append(record)
            continue

        dataset = spec.create_dataset(benchmark_root, scenes=scene_filter)
        scene_infos = dataset.list_scenes()
        record["scene_count"] = len(scene_infos)
        record["total_images"] = sum(scene_image_count(scene.metadata) for scene in scene_infos)
        record["scenes"] = [
            {
                "scene": scene.scene,
                "root_directory": str(scene.root_path),
                "total_images": scene_image_count(scene.metadata),
                "has_images": scene.image_path is not None,
                "has_depths": scene.depth_gt_path is not None,
                "has_sparse": scene.sparse_gt_path is not None,
                # Only MVRecon scores reconstruction geometry; skip the probe (filesystem checks) elsewhere.
                "has_gt_geometry": canonical_task == "MVRecon" and _has_gt_geometry(scene),
                # A SLAM sequence that names its range track (SpatialArena's) runs only in it (slam_protocol).
                **({"slam_range": scene.metadata["range"]}
                   if canonical_task == "SLAM" and (scene.metadata or {}).get("range") else {}),
            }
            for scene in scene_infos
        ]
        records.append(record)

    return records
