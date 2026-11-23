# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Dataset registry for the benchmark task layout.

Each `DatasetSpec` names a benchmark dataset of a task, where it lives under the
benchmark root and which adapter lists its scenes. `select_dataset_specs`,
`resolve_dataset_root` and `describe_benchmark_scenes` are the entry points.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterable

from spatialarena import paths
from spatialarena import spatialarena
from spatialarena.eval import bonn_dataset
from spatialarena.eval import canonical_dataset
from spatialarena.eval import imc_dataset
from spatialarena.eval import kitti_dataset
from spatialarena.eval import marigold_dataset
from spatialarena.eval import monodepth_datasets
from spatialarena.eval import nrgbd_dataset
from spatialarena.eval import pose_datasets
from spatialarena.eval import seven_scenes_dataset
from spatialarena.eval import sintel_dataset

DEFAULT_BENCHMARK_ROOT = pathlib.Path(f"{paths.SPATIALARENA_ROOT}/benchmarks")
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class DatasetSpec:
    """One benchmark dataset of a task.

    Attributes:
        task: Task name as registered (aliases allowed).
        dataset: Dataset name.
        relative_path: Dataset folder relative to the benchmark root.
        adapter_cls: Dataset adapter that lists the scenes.
        metadata_required: The dataset must have a metadata.json.
        legacy_relative_paths: Older locations, used if the primary one is
            missing.
        max_input_frames: Frames a full-depth/full-pose case feeds a method.
    """
    task: str
    dataset: str
    relative_path: pathlib.Path
    adapter_cls: type[
        canonical_dataset.CanonicalDataset] = canonical_dataset.CanonicalDataset
    metadata_required: bool = True
    legacy_relative_paths: tuple[pathlib.Path, ...] = ()
    # Frames a full-depth/full-pose case of this dataset feeds a method: None =
    # the suite's cap (100, DA3-BENCH's max_frames), 0 = every frame.
    # run_method_case_list.sh applies it per case (BENCHMARK_MAX_INPUT_FRAMES
    # and MAX_IMAGES_PER_RUN).
    max_input_frames: int | None = None

    def data_path(self, benchmark_root: pathlib.Path) -> pathlib.Path:
        """Returns the dataset folder, or an existing legacy one instead."""
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
        benchmark_root: pathlib.Path,
        scenes: list[str] | None = None,
    ) -> canonical_dataset.CanonicalDataset:
        """Creates the dataset adapter, restricted to `scenes` if given."""
        data_path = self.data_path(benchmark_root)
        task_name = canonical_task_name(self.task)
        if self.adapter_cls is canonical_dataset.CanonicalDataset:
            return self.adapter_cls(data_path,
                                    scenes=scenes,
                                    task_name=task_name)
        return self.adapter_cls(data_path, scenes=scenes)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SpatialArenaSpec(DatasetSpec):
    """SpatialArena (spatialarena/spatialarena.py) dataset of a task.

    It lives in `$SPATIALARENA_ROOT/<Task>` when that is set, else in the usual
    `<benchmark_root>/<Task>/spatialarena`.
    """

    def data_path(self, benchmark_root: pathlib.Path) -> pathlib.Path:
        """Returns the track root override if set, else the usual folder."""
        override = spatialarena.track_root_override(self.task)
        if override is not None:
            return override
        return super().data_path(benchmark_root)


# SpatialArena's VideoDepth clips (132-1,021 frames) are fed whole, its MVRecon
# scenes (every image, 363-1,806) and SfM scenes (about 1,100 images over the 3
# cameras) on 300 images; every other dataset keeps the suites' 100-frame cap.
SPATIALARENA_MAX_INPUT_FRAMES = {"VideoDepth": 0, "MVRecon": 300, "SfM": 300}
SPATIALARENA_SPECS = tuple(
    SpatialArenaSpec(task=task,
                     dataset=spatialarena.SPATIALARENA,
                     relative_path=pathlib.Path(task) /
                     spatialarena.SPATIALARENA,
                     max_input_frames=SPATIALARENA_MAX_INPUT_FRAMES.get(task))
    for task in spatialarena.TASKS)

DEFAULT_DATASET_SPECS = (
    DatasetSpec(
        task="MonoDepth",
        dataset="marigold",
        relative_path=pathlib.Path("MonoDepth/marigold"),
        adapter_cls=marigold_dataset.DatasetMarigold,
    ),
    DatasetSpec(
        task="MonoDepth",
        dataset="aerialmetric",
        relative_path=pathlib.Path("MonoDepth/aerialmetric"),
        adapter_cls=monodepth_datasets.DatasetAerialMetric,
    ),
    DatasetSpec(
        task="MonoDepth",
        dataset="hammer",
        relative_path=pathlib.Path("MonoDepth/hammer"),
        adapter_cls=monodepth_datasets.DatasetHammer,
    ),
    DatasetSpec(
        task="MonoDepth",
        dataset="ibims1",
        relative_path=pathlib.Path("MonoDepth/ibims1"),
        adapter_cls=monodepth_datasets.DatasetIBims1,
    ),
    DatasetSpec(
        task="MonoDepth",
        dataset="spring",
        relative_path=pathlib.Path("MonoDepth/spring"),
        adapter_cls=monodepth_datasets.DatasetSpring,
    ),
    DatasetSpec(
        task="MonoDepth",
        dataset="synth4k",
        relative_path=pathlib.Path("MonoDepth/synth4k"),
        adapter_cls=monodepth_datasets.DatasetSynth4K,
    ),
    DatasetSpec(
        task="VideoDepth",
        dataset="sintel",
        relative_path=pathlib.Path("VideoDepth/sintel"),
        adapter_cls=sintel_dataset.DatasetSintel,
    ),
    DatasetSpec(
        task="VideoDepth",
        dataset="bonn",
        relative_path=pathlib.Path("VideoDepth/bonn"),
        adapter_cls=bonn_dataset.DatasetBonn,
    ),
    DatasetSpec(task="VideoDepth",
                dataset="dycheck",
                relative_path=pathlib.Path("VideoDepth/dycheck")),
    DatasetSpec(
        task="VideoDepth",
        dataset="kitti",
        relative_path=pathlib.Path("VideoDepth/kitti"),
        adapter_cls=kitti_dataset.DatasetKitti,
    ),
    DatasetSpec(
        task="SfM",
        dataset="eth3d",
        relative_path=pathlib.Path("SfM/eth3d"),
        adapter_cls=pose_datasets.DatasetEth3DPose,
    ),
    DatasetSpec(
        task="SfM",
        dataset="tanks_and_temples",
        relative_path=pathlib.Path("3DGS/tanks_and_temples"),
        adapter_cls=pose_datasets.DatasetTanksAndTemplesPose,
        metadata_required=False,
        # The Hugging Face release has no 3DGS/ folder: it ships this dataset
        # as SfM/tanks_and_temples. The primary path stays first so existing
        # roots keep their resolved path (it is part of the prediction cache
        # key).
        legacy_relative_paths=(pathlib.Path("SfM/tanks_and_temples"),),
    ),
    DatasetSpec(
        task="SfM",
        dataset="imc",
        relative_path=pathlib.Path("SfM/imc"),
        adapter_cls=imc_dataset.DatasetIMC,
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="7scenes",
        relative_path=pathlib.Path("MVRecon/7scenes"),
        adapter_cls=seven_scenes_dataset.DatasetSevenScenes,
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="eth3d",
        relative_path=pathlib.Path("MVRecon/eth3d"),
        adapter_cls=pose_datasets.DatasetEth3DMVRecon,
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="tanks_and_temples",
        relative_path=pathlib.Path("MVRecon/tanks_and_temples"),
        adapter_cls=pose_datasets.DatasetTanksAndTemplesMVRecon,
        metadata_required=False,
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="dtu",
        relative_path=pathlib.Path("MVRecon/dtu"),
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="scannetpp",
        relative_path=pathlib.Path("MVRecon/scannetpp"),
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="hiroom",
        relative_path=pathlib.Path("MVRecon/hiroom"),
    ),
    DatasetSpec(
        task="SLAM",
        dataset="7scenes",
        relative_path=pathlib.Path("SLAM/7scenes"),
        adapter_cls=pose_datasets.DatasetSevenScenesSLAM,
    ),
    DatasetSpec(
        task="SfM",
        dataset="dtu",
        relative_path=pathlib.Path("SfM/dtu"),
        adapter_cls=pose_datasets.DatasetDTUPose,
        metadata_required=False,
        legacy_relative_paths=(pathlib.Path("3DGS/dtu"),),
    ),
    DatasetSpec(
        task="SfM",
        dataset="scannetpp",
        relative_path=pathlib.Path("SfM/scannetpp"),
        legacy_relative_paths=(pathlib.Path("MVRecon/scannetpp"),),
    ),
    DatasetSpec(
        task="SfM",
        dataset="hiroom",
        relative_path=pathlib.Path("SfM/hiroom"),
        legacy_relative_paths=(pathlib.Path("MVRecon/hiroom"),),
    ),
    DatasetSpec(
        task="SfM",
        dataset="texturepoorsfm",
        relative_path=pathlib.Path("SfM/texturepoorsfm"),
        legacy_relative_paths=(pathlib.Path("WeakTextureSfM/texturepoorsfm"),),
    ),
    DatasetSpec(
        task="MVRecon",
        dataset="nrgbd",
        relative_path=pathlib.Path("MVRecon/nrgbd"),
        adapter_cls=nrgbd_dataset.DatasetNRGBD,
    ),
    DatasetSpec(
        task="SLAM",
        dataset="tum_dynamics",
        relative_path=pathlib.Path("SLAM/tum_dynamics"),
        adapter_cls=pose_datasets.DatasetTUMDynamicsSLAM,
    ),
    DatasetSpec(
        task="SLAM",
        dataset="oxford_spires",
        relative_path=pathlib.Path("SLAM/oxford_spires"),
        adapter_cls=pose_datasets.DatasetOxfordSpiresSLAM,
    ),
    DatasetSpec(
        task="SLAM",
        dataset="vbr",
        relative_path=pathlib.Path("SLAM/vbr"),
        adapter_cls=pose_datasets.DatasetVBRSLAM,
    ),
    DatasetSpec(
        task="SLAM",
        dataset="kitti_odometry",
        relative_path=pathlib.Path("SLAM/kitti_odometry"),
        adapter_cls=pose_datasets.DatasetKITTIOdometrySLAM,
    ),
) + SPATIALARENA_SPECS

SCANNET_DATASET_SPECS = (DatasetSpec(
    task="MVRecon",
    dataset="scannet",
    relative_path=pathlib.Path("MVRecon/scannet"),
),)


def _normalize_filter(values: Iterable[str] | None) -> set[str]:
    return {canonical_task_name(value).lower() for value in values or []}


def canonical_task_name(task: str) -> str:
    """Maps a task alias (e.g. CameraPose) to its canonical task name."""
    return TASK_ALIASES.get(str(task or ""), str(task or ""))


def select_dataset_specs(
    tasks: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    include_scannet: bool = False,
) -> list[DatasetSpec]:
    """Selects the dataset specs matching the task and dataset filters.

    Args:
        tasks: Task names (aliases allowed); None or empty selects all.
        datasets: Dataset names; None or empty selects all.
        include_scannet: Also consider the ScanNet specs.

    Returns:
        The matching specs, in registry order.
    """
    task_filter = _normalize_filter(tasks)
    dataset_filter = _normalize_filter(datasets)
    specs = list(DEFAULT_DATASET_SPECS)
    if include_scannet:
        specs.extend(SCANNET_DATASET_SPECS)

    selected = []
    for spec in specs:
        if task_filter and canonical_task_name(
                spec.task).lower() not in task_filter:
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
    """Returns the spec of `dataset` for `task`, or None if not registered."""
    canonical_task = canonical_task_name(task).lower()
    wanted_dataset = str(dataset or "").lower()
    for spec in select_dataset_specs(include_scannet=include_scannet):
        if canonical_task_name(spec.task).lower() != canonical_task:
            continue
        if spec.dataset.lower() != wanted_dataset:
            continue
        return spec
    return None


def resolve_dataset_root(
    benchmark_root: pathlib.Path,
    task: str,
    dataset: str,
    *,
    include_scannet: bool = False,
) -> pathlib.Path:
    """Returns the dataset folder, or `<root>/<task>/<dataset>` if unknown."""
    benchmark_root = pathlib.Path(benchmark_root).expanduser()
    spec = find_dataset_spec(task, dataset, include_scannet=include_scannet)
    if spec is not None:
        return spec.data_path(benchmark_root)
    return benchmark_root / canonical_task_name(task) / dataset


def dataset_max_input_frames(task: str, dataset: str) -> int | None:
    """Returns the dataset's own frame cap per case, or None for the suite's.

    The cap is `DatasetSpec.max_input_frames`.
    """
    spec = find_dataset_spec(task, dataset)
    if spec is None:
        return None
    return spec.max_input_frames


def scene_image_count(scene_meta: dict | None) -> int:
    """Returns a scene's frame count from its metadata.json entry.

    That is `total_images`, or `num_images` as the SLAM inputs' metadata
    (scripts/preprocess/materialize_slam_inputs.py) records it.
    """
    if not scene_meta:
        return 0
    return int(scene_meta.get("total_images", scene_meta.get("num_images", 0)))


def _has_gt_geometry(scene) -> bool:
    # Deferred: mvrecon_evaluator imports this module.
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.eval import mvrecon_evaluator

    return mvrecon_evaluator.resolve_gt_geometry_path(scene) is not None


def _scene_record(scene, canonical_task: str) -> dict:
    """Returns the describe_benchmark_scenes record of one scene."""
    record = {
        "scene":
            scene.scene,
        "root_directory":
            str(scene.root_path),
        "total_images":
            scene_image_count(scene.metadata),
        "has_images":
            scene.image_path is not None,
        "has_depths":
            scene.depth_gt_path is not None,
        "has_sparse":
            scene.sparse_gt_path is not None,
        # Only MVRecon scores reconstruction geometry; skip the probe
        # (filesystem checks) elsewhere.
        "has_gt_geometry": (canonical_task == "MVRecon" and
                            _has_gt_geometry(scene)),
    }
    # A SLAM sequence that names its range track (SpatialArena's) runs only in
    # it (slam_protocol).
    if canonical_task == "SLAM" and (scene.metadata or {}).get("range"):
        record["slam_range"] = scene.metadata["range"]
    return record


def describe_benchmark_scenes(
    benchmark_root: pathlib.Path = DEFAULT_BENCHMARK_ROOT,
    tasks: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    scenes: Iterable[str] | None = None,
    include_scannet: bool = False,
) -> list[dict]:
    """Describes the benchmark datasets and their scenes as found on disk.

    Args:
        benchmark_root: Benchmark root folder.
        tasks: Task filter; None or empty selects all.
        datasets: Dataset filter; None or empty selects all.
        scenes: Scene filter passed to the adapters.
        include_scannet: Also describe the ScanNet datasets.

    Returns:
        One record per dataset spec, with its per-scene records.
    """
    benchmark_root = pathlib.Path(benchmark_root).expanduser()
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
        record["total_images"] = sum(
            scene_image_count(scene.metadata) for scene in scene_infos)
        record["scenes"] = [
            _scene_record(scene, canonical_task) for scene in scene_infos
        ]
        records.append(record)

    return records
