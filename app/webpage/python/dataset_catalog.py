# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Reader-facing catalog of the benchmark's evaluation datasets.

The dashboard's dataset gallery (build_dataset_gallery.py) shows one card per
dataset. Counts, scene types and license status mirror docs/evaluation.md;
update both together.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class DatasetCard:
    """One dataset's card in the gallery.

    Attributes:
        key: The dataset key used by the benchmark, e.g. "marigold".
        name: The display name.
        intro: A one- or two-sentence description.
        citation: The short citation, e.g. "Ke et al., CVPR 2024".
        url: The dataset's project page.
        scenes: The number of scenes.
        total_images: The number of images in the source dataset.
        evaluated_images: The number of images the benchmark evaluates.
        scene_type: The kind of scenes, e.g. "indoor, real".
        license_status: The dataset's license, as far as it is known.
        thumbnail_scene: A visually representative scene for the gallery
            thumbnail; the first registered scene is used when it is not
            present.
        thumbnail_image_hint: A hint for which image of the scene to use.
    """

    key: str
    name: str
    intro: str
    citation: str
    url: str
    scenes: int
    total_images: int
    evaluated_images: int
    scene_type: str
    license_status: str
    thumbnail_scene: str = ""
    thumbnail_image_hint: str = ""

    def as_dict(self) -> dict:
        """Returns the card's fields as a dict."""
        return dataclasses.asdict(self)


DATASET_CATALOG: tuple[DatasetCard, ...] = (
    DatasetCard(
        key="marigold",
        name="Marigold Eval Suite",
        intro=("Five classic monocular depth test splits (NYUv2, KITTI, "
               "ETH3D, DIODE, ScanNet) repackaged by the Marigold authors; "
               "each split is evaluated as one scene."),
        citation="Ke et al., CVPR 2024",
        url="https://marigoldmonodepth.github.io/",
        scenes=5,
        total_images=3331,
        evaluated_images=500,
        scene_type="mixed indoor/outdoor, real",
        license_status="mixed per split",
        thumbnail_scene="nyuv2",
        thumbnail_image_hint="rgb",
    ),
    DatasetCard(
        key="aerialmetric",
        name="AerialMetric",
        intro=("UAV views with metric depth: four sites flown at controlled "
               "pitch, altitude and FOV (Decoupled), oblique photogrammetry "
               "captures, and frames from online drone videos (Wild)."),
        citation="Song et al., ECCV 2026",
        url="https://github.com/kuieless/AerialMetric",
        scenes=3,
        total_images=300,
        evaluated_images=300,
        scene_type="outdoor, aerial (UAV), real",
        license_status="academic only / unconfirmed (Wild)",
        thumbnail_scene="decoupled",
    ),
    DatasetCard(
        key="hammer",
        name="HAMMER",
        intro=("Tabletop scenes with and without household objects, "
               "captured by a robot arm, with dense GT depth rendered from "
               "high-accuracy scanned meshes; sharp object boundaries."),
        citation="Jung et al., CVPR 2023",
        url="https://campar.in.tum.de/public_datasets/2022_arxiv_jung/",
        scenes=1,
        total_images=775,
        evaluated_images=100,
        scene_type="indoor tabletop, real",
        license_status="MIT",
        thumbnail_scene="hammer",
    ),
    DatasetCard(
        key="ibims1",
        name="iBims-1",
        intro=("100 indoor images with high-quality depth from a "
               "terrestrial laser scanner, a standard benchmark for "
               "planarity and depth-boundary accuracy."),
        citation="Koch et al., ECCV-W 2018",
        url="https://www.asg.ed.tum.de/lmf/ibims1/",
        scenes=1,
        total_images=100,
        evaluated_images=100,
        scene_type="indoor, real",
        license_status="CC BY 4.0",
        thumbnail_scene="ibims1",
    ),
    DatasetCard(
        key="spring",
        name="Spring",
        intro=("Frames rendered from the Blender open movie Spring: dense "
               "synthetic depth with very fine structure (fur, foliage, "
               "branches). Excluded from the aggregate ranking (train-split "
               "overlap)."),
        citation="Mehl et al., CVPR 2023",
        url="https://spring-benchmark.org/",
        scenes=1,
        total_images=1000,
        evaluated_images=100,
        scene_type="synthetic film, dynamic",
        license_status="CC BY 4.0",
        thumbnail_scene="spring",
    ),
    DatasetCard(
        key="synth4k",
        name="Synth4K",
        intro=("3840x2160 RGB-D frames captured from five video games "
               "(Cyberpunk 2077, Dead Island 2, two Spider-Man titles, "
               "Watch Dogs: Legion), testing fine detail at 4K."),
        citation="Yu et al., 2026 (InfiniDepth)",
        url="https://github.com/zju3dv/InfiniDepth",
        scenes=5,
        total_images=500,
        evaluated_images=500,
        scene_type="game-captured, synthetic",
        license_status="unconfirmed",
        thumbnail_scene="spiderman2",
    ),
    DatasetCard(
        key="sintel",
        name="MPI Sintel",
        intro=("Short clips from the open-source animated film Sintel with "
               "ground-truth depth: fast camera motion, moving characters "
               "and cloth/water simulation, fully rendered."),
        citation="Butler et al., ECCV 2012",
        url="http://sintel.is.tue.mpg.de/",
        scenes=23,
        total_images=1064,
        evaluated_images=1064,
        scene_type="indoor+outdoor, dynamic, synthetic",
        license_status="unconfirmed",
        thumbnail_scene="alley_1",
    ),
    DatasetCard(
        key="bonn",
        name="Bonn RGB-D Dynamic",
        intro=("Kinect RGB-D sequences of people moving boxes and balloons "
               "in front of a static background, a standard stress test for "
               "video depth under dynamic content."),
        citation="Palazzolo et al., IROS 2019",
        url="https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/",
        scenes=26,
        total_images=2860,
        evaluated_images=2600,
        scene_type="indoor, dynamic, real",
        license_status="CC BY-NC-SA 3.0",
        thumbnail_scene="rgbd_bonn_balloon",
    ),
    DatasetCard(
        key="kitti",
        name="KITTI",
        intro=("Raw KITTI driving sequences with LiDAR-derived depth, "
               "covering urban and residential streets with other traffic "
               "and pedestrians."),
        citation="Geiger et al., CVPR 2012",
        url="https://www.cvlibs.net/datasets/kitti/",
        scenes=13,
        total_images=1269,
        evaluated_images=1189,
        scene_type="outdoor, dynamic, real",
        license_status="CC BY-NC-SA 3.0",
    ),
    DatasetCard(
        key="dycheck",
        name="DyCheck",
        intro=("Handheld iPhone captures of people, pets and objects in "
               "motion (camera 0 of the MegaSaM-distributed DyCheck "
               "evaluation archive), testing video depth on casual dynamic "
               "scenes."),
        citation="Gao et al., NeurIPS 2022",
        url="https://github.com/KAIR-BAIR/dycheck",
        scenes=12,
        total_images=3952,
        evaluated_images=1200,
        scene_type="indoor/outdoor, dynamic, real",
        license_status="Apache 2.0",
    ),
    DatasetCard(
        key="spatialarena",
        name="SpatialArena",
        intro=("Our own captures with a handheld 3-camera LiDAR rig: 30 "
               "indoor and outdoor scenes (stations, museums, gardens, "
               "streets) with refined metric poses, LiDAR depth and GGGS "
               "dense depth, feeding all five tasks."),
        citation="SpatialArena, 2026",
        url="https://huggingface.co/datasets/Bluefish2333/SpatialArena-Dataset",
        scenes=30,
        total_images=34994,
        evaluated_images=10072,
        scene_type="indoor/outdoor, static and crowded, real",
        license_status="CC BY-NC 4.0",
    ),
    DatasetCard(
        key="eth3d",
        name="ETH3D",
        intro=("High-resolution DSLR captures of indoor rooms and outdoor "
               "courtyards with laser-scanned geometry, used for both "
               "camera pose and multi-view reconstruction."),
        citation="Schöps et al., CVPR 2017",
        url="https://www.eth3d.net/",
        scenes=13,
        total_images=908,
        evaluated_images=856,
        scene_type="indoor+outdoor, static, real",
        license_status="CC BY-NC-SA 4.0",
        thumbnail_scene="playground",
    ),
    DatasetCard(
        key="tanks_and_temples",
        name="Tanks and Temples",
        intro=("Large-scale handheld video captures of statues, vehicles, "
               "buildings and interiors, a long-standing reference for "
               "reconstruction at scene scale."),
        citation="Knapitsch et al., TOG 2017",
        url="https://www.tanksandtemples.org/",
        scenes=6,
        total_images=2784,
        evaluated_images=600,
        scene_type="object/indoor/outdoor, static, real",
        license_status="CC BY 4.0, non-commercial use",
        thumbnail_scene="Truck",
    ),
    DatasetCard(
        key="imc",
        name="Image Matching Challenge",
        intro=("Phototourism collections of world landmarks from community "
               "photos: unstructured viewpoints, cameras and lighting make "
               "pose estimation hard."),
        citation="Jin et al., IJCV 2021",
        url="https://www.cs.ubc.ca/research/image-matching-challenge/",
        scenes=8,
        total_images=3145,
        evaluated_images=800,
        scene_type="outdoor landmarks, static, real",
        license_status="mixed per-image licenses",
        thumbnail_scene="sagrada_familia",
    ),
    DatasetCard(
        key="7scenes",
        name="7-Scenes",
        intro=("Handheld Kinect sequences through seven small rooms such as "
               "an office, kitchen and stairwell; used by the "
               "reconstruction task, and by the SLAM task as its 18 "
               "official test sequences."),
        citation="Shotton et al., CVPR 2013",
        url=
        "https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/",
        scenes=7,
        total_images=43000,
        evaluated_images=700,
        scene_type="indoor, static, real",
        license_status="MSR-LA, non-commercial",
        thumbnail_scene="redkitchen",
    ),
    DatasetCard(
        key="dtu",
        name="DTU MVS",
        intro=("Tabletop objects captured by a robot-arm camera rig under "
               "studio lighting, with structured-light reference scans for "
               "precise geometry evaluation."),
        citation="Aanæs et al., IJCV 2016",
        url="https://roboimagedata.compute.dtu.dk/?page_id=36",
        scenes=15,
        total_images=855,
        evaluated_images=855,
        scene_type="object (tabletop), static, real",
        license_status="unconfirmed (research use)",
    ),
    DatasetCard(
        key="scannetpp",
        name="ScanNet++",
        intro=("High-fidelity laser and DSLR scans of real rooms, "
               "apartments and offices. Excluded from the aggregate "
               "leaderboard because many methods train on it."),
        citation="Yeshwanth et al., ICCV 2023",
        url="https://kaldir.vc.in.tum.de/scannetpp/",
        scenes=20,
        total_images=178345,
        evaluated_images=2000,
        scene_type="indoor, static, real",
        license_status="ScanNet++ ToU, non-commercial",
    ),
    DatasetCard(
        key="hiroom",
        name="HiRoom",
        intro=("Photorealistic Blender-rendered living rooms and kitchens "
               "from the Depth Anything 3 benchmark; never used as training "
               "data, so it carries no leakage risk."),
        citation="Depth Anything 3 (DA3-BENCH), 2025",
        url="https://huggingface.co/datasets/depth-anything/DA3-BENCH",
        scenes=30,
        total_images=400,
        evaluated_images=400,
        scene_type="indoor, static, synthetic",
        license_status="research/evaluation only",
    ),
    DatasetCard(
        key="texturepoorsfm",
        name="TexturePoorSfM",
        intro=("Low-texture household objects on a tabletop, where keypoint "
               "matching breaks down; the benchmark uses 30 bags from five "
               "held-out objects (split v2)."),
        citation="He et al., CVPR 2024",
        url="https://zju3dv.github.io/DetectorFreeSfM/",
        scenes=30,
        total_images=350,
        evaluated_images=350,
        scene_type="object (tabletop), static, real",
        license_status="unconfirmed",
    ),
    DatasetCard(
        key="nrgbd",
        name="Neural RGB-D",
        intro=("Rendered indoor scenes (dining rooms, kitchens, a "
               "staircase) with clean ground-truth meshes for measuring "
               "reconstruction accuracy and completeness."),
        citation="Azinović et al., CVPR 2022",
        url="https://github.com/dazinovic/neural-rgbd-surface-reconstruction",
        scenes=9,
        total_images=10970,
        evaluated_images=900,
        scene_type="indoor, static, synthetic",
        license_status="CC-BY / CC-0 per scene",
        thumbnail_scene="breakfast_room",
    ),
    DatasetCard(
        key="tum_dynamics",
        name="TUM RGB-D Dynamic",
        intro=("Office sequences from the TUM RGB-D benchmark in which "
               "people sit, move and walk through the view, testing camera "
               "tracking in dynamic scenes."),
        citation="Sturm et al., IROS 2012",
        url="https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
        scenes=9,
        total_images=10806,
        evaluated_images=900,
        scene_type="indoor, dynamic (people), real",
        license_status="CC BY 4.0",
        thumbnail_scene="rgbd_dataset_freiburg3_walking_xyz",
    ),
    DatasetCard(
        key="oxford_spires",
        name="Oxford Spires",
        intro=("Handheld walks around Oxford colleges and libraries with "
               "LiDAR-derived ground-truth poses: long outdoor trajectories "
               "with loops, used whole for large-scale camera tracking "
               "(large SLAM range)."),
        citation="Tao et al., IJRR 2026",
        url="https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/",
        scenes=10,
        total_images=38380,
        evaluated_images=38380,
        scene_type="outdoor, campus-scale, real",
        license_status="CC BY-NC-SA 4.0",
        thumbnail_scene="christ-church-02",
    ),
    DatasetCard(
        key="vbr",
        name="VBR (Vision Benchmark in Rome)",
        intro=("Kilometre-scale handheld and car-mounted sequences across "
               "Rome (campus, Colosseum, Spanish Steps, Pincio) with "
               "LiDAR-inertial ground truth; the longest inputs of the "
               "large SLAM range."),
        citation="Brizi et al., ICRA 2024",
        url="https://rvp-group.net/slam-dataset.html",
        scenes=7,
        total_images=86678,
        evaluated_images=86678,
        scene_type="outdoor, city-scale, real",
        license_status="unconfirmed (no data license stated)",
        thumbnail_scene="colosseo_train0",
    ),
    DatasetCard(
        key="kitti_odometry",
        name="KITTI Odometry",
        intro=("Car-mounted drives through Karlsruhe streets, residential "
               "areas and highways with GPS/INS ground truth: the classic "
               "visual-odometry benchmark, sequences 00-10 used whole at 10 "
               "Hz (large SLAM range)."),
        citation="Geiger et al., CVPR 2012",
        url="https://www.cvlibs.net/datasets/kitti/eval_odometry.php",
        scenes=11,
        total_images=23201,
        evaluated_images=23201,
        scene_type="outdoor, driving, real",
        license_status="CC BY-NC-SA 3.0",
        thumbnail_scene="00",
    ),
)

DATASET_CATALOG_BY_KEY = {card.key: card for card in DATASET_CATALOG}
