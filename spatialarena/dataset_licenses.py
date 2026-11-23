# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Original-source license/attribution registry for the benchmark datasets.

Every entry here was verified against that source's own official page (or,
where no official page states a formal license, that absence is recorded
honestly as `status="unconfirmed"` rather than guessed at) as of 2026-09.
`scripts/release/pack_and_upload_benchmark_datasets.py` uses this registry
to write a LICENSE_NOTICE.md into every packaged dataset folder and to
decide which datasets are safe to include in a public re-release by
default.

`status` values:
  - "permissive"             -- a named open license (MIT, CC BY, ...).
  - "restricted_noncommercial" -- a named license/agreement that limits use
    to non-commercial research (CC BY-NC-*, a signed DUA, a per-download
    "MSR-LA" style agreement, ...). Redistributing this data publicly still
    requires the redistributor to hold the rights to do so -- this registry
    does not grant that, it only records what the *original* source says.
  - "unconfirmed"            -- no formal, citable license/terms-of-use
    document was found on the source's own site during this research pass.
    Treated as research-only until someone confirms otherwise directly
    with the source.
  - "no_redistribution"      -- the source's terms forbid redistributing the
    data. Never packaged, not even with --include_unconfirmed; a dataset that
    bundles such a source ships without the scenes listed in WITHHELD_SCENES.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SourceLicense:
    """License and attribution of one original data source.

    Attributes:
        name: Display name of the source.
        homepage: The source's official page.
        status: "permissive", "restricted_noncommercial", "unconfirmed" or
            "no_redistribution" (see the module docstring).
        license_name: The license or terms, as the source names them.
        license_url: Where the license or terms are published.
        citation: How the source asks to be cited.
        notes: Anything else a user of the data must know.
    """

    name: str
    homepage: str
    # "permissive" | "restricted_noncommercial" | "unconfirmed" |
    # "no_redistribution"
    status: str
    license_name: str
    license_url: str
    citation: str
    notes: str = ""


SOURCE_LICENSES: dict[str, SourceLicense] = {
    "nyuv2":
        SourceLicense(
            name="NYU Depth V2",
            homepage="https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html",
            status="unconfirmed",
            license_name=("No formal license document published by the source "
                          "(attribution/citation requested)"),
            license_url=
            "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html",
            citation=(
                'Silberman, N., Hoiem, D., Kohli, P., Fergus, R. "Indoor '
                'Segmentation and Support Inference from RGBD Images." ECCV '
                "2012."),
        ),
    "scannet":
        SourceLicense(
            name="ScanNet",
            homepage="http://www.scan-net.org/",
            status="no_redistribution",
            license_name=(
                "ScanNet Terms of Use (non-commercial research and educational "
                "use; no redistribution)"),
            license_url="https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf",
            citation=(
                "Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., "
                'Niessner, M. "ScanNet: Richly-annotated 3D Reconstructions of '
                'Indoor Scenes." CVPR 2017.'),
            notes=("Marigold's 800-frame ScanNet evaluation subset "
                   "(scannet_val_sampled_800_1)."),
        ),
    "kitti_depth":
        SourceLicense(
            name="KITTI (depth selection / raw data)",
            homepage="https://www.cvlibs.net/datasets/kitti/",
            status="restricted_noncommercial",
            license_name="CC BY-NC-SA 3.0",
            license_url="https://creativecommons.org/licenses/by-nc-sa/3.0/",
            citation=(
                'Geiger, A., Lenz, P., Urtasun, R. "Are we ready for '
                'Autonomous Driving? The KITTI Vision Benchmark Suite." CVPR '
                "2012."),
        ),
    "eth3d":
        SourceLicense(
            name="ETH3D",
            homepage="https://www.eth3d.net/",
            status="restricted_noncommercial",
            license_name="CC BY-NC-SA 4.0",
            license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
            citation=(
                "Schöps, T., Schönberger, J.L., Galliani, S., Sattler, T., "
                'Schindler, K., Pollefeys, M., Geiger, A. "A Multi-View Stereo '
                "Benchmark with High-Resolution Images and Multi-Camera "
                'Videos." CVPR 2017.'),
        ),
    "diode":
        SourceLicense(
            name="DIODE",
            homepage="https://diode-dataset.org/",
            status="permissive",
            license_name="MIT License",
            license_url=
            "https://github.com/diode-dataset/diode-devkit/blob/master/LICENSE",
            citation=(
                'Vasiljevic, I. et al. "DIODE: A Dense Indoor and Outdoor '
                'DEpth Dataset." arXiv:1908.00463, 2019.'),
        ),
    "sintel":
        SourceLicense(
            name="MPI-Sintel (depth extension)",
            homepage="http://sintel.is.tue.mpg.de/",
            status="unconfirmed",
            license_name=(
                "No formal license document found on the official site as of "
                "this research pass"),
            license_url="http://sintel.is.tue.mpg.de/",
            citation=(
                'Butler, D.J., Wulff, J., Stanley, G.B., Black, M.J. "A '
                'naturalistic open source movie for optical flow evaluation." '
                "ECCV 2012. (Depth extension: Wulff, J. et al., MPI-Sintel "
                "depth data.)"),
            notes=("Verify directly with the Max Planck Institute "
                   "(sintel@tue.mpg.de) before public redistribution."),
        ),
    "bonn_rgbd_dynamic":
        SourceLicense(
            name="Bonn RGB-D Dynamic Dataset",
            homepage="https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/",
            status="restricted_noncommercial",
            license_name="CC BY-NC-SA 3.0",
            license_url="https://creativecommons.org/licenses/by-nc-sa/3.0/",
            citation=(
                "Palazzolo, E., Behley, J., Lottes, P., Giguère, P., "
                'Stachniss, C. "ReFusion: 3D Reconstruction in Dynamic '
                'Environments for RGB-D Cameras Exploiting Residuals." IROS '
                "2019."),
        ),
    "7scenes":
        SourceLicense(
            name="Microsoft 7-Scenes",
            homepage=
            "https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/",
            status="restricted_noncommercial",
            license_name=(
                "Microsoft Research License Agreement (non-commercial use "
                "only)"),
            license_url=
            ("https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/7-scenes-msr-la-dataset-7-scenes.rtf"
            ),
            citation=(
                "Shotton, J., Glocker, B., Zach, C., Izadi, S., Criminisi, A., "
                'Fitzgibbon, A. "Scene Coordinate Regression Forests for '
                'Camera Relocalization in RGB-D Images." CVPR 2013.'),
        ),
    "dtu_mvs":
        SourceLicense(
            name="DTU MVS / Robot Image Data Sets",
            homepage="https://roboimagedata.compute.dtu.dk/?page_id=36",
            status="unconfirmed",
            license_name=(
                "No formal license document published by the source; site "
                "describes the data as 'freely available' for research, "
                "citation requested"),
            license_url="https://roboimagedata.compute.dtu.dk/?page_id=36",
            citation=(
                "Aanæs, H., Jensen, R.R., Vogiatzis, G., Tola, E., Dahl, A.B. "
                '"Large-Scale Data for Multiple-View Stereopsis." IJCV 2016.'),
        ),
    "tanks_and_temples":
        SourceLicense(
            name="Tanks and Temples",
            homepage="https://www.tanksandtemples.org/",
            status="restricted_noncommercial",
            license_name=(
                "CC BY 4.0 for citation, plus a non-commercial-research-only "
                "install/use restriction stated in the benchmark's own license "
                "page"),
            license_url="https://www.tanksandtemples.org/license/",
            citation=(
                'Knapitsch, A., Park, J., Zhou, Q.Y., Koltun, V. "Tanks and '
                'Temples: Benchmarking Large-Scale Scene Reconstruction." ACM '
                "TOG 2017."),
            notes=(
                "This repo's copy is downloaded via Free View Synthesis's "
                "`ibr3d_tat.tar.gz` repackaging (Riegler & Koltun, ECCV 2020) "
                "-- check that repackaging's own terms in addition to the "
                "original Tanks and Temples license."),
        ),
    "imc_phototourism":
        SourceLicense(
            name="Image Matching Challenge (IMW2020) Phototourism TestData",
            homepage=
            "https://www.cs.ubc.ca/research/image-matching-challenge/2021/data/",
            status="unconfirmed",
            license_name=(
                "Mixed per-image licenses (sourced from Flickr/YFCC100M "
                "community photos) -- check LICENSE.txt in each scene folder"),
            license_url=
            "https://www.cs.ubc.ca/research/image-matching-challenge/2021/data/",
            citation=(
                'Jin, Y. et al. "Image Matching across Wide Baselines: From '
                'Paper to Practice." IJCV 2021.'),
        ),
    "scannetpp":
        SourceLicense(
            name="ScanNet++",
            homepage="https://scannetpp.mlsg.cit.tum.de/scannetpp/",
            status="restricted_noncommercial",
            license_name=(
                "ScanNet++ Terms of Use (non-commercial research/educational "
                "use only; no redistribution without permission)"),
            license_url=
            "https://scannetpp.mlsg.cit.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf",
            citation=(
                'Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A. "ScanNet++: A '
                'High-Fidelity Dataset of 3D Indoor Scenes." ICCV 2023.'),
            notes=(
                "Signed data-use agreement required with TUM to obtain this "
                "data in the first place. Redistribution requires separate, "
                "explicit permission from TUM; the repository maintainer "
                "confirmed on 2026-09-30 that this benchmark subset may be "
                "redistributed. Any use remains bound by the ScanNet++ Terms "
                "of Use (non-commercial)."),
        ),
    "hiroom":
        SourceLicense(
            name="HiRoom (via DA3-BENCH)",
            homepage="https://huggingface.co/datasets/depth-anything/DA3-BENCH",
            status="restricted_noncommercial",
            license_name=(
                'HF license tag "other" -- DA3-BENCH\'s own card states '
                'research/evaluation use only and asks users to "obtain proper '
                'permissions" before other use'),
            license_url=
            "https://huggingface.co/datasets/depth-anything/DA3-BENCH",
            citation=(
                "Lin, H., Chen, S., Liew, J.H., Chen, D.Y., Li, Z., Shi, G., "
                'Feng, J., Kang, B. "Depth Anything 3: Recovering the Visual '
                'Space from Any Views." arXiv preprint, 2025.'),
            notes=(
                "HiRoom is an original synthetic (Blender-rendered) dataset "
                "introduced by the Depth Anything 3 paper itself, not a "
                "third-party dataset DA3-BENCH is re-hosting -- there is no "
                "separate upstream license to point to beyond DA3-BENCH's own "
                "terms."),
        ),
    "nrgbd":
        SourceLicense(
            name="Neural RGB-D Surface Reconstruction dataset",
            homepage=
            "https://github.com/dazinovic/neural-rgbd-surface-reconstruction",
            status="permissive",
            license_name=(
                "Per-scene CC-BY or CC-0 (9 scenes sourced from BlendSwap "
                "artists; see the repo's own per-scene license table)"),
            license_url=
            "https://github.com/dazinovic/neural-rgbd-surface-reconstruction#dataset",
            citation=(
                "Azinović, D., Martin-Brualla, R., Goldman, D.B., Nießner, M., "
                'Thies, J. "Neural RGB-D Surface Reconstruction." CVPR 2022.'),
            notes=(
                "Each of the 9 scenes (breakfast_room, complete_kitchen, "
                "green_room, grey_white_room, kitchen, morning_apartment, "
                "staircase, thin_geometry, whiteroom) is individually CC-BY or "
                "CC-0 per its own BlendSwap source -- check the repo's table "
                "before assuming uniform terms across all 9."),
        ),
    "tum_dynamics":
        SourceLicense(
            name="TUM RGB-D SLAM Dataset (Dynamic Objects category)",
            homepage="https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
            status="permissive",
            license_name=("CC BY 4.0 (data); accompanying source "
                          "code is BSD-2-Clause"),
            license_url="https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
            citation=(
                "Sturm, J., Engelhard, N., Endres, F., Burgard, W., Cremers, "
                'D. "A Benchmark for the Evaluation of RGB-D SLAM Systems." '
                "IROS 2012."),
        ),
    "oxford_spires":
        SourceLicense(
            name="The Oxford Spires Dataset",
            homepage="https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/",
            status="restricted_noncommercial",
            license_name="CC BY-NC-SA 4.0",
            license_url=
            "https://huggingface.co/datasets/ori-drs/oxford_spires_dataset",
            citation=(
                "Tao, Y., Muñoz-Bañón, M.Á., Zhang, L., Wang, J., Fu, L.F.T., "
                'Fallon, M. "The Oxford Spires Dataset: Benchmarking '
                "Large-Scale LiDAR-Visual Localisation, Reconstruction and "
                'Radiance Field Methods." International Journal of Robotics '
                "Research 45(6), 2026."),
            notes=(
                "Rectified cam0 frames and GT poses derived with LingBot-Map's "
                "preprocess/oxford.py (images only)."),
        ),
    "vbr":
        SourceLicense(
            name="VBR: A Vision Benchmark in Rome",
            homepage="https://rvp-group.net/slam-dataset.html",
            status="unconfirmed",
            license_name=(
                "No data license stated (site footer: (c) RVP-Group, All "
                "Rights Reserved); devkit BSD-3-Clause"),
            license_url="https://github.com/rvp-group/vbr-devkit",
            citation=(
                "Brizi, L., Giacomini, E., Di Giammarino, L., Ferrari, S., "
                'Salem, O., De Rebotti, L., Grisetti, G. "VBR: A Vision '
                'Benchmark in Rome." ICRA 2024.'),
            notes=("Used in the processed form of LoGeR (HF "
                   "Junyi42/vbr_processed, no license stated either); evaluate "
                   "only, never redistributed."),
        ),
    "kitti_odometry":
        SourceLicense(
            name="KITTI Odometry",
            homepage="https://www.cvlibs.net/datasets/kitti/eval_odometry.php",
            status="restricted_noncommercial",
            license_name="CC BY-NC-SA 3.0",
            license_url="https://creativecommons.org/licenses/by-nc-sa/3.0/",
            citation=(
                'Geiger, A., Lenz, P., Urtasun, R. "Are we ready for '
                'Autonomous Driving? The KITTI Vision Benchmark Suite." CVPR '
                "2012."),
            notes=(
                "Sequences 00-10 (public ground truth), left color camera; GT "
                "poses moved from cam0 to cam2."),
        ),
    "texturepoorsfm":
        SourceLicense(
            name="TexturePoorSfM",
            homepage="https://github.com/zju3dv/DetectorFreeSfM",
            status="unconfirmed",
            license_name="Dataset license not stated by the authors",
            license_url="",
            citation=(
                "He, X., Sun, J., Wang, Y., Peng, S., Huang, Q., Bao, H., "
                'Zhou, X. "Detector-Free Structure from Motion." CVPR 2024.'),
            notes=(
                "Source identified: scripts/download/texturepoorsfm.sh fetches "
                "Google Drive ID 1UDo8K0uYCi-YtpwLviE6hT-6Mxm5jHYH, the "
                "TexturePoorSfM archive linked from DetectorFreeSfM's "
                "DATASET_PREPARE.md. The authors publish no dataset license "
                "(their code was Apache-2.0 until 2026-09-15, then PRL v1.0; "
                "neither text covers the data). Do not include this dataset in "
                "a public release until the data license is confirmed with the "
                "authors."),
        ),
    "dycheck":
        SourceLicense(
            name=("DyCheck (iPhone dataset, MegaSaM-distributed evaluation "
                  "archive)"),
            homepage="https://github.com/KAIR-BAIR/dycheck",
            status="permissive",
            license_name=(
                "Apache License 2.0 (dataset); MegaSaM's non-software "
                "materials CC BY 4.0"),
            license_url="https://github.com/KAIR-BAIR/dycheck#license",
            citation=(
                "Gao, H., Li, R., Tulsiani, S., Russell, B., Kanazawa, A. "
                '"Monocular Dynamic View Synthesis: A Reality Check." NeurIPS '
                "2022."),
            notes=(
                "The DyCheck README states 'Both our code and dataset are "
                "released under the Apache 2.0 license.' The evaluation "
                "archive comes from MegaSaM's Google Drive folder "
                "(github.com/mega-sam/mega-sam), whose README licenses "
                "software under Apache 2.0 and all other materials under CC BY "
                "4.0; both allow redistribution with attribution. Also cite "
                "MegaSaM (Li et al., CVPR 2025). Camera-0 RGB/depth are linked "
                "by scripts/preprocess/materialize_dycheck.py. Verified "
                "2026-10-24."),
        ),
    "spatialarena":
        SourceLicense(
            name=(
                "SpatialArena (the benchmark's own handheld 3-camera LiDAR rig "
                "captures)"),
            homepage=("https://huggingface.co/datasets/"
                      "Bluefish2333/SpatialArena-Dataset"),
            status="restricted_noncommercial",
            license_name="CC BY-NC 4.0",
            license_url="https://creativecommons.org/licenses/by-nc/4.0/",
            citation=("SpatialArena: an arena for geometric foundation models "
                      "(FeedForward-Benchmark), 2026."),
            notes=(
                "Captured for this benchmark (docs/benchmark/spatialarena.md) "
                "and released by its maintainers under CC BY-NC 4.0 (decision "
                "of 2026-10-29). Written by "
                "scripts/preprocess/materialize_spatialarena.py. In MonoDepth, "
                "VideoDepth and MVRecon, people and vehicles are blacked out "
                "of the images; SfM and SLAM frames are released unmasked."),
        ),
    "hammer":
        SourceLicense(
            name=("HAMMER (Highly Accurate Multi-Modal Dataset for DEnse 3D "
                  "Scene Regression)"),
            homepage=(
                "https://campar.in.tum.de/public_datasets/2022_arxiv_jung/"),
            status="permissive",
            license_name=(
                "MIT License (stated on the dataset page: 'published under MIT "
                "license')"),
            license_url=
            "https://campar.in.tum.de/public_datasets/2022_arxiv_jung/",
            citation=(
                "Jung, H., Ruhkamp, P., Zhai, G., Brasch, N., Li, Y., Verdie, "
                "Y., Song, J., Zhou, Y., Armagan, A., Ilic, S., Leonardis, A., "
                'Navab, N., Busam, B. "On the Importance of Accurate Geometry '
                'Data for Dense 3D Vision Tasks." CVPR 2023.'),
            notes=(
                "Used in the processed form of MoGe (HF "
                "Ruicheng/monocular-geometry-evaluation, HAMMER.zip; the card "
                "states no license of its own). Verified 2026-10-16."),
        ),
    "ibims1":
        SourceLicense(
            name=("iBims-1 (independent Benchmark images and matched scans, "
                  "version 1)"),
            homepage="https://www.asg.ed.tum.de/lmf/ibims1/",
            status="permissive",
            license_name="CC BY 4.0 (mediaTUM record 1455541, 'Rechte: by')",
            license_url="https://mediatum.ub.tum.de/1455541",
            citation=(
                'Koch, T., Liebel, L., Fraundorfer, F., Körner, M. "Evaluation '
                'of CNN-Based Single-Image Depth Estimation Methods." ECCV '
                "Workshops 2018."),
            notes=("Used in the processed form of MoGe (HF "
                   "Ruicheng/monocular-geometry-evaluation, iBims-1.zip). "
                   "Verified 2026-10-16."),
        ),
    "spring":
        SourceLicense(
            name="Spring",
            homepage="https://spring-benchmark.org/",
            status="permissive",
            license_name=(
                "CC BY 4.0 (DaRUS record doi:10.18419/DARUS-3376; Blender "
                "Foundation movie assets CC BY 4.0)"),
            license_url="https://doi.org/10.18419/darus-3376",
            citation=(
                "Mehl, L., Schmalfuss, J., Jahedi, A., Nalivayko, Y., Bruhn, "
                'A. "Spring: A High-Resolution High-Detail Dataset and '
                'Benchmark for Scene Flow, Optical Flow and Stereo." CVPR 2023.'
            ),
            notes=(
                "Train split only (test GT is withheld). Used in the processed "
                "form of MoGe (HF Ruicheng/monocular-geometry-evaluation, "
                "Spring.zip: every 5th train frame). Verified 2026-10-16."),
        ),
    "synth4k":
        SourceLicense(
            name="Synth4K (InfiniDepth game-captured 4K benchmark)",
            homepage="https://github.com/zju3dv/InfiniDepth",
            status="unconfirmed",
            license_name=
            ("No license published for the data (HF ritianyu/game_4k_data has "
             "no license or card)"),
            license_url="https://huggingface.co/datasets/ritianyu/game_4k_data",
            citation=(
                "Yu, H., Lin, H., Wang, J., Li, J., Wang, Y., Zhang, X., Wang, "
                'Y., Zhou, X., Hu, R., Peng, S. "InfiniDepth: '
                "Arbitrary-Resolution and Fine-Grained Depth Estimation with "
                'Neural Implicit Fields." arXiv:2601.03252, 2026.'),
            notes=(
                "Frames captured from commercial games (Cyberpunk 2077, Dead "
                "Island 2, Marvel's Spider-Man 2, Marvel's Spider-Man: Miles "
                "Morales, Watch Dogs: Legion); the game imagery belongs to the "
                "publishers. Paper: https://arxiv.org/abs/2601.03252; code: "
                "https://github.com/zju3dv/InfiniDepth; data: "
                "https://huggingface.co/datasets/ritianyu/game_4k_data (half "
                "of each game's scorable frames, 1,960 in total, re-packaged "
                "here, with attribution, by the repository maintainer's "
                "decision of 2026-10-16)."),
        ),
    "aerialmetric_decoupled":
        SourceLicense(
            name="AerialMetric-Decoupled",
            homepage="https://github.com/kuieless/AerialMetric",
            status="restricted_noncommercial",
            license_name=(
                "AerialMetric data terms: academic research and educational "
                "use only, no commercial use"),
            license_url=
            "https://github.com/kuieless/AerialMetric/blob/main/DATA_LICENSE.md",
            citation=(
                "Song, Z., Chen, G., Zhang, Y., Zou, Y., Fu, C., Yuan, Z., "
                'Huang, C., Cui, S., Cao, X. "AerialMetric: Benchmarking and '
                "Adapting UAV Monocular Metric Depth Estimation in the Real "
                'World." ECCV 2026 (arXiv:2606.29716).'),
            notes=(
                "Collected and released by the AerialMetric authors. Paper: "
                "https://arxiv.org/abs/2606.29716; data: "
                "https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026. "
                "Verified 2026-10-16."),
        ),
    "aerialmetric_oblique":
        SourceLicense(
            name="AerialMetric-Oblique (test split)",
            homepage="https://github.com/kuieless/AerialMetric",
            status="restricted_noncommercial",
            license_name=(
                "AerialMetric data terms (academic research and educational "
                "use only) plus the terms of each source dataset of the RGB "
                "images (UrbanBIS, GauU-Scene V2, UAVScenes, UrbanScene3D, "
                "OpenDroneMap, Esri)"),
            license_url=
            "https://github.com/kuieless/AerialMetric/blob/main/DATA_LICENSE.md",
            citation=(
                "Song, Z., Chen, G., Zhang, Y., Zou, Y., Fu, C., Yuan, Z., "
                'Huang, C., Cui, S., Cao, X. "AerialMetric: Benchmarking and '
                "Adapting UAV Monocular Metric Depth Estimation in the Real "
                'World." ECCV 2026 (arXiv:2606.29716).'),
            notes=(
                "RGB copyright stays with the original data providers; their "
                "terms take precedence. Paper: "
                "https://arxiv.org/abs/2606.29716; data: "
                "https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026. "
                "Verified 2026-10-16."),
        ),
    "aerialmetric_wild":
        SourceLicense(
            name="AerialMetric-Wild",
            homepage="https://github.com/kuieless/AerialMetric",
            status="unconfirmed",
            license_name=(
                "No rights granted to the images (frames of public online UAV "
                "videos, YouTube Terms of Service)"),
            license_url=
            "https://github.com/kuieless/AerialMetric/blob/main/DATA_LICENSE.md",
            citation=(
                "Song, Z., Chen, G., Zhang, Y., Zou, Y., Fu, C., Yuan, Z., "
                'Huang, C., Cui, S., Cao, X. "AerialMetric: Benchmarking and '
                "Adapting UAV Monocular Metric Depth Estimation in the Real "
                'World." ECCV 2026 (arXiv:2606.29716).'),
            notes=(
                "AerialMetric's annotations do not grant rights to the "
                "underlying video content. Paper: "
                "https://arxiv.org/abs/2606.29716; data: "
                "https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026 "
                "(half of the site's scorable frames re-packaged here, with "
                "attribution, by the repository maintainer's decision of "
                "2026-10-16)."),
        ),
}

# Maps a registered (task, dataset) pair -- see
# spatialarena/eval/dataset_registry.py -- to the SOURCE_LICENSES key(s) that
# apply to it. A packaged dataset folder gets one LICENSE_NOTICE.md section per
# source listed here.
DATASET_SOURCES: dict[tuple[str, str], tuple[str, ...]] = {
    ("MonoDepth", "marigold"):
        ("nyuv2", "kitti_depth", "eth3d", "diode", "scannet"),
    ("MonoDepth", "aerialmetric"):
        ("aerialmetric_decoupled", "aerialmetric_oblique", "aerialmetric_wild"),
    ("MonoDepth", "hammer"): ("hammer",),
    ("MonoDepth", "ibims1"): ("ibims1",),
    ("MonoDepth", "spring"): ("spring",),
    ("MonoDepth", "synth4k"): ("synth4k",),
    ("VideoDepth", "sintel"): ("sintel",),
    ("VideoDepth", "bonn"): ("bonn_rgbd_dynamic",),
    ("VideoDepth", "kitti"): ("kitti_depth",),
    ("VideoDepth", "dycheck"): ("dycheck",),
    ("SfM", "eth3d"): ("eth3d",),
    ("MVRecon", "eth3d"): ("eth3d",),
    ("SfM", "tanks_and_temples"): ("tanks_and_temples",),
    ("MVRecon", "tanks_and_temples"): ("tanks_and_temples",),
    ("SfM", "imc"): ("imc_phototourism",),
    ("MVRecon", "7scenes"): ("7scenes",),
    ("SLAM", "7scenes"): ("7scenes",),
    ("SfM", "dtu"): ("dtu_mvs",),
    ("MVRecon", "dtu"): ("dtu_mvs",),
    ("SfM", "scannetpp"): ("scannetpp",),
    ("MVRecon", "scannetpp"): ("scannetpp",),
    ("SfM", "hiroom"): ("hiroom",),
    ("MVRecon", "hiroom"): ("hiroom",),
    ("SfM", "texturepoorsfm"): ("texturepoorsfm",),
    ("MVRecon", "nrgbd"): ("nrgbd",),
    ("SLAM", "tum_dynamics"): ("tum_dynamics",),
    ("SLAM", "oxford_spires"): ("oxford_spires",),
    ("SLAM", "vbr"): ("vbr",),
    ("SLAM", "kitti_odometry"): ("kitti_odometry",),
}
for _task in ("MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM"):
    DATASET_SOURCES[(_task, "spatialarena")] = ("spatialarena",)
del _task

# Scenes a packaged dataset ships without, because their source may not be
# redistributed: scene name -> (SOURCE_LICENSES key, how a user rebuilds the
# scene from the source). The packer leaves the scene's folder out and drops it
# from the shipped metadata.json, so the downloaded dataset evaluates without
# it.
WITHHELD_SCENES: dict[tuple[str, str], dict[str, tuple[str, str]]] = {
    ("MonoDepth", "marigold"): {
        "scannet": (
            "scannet",
            "Download Marigold's copy (`wget -P "
            "<benchmark_root>/MonoDepth/marigold/scannet "
            "https://share.phys.ethz.ch/~pf/bingkedata/marigold/"
            "evaluation_dataset/scannet/scannet_val_sampled_800_1.tar"
            "`) "
            "and run `python "
            "scripts/preprocess/materialize_marigold_sample.py "
            "--marigold_root <benchmark_root>/MonoDepth/marigold --scenes "
            "scannet --max_samples -1`, which adds the 800 frames and the "
            "scene's metadata.json entry.",
        ),
    },
}


def withheld_scenes(task: str, dataset: str) -> dict[str, tuple[str, str]]:
    """Returns the scenes of task/dataset the packer leaves out.

    See WITHHELD_SCENES.
    """
    return WITHHELD_SCENES.get((task, dataset), {})


def _shipped_source_keys(task: str, dataset: str) -> tuple[str, ...]:
    """Returns the sources still shipped once the withheld scenes are out."""
    withheld = {key for key, _ in withheld_scenes(task, dataset).values()}
    return tuple(key for key in DATASET_SOURCES.get((task, dataset), ())
                 if key not in withheld)


def is_redistributable(task: str, dataset: str) -> bool:
    """Returns whether a dataset may be redistributed.

    It may not if a source still shipped with it (after its withheld scenes)
    forbids redistribution.
    """
    return all(SOURCE_LICENSES[key].status != "no_redistribution"
               for key in _shipped_source_keys(task, dataset))


def sources_for(task: str, dataset: str) -> tuple[SourceLicense, ...]:
    """Returns the source licenses of a registered (task, dataset) pair."""
    keys = DATASET_SOURCES.get((task, dataset), ())
    return tuple(SOURCE_LICENSES[key] for key in keys)


def is_safe_for_default_public_release(task: str, dataset: str) -> bool:
    """Returns whether a dataset may go into a default public release.

    It may not if any source shipped with it (its withheld scenes aside) has
    unconfirmed provenance/license -- those need a human to confirm before
    shipping in a public release, so the packaging script excludes them unless
    explicitly requested -- or forbids redistribution. A dataset without
    registered sources may not either.
    """
    keys = _shipped_source_keys(task, dataset)
    if not keys:
        return False
    return is_redistributable(task, dataset) and all(
        SOURCE_LICENSES[key].status != "unconfirmed" for key in keys)
