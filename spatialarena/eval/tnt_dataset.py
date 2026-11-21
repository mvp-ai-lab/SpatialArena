# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from pathlib import Path

from spatialarena.eval.utils import Dataset, SceneInfo


class DatasetTanksAndTemples(Dataset):
    """
    The Tanks and Temples dataset is preprocessed with our script at:
        scripts/preprocess/data_conversion/tnt_to_colmap.py
    The original url for downloading the Tanks and Temples dataset is available at:
        https://www.tanksandtemples.org/download/

    ##########################################################################################
    NOTE: For benchmarking reconstruction quality of 3DGS/NeRF, please download the
    Tanks-and-Temples dataset preprocessed by GOF at:
        https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/blob/main/TNT_GOF.zip
    ##########################################################################################

    Required input:
        - Images: {scene}_images/: input images for 3D reconstruction tasks
        - Camera Poses: {scene}_COLMAP_SfM.log
        - Alignment file: {scene}_trans.txt, a 4*4 transformation matrix which aligns
                          the COLMAP reconstruction to the according ground-truth 
                          point cloud
        - Crop file: {scene}.json, includes the bounding box coordinates for each model
        - Ground-truth point cloud: {scene}.ply
        
    Then, use our preprocessing script to convert it to COLMAP format.
    After processing, your Tanks and Temples dataset folder structure should look like:
    ```
        tanks_and_temples:
            - Barn
                - Barn.ply
                - Barn_COLMAP_SfM.log
                - Barn.json
                - Barn_trans.txt
                - database.db
                - images/
                - sparse/
            - Courthouse
            ...
    ```

    """

    def __init__(
        self,
        data_path: Path,
        categories: list[str],
        scenes: list[Path],
        method: str,
        run_path: Path,
        run_name: str,
    ):
        super().__init__()
        self.data_path = data_path
        self.categories = categories
        self.scenes = scenes
        self.method = method
        self.run_path = run_path
        self.run_name = run_name

    @property
    def position_accuracy_gt(self):
        return 1

    @property
    def rotation_accuracy_gt(self):
        return 1

    def list_scenes(
        self,
        dataset_folder: str = "tanks_and_temples",
        list_filename: str = ""  # pylint: disable=unused-argument
    ):
        scene_infos = []

        tnt_path = self.data_path / dataset_folder

        all_scenes = ["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"]

        # for scene_path in sorted(tnt_path.iterdir()):
        for scene in all_scenes:
            scene_path = tnt_path / scene
            if not scene_path.is_dir():
                continue

            # Check if this is a scan folder (typically named like "scan1", "scan2", etc.)
            scene = scene_path.name

            if self.scenes and scene not in self.scenes:
                continue

            workspace_path = (
                self.run_path / self.run_name / "tanks_and_temples" / self.method / scene
            )

            # DTU dataset structure
            image_path = scene_path / "images"
            sparse_gt_path = scene_path / "sparse"

            gt_point_path = scene_path / f'{scene}.ply'
            assert gt_point_path.exists()

            scene_info = SceneInfo(
                dataset=dataset_folder,
                category="eval",  # tnt doesn't have categories like ETH3D
                scene=scene,
                workspace_path=workspace_path,
                image_path=image_path,
                sparse_gt_path=sparse_gt_path,
                dense_gt_path=gt_point_path,
                depth_gt_path=None,
                mask_path=None,
                camera_priors_from_sparse_gt=True,
                colmap_extra_args=[],  # No special colmap args needed for DTU
            )

            scene_infos.append(scene_info)

        return scene_infos

    def prepare_scene(self, scene_info):
        # DTU dataset might need some preparation, but typically it's ready to use
        # You could add logic here to handle DTU-specific preprocessing if needed
        pass
