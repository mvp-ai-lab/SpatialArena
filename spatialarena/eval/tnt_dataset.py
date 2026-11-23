# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Dataset adapter for the Tanks and Temples layout."""

import pathlib

from spatialarena.eval import utils


class DatasetTanksAndTemples(utils.Dataset):
    """Lists the Tanks and Temples evaluation scenes.

    The Tanks and Temples dataset is preprocessed with our script at
    scripts/preprocess/data_conversion/tnt_to_colmap.py. The original dataset
    can be downloaded from https://www.tanksandtemples.org/download/.

    NOTE: For benchmarking reconstruction quality of 3DGS/NeRF, please
    download the Tanks-and-Temples dataset preprocessed by GOF at:
    https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/blob/main/TNT_GOF.zip

    Required input:
        - Images: {scene}_images/: input images for 3D reconstruction tasks.
        - Camera Poses: {scene}_COLMAP_SfM.log.
        - Alignment file: {scene}_trans.txt, a 4*4 transformation matrix
          which aligns the COLMAP reconstruction to the according
          ground-truth point cloud.
        - Crop file: {scene}.json, includes the bounding box coordinates for
          each model.
        - Ground-truth point cloud: {scene}.ply.

    Then, use our preprocessing script to convert it to COLMAP format.
    After processing, your Tanks and Temples dataset folder structure should
    look like:

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

    Attributes:
        data_path: Dataset root.
        categories: Unused; Tanks and Temples has no categories.
        scenes: Scene names to keep; empty keeps every scene.
        method: Method name used in the workspace path.
        run_path: Root of the method runs.
        run_name: Run directory name under run_path.
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        categories: list[str],
        scenes: list[pathlib.Path],
        method: str,
        run_path: pathlib.Path,
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
        """Lists the six evaluation scenes present under dataset_folder.

        Args:
            dataset_folder: Dataset directory under data_path.
            list_filename: Unused.

        Returns:
            One `utils.SceneInfo` per scene directory found.
        """
        scene_infos = []

        tnt_path = self.data_path / dataset_folder

        all_scenes = [
            "Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom",
            "Truck"
        ]

        for scene in all_scenes:
            scene_path = tnt_path / scene
            if not scene_path.is_dir():
                continue

            scene = scene_path.name

            if self.scenes and scene not in self.scenes:
                continue

            workspace_path = (self.run_path / self.run_name /
                              "tanks_and_temples" / self.method / scene)

            image_path = scene_path / "images"
            sparse_gt_path = scene_path / "sparse"

            gt_point_path = scene_path / f'{scene}.ply'
            assert gt_point_path.exists()

            scene_info = utils.SceneInfo(
                dataset=dataset_folder,
                category="eval",  # TnT has no categories like ETH3D.
                scene=scene,
                workspace_path=workspace_path,
                image_path=image_path,
                sparse_gt_path=sparse_gt_path,
                dense_gt_path=gt_point_path,
                depth_gt_path=None,
                mask_path=None,
                camera_priors_from_sparse_gt=True,
                colmap_extra_args=[],
            )

            scene_infos.append(scene_info)

        return scene_infos

    def prepare_scene(self, scene_info):
        # Nothing to prepare for Tanks and Temples.
        pass
