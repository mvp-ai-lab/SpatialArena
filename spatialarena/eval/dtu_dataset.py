# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import re
from pathlib import Path

from spatialarena.eval.utils import Dataset, SceneInfo


class DatasetDTU(Dataset):
    """
    The DTU dataset is preprocessed with our script at:
        scripts/preprocess/data_conversion/dtu_to_colmap.py
    The original url for downloading the DTU dataset is available at:
        https://roboimagedata.compute.dtu.dk/?page_id=36

    Required input:
        - Training set: https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view
        - Test set: https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_
        - Sample set: http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip
        - STL point clouds: http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip
        - GT depth map: GT Depths map:
          https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/
          Depths_raw.zip
    
    Then, use our preprocessing script to convert it to COLMAP format.
    You also need to copy the ObsMask from `SampleSet/MVS Data/ObsMask` to the dtu folder.
    After processing, your DTU dataset folder structure should look like:
    ```
        dtu:
            - Calibration
            - Points
            - ObsMask
            - scan_id:
                - cams
                - depths
                - images
                - masks
                - sparse
                - pair.txt
    ```

    NOTE:
    If you are evaluating 3DGS, please download the preprocessed data at:
        https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9

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
        dataset_folder: str = "dtu",
        list_filename: str = "scan_list_test"
    ):
        scene_infos = []

        dtu_path = self.data_path / dataset_folder
        data_list = dtu_path / f"{list_filename}.txt"
        stl_points_path = dtu_path / "Points/stl"
        obs_mask_path = dtu_path / "ObsMask"

        all_scenes = open(data_list, 'r', encoding='utf-8').readlines()
        all_scenes = list(map(str.strip, all_scenes))

        # DTU dataset typically has scenes organized by scan numbers
        # for scene_path in sorted(dtu_path.iterdir()):
        for scene in all_scenes:
            scene_path = dtu_path / scene
            if not scene_path.is_dir():
                continue

            # Check if this is a scan folder (typically named like "scan1", "scan2", etc.)
            scene = scene_path.name
            if not scene.startswith('scan'):
                continue

            if self.scenes and scene not in self.scenes:
                continue
            
            scan_number = int(re.findall(r'\d+', scene)[0])
            workspace_path = (
                self.run_path / self.run_name / "dtu" / self.method / scene
            )

            # DTU dataset structure
            image_path = scene_path / "images"
            sparse_gt_path = scene_path / "sparse"

            gt_point_path = stl_points_path / f'stl{scan_number:03d}_total.ply'
            assert gt_point_path.exists()

            obj_mask_path = obs_mask_path / f'ObsMask{scan_number}_10.mat'
            assert obj_mask_path.exists()

            plane_mask_path = obs_mask_path / f'Plane{scan_number}.mat'
            assert plane_mask_path.exists()

            scene_info = SceneInfo(
                dataset="dtu",
                category="dtu",  # DTU doesn't have categories like ETH3D
                scene=scene,
                workspace_path=workspace_path,
                image_path=image_path,
                sparse_gt_path=sparse_gt_path,
                dense_gt_path=gt_point_path,
                depth_gt_path=None,
                mask_path=[obj_mask_path, plane_mask_path],
                camera_priors_from_sparse_gt=True,
                colmap_extra_args=[],  # No special colmap args needed for DTU
            )

            scene_infos.append(scene_info)

        return scene_infos

    def prepare_scene(self, scene_info):
        # DTU dataset might need some preparation, but typically it's ready to use
        # You could add logic here to handle DTU-specific preprocessing if needed
        pass
