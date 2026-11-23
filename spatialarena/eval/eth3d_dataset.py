# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Dataset adapter for the raw ETH3D layout (<data_path>/eth3d/<category>)."""

import pathlib

from spatialarena.eval import utils


class DatasetETH3D(utils.Dataset):
    """Lists the ETH3D scenes of each category (e.g. dslr, rig).

    Attributes:
        data_path: Dataset root; scenes are under data_path/eth3d.
        categories: Categories to keep; empty keeps every category.
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
        return 0.001

    @property
    def rotation_accuracy_gt(self):
        return 1

    def list_scenes(self):
        scene_infos = []
        for category_path in (self.data_path / "eth3d").iterdir():
            if not category_path.is_dir() or (self.categories and
                                              category_path.name
                                              not in self.categories):
                continue

            category = category_path.name

            for scene_path in sorted(category_path.iterdir()):
                if not scene_path.is_dir():
                    continue

                scene = scene_path.name
                if self.scenes and scene not in self.scenes:
                    continue

                workspace_path = (self.run_path / self.run_name / "eth3d" /
                                  category / self.method / scene)
                # ETH3D stores both original DSLR images and the images used
                # by the undistorted COLMAP calibration. The latter are the
                # canonical benchmark inputs.
                image_path = scene_path / "images" / "dslr_images_undistorted"
                sparse_gt_path = list(
                    scene_path.glob("*_calibration_undistorted"))[0]

                dense_gt_path = scene_path / f"{category}_scan_eval"
                if dense_gt_path.exists() is False:
                    dense_gt_path = None

                depth_path = scene_path / "ground_truth_depth" / "dslr_images"
                if not depth_path.exists():
                    depth_path = None

                colmap_extra_args = []
                if category == "dslr":
                    colmap_extra_args.extend(["--data_type", "individual"])
                elif category == "rig":
                    colmap_extra_args.extend(["--data_type", "video"])

                scene_info = utils.SceneInfo(
                    dataset="eth3d",
                    category=category,
                    scene=scene,
                    workspace_path=workspace_path,
                    image_path=image_path,
                    sparse_gt_path=sparse_gt_path,
                    dense_gt_path=dense_gt_path,
                    depth_gt_path=depth_path,
                    mask_path=[],
                    camera_priors_from_sparse_gt=True,
                    colmap_extra_args=colmap_extra_args,
                )

                scene_infos.append(scene_info)

        return scene_infos

    def prepare_scene(self, scene_info):
        # Nothing to prepare for ETH3D.
        pass
