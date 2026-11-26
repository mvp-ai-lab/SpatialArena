# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""OmniVGGT, including its depth- and camera-conditioned tracks.

The images are resized and padded to a square at --image_load_resolution and
written to disk (with the matching depth and camera priors) for OmniVGGT's
own loader, so depth and intrinsics of a depth-only run are restored to each
source image. The scene is predicted in chunks of --max_images_per_run
images (one image per chunk in single-view mode); --max_images_per_run also
caps the number of frames, so there is one chunk otherwise.
"""

from __future__ import annotations

import pathlib
import shutil
import time

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

TRACKS = (
    "rgb",
    "rgb_depth",
    "rgb_cameras",
    "rgb_masked_depth",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
)
CAMERA_TRACKS = ("rgb_cameras", "rgb_masked_depth_cameras",
                 "rgb_sparse_depth_cameras")
DEPTH_TRACKS = (
    "rgb_depth",
    "rgb_masked_depth",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
)
MASKED_DEPTH_TRACKS = ("rgb_masked_depth", "rgb_masked_depth_cameras")
SPARSE_DEPTH_TRACKS = ("rgb_sparse_depth", "rgb_sparse_depth_cameras")
# The input's long side the sparse points are drawn for
# (utils.load_sparse_depth_priors).
SPARSE_MODEL_LONG_SIDE = 518
INPUTS_SUBDIR = "_omnivggt_inputs"


def _load_track_depth_priors(data_path, image_paths, track):
    """Loads the depth priors of a depth track.

    Masked tracks use masked ground truth (spatialarena/depth_prior_masking.py);
    sparse tracks use real sparse depth, LiDAR or COLMAP keypoints
    (spatialarena/sparse_depth.py).
    """
    if track in SPARSE_DEPTH_TRACKS:
        return utils.load_sparse_depth_priors(
            data_path, image_paths, model_long_side=SPARSE_MODEL_LONG_SIDE)
    if track in MASKED_DEPTH_TRACKS:
        return utils.load_masked_depth_priors(data_path, image_paths)
    return utils.load_optional_depth_priors(data_path, image_paths)


def resolve_checkpoint(ckpt_path):
    """Returns a local checkpoint file, downloading a HF repo id's."""
    path = pathlib.Path(ckpt_path).expanduser()
    if path.is_file():
        return str(path)
    import huggingface_hub  # pylint: disable=import-outside-toplevel

    return huggingface_hub.hf_hub_download(repo_id=ckpt_path,
                                           filename="OmniVGGT.safetensors")


def _clear_files(directory):
    """Creates `directory` and removes the files and links in it."""
    directory.mkdir(parents=True, exist_ok=True)
    for existing in directory.iterdir():
        if existing.is_file() or existing.is_symlink():
            existing.unlink()


def resize_depth_prior_for_omnivggt(depth, target_size):
    """Resizes and pads a depth prior to the square OmniVGGT input."""
    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim == 3:
        depth_array = depth_array[..., 0]
    height, width = depth_array.shape
    scale = min(target_size / float(width), target_size / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    try:
        import cv2  # pylint: disable=import-outside-toplevel

        resized = cv2.resize(
            depth_array,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
    except ImportError:
        from PIL import Image  # pylint: disable=import-outside-toplevel

        resized = np.asarray(
            Image.fromarray(depth_array).resize(
                (resized_width, resized_height),
                Image.Resampling.NEAREST,
            ),
            dtype=np.float32,
        )
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    offset_x = (target_size - resized_width) // 2
    offset_y = (target_size - resized_height) // 2
    canvas[offset_y:offset_y + resized_height,
           offset_x:offset_x + resized_width] = resized
    return canvas


def adjust_intrinsics_for_square_materialization(intrinsic, transform):
    """Applies a (scale, offset_x, offset_y) resize and pad to intrinsics."""
    scale, offset_x, offset_y = transform
    adjusted = np.asarray(intrinsic, dtype=np.float32).copy()
    adjusted[0, 0] *= scale
    adjusted[1, 1] *= scale
    adjusted[0, 2] = adjusted[0, 2] * scale + float(offset_x)
    adjusted[1, 2] = adjusted[1, 2] * scale + float(offset_y)
    return adjusted


def materialize_inputs(args, image_paths):
    """Writes the padded images and the track's priors for OmniVGGT's loader.

    Args:
        args: The parsed options.
        image_paths: Source images of one chunk.

    Returns:
        An (image_dir, depth_dir or None, camera_dir or None, transforms)
        tuple; transforms are the per-image `utils.SpatialTransform`s.

    Raises:
        FileNotFoundError: A depth track found no depth prior.
    """
    # pylint: disable-next=import-outside-toplevel
    from PIL import Image, ImageOps

    input_root = args.output_path / INPUTS_SUBDIR
    image_dir = input_root / "images"
    depth_dir = input_root / "depths"
    camera_dir = input_root / "cameras"
    _clear_files(image_dir)

    target_size = int(args.image_load_resolution)
    stems = []
    image_transforms = []
    spatial_transforms = []
    for index, image_path in enumerate(image_paths):
        src = pathlib.Path(image_path)
        stem = f"{index:06d}_{src.stem}"
        stems.append(stem)
        dst = image_dir / f"{stem}.png"
        with Image.open(src) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            scale = min(target_size / float(width), target_size / float(height))
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            offset_x = (target_size - resized_width) // 2
            offset_y = (target_size - resized_height) // 2
            resized = image.resize((resized_width, resized_height),
                                   Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            canvas.paste(resized, (offset_x, offset_y))
            canvas.save(dst)
            image_transforms.append((scale, offset_x, offset_y))
            spatial_transforms.append(
                utils.SpatialTransform(
                    source_wh=(width, height),
                    resized_wh=(resized_width, resized_height),
                    padding=(
                        offset_x,
                        offset_y,
                        target_size - resized_width - offset_x,
                        target_size - resized_height - offset_y,
                    ),
                    policy="omnivggt_resize_pad_square",
                ))

    active_depth_dir = None
    if args.track in DEPTH_TRACKS:
        _clear_files(depth_dir)
        depths, has_depth, _ = _load_track_depth_priors(args.data_path,
                                                        image_paths, args.track)
        if not has_depth.any():
            raise FileNotFoundError(
                f"No matched depth priors found for {args.data_path}")
        for stem, depth in zip(stems, depths, strict=True):
            if depth is None:
                continue
            np.save(
                depth_dir / f"{stem}.npy",
                resize_depth_prior_for_omnivggt(depth,
                                                target_size).astype(np.float32,
                                                                    copy=False),
            )
        active_depth_dir = depth_dir

    active_camera_dir = None
    if args.track in CAMERA_TRACKS:
        _clear_files(camera_dir)
        c2w, intrinsics = utils.load_colmap_camera_priors(
            args.data_path,
            image_paths,
            colmap_path=args.colmap_path,
            colmap_model_subdir=args.colmap_model_subdir,
        )
        for stem, pose, intrinsic, transform in zip(stems, c2w, intrinsics,
                                                    image_transforms):
            adjusted_intrinsic = adjust_intrinsics_for_square_materialization(
                intrinsic, transform)
            with (camera_dir / f"{stem}.txt").open("w",
                                                   encoding="utf-8") as handle:
                for row in pose[:3, :4]:
                    handle.write(" ".join(f"{value:.9g}" for value in row) +
                                 "\n")
                for row in adjusted_intrinsic:
                    handle.write(" ".join(f"{value:.9g}" for value in row) +
                                 "\n")
        active_camera_dir = camera_dir

    return image_dir, active_depth_dir, active_camera_dir, spatial_transforms


@registry.register_model
class OmniVGGT(base.BaseModel):
    """OmniVGGT."""

    name = "OMNIVGGT"
    aliases = ("OMNI_VGGT",)
    description = "Run OmniVGGT and export benchmark prediction artifacts."
    code_path_env = "OMNIVGGT_CODE_PATH"
    code_packages = ("omnivggt",)
    ckpt_env = "OMNIVGGT_CKPT_PATH"
    ckpt_default = "Livioni/OmniVGGT"
    uses_resolution = False
    tracks = TRACKS

    def __init__(self, args):
        super().__init__(args)
        self.chunk_size = 0
        self.peak_memory = 0.0
        self.depth_prior_frames = 0
        self.camera_prior_frames = 0

    def add_code_path(self):
        utils.add_code_path(self.args.code_path, self.code_packages)

    def select_images(self):
        image_paths = utils.find_image_paths(self.images_dir)
        if self.args.track in CAMERA_TRACKS:
            image_paths = utils.resolve_camera_conditioned_image_paths(
                self.args.data_path,
                image_paths,
                colmap_path=self.args.colmap_path,
            )
        # --max_images_per_run also caps the frames: chunks are independent
        # predictions in their own coordinate frames, simply concatenated,
        # so capping the frames first makes it one chunk.
        return utils.select_interval(image_paths, self.args.interval,
                                     self.args.max_images_per_run)

    def load(self, device):
        # pylint: disable=import-outside-toplevel
        from omnivggt.models import omnivggt as omnivggt_net
        from safetensors import torch as safetensors_torch

        model = omnivggt_net.OmniVGGT()
        model.load_state_dict(safetensors_torch.load_file(
            resolve_checkpoint(self.args.ckpt_path)),
                              strict=True)
        model.to(device).eval()
        return model

    def preprocess(self, image_paths, batch_index=None):
        # Writes the chunk's inputs and loads them with OmniVGGT's loader.
        import visual_util  # pylint: disable=import-outside-toplevel

        args = self.args
        image_dir, depth_dir, camera_dir, transforms = materialize_inputs(
            args, image_paths)
        (images, extrinsics, intrinsics, depthmaps, masks, depth_indices,
         camera_indices) = visual_util.load_images_and_cameras(
             str(image_dir),
             str(camera_dir) if camera_dir is not None else None,
             str(depth_dir) if depth_dir is not None else None,
             args.image_load_resolution,
         )
        utils.record_model_input(images.shape[-1], images.shape[-2])
        device = self.device
        data = {
            "images": images.to(device),
            "extrinsics": extrinsics.to(device),
            "intrinsics": intrinsics.to(device),
            "depth": depthmaps.to(device),
            "mask": masks.to(device),
            "depth_gt_index": depth_indices,
            "camera_gt_index": camera_indices,
        }
        return base.ModelInput(image_paths,
                               data,
                               transforms,
                               batch_index,
                               scratch_dirs=[args.output_path / INPUTS_SUBDIR],
                               extra={"images": images})

    def forward(self, inputs):
        # pylint: disable=import-outside-toplevel
        import torch
        from omnivggt.utils import pose_enc

        images = inputs.extra["images"]
        with torch.no_grad():
            predictions = self.model.inference(**inputs.data)
            pred_extrinsics, pred_intrinsics = (
                pose_enc.pose_encoding_to_extri_intri(
                    predictions["pose_enc"],
                    images.shape[-2:],
                ))
        depths = utils.as_numpy(predictions["depth"]).astype(np.float32)
        if depths.ndim == 5:
            depths = depths[0, ..., 0]
        confidence = predictions.get("depth_conf", None)
        if confidence is not None:
            confidence = utils.as_numpy(confidence)
            if confidence.ndim == 4:
                confidence = confidence[0]
        prediction = base.Prediction(depth=depths, confidence=confidence)
        if self.args.export_colmap:
            prediction.extrinsics_w2c = utils.as_numpy(
                pred_extrinsics)[0].astype(np.float32)
            prediction.intrinsics = utils.as_numpy(pred_intrinsics)[0].astype(
                np.float32)
        if not self.args.depth_only:
            prediction.colors = utils.colors_from_torch_images(images)
        return prediction

    def predict(self, image_paths, batch_index=None):
        # Predicts the images in chunks and concatenates the chunks.
        import torch  # pylint: disable=import-outside-toplevel

        self.chunk_size = self.args.max_images_per_run
        if self.chunk_size <= 0:
            self.chunk_size = len(image_paths)
        if utils.single_view_mode():
            self.chunk_size = 1  # MonoDepth: one image per forward pass.
        fields = ("depth", "confidence", "extrinsics_w2c", "intrinsics",
                  "colors")
        chunks = {field: [] for field in fields}
        transforms = []
        keep = utils.should_keep_intermediate(self.args)
        for start in range(0, len(image_paths), self.chunk_size):
            inputs = self.preprocess(image_paths[start:start + self.chunk_size])
            transforms.extend(inputs.transforms)
            prediction = self.forward(inputs)
            for field in fields:
                value = getattr(prediction, field)
                if value is not None:
                    chunks[field].append(value)
            self.depth_prior_frames += len(inputs.data["depth_gt_index"])
            self.camera_prior_frames += len(inputs.data["camera_gt_index"])
            if torch.cuda.is_available():
                self.peak_memory = max(
                    self.peak_memory,
                    torch.cuda.max_memory_allocated() / (1024.0**2))
            utils.cleanup_intermediate_paths(list(inputs.scratch_dirs),
                                             keep=keep)

        merged = base.Prediction(
            **{
                field: np.concatenate(values, axis=0) if values else None
                for field, values in chunks.items()
            })
        return merged, base.ModelInput(
            image_paths,
            transforms=transforms,
            batch_index=batch_index,
            scratch_dirs=[self.args.output_path / INPUTS_SUBDIR])

    def metrics_extra(self, depth_only, images_per_run=None):
        del depth_only, images_per_run  # Same fields in both modes.
        return {
            "Peak GPU Memory": self.peak_memory,
            "track": self.args.track,
            "image_load_resolution": self.args.image_load_resolution,
            "depth_prior_frames": self.depth_prior_frames,
            "camera_prior_frames": self.camera_prior_frames,
            "max_images_per_run": self.chunk_size,
        }

    def cleanup(self, inputs):
        super().cleanup(inputs)
        if not utils.should_keep_intermediate(self.args):
            shutil.rmtree(self.args.output_path / INPUTS_SUBDIR,
                          ignore_errors=True)

    def run(self):
        """Runs OmniVGGT on the scene and writes its artifacts.

        Raises:
            FileNotFoundError: The scene has no images directory.
            ValueError: No images, or --depth_only without --export_depth.
        """
        args = self.args
        self.images_dir = utils.canonical_image_root(args.data_path)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"images directory not found: {self.images_dir}")
        self.image_paths = self.select_images()
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")
        if args.dry_run:
            utils.print_dry_run(args, self.method_label, len(self.image_paths))
            return
        if args.depth_only and not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")

        self.add_code_path()
        self.device = self.select_device()
        import torch  # pylint: disable=import-outside-toplevel

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # "inference time" includes loading the model.
        self.started_at = time.time()
        self.model = self.load(self.device)
        utils.mark_model_loaded()
        if args.depth_only:
            self.run_depth_only()
        else:
            self.run_full()

    def run_depth_only(self):
        """Exports depth and cameras of all chunks, restored."""
        args = self.args
        prediction, inputs = self.predict(self.image_paths)
        utils.export_spatial_depth_predictions(args.output_path,
                                               self.image_paths,
                                               prediction.depth,
                                               prediction.confidence,
                                               inputs.transforms)
        cameras = self.cameras_for_export(inputs, prediction)
        if args.export_colmap and cameras is not None:
            extrinsics, intrinsics, (width, height), _ = cameras
            utils.export_camera_pose_priors(
                args.output_path,
                self.images_dir,
                self.image_paths,
                extrinsics,
                intrinsics,
                width=width,
                height=height,
                colmap_model_subdir=args.colmap_model_subdir,
            )
        self.write_metrics(depth_only=True,
                           metrics_extra=self.metrics_extra(depth_only=True))
        self.cleanup(inputs)
