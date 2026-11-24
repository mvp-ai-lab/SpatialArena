# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""DUSt3R, MASt3R and Pow3R through MapAnything's external-model wrappers.

MapAnything wraps each of these pairwise methods with its global alignment
(`mapanything.models.external.<method>`): DUSt3R and Pow3R with bundle
adjustment, MASt3R with sparse global alignment. The three read images the
same way, so they share one preprocessing: each image is padded into the
16 px patch grid, and depth and intrinsics are restored to the source
images. A single image (single-view mode) runs as the pair (image, image),
DUSt3R's own monocular mode.
"""

from __future__ import annotations

import errno
import getpass
import importlib
import os
import pathlib
import shutil
import sys
import tempfile
import time
from typing import Any, ClassVar
from urllib import parse

import numpy as np

from spatialarena.models import base
from spatialarena.models import registry
from spatialarena.models import utils

# Wrappers that predict from image pairs: a single image runs as the pair
# (image, image) and only the first view's prediction is kept.
PAIRWISE_WRAPPERS = ("DUSt3RBAWrapper", "MASt3RSGAWrapper", "Pow3RBAWrapper")

_CHECKPOINT_SUFFIXES = (".pth", ".pt", ".bin", ".safetensors")

# The scene of the last DUSt3R-family global alignment. MapAnything's wrappers
# drop its per-view confidence from their outputs; attach_aligned_confidence
# puts it back.
_ALIGNED_SCENE: dict[str, Any] = {}


def spatialarena_root() -> str:
    """Returns the benchmark root (spatialarena/paths.py)."""
    return os.path.expanduser(
        os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))


def default_mapanything_code_path() -> pathlib.Path:
    """Returns the default MapAnything checkout of the wrappers.

    $MAPANYTHING_CODE_PATH, else the code default of the MAPANYTHING
    MethodSpec ($METHOD_CODE_ROOT/map-anything).
    """
    env_path = os.environ.get("MAPANYTHING_CODE_PATH")
    if env_path:
        return pathlib.Path(env_path)
    code_default = base.BaseModel.spec_default(
        "code_default", "$METHOD_CODE_ROOT/map-anything", "MAPANYTHING")
    return pathlib.Path(base.expand_code_root(code_default))


def add_optional_path(path, package_names):
    """Puts a method's checkout on sys.path when one is given."""
    if path is not None:
        utils.add_code_path(path, package_names)


def add_nested_method_paths(path):
    """Puts the nested dust3r and croco submodules of a checkout on sys.path.

    Args:
        path: The method's checkout, or None.
    """
    if path is None:
        return
    root = pathlib.Path(path).expanduser().resolve()
    for nested in (root / "dust3r", root / "dust3r" / "croco", root / "croco"):
        if nested.is_dir():
            utils.add_code_path(nested, ())


def import_symbol(module_name, symbol_name):
    """Imports `module_name.symbol_name`.

    Args:
        module_name: The module.
        symbol_name: The attribute.

    Returns:
        The attribute.

    Raises:
        RuntimeError: The module cannot be imported or lacks the attribute.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise RuntimeError(
            f"Could not import {module_name!r}. Install/clone the method "
            "dependencies and pass the appropriate *_CODE_PATH to the "
            "benchmark runner.") from error
    try:
        return getattr(module, symbol_name)
    except AttributeError as error:
        raise RuntimeError(
            f"{module_name!r} does not expose {symbol_name!r}") from error


def describe_structure(value, *, depth=0):
    """Returns a compact nested type summary, for diagnostics.

    Args:
        value: Any value.
        depth: Nesting level of `value`; deeper levels print only the type.

    Returns:
        The summary.
    """
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = ", ".join(str(key) for key in keys[:8])
        if len(keys) > 8:
            preview += ", ..."
        return f"dict(keys=[{preview}])"
    if isinstance(value, (list, tuple)):
        inner = "empty"
        if value:
            inner = describe_structure(value[0], depth=depth + 1)
        return f"{type(value).__name__}(len={len(value)}, first={inner})"
    if hasattr(value, "shape"):
        return f"{type(value).__name__}(shape={getattr(value, 'shape', None)})"
    return type(value).__name__


def capture_aligned_scene(wrapper_module):
    """Keeps the scene MASt3R's `sparse_global_alignment` returns.

    DUSt3R and Pow3R keep theirs through `patch_dust3r_global_aligner`.

    Args:
        wrapper_module: The MapAnything wrapper module.
    """
    try:
        module = importlib.import_module(wrapper_module)
    except Exception:  # pylint: disable=broad-exception-caught
        return
    original = getattr(module, "sparse_global_alignment", None)
    if original is None or getattr(original, "_feedforward_capture", False):
        return

    def capturing_alignment(*args, **kwargs):
        scene = original(*args, **kwargs)
        _ALIGNED_SCENE["scene"] = scene
        return scene

    capturing_alignment._feedforward_capture = True
    module.sparse_global_alignment = capturing_alignment


def attach_aligned_confidence(outputs):
    """Gives each view's output the aligned scene's confidence as "conf".

    The confidence is the model's own (exp scale). DUSt3R and Pow3R
    (PointCloudOptimizer) hold it in `scene.im_conf`; MASt3R's sparse scene
    returns it from `get_dense_pts3d()`. Outputs that already have "conf",
    or a scene without confidence, are left alone.

    Args:
        outputs: Per-view output dicts of the wrapper, changed in place.
    """
    scene = _ALIGNED_SCENE.pop("scene", None)
    if scene is None or not outputs:
        return
    if all("conf" in output for output in outputs):
        return
    if hasattr(scene, "im_conf"):
        confs = list(scene.im_conf)
    elif hasattr(scene, "get_dense_pts3d"):
        confs = list(scene.get_dense_pts3d()[2])
    else:
        return
    if len(confs) != len(outputs):
        return
    for output, conf in zip(outputs, confs):
        depth = output.get("depth_along_ray")
        if depth is None:
            continue
        height, width = utils.as_numpy(depth).squeeze().shape[:2]
        conf = utils.as_numpy(conf).astype(np.float32).reshape(-1)
        if conf.size == height * width:
            output["conf"] = conf.reshape(height, width)


def _has_dust3r_keys(value):
    return isinstance(value, dict) and {"view1", "view2", "pred1", "pred2"
                                       }.issubset(value)


def _tuple_to_dust3r_dict(value):
    """Returns a (view1, view2, pred1, pred2, ...) sequence as a dict."""
    if isinstance(value, (list, tuple)) and len(value) >= 4 and all(
            isinstance(item, dict) for item in value[:4]):
        return {
            "view1": value[0],
            "view2": value[1],
            "pred1": value[2],
            "pred2": value[3],
        }
    return None


def _normalize_dust3r_output(dust3r_output, collate_with_cat):
    """Turns list-form DUSt3R inference output into the aligner's dict."""
    if _has_dust3r_keys(dust3r_output):
        return dust3r_output
    if isinstance(dust3r_output, (list, tuple)) and dust3r_output:
        if len(dust3r_output) == 1:
            return _normalize_dust3r_output(dust3r_output[0], collate_with_cat)
        if all(_has_dust3r_keys(item) for item in dust3r_output):
            dust3r_output = collate_with_cat(list(dust3r_output), lists=True)
        else:
            tuple_dict = _tuple_to_dust3r_dict(dust3r_output)
            if tuple_dict is not None:
                return tuple_dict
            if all(
                    _tuple_to_dust3r_dict(item) is not None
                    for item in dust3r_output):
                collated_tuple = collate_with_cat(list(dust3r_output),
                                                  lists=True)
                tuple_dict = _tuple_to_dust3r_dict(collated_tuple)
                if tuple_dict is not None:
                    return tuple_dict
        print(
            "Unrecognized DUSt3R-family output structure: "
            f"{describe_structure(dust3r_output)}",
            file=sys.stderr,
            flush=True,
        )
    return dust3r_output


def patch_dust3r_global_aligner(wrapper_module):
    """Makes MapAnything's DUSt3R-family wrappers accept list-form output.

    Some DUSt3R/Pow3R revisions return a list of per-batch dicts when the
    input pairs mix image shapes. MapAnything's wrappers pass that value
    directly to the `global_aligner` imported from `dust3r.cloud_opt`, which
    expects one dict with view1/view2/pred1/pred2 keys. The patched aligner
    normalizes the output first and keeps the scene for
    `attach_aligned_confidence`, without patching the external checkouts.

    Args:
        wrapper_module: The MapAnything wrapper module.
    """
    if not (wrapper_module.endswith(".dust3r") or
            wrapper_module.endswith(".pow3r") or
            wrapper_module.endswith(".mast3r")):
        return
    try:
        module = importlib.import_module(wrapper_module)
        device_utils = importlib.import_module("dust3r.utils.device")
    except Exception:  # pylint: disable=broad-exception-caught
        return
    original_global_aligner = getattr(module, "global_aligner", None)
    collate_with_cat = getattr(device_utils, "collate_with_cat", None)
    if original_global_aligner is None or collate_with_cat is None:
        return
    if getattr(original_global_aligner, "_feedforward_list_patch", False):
        return

    def patched_global_aligner(dust3r_output, *args, **kwargs):
        normalized_output = _normalize_dust3r_output(dust3r_output,
                                                     collate_with_cat)
        try:
            scene = original_global_aligner(normalized_output, *args, **kwargs)
        except TypeError:
            print(
                "DUSt3R-family global_aligner failed with structures: "
                f"original={describe_structure(dust3r_output)}, "
                f"normalized={describe_structure(normalized_output)}",
                file=sys.stderr,
                flush=True,
            )
            raise
        _ALIGNED_SCENE["scene"] = scene
        return scene

    patched_global_aligner._feedforward_list_patch = True
    module.global_aligner = patched_global_aligner


def candidate_checkpoint_files(root):
    """Returns the checkpoint files under `root`, by suffix, then sorted."""
    candidates = []
    for suffix in _CHECKPOINT_SUFFIXES:
        for path in sorted(root.rglob(f"*{suffix}")):
            name = path.name.lower()
            if "retrieval" not in name and "codebook" not in name:
                candidates.append(path)
    return candidates


def resolve_checkpoint_arg(checkpoint):
    """Resolves --ckpt_path to a checkpoint file.

    Args:
        checkpoint: A local file or directory, a Hugging Face repo id or a
            direct http(s) URL (downloaded once into $TORCH_HOME).

    Returns:
        The checkpoint file, or `checkpoint` unchanged when it is none of
        these.

    Raises:
        FileNotFoundError: The directory or repo holds no checkpoint file.
        RuntimeError: huggingface_hub is needed but not installed.
    """
    if not checkpoint:
        return checkpoint
    parsed = parse.urlparse(checkpoint)
    if parsed.scheme in {"http", "https"}:
        import torch  # pylint: disable=import-outside-toplevel

        cache_root = pathlib.Path(
            os.environ.get("TORCH_HOME",
                           pathlib.Path.home() / ".cache" / "torch"))
        target_dir = cache_root / "hub" / "checkpoints"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / pathlib.Path(parsed.path).name
        if not target_path.is_file():
            torch.hub.download_url_to_file(checkpoint,
                                           str(target_path),
                                           progress=True)
        return str(target_path)

    path = pathlib.Path(checkpoint).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidates = candidate_checkpoint_files(path)
        if candidates:
            return str(candidates[0])
        raise FileNotFoundError(f"No checkpoint file found under {path}")

    if "/" in checkpoint:
        try:
            import huggingface_hub  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            raise RuntimeError(
                f"Checkpoint {checkpoint!r} looks like an HF repo id, but "
                "huggingface_hub is unavailable.") from error
        snapshot_dir = pathlib.Path(
            huggingface_hub.snapshot_download(repo_id=checkpoint,
                                              repo_type="model"))
        candidates = candidate_checkpoint_files(snapshot_dir)
        if candidates:
            return str(candidates[0])
        raise FileNotFoundError(
            f"No checkpoint file found in HF snapshot for {checkpoint}")

    return checkpoint


def default_intrinsics(width, height):
    """Returns a pinhole with focal max(width, height), centered principal."""
    import torch  # pylint: disable=import-outside-toplevel

    focal = float(max(width, height))
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics[0, 0] = focal
    intrinsics[1, 1] = focal
    intrinsics[0, 2] = (width - 1) / 2.0
    intrinsics[1, 2] = (height - 1) / 2.0
    return intrinsics


def preprocess_dust3r_family(image_paths,
                             *,
                             args,
                             device=None,
                             norm_type="dust3r",
                             **unused):
    """Builds the view dicts of DUSt3R, MASt3R and Pow3R.

    The views match `mapanything.utils.image.preprocess_inputs` (img,
    data_norm_type, true_shape, label, instance, idx, camera_intrinsics,
    camera_pose, depthmap) but come from a per-image pad into the 16 px
    patch grid. `preprocess_inputs` would crop every image to one target
    size computed from the batch's average aspect ratio, losing real,
    batch-dependent content. These wrappers only ever get plain images, so
    the pad needs no prior to follow it.

    Args:
        image_paths: Source images.
        args: The parsed options.
        device: Unused; the views stay on the CPU.
        norm_type: UniCeption normalization of the images.
        **unused: Further `utils.prepare_model_input` arguments.

    Returns:
        A (views, transforms) tuple.
    """
    del device, unused  # The views are built on the CPU.
    import torch  # pylint: disable=import-outside-toplevel

    # 512 is the only grid of the released checkpoints; any other
    # --image_load_resolution falls back to it.
    size = 512
    # exif_transpose=True as mapanything.utils.image.preprocess_inputs.
    images, transforms = utils.preprocess_images(image_paths,
                                                 method="vggt_omega",
                                                 size=size,
                                                 mode="pad",
                                                 exif_transpose=True)
    # The "dust3r" normalization is ToTensor (to [0, 1]) and
    # Normalize(0.5, 0.5); the images are already in [0, 1].
    images = images * 2.0 - 1.0
    views = []
    for index, image_path in enumerate(image_paths):
        frame = images[index:index + 1]
        height, width = frame.shape[-2:]
        views.append({
            "img": frame,
            "data_norm_type": [norm_type],
            "true_shape": np.int32([[height, width]]),
            "label": [pathlib.Path(args.data_path).name],
            "instance": [pathlib.Path(image_path).name],
            "idx": index,
            "camera_intrinsics": default_intrinsics(width, height)[None],
            "camera_pose": torch.eye(4, dtype=torch.float32)[None],
            "depthmap": torch.zeros((1, height, width, 1), dtype=torch.float32),
        })
    return views, transforms


utils.register_preprocessor("DUST3R", preprocess_dust3r_family)
utils.register_preprocessor("MAST3R", preprocess_dust3r_family)
utils.register_preprocessor("POW3R", preprocess_dust3r_family)


def move_views_to_device(views, device):
    """Moves the tensors of the views to `device`, in place.

    Args:
        views: The view dicts.
        device: The device.

    Returns:
        `views`.
    """
    ignore = {"label", "instance", "data_norm_type", "idx", "true_shape"}
    for view in views:
        for key, value in list(view.items()):
            if key in ignore:
                continue
            if hasattr(value, "to"):
                view[key] = value.to(device)
    return views


def quats_trans_to_c2w(quats, trans):
    """Returns camera-to-world matrices from xyzw quaternions and centers."""
    quat_array = np.asarray(quats, dtype=np.float32)
    trans_array = np.asarray(trans, dtype=np.float32)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None],
                    quat_array.shape[0],
                    axis=0)
    c2w[:, :3, :3] = utils.quaternion_xyzw_to_matrix(quat_array)
    c2w[:, :3, 3] = trans_array
    return c2w


def infer_intrinsics(outputs, height, width):
    """Returns per-view intrinsics from the predicted ray directions.

    The focal lengths come from the ray step next to the image center; a
    view without ray directions gets `default_intrinsics`.

    Args:
        outputs: Per-view output dicts of the wrapper.
        height: Height of the depth maps.
        width: Width of the depth maps.

    Returns:
        The (N, 3, 3) intrinsics.
    """
    intrinsics = []
    for output in outputs:
        ray_dirs = output.get("ray_directions")
        if ray_dirs is None:
            intrinsics.append(utils.as_numpy(default_intrinsics(width, height)))
            continue
        rays = utils.as_numpy(ray_dirs)[0]
        center = rays[height // 2, width // 2]
        fx = abs(center[2]) / max(
            abs(rays[height // 2,
                     min(width - 1, width // 2 + 1), 0] - center[0]), 1e-6)
        fy = abs(center[2]) / max(
            abs(rays[min(height - 1, height // 2 + 1), width // 2, 1] -
                center[1]), 1e-6)
        matrix = np.eye(3, dtype=np.float32)
        matrix[0, 0] = fx
        matrix[1, 1] = fy
        matrix[0, 2] = (width - 1) / 2.0
        matrix[1, 2] = (height - 1) / 2.0
        intrinsics.append(matrix)
    return np.stack(intrinsics, axis=0)


def _confidence_maps(outputs, depths):
    """Returns one confidence map per view, broadcast to the depth grid."""
    maps = []
    for index, output in enumerate(outputs):
        conf = utils.as_numpy(output.get("conf", np.ones_like(
            depths[index]))).squeeze()
        if conf.shape == depths[index].shape:
            maps.append(conf)
        elif conf.ndim == 1 and conf.shape[0] == depths.shape[1]:
            maps.append(np.broadcast_to(conf[:, None], depths[index].shape))
        elif conf.ndim == 1 and conf.shape[0] == depths.shape[2]:
            maps.append(np.broadcast_to(conf[None, :], depths[index].shape))
        else:
            maps.append(np.ones_like(depths[index], dtype=np.float32))
    return maps


def outputs_to_standard(outputs, views):
    """Converts the wrapper's per-view outputs to the benchmark's arrays.

    Args:
        outputs: Per-view output dicts of the wrapper.
        views: The input views (their images give the point colors).

    Returns:
        A (depths, points, colors, confidence, extrinsics_w2c, intrinsics)
        tuple of numpy arrays; confidence is None when the model has no
        confidence at all (point sampling then treats it as all ones).
    """
    # pylint: disable-next=import-outside-toplevel
    from mapanything.utils import image as mapanything_image

    points = utils.normalize_points(
        [utils.as_numpy(output["pts3d"])[0] for output in outputs])
    if "pts3d_cam" in outputs[0]:
        depths = utils.normalize_depths([
            utils.as_numpy(output["pts3d_cam"])[0, ..., 2] for output in outputs
        ])
    else:
        depths = utils.normalize_depths([
            utils.as_numpy(output["depth_along_ray"])[0, ..., 0]
            for output in outputs
        ])
    confidence_maps = _confidence_maps(outputs, depths)
    confidence = None
    if any("conf" in output for output in outputs):
        confidence = utils.normalize_confidence(confidence_maps, depths.shape)
    colors = []
    for view in views:
        image = mapanything_image.rgb(view["img"][0].detach().cpu(),
                                      view["data_norm_type"][0])
        colors.append((image.clip(0.0, 1.0) * 255).astype(np.uint8))
    colors = np.stack(colors, axis=0)
    c2w = quats_trans_to_c2w(
        [utils.as_numpy(output["cam_quats"])[0] for output in outputs],
        [utils.as_numpy(output["cam_trans"])[0] for output in outputs],
    )
    extrinsics_w2c = utils.c2w_to_w2c(c2w)
    intrinsics = infer_intrinsics(outputs, depths.shape[1], depths.shape[2])
    return depths, points, colors, confidence, extrinsics_w2c, intrinsics


class MapAnythingWrapperModel(base.BaseModel):
    """A method run through one of MapAnything's external-model wrappers.

    Attributes:
        wrapper_module: The MapAnything module of the wrapper.
        wrapper_class: The wrapper class in it.
        wrapper_kwargs: Keyword arguments of the wrapper.
    """

    wrapper_module: ClassVar[str] = ""
    wrapper_class: ClassVar[str] = ""
    wrapper_kwargs: ClassVar[dict[str, Any]] = {}
    # The images are always padded into the 512 grid of the released
    # checkpoints (preprocess_dust3r_family).
    uses_resolution = False
    uses_image_load_resolution = False

    def __init__(self, args):
        super().__init__(args)
        self.checkpoint_path = None

    @classmethod
    def add_arguments(cls, parser):
        """Adds --mapanything_code_path, the checkout of the wrappers.

        Args:
            parser: The parser.
        """
        parser.add_argument(
            "--mapanything_code_path",
            type=pathlib.Path,
            default=default_mapanything_code_path(),
            help=("MapAnything checkout with the wrapper (default: "
                  "$MAPANYTHING_CODE_PATH, else "
                  "$METHOD_CODE_ROOT/map-anything)."),
        )

    @property
    def checkpoint(self):
        """The resolved checkpoint file, written to metrics.json."""
        return self.checkpoint_path

    def add_code_path(self):
        """Puts MapAnything and the method's checkout on sys.path."""
        add_optional_path(self.args.mapanything_code_path, ("mapanything",))
        add_optional_path(self.args.code_path, self.code_packages)
        add_nested_method_paths(self.args.code_path)

    def select_device(self):
        import torch  # pylint: disable=import-outside-toplevel

        return "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        """Runs the wrapper on the scene and writes its artifacts.

        Raises:
            ValueError: No images were found, no checkpoint was given, or
                --depth_only is set without --export_depth.
        """
        args = self.args
        self.images_dir = utils.canonical_image_root(args.data_path)
        self.image_paths = self.select_images()
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")
        if args.dry_run:
            utils.print_dry_run(args, self.method_label, len(self.image_paths))
            return
        if not args.ckpt_path:
            raise ValueError(
                f"{self.name} requires --ckpt_path or {self.ckpt_env}.")
        self.checkpoint_path = resolve_checkpoint_arg(args.ckpt_path)

        self.add_code_path()
        self.device = self.select_device()
        # The legacy "inference time" of metrics.json includes the load.
        self.started_at = time.time()
        try:
            self.model = self.load(self.device)
            utils.mark_model_loaded()
            if args.depth_only:
                if not args.export_depth:
                    raise ValueError("--depth_only requires --export_depth")
                self.run_depth_only()
            else:
                self.run_full()
        finally:
            self.remove_scratch()

    def make_wrapper_kwargs(self):
        """Returns the wrapper's keyword arguments for this run."""
        return dict(self.wrapper_kwargs)

    def remove_scratch(self):
        """Removes what `make_wrapper_kwargs` created. Default: nothing."""

    def load(self, device):
        wrapper_cls = import_symbol(self.wrapper_module, self.wrapper_class)
        patch_dust3r_global_aligner(self.wrapper_module)
        capture_aligned_scene(self.wrapper_module)
        kwargs = self.make_wrapper_kwargs()
        wrapper = wrapper_cls(name=self.name,
                              ckpt_path=self.checkpoint_path,
                              **kwargs)
        return wrapper.to(device).eval()

    def preprocess(self, image_paths, batch_index=None):
        # A depth-only batch of one image runs as the pair (image, image).
        duplicate_pair = (batch_index is not None and len(image_paths) == 1 and
                          self.wrapper_class in PAIRWISE_WRAPPERS)
        view_paths = image_paths
        if duplicate_pair:
            view_paths = image_paths * 2
        views, transforms = preprocess_dust3r_family(view_paths,
                                                     args=self.args,
                                                     device=self.device)
        views = move_views_to_device(views, self.device)
        if duplicate_pair:
            transforms = transforms[:1]
        return base.ModelInput(image_paths,
                               views,
                               transforms,
                               batch_index,
                               extra={"duplicate_pair": duplicate_pair})

    def forward(self, inputs):
        import torch  # pylint: disable=import-outside-toplevel

        views = inputs.data
        with torch.no_grad():
            outputs = self.model(views)
        if inputs.extra["duplicate_pair"]:
            outputs = outputs[:1]
            views = views[:1]
        attach_aligned_confidence(outputs)
        depths, points, colors, confidence, extrinsics_w2c, intrinsics = (
            outputs_to_standard(outputs, views))
        return base.Prediction(
            depth=depths,
            confidence=confidence,
            extrinsics_w2c=extrinsics_w2c,
            intrinsics=intrinsics,
            points=points,
            colors=colors,
        )

    def metrics_extra(self, depth_only, images_per_run=None):
        extra = {"track": self.args.track, "adapter": self.wrapper_module}
        if depth_only:
            extra["max_images_per_run"] = images_per_run
            extra["Peak GPU Memory"] = base.peak_gpu_memory_mib()
        else:
            extra["selected_images"] = len(self.image_paths)
        return extra


@registry.register_model
class DUSt3R(MapAnythingWrapperModel):
    """DUSt3R with bundle adjustment (MapAnything's DUSt3RBAWrapper)."""

    name = "DUST3R"
    description = "Run DUST3R and export benchmark artifacts."
    code_packages = ("dust3r",)
    ckpt_env = "DUST3R_CKPT_PATH"
    code_path_env = "DUST3R_CODE_PATH"
    wrapper_module = "mapanything.models.external.dust3r"
    wrapper_class = "DUSt3RBAWrapper"
    wrapper_kwargs = {
        "scene_graph": "complete",
        "inference_batch_size": 8,
        "global_optim_niter": 120,
    }


@registry.register_model
class MASt3R(MapAnythingWrapperModel):
    """MASt3R with sparse global alignment (MASt3RSGAWrapper).

    The alignment caches pair features on disk, in a temporary directory
    under $BENCHMARK_TMP_ROOT that is removed after the run. Depth-only
    batches hold at most $MAST3R_DEPTH_BATCH_SIZE images (default 16, 0 for
    no limit).
    """

    name = "MAST3R"
    description = "Run MAST3R and export benchmark artifacts."
    code_packages = ("mast3r", "dust3r")
    ckpt_env = "MAST3R_CKPT_PATH"
    code_path_env = "MAST3R_CODE_PATH"
    wrapper_module = "mapanything.models.external.mast3r"
    wrapper_class = "MASt3RSGAWrapper"
    wrapper_kwargs = {
        "scene_graph": "complete",
        "sparse_ga_niter1": 120,
        "sparse_ga_niter2": 120,
    }

    def __init__(self, args):
        super().__init__(args)
        self.cache_dir = None

    def make_wrapper_kwargs(self):
        kwargs = super().make_wrapper_kwargs()
        self.cache_dir = pathlib.Path(
            tempfile.mkdtemp(prefix="mast3r_sga_", dir=_mast3r_cache_root()))
        kwargs.setdefault("cache_dir", str(self.cache_dir))
        return kwargs

    def remove_scratch(self):
        if self.cache_dir is not None:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def images_per_run(self):
        images_per_run = super().images_per_run()
        limit = int(os.environ.get("MAST3R_DEPTH_BATCH_SIZE", "16"))
        if limit > 0:
            images_per_run = min(images_per_run, limit)
        return images_per_run


def _scratch_scope():
    """Returns the user / job path suffix of the scratch directories."""
    user = os.environ.get("USER") or getpass.getuser()
    job = os.environ.get("SLURM_JOB_ID", os.environ.get("RUN_ID", "manual"))
    return pathlib.Path(user) / job


def _mast3r_cache_root():
    """Returns (and creates) the parent of MASt3R's cache directory.

    $BENCHMARK_TMP_ROOT, by default /tmp/<user>/feedforward_benchmark/<job>.
    When that disk is full, the shared benchmark scratch
    ($BENCHMARK_AUX_ROOT/_scratch/<user>/<job>), never the home folder.

    Raises:
        OSError: The directory cannot be created for another reason.
    """
    default_root = (pathlib.Path("/tmp") / _scratch_scope().parts[0] /
                    "feedforward_benchmark" / _scratch_scope().parts[1])
    cache_root = pathlib.Path(
        os.environ.get("BENCHMARK_TMP_ROOT", str(default_root))).expanduser()
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        if error.errno != errno.ENOSPC:
            raise
        aux_root = os.environ.get("BENCHMARK_AUX_ROOT",
                                  f"{spatialarena_root()}/benchmarks")
        cache_root = pathlib.Path(aux_root) / "_scratch" / _scratch_scope()
        cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


@registry.register_model
class Pow3R(MapAnythingWrapperModel):
    """Pow3R with bundle adjustment, without geometric inputs."""

    name = "POW3R"
    description = "Run POW3R and export benchmark artifacts."
    code_packages = ("pow3r", "dust3r")
    ckpt_env = "POW3R_CKPT_PATH"
    code_path_env = "POW3R_CODE_PATH"
    wrapper_module = "mapanything.models.external.pow3r"
    wrapper_class = "Pow3RBAWrapper"
    wrapper_kwargs = {
        "geometric_input_config": {
            "overall_prob": 0.0,
            "ray_dirs_prob": 0.0,
            "depth_prob": 0.0,
            "cam_prob": 0.0,
        },
        "scene_graph": "complete",
        "inference_batch_size": 4,
        "global_optim_niter": 120,
    }
