# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import glob
import os
import sys
import time
import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from scripts.inference.adapter_utils import (
    save_prediction_map,
    export_confidence_predictions,
    export_depths_for_policy,
    mark_model_loaded,
    plan_spatial_transform,
    prepare_model_input,
    record_model_input,
    register_preprocessor,
    runtime_metrics,
    canonical_image_root,
    fold_single_view_outputs,
    resolve_camera_conditioned_inputs,
    resolve_capped_image_indices,
    resolve_colmap_prior_path,
    restore_camera_export,
    restrict_to_image_subset,
    single_view_mode,
    spatial_policy_metrics,
    validate_shared_spatial_policy_depth_only,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Run Depth Anything 3 and export benchmark prediction artifacts")
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Dataset root containing images/ directory",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="./colmap_export_custom",
        help="Output directory (will create sparse/ with COLMAP files)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="depth-anything/DA3-LARGE-1.1",
        help="Model checkpoint file path",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="DA3",
        help="Method label written to dry-run output and metrics.",
    )
    parser.add_argument(
        "--da3_code_path",
        type=Path,
        default=Path(os.environ["DA3_CODE_DIR"])
        if "DA3_CODE_DIR" in os.environ else None,
        help="Optional path containing the depth_anything_3 package.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Intervals of frames to sample",
    )
    parser.add_argument(
        "--max_num_image_per_batch",
        type=int,
        default=0,
        help="Maximum images per model call when COLMAP camera priors are used.",
    )
    parser.add_argument(
        "--max_images_per_run",
        type=int,
        default=0,
        help="Maximum images per model call for depth-only full evaluation.",
    )
    parser.add_argument(
        "--max_input_frames",
        type=int,
        default=int(os.environ.get("BENCHMARK_MAX_INPUT_FRAMES", "0") or 0),
        help="Maximum sampled frames per scene before batching.",
    )
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=3.0,
        help="Depth confidence threshold to filter low-confidence depth",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=1_000_000,
        help="Max number of 3D points to keep when exporting",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=480,
        help="Max number of 3D points to keep when exporting",
    )
    parser.add_argument(
        "--align_scale_shift",
        action="store_true",
        help="Align scale and shift for per-image depth",
    )
    parser.add_argument(
        "--export_depth",
        action="store_true",
        help="Export depth to output path",
    )
    parser.add_argument(
        "--depth_only",
        action="store_true",
        help=
        "Only export depth predictions; skip dense point cloud and COLMAP artifacts.",
    )
    parser.add_argument(
        "--export_colmap",
        action="store_true",
        help="Export predicted extrinsics & intrinsics to colmap format",
    )
    parser.add_argument(
        "--load_from_colmap",
        action="store_true",
        help="Load extrinsics/intrinsics from COLMAP",
    )
    parser.add_argument(
        "--colmap_path",
        type=str,
        default="./sparse/0",
        help="Root directory containing colmap output",
    )
    parser.add_argument(
        "--colmap_model_subdir",
        type=str,
        default="0",
        help="Subdirectory under output_path/sparse for COLMAP files",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report planned inputs and outputs without loading the model",
    )
    parser.add_argument(
        "--spatial_policy",
        choices=("legacy", "shared"),
        default="legacy",
        help=
        ("Temporary port of PR #21 (eval_refactor_hsy): 'shared' routes depth-only inference "
         "through adapter_utils.plan_spatial_transform's shared da3 policy instead of DA3's own "
         "process_res_method, so depth predictions line up with the benchmark's shared GT-space "
         "evaluation grid. Currently only supported for --depth_only without --export_colmap."
        ),
    )
    args = parser.parse_args()
    try:
        validate_shared_spatial_policy_depth_only(
            is_shared=args.spatial_policy == "shared",
            depth_only=args.depth_only)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def find_image_paths(images_dir: Path) -> list[str]:
    image_extensions = [
        '*.jpg',
        '*.JPG',
        '*.jpeg',
        '*.JPEG',
        '*.png',
        '*.PNG',
        '*.gif',
        '*.GIF',
        '*.bmp',
        '*.BMP',
        '*.tiff',
        '*.TIFF',
        '*.webp',
        '*.WEBP',
    ]
    image_path_list = []
    for extension in image_extensions:
        pattern = os.path.join(str(images_dir), "**", extension)
        image_path_list.extend(glob.glob(pattern, recursive=True))
    return sorted(image_path_list)


def load_prior_model(args: argparse.Namespace) -> SimpleNamespace:
    """The camera-prior COLMAP model (cameras, images, points3D) that ``--load_from_colmap`` conditions on: the same
    directory ``resolve_camera_conditioned_inputs`` reads the priors from."""
    from spatialarena.geometry.read_write_model import read_model  # pylint: disable=import-outside-toplevel

    if args.colmap_path is not None:
        sparse_dir = Path(args.colmap_path)
    else:
        sparse_dir = resolve_colmap_prior_path(
            args.data_path, colmap_model_subdir=args.colmap_model_subdir)
    if sparse_dir is None:
        raise FileNotFoundError(f"No COLMAP prior model for {args.data_path}")
    cameras, images, points3d = read_model(str(sparse_dir))
    return SimpleNamespace(cameras=cameras, images=images, points3D=points3d)


def planned_layout(args: argparse.Namespace, num_images: int) -> dict:
    return {
        "method":
            args.method_name,
        "data_path":
            str(args.data_path),
        "output_path":
            str(args.output_path),
        "num_images":
            num_images,
        "metrics":
            str(args.output_path / "metrics.json"),
        "dense_points":
            None if args.depth_only else str(args.output_path /
                                             "points3D_dense.ply"),
        "depths":
            str(args.output_path / "depths") if args.export_depth else None,
        "sparse":
            (str(args.output_path / "sparse" /
                 args.colmap_model_subdir) if args.export_colmap else None),
    }


def add_da3_code_path(path: Path | None) -> None:
    if path is None:
        return
    code_path = Path(path).expanduser().resolve()
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))


def export_depth_predictions(output_path: Path,
                             image_paths: list[str],
                             depths: np.ndarray,
                             confidence: np.ndarray | None = None) -> None:
    """Export one float32 depth map per frame for benchmark evaluation (and its float16 confidence when given), as
    compressed .npz."""

    depth_dir = output_path / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)
    for image_path, depth in zip(image_paths, depths):
        depth = depth.astype(np.float32, copy=False)
        depth_stem = Path(image_path).stem
        save_prediction_map(depth_dir, depth_stem, depth)
    if confidence is not None:
        export_confidence_predictions(output_path, image_paths, confidence,
                                      np.shape(depths)[:3])


def center_crop_spatial(array: np.ndarray, target_h: int,
                        target_w: int) -> np.ndarray:
    """Center-crop a batched ``N,H,W[,C]`` array to a common spatial grid."""

    if array.ndim < 3:
        return array
    height, width = array.shape[1], array.shape[2]
    if height == target_h and width == target_w:
        return array
    if target_h > height or target_w > width:
        raise ValueError(
            f"Cannot crop spatial shape {(height, width)} to larger target {(target_h, target_w)}"
        )
    y0 = (height - target_h) // 2
    x0 = (width - target_w) // 2
    return array[:, y0:y0 + target_h, x0:x0 + target_w, ...]


def concatenate_prediction_batches(prediction: SimpleNamespace) -> None:
    """Concatenate DA3 batches whose internal crop sizes differ by batch."""

    dense_batches = prediction.processed_images + prediction.depth + prediction.conf
    target_h = min(array.shape[1] for array in dense_batches if array.ndim >= 3)
    target_w = min(array.shape[2] for array in dense_batches if array.ndim >= 3)
    prediction.processed_images = np.concatenate(
        [
            center_crop_spatial(array, target_h, target_w)
            for array in prediction.processed_images
        ],
        axis=0,
    )
    prediction.depth = np.concatenate(
        [
            center_crop_spatial(array, target_h, target_w)
            for array in prediction.depth
        ],
        axis=0,
    )
    prediction.conf = np.concatenate(
        [
            center_crop_spatial(array, target_h, target_w)
            for array in prediction.conf
        ],
        axis=0,
    )
    prediction.extrinsics = np.concatenate(prediction.extrinsics, axis=0)
    prediction.intrinsics = np.concatenate(prediction.intrinsics, axis=0)


def run_depth_inference_with_camera_fallback(
    model,
    image_paths: list[str],
    extrinsics: np.ndarray | None,
    intrinsics: np.ndarray | None,
    process_res: int,
):
    try:
        return model.inference(
            image_paths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=process_res,
        )
    except Exception as exc:
        message = str(exc)
        degenerate_alignment = ("Degenerate covariance rank" in message or
                                "Umeyama alignment is not possible" in message)
        if extrinsics is None or not degenerate_alignment:
            raise
        print(
            "WARNING: DA3 camera-prior alignment was degenerate; "
            "rerunning this depth batch without camera priors.",
            file=sys.stderr,
        )
        return model.inference(
            image_paths,
            extrinsics=None,
            intrinsics=None,
            process_res=process_res,
        )


def preprocess_da3(image_paths: list[str],
                   *,
                   args: argparse.Namespace,
                   device=None,
                   **_unused):
    """DA3's preprocess_<method>(), registered below for prepare_model_input("DA3", ...).

    DA3's own model.inference() does the actual resize internally (its official
    upper_bound_resize, forced via the inference_recording_input_size wrapper when
    --spatial_policy shared is set) -- there is no separate tensor to build here, so this only
    computes the matching SpatialTransform for restoration, trusting that real resize rather than
    reimplementing it (the same "trust the vendored resize" pattern as iggt.py).

    --spatial_policy legacy (default): returns (image_paths, None) unchanged.
    --spatial_policy shared: returns (image_paths, transforms).
    """
    if args.spatial_policy != "shared":
        return image_paths, None
    sizes = [Image.open(path).size for path in image_paths]
    if len(set(sizes)) != 1:
        raise ValueError(
            "Shared DA3 mapping rejects mixed shapes within one depth-only batch"
        )
    transforms = [
        plan_spatial_transform(size, method="da3", size=args.resolution)
        for size in sizes
    ]
    return image_paths, transforms


register_preprocessor("DA3", preprocess_da3)


def is_nested_da3(model) -> bool:
    """Whether ``model`` is the nested (metric-aligned) DA3, which fits one depth scale and one sky depth over the
    whole input batch, so batched samples would not be independent."""
    return type(getattr(model, "model",
                        None)).__name__ == "NestedDepthAnything3Net"


def run_single_view_inference(model, image_paths: list[str], process_res: int):
    """DA3 prediction for ``image_paths`` as independent single-view samples: DA3's own pre- and post-processing,
    with the network run on [B, 1, 3, H, W] instead of one B-view sequence [1, B, 3, H, W]. The nested model runs
    one image per call (see is_nested_da3) and its predictions are concatenated."""
    if is_nested_da3(model):
        predictions = [
            model.inference([path], process_res=process_res)
            for path in image_paths
        ]
        merged = predictions[0]
        for field in ("depth", "conf", "extrinsics", "intrinsics",
                      "processed_images"):
            values = [
                getattr(prediction, field, None) for prediction in predictions
            ]
            # DA3-NGL fills all of these for one image (checked 2026-10-21); a field it leaves out stays None.
            if all(value is not None for value in values):
                setattr(merged, field, np.concatenate(values, axis=0))
            elif any(value is not None for value in values):
                raise ValueError(
                    f"DA3 returned {field} for only some of the {len(values)} images"
                )
        return merged

    sequence_forward = model._run_model_forward  # pylint: disable=protected-access

    def independent_forward(imgs, ex_t, in_t, *forward_args, **forward_kwargs):
        num_samples = imgs.shape[1]
        output = sequence_forward(imgs.transpose(0, 1), ex_t, in_t,
                                  *forward_args, **forward_kwargs)
        return fold_single_view_outputs(output, num_samples)

    model._run_model_forward = independent_forward  # pylint: disable=protected-access
    try:
        return model.inference(image_paths, process_res=process_res)
    finally:
        del model._run_model_forward  # back to the class method


def inference():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_dir = canonical_image_root(args.data_path)
    if not images_dir.exists():
        print(f"❌ images directory not found: {images_dir}")
        return

    extrinsics_prior, intrinsics_prior = None, None

    if args.load_from_colmap:
        resolved = resolve_camera_conditioned_inputs(
            args.data_path,
            colmap_path=args.colmap_path,
            colmap_model_subdir=args.colmap_model_subdir,
        )
        image_path_list = resolved.image_paths
        extrinsics_prior = np.linalg.inv(resolved.poses)
        intrinsics_prior = resolved.intrinsics
    else:
        image_path_list = find_image_paths(images_dir)

    if len(image_path_list) == 0:
        print(f"❌ No images found in {images_dir}")
        return

    # Route through the same shared selection adapter_utils.select_interval
    # uses for every other method (interval + DA3-BENCH-matching
    # BENCHMARK_MAX_INPUT_FRAMES cap), instead of da3.py's own separate
    # reimplementation (previously a different RNG: np.random.default_rng
    # vs random.Random) -- kept index-based here (rather than calling
    # select_interval directly) so extrinsics_prior/intrinsics_prior stay in
    # lockstep with image_path_list. --max_input_frames still works as a CLI
    # override (its default already mirrors BENCHMARK_MAX_INPUT_FRAMES); the
    # separate --max_images_per_run/--max_num_image_per_batch remain pure
    # per-batch sizes, applied later, not an additional total-frame cap here.
    os.environ["BENCHMARK_MAX_INPUT_FRAMES"] = str(int(args.max_input_frames))
    indices = resolve_capped_image_indices(len(image_path_list),
                                           int(args.interval))
    # One same-size group of a single-view scene that mixes image sizes (BENCHMARK_IMAGE_SUBSET).
    subset = set(
        restrict_to_image_subset([image_path_list[index] for index in indices]))
    indices = [index for index in indices if image_path_list[index] in subset]
    image_path_list = [image_path_list[index] for index in indices]
    if extrinsics_prior is not None:
        extrinsics_prior = extrinsics_prior[indices, :, :]
    if intrinsics_prior is not None:
        intrinsics_prior = intrinsics_prior[indices, :, :]

    if args.dry_run:
        print(json.dumps(planned_layout(args, len(image_path_list)), indent=2))
        return

    args.output_path.mkdir(parents=True, exist_ok=True)

    add_da3_code_path(args.da3_code_path)

    from depth_anything_3.api import DepthAnything3
    from spatialarena.geometry.geometry import unproject_depth_map_to_point_map
    from spatialarena.utils.dense_points_sample_for_gs import PointSamplerFor3DGS
    from spatialarena.geometry.utils import randomly_limit_trues
    from spatialarena.utils.point_utils import export_ply

    model = DepthAnything3.from_pretrained(args.ckpt_path)
    model = model.to(device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def inference_recording_input_size(*args_, **kwargs_):
        # DA3 resizes inside model.inference; processed_images ([N, H, W, 3]) is what the network received.
        if args.spatial_policy == "shared":
            # Keep DA3's own official preprocessing as the sole transform so the
            # shared spatial_policy's restore_prediction mapping stays accurate.
            kwargs_["process_res_method"] = "upper_bound_resize"
        result = original_inference(*args_, **kwargs_)
        processed = getattr(result, "processed_images", None)
        if processed is not None and np.ndim(processed) == 4:
            record_model_input(np.shape(processed)[2], np.shape(processed)[1])
        return result

    original_inference = model.inference
    model.inference = inference_recording_input_size
    time_start = mark_model_loaded()

    process_res = args.resolution

    print(f'Processing {len(image_path_list)} images...')
    if args.depth_only:
        if not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")
        images_per_run = args.max_images_per_run if args.max_images_per_run > 0 else len(
            image_path_list)
        images_per_run = int(
            os.environ.get("DEPTH_INFERENCE_BATCH_SIZE", images_per_run))
        if images_per_run < 1:
            raise ValueError(
                "DEPTH_INFERENCE_BATCH_SIZE/--max_images_per_run must be >= 1")
        single_view = single_view_mode()
        if single_view and args.load_from_colmap:
            raise ValueError(
                "single-view (MonoDepth) inference takes no camera priors")
        pose_image_paths: list[str] = []
        pose_extrinsics: list[np.ndarray] = []
        pose_intrinsics: list[np.ndarray] = []
        pose_output_size: tuple[int, int] | None = None
        for start in range(0, len(image_path_list), images_per_run):
            batch_images = image_path_list[start:start + images_per_run]
            batch_extrinsics = (extrinsics_prior[start:start +
                                                 images_per_run, :, :]
                                if extrinsics_prior is not None else None)
            batch_intrinsics = (intrinsics_prior[start:start +
                                                 images_per_run, :, :]
                                if intrinsics_prior is not None else None)
            print(f"Depth-only batch {start // images_per_run + 1}: "
                  f"{len(batch_images)} images" +
                  (" (independent single views)" if single_view else ""))
            if single_view:
                batch_pred = run_single_view_inference(model, batch_images,
                                                       process_res)
            else:
                batch_pred = run_depth_inference_with_camera_fallback(
                    model,
                    batch_images,
                    batch_extrinsics,
                    batch_intrinsics,
                    process_res,
                )
            _paths, transforms = prepare_model_input("DA3",
                                                     batch_images,
                                                     args=args,
                                                     device=device)
            export_depths_for_policy(
                args.output_path,
                batch_images,
                batch_pred.depth,
                batch_pred.conf,
                transforms,
                legacy_export=lambda: export_depth_predictions(
                    args.output_path, batch_images, batch_pred.depth, batch_pred
                    .conf),
            )
            if args.export_colmap:
                # Mirrors the non-depth-only branch's preference (below) for
                # the given camera prior over the model's own re-estimate,
                # except when this batch fell back to no-prior inference
                # (see run_depth_inference_with_camera_fallback), in which
                # case the prior was never actually used.
                pose_image_paths.extend(batch_images)
                pose_extrinsics.append(batch_extrinsics if batch_extrinsics
                                       is not None else batch_pred.extrinsics)
                batch_intrinsics_final = (batch_intrinsics
                                          if batch_intrinsics is not None else
                                          batch_pred.intrinsics)
                if transforms is not None:
                    restored_intrinsic, width, height = restore_camera_export(
                        batch_intrinsics_final, transforms)
                    pose_intrinsics.append(restored_intrinsic)
                    pose_output_size = (width, height)
                else:
                    pose_intrinsics.append(batch_intrinsics_final)
                    if pose_output_size is None:
                        depth_array = np.asarray(batch_pred.depth)
                        pose_output_size = (depth_array.shape[2],
                                            depth_array.shape[1])
            del batch_pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if pose_extrinsics:
            from scripts.inference.adapter_utils import export_camera_pose_priors

            width, height = pose_output_size
            export_camera_pose_priors(
                args.output_path,
                images_dir,
                pose_image_paths,
                np.concatenate(pose_extrinsics, axis=0),
                np.concatenate(pose_intrinsics, axis=0),
                width=width,
                height=height,
                colmap_model_subdir=args.colmap_model_subdir,
            )

        render_time = time.time() - time_start
        runtime = runtime_metrics(time_start)
        render_max_mem = (torch.cuda.max_memory_allocated() /
                          (1024.0**2) if torch.cuda.is_available() else 0.0)
        metrics = {
            "method":
                args.method_name,
            "Scene":
                str(args.data_path),
            "Preprocessed Images":
                len(image_path_list),
            "inference time":
                render_time,
            **runtime,
            "Peak GPU Memory":
                render_max_mem,
            "depth_only":
                True,
            "max_images_per_run":
                images_per_run,
            "single_view":
                single_view,
            "nested_one_image_per_call":
                single_view and is_nested_da3(model),
            "camera_priors":
                bool(args.load_from_colmap),
            **spatial_policy_metrics(args.spatial_policy,
                                     is_shared=args.spatial_policy == "shared"),
        }
        metric_file = os.path.join(args.output_path, 'metrics.json')
        print(f'Saving metrics to {metric_file}')
        with open(metric_file, 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(metrics, indent=4))
        return

    if args.load_from_colmap:
        prediction = SimpleNamespace(
            processed_images=[],
            depth=[],
            conf=[],
            extrinsics=[],
            intrinsics=[],
        )
        images_per_batch = args.max_num_image_per_batch if args.max_num_image_per_batch > 0 else len(
            image_path_list)
        for i in range(0, len(image_path_list), images_per_batch):
            batch_images = image_path_list[i:i + images_per_batch]
            batch_extrinsics = extrinsics_prior[i:i + images_per_batch, :, :]
            batch_intrinsics = intrinsics_prior[i:i + images_per_batch, :, :]

            current_batch_size = len(batch_images)

            # Check if this is the last batch and it's too short
            if current_batch_size < images_per_batch and i + images_per_batch >= len(
                    image_path_list):
                print(
                    f"Last batch has only {current_batch_size} images, " +
                    f"padding with images from beginning to reach {images_per_batch}"
                )

                # Calculate how many more images we need
                needed = images_per_batch - current_batch_size

                # Pad with images from the beginning of the sequence
                padded_images = batch_images + image_path_list[:needed]
                padded_extrinsics = None
                padded_intrinsics = None

                if batch_extrinsics is not None:
                    padded_extrinsics = np.concatenate(
                        [batch_extrinsics, extrinsics_prior[:needed, :, :]],
                        axis=0)
                if batch_intrinsics is not None:
                    padded_intrinsics = np.concatenate(
                        [batch_intrinsics, intrinsics_prior[:needed, :, :]],
                        axis=0)

                batch_pred = model.inference(
                    padded_images,
                    extrinsics=padded_extrinsics,
                    intrinsics=padded_intrinsics,
                    process_res=process_res,
                )

                # Only keep the original batch size results (trim off the padded ones)
                batch_pred.processed_images = batch_pred.processed_images[:
                                                                          current_batch_size]
                batch_pred.depth = batch_pred.depth[:current_batch_size]
                batch_pred.conf = batch_pred.conf[:current_batch_size]
                batch_pred.extrinsics = batch_pred.extrinsics[:
                                                              current_batch_size, :, :]
                batch_pred.intrinsics = batch_pred.intrinsics[:
                                                              current_batch_size, :, :]
            else:
                # Normal batch processing
                batch_pred = model.inference(
                    batch_images,
                    extrinsics=batch_extrinsics,
                    intrinsics=batch_intrinsics,
                    process_res=process_res,
                )

            prediction.processed_images.append(batch_pred.processed_images)
            prediction.depth.append(batch_pred.depth)
            prediction.conf.append(batch_pred.conf)
            prediction.extrinsics.append(batch_pred.extrinsics)
            prediction.intrinsics.append(batch_pred.intrinsics)

        concatenate_prediction_batches(prediction)
    else:
        prediction = model.inference(
            image_path_list,
            extrinsics=extrinsics_prior,
            intrinsics=intrinsics_prior,
            process_res=process_res,
        )

    render_time = time.time() - time_start
    runtime = runtime_metrics(time_start)
    render_max_mem = (torch.cuda.max_memory_allocated() /
                      (1024.0**2) if torch.cuda.is_available() else 0.0)
    metrics = {
        "method": args.method_name,
        "Scene": str(args.data_path),
        "Preprocessed Images": len(image_path_list),
        "inference time": render_time,
        **runtime,
        "Peak GPU Memory": render_max_mem,
    }
    metric_file = os.path.join(args.output_path, 'metrics.json')
    json_obj = json.dumps(metrics, indent=4)
    print(f'Saving metrics to {metric_file}')
    with open(metric_file, 'w', encoding='utf-8') as json_file:
        json_file.write(json_obj)

    preprocessed_images = torch.from_numpy(
        prediction.processed_images)  # [N,H,W,3]
    points_rgb = preprocessed_images.cpu().numpy().astype(np.uint8)
    _, grid_h, grid_w, _ = preprocessed_images.shape
    depths = prediction.depth  # [N,H,W]
    confs = prediction.conf
    extrinsics = prediction.extrinsics if extrinsics_prior is None else extrinsics_prior
    intrinsics = prediction.intrinsics  # if intrinsics_prior is None else intrinsics_prior

    # Confidence filtering and random downsample
    conf_mask = confs >= args.depth_conf_thresh
    conf_mask = randomly_limit_trues(conf_mask, args.max_points)
    points_rgb = points_rgb[conf_mask]

    if args.load_from_colmap:
        prior_model = load_prior_model(args)
        print('Estimating global scale and shift...')
        # Collect depth predictions into dictionary
        depth_predictions = {}
        for i, img_path in enumerate(image_path_list):
            depth_predictions[os.path.basename(img_path)] = depths[i]

        # Estimate and apply global alignment
        if args.align_scale_shift:
            from spatialarena.utils.depth_scale_shift_estimator import (
                estimate_and_apply_local_alignment,)

            # aligned_depth_maps, alignment_info = estimate_and_apply_global_alignment(
            aligned_depth_maps, alignment_info = estimate_and_apply_local_alignment(
                cameras=prior_model.cameras,
                images=prior_model.images,
                points3d=prior_model.points3D,
                depth_predictions=depth_predictions,
                model_resolution=(grid_h, grid_w),
                outlier_rejection=True,
                n_jobs=-1,
            )
            aligned_depths = []
            for i, img_path in enumerate(image_path_list):
                depth = aligned_depth_maps[os.path.basename(img_path)]
                aligned_depths.append(depth)
            aligned_depths = np.stack(aligned_depths, axis=0)
        else:
            aligned_depths = depths

        aligned_points_3d = unproject_depth_map_to_point_map(
            aligned_depths, extrinsics, intrinsics)

        # Sample points using edge-aware strategy
        sampler = PointSamplerFor3DGS(
            max_points=args.max_points,
            edge_weight=3.0,
            corner_weight=5.0,
            use_confidence=True,
            confidence_threshold=args.depth_conf_thresh,
            use_redundancy_filter=True,
            voxel_size=0.01,
        )
        sampled_points, sampled_colors = sampler.sample_points_from_views(
            points_3d=aligned_points_3d,
            colors=preprocessed_images.cpu().numpy(),
            images=preprocessed_images.cpu().numpy(),
            confidence_maps=confs,
            masks=None,
        )

        colmap_points3d, colmap_points_rgb = [], []
        for _, point3D in prior_model.points3D.items():
            xyz, rgb = point3D.xyz, point3D.rgb
            colmap_points3d.append(xyz)
            colmap_points_rgb.append(rgb)
        if colmap_points3d:
            colmap_points3d = np.stack(colmap_points3d, axis=0)
            colmap_points_rgb = np.stack(colmap_points_rgb, axis=0)
            print(f'colmap_points3d shape: {colmap_points3d.shape}')
        else:
            colmap_points3d = np.empty((0, 3), dtype=np.float32)
            colmap_points_rgb = np.empty((0, 3), dtype=np.uint8)
            print(
                'colmap_points3d shape: (0, 3); sparse model has no 3D points')
        print(f'sampled_points shape: {sampled_points.shape}')

        # # Combine sampled dense points and colmap points for 3DGS initialization.
        # points_3d = np.concatenate([sampled_points, colmap_points3d], axis=0)
        # points_rgb = np.concatenate([sampled_colors, colmap_points_rgb], axis=0)
        points_3d = sampled_points
        points_rgb = sampled_colors
    else:
        # Back-project depth to 3D (camera/world coords as defined by util func)
        points_3d = unproject_depth_map_to_point_map(depths, extrinsics,
                                                     intrinsics)
        points_3d = points_3d[conf_mask]

    # Export .ply file for 3DGS initialization.
    gs_ply_path = os.path.join(args.output_path, "points3D_dense.ply")
    export_ply(gs_ply_path, points_3d, points_rgb)
    print(f'Number of 3D points for 3DGS: {points_3d.shape[0]}')

    if args.export_depth:
        print("Exporting predicted depth maps...")
        export_depth_predictions(args.output_path, image_path_list, depths,
                                 confs)

    # Export to colmap formats.
    if args.export_colmap:
        from spatialarena.geometry.utils import write_colmap_formats

        print(
            "Exporting predicted extrinsics + intrinsics + points to COLMAP format"
        )
        colmap_out_dir = os.path.join(args.output_path, 'sparse',
                                      args.colmap_model_subdir)
        os.makedirs(colmap_out_dir, exist_ok=True)
        write_colmap_formats(
            str(images_dir),
            extrinsics,
            intrinsics,
            image_path_list,
            points_3d,
            points_rgb,
            width=grid_w,
            height=grid_h,
            output_path=colmap_out_dir,
        )


if __name__ == "__main__":
    inference()
