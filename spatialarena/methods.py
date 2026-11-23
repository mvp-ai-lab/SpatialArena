# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Benchmark method and modality-track registry.

This is the single source of truth for supported method names, aliases,
adapter scripts, default checkpoint/code-path variables, and valid
input-modality tracks. Native adapters are implemented in
`scripts/inference`. Methods without native adapters can still be
benchmarked through `external_command_adapter.py` when a method-specific
command template is provided via environment variable.

Promoted from `scripts/test/benchmark_methods.py` so it is importable as
`spatialarena.methods` (e.g. by `BenchmarkEvaluator`).
`scripts/test/benchmark_methods.py` re-exports everything here and keeps
only its CLI wrapper, so the `--field ...` interface
`run_method_case_list.sh` already depends on is unchanged.

Typical use:

    spec = methods.resolve_method("vggt")
    methods.validate_track(spec.name, methods.TRACK_RGB_DEPTH)
"""

from __future__ import annotations

import dataclasses
import unicodedata
from typing import Iterable

# Imported by name because the METHOD_SPECS table interpolates it in many
# checkpoint paths.
from spatialarena.paths import SPATIALARENA_ROOT  # noqa: E402

TRACK_RGB = "rgb"
TRACK_RGB_DEPTH = "rgb_depth"
TRACK_RGB_CAMERAS = "rgb_cameras"
# GT depth with part of every map dropped, alone or with the GT cameras
# (spatialarena/depth_prior_masking.py).
TRACK_RGB_MASKED_DEPTH = "rgb_masked_depth"
TRACK_RGB_MASKED_DEPTH_CAMERAS = "rgb_masked_depth_cameras"
# Real sparse depth (KITTI LiDAR, COLMAP keypoints), alone or with the GT
# cameras (spatialarena/sparse_depth.py).
TRACK_RGB_SPARSE_DEPTH = "rgb_sparse_depth"
TRACK_RGB_SPARSE_DEPTH_CAMERAS = "rgb_sparse_depth_cameras"


@dataclasses.dataclass(frozen=True, kw_only=True)
class MethodSpec:
    """One benchmark method: its adapter, code, checkpoint and tracks.

    Attributes:
        name: Canonical method name.
        adapter: Adapter script under scripts/inference.
        code_env: Environment variable that points at the method code.
        code_default: Default method code path.
        ckpt_env: Environment variable that points at the checkpoint.
        ckpt_default: Default checkpoint path, URL or model id.
        native: Whether a native adapter exists.
        tracks: Input-modality tracks the method supports.
        aliases: Other accepted names.
        input_resolution: Model input resolution.
        image_load_resolution: Resolution images are loaded at.
        pixel_limit: Maximum pixel count per image; 0 means no limit.
        conf_thresh: Point confidence threshold.
        depth_conf_thresh: Depth confidence threshold.
        resolution_source: Where the resolution comes from.
        notes: Free-form notes.
        depth_output_space: Space of the predicted depth: "depth",
            "inverse_depth" or "log_depth".
        tasks: Tasks the method takes part in; empty means every task.
        point_grid_size: Query points per image axis of point-query VLMs.
    """

    name: str
    adapter: str
    code_env: str
    code_default: str
    ckpt_env: str = ""
    ckpt_default: str = ""
    native: bool = False
    tracks: tuple[str, ...] = (TRACK_RGB,)
    aliases: tuple[str, ...] = ()
    input_resolution: int = 518
    image_load_resolution: int = 518
    pixel_limit: int = 0
    conf_thresh: float = 0.0
    depth_conf_thresh: float = 0.0
    resolution_source: str = "repo_default"
    notes: str = ""
    depth_output_space: str = "depth"
    # Tasks the method takes part in; empty means every task. The long-sequence
    # SLAM systems are SLAM-only.
    tasks: tuple[str, ...] = ()
    # Point-query VLMs (scripts/inference/point_query_depth_adapter.py): query
    # points per image axis, so a dense map is interpolated from
    # point_grid_size**2 scalar answers. 0: not a point-query method (the
    # adapter's default, the image centre only, which yields a constant map).
    # The engine passes it as POINT_QUERY_DEPTH_GRID_SIZE.
    point_grid_size: int = 0

    @property
    def default_depth_protocol(self) -> str:
        """The depth alignment protocol in the model's native output space.

        Alignment runs in the space the model predicts in (depth, inverse
        depth or log depth); the metrics are always computed in depth space.

        Raises:
            ValueError: If `depth_output_space` is not a known space.
        """
        protocols = {
            "depth": "depth_huber_irls_gt",
            "inverse_depth": "inverse_huber_irls_gt",
            "log_depth": "log_huber_irls_gt"
        }
        if self.depth_output_space not in protocols:
            raise ValueError(
                f"Unsupported depth output space: {self.depth_output_space}")
        return protocols[self.depth_output_space]

    @property
    def external_command_env(self) -> str:
        """The environment variable holding the external command template."""
        return f"{self.name.replace('-', '_')}_COMMAND".upper()


METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        name="DEPTHCRAFTER",
        depth_output_space="inverse_depth",
        adapter="video_diffusion_depth.py",
        code_env="DEPTHCRAFTER_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/DepthCrafter",
        ckpt_env="DEPTHCRAFTER_CKPT_PATH",
        ckpt_default="tencent/DepthCrafter",
        native=True,
        input_resolution=1024,
        image_load_resolution=1024,
        aliases=("DEPTHCRAFT",),
        tasks=("VideoDepth",),
        resolution_source=(
            "Official long-edge cap 1024; dimensions rounded to 64, "
            "restored to source grid."),
        notes=("Official video diffusion pipeline; 5 denoising steps; "
               "RGB-only, no GT input."),
    ),
    MethodSpec(
        name="CHRONODEPTH",
        adapter="video_diffusion_depth.py",
        code_env="CHRONODEPTH_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/ChronoDepth",
        ckpt_env="CHRONODEPTH_CKPT_PATH",
        ckpt_default="jhshao/ChronoDepth-v1",
        native=True,
        input_resolution=1024,
        image_load_resolution=1024,
        aliases=("CHRONO_DEPTH",),
        tasks=("VideoDepth",),
        resolution_source=(
            "Official long-edge cap 1024; dimensions rounded to 64, "
            "restored to source grid."),
        notes=("Official video diffusion pipeline; 5 denoising steps; "
               "RGB-only, no GT input."),
    ),
    MethodSpec(
        name="DA3-L",
        adapter="da3.py",
        code_env="DA3_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/depth-anything-3/src",
        ckpt_env="DA3_L_CKPT_PATH",
        ckpt_default="depth-anything/DA3-LARGE-1.1",
        native=True,
        aliases=("DA3_L", "DEPTHANYTHING3-L", "DEPTHANYTHING3_L"),
        input_resolution=504,
        image_load_resolution=504,
        resolution_source="MapAnything benchmark profile reports DA3 at 504.",
        notes=("Depth Anything 3 Large checkpoint. code_default points at the "
               "official ByteDance-Seed/depth-anything-3 repo's src/ (not a "
               "third-party vendored copy) -- see "
               "METHOD_CODE_REPOS['depth-anything-3']."),
    ),
    MethodSpec(
        name="DA3-G",
        adapter="da3.py",
        code_env="DA3_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/depth-anything-3/src",
        ckpt_env="DA3_G_CKPT_PATH",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks/"
                      "_checkpoints/local/da3_giant"),
        native=True,
        aliases=("DA3", "DA3_G", "DEPTHANYTHING3", "DEPTHANYTHING3-G",
                 "DEPTHANYTHING3_G"),
        input_resolution=504,
        image_load_resolution=504,
        resolution_source="MapAnything benchmark profile reports DA3 at 504.",
        notes=("Depth Anything 3 Giant checkpoint. Backward-compatible alias: "
               "DA3."),
    ),
    MethodSpec(
        name="DA3-NGL",
        adapter="da3.py",
        code_env="DA3_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/depth-anything-3/src",
        ckpt_env="DA3_NGL_CKPT_PATH",
        ckpt_default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        native=True,
        tracks=(TRACK_RGB, TRACK_RGB_CAMERAS),
        aliases=(
            "DA3_NGL",
            "DA3-NESTED-GIANT-LARGE",
            "DA3_NESTED_GIANT_LARGE",
            "DEPTHANYTHING3-NESTED-GIANT-LARGE",
            "DEPTHANYTHING3_NESTED_GIANT_LARGE",
        ),
        input_resolution=504,
        image_load_resolution=504,
        resolution_source="MapAnything benchmark profile reports DA3 at 504.",
        notes=(
            "Depth Anything 3 Nested-Giant-Large; camera track passes COLMAP "
            "priors when available."),
    ),
    MethodSpec(
        name="VGGT",
        adapter="vggt.py",
        code_env="VGGT_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/RobustVGGT",
        ckpt_env="VGGT_CKPT_PATH",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks/"
                      "_checkpoints/local/vggt/vggt.pt"),
        native=True,
        input_resolution=518,
        image_load_resolution=518,
        resolution_source="VGGT model and official loader default to 518.",
    ),
    MethodSpec(
        name="AMB3R",
        adapter="amb3r.py",
        code_env="AMB3R_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/amb3r",
        ckpt_env="AMB3R_CKPT_PATH",
        ckpt_default=(
            f"{SPATIALARENA_ROOT}/benchmarks/_checkpoints/local/amb3r/amb3r.pt"
        ),
        native=True,
        aliases=("AMB3R-BASE",),
        input_resolution=518,
        image_load_resolution=518,
        resolution_source=(
            "AMB3R's own benchmark harness (benchmark/eval_mvrecon.py etc.) "
            "hardcodes (518, 392) width x height for the base model -- "
            "non-square, unlike input_resolution's single-int convention here; "
            "the adapter hardcodes both dimensions itself."),
        notes=(
            "Front-end reuses VGGT's own camera/depth/point heads (AMB3R's "
            "frontend.py imports "
            "vggt.utils.pose_enc.pose_encoding_to_extri_intri from its "
            "vendored thirdparty/vggt), refined by a voxel-based backend "
            "(torch_scatter + pytorch3d.ops.knn_points, both built from "
            "source, plus flash-attn -- required by thirdparty/ptv3's "
            "PointTransformerV3, not optional). Runs in the shared si_bench "
            "env, whose build script compiles those extensions from source "
            "against its torch with --no-cache-dir (a cached wheel built "
            "against another torch ABI was silently reused once) and leaves "
            "out AMB3R's requirements.txt xformers pin (it downgrades torch, "
            "and its import is dead code in the vendored thirdparty/vggt). "
            "AMB3R's requirements.txt also hard-pins huggingface-hub==0.28.1, "
            "too old for transformers>=4.57 (needed for the GenCeption raymap "
            "prompt script); the build script upgrades it afterward since "
            "AMB3R's own code only needs huggingface_hub>=0.22.0."),
    ),
    MethodSpec(
        name="PI3_X",
        adapter="pi3.py",
        code_env="PI3_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/Pi3",
        ckpt_env="PI3_CKPT_PATH",
        ckpt_default="yyfz233/Pi3X",
        native=True,
        tracks=(
            TRACK_RGB,
            TRACK_RGB_DEPTH,
            TRACK_RGB_CAMERAS,
            TRACK_RGB_MASKED_DEPTH,
            TRACK_RGB_MASKED_DEPTH_CAMERAS,
            TRACK_RGB_SPARSE_DEPTH,
            TRACK_RGB_SPARSE_DEPTH_CAMERAS,
        ),
        aliases=("PI3", "PI3-X", "PIX3", "PI3X"),
        input_resolution=512,
        image_load_resolution=512,
        pixel_limit=262144,
        conf_thresh=0.0,
        depth_conf_thresh=0.0,
        resolution_source=(
            "Pi3 evaluation branch uses load_img_size=512 for MonoDepth, "
            "VideoDepth, and relative pose; MVRecon is 518."),
        notes="Pi3/Pi3-X. Non-RGB tracks require method-side prior support.",
    ),
    MethodSpec(
        name="FASTVGGT",
        adapter="fastvggt.py",
        code_env="FASTVGGT_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/RobustVGGT",
        ckpt_env="FASTVGGT_CKPT_PATH",
        ckpt_default=(
            f"{SPATIALARENA_ROOT}/benchmarks/_checkpoints/local/fastvggt/"
            "model_tracker_fixed_e20.pt"),
        native=True,
        input_resolution=518,
        image_load_resolution=518,
        resolution_source="FastVGGT follows VGGT-compatible 518 preprocessing.",
    ),
    MethodSpec(
        name="COME",
        adapter="come.py",
        code_env="COME_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/CoMe",
        ckpt_env="COME_CKPT_PATH",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks/_checkpoints/local/come/"
                      "checkpoint.pth"),
        native=True,
        input_resolution=504,
        image_load_resolution=504,
        resolution_source=(
            "CoME adapter/default config uses a 504-compatible VGGT token "
            "grid."),
        notes=(
            "Confidence-distill checkpoint for the VGGT-fused config "
            "(config/model/co_me_fused_vggt_3x3.yaml's ckpt_path). Ships "
            "committed inside the CoMe repo itself (no separate network "
            "download), but is canonicalized to "
            "checkpoint_root/local/come/checkpoint.pth like every other "
            "local-file checkpoint -- download_all_supported_models() copies "
            "it there from the cloned CoMe checkout (see "
            "SHIPPED_WITH_CODE_CHECKPOINTS in spatialarena/download.py)."),
    ),
    MethodSpec(
        name="CUT3R",
        adapter="cut3r.py",
        code_env="CUT3R_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/CUT3R",
        ckpt_env="CUT3R_CKPT_PATH",
        ckpt_default="$METHOD_CODE_ROOT/CUT3R/src/cut3r_512_dpt_4_64.pth",
        native=True,
        input_resolution=512,
        image_load_resolution=512,
        resolution_source=("CUT3R released checkpoint name and "
                           "adapter default use 512."),
        notes=("The authors' final checkpoint, published via Google Drive "
               "(see GDRIVE_CHECKPOINT_URLS in spatialarena/download.py); "
               "download_all_supported_models() fetches it with gdown."),
    ),
    MethodSpec(
        name="FAST3R",
        adapter="fast3r.py",
        code_env="FAST3R_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/fast3r",
        ckpt_env="FAST3R_CKPT_PATH",
        ckpt_default="jedyang97/Fast3R_ViT_Large_512",
        native=True,
        input_resolution=512,
        image_load_resolution=512,
        resolution_source="Fast3R released checkpoint is ViT-Large-512.",
    ),
    MethodSpec(
        name="VGGT_OMEGA",
        adapter="vggt_omega.py",
        code_env="VGGT_OMEGA_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/vggt-omega",
        ckpt_env="VGGT_OMEGA_CKPT_PATH",
        ckpt_default=(
            f"{SPATIALARENA_ROOT}/benchmarks/_checkpoints/local/vggt_omega/"
            "vggt_omega_1b_512.pt"),
        native=True,
        aliases=("VGGT-OMEGA",),
        input_resolution=512,
        image_load_resolution=512,
        resolution_source=(
            "VGGT-Omega benchmark adapter/checkpoint uses 512-scale "
            "preprocessing."),
    ),
    MethodSpec(
        name="MAPANYTHING",
        adapter="mapanything.py",
        code_env="MAPANYTHING_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/map-anything",
        ckpt_env="MAPANYTHING_CKPT_PATH",
        ckpt_default="facebook/map-anything",
        native=True,
        tracks=(
            TRACK_RGB,
            TRACK_RGB_DEPTH,
            TRACK_RGB_CAMERAS,
            TRACK_RGB_MASKED_DEPTH,
            TRACK_RGB_MASKED_DEPTH_CAMERAS,
            TRACK_RGB_SPARSE_DEPTH,
            TRACK_RGB_SPARSE_DEPTH_CAMERAS,
        ),
        aliases=("MAP_ANYTHING", "MAP-ANYTHING"),
        input_resolution=518,
        image_load_resolution=518,
        resolution_source="MapAnything official benchmark profile uses 518.",
        notes=("Native adapter supports RGB, RGB+depth, RGB+cameras, the "
               "masked-depth and the sparse-depth tracks."),
    ),
    MethodSpec(
        name="OMNIVGGT",
        adapter="omnivggt.py",
        code_env="OMNIVGGT_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/OmniVGGT-official",
        ckpt_env="OMNIVGGT_CKPT_PATH",
        ckpt_default="Livioni/OmniVGGT",
        native=True,
        tracks=(
            TRACK_RGB,
            TRACK_RGB_DEPTH,
            TRACK_RGB_CAMERAS,
            TRACK_RGB_MASKED_DEPTH,
            TRACK_RGB_MASKED_DEPTH_CAMERAS,
            TRACK_RGB_SPARSE_DEPTH,
            TRACK_RGB_SPARSE_DEPTH_CAMERAS,
        ),
        aliases=("OMNI_VGGT", "OMNI-VGGT"),
        input_resolution=518,
        image_load_resolution=518,
        resolution_source=(
            "OmniVGGT README/default loader uses target_size=518."),
        notes=("Native adapter supports RGB, RGB+depth, RGB+cameras, the "
               "masked-depth and the sparse-depth tracks."),
    ),
    MethodSpec(
        name="DEJAVIEW",
        adapter="dejaview.py",
        code_env="DEJAVIEW_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/dvlt",
        ckpt_env="DEJAVIEW_CKPT_PATH",
        ckpt_default="nvidia/dvlt",
        native=True,
        aliases=("DEJA_VIEW", "DEJA-VIEW", "DVLT"),
        input_resolution=504,
        image_load_resolution=504,
        resolution_source=(
            "Déjà View README and adapter default instantiate DVLT at "
            "img_size=504."),
        notes="Native DVLT adapter using dvlt.model.dvlt.model.DVLT.",
    ),
    MethodSpec(
        name="IGGT",
        adapter="iggt.py",
        code_env="IGGT_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/IGGT_official",
        ckpt_env="IGGT_CKPT_PATH",
        ckpt_default="lifuguan/IGGT_official",
        native=True,
        input_resolution=504,
        image_load_resolution=504,
        resolution_source="IGGT adapter default uses 504.",
        notes=(
            "Native adapter for IGGT official checkpoint/code. Runs in the "
            "shared si_bench env, not a dedicated one -- IGGT_official's own "
            "requirements.txt is a full training-env freeze (apex, detectron2, "
            "RAPIDS cudf/cuml, a pytorch3d commit that conflicts with AMB3R's "
            "own build in si_bench), but the adapter's actual import chain "
            "(iggt.models.vggt -> heads/layers, plus the vendored sam2 "
            "submodule for SamProjector) only needs torch, torchvision, "
            "transformers (Mask2FormerConfig), huggingface_hub, and scipy -- "
            "all already present, verified via a real inference smoke test "
            "(2026-09-25)."),
    ),
    MethodSpec(
        name="IGGT4D",
        adapter="iggt4d.py",
        code_env="IGGT4D_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/IGGT4D",
        ckpt_env="IGGT4D_CKPT_PATH",
        ckpt_default=(
            f"{SPATIALARENA_ROOT}/benchmarks/_checkpoints/local/iggt4d/"
            "model.safetensors"),
        native=True,
        input_resolution=504,
        image_load_resolution=504,
        resolution_source=
        ("IGGT4D inference() default process_res=504 (upper_bound_resize), as "
         "in streaming_infer.py."),
        notes=(
            "HorizonRobotics/IGGT4D (Apache-2.0): streaming_infer.py / "
            "iggt4d.api.IGGT4D.inference, a Depth Anything 3-style API "
            "returning depth, depth confidence, world-to-camera extrinsics and "
            "intrinsics (plus instance masks, unused here). Checkpoint: "
            "Hugging Face HorizonRobotics/IGGT4D model.safetensors. Runs "
            "offline full attention over each run's frames; the instance head "
            "is skipped."),
    ),
    MethodSpec(
        name="POW3R",
        adapter="pow3r.py",
        code_env="POW3R_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/pow3r",
        ckpt_env="POW3R_CKPT_PATH",
        ckpt_default=
        ("https://download.europe.naverlabs.com/ComputerVision/Pow3R/Pow3R_ViTLarge_BaseDecoder_512_linear.pth"
        ),
        native=True,
        aliases=("POW3R",),
        input_resolution=512,
        image_load_resolution=512,
        resolution_source=(
            "Pow3R README/model table reports 512 training/evaluation "
            "resolution."),
        notes="Native adapter via MapAnything Pow3R wrapper.",
    ),
    MethodSpec(
        name="DUST3R",
        adapter="dust3r.py",
        code_env="DUST3R_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/dust3r",
        ckpt_env="DUST3R_CKPT_PATH",
        ckpt_default=
        ("https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        ),
        native=True,
        aliases=("DUSt3R",),
        input_resolution=512,
        image_load_resolution=512,
        resolution_source="DUSt3R released checkpoints/wrappers use 512.",
        notes=(
            "Native adapter via MapAnything DUSt3R bundle-adjustment wrapper."),
    ),
    MethodSpec(
        name="MAST3R",
        adapter="mast3r.py",
        code_env="MAST3R_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/mast3r",
        ckpt_env="MAST3R_CKPT_PATH",
        ckpt_default=(
            "https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"),
        native=True,
        aliases=("MASt3R",),
        input_resolution=512,
        image_load_resolution=512,
        resolution_source=("MASt3R README examples and released "
                           "checkpoint use 512."),
        notes=("Native adapter via MapAnything MASt3R sparse-global-alignment "
               "wrapper."),
    ),
    MethodSpec(
        name="ROBUSTVGGT",
        adapter="robustvggt.py",
        code_env="ROBUSTVGGT_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/RobustVGGT",
        ckpt_env="ROBUSTVGGT_CKPT_PATH",
        ckpt_default="facebook/VGGT-1B",
        native=True,
        aliases=("ROBUST_VGGT", "ROBUST-VGGT"),
        input_resolution=518,
        image_load_resolution=518,
        resolution_source=(
            "RobustVGGT keeps the VGGT-compatible 518 model grid."),
        notes="Native adapter using the RobustVGGT VGGT-compatible code path.",
    ),
    MethodSpec(
        name="DEPTHVLM",
        adapter="depthvlm.py",
        code_env="DEPTHVLM_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/DepthVLM",
        ckpt_env="DEPTHVLM_CKPT_PATH",
        ckpt_default="JonnyYu828/DepthVLM-4B",
        # Exports dense depth only (no camera poses), so SfM, MVRecon and SLAM
        # do not apply.
        tasks=("MonoDepth", "VideoDepth"),
        native=True,
        aliases=("DEPTH_VLM",),
        input_resolution=224,
        image_load_resolution=224,
        resolution_source=("DepthVLM adapter/VLM processor path "
                           "uses 224 image inputs."),
        notes="Dense metric-depth VLM adapter using outputs.depth_pred.",
    ),
    MethodSpec(
        name="VLM3",
        adapter="vlm3.py",
        code_env="VLM3_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/VLM3",
        ckpt_env="VLM3_CKPT_PATH",
        ckpt_default="facebook/VLM3-depth",
        # Exports dense depth only (no camera poses), so SfM, MVRecon and SLAM
        # do not apply.
        tasks=("MonoDepth", "VideoDepth"),
        native=True,
        aliases=("VLM³", "VLM_3"),
        input_resolution=224,
        image_load_resolution=224,
        resolution_source=(
            "Point-query text-depth adapter uses 224 image inputs by "
            "default."),
        notes=(
            "Native point-query adapter; dense depth maps are interpolated "
            "from text scalar depth answers. ckpt_default matches the repo id "
            "facebookresearch/VLM3's own inference.ipynb uses, but that repo's "
            "README still lists its checkpoints "
            "(VLM3-depth/-corr/-object/-pose) as \"Coming Soon!\" -- not yet "
            "published, so download_all_supported_models() reports this as "
            "skipped (a 404, not gating) until Meta actually releases the "
            "weights."),
    ),
    MethodSpec(
        name="DEPTHLM",
        adapter="depthlm.py",
        code_env="DEPTHLM_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/DepthLM_Official",
        ckpt_env="DEPTHLM_CKPT_PATH",
        ckpt_default="facebook/DepthLM",
        # Exports dense depth only (no camera poses), so SfM, MVRecon and SLAM
        # do not apply.
        tasks=("MonoDepth", "VideoDepth"),
        native=True,
        aliases=("DEPTH_LM",),
        input_resolution=750,
        image_load_resolution=750,
        resolution_source=(
            "DepthLM_Official's own eval/inference code always resizes images "
            "so their focal length matches normalized_focal_length=750.0 "
            "(utils/datasets.py) before querying, since its metric-depth "
            "reasoning is tied to that normalized geometry. This repo's "
            "MonoDepth/VideoDepth canonical layout has no per-scene camera "
            "intrinsics to replicate that exactly, so 750px is used as a scale "
            "proxy (matching the *magnitude* DepthLM was evaluated at) rather "
            "than true per-image focal normalization -- a known limitation, "
            "not a guess dressed up as a fix."),
        notes=("Native point-query adapter; dense depth maps are interpolated "
               "from text scalar depth answers."),
        # 256 queries per image, as the SpatialArena rerun (FFB
        # supplement-pointgrid16-20261111).
        point_grid_size=16,
    ),
    MethodSpec(
        name="SENSENOVA_VISION",
        adapter="sensenova_vision_depth.py",
        code_env="SENSENOVA_VISION_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/SenseNova-Vision",
        ckpt_env="SENSENOVA_VISION_CKPT_PATH",
        ckpt_default="sensenova/SenseNova-Vision-7B-MoT",
        # Exports dense depth only (no camera poses), so SfM, MVRecon and SLAM
        # do not apply.
        tasks=("MonoDepth", "VideoDepth"),
        native=True,
        aliases=("SENSENOVA", "SENSENOVA-VISION", "SENSE_NOVA_VISION"),
        input_resolution=224,
        image_load_resolution=224,
        resolution_source="SenseNova-Vision adapter uses 224 image inputs.",
        notes=(
            "Dense-perception image-output adapter; exports normalized "
            "relative depth maps. Its vendored modeling/bagel/qwen2_navit.py "
            "calls torch.compile(flex_attention) at import time; the earlier "
            "Python 3.10 si_bench could not run it (stale files in its cloned "
            "torch tree broke torch._inductor), the rebuilt si_bench imports "
            "it (verified 2026-10-08)."),
    ),
    MethodSpec(
        name="QWEN3_VL",
        adapter="qwen3_vl.py",
        code_env="QWEN3_VL_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/qwen3-vl",
        ckpt_env="QWEN3_VL_CKPT_PATH",
        ckpt_default="Qwen/Qwen3-VL-8B-Instruct",
        # Exports dense depth only (no camera poses), so SfM, MVRecon and SLAM
        # do not apply.
        tasks=("MonoDepth", "VideoDepth"),
        native=True,
        aliases=("QWEN3-VL", "QWEN3VL"),
        input_resolution=224,
        image_load_resolution=224,
        resolution_source=(
            "Point-query VLM baseline adapter uses 224 image inputs by "
            "default."),
        notes=
        ("Native point-query baseline; dense depth maps are interpolated from "
         "prompted scalar depth answers."),
        # 256 queries per image, as the SpatialArena rerun (FFB
        # supplement-pointgrid16-20261111).
        point_grid_size=16,
    ),
    *(
        MethodSpec(
            name=name,
            adapter="openrouter_vlm_depth.py",
            # No local code or weights: the "checkpoint" is the OpenRouter model
            # slug.
            code_env="OPENROUTER_CODE_PATH",
            code_default="",
            ckpt_env=f"{name}_CKPT_PATH",
            ckpt_default=slug,
            # Commercial API VLMs, queried per image; MonoDepth only (VideoDepth
            # would cost ~10x more frames).
            tasks=("MonoDepth",),
            native=True,
            aliases=aliases,
            input_resolution=1024,
            image_load_resolution=1024,
            resolution_source=(
                "Images are sent with a 1024px long side so the 16x16 numbered "
                "marker grid stays legible; the providers resize further "
                "internally."),
            notes=(
                f"Commercial VLM via OpenRouter ({slug}); one call per image "
                "(clean + marked photo) returns metric depths for a 16x16 "
                "numbered marker grid, interpolated to a dense map "
                "(scripts/inference/openrouter_vlm_depth.py). Needs "
                "OPENROUTER_API_KEY or OPENROUTER_API_KEY_FILE."),
        ) for name, slug, aliases in (
            ("CLAUDE_OPUS_5_5", "anthropic/claude-opus-5.5",
             ("CLAUDE-OPUS-5.5", "OPUS_5_5")),
            ("GPT_6_ASTRA", "openai/gpt-6-astra", ("GPT-6-ASTRA",)),
            ("GEMINI_4_ARGON", "google/gemini-4-argon", ("GEMINI-4-ARGON",)),
        )),
    MethodSpec(
        name="GENCEPTION_DEPTH",
        adapter="genception.py",
        code_env="GENCEPTION_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/representations4d",
        ckpt_env="GENCEPTION_CKPT_DIR",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks/"
                      "_checkpoints/local/genception"),
        # Exports dense depth only (no camera poses), so SfM, MVRecon and SLAM
        # do not apply.
        tasks=("MonoDepth", "VideoDepth"),
        native=True,
        aliases=("GENCEPTION-DEPTH",),
        input_resolution=832,
        image_load_resolution=832,
        resolution_source=(
            "GenCeption's VAE/transformer are hardwired to 480x832 (height x "
            "width), anisotropic resize, no aspect-preserving crop -- see "
            "representations4d/genception/utils.py::preprocess_video. "
            "input_resolution here is the width; the adapter hardcodes both "
            "dimensions itself and resizes predictions back to the original "
            "size."),
        notes=(
            "Uses the officially released 'Input: RGB. Output: Depth.' prompt "
            "embedding + pipe.decode_depth() -- reliable, matches "
            "representations4d's own colab demo. Predicts relative "
            "(up-to-scale) depth only. Needs a dedicated conda env "
            "(GENCEPTION_BENCHMARK_PYTHON, JAX/Flax) built via "
            "scripts/slurm/build_genception_env.sh, and the checkpoint fetched "
            "via scripts/download/genception.sh (a multi-file bundle: "
            "transformer + VAE + prompt embeddings, not automatable via "
            "download_all_supported_models())."),
    ),
    MethodSpec(
        name="GENCEPTION_POSE",
        adapter="genception.py",
        code_env="GENCEPTION_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/representations4d",
        ckpt_env="GENCEPTION_CKPT_DIR",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks/"
                      "_checkpoints/local/genception"),
        # Exports camera poses only (no depth maps); not a SLAM-track member.
        tasks=("SfM",),
        native=True,
        aliases=("GENCEPTION-POSE",),
        input_resolution=832,
        image_load_resolution=832,
        resolution_source=(
            "Same 480x832 hardwired resolution as GENCEPTION_DEPTH, but "
            "resized to cover it with the aspect ratio kept and "
            "center-cropped, as the official camera-pose evaluation does; the "
            "fitted pinhole K is mapped back to the original image size."),
        notes=(
            "Official 'Input: RGB. Output: Raymap.' prompt embedding "
            "(raymap.npy) and the raymap decode of "
            "representations4d/genception/evals/eval_camera_pose.py, released "
            "with the answer to google-deepmind/representations4d#10 "
            "(spatialarena/geometry/genception_raymap.py): log-scaled origin "
            "block, 2-patch margins, robust2 rotation solver "
            "(GENCEPTION_POSE_SOLVER=svd for the other), clip-wide fitted K, "
            "temporally smoothed poses (GENCEPTION_RAW_POSES=1 for raw)."),
    ),
    # Official single-image MonoDepth baselines
    # (scripts/inference/single_image_depth.py): each predicts one RGB at a
    # time, on the image's own grid, with the method's own internal resizing
    # (input_resolution 0: not set by us).
    MethodSpec(
        name="MOGE",
        adapter="moge.py",
        code_env="MOGE_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/MoGe",
        ckpt_env="MOGE_CKPT_PATH",
        ckpt_default="Ruicheng/moge-vitl",
        native=True,
        tasks=("MonoDepth",),
        aliases=("MOGE1", "MOGE-1"),
        input_resolution=0,
        image_load_resolution=0,
        resolution_source=(
            "Official MoGe infer(resolution_level=9) internal resize; depth on "
            "the input grid."),
        notes=("Affine-invariant geometry; apply_mask=False, fp32, no GT "
               "intrinsics."),
    ),
    MethodSpec(
        name="MOGE2",
        adapter="moge.py",
        code_env="MOGE_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/MoGe",
        ckpt_env="MOGE2_CKPT_PATH",
        ckpt_default="Ruicheng/moge-2-vitl",
        native=True,
        tasks=("MonoDepth",),
        aliases=("MOGE-2",),
        input_resolution=0,
        image_load_resolution=0,
        resolution_source=
        ("Official MoGe-2 infer(resolution_level=9) internal resize; depth on "
         "the input grid."),
        notes=
        ("Metric depth; apply_mask=False, fp32, no GT intrinsics (as PR #21's "
         "moge_official.py)."),
    ),
    MethodSpec(
        name="MOGE3",
        adapter="moge.py",
        code_env="MOGE_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/MoGe",
        ckpt_env="MOGE3_CKPT_PATH",
        ckpt_default="Ruicheng/moge-3-vitl",
        native=True,
        tasks=("MonoDepth",),
        aliases=("MOGE-3",),
        input_resolution=0,
        image_load_resolution=0,
        resolution_source=
        ("Official MoGe-3 infer(resolution_level=9) internal resize; depth on "
         "the input grid."),
        notes="Metric depth; apply_mask=False, fp32, no GT intrinsics.",
    ),
    MethodSpec(
        name="MARIGOLD",
        adapter="marigold.py",
        code_env="MARIGOLD_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/Marigold",
        ckpt_env="MARIGOLD_CKPT_PATH",
        ckpt_default="prs-eth/marigold-depth-v1-0",
        native=True,
        tasks=("MonoDepth",),
        input_resolution=768,
        image_load_resolution=768,
        resolution_source=(
            "diffusers MarigoldDepthPipeline: checkpoint default processing "
            "resolution (768), match_input_resolution."),
        notes=
        ("Marigold v1-0. Affine-invariant depth in [0, 1]; checkpoint default "
         "denoising steps, ensemble 1, fp32."),
    ),
    MethodSpec(
        name="MARIGOLD_V1_1",
        adapter="marigold.py",
        code_env="MARIGOLD_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/Marigold",
        ckpt_env="MARIGOLD_V1_1_CKPT_PATH",
        ckpt_default="prs-eth/marigold-depth-v1-1",
        native=True,
        tasks=("MonoDepth",),
        aliases=("MARIGOLD-V1-1", "MARIGOLD_V1-1"),
        input_resolution=768,
        image_load_resolution=768,
        resolution_source=(
            "diffusers MarigoldDepthPipeline: checkpoint default processing "
            "resolution (768), match_input_resolution."),
        notes=("Marigold v1-1 (registered as MARIGOLD_V2 before "
               "2026-11-09; Marigold V2, the 2026 diffusion-transformer "
               "model, arXiv 2609.08084, is a different method). "
               "Affine-invariant depth in [0, 1]; checkpoint default "
               "denoising steps, ensemble 1, fp32."),
    ),
    MethodSpec(
        name="MARIGOLD_V2",
        depth_output_space="log_depth",
        adapter="marigold_v2.py",
        code_env="MARIGOLD_V2_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/marigold-v2",
        ckpt_env="MARIGOLD_V2_CKPT_PATH",
        ckpt_default="huawei-bayerlab/marigold-v2-0/depth/Log-stage2",
        native=True,
        tasks=("MonoDepth",),
        aliases=("MARIGOLD-V2",),
        input_resolution=0,
        image_load_resolution=0,
        resolution_source=(
            "Official Marigold V2 native image size rounded to a multiple "
            "of 16 (or MARIGOLD_V2_WIDTH x MARIGOLD_V2_HEIGHT); restored to "
            "the source grid in log-code space."),
        notes=("Marigold V2 (arXiv 2609.08084), Qwen-Image-Edit-2509 "
               "based, Log-stage2 checkpoint, run through its official "
               "scripts/infer.py. Raw normalized log depth in "
               "raw_log_depths/; scored only by log_huber_irls_gt "
               "(docs/benchmark/depth_methods.md)."),
    ),
    MethodSpec(
        name="LOTUS_D",
        depth_output_space="inverse_depth",
        adapter="lotus.py",
        code_env="LOTUS_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/Lotus",
        ckpt_env="LOTUS_CKPT_PATH",
        ckpt_default="jingheya/lotus-depth-d-v2-0-disparity",
        native=True,
        tasks=("MonoDepth",),
        aliases=("LOTUS", "LOTUS-D"),
        input_resolution=768,
        image_load_resolution=768,
        resolution_source=(
            "Official Lotus-D v2 disparity, single step, processing resolution "
            "768, match_input_res."),
        notes=("Normalized disparity kept in raw_disparities/; depths/ holds "
               "1/max(q, 0.001) (as PR #21)."),
    ),
    MethodSpec(
        name="DEPTH_PRO",
        adapter="depth_pro.py",
        code_env="DEPTH_PRO_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/ml-depth-pro",
        ckpt_env="DEPTH_PRO_CKPT_PATH",
        ckpt_default="apple/DepthPro",
        native=True,
        tasks=("MonoDepth",),
        aliases=("DEPTHPRO", "DEPTH-PRO"),
        input_resolution=1536,
        image_load_resolution=1536,
        resolution_source=(
            "Official Depth Pro: fixed 1536x1536 network input, depth resized "
            "back to the input grid."),
        notes=(
            "Metric depth; focal length from the model's FoV head (f_px=None, "
            "EXIF ignored), fp32."),
    ),
    MethodSpec(
        name="UNIDEPTH_V2",
        adapter="unidepth.py",
        code_env="UNIDEPTH_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/UniDepth",
        ckpt_env="UNIDEPTH_V2_CKPT_PATH",
        ckpt_default="lpiccinelli/unidepth-v2-vitl14",
        native=True,
        tasks=("MonoDepth",),
        aliases=("UNIDEPTH", "UNIDEPTHV2", "UNIDEPTH-V2"),
        input_resolution=0,
        image_load_resolution=0,
        resolution_source=(
            "Official UniDepthV2.infer: pad/resize within the training "
            "pixel bounds, depth on the input grid."),
        notes=(
            "Metric depth; no camera input (rgb track); resolution_level unset "
            "(full training pixel range)."),
    ),
    MethodSpec(
        name="INFINIDEPTH",
        adapter="infinidepth.py",
        code_env="INFINIDEPTH_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/InfiniDepth",
        ckpt_env="INFINIDEPTH_CKPT_PATH",
        ckpt_default="ritianyu/InfiniDepth",
        native=True,
        tasks=("MonoDepth",),
        aliases=("INFINI_DEPTH", "INFINI-DEPTH"),
        input_resolution=1024,
        image_load_resolution=1024,
        resolution_source=(
            "Official InfiniDepth: 768x1024 model input, implicit field "
            "queried on the source grid."),
        notes=(
            "RGB model (not the depth-sensor variant); metric scale from the "
            "bundled MoGe-2 checkpoint."),
    ),
    MethodSpec(
        name="DVD",
        depth_output_space="inverse_depth",
        adapter="dvd.py",
        code_env="DVD_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/DVD",
        ckpt_env="DVD_CKPT_PATH",
        ckpt_default="FayeHongfeiZhang/DVD",
        native=True,
        tasks=("VideoDepth",),
        aliases=("DVD_V11", "DVD-V1.1"),
        input_resolution=640,
        image_load_resolution=640,
        resolution_source=(
            "Official DVD training-scale resize (>= 640x480, sides rounded "
            "up to 16), restored to the source grid."),
        notes=("DVD v1.1 on Wan2.1-T2V-1.3B (WAN_MODEL_PATH); window 45 / "
               "overlap 9 stitching. Exports normalized disparity "
               "(raw_disparities/) and the relative-depth proxy 1/max(q, "
               "0.001) (depths/), not metric depth."),
    ),
    MethodSpec(
        name="VGGT_LONG",
        adapter="long_sequence_slam.py",
        code_env="VGGT_LONG_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/VGGT-Long",
        native=True,
        aliases=("VGGT-LONG",),
        tasks=("SLAM",),
        resolution_source=("VGGT-Long base_config.yaml (VGGT-1B "
                           "preprocessing, 518 px)."),
        notes=(
            "Long-sequence system run unmodified through "
            "scripts/inference/long_sequence_slam.py; chunks internally, so it "
            "has no single-pass frame budget. SLAM track only. VGGT-1B "
            "backbone, chunk 60 / overlap 30, SALAD loop closure."),
    ),
    MethodSpec(
        name="PI_LONG",
        adapter="long_sequence_slam.py",
        code_env="PI_LONG_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/Pi-Long",
        native=True,
        aliases=("PI-LONG", "PI3_LONG", "PI3-LONG"),
        tasks=("SLAM",),
        input_resolution=504,
        image_load_resolution=504,
        resolution_source="Pi-Long base_config.yaml (image_width 504).",
        notes=(
            "Long-sequence system run unmodified through "
            "scripts/inference/long_sequence_slam.py; chunks internally, so it "
            "has no single-pass frame budget. SLAM track only. Pi3 backbone, "
            "SALAD loop closure."),
    ),
    MethodSpec(
        name="VGGT_SLAM",
        adapter="long_sequence_slam.py",
        code_env="VGGT_SLAM_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/VGGT-SLAM",
        native=True,
        aliases=("VGGT-SLAM",),
        tasks=("SLAM",),
        resolution_source="VGGT-SLAM main.py (VGGT-1B preprocessing, 518 px).",
        notes=(
            "Long-sequence system run unmodified through "
            "scripts/inference/long_sequence_slam.py; chunks internally, so it "
            "has no single-pass frame budget. SLAM track only. SL(4) submap "
            "optimization with loop closure, run with its evals/eval_tum.sh "
            "settings; keyframe-based, so it returns poses for a subset of the "
            "input frames (reported as completeness)."),
    ),
    MethodSpec(
        name="LOGER",
        adapter="streaming_slam.py",
        code_env="LOGER_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/LoGeR",
        ckpt_env="LOGER_CKPT_PATH",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks"
                      "/_checkpoints/local/loger/LoGeR/latest.pt"),
        native=True,
        tasks=("SLAM",),
        input_resolution=512,
        image_load_resolution=512,
        resolution_source=("LoGeR eval/relpose (TUM): --size 512, "
                           "crop, aligned to 14."),
        notes=(
            "Streaming / long-context method run through "
            "scripts/inference/streaming_slam.py with its own released code "
            "and evaluation settings; SLAM track only. Pi3-based chunked model "
            "with TTT + sliding-window memory; checkpoint LoGeR, window 64 / "
            "overlap 3 (eval/eval.md TUM setting). The public code is a "
            "reimplementation with no license: evaluated, never redistributed."
        ),
    ),
    MethodSpec(
        name="ZIPMAP",
        adapter="streaming_slam.py",
        code_env="ZIPMAP_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/ZipMap",
        ckpt_env="ZIPMAP_CKPT_PATH",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks"
                      "/_checkpoints/local/zipmap/checkpoint_aff_inv.pt"),
        native=True,
        tasks=("SLAM",),
        resolution_source="ZipMap load_and_preprocess_images (518, crop).",
        notes=(
            "Streaming / long-context method run through "
            "scripts/inference/streaming_slam.py with its own released code "
            "and evaluation settings; SLAM track only. VGGT-based linear-time "
            "TTT model; main offline checkpoint checkpoint_aff_inv.pt, one "
            "bidirectional pass per sequence."),
    ),
    MethodSpec(
        name="LINGBOT_MAP",
        adapter="streaming_slam.py",
        code_env="LINGBOT_MAP_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/lingbot-map",
        ckpt_env="LINGBOT_MAP_CKPT_PATH",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks"
                      "/_checkpoints/local/lingbot_map/lingbot-map.pt"),
        native=True,
        tasks=("SLAM",),
        resolution_source=
        ("lingbot-map benchmark config: area budget 255,000 px, aligned to 14 "
         "(574x434 at 640x480)."),
        notes=(
            "Streaming / long-context method run through "
            "scripts/inference/streaming_slam.py with its own released code "
            "and evaluation settings; SLAM track only. Streaming geometric "
            "context transformer (paged KV cache); lingbot-map.pt with its own "
            "benchmark settings (8 scale frames, KV window 64, keyframe "
            "interval auto above 320 frames), SDPA attention."),
    ),
    MethodSpec(
        name="R3",
        adapter="streaming_slam.py",
        code_env="R3_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/R3",
        ckpt_env="R3_CKPT_PATH",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks"
                      "/_checkpoints/local/r3/r3.safetensors"),
        native=True,
        tasks=("SLAM",),
        input_resolution=504,
        image_load_resolution=504,
        resolution_source="R3 demo.py --size 504 (default).",
        notes=(
            "Streaming / long-context method run through "
            "scripts/inference/streaming_slam.py with its own released code "
            "and evaluation settings; SLAM track only. DA3-based relative-pose "
            "regression with a keyframe bank; paper checkpoint r3.safetensors "
            "with the indoor preset demo.py --mode local (dynamic KV cache, no "
            "fallback, no metric scale)."),
    ),
    MethodSpec(
        name="SCAL3R",
        adapter="streaming_slam.py",
        code_env="SCAL3R_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/Scal3R",
        ckpt_env="SCAL3R_CKPT_PATH",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks"
                      "/_checkpoints/local/scal3r/scal3r.pt"),
        native=True,
        tasks=("SLAM",),
        resolution_source=(
            "Scal3R release config (proc_max_size 518, aligned to 14, center "
            "crop)."),
        notes=(
            "Streaming / long-context method run through "
            "scripts/inference/streaming_slam.py with its own released code "
            "and evaluation settings; SLAM track only. VGGT-based chunked TTT "
            "model; release config (block 60, overlap 30, DINO-SALAD loop "
            "closure), single GPU. Released as inference only (no evaluation "
            "code)."),
    ),
    MethodSpec(
        name="ABOT_RECON",
        adapter="streaming_slam.py",
        code_env="ABOT_RECON_CODE_PATH",
        code_default="$METHOD_CODE_ROOT/ABot-Recon",
        ckpt_env="ABOT_RECON_CKPT_PATH",
        ckpt_default=(f"{SPATIALARENA_ROOT}/benchmarks"
                      "/_checkpoints/local/abot_recon/abot_recon.safetensors"),
        native=True,
        tasks=("SLAM",),
        input_resolution=504,
        image_load_resolution=504,
        resolution_source=(
            "ABot-Recon InferenceConfig: 504x280 (width-locked resize, center "
            "crop)."),
        notes=(
            "Streaming / long-context method run through "
            "scripts/inference/streaming_slam.py with its own released code "
            "and evaluation settings; SLAM track only. Pi3-based streaming "
            "model with a fixed 12-frame local window and relative-pose "
            "chaining; its loop-closure backend on, as by default (DINO-SALAD "
            "retrieval, sparse pose-graph optimization), SDPA attention."),
    ),
)


def normalize_key(value: str) -> str:
    """Returns the lookup key of a method name: ASCII, upper case, `_`."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(
        ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_text.upper().replace(" ", "_").replace("-", "_")


_METHOD_BY_KEY: dict[str, MethodSpec] = {
    normalize_key(key): spec for spec in METHOD_SPECS
    for key in (spec.name, *spec.aliases)
}


def method_names() -> list[str]:
    """Returns the canonical names of all registered methods."""
    return [spec.name for spec in METHOD_SPECS]


def resolve_method(value: str) -> MethodSpec:
    """Returns the method registered under a name or alias.

    Args:
        value: Method name or alias, in any case, with `-`, `_` or spaces.

    Returns:
        The method spec.

    Raises:
        KeyError: If no method has this name or alias.
    """
    key = normalize_key(value)
    if key not in _METHOD_BY_KEY:
        raise KeyError(value)
    return _METHOD_BY_KEY[key]


def valid_tracks(spec: MethodSpec | str) -> tuple[str, ...]:
    """Returns the tracks a method (spec, name or alias) supports."""
    if isinstance(spec, str):
        spec = resolve_method(spec)
    return spec.tracks


def validate_track(method: str, track: str) -> MethodSpec:
    """Returns the method if it supports a track.

    Args:
        method: Method name or alias.
        track: Input-modality track, e.g. `TRACK_RGB_DEPTH`.

    Returns:
        The method spec.

    Raises:
        KeyError: If the method is unknown.
        ValueError: If the method does not support the track.
    """
    spec = resolve_method(method)
    if track not in spec.tracks:
        raise ValueError(f"{spec.name} does not support track {track!r}; "
                         f"valid: {', '.join(spec.tracks)}")
    return spec


def specs_as_dicts(specs: Iterable[MethodSpec] = METHOD_SPECS) -> list[dict]:
    """Returns the methods as dicts, with their derived properties added.

    Args:
        specs: Methods to convert; all methods by default.

    Returns:
        One dict per method: its fields, then `external_command_env` and
        `default_depth_protocol`.
    """
    rows = []
    for spec in specs:
        row = dataclasses.asdict(spec)
        row["external_command_env"] = spec.external_command_env
        row["default_depth_protocol"] = spec.default_depth_protocol
        rows.append(row)
    return rows
