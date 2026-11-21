# Depth methods: output spaces, alignment and method-specific settings

How depth predictions are aligned to the ground truth (by each method's declared output space), and the settings
specific to a few depth methods: the Marigold versions, the point-query VLMs (DepthLM, Qwen3-VL) and the video
diffusion baselines (DepthCrafter, ChronoDepth). The metrics and per-dataset protocols are in `docs/evaluation.md`.

## Native output-space alignment

The standard engine selects Huber+IRLS alignment using the method registry's
`depth_output_space`: `depth` selects `depth_huber_irls_gt`; `inverse_depth`
selects `inverse_huber_irls_gt`. Geometry models (including VGGT-Omega and DA3)
export depth. Lotus-D, DVD and DepthCrafter declare inverse depth; ChronoDepth declares depth.
Benchmark configs that pin `MONO_DEPTH_PROTOCOL` / `VIDEO_DEPTH_PROTOCOL` pin the protocol of each model's
declared space. A changed checkpoint representation must update this declaration or explicitly
override its protocol; do not infer it from the model family name or output
filename.

For disparity, fit against inverse GT depth, preferring preserved
`raw_disparities` when available, then convert the aligned prediction to depth.
For native depth, fit directly against GT depth. Metrics are computed in depth
space in both cases. This changes alignment space, not the established
per-image MonoDepth / per-sequence VideoDepth aggregation, input frame selection,
GT masks, crop settings or depth ranges. It does not include alternative solvers
from the six-protocol diagnostic study.

An explicit `MONO_DEPTH_PROTOCOL` or `VIDEO_DEPTH_PROTOCOL` in a model's YAML
`env` takes precedence, for reproducible existing protocols and diagnostics.
Unregistered external commands retain legacy defaults; explicitly set their
protocol. This automatic choice applies to standard engine runs; direct calls
to the evaluator CLI still require the appropriate `--protocol` argument.

Registry declarations here cover depth and inverse-depth. Normalized log-depth
is a different representation; it must not be declared as either without a
correct decoder. It has its own protocol, `log_huber_irls_gt`
(`spatialarena/eval/log_depth_gt.py`), the one the FFB releases score
Marigold V2 (`MARIGOLD_V2`) with: the prediction is the normalized log depth
`q` in `raw_log_depths/` (the decoded `depths/` beside it are never used),
one affine fit `a * q + b` to `log(GT depth)` over the GT-valid pixels of each
image (or sequence) by Huber IRLS with delta 0.05 on the absolute log-depth
residual (unit 1 -- not divided by a GT median, since log depth is not a
positive scale), decoded as `exp(a * q + b)` and clipped to the dataset's depth
range. `MARIGOLD_V2` declares `depth_output_space="log_depth"`, so the engine
selects this protocol for it by default; it can also be selected by name
(`--protocol log_huber_irls_gt`, or `MONO_DEPTH_PROTOCOL` /
`VIDEO_DEPTH_PROTOCOL`), as the release evaluations record it. See
[Marigold versions](#marigold-versions) below.

### Constant predictions

A prediction that is the same value on every GT-valid pixel of a frame (MonoDepth) or sequence (VideoDepth) carries
no scale information, so the affine fit is undefined. Both Huber+IRLS protocols then fit the offset alone, with the
scale fixed at 0, under the same Huber objective (`huber_irls._fit_constant_offset`): the frame is scored as the best
single depth (or inverse depth) for it. The method keeps the full penalty for the flat answer, and its other frames
are still scored; before this, the evaluator raised `Degenerate constant prediction` and the whole scene failed. The
fit record marks such frames with `constant_prediction: true`. This happens in practice with text-answering VLMs (an
API model once answered 1.1 m for all 256 points of an iBims-1 image) and with single-point baselines such as
QWEN3_VL's default one-query grid.

## Marigold versions

Keep each method name tied to its actual checkpoint:

| Method ID | Checkpoint | Native output / alignment |
| --- | --- | --- |
| `MARIGOLD` | `prs-eth/marigold-depth-v1-0` | affine depth / `depth_huber_irls_gt` |
| `MARIGOLD_V1_1` | `prs-eth/marigold-depth-v1-1` | affine depth / `depth_huber_irls_gt` |
| `MARIGOLD_V2` | `huawei-bayerlab/marigold-v2-0/depth/Log-stage2` | normalized log-depth / `log_huber_irls_gt` |

Some older local results called the v1.1 checkpoint `MARIGOLD_V2`. Check their
recorded checkpoint before correcting a historical label; this code does not
rename existing artifacts or mix their scores with actual V2 results.
The Marigold dataset collection is unrelated to the model version names.

### ETH3D resolution

Marigold V2 runs with the other MonoDepth methods in `configs/benchmark/full_monodepth.yaml`, at its native size
rounded to a multiple of 16, except Marigold's ETH3D subset: **1008 × 672 (width × height)**, set by
`MARIGOLD_V2_SCENE_SIZES: marigold/eth3d=1008x672` in its model entry. Native resolution on ETH3D produced
substantially worse estimates in a local 13-image diagnostic. Every ETH3D image (454 in the canonical split) is
scored, as for every MonoDepth method; `MAX_IMAGES_PER_RUN: 0` makes each image-size group one adapter call, since the
official program loads the model on every call. Prepare the official V2 code (`MARIGOLD_V2_CODE_PATH`) and its model
assets (`MARIGOLD_V2_CKPT_PATH`) before running it.

`MARIGOLD_V2_SCENE_SIZES` (`<dataset>/<scene>=<W>x<H>`, comma-separated) sets single scenes;
`MARIGOLD_V2_WIDTH` and `MARIGOLD_V2_HEIGHT` set every other scene. Sizes must be positive multiples of 16; zero or
unset keeps the native resolution rounded to a multiple of 16.
Fixed-size predictions are bilinearly restored to the original image grid
**in log-code space**. The adapter only stages benchmark-selected images.

### Representation and scoring

`raw_log_depths/*.npz` stores signed normalized log codes; `depths/*.npz`
contains positive exponentiated proxies solely for matching image artifacts.
The evaluator fits `a*q+b` to `log(GT)` on GT-valid pixels per MonoDepth image,
then computes the existing depth metrics on `exp(a*q+b)`. Huber uses a fixed
log residual unit of 1 and delta 0.05, independent of depth units. Missing
raw codes, nonfinite predictions on valid pixels, or a depth/inverse-depth
protocol applied to these exports fail explicitly.

This is the benchmark's log-space alignment, not a reproduction of every detail of the
authors' evaluation. Existing depth/inverse-depth fitting and constant-output
handling are unchanged. V2 runtime includes subprocess model loading and is
labelled accordingly; it is not pure inference throughput.

## Point-query VLMs: strict 16 × 16 queries

DepthLM and Qwen3-VL produce text depth answers for queried pixels. The engine
already queries a 16 × 16 grid (256 points per image) for both: their registry
entries set `point_grid_size=16` (`spatialarena/methods.py`), which the engine
exports as `POINT_QUERY_DEPTH_GRID_SIZE` unless the config sets its own. To also
make the queries strict, set this in the formal evaluation YAML's `env` mapping:

```yaml
env:
  POINT_QUERY_REQUIRE_SPATIAL: "1"
```

The equivalent adapter arguments are `--point_grid_size 16
--require_spatial_queries`. The strict flag requires a grid of at least 2 × 2;
it does not select 16 × 16 by itself. Without it, the legacy fallback behavior
stays (the adapter alone, run by hand, still defaults to a single centre query).
Generate and review the job scripts through the standard coverage pipeline
before submitting.

The grid is defined on the loaded image after the adapter's existing image
resize. Each point is queried independently, in batches. Positive finite
parsed depths are valid. Missing answers are filled with the median of valid
answers **from that same image**, then the grid is bilinearly interpolated to
a dense depth map. Strict mode does not require all 256 answers to be valid.
If none are valid, the adapter fails instead of inventing a constant map or
borrowing a depth from a previous image.

For each fully queried image, including an all-invalid image that triggers a
failure, the adapter saves:

- `point_query_outputs.jsonl`: queried coordinates, raw answer text, and parsed
  depth; invalid answers are JSON `null`.
- `point_query_coverage.json`: an array containing the image path, attempted
  query count (`queries`), and valid answer count (`valid`).

Successful runs also include the strict flag, interpolation policy, coverage,
parse failures, and fallback counts in adapter runtime metadata. If a run
fails, evidence covers only images whose queries finished; dense predictions
are exported only after all selected images finish. Use a fresh output folder
when checking failures so older artifacts cannot be mistaken for new output.

A valid constant response (for example, 100 at every queried point) is still
exported as a constant map. Strict querying does not change alignment,
degenerate-prediction handling, scoring, or whether an evaluation is successful.
Query coverage alone is not evidence that all frames were scored successfully.

## Video diffusion baselines: DepthCrafter and ChronoDepth

Both RGB-only methods use `scripts/inference/video_diffusion_depth.py` with the
upstream pipelines, not MP4 subprocesses. The adapter reads precisely the frames
selected by the benchmark, preserves canonical names (including symlinks), and restores
numeric predictions to the source image grid. It rejects missing frames,
nonfinite outputs and constant DepthCrafter disparity.

| Method | Official implementation | Checkpoint | Steps | Window / advance |
| --- | --- | --- | --- | --- |
| DEPTHCRAFTER | https://github.com/Tencent/DepthCrafter | tencent/DepthCrafter | 5 | 110 / 85 |
| CHRONODEPTH | https://github.com/jiahao-shao1/ChronoDepth | jhshao/ChronoDepth-v1 | 5 | 10 / 5 |

Both use `stabilityai/stable-video-diffusion-img2vid-xt` as the base, fp16,
seed 42, official open-video long-edge limit 1024 and dimensions rounded to 64.
ChronoDepth uses `infer_mode=ours`, sigma epsilon -4, decode chunk 8.
DepthCrafter uses guidance 1 and sequence-wide normalization after decoding.
DepthCrafter's benchmark-specific fixed resize profiles are not selected: this
integration uses its official open-video preprocessing consistently on all datasets.
No temporal resampling or MP4 compression is introduced. Selected frames may
already be sparse; the adapter does not claim they constitute full continuous video.

DepthCrafter outputs disparity. Store it in `raw_disparities`; export the relative
depth proxy `1/max(disparity, 0.001)` in `depths`. ChronoDepth outputs normalized
depth directly, clipped to [0,1] and floored at 1e-6 for depth metrics. Predictions
are not metric depths. Record clipping fractions in adapter metrics. Use sequence
inverse-depth Huber+IRLS for DepthCrafter and sequence depth Huber+IRLS for
ChronoDepth in the local campaign; report this protocol distinction in comparisons.
Neither model receives GT during inference.

Set the method-specific CODE_PATH and CKPT_PATH variables; `SVD_BASE_PATH` can
point to a cached base model. The current local campaign covers Bonn, KITTI,
Sintel and DyCheck with the benchmark's case lists and frame caps. The shared suite
keeps its existing dataset selection. Generate YAML coverage, review scripts,
then submit. Native registration is not proof of successful GPU inference: verify
actual output counts, shapes and scoring reports before marking completed.

### Dependencies and validation

Clone each linked official repository under `METHOD_CODE_ROOT`, install its
upstream inference dependencies in a compatible Python environment, and provide
that interpreter through the YAML `PYTHON` environment entry. The adapter imports
PyTorch, diffusers, transformers, NumPy, Pillow and OpenCV. It does not change or
install packages at inference time. Model loading requires the official UNet,
SVD image encoder, VAE, scheduler and feature-extractor files; it does not require
the SVD base UNet when the method UNet is supplied. Pre-download these for offline
compute nodes. Use the official repositories' dependency constraints as the
starting point; a successful import is not a successful GPU forward pass.

For a real GPU check before a full suite, run the adapter on a small canonical
scene (`images/`), using `--method DEPTHCRAFTER` or `--method CHRONODEPTH`,
`--code_path`, `--data_path`, `--output_path`, and `--export_depth`. Check one
finite float32 map per selected frame at its original resolution. A 12-frame
check exercises ChronoDepth's default window advance; more than 110 frames
exercise DepthCrafter's default overlap. Keep diagnostics outside canonical
report directories and retain formal YAML coverage/review/submit for evaluation.
