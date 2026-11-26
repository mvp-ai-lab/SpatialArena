# Per-method image preprocessing and output postprocessing

Every benchmark scene's images are a fixed size, but each model wants its input at a size and
shape of its own choosing. This document explains, for each method, in plain terms: **what
happens to the image before the model sees it**, and **what happens to the model's depth output
before it is scored**, so that two different methods evaluated on the same scene are actually
being compared on the same ground truth.

This work resolves GitHub issue #19. SLAM's separate, already fixed-resolution input pipeline,
unaffected by any of this, is in `docs/benchmark/slam_protocol.md`.

## Why this matters, in one picture

A ground-truth depth map has one pixel per pixel of the original photo. If a model's own
pipeline resizes that photo before predicting depth, and the adapter exports the prediction
without undoing that resize, the evaluator ends up comparing a prediction grid of one shape
against a ground-truth grid of a different shape, stretching one onto the other with bilinear
interpolation. That stretch is wrong whenever the model's resize changed the aspect ratio (not
just the pixel count) -- which is exactly what happens when a model pads an image onto a square
canvas, or crops it to fit a fixed grid, rather than simply resizing it. The fix in this codebase
has two matching halves for every method below:

1. **Before inference**: record exactly what happened to each image on its way into the model --
   how much it was scaled, and how many pixels (if any) of padding or cropping were added on each
   side. This record is a `SpatialTransform` (defined in `spatialarena/models/utils.py`).
2. **After inference**: use that same record to undo it -- scale the predicted depth map back
   down, strip off any padding, and mark any border pixel the model never actually saw as invalid
   (not a guess). Camera intrinsics get the matching inverse transform, so a COLMAP export's focal
   length and principal point describe the *original* image, not the model's resized one.

Both halves are driven by one invocation every adapter makes, described next.

## The one call every adapter makes

Previously, each adapter called its own differently-named helper to do this -- `build_shared_views`
in one file, `build_pure_resize_transforms` in another, `materialize_resized_images` in a third --
so answering "how does method X preprocess its images" meant finding and reading a different
function every time, and it was easy for one adapter to duplicate (and subtly diverge from)
logic another adapter already had. There is now exactly one name to look for:

```python
model_input, transforms = prepare_model_input(METHOD, image_paths, args=args, device=device)
```

Methods already on `spatialarena.models.BaseModel` (VGGT, RobustVGGT, FastVGGT, VGGT-Omega, Pi3X, DA3; see
`docs/models.md`) implement this as their `preprocess()` method instead: it returns a `ModelInput` carrying the
same `transforms`, and the base class does the restoration below. The rest still use:

`prepare_model_input` (in `spatialarena/models/utils.py`) looks up `METHOD`'s own implementation -- a function
named `preprocess_<method>` that every adapter module registers once, near its definition, via
`register_preprocessor(METHOD, preprocess_<method>)` -- and calls it. The implementation still
varies by method (a tensor, a list of per-view dictionaries, a batch dict, or a list of file paths
a subprocess reads from disk, depending on what that model's own code actually needs), but the
name is always the same shape: `grep "^def preprocess_" scripts/inference/*.py` lists every
method's implementation in one place. `transforms` is the list of per-image `SpatialTransform`
records from step 1 above (or `None` for the handful of tracks explained in "What doesn't get
restored, and why" below).

On the output side, every method calls the same two functions: `restore_prediction` undoes a
`SpatialTransform` on one depth or confidence map, and `export_spatial_depth_predictions` calls it
on **both** the depth and the confidence map for a batch, so a method's confidence is always at
the same resolution as its depth -- never one restored and the other left stale.

## How each method actually resizes its images

Grouped by what geometric operation actually happens, in plain terms. "Lossless" means every
pixel of the original photo reaches the model somewhere on its resized/padded canvas; nothing is
thrown away before inference.

**Pad to a square, keeping the whole image (lossless).** The image is scaled down (or up) so its
*longer* side exactly fits the target canvas size, keeping the original aspect ratio, and then
centered inside a square canvas of that size -- the leftover strip on the shorter side (top+bottom
for a wide image, left+right for a tall one) is filled with black padding, never cropped.
- **VGGT, ROBUSTVGGT**: pads to a square sized to a multiple of VGGT's 14-pixel patch grid, then
  (for historical reasons kept only under the default, unrestored `--spatial_policy legacy`) resizes
  that square a second time to `--resolution` -- two resizes where one would do, kept only because
  changing it would change VGGT's own historical numbers. Under `--spatial_policy shared`, this
  collapses into the single pad-to-square step, straight to `--resolution`.
- **FASTVGGT**: the same pad-to-square step on VGGT's 14-pixel patch grid (`plan_spatial_transform`
  handles it in VGGT's branch, policy `fastvggt_resize_pad_square`; the canvas must be a multiple of 14).
- **CUT3R, FAST3R, DUST3R, MAST3R, POW3R**: same pad-to-square idea, at a canvas sized
  to a multiple of 16 instead of 14 (these share a 16-pixel patch grid). CUT3R, FAST3R, DUST3R,
  MASt3R and POW3R all originally came with their own loader that did something *lossy* instead
  (see "What was actually broken" below); they now build the same pad-to-square input directly,
  bypassing that loader entirely.
- **MAPANYTHING** (`rgb` track only): same idea, canvas sized to a multiple of 14 (MapAnything's
  real patch size, even though its own vendored loader's default convention is 16).
- **AMB3R**: pads to its fixed 518x392 working size (or, for SLAM's pre-sized frames, whatever
  size those already are) by asking its own loader not to crop (`disable_crop=True`), rather than
  reimplementing the pad math separately.
- **VGGT_OMEGA**: the same pad-to-square idea, canvas sized to a multiple of 16, under
  `--spatial_policy shared-pad` (its own name for "shared" here, since it has a few resize variants
  to choose from). On a SLAM-protocol fixed-size input, it instead only trims to the nearest
  16-pixel grid line (see "small, separate crops" below) -- there is nothing left to pad.
- **OMNIVGGT**: same idea; writes the padded images (and, for its camera/depth tracks, the padded
  depth and camera files alongside them) to a temporary directory for its own loader to read,
  rather than building a tensor directly in memory.
- **COME (Co-Me)**: pads to a square sized 504 (hardcoded by Co-Me's own code, which ignores any
  other size you ask for), written to disk for its own subprocess to read; that subprocess then
  resizes the already-square 504 canvas a second time to its working resolution. Since resizing a
  square image keeps it square, this second resize doesn't change the padding *proportion* -- so
  one combined pad-to-square transform straight from the true photo to the final working
  resolution describes the net effect of both steps correctly.

**Resize to fit, no padding (lossless, but changes the field of view when the aspect ratio
changes).** The image is scaled so it fits a target box, and that's it -- no cropping, but also
no padding, so a non-square photo coming out of this step has had its aspect ratio altered (a
square photo stretched into a wide one, for instance), unlike the pad methods above which keep
the aspect ratio and add blank borders instead.
- **DA3**: resizes so the longer side hits a target length, rounding to the nearest multiple of
  14 (rounding ties up, matching DA3's own convention exactly).
- **IGGT**: stretches the image directly onto a square canvas, abandoning the original aspect
  ratio entirely (confirmed by reading IGGT's own loader, which documents this itself) -- the only
  method here that does this.
- **PI3_X**: resizes the whole batch to one shared size; which exact formula depends on the
  track, but it is always a resize with no crop or pad margin.
- **DEPTHVLM, SENSENOVA_VISION, and the point-query methods (DEPTHLM, QWEN3_VL)**: resize
  per image (each one can land at a different size, since these methods' own vision towers choose
  their working resolution per image rather than per batch) with no crop or pad.

**Pad a whole batch to a shared size (lossless, batch-dependent canvas).**
- **DEJAVIEW**: resizes each image to fit a target box (preserving aspect ratio, same idea as the
  pad-to-square methods above, just not necessarily square), rounds each one up to the next
  multiple of DejaView's 14-pixel patch grid, and then pads every image in the batch out to
  whichever of them needed the largest canvas -- so the canvas size depends on the whole batch,
  not just one image, but every image still keeps all of its own content.
- **IGGT4D** (`rgb` track only): pads every image to one shared square canvas sized to a multiple
  of 14, computed up front, and tells IGGT4D's own code the canvas is already that exact size so
  its own resize step becomes a no-op (it would otherwise crop the whole batch down to whichever
  image came out smallest after its own per-image rounding -- seeing a fixed size ahead of time
  heads that off).

**Already fixed, nothing to resize.**
- **GENCEPTION_DEPTH**: resizes to its fixed 480x832 input and the depth back to each original size;
  nothing needed changing. **GENCEPTION_POSE** resizes to cover 480x832 keeping the aspect ratio and
  center-crops, as GenCeption's official camera-pose evaluation does, and maps the fitted K back to the
  original image size.
- **The 9 streaming SLAM methods** (VGGT_LONG, PI_LONG, VGGT_SLAM, LOGER, ZIPMAP, LINGBOT_MAP, R3,
  SCAL3R, ABOT_RECON): read frames that `materialize_slam_inputs.py` already resized to one fixed
  size per dataset, before any method sees them -- there is no further adapter-side resize to get
  right.

**Official single-image MonoDepth baselines and DVD: the method's own resize, checked on the way out.**
These run their upstream inference code unmodified and never see a benchmark resize; each returns
depth on the input image's grid, and `scripts/inference/single_image_depth.py` fails the case if a
prediction's shape is not exactly the source image's (H, W) -- nothing is resized afterwards.
- **MOGE / MOGE2 / MOGE3**: `MoGeModel.infer(resolution_level=9)` sizes its own token grid from the
  image's aspect ratio and returns depth at the input resolution.
- **MARIGOLD / MARIGOLD_V1_1**: diffusers' `MarigoldDepthPipeline` resizes the long side to the
  checkpoint's processing resolution (768) and back (`match_input_resolution`).
- **LOTUS_D**: the official pipeline's 768 long-side processing resolution, bilinear back to the
  input (`match_input_res`).
- **DEPTH_PRO**: the official transform feeds a fixed 1536x1536 network input; `DepthPro.infer`
  interpolates depth back to the input size.
- **UNIDEPTH_V2**: pads/resizes within its training pixel bounds internally and returns depth at the
  input size.
- **INFINIDEPTH**: stretches the image to a 768x1024 model input (official preprocessing) and
  queries its implicit depth field on the source pixel grid (`output_resolution_mode="original"`).
  Its loader applies EXIF orientation, so a rotated input would come back transposed and fail the
  shape check.
- **DVD** (VideoDepth): the official training-scale resize (`plan_spatial_transform(method="dvd")`:
  at least 640x480, each side rounded up to 16, no crop or pad); the stitched disparity is restored
  to the source grid with `restore_prediction` before inversion.
- **DEPTHCRAFTER / CHRONODEPTH** (VideoDepth, `scripts/inference/video_diffusion_depth.py`): the
  official open-video preprocessing -- each side rounded to 64, long edge at most 1024 -- applied to
  every frame of the sequence (frames must share one source size); the prediction is resized
  bilinearly back to the source grid. See `docs/benchmark/depth_methods.md`.

## What was actually broken, and is now fixed

Several methods' own, original loaders did not just resize -- they **cropped**, discarding real
image content before the model ever saw it, independent of anything this fix's restoration step
could undo afterward:

- **CUT3R, FAST3R** (and the shared DUST3R/MASt3R/POW3R loader) resized an image down, then
  center-cropped it to the nearest *lower* multiple of their 16-pixel grid -- and, for a
  perfectly square input specifically, cropped it down to a narrower 4:3 window on top of that.
- **DejaView**'s own loader rounded each image's size *down* to the nearest patch multiple (a
  crop) before padding the batch, instead of rounding up.
- **MapAnything**'s and the DUST3R-family's shared loader picked one target size from the
  *average* aspect ratio of the whole batch, then cropped every image in the batch to fit it --
  so how much of a given image got cropped depended on what other images happened to be batched
  with it.
- **`materialize_slam_inputs.py`** (feeding every SLAM method) center-cropped VBR and KITTI
  Odometry frames to the SLAM protocol's fixed aspect ratio.

All four are now fixed by building the model's input directly (bypassing the lossy loader) using
the pad-based conventions described above, so no source pixel is ever discarded.

**Co-Me** had a different kind of gap: it was never actually fixed at all, until this was
specifically checked against GitHub issue #19's list of affected methods -- only a defensive guard
against one narrower crop risk existed.

## What doesn't get restored, and why

For three tracks, the image is deliberately left un-restored -- the output stays at the model's
own working resolution, not the photo's original size -- because restoring it would require a
riskier trade: these tracks feed the model a real depth map or camera pose *alongside* the image
as a conditioning input, and that conditioning input has its own resize math that assumes it is
being resized together with the image it was computed from. Pre-padding the image to a canvas
picked without reference to that depth/camera prior would desync the two -- the model would end
up conditioning on a depth map or camera pose that no longer lines up with the image pixels it is
paired with, which is a worse failure mode than the misalignment this fix is otherwise closing.

- **MapAnything's `rgb_depth` and `rgb_cameras` tracks.**
- **IGGT4D's `rgb_cameras` track.**

This is flagged in each adapter's own code as a to-do, not silently left broken.

## Turning this on

Nine method specs -- **VGGT, ROBUSTVGGT, FASTVGGT, DA3-L, DA3-G, DA3-NGL, VGGT_OMEGA, CUT3R,
COME** -- keep their *original*, un-restored behavior by default, so a real benchmark run only
gets the fix if its config explicitly opts in with `env: {SPATIAL_POLICY: shared}` on that model
entry (`shared-pad` for `VGGT_OMEGA`); every production config
(`full_monodepth.yaml`/`full_videodepth.yaml`/`full_sfm.yaml`/`full_mvrecon.yaml`/`all_tracks.yaml`)
now sets this. Every other method listed as "restored" above has no such switch -- the fix is
simply always on whenever `--depth_only` is requested. This asymmetry exists only because those
9 methods already had real benchmark numbers computed with the old, unrestored behavior; keeping
`legacy` as their default means old results stay reproducible, and `shared` is an explicit,
visible opt-in to new ones.

## Small, separate crops that remain (not a bug, not touched by this fix)

A few methods' own patch-grid alignment still trims a handful of pixels *after* this fix's
lossless input already reaches them -- a much smaller effect than the real content loss described
above, and this fix does not touch it:

- CUT3R, FAST3R, and VGGT-Omega's own 16-pixel patch grid can drop a 4-6 pixel border when the
  SLAM protocol's fixed 504-pixel-wide input isn't itself an exact multiple of 16 (see
  `docs/benchmark/slam_protocol.md`'s "Why 504" section for why 504 was chosen anyway).

## Verification status per method

"Verified" means `scripts/test/run_resolution_consistency_smoke.py` actually ran that method's
real adapter, with its real checkpoint and real vendored model code (not a mock, not `--dry_run`),
against a tiny synthetic scene, and confirmed the exported depth map, confidence map, and restored
COLMAP intrinsics all land at the scene's true source resolution. "Code fix, unverified" means the
restoration math above is implemented and checked with a mocked model, but a real run hasn't
completed -- always because of something unrelated to this fix (a GPU-only requirement in that
method's own code, or a third-party library gap), never because the restoration logic itself is
in doubt.

| Method | Status | If not fully verified, why |
|---|---|---|
| VGGT / ROBUSTVGGT | Verified (CPU) | |
| FASTVGGT | Verified (CPU) | |
| DA3-L / DA3-G / DA3-NGL | Verified (CPU, both `rgb` and `rgb_cameras`) | |
| CUT3R | Verified (CPU) | |
| FAST3R | Verified (CPU) | |
| DUST3R / MAST3R / POW3R | Verified (CPU) | |
| MAPANYTHING (`rgb`) | Verified (CPU) | |
| OMNIVGGT (`rgb`, `rgb_depth`, `rgb_cameras`) | Verified (CPU) | |
| PI3_X (`rgb`, `rgb_depth`, `rgb_cameras`) | Verified (CPU) | |
| IGGT | Verified (CPU) | |
| IGGT4D (`rgb`) | Verified (CPU) | |
| DEJAVIEW | Verified (CPU) | |
| DEPTHVLM, DEPTHLM, QWEN3_VL | Verified (CPU) | |
| COME | Code fix, unverified | Co-Me's own vendored code loads its checkpoint with a CUDA-only call (no `map_location="cpu"`), so it cannot run on CPU at all |
| VGGT_OMEGA | Code fix, unverified | its own upstream implementation explicitly refuses to run without CUDA |
| AMB3R | Code fix, unverified | a third-party library its point-cloud refinement step depends on (`spconv`) has no CPU/bfloat16 support at all |
| SENSENOVA_VISION | Code fix, unverified | its vendored image decoder's weights are bfloat16 with no CPU-compatible path |
| GENCEPTION_DEPTH / GENCEPTION_POSE | Not run | already correct, no fix needed; its own package isn't installed in this environment and it is also a very large diffusion model, likely impractical on CPU regardless |
| VGGT_LONG, PI_LONG, VGGT_SLAM, LOGER, ZIPMAP, LINGBOT_MAP, R3, SCAL3R, ABOT_RECON | Not applicable | already correct (fixed-resolution SLAM input, confirmed by code review); a different CLI shape entirely, out of scope for this smoke test |
| MOGE, MOGE2, MOGE3, MARIGOLD, MARIGOLD_V1_1, LOTUS_D, DEPTH_PRO, UNIDEPTH_V2, INFINIDEPTH, DVD | Not yet run (GPU check queued) | added 2026-10-23; the smoke test above does not cover them (different CLI). Their first GPU run on `MonoDepth/ibims1` (DVD: `VideoDepth/sintel` `alley_1`) is checked with `scripts/test/check_prediction_resolution.py`; the adapters themselves fail any off-grid output |
| DEPTHCRAFTER, CHRONODEPTH | Not yet run (GPU check queued) | added 2026-10-30 (PR #21); CUDA-only, not covered by the smoke test above. Check the first VideoDepth run with `scripts/test/check_prediction_resolution.py` |

## Verifying a change

After touching any adapter's `preprocess_<method>` or its restoration path:

1. Fast check: a synthetic-image numeric test of that method's resize formula, and a mocked
   end-to-end `run()` test (see the VGGT, CUT3R, Co-Me and VGGT-Omega tests in `test/test_model_adapters.py` for the
   pattern).
2. Real check: `python scripts/test/run_resolution_consistency_smoke.py --methods <NAME>` (add
   `--device cuda` once GPU is available) -- runs the real adapter CLI with its real checkpoint and
   confirms the exported depth/confidence/COLMAP intrinsics land at the true source resolution.
   Mocked tests alone have missed real bugs (a missing view-dict key, an unhandled tensor shape)
   that only show up running the actual model forward pass.
