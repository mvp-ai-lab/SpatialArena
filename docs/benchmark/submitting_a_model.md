# Submitting a model: writing a compatible inference script

To evaluate a model we do not have an adapter for, the pipeline runs **one command** that you provide, once per
scene, and scores what it writes. This guide describes that command's inputs and outputs, gives a template to start
from, and shows how to check your script locally before you submit it on the
[Submit page](https://mvp-ai-lab.github.io/SpatialArena/#submit).

The short version:

1. Copy [`scripts/inference/submission_template.py`](../../scripts/inference/submission_template.py) into your
   repository (a fork or a branch is fine) and replace its `predict()` with your model.
2. Run it on a downloaded scene and check the output with
   [`scripts/inference/check_submission_output.py`](../../scripts/inference/check_submission_output.py).
3. Submit the repository, the commit and the command on the Submit page.

## How we run your command

You give a command template, for example

```text
python {code_path}/spatialarena_infer.py --images {images_dir} --output {output_path} --checkpoint {ckpt_path}
```

and the pipeline ([`external_command_adapter.py`](../../scripts/inference/external_command_adapter.py)) fills in the
placeholders and runs it in a shell:

- **Where:** the command runs from our SpatialArena checkout, not from your repository. Refer to your files as
  `{code_path}/...`; `{code_path}` is your repository, checked out at the commit you submit.
- **Environment:** the Python environment built from your install command (e.g. `pip install -e .`), or your Docker
  image. `python` is that environment's interpreter.
- **When:** once per scene of every dataset of the tasks you chose. Some runs give the command only part of a scene:
  MonoDepth scenes are split into groups of images of the same size, and long scenes of some datasets are capped at
  100 frames. Handle whatever images are in `{images_dir}`.
- **Result:** a zero exit status and the files described below. The command's output goes to `log.txt` in the output
  folder; a non-zero exit fails the scene.

### Placeholders

| Placeholder | Value |
| --- | --- |
| `{images_dir}` | the scene's input images (JPEG or PNG), named as the evaluator expects them back |
| `{data_path}` | the scene folder (holds `images/`, and the ground truth your track may use) |
| `{output_path}` | the folder to write the prediction to (created for you) |
| `{code_path}` | your repository |
| `{ckpt_path}` | your checkpoint, downloaded from the link you submit |
| `{camera_dir}` | ground-truth cameras as a COLMAP model, on the camera-conditioned tracks (`rgb_cameras`, ...) |
| `{depth_dir}` | ground-truth depth maps, on the depth-conditioned tracks (`rgb_depth`, ...) |
| `{track}` | the track being run, e.g. `rgb` |
| `{resolution}` | the input resolution we use for registered methods, if your script wants a hint |

On the plain `rgb` track your command must only read the images: the scene folder also holds the ground truth.

## What the command must write

Write everything under `{output_path}`. The simplest rule: **always write the depth maps, and write the cameras if
your model predicts them**; each task reads what it needs.

| Task | Reads |
| --- | --- |
| MonoDepth, VideoDepth | `depths/` |
| SfM, SLAM | `sparse/0/` |
| MVRecon | `depths/` and `sparse/0/` |

**Depth maps** (`depths/<image stem>.npz`): one file per input image, named after the image without its extension
(`frame_000.jpg` -> `depths/frame_000.npz`), holding a float32 array named `depth` of shape (height, width).

- Depth along the camera axis, not disparity or inverse depth.
- Any unit and any resolution: the evaluator resizes the map to the ground truth and aligns its scale (and shift,
  where the protocol does) before scoring, as [the evaluation protocol](../evaluation.md) describes.
- Mark invalid pixels with 0, NaN or inf.

```python
np.savez_compressed(output / "depths" / f"{image_path.stem}.npz", depth=depth.astype(np.float32))
```

**Cameras** (`sparse/0/`): a COLMAP model (`cameras`, `images` and `points3D`, as `.txt` or `.bin`).

- Image names are the input file names (`frame_000.jpg`).
- Poses are COLMAP's world-to-camera rotation (quaternion `QW QX QY QZ`) and translation, with x right, y down and z
  forward. Any world frame and scale: the evaluator aligns the trajectory to the ground truth (Sim(3)).
- Intrinsics in pixels of the input images; `PINHOLE` (or any COLMAP camera model) per image.
- `points3D` may be empty. Give every image a camera: a missing camera counts against the completeness metrics.

The template's `write_colmap_text()` writes such a model from world-to-camera matrices and intrinsics.

**Optional:** `confidences/<image stem>.npz` (array `confidence`, higher is more confident) and a dense point cloud
`points3D_dense.ply`, which the dashboard's viewer shows.

## Start from the template

[`submission_template.py`](../../scripts/inference/submission_template.py) needs only NumPy and Pillow. It reads the
images, calls `predict()`, and writes the depth maps, confidences and COLMAP model in the layout above. Its
placeholder `predict()` returns a constant depth and made-up cameras, so the script runs as it is; replace it with
your model:

```python
def predict(images, checkpoint, known_cameras):
    model = MyModel.from_pretrained(checkpoint)
    out = model(images)                          # your inference
    return Prediction(
        depths=list(out.depth),                  # N x (H, W)
        world_to_camera=out.extrinsics,          # (N, 4, 4), or None for depth only
        intrinsics=out.intrinsics,               # (N, 3, 3) in input pixels
    )
```

Its command template is the one shown at the top of this guide.

## Check it locally

1. Download a small SpatialArena scene (11 frames for MonoDepth, 321 for SfM) and unpack it:

   ```bash
   pip install -U huggingface_hub
   hf download mvp-lab/SpatialArena-Dataset --repo-type dataset --local-dir data \
       --include "extract_benchmark_archives.py" --include "MonoDepth/spatialarena/CambridgeUpstair.tar" \
       --include "SfM/spatialarena/CambridgeUpstair.tar"
   python data/extract_benchmark_archives.py data --delete
   ```

2. Run your command the way the pipeline does, through the adapter in a SpatialArena checkout
   (`git clone https://github.com/mvp-ai-lab/SpatialArena.git && pip install -e SpatialArena`):

   ```bash
   cd SpatialArena
   python scripts/inference/external_command_adapter.py \
       --data_path ../data/SfM/spatialarena/CambridgeUpstair --output_path ../out/SfM \
       --method_name MY_MODEL --code_path /path/to/your/repo --ckpt_path /path/to/weights \
       --command 'python {code_path}/spatialarena_infer.py --images {images_dir} --output {output_path} --checkpoint {ckpt_path}'
   ```

3. Check the output:

   ```bash
   python scripts/inference/check_submission_output.py \
       --images ../data/SfM/spatialarena/CambridgeUpstair/images --output ../out/SfM
   ```

   ```text
   MonoDepth: ok
   VideoDepth: ok
   MVRecon: ok
   SfM: ok
   SLAM: ok
   ```

   It reads the files the way the evaluator does and names what is missing or misnamed.

4. Optionally, score it with the full pipeline on the SpatialArena datasets: describe your model as a custom model
   (`name`, `track`, `command`, `code_path`, `ckpt_path`) in `evaluate.py`, as in the
   [README's quick start](../../README.md#quick-start-evaluate-your-model).

## Submit

On the [Submit page](https://mvp-ai-lab.github.io/SpatialArena/#submit), give the repository and the **commit** that
contains your script, the command template, the checkpoint link, the install command, the licenses of the code and of
the weights, the GPU memory the model needs (up to 32 GB on a V100 or 128 GB on an H200; less memory gets higher
priority) and the tasks and tracks. Until our evaluation backend is live, submissions of public models (code and
weights reachable without a login) are sent by email from that page.

Before you submit:

- [ ] The script is in the repository at the submitted commit, and the command refers to it as `{code_path}/...`.
- [ ] The install command sets up everything the script imports, on a fresh environment.
- [ ] The checkpoint link downloads without a login.
- [ ] `check_submission_output.py` reports `ok` for the tasks you chose.
- [ ] The model fits in the GPU memory you chose on a full scene (hundreds of frames for SfM and SLAM).
