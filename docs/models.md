# Inference models (`spatialarena.models`)

Every method the benchmark runs produces the same artifacts (`depths/`, `confidences/`, `sparse/<subdir>`,
`points3D_dense.ply`, `metrics.json`; see the README). What differs between methods is only how their own code
loads, preprocesses and runs; everything around that is shared. `spatialarena.models` puts the shared part in one
base class:

| Part | Where |
|---|---|
| The command line, the same for every model (see [Command line](#command-line)) | `BaseModel.add_common_arguments` |
| Frame discovery and sampling (interval, frame cap, image subset) | `BaseModel.select_images` |
| Dry run, timers and peak GPU memory, `metrics.json` | `BaseModel.run`, `write_metrics` |
| Depth-only batch loop, single-view (MonoDepth) batches | `BaseModel.run_depth_only`, `depth_batches` |
| Restoring depth, confidence and intrinsics from the model grid to each image's source grid | `BaseModel.export_depth`, `cameras_for_export` (via `utils.restore_prediction` / `restore_camera_export`) |
| Dense points + COLMAP export | `BaseModel.run_full` (`utils.export_standard_artifacts`) |
| Resize / crop / pad planning, prediction-map writers, depth and camera priors | `spatialarena/models/utils.py` |

A method implements three hooks and registers itself:

```python
from spatialarena import models
from spatialarena.models import utils


@models.register_model
class MyModel(models.BaseModel):
    """My model."""

    name = "MY_MODEL"  # Registry name and "method" in metrics.json.
    code_path_env = "MY_MODEL_CODE_DIR"
    ckpt_default = "org/my-model"
    default_resolution = 518
    spatial_policies = ("shared",)  # Restorable resize and padding.

    def load(self, device):
        import my_model  # The method's own code, imported only here.

        network = my_model.Net.from_pretrained(self.args.ckpt_path)
        return network.to(device).eval()

    def preprocess(self, image_paths, batch_index=None):
        images, transforms = utils.preprocess_images(
            image_paths, method="vggt", size=self.args.resolution)
        return models.ModelInput(image_paths, images.to(self.device),
                                 transforms, batch_index)

    def forward(self, inputs):
        output = self.model(inputs.data)
        return models.Prediction(
            depth=output["depth"].cpu().numpy(),
            confidence=output["conf"].cpu().numpy(),
            extrinsics_w2c=output["w2c"].cpu().numpy(),
            intrinsics=output["K"].cpu().numpy(),
        )
```

`Prediction` fields are frame-major numpy arrays on the **model grid**. When `ModelInput.transforms` describes how
each image reached that grid, the base class restores depth, confidence and intrinsics to the source images.
Otherwise (a method's own legacy resize) they are exported as returned, or through `export_depth_legacy`. Further
hooks (`postprocess`, `select_images`, `export_depth_legacy`, `legacy_camera_size`, `metrics_extra`, `run_full`,
...) exist for the places where a method really differs. The built-in models in `spatialarena/models/` show each
one: VGGT's legacy padding crop, DA3's camera priors and own dense export, Pi3X's depth- and camera-conditioned
tracks.

## Command line

Every model has the same command line, built by `BaseModel.add_common_arguments`; a model adds only options that
exist for it alone (`add_arguments`, e.g. FastVGGT's `--merging`, the long-sequence SLAM `--phase`).

| Options | Which models |
|---|---|
| `--data_path --output_path --ckpt_path --track --interval --max_images_per_run --export_depth --export_colmap --depth_only --colmap_model_subdir --keep_intermediate --dry_run` | all |
| `--code_path` | all with a code checkout (`uses_code_path`; not the API VLMs, GenCeption, Marigold v1) |
| `--resolution` | those with a model input resolution (`uses_resolution`) |
| `--image_load_resolution` | those that load images at a resolution of their own (`uses_image_load_resolution`) |
| `--max_points --conf_thresh --depth_conf_thresh` | those with a dense point export (`exports_points`) |
| `--colmap_path` | those with a camera-prior track (from the MethodSpec's tracks) |
| `--spatial_policy` | those with restorable resize policies (`spatial_policies`) |
| `--seed` | the diffusion and single-image models (`default_seed`) |

The **method name selects the model and its variant**: `run_model.py --model DA3-G` runs the DA3 model as DA3-G
(there is no `--method` / `--method_name` / `--task` option). The defaults come from the method's `MethodSpec` in
`spatialarena/methods.py`, the values the engine passes as well: `--ckpt_path` and `--code_path` from its
`ckpt_env` / `code_env` variables, else `ckpt_default` / `code_default` (with `$METHOD_CODE_ROOT` expanded),
`--track` choices from `tracks`, `--resolution`, `--image_load_resolution`, `--conf_thresh` and model-specific
fields such as Pi3's `--pixel_limit`. A model without a MethodSpec falls back to its `default_*` class attributes.
`--depth_conf_thresh` is unset by default and overrides `--conf_thresh` when given.

Run a model through the generic entry point, or through its own script in `scripts/inference/` (the same
command line; a script that serves several methods, e.g. `moge.py`, takes `--model` too):

```bash
python scripts/inference/run_model.py --model VGGT --data_path SCENE --output_path OUT --export_depth
python scripts/inference/run_model.py --model DA3-G --help
python scripts/inference/moge.py --model MOGE2 --data_path SCENE --output_path OUT --export_depth --depth_only
```

The evaluation engine (`spatialarena/templates/_engine/run_method_case_list.sh`) runs every method the same way,
`run_model.py --engine --model METHOD` with one option set; `--engine` drops the options in
`run_model.ENGINE_OPTIONS` that a model does not have. Model-specific settings that are not options (e.g.
`GENCEPTION_MODEL_SIZE`, `COME_MODEL`, `POINT_QUERY_DEPTH_GRID_SIZE`) are read from the environment, which the
engine exports.

To run it from Python:

```python
from spatialarena import models

model_cls = models.get_model("VGGT")
model = model_cls.from_argv(["--data_path", scene, "--output_path", out,
                             "--spatial_policy", "shared"])
model.run()  # The whole adapter run, as the engine does it.

prediction, inputs = model.predict(paths)  # One model call, model grid.
depths = model.restore_depths(prediction, inputs)  # Source image sizes.
```

## Evaluating your own model with SpatialArena as a dependency

Register the model with a `MethodSpec` as well. That adds it to the method registry, so the engine and
`BenchmarkEvaluator` can plan and run it like a built-in method:

```python
from spatialarena import methods
from spatialarena import models


@models.register_model(spec=methods.MethodSpec(
    name="MY_MODEL",
    adapter="run_model.py",
    code_env="MY_MODEL_CODE_DIR",
    code_default="",
    ckpt_env="MY_MODEL_CKPT_PATH",
    native=True,
    tracks=(methods.TRACK_RGB,),
    input_resolution=518,
))
class MyModel(models.BaseModel):
    ...
```

The engine calls every registered model as `scripts/inference/run_model.py --engine --model MY_MODEL` with the
usual options (`adapter="run_model.py"` names the entry point). The engine and its job scripts are separate processes, so they import your module by one of:

* an entry point in your package's `pyproject.toml`:

  ```toml
  [project.entry-points."spatialarena.models"]
  my_model = "my_package.my_model"
  ```

* or `SPATIALARENA_MODEL_PLUGINS=my_package.my_model` (comma-separated module names or `.py` paths) in the job
  environment, e.g. a model's `env:` in the run config.

Then `python scripts/test/benchmark_methods.py --method MY_MODEL --field adapter` prints `run_model.py`, and a run
config lists it like any other model (`models: [{name: MY_MODEL, track: rgb}]`).

`register_model` refuses a name that is already taken (pass `override=True` to replace a built-in model on purpose).
`get_model("my_package.my_model:MyModel")` also loads an unregistered class directly.

## Status

Every method in `spatialarena/methods.py` runs on `BaseModel`; `models.list_models()` lists them, and
`models.get_model()` accepts every name and alias the method registry does. The modules, by family:

| Family | Modules | Notes |
|---|---|---|
| VGGT-style multi-view | `vggt`, `fastvggt`, `vggt_omega`, `pi3`, `da3`, `amb3r`, `cut3r`, `fast3r`, `come`, `iggt`, `iggt4d`, `omnivggt`, `dejaview` | Co-Me runs its own Hydra entry point in a subprocess. |
| MapAnything and its wrappers | `mapanything`, `mapanything_wrappers` (DUSt3R, MASt3R, Pow3R) | |
| Single-image MonoDepth | `single_image` (family base), `moge`, `marigold`, `marigold_v2`, `lotus`, `depth_pro`, `unidepth`, `infinidepth`, `sensenova_vision` | Marigold V2 runs the official script in its own interpreter. |
| Video depth | `dvd`, `video_diffusion_depth` (DepthCrafter, ChronoDepth), `genception` | |
| VLM depth | `point_query` (VLM3, DepthLM, Qwen3-VL), `depthvlm`, `openrouter_vlm` | |
| SLAM | `slam` (family base), `streaming_slam`, `long_sequence_slam`, `slam_depth` | Whole sequences: `run()` only, no `predict()`. |

Methods whose official pipeline is a whole-scene or whole-sequence program (the SLAM systems, the video models, the
subprocess-based ones) override `run()` and keep their own command line; the others use the base pipeline. The
`scripts/inference/*.py` entry points keep their names and command lines, so the engine calls them unchanged; a few
(`single_image_depth`, `point_query_depth_adapter`, `openrouter_vlm_depth`, `slam_depth_export`,
`mapanything_dense_wrapper`, `adapter_utils`) remain as aliases of the new modules for code that imports them.
`fixed_input_launcher.py` and `long_sequence_phase_launcher.py` stay scripts: they run inside a method's own
interpreter.

Each migration was checked against the pre-refactor adapter on identical inputs with stand-in method code, output file
by output file. `scripts/test/compare_prediction_runs.py` compares a rerun on real data with released predictions and
scores (`configs/benchmark/model_refactor_check.yaml`).
