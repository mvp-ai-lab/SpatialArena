# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Registry of inference models: name -> `base.BaseModel` subclass.

Built-in models are imported on first lookup, so importing this module
imports neither torch nor any method's code.

An external model registers the same way as the built-in ones:

    from spatialarena import methods
    from spatialarena import models

    @models.register_model(spec=methods.MethodSpec(
        name="MY_MODEL", adapter="run_model.py", code_env="MY_MODEL_CODE",
        code_default="", native=True))
    class MyModel(models.BaseModel):
        name = "MY_MODEL"
        ...

`spec` also adds the method to `spatialarena.methods`, so the evaluation
engine can plan and run it. The engine runs in other processes, which find
the model module through the "spatialarena.models" entry-point group or the
SPATIALARENA_MODEL_PLUGINS environment variable (comma-separated module
names or .py paths); see `load_plugins`.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import pathlib

PLUGIN_ENV = "SPATIALARENA_MODEL_PLUGINS"
ENTRY_POINT_GROUP = "spatialarena.models"

# Built-in models: registry key -> "module:Class", imported on first lookup.
BUILTIN_MODELS = {
    "VGGT": "spatialarena.models.vggt:VGGT",
    "ROBUSTVGGT": "spatialarena.models.vggt:RobustVGGT",
    "FASTVGGT": "spatialarena.models.fastvggt:FastVGGT",
    "VGGT_OMEGA": "spatialarena.models.vggt_omega:VGGTOmega",
    "PI3_X": "spatialarena.models.pi3:Pi3",
    "DA3": "spatialarena.models.da3:DA3",
    "AMB3R": "spatialarena.models.amb3r:AMB3R",
    "CUT3R": "spatialarena.models.cut3r:CUT3R",
    "FAST3R": "spatialarena.models.fast3r:Fast3R",
    "COME": "spatialarena.models.come:Come",
    "IGGT": "spatialarena.models.iggt:IGGT",
    "IGGT4D": "spatialarena.models.iggt4d:IGGT4D",
    "OMNIVGGT": "spatialarena.models.omnivggt:OmniVGGT",
    "DEJAVIEW": "spatialarena.models.dejaview:DejaView",
    "DUST3R": "spatialarena.models.mapanything_wrappers:DUSt3R",
    "MAST3R": "spatialarena.models.mapanything_wrappers:MASt3R",
    "POW3R": "spatialarena.models.mapanything_wrappers:Pow3R",
    "MAPANYTHING": "spatialarena.models.mapanything:MapAnything",
    "MOGE": "spatialarena.models.moge:MoGe",
    "MOGE2": "spatialarena.models.moge:MoGe2",
    "MOGE3": "spatialarena.models.moge:MoGe3",
    "MARIGOLD": "spatialarena.models.marigold:Marigold",
    "MARIGOLD_V1_1": "spatialarena.models.marigold:MarigoldV11",
    "MARIGOLD_V2": "spatialarena.models.marigold_v2:MarigoldV2",
    "LOTUS_D": "spatialarena.models.lotus:LotusD",
    "DEPTH_PRO": "spatialarena.models.depth_pro:DepthPro",
    "UNIDEPTH_V2": "spatialarena.models.unidepth:UniDepthV2",
    "INFINIDEPTH": "spatialarena.models.infinidepth:InfiniDepth",
    "SENSENOVA_VISION": "spatialarena.models.sensenova_vision:SenseNovaVision",
    "VLM3": "spatialarena.models.point_query:VLM3",
    "DEPTHLM": "spatialarena.models.point_query:DepthLM",
    "QWEN3_VL": "spatialarena.models.point_query:Qwen3VL",
    "DEPTHVLM": "spatialarena.models.depthvlm:DepthVLM",
    "CLAUDE_OPUS_5_5": "spatialarena.models.openrouter_vlm:ClaudeOpus55",
    "GPT_6_ASTRA": "spatialarena.models.openrouter_vlm:Gpt6Astra",
    "GEMINI_4_ARGON": "spatialarena.models.openrouter_vlm:Gemini4Argon",
    "LOGER": "spatialarena.models.streaming_slam:LoGeR",
    "ZIPMAP": "spatialarena.models.streaming_slam:ZipMap",
    "LINGBOT_MAP": "spatialarena.models.streaming_slam:LingBotMap",
    "R3": "spatialarena.models.streaming_slam:R3",
    "SCAL3R": "spatialarena.models.streaming_slam:Scal3R",
    "ABOT_RECON": "spatialarena.models.streaming_slam:ABotRecon",
    "VGGT_LONG": "spatialarena.models.long_sequence_slam:VGGTLong",
    "PI_LONG": "spatialarena.models.long_sequence_slam:PiLong",
    "VGGT_SLAM": "spatialarena.models.long_sequence_slam:VGGTSlam",
    "DVD": "spatialarena.models.dvd:DVD",
    "DEPTHCRAFTER": "spatialarena.models.video_diffusion_depth:DepthCrafter",
    "CHRONODEPTH": "spatialarena.models.video_diffusion_depth:ChronoDepth",
    "GENCEPTION_DEPTH": "spatialarena.models.genception:GenCeptionDepth",
    "GENCEPTION_POSE": "spatialarena.models.genception:GenCeptionPose",
}
BUILTIN_ALIASES = {
    "PI3": "PI3_X",
    "DA3_L": "DA3",
    "DA3_G": "DA3",
    "DA3_NGL": "DA3",
    "AMB3R_BASE": "AMB3R",
    "OMNI_VGGT": "OMNIVGGT",
    "DEJA_VIEW": "DEJAVIEW",
    "DVLT": "DEJAVIEW",
    "MAP_ANYTHING": "MAPANYTHING",
    "MOGE1": "MOGE",
    "MOGE_1": "MOGE",
    "MOGE_2": "MOGE2",
    "MOGE_3": "MOGE3",
    "LOTUS": "LOTUS_D",
    "DEPTHPRO": "DEPTH_PRO",
    "UNIDEPTH": "UNIDEPTH_V2",
    "UNIDEPTHV2": "UNIDEPTH_V2",
    "INFINI_DEPTH": "INFINIDEPTH",
    "SENSENOVA": "SENSENOVA_VISION",
    "SENSE_NOVA_VISION": "SENSENOVA_VISION",
    "PI3_LONG": "PI_LONG",
}

_registry = {}
_aliases = {}
_loaded_plugins = set()


def _key(name):
    """Returns the registry key of a model name: upper case, underscores."""
    return name.upper().replace("-", "_").replace(" ", "_")


def register_model(cls=None, name=None, aliases=(), spec=None, override=False):
    """Registers a `base.BaseModel` subclass.

    Use it as `@register_model` or `@register_model(name=..., spec=...)`.

    Args:
        cls: The class, when used without arguments.
        name: Registry name; defaults to the class's `name`.
        aliases: Further names.
        spec: Optional `methods.MethodSpec` to add to the method registry.
        override: Replace a model already registered under the name.

    Returns:
        The class, or a decorator when `cls` is None.

    Raises:
        TypeError: `cls` is not a `base.BaseModel` subclass.
        ValueError: The class has no name, or the name is taken.
    """

    def decorate(model_cls):
        _register(model_cls, name, aliases, spec, override)
        return model_cls

    if cls is None:
        return decorate
    return decorate(cls)


def _register(model_cls, name, aliases, spec, override):
    # pylint: disable-next=import-outside-toplevel  # avoids an import cycle
    from spatialarena.models import base

    if not (isinstance(model_cls, type) and
            issubclass(model_cls, base.BaseModel)):
        raise TypeError(
            f"register_model expects a BaseModel subclass, got {model_cls!r}")
    model_name = name or model_cls.name
    if not model_name:
        raise ValueError(f"{model_cls.__name__} has no name; set the class "
                         "attribute `name` or pass name=")

    key = _key(model_name)
    existing = _registry.get(key)
    if existing is not None and existing is not model_cls and not override:
        raise ValueError(f"Model {model_name!r} is already registered "
                         f"({existing.__module__}.{existing.__qualname__}); "
                         "pass override=True to replace it")
    _registry[key] = model_cls
    for alias in (*model_cls.aliases, *aliases):
        _aliases[_key(alias)] = key

    if spec is not None:
        # pylint: disable-next=import-outside-toplevel  # optional dependency
        from spatialarena import methods

        methods.register_method(spec, override=override)


def unregister_model(name):
    """Removes a registered model and the aliases pointing at it."""
    key = _key(name)
    _registry.pop(key, None)
    for alias in [a for a, target in _aliases.items() if target == key]:
        del _aliases[alias]


def _import_target(target):
    """Imports "module:attribute" (or a module) and returns it."""
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    if attribute:
        return getattr(module, attribute)
    return module


def _import_file(path):
    """Imports a .py file as a module."""
    module_name = f"spatialarena_model_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model plugin {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _import_plugin(entry):
    entry = entry.strip()
    if not entry or entry in _loaded_plugins:
        return
    _loaded_plugins.add(entry)
    path = pathlib.Path(entry).expanduser()
    if entry.endswith(".py") and path.is_file():
        _import_file(path)
    else:
        _import_target(entry)


def load_plugins():
    """Imports the external model modules (each one only once).

    They come from the "spatialarena.models" entry-point group of installed
    packages and from SPATIALARENA_MODEL_PLUGINS.
    """
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.value not in _loaded_plugins:
            _loaded_plugins.add(entry_point.value)
            entry_point.load()
    for entry in os.environ.get(PLUGIN_ENV, "").split(","):
        _import_plugin(entry)


def _resolve_key(name):
    key = _key(name)
    return _aliases.get(key, BUILTIN_ALIASES.get(key, key))


def _method_key(name):
    """Returns the registry key of a method alias from spatialarena.methods.

    The method registry lists every accepted spelling of a method name
    (e.g. "DVD-V1.1", "PI3X"); this maps them onto the model registry
    without repeating them here.

    Args:
        name: A method name or alias.

    Returns:
        The key of the method's canonical name, or None when no method has
        this name.
    """
    # pylint: disable-next=import-outside-toplevel  # avoids an import cycle
    from spatialarena import methods

    try:
        spec = methods.resolve_method(name)
    except (KeyError, ValueError):
        return None
    return _resolve_key(spec.name)


def get_model(name):
    """Returns the model class registered under a name or alias.

    A built-in model is imported on first use, and the plugins are loaded
    when the name is not found otherwise.

    Args:
        name: Registry name or alias, or "package.module:Class" for an
            unregistered class.

    Returns:
        The `base.BaseModel` subclass.

    Raises:
        KeyError: No model has this name.
    """
    if ":" in name:
        model_cls = _import_target(name)
        register_model(model_cls)
        return model_cls

    key = _resolve_key(name)
    if key not in _registry and key not in BUILTIN_MODELS:
        key = _method_key(name) or key
    if key not in _registry and key in BUILTIN_MODELS:
        _import_target(BUILTIN_MODELS[key])  # The module registers itself.
    if key not in _registry:
        load_plugins()
        key = _resolve_key(name)
        if key not in _registry:
            key = _method_key(name) or key
    if key not in _registry:
        raise KeyError(
            f"Unknown model {name!r}; registered: {', '.join(list_models())}")
    return _registry[key]


def list_models():
    """Returns the names of the built-in and registered models."""
    return sorted(set(_registry) | set(BUILTIN_MODELS))
