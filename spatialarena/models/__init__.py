# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Inference models of the evaluation pipeline.

`BaseModel` holds the shared pre- and post-processing, the inference loop
and the artifact export. Each method subclasses it in
spatialarena/models/<method>.py and registers with `register_model`.
External packages register their own models the same way (see
`spatialarena.models.registry`). Importing this package does not import
torch.
"""

from spatialarena.models import base
from spatialarena.models import registry

BaseModel = base.BaseModel
CameraExport = base.CameraExport
ModelInput = base.ModelInput
Prediction = base.Prediction

get_model = registry.get_model
list_models = registry.list_models
load_plugins = registry.load_plugins
register_model = registry.register_model
unregister_model = registry.unregister_model
