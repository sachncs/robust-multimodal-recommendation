"""morel: Modality-aware recommendation via graph retrieval-enhanced completion.

Public surface is re-exported here. Library consumers should import from
``morel`` directly when possible.
"""

from __future__ import annotations

import logging

from morel.core import (
    Config,
    Embedding,
    Graph,
    Mask,
    Modality,
    configure_log,
    get_logger,
    seed_everything,
)
from morel.data.manifest import Manifest
from morel.pipeline import Output, Pipeline

__version__: str
try:
    from morel._version import __version__  # type: ignore[attr-defined]
except ImportError:
    __version__ = "0.0.0+unknown"


if not logging.getLogger("morel").handlers:
    logging.getLogger("morel").addHandler(logging.NullHandler())


LAZY_EXPORTS = {
    "Modality": "morel.core.types",
    "Mask": "morel.core.types",
    "Graph": "morel.core.types",
    "Embedding": "morel.core.types",
    "Device": "morel.core.device",
    "device": "morel.core.device",
    "MorelError": "morel.core.errors",
    "DataError": "morel.core.errors",
    "ConfigError": "morel.core.errors",
    "ModelError": "morel.core.errors",
    "GraphError": "morel.core.errors",
    "TrainError": "morel.core.errors",
    "EvalError": "morel.core.errors",
    "ShapeError": "morel.core.errors",
    "DeterminismError": "morel.core.errors",
}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'morel' has no attribute {name!r}")


__all__ = [
    "Config",
    "Embedding",
    "Graph",
    "Manifest",
    "Mask",
    "Modality",
    "Output",
    "Pipeline",
    "configure_log",
    "get_logger",
    "seed_everything",
]
