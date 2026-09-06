"""morel: Modality-aware recommendation via graph retrieval-enhanced completion.

Public surface is re-exported here. Library consumers should import from
``morel`` directly.
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

__version__ = "0.0.0+unknown"
try:
    from morel._version import __version__ as scm_version
except ImportError:
    scm_version = None
if scm_version is not None:
    __version__ = scm_version

if not logging.getLogger("morel").handlers:
    logging.getLogger("morel").addHandler(logging.NullHandler())


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
