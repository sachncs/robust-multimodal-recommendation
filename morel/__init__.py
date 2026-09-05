"""morel: Modality-aware recommendation via graph retrieval-enhanced completion.

Public surface is re-exported here. Library consumers should import from
``morel`` directly, not from submodules.
"""

from __future__ import annotations

import logging

__version__: str
try:
    from morel._version import __version__  # type: ignore[attr-defined]
except ImportError:
    __version__ = "0.0.0+unknown"


if not logging.getLogger("morel").handlers:
    logging.getLogger("morel").addHandler(logging.NullHandler())


__all__ = ["__version__"]
