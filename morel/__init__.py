"""morel: Modality-aware recommendation via graph retrieval-enhanced completion.

Public surface is re-exported here. Library consumers should import from
``morel`` directly, not from submodules.
"""

from __future__ import annotations

import logging
import os
import sys

if sys.version_info < (3, 10):
    raise RuntimeError("morel requires Python 3.10 or newer")

__version__: str
try:
    from morel._version import __version__  # type: ignore[attr-defined]
except ImportError:
    __version__ = "0.0.0+unknown"


if not logging.getLogger("morel").handlers:
    handler = logging.NullHandler()
    logging.getLogger("morel").addHandler(handler)


def _setup_path() -> None:
    """Add the package root to sys.path for CLI entry points."""
    if "MOREL_DATA_DIR" not in os.environ:
        os.environ.setdefault("MOREL_DATA_DIR", "data")


_setup_path()


__all__ = ["__version__"]
