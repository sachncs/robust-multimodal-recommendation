"""Marker for the morel.core package."""

from morel.core.config import Config
from morel.core.device import Device, device, to
from morel.core.distributed import (
    Cluster,
    barrier,
    cleanup,
    init,
    initialized,
    is_lead,
    local,
    mean,
    rank,
    size,
    state,
)
from morel.core.errors import (
    Cfg,
    Datum,
    Determinism,
    Rate,
    Net,
    Model,
    Error,
    Shape,
    Train,
)
from morel.core.fidelity import (
    Entry as FidelityEntry,
)
from morel.core.fidelity import (
    Status as FidelityStatus,
)
from morel.core.fidelity import (
    all as fidelity_all,
)
from morel.core.fidelity import (
    clear as fidelity_clear,
)
from morel.core.fidelity import (
    register as fidelity_register,
)
from morel.core.fidelity import (
    render as fidelity_render_json,
)
from morel.core.fidelity import (
    render as fidelity_render_markdown,
)
from morel.core.log import configure as configure_log
from morel.core.log import get as get_logger
from morel.core.log import log as log_metrics
from morel.core.path import (
    checkpoints,
    features,
    graphs,
    manifest,
    processed,
    raw,
    root,
    runs,
)
from morel.core.seed import State as SeedState
from morel.core.seed import deterministic
from morel.core.seed import restore as seed_restore
from morel.core.seed import seed as seed_everything
from morel.core.seed import state as seed_state
from morel.core.types import Embedding, Graph, Mask, Modality

__all__ = [
    "Cluster",
    "Config",
    "Cfg",
    "Datum",
    "Determinism",
    "Device",
    "Embedding",
    "Rate",
    "FidelityEntry",
    "FidelityStatus",
    "Graph",
    "Net",
    "Mask",
    "Modality",
    "Model",
    "Error",
    "SeedState",
    "Shape",
    "Train",
    "barrier",
    "checkpoints",
    "cleanup",
    "configure_log",
    "deterministic",
    "device",
    "features",
    "fidelity_all",
    "fidelity_clear",
    "fidelity_register",
    "fidelity_render_json",
    "fidelity_render_markdown",
    "get_logger",
    "graphs",
    "init",
    "initialized",
    "is_lead",
    "local",
    "log_metrics",
    "manifest",
    "mean",
    "processed",
    "rank",
    "raw",
    "root",
    "runs",
    "seed_everything",
    "seed_restore",
    "seed_state",
    "size",
    "state",
    "to",
]


# Register every paper-component fidelity entry on import. The call is
# idempotent so callers may also invoke it explicitly.
from morel.core import fidelity as fidelity_mod

fidelity_mod.register_all()
