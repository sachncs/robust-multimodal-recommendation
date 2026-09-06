"""Marker for the morel.core package."""

from morel.core.config import Config
from morel.core.device import Device, device, to
from morel.core.distributed import (
    DistributedState,
    barrier,
    cleanup,
    init,
    is_initialized,
    is_rank_zero,
    local_rank,
    rank,
    reduce_mean,
    state,
    world_size,
)
from morel.core.errors import (
    ConfigError,
    DataError,
    DeterminismError,
    EvalError,
    GraphError,
    ModelError,
    MorelError,
    ShapeError,
    TrainError,
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
    render_json as fidelity_render_json,
)
from morel.core.fidelity import (
    render_markdown as fidelity_render_markdown,
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
from morel.core.seed import restore as seed_restore
from morel.core.seed import seed as seed_everything
from morel.core.seed import state as seed_state
from morel.core.types import Embedding, Graph, Mask, Modality

__all__ = [
    "Config",
    "ConfigError",
    "DataError",
    "DeterminismError",
    "Device",
    "DistributedState",
    "Embedding",
    "EvalError",
    "FidelityEntry",
    "FidelityStatus",
    "Graph",
    "GraphError",
    "Mask",
    "Modality",
    "ModelError",
    "MorelError",
    "SeedState",
    "ShapeError",
    "TrainError",
    "barrier",
    "checkpoints",
    "cleanup",
    "configure_log",
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
    "is_initialized",
    "is_rank_zero",
    "local_rank",
    "log_metrics",
    "manifest",
    "processed",
    "rank",
    "raw",
    "reduce_mean",
    "root",
    "runs",
    "seed_everything",
    "seed_restore",
    "seed_state",
    "state",
    "to",
    "world_size",
]


# Register every paper-component fidelity entry on import. The call is
# idempotent so callers may also invoke it explicitly.
from morel.core import fidelity_registry

fidelity_registry.register_all()
