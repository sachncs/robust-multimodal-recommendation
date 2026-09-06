"""Public API for the morel.serve package."""

from morel.serve.app import (
    FeedbackRequest,
    FeedbackResponse,
    RollbackResponse,
    StatsResponse,
    create,
)
from morel.serve.auth import (
    Scope,
    admin_enabled,
    assert_configured,
    dependency,
    read_enabled,
    token_for_scope,
)
from morel.serve.loader import Loader
from morel.serve.lock import ReadGuard, RWLock, WriteGuard, reader, writer
from morel.serve.schema import (
    CompleteRequest,
    CompleteResponse,
    HealthResponse,
    RecommendItem,
    RecommendRequest,
    RecommendResponse,
    serialize_completed,
)
from morel.serve.update import (
    DefaultStep,
    Event,
    LossStep,
    PipelineUpdater,
    Signal,
    UpdateResult,
)

__all__ = [
    "CompleteRequest",
    "CompleteResponse",
    "DefaultStep",
    "Event",
    "FeedbackRequest",
    "FeedbackResponse",
    "HealthResponse",
    "Loader",
    "LossStep",
    "PipelineUpdater",
    "RWLock",
    "ReadGuard",
    "RecommendItem",
    "RecommendRequest",
    "RecommendResponse",
    "RollbackResponse",
    "Scope",
    "Signal",
    "StatsResponse",
    "UpdateResult",
    "WriteGuard",
    "admin_enabled",
    "assert_configured",
    "create",
    "dependency",
    "read_enabled",
    "reader",
    "serialize_completed",
    "token_for_scope",
    "writer",
]
