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
from morel.serve.lock import RWLock, ReadGuard, WriteGuard, reader, writer
from morel.serve.schema import (
    CompleteRequest,
    CompleteResponse,
    HealthResponse,
    RecommendItem,
    RecommendRequest,
    RecommendResponse,
    serialize_completed,
)
from morel.serve.update import DefaultLossStep, FeedbackEvent, LossStep, PipelineUpdater, Signal, UpdateResult


__all__ = [
    "CompleteRequest",
    "CompleteResponse",
    "DefaultLossStep",
    "FeedbackEvent",
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
