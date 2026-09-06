"""Public API for the morel.serve package."""

from morel.serve.app import (
    FeedbackRequest,
    FeedbackResponse,
    RollbackResponse,
    StatsResponse,
    create,
)
from morel.serve.auth import (
    admin_enabled,
    assert_configured,
    read_enabled,
    token_for_scope,
)
from morel.serve.loader import Loader
from morel.serve.lock import RWLock, reader, writer
from morel.serve.schema import (
    CompleteRequest,
    CompleteResponse,
    HealthResponse,
    RecommendItem,
    RecommendRequest,
    RecommendResponse,
    serialize_completed,
)
from morel.serve.update import FeedbackEvent, PipelineUpdater, Signal, UpdateResult

# Backwards-compat aliases under the old prefix-style names.
auth_assert_configured = assert_configured
auth_admin_enabled = admin_enabled
auth_read_enabled = read_enabled
auth_token_for_scope = token_for_scope


__all__ = [
    "CompleteRequest",
    "CompleteResponse",
    "FeedbackEvent",
    "FeedbackRequest",
    "FeedbackResponse",
    "HealthResponse",
    "Loader",
    "PipelineUpdater",
    "RWLock",
    "RecommendItem",
    "RecommendRequest",
    "RecommendResponse",
    "RollbackResponse",
    "Signal",
    "StatsResponse",
    "UpdateResult",
    "admin_enabled",
    "assert_configured",
    "auth_admin_enabled",
    "auth_assert_configured",
    "auth_read_enabled",
    "auth_token_for_scope",
    "create",
    "read_enabled",
    "reader",
    "serialize_completed",
    "token_for_scope",
    "writer",
]
