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
    is_admin,
    assert_set,
    dependency,
    is_read,
    token,
)
from morel.serve.loader import Loader
from morel.serve.lock import Read, RWLock, Write, reader, writer
from morel.serve.schema import (
    CompleteRequest,
    CompleteResponse,
    HealthResponse,
    Pick,
    RecommendRequest,
    RecommendResponse,
    serialize_completed,
)
from morel.serve.update import (
    DefaultStep,
    Event,
    Outcome,
    Signal,
    Step,
    Updater,
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
    "Outcome",
    "Pick",
    "RWLock",
    "Read",
    "RecommendRequest",
    "RecommendResponse",
    "RollbackResponse",
    "Scope",
    "Signal",
    "StatsResponse",
    "Step",
    "Updater",
    "Write",
    "is_admin",
    "assert_set",
    "create",
    "dependency",
    "is_read",
    "reader",
    "serialize_completed",
    "token",
    "writer",
]
