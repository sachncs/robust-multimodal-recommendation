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
    assert_set,
    dependency,
    is_admin,
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
    DefaultStp,
    Event,
    Outcome,
    Signal,
    Step,
    Updater,
)

__all__ = [
    "CompleteRequest",
    "CompleteResponse",
    "DefaultStp",
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
    "assert_set",
    "create",
    "dependency",
    "is_admin",
    "is_read",
    "reader",
    "serialize_completed",
    "token",
    "writer",
]
