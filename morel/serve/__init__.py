"""Public API for the morel.serve package."""

from morel.serve.app import (
    Ask,
    Tell,
    Rollback,
    Stats,
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
    Done,
    Health,
    Pick,
    RecommendRequest,
    RecommendResponse,
    serialize,
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
    "Done",
    "DefaultStp",
    "Event",
    "Ask",
    "Tell",
    "Health",
    "Loader",
    "Outcome",
    "Pick",
    "RWLock",
    "Read",
    "RecommendRequest",
    "RecommendResponse",
    "Rollback",
    "Scope",
    "Signal",
    "Stats",
    "Step",
    "Updater",
    "Write",
    "assert_set",
    "create",
    "dependency",
    "is_admin",
    "is_read",
    "reader",
    "serialize",
    "token",
    "writer",
]
