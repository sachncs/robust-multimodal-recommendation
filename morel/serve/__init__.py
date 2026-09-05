"""Public API for the morel.serve package."""

from morel.serve.app import create
from morel.serve.auth import assert_configured as auth_assert_configured
from morel.serve.auth import enabled as auth_enabled
from morel.serve.loader import Loader
from morel.serve.schema import (
    CompleteRequest,
    CompleteResponse,
    HealthResponse,
    RecommendItem,
    RecommendRequest,
    RecommendResponse,
    serialize_completed,
)

__all__ = [
    "CompleteRequest",
    "CompleteResponse",
    "HealthResponse",
    "Loader",
    "RecommendItem",
    "RecommendRequest",
    "RecommendResponse",
    "auth_assert_configured",
    "auth_enabled",
    "create",
    "serialize_completed",
]
