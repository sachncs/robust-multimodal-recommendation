"""Bearer-token authentication middleware."""

from __future__ import annotations

import os

from fastapi import HTTPException, Request

from morel.core.errors import ConfigError


def enabled() -> bool:
    """Return whether bearer-token auth is enabled."""
    value = os.environ.get("MOREL_AUTH_TOKEN", "").strip()
    return bool(value)


def require(request: Request) -> None:
    """Validate the bearer token if auth is enabled.

    Raises
    ------
        HTTPException: 401 if the token is missing or wrong.
    """
    token = os.environ.get("MOREL_AUTH_TOKEN", "").strip()
    if not token:
        return
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    presented = header.removeprefix("Bearer ").strip()
    if presented != token:
        raise HTTPException(status_code=401, detail="invalid bearer token")


def assert_configured() -> None:
    """Raise if a deployment attempted to enable auth without setting the token."""
    if os.environ.get("MOREL_AUTH_ENABLED") == "1" and not enabled():
        raise ConfigError("MOREL_AUTH_ENABLED=1 requires MOREL_AUTH_TOKEN")


__all__ = ["enabled", "require", "assert_configured"]
