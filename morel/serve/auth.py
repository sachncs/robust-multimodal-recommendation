"""Bearer-token authentication with separate read and admin scopes.

The serve stack now distinguishes:

- ``MOREL_AUTH_TOKEN_READ`` gates the public endpoints ``/v1/complete`` and
  ``/v1/recommend``.
- ``MOREL_AUTH_TOKEN_ADMIN`` gates ``/v1/feedback``, ``/v1/rollback``, and
  ``/v1/stats``.
- ``MOREL_AUTH_TOKEN`` is preserved for backwards compatibility and is
  treated as covering both scopes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal

from fastapi import HTTPException, Request

from morel.core.errors import ConfigError

Scope = Literal["read", "admin"]


def admin_enabled() -> bool:
    """Return whether admin-scope auth is configured."""
    return bool(os.environ.get("MOREL_AUTH_TOKEN_ADMIN") or os.environ.get("MOREL_AUTH_TOKEN"))


def read_enabled() -> bool:
    """Return whether read-scope auth is configured."""
    return bool(os.environ.get("MOREL_AUTH_TOKEN_READ") or os.environ.get("MOREL_AUTH_TOKEN"))


def token_for_scope(scope: Scope) -> str | None:
    """Return the configured token for ``scope``, or ``None`` if auth is off."""
    legacy = os.environ.get("MOREL_AUTH_TOKEN", "").strip()
    if scope == "admin":
        explicit = os.environ.get("MOREL_AUTH_TOKEN_ADMIN", "").strip()
    else:
        explicit = os.environ.get("MOREL_AUTH_TOKEN_READ", "").strip()
    return explicit or (legacy or None)


def require(request: Request, scope: Scope = "read") -> None:
    """Validate the bearer token for the requested scope.

    No-op when no token is configured for ``scope``.

    Args
    ----
    request : Request
        Incoming FastAPI request.
    scope : Scope
        Which scope to enforce. ``"admin"`` for feedback/rollback/stats,
        ``"read"`` for complete/recommend.

    Raises
    ------
    HTTPException
        401 if the token is missing or wrong.
    """
    expected = token_for_scope(scope)
    if not expected:
        return
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail=f"missing bearer token for {scope}")
    presented = header.removeprefix("Bearer ").strip()
    if presented != expected:
        raise HTTPException(status_code=401, detail=f"invalid bearer token for {scope}")


def dependency(scope: Scope) -> Callable[[Request], None]:
    """Return a FastAPI-compatible dependency callable for the given scope.

    The returned callable preserves the ``(request: Request) -> None``
    signature so FastAPI can introspect the parameter list and inject the
    incoming request.
    """

    def scoped_dependency(request: Request) -> None:
        require(request, scope=scope)

    return scoped_dependency


def assert_configured() -> None:
    """Raise if a deployment attempted to enable auth without setting any token."""
    if os.environ.get("MOREL_AUTH_ENABLED") == "1" and not (admin_enabled() or read_enabled()):
        raise ConfigError("MOREL_AUTH_ENABLED=1 requires MOREL_AUTH_TOKEN[_READ|_ADMIN]")


__all__ = [
    "Scope",
    "admin_enabled",
    "assert_configured",
    "dependency",
    "read_enabled",
    "require",
    "token_for_scope",
]
