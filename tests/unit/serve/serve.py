"""Tests for morel.serve."""

from __future__ import annotations

import httpx
import pytest

from morel.serve import Loader, create


async def get(client: httpx.AsyncClient, path: str, **kwargs) -> httpx.Response:
    """Issue a GET and return the response."""
    return await client.get(path, **kwargs)


async def post(client: httpx.AsyncClient, path: str, **kwargs) -> httpx.Response:
    """Issue a POST and return the response."""
    return await client.post(path, **kwargs)


def make(app) -> httpx.AsyncClient:
    """Build an async httpx test client.

    Uses ``httpx.ASGITransport`` directly, which dispatches through the
    ASGI app without going through anyio.abc.BlockingPortal (deprecated
    by anyio 4).
    """
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


class Checker:
    """Aggregated test methods for this module."""

    def capacity() -> None:
        with pytest.raises(ValueError, match="capacity must be positive"):
            Loader(capacity=0)

    def caches() -> None:
        loader = Loader(capacity=2)
        calls = []

        def factory():
            calls.append(1)
            return object()

        a = loader.get("k1", factory)
        b = loader.get("k1", factory)
        assert a is b
        assert len(calls) == 1

    def evicts() -> None:
        loader = Loader(capacity=2)
        loader.get("a", lambda: object())
        loader.get("b", lambda: object())
        loader.get("c", lambda: object())
        assert "a" not in loader.cache

    def key() -> None:
        loader = Loader()
        with pytest.raises(ValueError, match="key must be a non-empty string"):
            loader.get("", lambda: object())


class Checker:
    """Aggregated test methods for this module."""


class Checker:
    """Aggregated test methods for this module."""


class Checker:
    """Aggregated test methods for this module."""


class Checker:
    """Aggregated test methods for this module."""


class Checker:
    """Aggregated test methods for this module."""

async def test_health_endpoint() -> None:
                            async with make(create()) as client:
                                r = await get(client, "/health")
                            assert r.status_code == 200
                            assert r.json()["status"] == "ok"

async def test_metrics_endpoint() -> None:
                            async with make(create()) as client:
                                r = await get(client, "/metrics")
                            assert r.status_code == 200

async def test_complete_endpoint() -> None:
                            async with make(create()) as client:
                                r = await post(client, "/v1/complete", json={"items": [0, 1, 2]})
                            assert r.status_code == 200
                            assert "visual" in r.json()["completed"]

async def test_recommend_endpoint() -> None:
                            async with make(create()) as client:
                                r = await post(client, "/v1/recommend", json={"user": 0, "top": 3})
                            assert r.status_code == 200
                            assert len(r.json()["items"]) == 3

async def test_auth_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
                            monkeypatch.setenv("MOREL_AUTH_TOKEN", "s3cr3t")
                            async with make(create()) as client:
                                r = await post(client, "/v1/complete", json={"items": [0]})
                                assert r.status_code == 401

                                r = await post(
                                    client,
                                    "/v1/complete",
                                    json={"items": [0]},
                                    headers={"authorization": "Bearer wrong"},
                                )
                                assert r.status_code == 401

                                r = await post(
                                    client,
                                    "/v1/complete",
                                    json={"items": [0]},
                                    headers={"authorization": "Bearer s3cr3t"},
                                )
                                assert r.status_code == 200
