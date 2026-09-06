"""Tests for morel.serve."""

from __future__ import annotations

import pytest

from morel.serve import Loader, create


def test_loader_capacity() -> None:
    with pytest.raises(ValueError):
        Loader(capacity=0)


def test_loader_get_caches() -> None:
    loader = Loader(capacity=2)
    calls = []

    def factory():
        calls.append(1)
        return object()

    a = loader.get("k1", factory)
    b = loader.get("k1", factory)
    assert a is b
    assert len(calls) == 1


def test_loader_evicts() -> None:
    loader = Loader(capacity=2)
    loader.get("a", lambda: object())
    loader.get("b", lambda: object())
    loader.get("c", lambda: object())
    assert "a" not in loader.keys()


def test_loader_empty_key() -> None:
    loader = Loader()
    with pytest.raises(ValueError):
        loader.get("", lambda: object())


def test_health_endpoint() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_endpoint() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create())
    r = client.get("/metrics")
    assert r.status_code == 200


def test_complete_endpoint() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create())
    r = client.post("/v1/complete", json={"items": [0, 1, 2]})
    assert r.status_code == 200
    assert "visual" in r.json()["completed"]


def test_recommend_endpoint() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create())
    r = client.post("/v1/recommend", json={"user": 0, "top": 3})
    assert r.status_code == 200
    assert len(r.json()["items"]) == 3


def test_auth_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MOREL_AUTH_TOKEN", "s3cr3t")
    app = create()
    client = TestClient(app)
    assert client.post("/v1/complete", json={"items": [0]}).status_code == 401
    assert (
        client.post(
            "/v1/complete",
            json={"items": [0]},
            headers={"authorization": "Bearer wrong"},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/v1/complete",
            json={"items": [0]},
            headers={"authorization": "Bearer s3cr3t"},
        ).status_code
        == 200
    )
