"""Tests for morel.core.registry."""

from __future__ import annotations

import pytest

from morel.core.errors import ConfigError
from morel.core.registry import Registry


def test_register_and_create() -> None:
    registry: Registry[str] = Registry("widget")
    registry.register("plain", lambda **kw: f"plain{kw.get('size', 0)}")
    assert registry.create("plain", size=3) == "plain3"


def test_register_as_decorator_returns_the_function() -> None:
    registry: Registry[str] = Registry("widget")

    @registry.register("fancy")
    def build_fancy(**_: object) -> str:
        return "fancy"

    assert registry.create("fancy") == "fancy"
    assert build_fancy() == "fancy", "the decorator must not replace the function"


def test_unknown_key_lists_the_available_names() -> None:
    registry: Registry[str] = Registry("widget")
    registry.register("alpha", lambda **_: "a")
    registry.register("beta", lambda **_: "b")

    with pytest.raises(ConfigError, match=r"unknown widget 'gamma'; available: alpha, beta"):
        registry.get("gamma")


def test_unknown_key_on_empty_registry_is_still_readable() -> None:
    registry: Registry[str] = Registry("widget")
    with pytest.raises(ConfigError, match=r"available: \(none\)"):
        registry.get("anything")


def test_duplicate_registration_is_rejected() -> None:
    """Two components claiming one name must not silently shadow each other."""
    registry: Registry[str] = Registry("widget")
    registry.register("dup", lambda **_: "first")

    with pytest.raises(ConfigError, match="already registered"):
        registry.register("dup", lambda **_: "second")

    assert registry.create("dup") == "first"


def test_duplicate_registration_allowed_when_explicit() -> None:
    registry: Registry[str] = Registry("widget")
    registry.register("dup", lambda **_: "first")
    registry.register("dup", lambda **_: "second", replace=True)
    assert registry.create("dup") == "second"


def test_empty_name_is_rejected() -> None:
    registry: Registry[str] = Registry("widget")
    with pytest.raises(ConfigError, match="non-empty"):
        registry.register("", lambda **_: "x")


def test_container_protocol() -> None:
    registry: Registry[str] = Registry("widget")
    registry.register("b", lambda **_: "b")
    registry.register("a", lambda **_: "a")

    assert "a" in registry
    assert "missing" not in registry
    assert len(registry) == 2
    assert registry.available() == ("a", "b")
    assert list(registry) == ["a", "b"]
    assert "widget" in repr(registry)


def test_unregister_removes_an_entry() -> None:
    registry: Registry[str] = Registry("widget")
    registry.register("temp", lambda **_: "t")
    assert "temp" in registry
    registry.unregister("temp")
    assert "temp" not in registry


def test_unregister_unknown_key_is_rejected() -> None:
    registry: Registry[str] = Registry("widget")
    with pytest.raises(ConfigError, match="cannot unregister"):
        registry.unregister("nope")
