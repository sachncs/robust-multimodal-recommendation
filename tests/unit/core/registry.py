"""Tests for morel.core.registry."""

from __future__ import annotations

import pytest

from morel.core.errors import ConfigError
from morel.core.registry import Registry


class Checker:
    """Aggregated test methods for this module."""

    def register() -> None:
        registry: Registry[str] = Registry("widget")
        registry.register("plain", lambda **kw: f"plain{kw.get('size', 0)}")
        assert registry.create("plain", size=3) == "plain3"

    def decorator() -> None:
        registry: Registry[str] = Registry("widget")

        @registry.register("fancy")
        def build_fancy(**_: object) -> str:
            return "fancy"

        assert registry.create("fancy") == "fancy"
        assert build_fancy() == "fancy", "the decorator must not replace the function"

    def unknown() -> None:
        registry: Registry[str] = Registry("widget")
        registry.register("alpha", lambda **_: "a")
        registry.register("beta", lambda **_: "b")

        with pytest.raises(ConfigError, match=r"unknown widget 'gamma'; available: alpha, beta"):
            registry.get("gamma")

    def key() -> None:
        registry: Registry[str] = Registry("widget")
        with pytest.raises(ConfigError, match=r"available: \(none\)"):
            registry.get("anything")

    def duplicate() -> None:
        """Two components claiming one name must not silently shadow each other."""
        registry: Registry[str] = Registry("widget")
        registry.register("dup", lambda **_: "first")

        with pytest.raises(ConfigError, match="already registered"):
            registry.register("dup", lambda **_: "second")

        assert registry.create("dup") == "first"

    def registration() -> None:
        registry: Registry[str] = Registry("widget")
        registry.register("dup", lambda **_: "first")
        registry.register("dup", lambda **_: "second", replace=True)
        assert registry.create("dup") == "second"

    def empty() -> None:
        registry: Registry[str] = Registry("widget")
        with pytest.raises(ConfigError, match="non-empty"):
            registry.register("", lambda **_: "x")

    def container() -> None:
        registry: Registry[str] = Registry("widget")
        registry.register("b", lambda **_: "b")
        registry.register("a", lambda **_: "a")

        assert "a" in registry
        assert "missing" not in registry
        assert len(registry) == 2
        assert registry.available() == ("a", "b")
        assert list(registry) == ["a", "b"]
        assert "widget" in repr(registry)

    def unregister() -> None:
        registry: Registry[str] = Registry("widget")
        registry.register("temp", lambda **_: "t")
        assert "temp" in registry
        registry.unregister("temp")
        assert "temp" not in registry

    def rejected() -> None:
        registry: Registry[str] = Registry("widget")
        with pytest.raises(ConfigError, match="cannot unregister"):
            registry.unregister("nope")
