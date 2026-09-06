"""Thread-safe model loader with LRU cache."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class Loader:
    """Cache of named model pipelines keyed by their checkpoint path.

    The loader is safe to share across request threads.
    """

    def __init__(self, *, capacity: int = 4) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.cache: "OrderedDict[str, Any]" = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str, factory: Any) -> Any:
        """Return a cached pipeline or build and cache one via ``factory``."""
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        model = factory()
        if model is None:
            raise ModelError(f"loader factory returned None for key {key!r}")
        with self.lock:
            if key in self.cache:
                existing = self.cache[key]
                self.cache.move_to_end(key)
                return existing
            self.cache[key] = model
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return model

    def clear(self) -> None:
        """Drop every cached pipeline."""
        with self.lock:
            self.cache.clear()

    def keys(self) -> list[str]:
        """Return currently cached keys (insertion-ordered)."""
        with self.lock:
            return list(self.cache.keys())


__all__ = ["Loader"]
