"""Thread-safe model loader with LRU cache."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from morel.core.errors import ModelError


class Loader:
    """Cache of named model pipelines keyed by their checkpoint path.

    The loader is safe to share across request threads.
    """

    def __init__(self, *, capacity: int = 4) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, factory: Any) -> Any:
        """Return a cached pipeline or build and cache one via ``factory``."""
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        model = factory()
        if model is None:
            raise ModelError(f"loader factory returned None for key {key!r}")
        with self._lock:
            if key in self._cache:
                existing = self._cache[key]
                self._cache.move_to_end(key)
                return existing
            self._cache[key] = model
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        return model

    def clear(self) -> None:
        """Drop every cached pipeline."""
        with self._lock:
            self._cache.clear()

    def keys(self) -> list[str]:
        """Return currently cached keys (insertion-ordered)."""
        with self._lock:
            return list(self._cache.keys())

    def load_path(self, path: Path | str) -> Any:
        """Load a torch checkpoint, returning the state dict."""
        path = Path(path)
        if not path.exists():
            raise ModelError(f"checkpoint not found: {path}")
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)


__all__ = ["Loader"]
