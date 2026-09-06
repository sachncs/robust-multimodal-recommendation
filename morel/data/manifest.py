"""Artifact manifest.

Every persisted artifact has a sidecar ``.manifest.json`` carrying enough
metadata to answer: what dataset, what version, what preprocessing, what
feature extractor, what configuration, what random seed, what code version.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from morel.core.errors import DataError

VERSION = 1


@dataclass
class Manifest:
    """Manifest sidecar for a data artifact."""

    dataset: str
    version: str
    code: str
    seed: int
    extractor: str
    config_hash: str
    parents: list[str] = field(default_factory=list)
    schema: int = VERSION
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        """Deserialize from JSON."""
        payload = json.loads(text)
        return cls(**payload)


def path_for(artifact: Path | str) -> Path:
    """Return the sidecar manifest path for an artifact."""
    target = Path(artifact)
    return target.with_suffix(target.suffix + ".manifest.json")


def save(artifact: Path | str, manifest: Manifest) -> Path:
    """Atomically save a manifest next to an artifact.

    Args:
        artifact: Path to the artifact the manifest describes.
        manifest: Manifest payload.

    Returns
    -------
        The manifest path that was written.
    """
    target = Path(artifact).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path_for(target)
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(manifest.to_json(), encoding="utf-8")
    os.replace(tmp, sidecar)
    return sidecar


def load(artifact: Path | str, *, expected_config_hash: str | None = None) -> Manifest:
    """Load a manifest sidecar.

    Args:
        artifact: Path to the artifact whose manifest to load.
        expected_config_hash: If provided, raise DataError on mismatch.

    Returns
    -------
        The loaded Manifest.

    Raises
    ------
        DataError: If the sidecar is missing or its config hash mismatches.
    """
    sidecar = path_for(artifact)
    if not sidecar.exists():
        raise DataError(f"manifest not found for {artifact}: {sidecar}")
    manifest = Manifest.from_json(sidecar.read_text(encoding="utf-8"))
    if expected_config_hash is not None and manifest.config_hash != expected_config_hash:
        raise DataError(
            f"manifest config hash mismatch for {artifact}: "
            f"got {manifest.config_hash}, expected {expected_config_hash}"
        )
    return manifest


def checksum(path: Path | str) -> str:
    """Compute SHA256 of a file's contents."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "VERSION",
    "Manifest",
    "checksum",
    "load",
    "path_for",
    "save",
]
