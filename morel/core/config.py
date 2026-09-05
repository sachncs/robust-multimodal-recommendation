"""Frozen, hierarchical configuration for morel.

Loaded from YAML, env, or CLI flags. Precedence is `CLI > env > YAML > default`.
Every section is validated on construction.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from morel.core.errors import ConfigError


@dataclass(frozen=True)
class Data:
    """Data pipeline configuration."""

    raw: str = "data/raw"
    processed: str = "data/processed"
    category: str = "Beauty"
    min: int = 5
    seed: int = 42


@dataclass(frozen=True)
class Encoder:
    """Feature encoder configuration."""

    text: str = "sentence-transformers/all-MiniLM-L6-v2"
    visual: str = "resnet50"
    text_dim: int = 384
    visual_dim: int = 2048
    batch: int = 64


@dataclass(frozen=True)
class Masking:
    """Masking configuration."""

    kind: str = "bernoulli"
    ratio: float = 0.4
    seed: int = 42


@dataclass(frozen=True)
class Retrieve:
    """Retrieval configuration."""

    kind: str = "mage"
    anchors: int = 10
    iters: int = 10


@dataclass(frozen=True)
class Encode:
    """Joint encoder configuration."""

    kind: str = "transformer"
    hidden: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.5
    pe: int = 20


@dataclass(frozen=True)
class Route:
    """Routing configuration."""

    kind: str = "top"
    p: int = 4
    tau: float = 0.5


@dataclass(frozen=True)
class Codebook:
    """Codebook configuration."""

    kind: str = "gumbel"
    size: int = 100


@dataclass(frozen=True)
class Complete:
    """Modality completion configuration."""

    kind: str = "mlp"
    hidden: int = 128


@dataclass(frozen=True)
class Recommend:
    """Downstream recommender configuration."""

    kind: str = "light"
    embed: int = 64
    layers: int = 3


@dataclass(frozen=True)
class Completion:
    """Completion training configuration."""

    epochs: int = 100
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    usage: float = 1.0
    balance: float = 1.0
    grad_clip: float = 1.0
    val: float = 0.1
    patience: int = 10
    amp: bool = False


@dataclass(frozen=True)
class Recommendation:
    """Recommendation training configuration."""

    epochs: int = 100
    batch: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    negatives: int = 1
    grad_clip: float = 1.0
    val: float = 0.1
    patience: int = 10
    amp: bool = False


@dataclass(frozen=True)
class Eval:
    """Evaluation configuration."""

    ks: tuple[int, ...] = (10, 20)
    robustness: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ablations: tuple[str, ...] = ("no_retrieval", "no_pe", "no_codebook")


@dataclass(frozen=True)
class Serve:
    """Inference server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    auth: bool = False


@dataclass(frozen=True)
class Log:
    """Logging configuration."""

    level: str = "INFO"
    structured: bool = True
    directory: str = "runs"


@dataclass(frozen=True)
class Config:
    """Top-level morel configuration."""

    seed: int = 42
    device: str = "auto"
    data: Data = field(default_factory=Data)
    encoder: Encoder = field(default_factory=Encoder)
    masking: Masking = field(default_factory=Masking)
    retrieve: Retrieve = field(default_factory=Retrieve)
    encode: Encode = field(default_factory=Encode)
    route: Route = field(default_factory=Route)
    codebook: Codebook = field(default_factory=Codebook)
    complete: Complete = field(default_factory=Complete)
    recommend: Recommend = field(default_factory=Recommend)
    completion: Completion = field(default_factory=Completion)
    recommendation: Recommendation = field(default_factory=Recommendation)
    eval: Eval = field(default_factory=Eval)
    serve: Serve = field(default_factory=Serve)
    log: Log = field(default_factory=Log)

    def validate(self) -> None:
        """Raise ConfigError on invalid values."""
        if self.seed < 0:
            raise ConfigError(f"seed must be non-negative, got {self.seed}")
        if not 0.0 <= self.masking.ratio <= 1.0:
            raise ConfigError(f"masking.ratio must be in [0, 1], got {self.masking.ratio}")
        if self.route.p <= 0:
            raise ConfigError(f"route.p must be positive, got {self.route.p}")
        if self.codebook.size <= 0:
            raise ConfigError(f"codebook.size must be positive, got {self.codebook.size}")
        if self.encode.hidden <= 0:
            raise ConfigError(f"encode.hidden must be positive, got {self.encode.hidden}")
        if any(k <= 0 for k in self.eval.ks):
            raise ConfigError(f"eval.ks must all be positive, got {self.eval.ks}")
        if not 0.0 <= self.completion.val <= 1.0:
            raise ConfigError("completion.val must be in [0, 1]")
        if not 0.0 <= self.recommendation.val <= 1.0:
            raise ConfigError("recommendation.val must be in [0, 1]")

    def hash(self) -> str:
        """Return a stable SHA256 of the configuration for manifest binding."""
        canonical = yaml.safe_dump(self.dump(), sort_keys=True, allow_unicode=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def dump(self) -> dict[str, Any]:
        """Return a nested dict suitable for YAML serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Config":
        """Build a Config from a nested dict.

        Unknown keys raise ConfigError. Missing keys take defaults.
        """
        if not isinstance(payload, dict):
            raise ConfigError(f"config payload must be a dict, got {type(payload).__name__}")
        cls_fields = {f.name for f in fields(cls)}
        clean_top: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in cls_fields:
                raise ConfigError(f"unknown top-level config key: {key!r}")
            clean_top[key] = value
        coerced = _coerce(cls, clean_top)
        result: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in coerced:
                f_type = _resolve_dataclass(f.type)
                if is_dataclass(f_type) and isinstance(coerced[f.name], dict):
                    result[f.name] = f_type(**coerced[f.name])
                else:
                    result[f.name] = coerced[f.name]
        return cls(**result)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """Load configuration from a YAML file."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config with values overridden from ``MOREL_*`` env vars.

        Recognized keys: ``MOREL_SEED``, ``MOREL_DEVICE``. Nested keys are not
        exposed; use a YAML file for full control.
        """
        overrides: dict[str, Any] = {}
        seed_env = os.environ.get("MOREL_SEED")
        if seed_env is not None:
            overrides["seed"] = int(seed_env)
        device_env = os.environ.get("MOREL_DEVICE")
        if device_env is not None:
            overrides["device"] = device_env
        if not overrides:
            return cls()
        merged = cls().dump()
        merged.update(overrides)
        return cls.from_dict(merged)

    def to_yaml(self, path: Path | str) -> None:
        """Write the configuration to a YAML file."""
        text = yaml.safe_dump(self.dump(), sort_keys=True, allow_unicode=True)
        Path(path).write_text(text, encoding="utf-8")


def _coerce(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively coerce nested dicts into nested dataclass instances.

    Returns a dict suitable for ``cls(**result)``. Nested dataclass fields are
    replaced with their ``_coerce``d dicts so that ``cls(**...)`` constructs
    them in turn.
    """
    out: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in payload:
            continue
        value = payload[f.name]
        f_type = _resolve_dataclass(f.type)
        if is_dataclass(f_type) and isinstance(value, dict):
            out[f.name] = _coerce(f_type, value)
        else:
            out[f.name] = value
    return out


def _resolve_dataclass(annotation: Any) -> type:
    """Return the dataclass type from a string annotation or class.

    Returns the annotation unchanged if it is not a string and not a dataclass
    (e.g. ``int``, ``str``, ``tuple``).
    """
    import dataclasses

    if isinstance(annotation, str):
        import importlib

        module = importlib.import_module("morel.core.config")
        resolved = getattr(module, annotation, None)
        if resolved is None:
            return annotation
        return resolved
    if dataclasses.is_dataclass(annotation):
        return annotation
    return annotation


__all__ = [
    "Data",
    "Encoder",
    "Masking",
    "Retrieve",
    "Encode",
    "Route",
    "Codebook",
    "Complete",
    "Recommend",
    "Completion",
    "Recommendation",
    "Eval",
    "Serve",
    "Log",
    "Config",
]
