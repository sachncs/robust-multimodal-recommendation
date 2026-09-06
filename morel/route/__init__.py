"""Public API for the morel.route package."""

from morel.core.errors import ConfigError
from morel.route.router import Dense, Fixed, Gumbel, Router, Top, Weights, build

#: Map from config name to router factory. Module-local; no global registry.
KIND: dict[str, type[Router]] = {
    "top": Top,
    "dense": Dense,
    "gumbel": Gumbel,
    "fixed": Fixed,
}


__all__ = [
    "KIND",
    "Dense",
    "Fixed",
    "Gumbel",
    "Router",
    "Top",
    "Weights",
    "build",
]
