"""Public API for the morel.route package."""

from morel.core.registry import Registry
from morel.route.router import Dense, Fixed, Gumbel, Router, Top, Weights, build

#: Selectable routers, keyed by ``config.route.kind``.
ROUTERS: Registry[Router] = Registry("router")


@ROUTERS.register("top")
def build_top(*, dim: int, k: int, p: int, tau: float) -> Router:
    """Build the top-p router used by the full model."""
    return Top(dim, k, p=p, tau=tau)


@ROUTERS.register("dense")
def build_dense(*, dim: int, k: int, p: int, tau: float) -> Router:
    """Build the dense softmax router; ``p`` does not apply."""
    del p
    return Dense(dim, k, tau=tau)


@ROUTERS.register("gumbel")
def build_gumbel_router(*, dim: int, k: int, p: int, tau: float) -> Router:
    """Build the Gumbel-softmax router; ``p`` does not apply."""
    del p
    return Gumbel(dim, k, tau=tau)


@ROUTERS.register("fixed")
def build_fixed(*, dim: int, k: int, p: int, tau: float) -> Router:
    """Build the fixed-index router used when routing is supplied externally."""
    del dim, p, tau
    return Fixed(k)


__all__ = [
    "ROUTERS",
    "Dense",
    "Fixed",
    "Gumbel",
    "Router",
    "Top",
    "Weights",
    "build",
    "build_dense",
    "build_fixed",
    "build_gumbel_router",
    "build_top",
]
