"""FastAPI app factory for morel inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from morel.core.errors import Error
from morel.serve import auth
from morel.serve.loader import Loader
from morel.serve.schema import (
    Done,
    Fill,
    Health,
    List,
    Pick,
    Query,
    serialize,
)

if TYPE_CHECKING:
    from morel.pipeline import Pipeline


class Ask(BaseModel):
    """One feedback event submitted by a client."""

    user: int = Field(..., description="User id.")
    item: int = Field(..., description="Item id.")
    signal: str = Field(..., description="One of 'like', 'dislike', 'view', 'purchase'.")


class Tell(BaseModel):
    """Response from /v1/feedback."""

    queued: bool
    buffer_size: int


class Rollback(BaseModel):
    """Response from /v1/rollback."""

    restored_version: int


class Stats(BaseModel):
    """Response from /v1/stats."""

    events_buffered: int
    updates_applied: int
    last_loss: float
    valid_loss: float
    current_version: int
    cooldown_until: float


def create(loader: Loader | None = None) -> FastAPI:
    """Create a FastAPI app.

    Args:
        loader: Optional pre-configured ``Loader``; a default one is created
            if not provided.

    Returns
    -------
        Configured ``FastAPI`` instance.
    """
    app = FastAPI(title="morel inference", version="0.1.0")
    app.state.loader = loader or Loader()
    app.state.updater_enabled = True

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", version="0.1.0")

    @app.get("/metrics")
    def metrics() -> dict[str, float]:
        return {"requests": float(request_count(app))}

    @app.post("/v1/complete", response_model=Done)
    def complete(payload: Fill, _: None = Depends(auth.dependency("read"))) -> Done:
        try:
            pipeline = app.state.loader.get("default", default)
        except Error as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        completed = run(pipeline, payload)
        return Done(completed=serialize(completed))

    @app.post("/v1/recommend", response_model=List)
    def recommend(payload: Query, _: None = Depends(auth.dependency("read"))) -> List:
        try:
            pipeline = app.state.loader.get("default", default)
        except Error as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        items = recommend_items(pipeline, payload)
        return List(items=items)

    @app.post("/v1/feedback", response_model=Tell)
    def feedback(payload: Ask, _: None = Depends(auth.dependency("admin"))) -> Tell:
        if not getattr(app.state, "updater_enabled", True):
            raise HTTPException(status_code=503, detail="Updater disabled")
        updater = getattr(app.state, "updater", None)
        if updater is None:
            raise HTTPException(status_code=503, detail="Updater not mounted")
        updater.accept(user=payload.user, item=payload.item, signal=payload.signal)
        return Tell(queued=True, buffer_size=updater.stats()["events_buffered"])

    @app.post("/v1/rollback", response_model=Rollback)
    def rollback(steps: int = 1, _: None = Depends(auth.dependency("admin"))) -> Rollback:
        updater = getattr(app.state, "updater", None)
        if updater is None:
            raise HTTPException(status_code=503, detail="Updater not mounted")
        return Rollback(restored_version=updater.rollback(steps=steps))

    @app.get("/v1/stats", response_model=Stats)
    def stats(_: None = Depends(auth.dependency("admin"))) -> Stats:
        updater = getattr(app.state, "updater", None)
        if updater is None:
            raise HTTPException(status_code=503, detail="Updater not mounted")
        s = updater.stats()
        return Stats(
            events_buffered=int(s["events_buffered"]),
            updates_applied=int(s["updates_applied"]),
            last_loss=float(s["last_loss"]),
            valid_loss=float(s["valid_loss"]),
            current_version=int(s["current_version"]),
            cooldown_until=float(s["cooldown_until"]),
        )

    return app


# Re-export for callers that want the dependency callables by name.
require_read = auth.dependency("read")
require_admin = auth.dependency("admin")


def default() -> object:
    """Build a pipeline for inference.

    Returns a tiny stub pipeline. Replace with a real loader in production.
    """
    from morel.core.config import Config
    from morel.pipeline import Pipeline

    return Pipeline(Config(), dims={"visual": 4, "text": 2})


def run(pipeline: object, payload: Fill) -> dict[str, Any]:
    """Run the completion forward pass."""
    import numpy as np
    import scipy.sparse as sp
    import torch

    from morel.data.mask import bernoulli

    items = payload.items
    if not items:
        raise HTTPException(status_code=400, detail="items must be non-empty")
    pipeline_obj = cast("Pipeline", pipeline)
    dims = getattr(pipeline_obj, "dims", {"visual": 4, "text": 2})
    mask = bernoulli(len(items), len(dims), 0.4, seed=0).to_numpy()
    features = {
        name: torch.from_numpy(np.zeros((len(items), dim), dtype=np.float32))
        for name, dim in dims.items()
    }
    mask_t = torch.from_numpy(mask)
    # Build the empty adjacency sparsely. Materializing a dense (n, n) block
    # first costs O(n^2) memory for a matrix that has no nonzeros at all,
    # which put a hard ceiling on the number of items a request could carry.
    adjacency = sp.csr_matrix((len(items), len(items)), dtype=np.float32)
    output = pipeline_obj(features, mask_t, adjacency=adjacency, training=False)
    if payload.modalities:
        return {
            name: tensor for name, tensor in output.completed.items() if name in payload.modalities
        }
    completed: dict[str, Any] = output.completed
    return completed


def recommend_items(pipeline: object, payload: Query) -> list[Pick]:
    """Return the top-``top`` items for a user.

    Stub implementation: returns a uniformly-ranked slice of the catalogue.
    """
    top = max(1, int(payload.top))
    return [Pick(item=i, score=1.0 / (i + 1)) for i in range(top)]


def request_count(app: FastAPI) -> int:
    return int(getattr(app.state, "request_count", 0))


# (Dependencies are installed directly via Depends(_require_read).)


__all__ = [
    "Ask",
    "Rollback",
    "Stats",
    "Tell",
    "create",
]
