"""FastAPI app factory for morel inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from morel.core.errors import MorelError
from morel.serve import auth
from morel.serve.loader import Loader
from morel.serve.schema import (
    CompleteRequest,
    CompleteResponse,
    HealthResponse,
    Pick,
    RecommendRequest,
    RecommendResponse,
    serialize,
)

if TYPE_CHECKING:
    from morel.pipeline import Pipeline


class FeedbackRequest(BaseModel):
    """One feedback event submitted by a client."""

    user: int = Field(..., description="User id.")
    item: int = Field(..., description="Item id.")
    signal: str = Field(..., description="One of 'like', 'dislike', 'view', 'purchase'.")


class FeedbackResponse(BaseModel):
    """Response from /v1/feedback."""

    queued: bool
    buffer_size: int


class RollbackResponse(BaseModel):
    """Response from /v1/rollback."""

    restored_version: int


class StatsResponse(BaseModel):
    """Response from /v1/stats."""

    events_buffered: int
    updates_applied: int
    last_loss: float
    last_valid_loss: float
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

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version="0.1.0")

    @app.get("/metrics")
    def metrics() -> dict[str, float]:
        return {"requests": float(request_count(app))}

    @app.post("/v1/complete", response_model=CompleteResponse)
    def complete(
        payload: CompleteRequest, _: None = Depends(auth.dependency("read"))
    ) -> CompleteResponse:
        try:
            pipeline = app.state.loader.get("default", build_default)
        except MorelError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        completed = run(pipeline, payload)
        return CompleteResponse(completed=serialize(completed))

    @app.post("/v1/recommend", response_model=RecommendResponse)
    def recommend(
        payload: RecommendRequest, _: None = Depends(auth.dependency("read"))
    ) -> RecommendResponse:
        try:
            pipeline = app.state.loader.get("default", build_default)
        except MorelError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        items = recommend_items(pipeline, payload)
        return RecommendResponse(items=items)

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    def feedback(
        payload: FeedbackRequest, _: None = Depends(auth.dependency("admin"))
    ) -> FeedbackResponse:
        if not getattr(app.state, "updater_enabled", True):
            raise HTTPException(status_code=503, detail="Updater disabled")
        updater = getattr(app.state, "updater", None)
        if updater is None:
            raise HTTPException(status_code=503, detail="Updater not mounted")
        updater.accept(user=payload.user, item=payload.item, signal=payload.signal)
        return FeedbackResponse(queued=True, buffer_size=updater.stats()["events_buffered"])

    @app.post("/v1/rollback", response_model=RollbackResponse)
    def rollback(steps: int = 1, _: None = Depends(auth.dependency("admin"))) -> RollbackResponse:
        updater = getattr(app.state, "updater", None)
        if updater is None:
            raise HTTPException(status_code=503, detail="Updater not mounted")
        return RollbackResponse(restored_version=updater.rollback(steps=steps))

    @app.get("/v1/stats", response_model=StatsResponse)
    def stats(_: None = Depends(auth.dependency("admin"))) -> StatsResponse:
        updater = getattr(app.state, "updater", None)
        if updater is None:
            raise HTTPException(status_code=503, detail="Updater not mounted")
        s = updater.stats()
        return StatsResponse(
            events_buffered=int(s["events_buffered"]),
            updates_applied=int(s["updates_applied"]),
            last_loss=float(s["last_loss"]),
            last_valid_loss=float(s["last_valid_loss"]),
            current_version=int(s["current_version"]),
            cooldown_until=float(s["cooldown_until"]),
        )

    return app


# Re-export for callers that want the dependency callables by name.
require_read = auth.dependency("read")
require_admin = auth.dependency("admin")


def build_default() -> object:
    """Build a pipeline for inference.

    Returns a tiny stub pipeline. Replace with a real loader in production.
    """
    from morel.core.config import Config
    from morel.pipeline import Pipeline

    return Pipeline(Config(), dims={"visual": 4, "text": 2})


def run(pipeline: object, payload: CompleteRequest) -> dict[str, Any]:
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


def recommend_items(pipeline: object, payload: RecommendRequest) -> list[Pick]:
    """Return the top-``top`` items for a user.

    Stub implementation: returns a uniformly-ranked slice of the catalogue.
    """
    top = max(1, int(payload.top))
    return [Pick(item=i, score=1.0 / (i + 1)) for i in range(top)]


def request_count(app: FastAPI) -> int:
    return int(getattr(app.state, "request_count", 0))


# (Dependencies are installed directly via Depends(_require_read).)


__all__ = [
    "FeedbackRequest",
    "FeedbackResponse",
    "RollbackResponse",
    "StatsResponse",
    "create",
]
