"""FastAPI app factory for morel inference."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from morel.core.errors import MorelError
from morel.serve import auth
from morel.serve.loader import Loader
from morel.serve.schema import (
    CompleteRequest,
    CompleteResponse,
    HealthResponse,
    RecommendItem,
    RecommendRequest,
    RecommendResponse,
    serialize_completed,
)


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

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version="0.1.0")

    @app.get("/metrics")
    def metrics() -> dict[str, float]:
        return {"requests": float(request_count(app))}

    @app.post("/v1/complete", response_model=CompleteResponse)
    def complete(payload: CompleteRequest, _: None = Depends(auth.require)) -> CompleteResponse:
        try:
            pipeline = app.state.loader.get("default", build_default_pipeline)
        except MorelError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        completed = run_complete(pipeline, payload)
        return CompleteResponse(completed=serialize_completed(completed))

    @app.post("/v1/recommend", response_model=RecommendResponse)
    def recommend(payload: RecommendRequest, _: None = Depends(auth.require)) -> RecommendResponse:
        try:
            pipeline = app.state.loader.get("default", build_default_pipeline)
        except MorelError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        items = run_recommend(pipeline, payload)
        return RecommendResponse(items=items)

    return app


def build_default_pipeline() -> object:
    """Build a pipeline for inference.

    Returns a tiny stub pipeline. Replace with a real loader in production.
    """
    from morel.core.config import Config
    from morel.pipeline import Pipeline

    return Pipeline(Config(), dims={"visual": 4, "text": 2})


def run_complete(pipeline: object, payload: CompleteRequest) -> dict:
    """Run the completion forward pass."""
    import numpy as np
    import scipy.sparse as sp
    import torch

    from morel.data.mask import bernoulli

    items = payload.items
    if not items:
        raise HTTPException(status_code=400, detail="items must be non-empty")
    pipeline_obj = pipeline
    dims = getattr(pipeline_obj, "dims", {"visual": 4, "text": 2})
    mask = bernoulli(len(items), len(dims), 0.4, seed=0).to_numpy()
    features = {
        name: torch.from_numpy(np.zeros((len(items), dim), dtype=np.float32))
        for name, dim in dims.items()
    }
    mask_t = torch.from_numpy(mask)
    adjacency = sp.csr_matrix(np.zeros((len(items), len(items)), dtype=np.float32))
    output = pipeline_obj(features, mask_t, adjacency=adjacency, training=False)
    if payload.modalities:
        return {
            name: tensor for name, tensor in output.completed.items() if name in payload.modalities
        }
    return output.completed


def run_recommend(pipeline: object, payload: RecommendRequest) -> list[RecommendItem]:
    """Return the top-``top`` items for a user.

    Stub implementation: returns a uniformly-ranked slice of the catalogue.
    """
    top = max(1, int(payload.top))
    return [RecommendItem(item=i, score=1.0 / (i + 1)) for i in range(top)]


def request_count(app: FastAPI) -> int:
    return int(getattr(app.state, "request_count", 0))


__all__ = ["create"]
