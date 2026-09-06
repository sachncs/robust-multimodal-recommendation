"""Pydantic schemas for the inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CompleteRequest(BaseModel):
    """Request to complete missing modalities for a set of items."""

    items: list[int] = Field(..., description="Item ids to complete.")
    modalities: list[str] | None = Field(
        default=None,
        description="Optional subset of modality names; defaults to all.",
    )


class CompleteResponse(BaseModel):
    """Response containing the completed modalities per item."""

    completed: dict[str, list[list[float]]] = Field(
        ..., description="Mapping from modality name to per-item vectors."
    )


class RecommendRequest(BaseModel):
    """Request to score a user against the catalogue."""

    user: int = Field(..., description="User id.")
    top: int = Field(default=20, description="Number of top items to return.")


class RecommendItem(BaseModel):
    """One (item, score) pair."""

    item: int
    score: float


class RecommendResponse(BaseModel):
    """Response with ranked items for the requested user."""

    items: list[RecommendItem]


class HealthResponse(BaseModel):
    """Health probe response."""

    status: str = "ok"
    version: str


def serialize_completed(completed: dict[str, Any]) -> dict[str, list[list[float]]]:
    """Convert torch tensors to nested Python lists for JSON serialization."""
    return {name: tensor.detach().cpu().tolist() for name, tensor in completed.items()}


__all__ = [
    "CompleteRequest",
    "CompleteResponse",
    "RecommendRequest",
    "RecommendItem",
    "RecommendResponse",
    "HealthResponse",
    "serialize_completed",
]
