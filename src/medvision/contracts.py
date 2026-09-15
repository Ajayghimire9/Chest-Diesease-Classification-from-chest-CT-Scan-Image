from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    predicted_class: str | None
    probability: float
    status: Literal["accepted", "review"]
    latency_ms: float = Field(ge=0)


@dataclass(frozen=True)
class ImagePolicy:
    min_size: int = 32
    max_size: int = 4096
    max_bytes: int = 10_000_000
    review_threshold: float = 0.70
