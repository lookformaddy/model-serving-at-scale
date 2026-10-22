"""Core types and deterministic fake models."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .errors import ValidationError


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str
    weights: tuple[float, ...]
    bias: float = 0.0
    threshold: float = 0.5
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValidationError("model name and version are required")
        if not self.weights:
            raise ValidationError("weights must be non-empty")
        if not (0.0 < self.threshold < 1.0):
            raise ValidationError("threshold must be in (0, 1)")

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"


class LinearScorer:
    """Deterministic fake model: sigmoid(weights·features + bias)."""

    def __init__(self, spec: ModelSpec, *, fail: bool = False, delay_ms: float = 0.0):
        self.spec = spec
        self.fail = fail
        self.delay_ms = delay_ms

    def predict_one(self, features: tuple[float, ...] | list[float]) -> float:
        if self.fail:
            raise RuntimeError(f"model {self.spec.key} simulated failure")
        if len(features) != len(self.spec.weights):
            raise ValidationError(
                f"expected {len(self.spec.weights)} features, got {len(features)}"
            )
        dot = sum(w * float(f) for w, f in zip(self.spec.weights, features)) + self.spec.bias
        return _sigmoid(dot)

    def predict_batch(self, batch: list[tuple[float, ...] | list[float]]) -> list[float]:
        return [self.predict_one(f) for f in batch]

    def label(self, score: float) -> int:
        return 1 if score >= self.spec.threshold else 0


@dataclass(frozen=True)
class PredictRequest:
    request_id: str
    features: tuple[float, ...]
    model_name: str = "churn"
    user_id: str = ""
    deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValidationError("request_id is required")
        if not self.features:
            raise ValidationError("features must be non-empty")


@dataclass
class PredictResponse:
    request_id: str
    model_name: str
    model_version: str
    score: float
    label: int
    variant: str = "stable"
    latency_ms: float = 0.0
    batched: bool = False
    batch_size: int = 1
    shadow_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "score": round(self.score, 6),
            "label": self.label,
            "variant": self.variant,
            "latency_ms": round(self.latency_ms, 3),
            "batched": self.batched,
            "batch_size": self.batch_size,
            "shadow_score": None if self.shadow_score is None else round(self.shadow_score, 6),
        }
