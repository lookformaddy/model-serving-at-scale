"""Deterministic A/B variant routing by stable hash."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .errors import ValidationError


def hash_bucket(key: str, buckets: int = 100) -> int:
    """Stable bucket in [0, buckets) — same key always routes the same."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


@dataclass(frozen=True)
class Variant:
    name: str
    model_version: str
    weight: int  # percent, 0..100

    def __post_init__(self) -> None:
        if not (0 <= self.weight <= 100):
            raise ValidationError("variant weight must be in [0, 100]")


class ABRouter:
    """Weighted deterministic routing. Weights must sum to 100."""

    def __init__(self, variants: list[Variant]):
        if not variants:
            raise ValidationError("at least one variant required")
        total = sum(v.weight for v in variants)
        if total != 100:
            raise ValidationError(f"variant weights must sum to 100, got {total}")
        self._variants = list(variants)
        # cumulative boundaries
        self._bounds: list[tuple[int, Variant]] = []
        acc = 0
        for v in variants:
            acc += v.weight
            self._bounds.append((acc, v))

    @property
    def variants(self) -> list[Variant]:
        return list(self._variants)

    def route(self, key: str) -> Variant:
        bucket = hash_bucket(key or "anonymous")
        for bound, variant in self._bounds:
            if bucket < bound:
                return variant
        return self._variants[-1]  # unreachable when weights sum to 100

    def distribution(self, keys: list[str]) -> dict[str, int]:
        out: dict[str, int] = {v.name: 0 for v in self._variants}
        for k in keys:
            out[self.route(k).name] += 1
        return out
