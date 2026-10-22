"""Latency percentiles, throughput and error metrics."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if not (0.0 <= pct <= 100.0):
        raise ValueError("pct must be in [0, 100]")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = math.floor(rank); hi = math.ceil(rank)
    if lo == hi:
        return s[int(rank)]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


@dataclass(frozen=True)
class LatencyStats:
    count: int
    p50: float
    p95: float
    p99: float
    mean: float
    max_value: float


@dataclass(frozen=True)
class MetricsSnapshot:
    total_requests: int
    total_errors: int
    error_rate: float
    throughput_rps: float
    latency: LatencyStats
    shed_count: int
    batch_count: int
    avg_batch_size: float

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests, "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 6), "throughput_rps": round(self.throughput_rps, 3),
            "latency_p50_ms": round(self.latency.p50, 3), "latency_p95_ms": round(self.latency.p95, 3),
            "latency_p99_ms": round(self.latency.p99, 3), "latency_mean_ms": round(self.latency.mean, 3),
            "shed_count": self.shed_count, "batch_count": self.batch_count,
            "avg_batch_size": round(self.avg_batch_size, 3),
        }


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: list[float] = []
        self._errors = 0
        self._shed = 0
        self._batches = 0
        self._batched_items = 0
        self._started: float | None = None

    def record(self, latency_ms: float, *, error: bool = False) -> None:
        with self._lock:
            self._latencies.append(latency_ms)
            if error:
                self._errors += 1

    def record_shed(self) -> None:
        with self._lock:
            self._shed += 1

    def record_batch(self, size: int) -> None:
        with self._lock:
            self._batches += 1
            self._batched_items += size

    @property
    def total(self) -> int:
        return len(self._latencies)

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def shed_count(self) -> int:
        return self._shed

    def error_rate(self) -> float:
        return self._errors / len(self._latencies) if self._latencies else 0.0

    def latency_stats(self) -> LatencyStats:
        vals = list(self._latencies)
        if not vals:
            return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return LatencyStats(len(vals), percentile(vals, 50), percentile(vals, 95),
                            percentile(vals, 99), sum(vals) / len(vals), max(vals))

    def snapshot(self, elapsed_s: float = 0.0) -> MetricsSnapshot:
        total = len(self._latencies)
        tps = total / elapsed_s if elapsed_s > 0 else 0.0
        avg = self._batched_items / self._batches if self._batches else 0.0
        return MetricsSnapshot(total, self._errors, self.error_rate(), tps,
                               self.latency_stats(), self._shed, self._batches, avg)
