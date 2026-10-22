"""Canary releases: traffic splitting + auto-rollback."""
from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ValidationError
from .router import hash_bucket


@dataclass(frozen=True)
class CanaryConfig:
    traffic_pct: int = 5
    error_rate_threshold: float = 0.05
    p95_latency_ms: float = 200.0
    min_requests: int = 20

    def __post_init__(self) -> None:
        if not (0 <= self.traffic_pct <= 100):
            raise ValidationError("traffic_pct must be in [0, 100]")
        if not (0.0 <= self.error_rate_threshold <= 1.0):
            raise ValidationError("error_rate_threshold must be in [0, 1]")


@dataclass
class CanaryStats:
    requests: int = 0
    errors: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0


class CanaryController:
    """Routes a % of traffic to canary; auto-rollbacks on breach."""

    def __init__(self, config: CanaryConfig | None = None):
        self.config = config or CanaryConfig()
        self.stats = CanaryStats()
        self.rolled_back = False
        self.rollback_reason: str | None = None

    def should_route_canary(self, key: str) -> bool:
        if self.rolled_back or self.config.traffic_pct == 0:
            return False
        if self.config.traffic_pct == 100:
            return True
        return hash_bucket(f"canary:{key}") < self.config.traffic_pct

    def record(self, latency_ms: float, *, error: bool = False) -> None:
        self.stats.requests += 1
        self.stats.latencies.append(latency_ms)
        if error:
            self.stats.errors += 1

    def _p95(self) -> float:
        from .metrics import percentile
        return percentile(list(self.stats.latencies), 95) if self.stats.latencies else 0.0

    def evaluate(self) -> bool:
        """Returns True if rollback was triggered by this evaluation."""
        if self.rolled_back or self.stats.requests < self.config.min_requests:
            return False
        if self.stats.error_rate > self.config.error_rate_threshold:
            return self._rollback(f"error rate {self.stats.error_rate:.3f} > {self.config.error_rate_threshold}")
        p95 = self._p95()
        if p95 > self.config.p95_latency_ms:
            return self._rollback(f"p95 latency {p95:.1f}ms > {self.config.p95_latency_ms}ms")
        return False

    def _rollback(self, reason: str) -> bool:
        self.rolled_back = True
        self.rollback_reason = reason
        return True

    def promote(self) -> None:
        """Mark canary healthy for promotion (caller promotes in registry)."""
        if self.rolled_back:
            raise ValidationError(f"cannot promote a rolled-back canary: {self.rollback_reason}")

    def status(self) -> dict:
        return {"traffic_pct": self.config.traffic_pct, "requests": self.stats.requests,
                "errors": self.stats.errors, "error_rate": round(self.stats.error_rate, 6),
                "p95_ms": round(self._p95(), 3), "rolled_back": self.rolled_back,
                "rollback_reason": self.rollback_reason}
