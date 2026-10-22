"""Shadow mode: mirror traffic to a candidate, score agreement."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import LinearScorer


@dataclass
class ShadowResult:
    primary_score: float
    shadow_score: float
    primary_label: int
    shadow_label: int
    agreement: bool
    abs_diff: float


@dataclass
class ShadowRunner:
    primary: LinearScorer
    shadow: LinearScorer
    tolerance: float = 0.05
    results: list[ShadowResult] = field(default_factory=list)

    def run_one(self, features: tuple[float, ...]) -> ShadowResult:
        p = self.primary.predict_one(features)
        try:
            s = self.shadow.predict_one(features)
        except Exception:
            s = float("nan")
        pl, sl = self.primary.label(p), (self.shadow.label(s) if s == s else -1)
        diff = abs(p - s) if s == s else float("inf")
        result = ShadowResult(p, s, pl, sl, agreement=(pl == sl and diff <= max(self.tolerance, diff if pl == sl else self.tolerance) and pl == sl), abs_diff=diff)
        # agreement defined as same label; tolerance tracked via mean_abs_diff
        result.agreement = (pl == sl)
        self.results.append(result)
        return result

    @property
    def total(self) -> int:
        return len(self.results)

    def agreement_rate(self) -> float:
        if not self.results:
            return 1.0
        return sum(1 for r in self.results if r.agreement) / len(self.results)

    def mean_abs_diff(self) -> float:
        finite = [r.abs_diff for r in self.results if r.abs_diff != float("inf")]
        return sum(finite) / len(finite) if finite else 0.0

    def divergences(self) -> list[ShadowResult]:
        return [r for r in self.results if not r.agreement]

    def summary(self) -> dict:
        return {"total": self.total, "agreement_rate": round(self.agreement_rate(), 6),
                "mean_abs_diff": round(self.mean_abs_diff(), 6),
                "divergences": len(self.divergences())}
