"""Serving orchestration: registry + batcher + router + canary + shadow.

Core works WITHOUT FastAPI. create_app() guards the import.
Features: deadline/timeout handling, load-shedding (max in-flight),
latency metrics, canary auto-rollback, shadow agreement scoring.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from .batcher import BatchConfig, DynamicBatcher
from .canary import CanaryController
from .errors import DeadlineExceededError, OverloadedError, ValidationError
from .metrics import MetricsRegistry
from .models import PredictRequest, PredictResponse
from .registry import ModelRegistry, build_demo_registry
from .router import ABRouter
from .shadow import ShadowRunner


class ModelServingService:
    def __init__(
        self,
        registry: ModelRegistry | None = None,
        *,
        router: ABRouter | None = None,
        canary: CanaryController | None = None,
        shadow: ShadowRunner | None = None,
        metrics: MetricsRegistry | None = None,
        batch_config: BatchConfig | None = None,
        max_inflight: int = 64,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.registry = registry or build_demo_registry()
        self.router = router
        self.canary = canary
        self.shadow = shadow
        self.metrics = metrics or MetricsRegistry()
        self.batch_config = batch_config or BatchConfig()
        self.max_inflight = max_inflight
        self._clock = clock
        self._inflight = 0
        self._lock = threading.Lock()
        self._started = time.monotonic()

    @property
    def inflight(self) -> int:
        return self._inflight

    def _acquire(self) -> None:
        with self._lock:
            if self._inflight >= self.max_inflight:
                self.metrics.record_shed()
                raise OverloadedError(f"load shed: {self._inflight} in-flight >= {self.max_inflight}")
            self._inflight += 1

    def _release(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    def _resolve(self, request: PredictRequest) -> tuple[str, str]:
        """Returns (version, variant). Canary takes precedence, then A/B, else stable."""
        name = request.model_name
        stable = self.registry.active_version(name)
        key = request.user_id or request.request_id
        if self.canary is not None and self.canary.should_route_canary(key):
            # canary version = newest non-active version if present
            versions = self.registry.list_versions(name)
            others = [v for v in versions if v != stable]
            if others:
                return others[-1], "canary"
        if self.router is not None:
            variant = self.router.route(key)
            # only honour router version if registered
            if variant.model_version in self.registry.list_versions(name):
                return variant.model_version, variant.name
        return stable, "stable"

    def predict(self, request: PredictRequest, *, use_batcher: bool = False) -> PredictResponse:
        started = self._clock()
        self._acquire()
        try:
            version, variant = self._resolve(request)
            scorer = self.registry.get(request.model_name, version)
            if use_batcher:
                with DynamicBatcher(lambda b: scorer.predict_batch(b), self.batch_config) as batcher:
                    score, bsize = batcher.submit(request.features)
                batched, batch_size = True, bsize
            else:
                score, batched, batch_size = scorer.predict_one(request.features), False, 1
            latency_ms = (self._clock() - started) * 1000.0
            # deadline handling
            if request.deadline_ms is not None and latency_ms > request.deadline_ms:
                self.metrics.record(latency_ms, error=True)
                raise DeadlineExceededError(
                    f"latency {latency_ms:.1f}ms exceeded deadline {request.deadline_ms}ms")
            shadow_score = None
            if self.shadow is not None:
                result = self.shadow.run_one(request.features)
                shadow_score = result.shadow_score
            if variant == "canary" and self.canary is not None:
                self.canary.record(latency_ms, error=False)
                self.canary.evaluate()
            self.metrics.record(latency_ms, error=False)
            return PredictResponse(
                request_id=request.request_id, model_name=request.model_name,
                model_version=version, score=score, label=scorer.label(score),
                variant=variant, latency_ms=latency_ms, batched=batched,
                batch_size=batch_size, shadow_score=shadow_score)
        except (DeadlineExceededError, OverloadedError, ValidationError):
            raise
        except Exception:
            latency_ms = (self._clock() - started) * 1000.0
            self.metrics.record(latency_ms, error=True)
            if variant_is_canary(self, request):
                pass
            raise
        finally:
            self._release()

    def predict_batch(self, requests: list[PredictRequest]) -> list[PredictResponse]:
        return [self.predict(r) for r in requests]

    def health(self) -> dict:
        return {"status": "ok", "models": self.registry.list_models(),
                "inflight": self._inflight, "max_inflight": self.max_inflight}

    def metrics_snapshot(self) -> dict:
        elapsed = max(time.monotonic() - self._started, 1e-9)
        snap = self.metrics.snapshot(elapsed_s=elapsed).to_dict()
        if self.canary is not None:
            snap["canary"] = self.canary.status()
        if self.shadow is not None:
            snap["shadow"] = self.shadow.summary()
        return snap


def variant_is_canary(service: ModelServingService, request: PredictRequest) -> bool:
    try:
        _, v = service._resolve(request)
        return v == "canary"
    except Exception:
        return False


def build_demo_service(**kwargs) -> ModelServingService:
    return ModelServingService(**kwargs)


def create_app(service: ModelServingService | None = None):
    """FastAPI app factory — guarded import so core works without it."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError(
            "FastAPI not installed. Install with: pip install -e '.[api]'") from exc
    svc = service or build_demo_service()
    app = FastAPI(title="Model Serving at Scale", version="0.1.0")

    class PredictIn(BaseModel):
        request_id: str
        features: list[float]
        model_name: str = "churn"
        user_id: str = ""
        deadline_ms: float | None = None

    @app.get("/health")
    def health():
        return svc.health()

    @app.get("/metrics")
    def metrics():
        return svc.metrics_snapshot()

    @app.get("/models")
    def models():
        return {n: svc.registry.list_versions(n) for n in svc.registry.list_models()}

    @app.post("/predict")
    def predict(body: PredictIn):
        try:
            req = PredictRequest(request_id=body.request_id, features=tuple(body.features),
                                 model_name=body.model_name, user_id=body.user_id,
                                 deadline_ms=body.deadline_ms)
            return svc.predict(req).to_dict()
        except OverloadedError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except DeadlineExceededError as e:
            raise HTTPException(status_code=504, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    app.state.service = svc
    return app


try:
    app = create_app()
except ImportError:
    app = None  # type: ignore[assignment]
