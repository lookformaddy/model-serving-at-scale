"""End-to-end demo: batching, canary, shadow, A/B, load-shedding, metrics."""
from __future__ import annotations

import time

from .batcher import BatchConfig, DynamicBatcher
from .canary import CanaryConfig, CanaryController
from .models import LinearScorer, ModelSpec, PredictRequest
from .registry import ModelRegistry, build_demo_registry
from .router import ABRouter, Variant
from .service import ModelServingService
from .shadow import ShadowRunner


def run_demo() -> None:
    print("=== Model Serving at Scale — demo ===")
    registry = build_demo_registry()
    print(f"registry: {registry.list_models()} churn versions={registry.list_versions('churn')} active={registry.active_version('churn')}")

    # 1. Single prediction
    svc = ModelServingService(registry)
    resp = svc.predict(PredictRequest("req-001", (0.9, 0.2, 0.7, 0.1), user_id="user-1"))
    print(f"single: score={resp.score:.4f} label={resp.label} version={resp.model_version} variant={resp.variant}")

    # 2. Dynamic batching (concurrent submissions coalesce)
    scorer = registry.get("churn", "v1")
    with DynamicBatcher(scorer.predict_batch, BatchConfig(max_batch_size=8, max_wait_ms=50)) as b:
        import threading
        results: list = []
        threads = [threading.Thread(target=lambda f=f: results.append(b.submit(f))) for f in [(0.1,0.2,0.3,0.4),(0.9,0.1,0.8,0.2),(0.5,0.5,0.5,0.5)]]
        for t in threads: t.start()
        for t in threads: t.join()
        print(f"batching: {len(results)} requests -> batches_executed={b.batches_executed} items={b.items_processed} batch_sizes={[s for _, s in results]}")

    # 3. A/B routing determinism
    router = ABRouter([Variant("control", "v1", 50), Variant("treatment", "v2", 50)])
    dist = router.distribution([f"user-{i}" for i in range(200)])
    print(f"A/B routing over 200 users: {dist} (deterministic — same user always same variant)")

    # 4. Canary with auto-rollback on errors
    canary = CanaryController(CanaryConfig(traffic_pct=50, error_rate_threshold=0.1, min_requests=10))
    for i in range(15):
        canary.record(10.0, error=(i < 5))
    triggered = canary.evaluate()
    print(f"canary: requests={canary.stats.requests} error_rate={canary.stats.error_rate:.2f} rolled_back={canary.rolled_back} triggered={triggered} reason={canary.rollback_reason}")

    # 5. Shadow mode agreement
    shadow = ShadowRunner(registry.get("churn", "v1"), registry.get("churn", "v2"))
    for feats in [(0.9,0.2,0.7,0.1),(0.1,0.9,0.1,0.0),(0.5,0.5,0.5,0.5),(1.0,0.0,1.0,0.5)]:
        shadow.run_one(feats)
    print(f"shadow: {shadow.summary()}")

    # 6. Service with canary + metrics
    svc2 = ModelServingService(build_demo_registry(), canary=CanaryController(CanaryConfig(traffic_pct=20, min_requests=5)))
    for i in range(20):
        svc2.predict(PredictRequest(f"r-{i}", (0.3, 0.4, 0.5, 0.6), user_id=f"user-{i}"))
    print(f"metrics: {svc2.metrics_snapshot()}")
    print("demo OK")


if __name__ == "__main__":
    run_demo()
