"""Model Serving at Scale.

Clean-room reference for production model serving: dynamic request
batching, canary releases with auto-rollback, shadow mode with
agreement scoring, deterministic A/B routing, a model version
registry, latency/throughput metrics, deadlines and load-shedding.
Deterministic fake (linear) models — runs fully offline, stdlib core.
"""
from .batcher import BatchConfig, DynamicBatcher, collect_batch
from .canary import CanaryConfig, CanaryController
from .errors import DeadlineExceededError, ModelNotFoundError, OverloadedError, ValidationError
from .metrics import MetricsRegistry, percentile
from .models import LinearScorer, ModelSpec, PredictRequest, PredictResponse
from .registry import ModelRegistry, build_demo_registry
from .router import ABRouter, Variant, hash_bucket
from .service import ModelServingService, build_demo_service
from .shadow import ShadowRunner

__version__ = "0.1.0"
__all__ = ["ABRouter","BatchConfig","CanaryConfig","CanaryController","DeadlineExceededError",
"DynamicBatcher","LinearScorer","MetricsRegistry","ModelNotFoundError","ModelRegistry",
"ModelServingService","ModelSpec","OverloadedError","PredictRequest","PredictResponse",
"ShadowRunner","ValidationError","Variant","build_demo_registry","build_demo_service",
"collect_batch","hash_bucket","percentile"]
