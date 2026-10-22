import pytest
from model_serving.canary import CanaryConfig, CanaryController
from model_serving.errors import ValidationError
from model_serving.router import ABRouter, Variant, hash_bucket

def test_hash_bucket_stable_and_range():
    assert hash_bucket("user-1")==hash_bucket("user-1")
    assert 0 <= hash_bucket("anything") < 100
    assert 0 <= hash_bucket("x", buckets=10) < 10

def test_hash_bucket_spreads():
    buckets={hash_bucket(f"user-{i}") for i in range(100)}
    assert len(buckets) > 20

def test_router_deterministic():
    r=ABRouter([Variant("a","v1",50),Variant("b","v2",50)])
    assert r.route("user-42")==r.route("user-42")

def test_router_weights_must_sum_100():
    with pytest.raises(ValidationError): ABRouter([Variant("a","v1",30)])
    with pytest.raises(ValidationError): ABRouter([])

def test_router_single_variant():
    r=ABRouter([Variant("only","v1",100)])
    assert all(r.route(f"u{i}").name=="only" for i in range(20))

def test_router_zero_weight_never_chosen():
    r=ABRouter([Variant("a","v1",100),Variant("b","v2",0)])
    assert all(r.route(f"u{i}").name=="a" for i in range(50))

def test_router_distribution_counts():
    r=ABRouter([Variant("a","v1",50),Variant("b","v2",50)])
    d=r.distribution([f"user-{i}" for i in range(100)])
    assert sum(d.values())==100 and d["a"]>0 and d["b"]>0

def test_variant_weight_validation():
    with pytest.raises(ValidationError): Variant("a","v1",101)

def test_canary_zero_traffic():
    c=CanaryController(CanaryConfig(traffic_pct=0))
    assert not any(c.should_route_canary(f"u{i}") for i in range(50))

def test_canary_full_traffic():
    c=CanaryController(CanaryConfig(traffic_pct=100))
    assert all(c.should_route_canary(f"u{i}") for i in range(20))

def test_canary_deterministic_routing():
    c=CanaryController(CanaryConfig(traffic_pct=50))
    assert c.should_route_canary("user-1")==c.should_route_canary("user-1")

def test_canary_rollback_on_errors():
    c=CanaryController(CanaryConfig(traffic_pct=10, error_rate_threshold=0.1, min_requests=10))
    for i in range(10): c.record(5.0, error=(i<3))
    assert c.evaluate() is True and c.rolled_back and c.rollback_reason
    assert c.should_route_canary("anyone") is False

def test_canary_rollback_on_latency():
    c=CanaryController(CanaryConfig(traffic_pct=10, p95_latency_ms=100.0, min_requests=5))
    for _ in range(10): c.record(500.0)
    assert c.evaluate() is True and "p95" in (c.rollback_reason or "")

def test_canary_no_rollback_below_min_requests():
    c=CanaryController(CanaryConfig(min_requests=100, error_rate_threshold=0.0))
    for _ in range(5): c.record(5.0, error=True)
    assert c.evaluate() is False and not c.rolled_back

def test_canary_healthy_no_rollback():
    c=CanaryController(CanaryConfig(min_requests=5))
    for _ in range(10): c.record(10.0)
    assert c.evaluate() is False

def test_canary_promote_rolled_back_fails():
    c=CanaryController(CanaryConfig(min_requests=2, error_rate_threshold=0.1))
    c.record(1.0, error=True); c.record(1.0, error=True); c.evaluate()
    with pytest.raises(ValidationError): c.promote()

def test_canary_status_shape():
    s=CanaryController().status()
    assert set(s)>={"traffic_pct","requests","rolled_back"}
