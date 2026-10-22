import pytest
from model_serving.canary import CanaryConfig, CanaryController
from model_serving.errors import DeadlineExceededError, OverloadedError
from model_serving.models import PredictRequest
from model_serving.router import ABRouter, Variant
from model_serving.service import ModelServingService, build_demo_service
from model_serving.shadow import ShadowRunner
from model_serving.registry import build_demo_registry

def req(rid="r1", feats=(0.9,0.2,0.7,0.1), **kw):
    return PredictRequest(rid, feats, **kw)

def test_predict_happy(service):
    resp=service.predict(req())
    assert 0.0 < resp.score < 1.0 and resp.model_version=="v1" and resp.variant=="stable"
    assert resp.label in (0,1)

def test_predict_fraud_model(service):
    resp=service.predict(PredictRequest("r1",(0.5,0.5,0.5),model_name="fraud"))
    assert resp.model_name=="fraud"

def test_predict_batch_endpoint(service):
    resps=service.predict_batch([req(f"r{i}") for i in range(5)])
    assert len(resps)==5

def test_predict_batched_path(service):
    resp=service.predict(req(), use_batcher=True)
    assert resp.batched is True and resp.batch_size>=1

def test_predict_unknown_model(service):
    with pytest.raises(Exception): service.predict(PredictRequest("r",(1.0,),model_name="nope"))

def test_deadline_exceeded_with_fake_clock():
    times=iter([0.0, 10.0])  # start, end -> 10000ms latency
    svc=ModelServingService(build_demo_registry(), clock=lambda: next(times))
    with pytest.raises(DeadlineExceededError):
        svc.predict(PredictRequest("r",(0.1,0.2,0.3,0.4),deadline_ms=100.0))

def test_load_shedding():
    svc=build_demo_service(max_inflight=0)
    with pytest.raises(OverloadedError): svc.predict(req())

def test_ab_routing_in_service():
    router=ABRouter([Variant("control","v1",100),Variant("treatment","v2",0)])
    svc=ModelServingService(build_demo_registry(), router=router)
    resp=svc.predict(req(user_id="user-9"))
    assert resp.variant=="control" and resp.model_version=="v1"

def test_shadow_attached():
    reg=build_demo_registry()
    shadow=ShadowRunner(reg.get("churn","v1"), reg.get("churn","v2"))
    svc=ModelServingService(reg, shadow=shadow)
    resp=svc.predict(req())
    assert resp.shadow_score is not None and shadow.total==1

def test_canary_routing_in_service():
    canary=CanaryController(CanaryConfig(traffic_pct=100, min_requests=1000))
    svc=ModelServingService(build_demo_registry(), canary=canary)
    resp=svc.predict(req(user_id="user-1"))
    assert resp.variant=="canary" and resp.model_version=="v2"

def test_health_and_metrics(service):
    service.predict(req())
    h=service.health(); assert h["status"]=="ok" and "churn" in h["models"]
    m=service.metrics_snapshot(); assert m["total_requests"]>=1 and "latency_p95_ms" in m

def test_response_to_dict(service):
    d=service.predict(req()).to_dict()
    assert set(d)>={"request_id","score","label","variant"}

def test_create_app_guarded():
    pytest.importorskip("fastapi")
    from model_serving.service import create_app
    from fastapi.testclient import TestClient
    client=TestClient(create_app(build_demo_service()))
    assert client.get("/health").status_code==200
    r=client.post("/predict", json={"request_id":"x","features":[0.9,0.2,0.7,0.1]})
    assert r.status_code==200 and "score" in r.json()
    assert client.get("/metrics").status_code==200
    assert client.get("/models").status_code==200

def test_cli_models_and_predict(capsys):
    from model_serving.cli import main
    assert main(["models"])==0; assert "churn" in capsys.readouterr().out
    assert main(["predict","--features","0.9,0.2,0.7,0.1"])==0
    assert "score" in capsys.readouterr().out

def test_demo_runs(capsys):
    from model_serving.demo import run_demo
    run_demo(); out=capsys.readouterr().out
    assert "demo OK" in out and "batching" in out and "shadow" in out
