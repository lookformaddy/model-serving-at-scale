import pytest
from model_serving.errors import ModelNotFoundError, RegistryError, ValidationError
from model_serving.models import LinearScorer, ModelSpec, PredictRequest
from model_serving.registry import ModelRegistry, build_demo_registry

def test_linear_scorer_deterministic():
    m = LinearScorer(ModelSpec("m","v1",(1.0,1.0),bias=0.0))
    assert m.predict_one((0.0,0.0)) == pytest.approx(0.5)
    assert m.predict_one((1.0,1.0)) > 0.8
    assert m.predict_one((0.5,0.5)) == m.predict_one((0.5,0.5))

def test_scorer_batch_matches_single():
    m = LinearScorer(ModelSpec("m","v1",(0.5,-0.5,1.0)))
    batch = [(0.1,0.2,0.3),(1.0,0.0,0.5)]
    assert m.predict_batch(batch) == [m.predict_one(f) for f in batch]

def test_scorer_label_threshold():
    m = LinearScorer(ModelSpec("m","v1",(1.0,),threshold=0.5))
    assert m.label(0.9)==1 and m.label(0.1)==0

def test_scorer_wrong_dim():
    m = LinearScorer(ModelSpec("m","v1",(1.0,1.0)))
    with pytest.raises(ValidationError): m.predict_one((1.0,))

def test_scorer_fail_hook():
    m = LinearScorer(ModelSpec("m","v1",(1.0,)), fail=True)
    with pytest.raises(RuntimeError): m.predict_one((1.0,))

def test_spec_validation():
    with pytest.raises(ValidationError): ModelSpec("","v1",(1.0,))
    with pytest.raises(ValidationError): ModelSpec("m","v1",())
    with pytest.raises(ValidationError): ModelSpec("m","v1",(1.0,),threshold=1.5)

def test_request_validation():
    with pytest.raises(ValidationError): PredictRequest("", (1.0,))
    with pytest.raises(ValidationError): PredictRequest("r", ())

def test_registry_register_and_active():
    reg = build_demo_registry()
    assert reg.active_version("churn")=="v1"
    assert set(reg.list_versions("churn"))=={"v1","v2"}
    assert "fraud" in reg.list_models()

def test_registry_promote_and_rollback():
    reg = build_demo_registry()
    reg.promote("churn","v2"); assert reg.active_version("churn")=="v2"
    prev = reg.rollback("churn"); assert prev=="v1" and reg.active_version("churn")=="v1"

def test_registry_duplicate_rejected():
    reg = ModelRegistry()
    reg.register(LinearScorer(ModelSpec("m","v1",(1.0,))))
    with pytest.raises(RegistryError): reg.register(LinearScorer(ModelSpec("m","v1",(1.0,))))

def test_registry_not_found():
    reg = build_demo_registry()
    with pytest.raises(ModelNotFoundError): reg.get("nope")
    with pytest.raises(ModelNotFoundError): reg.get("churn","v99")

def test_registry_rollback_without_history():
    reg = ModelRegistry(); reg.register(LinearScorer(ModelSpec("m","v1",(1.0,))))
    with pytest.raises(RegistryError): reg.rollback("m")

def test_registry_len():
    assert len(build_demo_registry())==3
