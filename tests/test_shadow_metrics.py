import pytest
from model_serving.metrics import MetricsRegistry, percentile
from model_serving.models import LinearScorer, ModelSpec
from model_serving.shadow import ShadowRunner

def test_percentile_basics():
    assert percentile([],50)==0.0
    assert percentile([10.0],95)==10.0
    assert percentile([1,2,3,4],50)==pytest.approx(2.5)
    assert percentile([1,2,3,4],0)==1 and percentile([1,2,3,4],100)==4
    with pytest.raises(ValueError): percentile([1],101)

def test_percentile_ordering():
    vals=[float(i) for i in range(100)]
    s=MetricsRegistry()
    for v in vals: s.record(v)
    stats=s.latency_stats()
    assert stats.p50 < stats.p95 <= stats.p99 <= stats.max_value
    assert stats.count==100

def test_metrics_error_rate():
    s=MetricsRegistry(); s.record(10.0); s.record(10.0, error=True)
    assert s.total==2 and s.error_rate()==pytest.approx(0.5)

def test_metrics_batch_and_shed():
    s=MetricsRegistry(); s.record_batch(4); s.record_batch(6); s.record_shed()
    snap=s.snapshot(elapsed_s=1.0)
    assert snap.batch_count==2 and snap.avg_batch_size==pytest.approx(5.0) and snap.shed_count==1

def test_metrics_empty_snapshot():
    snap=MetricsRegistry().snapshot()
    assert snap.total_requests==0 and snap.error_rate==0.0

def _runner():
    p=LinearScorer(ModelSpec("m","v1",(1.0,1.0),bias=0.0))
    s=LinearScorer(ModelSpec("m","v2",(1.0,1.0),bias=0.0))
    return ShadowRunner(p,s)

def test_shadow_identical_models_agree():
    r=_runner()
    for f in [(0.0,0.0),(1.0,1.0),(-1.0,0.0)]: r.run_one(f)
    assert r.agreement_rate()==1.0 and r.mean_abs_diff()==pytest.approx(0.0)
    assert r.divergences()==[]

def test_shadow_divergent_models():
    p=LinearScorer(ModelSpec("m","v1",(2.0,),bias=0.0))
    s=LinearScorer(ModelSpec("m","v2",(-2.0,),bias=0.0))
    r=ShadowRunner(p,s); r.run_one((1.0,)); r.run_one((-1.0,))
    assert r.agreement_rate()==0.0 and len(r.divergences())==2

def test_shadow_summary_and_empty():
    r=_runner()
    assert r.agreement_rate()==1.0  # empty -> 1.0 by convention
    r.run_one((0.5,0.5))
    summ=r.summary(); assert summ["total"]==1 and "agreement_rate" in summ
