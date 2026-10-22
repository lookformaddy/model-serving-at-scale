import threading
import pytest
from model_serving.batcher import BatchConfig, DynamicBatcher, collect_batch
from model_serving.errors import ValidationError
from model_serving.models import LinearScorer, ModelSpec

def scorer_fn():
    m = LinearScorer(ModelSpec("m","v1",(1.0,1.0)))
    return m.predict_batch

def test_batch_config_validation():
    with pytest.raises(ValidationError): BatchConfig(max_batch_size=0)
    with pytest.raises(ValidationError): BatchConfig(max_wait_ms=0)
    with pytest.raises(ValidationError): BatchConfig(max_queue_size=0)

def test_collect_batch_chunks():
    assert collect_batch([1,2,3,4,5],2)==[[1,2],[3,4],[5]]
    assert collect_batch([],3)==[]
    assert collect_batch([1,2],10)==[[1,2]]
    with pytest.raises(ValidationError): collect_batch([1],0)

def test_batcher_single_submit():
    with DynamicBatcher(scorer_fn(), BatchConfig(max_batch_size=4, max_wait_ms=30)) as b:
        score, size = b.submit((0.0,0.0))
        assert score == pytest.approx(0.5); assert size>=1

def test_batcher_coalesces_concurrent():
    with DynamicBatcher(scorer_fn(), BatchConfig(max_batch_size=8, max_wait_ms=100)) as b:
        results=[]
        threads=[threading.Thread(target=lambda f=f: results.append(b.submit(f))) for f in [(0.1,0.1),(0.2,0.2),(0.3,0.3),(0.4,0.4)]]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(results)==4
        # at least one batch larger than 1 proves coalescing
        assert max(s for _,s in results) > 1
        assert b.items_processed==4

def test_batcher_max_size_flush():
    with DynamicBatcher(scorer_fn(), BatchConfig(max_batch_size=2, max_wait_ms=5000)) as b:
        results=[]
        threads=[threading.Thread(target=lambda f=f: results.append(b.submit(f))) for f in [(0.0,0.0),(1.0,1.0)]]
        for t in threads: t.start()
        for t in threads: t.join(timeout=3)
        assert len(results)==2  # flushed by size, not by 5s wait

def test_batcher_error_propagates():
    def boom(batch): raise RuntimeError("boom")
    with DynamicBatcher(boom, BatchConfig(max_wait_ms=20)) as b:
        with pytest.raises(RuntimeError): b.submit((1.0,1.0))
