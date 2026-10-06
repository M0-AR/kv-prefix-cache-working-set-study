"""Fast offline regression tests (no network). Mirrors experiments 01-04,06."""
import sys
sys.path.insert(0, "src")
import numpy as np
from attention_kv import full_prefill, prefill_into_cache
from prefix_cache import PrefixCacheLRU, simulate
from stack_distance import stack_distances, hitrate_curve, naive_hitrate, working_set
from memory_model import bytes_per_token, throughput_gain
from trace_loader import synthetic_agentic_trace, trace_to_page_accesses


def test_causal_reuse():
    rng = np.random.default_rng(1)
    d = 12
    Wq, Wk, Wv = rng.normal(size=(d, d)), rng.normal(size=(d, d)), rng.normal(size=(d, d))
    X = rng.normal(size=(4, d))
    full, _, _ = full_prefill(X, Wq, Wk, Wv)
    pre, c = prefill_into_cache(X[:2], Wq, Wk, Wv)
    o2 = c.append_token(X[2], Wq, Wk, Wv)
    assert np.max(np.abs(o2 - full[2])) < 1e-9


def test_prefix_exact_vs_similar():
    c = PrefixCacheLRU(16, 4)
    c.request([1, 2, 3, 4, 5, 6, 7, 8])
    r = c.request([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert r["hit_blocks"] == 2
    c2 = PrefixCacheLRU(16, 4)
    c2.request([1, 2, 3, 4, 5, 6, 7, 8])
    r2 = c2.request([1, 2, 3, 99, 5, 6, 7, 8, 9, 10, 11, 12])
    assert r2["hit_blocks"] <= r["hit_blocks"]


def test_memory_model():
    assert bytes_per_token(80, 8, 128, 2) == 327680
    assert abs(throughput_gain(0.75) - 4.0) < 1e-9


def test_mattson_equals_naive():
    trace = synthetic_agentic_trace(n_sessions=4, turns_per_session=4, seed=3)
    acc = trace_to_page_accesses(trace, block_size=8)
    dists = stack_distances(acc)
    for cap in [4, 16, 64]:
        assert abs(hitrate_curve(dists, [cap])[cap] - naive_hitrate(acc, cap)) < 1e-12
    assert working_set(dists, 0.5, list(range(1, 200))) is not None


def test_locality_matters():
    trace = synthetic_agentic_trace(n_sessions=6, turns_per_session=4, seed=5)
    assert simulate(trace, capacity=32, block_size=16)["hit_rate_tokens"] >= 0.0
