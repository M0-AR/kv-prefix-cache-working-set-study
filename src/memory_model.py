"""KV-bytes + prefill-throughput cost model (from 2026 literature).

bytes/token = 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem
T(r) = T0 / (1 - r)  (KVSET Eq.1 idealized prefill throughput at hit rate r)
"""
from __future__ import annotations


def bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem


def cache_gib(n_tokens: int, n_layers: int, n_kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> float:
    return n_tokens * bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_elem) / (1024**3)


# Reference configs (rounded public specs)
MODELS = {
    "llama-8B-GQA": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
    "llama-70B-GQA": {"n_layers": 80, "n_kv_heads": 8, "head_dim": 128},
    "qwen-14B-GQA": {"n_layers": 48, "n_kv_heads": 8, "head_dim": 128},
}


def throughput_gain(hit_rate: float) -> float:
    """Idealized T/T0 = 1/(1-r)."""
    assert 0.0 <= hit_rate < 1.0
    return 1.0 / (1.0 - hit_rate)
