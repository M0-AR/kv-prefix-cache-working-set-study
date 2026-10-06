"""Tiny causal attention with explicit K/V cache.

Verifies the prompt's core claim: in causal attention earlier tokens'
keys/values stay valid when new tokens arrive, so incremental decoding
with a cache equals full recomputation.
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def full_prefill(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full O(n^2) causal forward. X: (n,d). Returns (out, K, V)."""
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    n = X.shape[0]
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask, -1e9, scores)
    attn = softmax(scores, axis=-1)
    return attn @ V, K, V


class KVCache:
    """Incremental causal decoder holding K/V rows."""

    def __init__(self, d_k: int, d_v: int):
        self.K = np.zeros((0, d_k))
        self.V = np.zeros((0, d_v))

    def __len__(self) -> int:
        return self.K.shape[0]

    def extend(self, k_new: np.ndarray, v_new: np.ndarray):
        self.K = np.concatenate([self.K, k_new], axis=0)
        self.V = np.concatenate([self.V, v_new], axis=0)
        return self

    def append_token(self, x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray) -> np.ndarray:
        q = x @ Wq  # (d_k,)
        k = x @ Wk  # (d_k,)
        v = x @ Wv  # (d_v,)
        self.K = np.concatenate([self.K, k[None, :]], axis=0)
        self.V = np.concatenate([self.V, v[None, :]], axis=0)
        scores = (self.K @ q) / np.sqrt(q.shape[0])  # (t,)
        attn = softmax(scores, axis=0)
        return attn @ self.V  # (d_v,)


def prefill_into_cache(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray) -> tuple[np.ndarray, KVCache]:
    """Prefill n tokens at once, populate cache, return outputs + cache."""
    out, K, V = full_prefill(X, Wq, Wk, Wv)
    c = KVCache(K.shape[1], V.shape[1])
    c.K = K.copy()
    c.V = V.copy()
    return out, c
