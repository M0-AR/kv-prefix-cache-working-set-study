"""Trace generation + public-data loaders (live-first, deterministic fallback).

- synthetic_agentic_trace(): deterministic multi-session agent workload with
  a long shared system prefix + growing per-session history (session reuse)
  plus a repeated tool-output template pool (structural reuse, cf. UniCache 2026).
- load_burstgpt_sample(): tries live download of BurstGPT_1.csv (first N rows);
  falls back to synthetic length distribution with a flag.
- No local-repo reads: everything is generated or fetched live here.
"""
from __future__ import annotations

import csv
import random
import urllib.request


def synthetic_agentic_trace(
    n_sessions: int = 12,
    turns_per_session: int = 8,
    system_tokens: int = 64,
    per_turn_new_tokens: int = 24,
    structural_pool: int = 6,
    seed: int = 0,
) -> list[list[int]]:
    """Each request = full token list (system + history-so-far + new turn).

    Token ids are synthetic but prefix-structured: session s owns base range,
    history grows monotonically so consecutive turns share exact prefixes.
    The structural pool injects cross-session repeated chunks (tool templates).
    """
    rng = random.Random(seed)
    SYS = list(range(1, 1 + system_tokens))
    pool_chunks = [[100000 + p * 100 + i for i in range(16)] for p in range(structural_pool)]
    trace: list[list[int]] = []
    for s in range(n_sessions):
        base = 200000 + s * 100000
        history: list[int] = []
        for t in range(turns_per_session):
            new = [base + t * per_turn_new_tokens + i for i in range(per_turn_new_tokens)]
            # every 3rd turn carries a shared structural chunk (cross-session reuse)
            struct = pool_chunks[rng.randrange(structural_pool)] if t % 3 == 2 else []
            history = history + struct + new
            trace.append(SYS + history)
    # interleave sessions round-robin (realistic concurrency, stresses LRU)
    interleaved: list[list[int]] = []
    per_session = [trace[s * turns_per_session:(s + 1) * turns_per_session] for s in range(n_sessions)]
    for t in range(turns_per_session):
        for s in range(n_sessions):
            interleaved.append(per_session[s][t])
    return interleaved


def trace_to_page_accesses(trace: list[list[int]], block_size: int = 16) -> list[tuple]:
    """Expand requests to page-key accesses (prefix+block tuple, vLLM hash model)."""
    acc: list[tuple] = []
    for toks in trace:
        n_full = len(toks) // block_size
        for b in range(n_full):
            acc.append(tuple(toks[: (b + 1) * block_size]))
    return acc


def load_burstgpt_sample(max_rows: int = 2000, timeout: int = 15) -> dict:
    """Live-fetch BurstGPT_1.csv head; fallback to synthetic Zipf lengths.

    Returns {"source": "live-burstgpt"|"synthetic-fallback", "req_tokens": [...], ...}.
    """
    url = "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            head = "".join(r.read(300000).decode("utf-8", "ignore").splitlines(True)[: max_rows + 1])
        rows = list(csv.DictReader(head.splitlines()))
        req = [int(f["Request tokens"]) for f in rows if f.get("Request tokens", "").isdigit()][:max_rows]
        rsp = [int(f["Response tokens"]) for f in rows if f.get("Response tokens", "").isdigit()][:max_rows]
        if len(req) < 50:
            raise ValueError("too few live rows")
        return {"source": "live-burstgpt", "url": url, "req_tokens": req, "rsp_tokens": rsp, "n": len(req)}
    except Exception as e:  # offline sandbox -> deterministic fallback
        rng = random.Random(1234)
        req = [max(4, int(rng.paretovariate(1.16) * 22)) for _ in range(max_rows)]
        rsp = [max(1, int(rng.paretovariate(1.25) * 18)) for _ in range(max_rows)]
        return {"source": f"synthetic-fallback({type(e).__name__})", "url": url, "req_tokens": req, "rsp_tokens": rsp, "n": len(req)}
