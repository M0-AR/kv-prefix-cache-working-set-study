# Prefix Caching Is a Memory–Compute Trade, Not a Free Lunch: Verifying KV Reuse, LRU Working Sets, and the Capacity–Hit-Rate Knee from First Principles to Live Traces

**A reproducible, from-scratch empirical study. No local code was reused; every claim below is backed by an executable experiment in this repo. All numbers are measured outputs, not illustrations.**

| Artifact | Location |
|---|---|
| Experiments 01–06 | `experiments/` |
| Library (attention, LRU cache, Mattson, memory model, traces) | `src/` |
| Measured results (regenerated on every run) | `results/` |
| Reproducible runner | `Dockerfile`, `docker-compose.yml`, `Makefile` |
| Regression tests | `tests/` |

**Reproduce in one command:**

```bash
docker compose up --build
# or without docker:
pip install -r requirements.txt
python experiments/run_all.py
python -m pytest tests/ -q
```

Verified environment: Python 3.12, `numpy==1.26.4`, Docker 29.1.3 / Compose 2.40.3. The container run reproduces every number in this paper bit-identically (see Section 6).

---

## Abstract

Agentic LLM workloads resend growing conversations — system instructions, dialogue history, tool-use logs — on every model call. Without reuse, the server re-runs the *prefill* stage (prompt processing) over the full context before generating the next token. This study verifies, end to end and from scratch, the standard account of how production systems avoid that repeated work:

1. **Mechanism.** In causal attention each token emits a key (K) and value (V); a later query scores past keys to weight past values. Because earlier tokens never attend to later ones, their K/V rows stay valid as the sequence grows and can be cached.
2. **Cross-request reuse.** A serving system can retain those K/V blocks and reuse the longest matching *token* prefix of a new request, skipping prefill for the cached span. The match is on exact tokens (plus model and cache configuration), never on semantic similarity.
3. **Cost.** Each retained token costs K/V bytes on every layer. Finite capacity plus LRU eviction bounds reuse; evicted state must be recomputed.
4. **Sizing.** The hit-rate-vs-capacity curve has a knee: beyond some point more memory buys little extra reuse. The minimum capacity reaching a target hit rate — the *working set* — can be estimated in one pass with Mattson's (1970) stack-distance algorithm instead of one simulation per capacity (the KVSET method, Li et al., Sept 2026).

We implement a toy causal attention core, a vLLM-faithful block LRU prefix cache, a one-pass Mattson analyzer, a byte-level memory model, and a deterministic agentic trace generator; we replay both synthetic and **live public data** (BurstGPT, fetched at run time). We confirm: cached incremental decoding equals full recomputation to 1.2e-14; exact prefixes hit while paraphrases (nearly) do not; a 70B-GQA-class model costs 320 KiB/token (10 GiB per 32K context); Mattson's curve matches naive per-capacity simulation to 0.0 exactly; the demo workload needs 122 pages for 70% hits and 136 for 80%, while 90% is *unreachable* on that trace (max 0.845) — the same non-convergence at extreme coverage the KVSET paper reports on production traces. We surface four hidden patterns, the most actionable being a **measurement pitfall**: comparing block sizes at equal *block counts* inflates coarse-block hit rates by up to 16x in memory; all fair comparisons must fix the *token* budget.

**Contributions.**

- (C1) A minimal, checkable verification of every sentence of the causal-KV-reuse account (Sections 2–4).
- (C2) An independent replication of the KVSET one-pass theorem with exact equality against naive simulation, plus working-set extraction and knee detection (EXP-04).
- (C3) Live-data grounding: BurstGPT length statistics fetched at experiment time drive a scale-aware replay (EXP-05).
- (C4) Four measured hidden patterns, including the block-count normalization pitfall and a 0.458 session-locality gap under concurrency (EXP-06).
- (C5) A fully reproducible artifact (Docker + pinned `numpy`, seeded traces, offline tests) suitable as a PhD-paper methods supplement.

---

## 1. Introduction

### 1.1 Problem

An AI agent that reads a paper, calls a tool, and continues the conversation resends its full history on each turn. A naive server would redo all prompt computation every time. At 128K context on a 70B-class model a single request's K/V state is ~30 GiB; ten concurrent users exceed a whole HBM package. Provisioning for "keep everything" is infeasible; provisioning blindly small destroys hit rates. Operators need a principled answer to: *how much cache gives the hit rate we need?*

### 1.2 What this paper does

We rebuild the answer from zero: attention math, prefix-match semantics, memory accounting, eviction dynamics, working-set estimation, and validation on public traces — each step as runnable code with asserted checks. We then push one step further and document reuse patterns that only appear once you measure.

### 1.3 Scope and non-claims

- Single-layer toy attention verifies *correctness of reuse*, not model quality.
- Our simulator models full-attention, LRU, exact-prefix block caching (the vLLM/SGLang mainstream). Sliding-window, Mamba/SSM, sparse/checkpointed, and compressed caches are out of scope — and, per the literature, change the conclusions (see Section 3).
- Trace results describe *today's* workload shape, not a constant of nature. Capacity numbers are in simulator pages; translate to bytes via Section 4.

---

## 2. Background: prefill, attention, and why caching is legal

### 2.1 Prefill vs. decode

LLM generation has two phases. **Prefill** processes the whole prompt at once (compute-bound, quadratic in length for dense attention) and emits the first token. **Decode** emits one token at a time (memory-bound), each step attending over all prior tokens. Time-to-first-token (TTFT) is dominated by prefill; time-per-output-token (TPOT) by decode. Anything that shrinks prefill — such as skipping already-computed prefixes — directly cuts TTFT and, via the idealized model `T = T0 / (1 − r)` at hit rate `r`, multiplies prefill throughput (4x at `r = 0.75`, 10x at `r = 0.90`; verified in EXP-03).

### 2.2 One attention layer, precisely

For input rows `X` (n × d) with projections `Wq, Wk, Wv`, head output is `softmax(QKᵀ/√dk + causal_mask)V`. Each position `i` produces one key row and one value row. Position `i`'s query scores keys `1..i` only — never future keys. Hence when token `n+1` arrives, rows `1..n` are unchanged: their stored K/V remain the correct inputs for the new query. Full recomputation and incremental cached decoding are mathematically identical (EXP-01 measures max abs error **1.15e-14**).

### 2.3 From one sequence to many requests

An agent's turn `t+1` contains turn `t`'s tokens as a prefix (plus new tool outputs). Across requests the server can therefore index cached K/V blocks by their token content and reuse the longest cached prefix. vLLM hashes each block by its tokens *plus all prefix tokens before it*; SGLang traverses a radix tree of token sequences. Both enforce the same invariant our simulator enforces: **only complete leading blocks hit; the first miss truncates the hit prefix; partial tails never hit.** Model weights, adapter, block size, and cache salt are part of the effective key — change any and the hit disappears even for identical text.

---

## 3. Related work (2024–2026)

**Prefix-caching substrates.** vLLM's PagedAttention partitions K/V into blocks in non-contiguous memory and adds hash-based automatic prefix caching with an LRU free queue, reference counts, and per-request `cache_salt` isolation (mitigating CVE-2025-46570 prefix-timing side channels, ROC AUC 0.99 at 8 tokens). SGLang's RadixAttention uses a token-sequence radix tree with reference-counted nodes. Our `PrefixCacheLRU` implements the shared semantics (complete-block prefix hits, LRU eviction) without claiming either engine's internals.

**Working-set estimation (the "K V Set" paper).** Li et al., *The KV Cache Working Set* (arXiv:2609.27746, Sept 2026, Kingsoft Cloud) define the working set as the minimum capacity achieving a target hit rate and estimate it online with Mattson's stack algorithm plus a Fenwick tree: one pass yields the whole hit-rate curve under LRU; capacity follows from max LRU depth at the target. On a 24K-request coding-agent trace ~1 TiB preserves per-request hit rates for 95% of requests and ~5 TiB for 99%, with **no convergence at 99.9%**. Open implementation: `llc-kc/kv_cache_capacity_estimator`. Our EXP-04 is an independent replication of the one-pass theorem and reproduces the non-convergence phenomenon on our trace (90% unreachable, max 0.845).

**Reuse-pattern heterogeneity.** UniCache (SIGMETRICS 2026) shows conversational vs. API traffic follow different patterns — *session reuse* vs. *structural reuse* — and a unified eviction policy gains up to 17.32% hit rate / 3.63x latency. Our EXP-06 H1 directly measures the session-locality axis (sequential vs. interleaved).

**Granularity, offload, and restore-vs-recompute.** GraniKV (asymmetric HOT/COLD paging, up to 2.16x), ContiguousKV (chunk-aligned offload, 3.85x re-prefill), Tutti (GPU-centric SSD path, −78.3% TTFT), LMCache tiering on GKE, and the HotStorage'26 restore-vs-recompute policy (up to 10x over static policies) all address what happens when the working set exceeds HBM. FastKV (1.82x prefill / 2.87x decode) decouples prefill reduction from decode budgets. PrefillShare (4.5x lower p95 across models) tackles cross-model prefix incompatibility. Sparse-prefix (checkpoint placement DP), SparseX (segment-level non-prefix reuse), BiCache (diffusion LMs), and the multi-instruction compression pitfalls paper (uneven degradation, eviction bias, system-prompt leakage) mark the boundaries of our scope: we study dense, full-attention, exact-prefix, LRU caching only.

**Traces.** BurstGPT (2401.17644; 10.31M Azure OpenAI traces over 213 days, public CSVs with request/response lengths, session IDs, timestamps) and ShareGPT conversation dumps are the community standards for realistic length/concurrency distributions. We fetch BurstGPT live in EXP-05 and synthesize prefix-structured agent traffic with matching scale.

**Foundations.** Mattson, Gecsei, Slutz & Traiger (IBM Systems J. 1970) — one-pass LRU miss-ratio curves from stack distances. Our implementation is a direct, credited application, not a novel algorithm.

---

## 4. Cost model: bytes per token and throughput leverage

For BF16/FP16 K/V on every layer:

```
bytes/token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_elem
```

Measured in EXP-03 (asserted in code):

| Model class | Config | Bytes/token | Per 32K context |
|---|---|---|---|
| Llama-8B-GQA | 32 layers, 8 KV heads, d=128 | 131,072 (128 KiB) | 4.0 GiB |
| Llama-70B-GQA | 80 layers, 8 KV heads, d=128 | 327,680 (320 KiB) | 10.0 GiB |
| Qwen-14B-GQA | 48 layers, 8 KV heads, d=128 | 196,608 (192 KiB) | 6.0 GiB |

The 320 KiB/token figure matches published 2026 engineering guides; 100 concurrent 32K sessions approach 1 TiB — the regime where KVSET's 1–5 TiB working sets bite. Idealized throughput leverage `1/(1−r)` gives **4.0x at r=0.75 and 10.0x at r=0.90** (EXP-03, asserted). Real gains are lower once lookup, transfer, and queuing are counted — our numbers are upper bounds, stated as such.

---

## 5. Methods

### 5.1 Modules (`src/`)

- `attention_kv.py` — full causal forward + incremental `KVCache` (prefill-then-decode).
- `prefix_cache.py` — `PrefixCacheLRU(capacity_blocks, block_size)`; block key = tuple of all tokens through block end (vLLM hash abstraction); longest complete-block prefix hits; LRU eviction via `OrderedDict`; token and block hit rates.
- `stack_distance.py` — exact Mattson stack distances (first use = inf), `hitrate_curve` (`P(distance ≤ C)`), `working_set` (min capacity at target), and `naive_hitrate` (independent LRU sim = ground truth).
- `memory_model.py` — byte accounting + `1/(1−r)` model + reference GQA configs.
- `trace_loader.py` — deterministic agentic generator (shared system prefix + growing history + cross-session structural pool, round-robin interleaved); `trace_to_page_accesses` (request → page keys); `load_burstgpt_sample` (live HTTP fetch of `BurstGPT_1.csv`, first 2000 rows; deterministic Zipf/Pareto fallback only if offline, with source flag recorded).

### 5.2 Experiments (`experiments/`)

| ID | Claim under test | Assertion |
|---|---|---|
| 01 | Cached incremental decode == full recompute | max abs err < 1e-9 |
| 02 | Exact token prefix hits; paraphrase does not | exact hits > 0 and ≥ paraphrase hits |
| 03 | Memory model + eviction recompute + throughput leverage | 70B bytes == 327680; small-cache hit < big-cache hit |
| 04 | Mattson one-pass == naive sim; working set + knee | max curve diff < 1e-12; ws70 ≤ ws80 |
| 05 | Live public-data replay | live rows ≥ 50; ws80 exists |
| 06 | Hidden patterns H1/H2/H4 | sequential ≥ interleaved; compulsory > 0 |

### 5.3 Best-practice compliance

One web-search at a time during research (rate-limit discipline); pinned dependencies; seeded RNGs (`PYTHONHASHSEED=0` in compose); live-data source recorded in output JSON (no silent fallback); Docker reproduction bit-identical to host; fast offline `pytest` suite independent of network.

---

## 6. Results (all values are executed outputs in `results/`)

### EXP-01 — Reuse is exact

Max abs deviation between cached incremental decoding and full recomputation: **1.15e-14** (float noise; `results/01.json`). The K/V-reuse step introduces no approximation.

### EXP-02 — Prefix means tokens, not meaning

With block size 4: exact continuation hits **3/3 blocks**; a paraphrase with nearly identical meaning but different tokens hits **1/3** (only the leading `system you are a`-style block that happens to share tokens). `results/02.json`. Similarity ≠ cacheability.

### EXP-03 — Memory is the binding constraint; eviction forces recompute

Table in Section 4 (all asserted). Thrashing demo on two alternating 32-token requests: 4-block cache hit rate **0.0** vs. 32-block cache **0.667**. Throughput leverage confirmed at **4.0x / 10.0x**. `results/03.json`.

### EXP-04 — One pass equals N simulations; working set with a knee — and a ceiling

Synthetic agent trace: 10 sessions × 8 turns → **930 page accesses** (block 16). Mattson vs. naive per-capacity LRU agree to **0.0** at every probed capacity:

| Capacity (pages) | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|
| Hit rate (both methods) | 0.000 | 0.086 | 0.303 | 0.351 | 0.415 | 0.755 | 0.845 | 0.845 |

Working sets: **122 pages @ 70%**, **136 pages @ 80%**, **90% unreachable** (max 0.845); knee (marginal doubling gain < 0.02) at **256 pages**. `results/04.json`, `results/04_curve.csv`. The unreachable-90% finding replicates KVSET's production non-convergence at extreme coverage, on our own trace and in miniature.

### EXP-05 — Live public data (BurstGPT)

Fetch at run time succeeded (`burst_source: live-burstgpt`, `results/05.json`): **n=2000 live rows**, mean request **551.7 tokens**, p50 **334**, p99 **2018**. Scale-aware replay (2220 page accesses) yields hit curve 0.022 → 0.814 across capacities 8 → 512 and working set **404 pages @ 80%**. Heavy-tailed lengths (p99 ≈ 6x p50) confirm why synthetic Poisson benchmarks understate SLO pressure, consistent with ShareGPT workload studies.

### EXP-06 — Hidden patterns (measured, not hypothesized)

- **H1 — Concurrency shreds session locality.** Same requests, same 64-page cache: sequential-by-session hit **0.846** vs. round-robin interleaved **0.388** (**Δ = 0.458**). Scheduling/admission that preserves session affinity is worth more than raw capacity in this regime — the UniCache session-vs-structural split, quantified.
- **H2 — Block-size "tax" is real but easily mismeasured (methodological pitfall).** At a fixed **1024-token** budget: bs=4 → 0.375, bs=16 → 0.388, bs=64 → 0.463. Coarse blocks still win here because long stable prefixes amortize partial-tail waste. But comparing at equal **block counts** (64 blocks each) gives 0.253 vs. 0.846 — a **16x memory handicap** disguised as an algorithmic win. Rule: always normalize by tokens/bytes, never by block count.
- **H4 — Cold start is compulsory.** First-epoch unique-page share **0.244**: roughly a quarter of early accesses *cannot* hit at any capacity. Warm-up and admission policy, not sizing, own this slice.

### Reproduction record

- `python experiments/run_all.py` → **ALL EXPERIMENTS PASSED** (host).
- `python -m pytest tests/ -q` → **5 passed**.
- `docker compose build` → exit 0; `docker compose up --build` → exit 0 with **identical JSON** to host run (deterministic seeds; live BurstGPT fetch also succeeded inside the container).

---

## 7. Discussion

**What the curve teaches.** Hit rate is concave in capacity with a flat tail: the knee (256 pages in EXP-04) is where operators should stop buying memory and start buying locality (session pinning, structural-template consolidation, admission control). The working set is not a property of the model — it is a property of *(workload × target × eviction policy)*. Change the target from 70% to 80% and the answer moves 122 → 136; demand 90% and the answer is "not on this trace at any tested capacity."

**Why paraphrases miss.** Block hashes bind prefix + block tokens. One substituted word changes that block's key *and every later block's key* (they embed the prefix). Semantic closeness is invisible to the cache — a security feature (no cross-user leakage without shared salt) and a performance fact (prompt-engineering churn defeats reuse).

**KV cache is state, not knowledge.** Nothing in Sections 2–6 changes model weights or stores facts. Evict the cache and the model answers identically, just slower. Conflating the two leads to over-retention (privacy risk) and under-provisioning (latency risk).

## 8. Threats to validity

- Toy single-head attention; production kernels (FlashAttention, GQA/MLA, quantization) change constants, not the reuse theorem — but absolute byte/latency numbers must be remeasured per deployment.
- LRU-only analysis (Mattson-exact). FIFO/LFU/SLRU/priority policies need their own simulations; our curve is a baseline for them, not a bound.
- Synthetic prefix structure is regular by construction; real agent harnesses (compaction, truncation, tool-breadth changes) shift reuse and stale-date any working set — hence KVSET's online (continuous) framing, which we endorse.
- Live BurstGPT sample is 2000 head rows (lengths only, no text for privacy); full-distribution studies should page the whole release and stratify by service type (conversation vs. API) and model.

## 9. Security & ethics note

Prefix reuse is a timing side channel (CVE-2025-46570): TTFT differences reveal cached prefixes across tenants. Deployments must use per-tenant `cache_salt` (vLLM) or equivalent isolation, and treat shared-tier provenance (adapter, weights, sharing domain) as part of the cache key — see the 2026 provenance-blind-reuse study. Our traces are synthetic or length-only public data; no user text is stored.

## 10. What to try next (PhD-paper extensions)

1. **Online working-set tracking** on a live gateway: stream stack distances, alert when ws80 drifts >20% week-over-week.
2. **Session-aware admission**: route same-session turns to the same prefill worker (PrefillShare-style pinning) and re-measure H1 on production traces.
3. **Token-budget-normalized block-size sweep** across ShareGPT/BurstGPT mixes to find where the H2 coarse-block advantage inverts.
4. **Heterogeneous policy bake-off** (LRU vs. SLRU vs. frequency-aware) against tree-constrained Belady oracle on the same trace.
5. **Restore-vs-recompute frontier** per hardware tier (HBM → DRAM → SSD) with measured TTFT SLOs, not just hit rates.

## 11. How to extend this repo

- New workload: add a generator in `src/trace_loader.py`, expand via `trace_to_page_accesses`, analyze with `stack_distances`/`hitrate_curve`/`working_set`.
- New policy: subclass the `PrefixCacheLRU.request` loop; verify against `naive_hitrate` at matched capacities.
- New figure: `results/04_curve.csv` is the paper-figure source; plot capacity (log-x) vs. hit rate with knee + working-set markers.

## References

- Li, Wang, Ruan, Li & Gong. *The KV Cache Working Set: Online Capacity Planning for LLM Inference Systems.* arXiv:2609.27746 (2026). — working-set definition; Mattson + Fenwick one-pass; 1 TiB @95% / 5 TiB @99% / no convergence @99.9%; open code `llc-kc/kv_cache_capacity_estimator`.
- Mattson, Gecsei, Slutz & Traiger. *Evaluation Techniques for Storage Hierarchies.* IBM Systems J. 9(2) (1970). — stack-distance theorem.
- vLLM docs. *Automatic Prefix Caching*; *Hybrid KV Cache Manager*; *Security: cache_salt / CVE-2025-46570.* — hash(prefix+block), LRU free queue, isolation.
- Wang et al. *UniCache.* SIGMETRICS (2026). — session vs. structural reuse; +17.32% / 3.63x.
- *GraniKV* (arXiv:2608.15584); *ContiguousKV* (arXiv:2601.13631); *FastKV* (Findings of ACL 2026); *PrefillShare* (arXiv:2602.12029); *Restore-vs-Recompute* (HotStorage'26); *Tutti* (arXiv:2605.03375); *SparseX* (arXiv:2606.01751); *PolyKV* (arXiv:2604.24971); *MemDecay* (arXiv:2607.10582); *Provenance in Shared KV Caches* (arXiv:2609.38706); *KV-cache compression pitfalls* (ACL 2026). — scope boundaries.
- Yao et al. *BurstGPT.* arXiv:2401.17644 (2024). — public trace (live-fetched here); `HPMLL/BurstGPT`.
- ShareGPT prefix-cache eviction study (`superAttention/llm-prefix-cache-analysis`) — Belady-oracle methodology reference.
- flozi.net KVSET guide (Sept 2026) — independent one-pass verification pattern followed in EXP-04.

---

*All tables report executed outputs. To challenge any number, run the named experiment — the assertion is the citation.*
