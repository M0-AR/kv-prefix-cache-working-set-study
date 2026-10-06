"""EXP-06: hidden-pattern hunt (session vs structural reuse, FP-rate illusion).

Findings reported only if measured here:
 H1 session reuse >> structural reuse (UniCache-style split)
 H2 block-size coarsening taxes hit rate (partial-tail waste)
 H3 interleaving (concurrency) deflates hits at fixed capacity
 H4 cold-start: first-epoch requests can never hit (compulsory misses)
"""
import json, sys
sys.path.insert(0, "src")
from collections import Counter
from trace_loader import synthetic_agentic_trace, trace_to_page_accesses
from prefix_cache import simulate

# H1: isolate session reuse (per-session sequential) vs structural-only shuffle
trace = synthetic_agentic_trace(n_sessions=12, turns_per_session=8, seed=21)
seq = [t for t in trace]  # interleaved by construction
# sequential-by-session order maximizes session locality
flat_by_session = []
for s in range(12):
    flat_by_session.extend(trace[s::12] if False else [])  # placeholder guard
# rebuild properly: trace is interleaved round-robin; de-interleave:
turns = 8
per_session = [[None]*turns for _ in range(12)]
idx = 0
for t in range(turns):
    for s in range(12):
        per_session[s][t] = trace[idx]; idx += 1
sequential = [r for s in range(12) for r in per_session[s]]

h_seq = simulate(sequential, capacity=64, block_size=16)["hit_rate_tokens"]
h_int = simulate(seq, capacity=64, block_size=16)["hit_rate_tokens"]

# H2: block-size tax at fixed TOKEN budget (fair comparison).
# Naive block-count comparison is misleading because 64 blocks x 64 tok = 16x
# the memory of 64 blocks x 4 tok. Fix token budget at 1024 tokens:
h_bs4 = simulate(seq, capacity=256, block_size=4)["hit_rate_tokens"]    # 256*4=1024 tok
h_bs16 = simulate(seq, capacity=64, block_size=16)["hit_rate_tokens"]  # 64*16=1024 tok
h_bs64 = simulate(seq, capacity=16, block_size=64)["hit_rate_tokens"]  # 16*64=1024 tok
# H2b (documented pitfall): same block COUNT inflates coarse-block hits
h_bs4_samecount = simulate(seq, capacity=64, block_size=4)["hit_rate_tokens"]
h_bs64_samecount = simulate(seq, capacity=64, block_size=64)["hit_rate_tokens"]

# H3/H4: compulsory-miss share = unique-page fraction on first epoch
acc = trace_to_page_accesses(seq, block_size=16)
first_epoch = acc[: len(acc)//4]
compulsory = len(set(first_epoch)) / len(first_epoch)

out = {"experiment": "06_hidden_patterns",
       "H1_session_locality": {"sequential_hit": h_seq, "interleaved_hit": h_int,
                               "delta": h_seq - h_int},
       "H2_blocksize_tax_token_budget_1024": {"bs4": h_bs4, "bs16": h_bs16, "bs64": h_bs64},
       "H2b_pitfall_same_block_count": {"bs4_64blocks": h_bs4_samecount,
                                        "bs64_64blocks": h_bs64_samecount,
                                        "note": "same-block-count favors coarse blocks by 16x memory; always normalize by tokens"},
       "H4_compulsory_miss_share_first_epoch": compulsory}
print(json.dumps(out, indent=2))
assert h_seq >= h_int, "locality must help"
assert compulsory > 0
with open("results/06.json", "w") as f:
    json.dump(out, f, indent=2)
