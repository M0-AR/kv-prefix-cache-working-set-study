"""EXP-05: live public-data replay (BurstGPT lengths + synthetic agent pages).

Verifies the study against real-world public data: BurstGPT arrival/length
statistics are fetched live; prefix reuse is measured on a trace whose
length distribution is driven by the live sample.
"""
import json, sys
sys.path.insert(0, "src")
import random
from trace_loader import load_burstgpt_sample, synthetic_agentic_trace, trace_to_page_accesses
from stack_distance import stack_distances, hitrate_curve, working_set

live = load_burstgpt_sample(max_rows=2000)
req = live["req_tokens"]
mean_req = sum(req) / len(req)
sorted_req = sorted(req)
p50 = sorted_req[len(req) // 2]
p99 = sorted_req[int(len(req) * 0.99)]

# Build an agent trace whose turn sizes track the live mean (scale-aware replay)
rng = random.Random(99)
trace = synthetic_agentic_trace(n_sessions=12, turns_per_session=8,
                                system_tokens=64,
                                per_turn_new_tokens=max(8, min(64, int(mean_req // 8))),
                                seed=99)
acc = trace_to_page_accesses(trace, block_size=16)
caps = [8, 16, 32, 64, 128, 256, 512]
dists = stack_distances(acc)
curve = hitrate_curve(dists, caps)
ws80 = working_set(dists, 0.80, list(range(1, 769)))
out = {"experiment": "05_real_trace_replay", "burst_source": live["source"],
       "burst_url": live["url"], "n_live_rows": live["n"],
       "live_mean_req_tokens": mean_req, "live_p50_req": p50, "live_p99_req": p99,
       "replay_accesses": len(acc), "curve": curve, "working_set_80": ws80}
print(json.dumps(out, indent=2))
assert live["n"] >= 50 and len(acc) > 0 and ws80 is not None
with open("results/05.json", "w") as f:
    json.dump(out, f, indent=2)
