"""EXP-04: one-pass Mattson curve == naive per-capacity sim; working set + knee."""
import json, sys
sys.path.insert(0, "src")
from trace_loader import synthetic_agentic_trace, trace_to_page_accesses
from stack_distance import stack_distances, hitrate_curve, working_set, naive_hitrate

trace = synthetic_agentic_trace(n_sessions=10, turns_per_session=8, seed=7)
acc = trace_to_page_accesses(trace, block_size=16)
caps = [4, 8, 16, 32, 64, 128, 256, 512]
dists = stack_distances(acc)
curve = hitrate_curve(dists, caps)
naive = {c: naive_hitrate(acc, c) for c in caps}
maxdiff = max(abs(curve[c] - naive[c]) for c in caps)
# working sets at two targets (None = unreachable: max hit rate caps the curve,
# exactly the "no convergence at extreme coverage" behavior reported in KVSET)
ws70 = working_set(dists, 0.70, list(range(1, 1025)))
ws80 = working_set(dists, 0.80, list(range(1, 1025)))
ws90 = working_set(dists, 0.90, list(range(1, 1025)))
max_achievable = max(curve.values())
# knee: capacity where marginal gain over next doubling < 0.02
ordered = sorted(curve)
knee = None
for i in range(len(ordered) - 1):
    if curve[ordered[i + 1]] - curve[ordered[i]] < 0.02:
        knee = ordered[i]
        break
out = {"experiment": "04_hitrate_capacity", "n_accesses": len(acc),
       "curve": curve, "naive": naive, "max_abs_diff": maxdiff,
       "working_set_70": ws70, "working_set_80": ws80, "working_set_90": ws90,
       "max_achievable_hit_rate": max_achievable, "knee": knee}
print(json.dumps(out, indent=2))
assert maxdiff < 1e-12, maxdiff
assert ws70 is not None and ws70 <= (ws80 if ws80 is not None else 10**9)
# ws90 may legitimately be None (unreachable); record, don't force
with open("results/04.json", "w") as f:
    json.dump(out, f, indent=2)
# csv for the paper figure
with open("results/04_curve.csv", "w") as f:
    f.write("capacity,mattson,naive\n")
    for c in caps:
        f.write(f"{c},{curve[c]:.6f},{naive[c]:.6f}\n")
