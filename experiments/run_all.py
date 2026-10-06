"""Run all experiments in order; fail fast on any assertion."""
import subprocess, sys

ORDER = ["01_causal_reuse_correctness", "02_prefix_vs_similar", "03_memory_eviction",
         "04_hitrate_capacity_curve", "05_real_trace_replay", "06_hidden_patterns"]
fails = 0
for name in ORDER:
    print(f"=== {name} ===", flush=True)
    r = subprocess.run([sys.executable, f"experiments/{name}.py"])
    if r.returncode != 0:
        print(f"FAILED: {name}")
        fails += 1
        break
if fails:
    sys.exit(1)
print("ALL EXPERIMENTS PASSED")
