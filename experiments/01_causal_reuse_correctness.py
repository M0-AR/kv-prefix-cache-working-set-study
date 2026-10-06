"""EXP-01: causal K/V reuse == full recompute (numpy, seeded)."""
import json, sys
sys.path.insert(0, "src")
import numpy as np
from attention_kv import full_prefill, KVCache

rng = np.random.default_rng(0)
d, dk = 16, 16
Wq, Wk, Wv = rng.normal(size=(d, dk)), rng.normal(size=(d, dk)), rng.normal(size=(d, dk))
X = rng.normal(size=(5, d))

full_out, K, V = full_prefill(X, Wq, Wk, Wv)
# incremental: prefill first 3, decode 2
from attention_kv import prefill_into_cache
pre, cache = prefill_into_cache(X[:3], Wq, Wk, Wv)
o3 = pre
o4 = cache.append_token(X[3], Wq, Wk, Wv)
o5 = cache.append_token(X[4], Wq, Wk, Wv)
inc = np.stack([o3[2], o4, o5])  # last-row outputs for positions 3,4,5
ref = np.stack([full_out[2], full_out[3], full_out[4]])
max_err = float(np.max(np.abs(inc - ref)))
print(json.dumps({"experiment": "01_causal_reuse", "max_abs_err": max_err, "pass": bool(max_err < 1e-9)}))
assert max_err < 1e-9, max_err
with open("results/01.json", "w") as f:
    json.dump({"max_abs_err": max_err, "pass": True}, f, indent=2)
