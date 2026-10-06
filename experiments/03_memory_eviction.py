"""EXP-03: memory cost per token + eviction forces recompute."""
import json, sys
sys.path.insert(0, "src")
from memory_model import MODELS, bytes_per_token, cache_gib, throughput_gain
from prefix_cache import PrefixCacheLRU

rows = {}
for name, cfg in MODELS.items():
    bpt = bytes_per_token(cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"], 2)
    rows[name] = {"bytes_per_token": bpt, "kib_per_token": bpt / 1024,
                  "gib_per_32k": cache_gib(32768, **cfg)}
# eviction demo: tiny cache thrashes, big cache reuses
toks_a = list(range(100, 132))  # 32 toks = 8 blocks @ bs=4
toks_b = list(range(200, 232))
small = PrefixCacheLRU(4, 4)
big = PrefixCacheLRU(32, 4)
for _ in range(3):
    small.request(toks_a); small.request(toks_b)
    big.request(toks_a); big.request(toks_b)
out = {"experiment": "03_memory_eviction", "models": rows,
       "small_cache_hit_rate": small.hit_rate_blocks,
       "big_cache_hit_rate": big.hit_rate_blocks,
       "throughput_gain_at_75pct": throughput_gain(0.75),
       "throughput_gain_at_90pct": throughput_gain(0.90)}
print(json.dumps(out, indent=2))
assert rows["llama-70B-GQA"]["bytes_per_token"] == 327680
assert small.hit_rate_blocks < big.hit_rate_blocks
with open("results/03.json", "w") as f:
    json.dump(out, f, indent=2)
