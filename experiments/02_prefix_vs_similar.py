"""EXP-02: exact token prefix hits; merely-similar text misses."""
import json, sys
sys.path.insert(0, "src")
from prefix_cache import PrefixCacheLRU, tokenize_words

c = PrefixCacheLRU(capacity_blocks=64, block_size=4)
agent_turn1 = tokenize_words("system you are a coding assistant with tool history file main py")
agent_turn2 = tokenize_words("system you are a coding assistant with tool history file main py run tests now")
agent_turn2_paraphrase = tokenize_words("system you are a coding helper with tool log file main py run tests now")

r1 = c.request(agent_turn1)
r2 = c.request(agent_turn2)  # exact prefix -> hits
c2 = PrefixCacheLRU(capacity_blocks=64, block_size=4)
c2.request(agent_turn1)
r3 = c2.request(agent_turn2_paraphrase)  # similar wording, different tokens -> ~no hit

out = {
    "experiment": "02_prefix_vs_similar",
    "exact_prefix_hit_blocks": r2["hit_blocks"],
    "exact_prefix_total": r2["total_blocks"],
    "paraphrase_hit_blocks": r3["hit_blocks"],
    "paraphrase_total": r3["total_blocks"],
}
print(json.dumps(out, indent=2))
assert r2["hit_blocks"] > 0, "exact prefix must hit"
assert r3["hit_blocks"] <= r2["hit_blocks"], "paraphrase must not beat exact prefix"
with open("results/02.json", "w") as f:
    json.dump(out, f, indent=2)
