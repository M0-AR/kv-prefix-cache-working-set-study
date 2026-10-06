"""Block-level LRU prefix cache (vLLM-style hash = prefix + block).

Each block key = tuple(tokens up to end of block). A request hits the
longest prefix of *complete* blocks already cached. Partial-block tails
never hit (matches vLLM/SGLang page_size semantics).
"""
from __future__ import annotations

from collections import OrderedDict


def tokenize_words(text: str) -> list[int]:
    """Deterministic toy tokenizer: word -> int via stable hash."""
    toks = []
    for w in text.split():
        h = 0
        for ch in w:
            h = (h * 31 + ord(ch)) & 0x7FFFFFFF
        toks.append(h % 50000 + 1)
    return toks


class PrefixCacheLRU:
    def __init__(self, capacity_blocks: int, block_size: int = 4):
        assert capacity_blocks > 0 and block_size > 0
        self.capacity = capacity_blocks
        self.block_size = block_size
        self.store: OrderedDict[tuple, None] = OrderedDict()  # key -> None (LRU: MRU at end)
        self.hits_blocks = 0
        self.total_blocks = 0
        self.hits_tokens = 0
        self.total_tokens = 0

    def _block_keys(self, tokens: list[int]) -> list[tuple]:
        """Full-block prefix keys only (drop partial tail)."""
        n_full = len(tokens) // self.block_size
        keys = []
        for b in range(n_full):
            end = (b + 1) * self.block_size
            keys.append(tuple(tokens[:end]))
        return keys

    def request(self, tokens: list[int]) -> dict:
        keys = self._block_keys(tokens)
        hit_prefix_blocks = 0
        for k in keys:
            if k in self.store:
                hit_prefix_blocks += 1
                self.store.move_to_end(k)
            else:
                break  # prefix property: stop at first miss
        # insert missed suffix blocks in order (each may evict LRU)
        for k in keys[hit_prefix_blocks:]:
            if k in self.store:
                self.store.move_to_end(k)
                continue
            self.store[k] = None
            while len(self.store) > self.capacity:
                self.store.popitem(last=False)
        n_full_toks = len(keys) * self.block_size
        self.hits_blocks += hit_prefix_blocks
        self.total_blocks += len(keys)
        self.hits_tokens += hit_prefix_blocks * self.block_size
        self.total_tokens += n_full_toks
        return {
            "hit_blocks": hit_prefix_blocks,
            "total_blocks": len(keys),
            "hit_tokens": hit_prefix_blocks * self.block_size,
            "total_tokens": n_full_toks,
        }

    @property
    def hit_rate_blocks(self) -> float:
        return self.hits_blocks / self.total_blocks if self.total_blocks else 0.0

    @property
    def hit_rate_tokens(self) -> float:
        return self.hits_tokens / self.total_tokens if self.total_tokens else 0.0


def simulate(tokens_list: list[list[int]], capacity: int, block_size: int = 4) -> dict:
    c = PrefixCacheLRU(capacity, block_size)
    for t in tokens_list:
        c.request(t)
    return {
        "capacity": capacity,
        "hit_rate_blocks": c.hit_rate_blocks,
        "hit_rate_tokens": c.hit_rate_tokens,
        "hits_blocks": c.hits_blocks,
        "total_blocks": c.total_blocks,
    }
