"""Mattson (1970) LRU stack-distance working-set analysis for KV pages.

One pass over the access stream yields the hit-rate curve for *all*
capacities, and the working set = min capacity achieving target hit rate.
Verified against naive per-capacity LRU simulation in experiments.
"""
from __future__ import annotations


def stack_distances(accesses: list[hashable]) -> list[int | float]:  # type: ignore[valid-type]
    """LRU stack distance per access. inf (float('inf')) = first-ever use (cold miss)."""
    # O(N * U) simple version: exact, fine for traces up to ~100k accesses.
    last_pos: dict = {}
    # maintain LRU stack as list of keys, MRU at end
    stack: list = []
    pos_in_stack: dict = {}
    dists: list[int | float] = []
    for key in accesses:
        if key not in pos_in_stack:
            dists.append(float("inf"))
            stack.append(key)
            pos_in_stack[key] = len(stack) - 1
        else:
            idx = pos_in_stack[key]
            d = (len(stack) - 1) - idx  # # distinct keys accessed since last use
            # stack distance convention: number of *distinct* intervening keys + 1?
            # We use standard "reuse distance" = # distinct keys since last ref.
            # Hit under capacity C iff d < C (room for d intervening + the line itself).
            # To keep `hit iff d <= C` with 1-indexed depth, store d+1.
            dists.append(d + 1)
            # move to MRU
            stack.pop(idx)
            stack.append(key)
            # rebuild index for shifted suffix (O(U) worst case; ok for study scale)
            for i in range(idx, len(stack)):
                pos_in_stack[stack[i]] = i
        last_pos[key] = True
    return dists


def hitrate_curve(dists: list[int | float], capacities: list[int]) -> dict[int, float]:
    """Hit(C) = P(stack_distance <= C). Cold misses (inf) never hit."""
    n = len(dists)
    out: dict[int, float] = {}
    for c in capacities:
        h = sum(1 for d in dists if d != float("inf") and d <= c)
        out[c] = h / n if n else 0.0
    return out


def working_set(dists: list[int | float], target: float, capacities: list[int]) -> int | None:
    """Minimum capacity with hit rate >= target. None if unreachable in range."""
    curve = hitrate_curve(dists, sorted(capacities))
    for c in sorted(capacities):
        if curve[c] >= target:
            return c
    return None


def naive_hitrate(accesses: list, capacity: int) -> float:
    """Independent LRU simulation at one capacity (ground truth for verification)."""
    from collections import OrderedDict

    cache: OrderedDict = OrderedDict()
    hits = 0
    for a in accesses:
        if a in cache:
            hits += 1
            cache.move_to_end(a)
        else:
            cache[a] = None
            while len(cache) > capacity:
                cache.popitem(last=False)
    return hits / len(accesses) if accesses else 0.0
