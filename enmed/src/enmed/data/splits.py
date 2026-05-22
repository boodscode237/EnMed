"""Few-shot demonstration selection strategies.

Three strategies are compared in the paper:
  1. random            — uniform sampling from the train pool
  2. stratified        — balanced across medical specialties
  3. similarity        — semantically nearest examples (embedding cosine)

The strategy is chosen in the eval config; `select_shots` dispatches on it.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence


def select_shots(
    query: dict,
    pool: Sequence[dict],
    n: int,
    strategy: str = "random",
    seed: int = 12181531,
    embedder=None,
) -> list[dict]:
    """Return `n` demonstration examples for a query using the given strategy."""
    if n == 0 or not pool:
        return []
    rng = random.Random(seed + hash(query.get("id", "")) % 10_000)

    if strategy == "random":
        return rng.sample(list(pool), min(n, len(pool)))

    if strategy == "stratified":
        return _stratified(query, pool, n, rng)

    if strategy == "similarity":
        if embedder is None:
            raise ValueError("similarity strategy requires an `embedder`")
        return _similarity(query, pool, n, embedder)

    raise ValueError(f"Unknown few-shot strategy: {strategy}")


def _stratified(query, pool, n, rng) -> list[dict]:
    """Sample examples balanced across specialties, preferring the query's."""
    by_spec: dict[str, list] = defaultdict(list)
    for ex in pool:
        by_spec[ex.get("specialty", "unknown")].append(ex)

    q_spec = query.get("specialty", "unknown")
    order = [q_spec] + [s for s in by_spec if s != q_spec]
    out: list[dict] = []
    i = 0
    while len(out) < n and any(by_spec.values()):
        spec = order[i % len(order)]
        bucket = by_spec.get(spec, [])
        if bucket:
            out.append(bucket.pop(rng.randrange(len(bucket))))
        i += 1
        if i > 10 * n:  # safety
            break
    return out[:n]


def _similarity(query, pool, n, embedder) -> list[dict]:
    """Pick the `n` examples whose question embeddings are closest to the query."""
    import numpy as np

    q_vec = embedder([query["question"]])[0]
    pool_vecs = embedder([ex["question"] for ex in pool])
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    pool_vecs = pool_vecs / (np.linalg.norm(pool_vecs, axis=1, keepdims=True) + 1e-9)
    sims = pool_vecs @ q_vec
    top = np.argsort(-sims)[:n]
    return [pool[i] for i in top]
