"""Tiny helpers to build deterministic, L2-normalized fake embeddings.

Used so pool/monitor/selector tests don't need a real embedding model.
"""

from __future__ import annotations

import numpy as np


def vec(*components: float, dim: int = 8) -> np.ndarray:
    """Build a dim-d unit vector from the given leading components."""
    arr = np.zeros(dim, dtype=np.float32)
    for i, c in enumerate(components):
        if i >= dim:
            break
        arr[i] = c
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr


def family(seed: int, jitter: float = 0.05, n: int = 1, dim: int = 8) -> list[np.ndarray]:
    """Produce *n* vectors clustered around one of the canonical axes.

    seed: 0..dim-1 picks which axis to cluster on. jitter controls intra-
    cluster spread (smaller = tighter family).
    """
    rng = np.random.default_rng(seed * 1009 + 7)
    base = np.zeros(dim, dtype=np.float32)
    base[seed % dim] = 1.0
    out = []
    for _ in range(n):
        noise = rng.normal(0, jitter, size=dim).astype(np.float32)
        v = base + noise
        n_ = float(np.linalg.norm(v))
        out.append(v / n_)
    return out
