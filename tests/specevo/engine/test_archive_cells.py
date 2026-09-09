"""Cell-count invariant tests for ClusterArchive.

These tests pin the fix that the previous SpecEvo runs collapsed to
``num_occupied_cells = 16`` for the entire evolution despite
``n_cells = 32`` being requested. The root cause was twofold:

1. ``KMeans`` was being asked for ``k = min(n_cells, n)`` clusters,
   so once ``n`` dropped (coalescing after a recluster) ``k`` stayed
   low forever.
2. The coalesce step kept only the top-score program per cell, so the
   population shrank to ``num_occupied_cells`` and the next recluster
   could not see enough points to grow.

The new contract: ``n_cells`` is held FIXED across the run; KMeans is
not fit until at least ``n_cells`` programs have been admitted; and
once fit, ``len(centroids) == n_cells`` always.
"""

from __future__ import annotations

import numpy as np

from specevo.simple.archive import ArchiveConfig, ClusterArchive, Program


def _make_program(score: float, embedding: np.ndarray, seed_code: int) -> Program:
    code = f"def solve(x):\n    return x + {seed_code}\n"
    return Program(
        code=code,
        description=f"score={score:.2f} variant{seed_code}",
        score=score,
        embedding=embedding,
    )


def _admit_many(archive: ClusterArchive, n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n):
        emb = rng.normal(size=64).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        archive.add(_make_program(score=float(rng.random()), embedding=emb, seed_code=i))


def test_kmeans_runs_with_full_n_cells_after_threshold() -> None:
    """Once ``n_cells`` programs have been admitted, KMeans is fit with
    exactly ``k = n_cells`` clusters — never min(n_cells, n)."""
    cfg = ArchiveConfig(n_cells=50, min_admits_before_cluster=16, recluster_every=30)
    archive = ClusterArchive(cfg)

    _admit_many(archive, n=20)
    # Below n_cells=50: pre-cluster mode, no KMeans fit yet.
    assert archive._centroids is None

    _admit_many(archive, n=40, seed=1)
    # Crossed 50: KMeans fits with k=50.
    assert archive._centroids is not None
    assert archive._centroids.shape[0] == 50


def test_recluster_keeps_centroids_at_n_cells_after_coalesce() -> None:
    """Even after coalescing shrinks the population, the next recluster
    must fit ``k = n_cells`` centroids, not ``min(n_cells, n)``."""
    cfg = ArchiveConfig(n_cells=50, min_admits_before_cluster=16, recluster_every=30)
    archive = ClusterArchive(cfg)

    # Push the archive well past n_cells with diverse embeddings.
    _admit_many(archive, n=120, seed=0)

    # After many admits, programs in the archive may have coalesced
    # (best-per-cell), but the centroid grid is still 50.
    assert archive._centroids is not None
    assert archive._centroids.shape[0] == 50
    assert archive.num_occupied_cells() <= 50

    # Force another recluster cycle and verify centroids stay at 50.
    _admit_many(archive, n=60, seed=42)
    assert archive._centroids.shape[0] == 50


def test_num_occupied_cells_can_grow_after_initial_fit() -> None:
    """A central regression: with the old code, ``num_occupied_cells``
    plateaued at ~16 (the floor) for the whole run. With the new code,
    occupied-cell count can grow toward ``n_cells`` as more programs
    arrive."""
    cfg = ArchiveConfig(n_cells=50, min_admits_before_cluster=16, recluster_every=30)
    archive = ClusterArchive(cfg)

    # Cross the floor with very-diverse embeddings so KMeans separates
    # them into distinct cells.
    _admit_many(archive, n=50, seed=0)
    occupied_after_first_fit = archive.num_occupied_cells()

    # Add more diverse programs. Some land in fresh cells.
    _admit_many(archive, n=200, seed=99)
    occupied_after_more = archive.num_occupied_cells()

    # We don't insist on a strict ratio (random embeddings can collide),
    # but the post-growth count should not be *worse* than the initial
    # fit, and should be > 16 (the old plateau).
    assert occupied_after_more >= occupied_after_first_fit
    assert occupied_after_more > 16, (
        f"num_occupied_cells={occupied_after_more} still stuck at the old plateau"
    )


def test_cell_ids_are_in_range_0_n_cells() -> None:
    """All admitted programs must carry ``0 <= cell_id < n_cells``
    after the first KMeans fit. This is what allows downstream code
    (e.g. RankSampler.select_two_parents) to reason about cell
    diversity without bounds-checking."""
    cfg = ArchiveConfig(n_cells=50, min_admits_before_cluster=16, recluster_every=30)
    archive = ClusterArchive(cfg)
    _admit_many(archive, n=120, seed=7)
    for p in archive.programs():
        assert 0 <= p.cell_id < 50, p.cell_id
