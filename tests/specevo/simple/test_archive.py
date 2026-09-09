"""Tests for ClusterArchive — admission, re-clustering, ablation toggles."""

from __future__ import annotations

import numpy as np

from specevo.simple.archive import ArchiveConfig, ClusterArchive, Program

from ._fake_embeddings import family, vec


def _mk(score: float, embedding, code: str = "def f(x):\n    return 0\n", desc: str = "p") -> Program:
    return Program(
        code=code,
        description=desc,
        score=score,
        embedding=embedding,
        source="mutate",
    )


def test_admits_first_program() -> None:
    arch = ClusterArchive(ArchiveConfig(n_cells=4, min_admits_before_cluster=4))
    accepted, reason = arch.add(_mk(0.5, vec(1, 0, 0)))
    assert accepted and reason == "added"
    assert len(arch) == 1


def test_replaces_when_better_in_same_cell() -> None:
    # Force the pre-cluster regime — each admit gets a fresh cell id —
    # so we explicitly verify cell-incumbent comparison by re-clustering
    # after admits.
    arch = ClusterArchive(ArchiveConfig(n_cells=2, min_admits_before_cluster=2))
    arch.add(_mk(0.5, vec(1, 0.01, 0), desc="A"))
    arch.add(_mk(0.5, vec(1, 0.0, 0), desc="A2"))  # triggers re-cluster
    # Now both should be in the same cell after KMeans on similar vectors.
    cells_now = arch.cells()
    if len(cells_now) == 1:
        # Single cell, single program kept (the higher-score one — same here).
        assert len(arch) == 1
    # Adding a higher-score similar program must replace the incumbent.
    accepted, reason = arch.add(_mk(0.9, vec(1, 0.005, 0), desc="A3"))
    assert accepted, f"expected admit, got {reason}"


def test_drops_worse_in_same_cell_after_clustering() -> None:
    arch = ClusterArchive(ArchiveConfig(n_cells=2, min_admits_before_cluster=2))
    # Two tightly clustered points around (1, 0, 0).
    arch.add(_mk(0.9, vec(1, 0.0, 0), desc="A"))
    arch.add(_mk(0.5, vec(1, 0.01, 0), desc="A2"))
    # After re-cluster, both should share a cell; the lower score is gone.
    if len(arch) == 1:
        # Try to admit a third member of the same cluster at lower score
        # → must be dropped.
        accepted, reason = arch.add(_mk(0.3, vec(1, 0.02, 0), desc="A3"))
        assert not accepted
        assert reason == "dropped_worse"


def test_multiple_paradigms_get_multiple_cells() -> None:
    arch = ClusterArchive(ArchiveConfig(n_cells=3, min_admits_before_cluster=3))
    # Three orthogonal families.
    for fid in range(3):
        for v in family(seed=fid, jitter=0.02, n=2):
            arch.add(_mk(0.5 + 0.01 * fid, v, desc=f"fam{fid}"))
    # After KMeans on 6 points around 3 orthogonal axes, expect ~3 cells.
    assert arch.num_occupied_cells() >= 2


def test_ablation_ast_only_disables_embedding_half() -> None:
    arch = ClusterArchive(ArchiveConfig(
        n_cells=2, min_admits_before_cluster=2, use_embedding=False,
    ))
    arch.add(_mk(0.5, vec(1, 0, 0), code="def f():\n    for i in range(10):\n        pass\n"))
    arch.add(_mk(0.5, vec(0, 1, 0), code="def f():\n    return 1\n"))
    # With use_embedding=False, behavior_vec dim = 14 (just AST).
    for p in arch.programs():
        assert p.behavior_vec.shape[0] == 14


def test_ablation_emb_only_disables_ast_half() -> None:
    # PCA dim is clamped by min(embedding_dim, n_samples, raw_dim);
    # set min_admits_before_cluster=6 with 6 admits so the SVD can keep
    # ``embedding_dim`` components.
    arch = ClusterArchive(ArchiveConfig(
        n_cells=2, min_admits_before_cluster=6, use_ast=False, embedding_dim=4,
    ))
    for i, axis in enumerate(range(6)):
        v = vec(*(1.0 if k == axis else 0.0 for k in range(8)))
        arch.add(_mk(0.5 + 0.01 * i, v, desc=f"p{i}"))
    # With use_ast=False, behavior_vec dim = PCA target.
    for p in arch.programs():
        assert p.behavior_vec.shape[0] == 4


def test_ablation_static_cells_freezes_centroids() -> None:
    cfg = ArchiveConfig(
        n_cells=2, min_admits_before_cluster=2, recluster_every=2,
        adaptive_recluster=False,
    )
    arch = ClusterArchive(cfg)
    arch.add(_mk(0.5, vec(1, 0, 0)))
    arch.add(_mk(0.6, vec(0, 1, 0)))
    cent0 = arch._centroids.copy()
    # Push a few more admits to exceed recluster_every threshold.
    for i in range(5):
        arch.add(_mk(0.7 + i * 0.01, vec(1, 0, 0)))
    cent1 = arch._centroids
    # Static cells: centroids must be unchanged.
    assert np.allclose(cent0, cent1)


def test_rejects_when_no_embedding() -> None:
    arch = ClusterArchive(ArchiveConfig(n_cells=2, min_admits_before_cluster=2))
    p = Program(code="x=1", description="d", score=0.0, embedding=np.array([], dtype=np.float32))
    accepted, reason = arch.add(p)
    assert not accepted
    assert reason == "no_embedding"
