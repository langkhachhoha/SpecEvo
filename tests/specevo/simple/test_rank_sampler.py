"""Tests for the rank-based parent / inspiration sampler."""

from __future__ import annotations

import random

import numpy as np

from specevo.simple.archive import Program
from specevo.simple.rank_sampler import RankSampler, RankSamplerConfig


def _mk(score: float, cell_id: int = 0) -> Program:
    return Program(
        code=f"# s={score}",
        description=f"prog {score}",
        score=score,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        source="mutate",
        cell_id=cell_id,
    )


def test_beta_decreases_with_stagnation() -> None:
    s = RankSampler(RankSamplerConfig(beta_max=2.0, beta_min=0.3))
    assert s.beta(0.0) == 2.0
    assert s.beta(1.0) == 0.3
    # Monotone.
    assert s.beta(0.5) > s.beta(0.8)


def test_select_parent_prefers_top_at_low_stagnation() -> None:
    programs = [_mk(i) for i in range(10)]
    sampler = RankSampler(RankSamplerConfig(beta_max=3.0, beta_min=0.3))
    rng = random.Random(0)
    counts: dict[float, int] = {}
    N = 2000
    for _ in range(N):
        p = sampler.select_parent(programs, stagnation=0.0, rng=rng)
        counts[p.score] = counts.get(p.score, 0) + 1
    # Top score must be selected far more often than the bottom.
    assert counts.get(9.0, 0) > counts.get(0.0, 0) * 5


def test_select_parent_flattens_at_high_stagnation() -> None:
    programs = [_mk(i) for i in range(10)]
    sampler = RankSampler(RankSamplerConfig(beta_max=3.0, beta_min=0.3))
    rng = random.Random(0)
    counts: dict[float, int] = {}
    N = 4000
    for _ in range(N):
        p = sampler.select_parent(programs, stagnation=1.0, rng=rng)
        counts[p.score] = counts.get(p.score, 0) + 1
    # With β=0.3 the tail must get reasonable mass — top should still
    # dominate, but every program must be touched.
    assert all(counts.get(float(i), 0) > 0 for i in range(10))


def test_two_parents_prefer_different_cells() -> None:
    a = _mk(1.0, cell_id=0)
    b = _mk(0.5, cell_id=1)
    c = _mk(0.4, cell_id=0)
    sampler = RankSampler()
    rng = random.Random(0)
    pair = sampler.select_two_parents([a, b, c], stagnation=0.0, rng=rng)
    assert pair is not None
    p1, p2 = pair
    assert p1.cell_id != p2.cell_id


def test_two_parents_falls_back_to_same_cell_when_alone() -> None:
    a = _mk(1.0, cell_id=0)
    b = _mk(0.5, cell_id=0)
    sampler = RankSampler()
    rng = random.Random(0)
    pair = sampler.select_two_parents([a, b], stagnation=0.0, rng=rng)
    assert pair is not None  # only one cell, but we must still return a pair


def test_inspirations_unique() -> None:
    programs = [_mk(i, cell_id=i % 3) for i in range(10)]
    sampler = RankSampler(RankSamplerConfig(n_inspirations=4))
    rng = random.Random(0)
    insps = sampler.select_inspirations(programs, exclude=[programs[0]], stagnation=0.5, rng=rng)
    assert len(insps) == 4
    assert len({id(p) for p in insps}) == 4
    assert programs[0] not in insps
