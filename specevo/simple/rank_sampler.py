"""Parameter-free rank-based parent sampler for SpecEvo.

Replaces the UCB selector used in prior versions. The design is taken
directly from the framework's ``AdaptiveRankSampler`` (Zipfian over score rank,
β driven by stagnation) because it has two properties the prior
threshold-heavy selector lacked:

* **Scale-invariant.** Probability depends only on the *rank* of a
  program inside the archive, not the raw score gaps. A small
  improvement at the top of the archive does not collapse the
  distribution to near-deterministic mode.

* **Single derived knob.** β(stagnation) interpolates between argmax
  (high β when search is healthy) and uniform (low β when stuck).
  There is no separate "boost", "grace", or "diversity penalty" — the
  rank already encodes everything.

Crossover parent selection uses a tiny extra rule: the second parent is
drawn from a *different cell* than the first whenever possible, so the
mutation model sees two paradigms to hybridise. This is the only cell-
aware sampling in the system; it does not introduce any new threshold
because cell membership is already discrete.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .archive import Program

logger = logging.getLogger(__name__)


@dataclass
class RankSamplerConfig:
    beta_max: float = 2.0
    """Zipfian exponent when the search is fresh (stagnation = 0).
    P(rank=r) ∝ (r+1)^(-β). β=2 ≈ top-3 dominate; β=0 ≈ uniform."""

    beta_min: float = 0.3
    """Floor for β under maximum stagnation. Pure uniform (β=0) was
    observed to spend too long on the long tail; 0.3 keeps a mild
    preference for high-score parents even when the search is fully
    stuck, while still letting the long tail get sampled often enough
    to break plateaus."""

    n_inspirations: int = 3
    """How many additional programs to surface as inspirations in the
    mutate / crossover prompt. Description-only (no code) so the model
    treats them as ideas, not copy targets."""


@dataclass
class RankSampler:
    config: RankSamplerConfig = field(default_factory=RankSamplerConfig)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def beta(self, stagnation: float) -> float:
        """β(stagnation): linear interpolation between beta_max (fresh)
        and beta_min (stuck), saturating at the bounds."""
        s = max(0.0, min(1.0, float(stagnation)))
        return self.config.beta_min + (1.0 - s) * (self.config.beta_max - self.config.beta_min)

    def select_parent(
        self,
        programs: Sequence[Program],
        *,
        stagnation: float,
        rng: random.Random | None = None,
    ) -> Program | None:
        """Draw one parent by Zipfian rank over score."""
        if not programs:
            return None
        return self._zipfian_draw(programs, stagnation=stagnation, rng=rng)

    def select_two_parents(
        self,
        programs: Sequence[Program],
        *,
        stagnation: float,
        rng: random.Random | None = None,
    ) -> tuple[Program, Program] | None:
        """Draw two parents; prefer different cells so crossover sees
        two paradigms. Falls back to same-cell when only one cell is
        occupied (early bootstrap)."""
        if len(programs) < 2:
            return None
        p1 = self._zipfian_draw(programs, stagnation=stagnation, rng=rng)
        assert p1 is not None
        rest = [p for p in programs if p is not p1]
        cross_cell = [p for p in rest if p.cell_id != p1.cell_id]
        pool_for_p2 = cross_cell if cross_cell else rest
        p2 = self._zipfian_draw(pool_for_p2, stagnation=stagnation, rng=rng)
        assert p2 is not None
        return p1, p2

    def select_inspirations(
        self,
        programs: Sequence[Program],
        *,
        exclude: Sequence[Program],
        stagnation: float,
        rng: random.Random | None = None,
        k: int | None = None,
    ) -> list[Program]:
        """Draw *k* inspirations (without replacement, Zipfian)."""
        k = self.config.n_inspirations if k is None else k
        if k <= 0:
            return []
        exclude_ids = {id(p) for p in exclude}
        pool = [p for p in programs if id(p) not in exclude_ids]
        if not pool:
            return []
        picked: list[Program] = []
        rng = rng or random
        for _ in range(min(k, len(pool))):
            choice = self._zipfian_draw(pool, stagnation=stagnation, rng=rng)
            if choice is None:
                break
            picked.append(choice)
            pool = [p for p in pool if p is not choice]
        return picked

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _zipfian_draw(
        self,
        programs: Sequence[Program],
        *,
        stagnation: float,
        rng: random.Random | None,
    ) -> Program | None:
        if not programs:
            return None
        beta = self.beta(stagnation)
        # Sort by score desc; deterministic tie-break by id to stabilise
        # reproductions across asyncio interleavings.
        ranked = sorted(programs, key=lambda p: (-p.score, id(p)))
        n = len(ranked)
        ranks = np.arange(1, n + 1, dtype=np.float64)
        weights = ranks ** (-beta)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            return ranked[0]
        probs = weights / total
        rng = rng or random
        # Inverse-CDF sampling (avoid numpy random state contention with
        # multi-worker asyncio code).
        u = rng.random()
        cum = 0.0
        for p, prob in zip(ranked, probs):
            cum += float(prob)
            if u <= cum:
                return p
        return ranked[-1]
