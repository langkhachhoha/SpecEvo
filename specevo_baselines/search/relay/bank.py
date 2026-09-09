"""The relay objective, the online relay bank, and offline seed curation.

Implements Eq. (2)-(7) and Eq. (11)-(13) of RelayEvolve:

* ``Q_r(S)``  — mean of the top-``r`` normalised qualities in the bank (Eq. 2).
* ``D^q_C(S)`` — quality-weighted facility-location coverage of the candidate
  pool by the bank (Eq. 4).
* ``F_C(S) = λ Q_r(S) + (1-λ) D^q_C(S)`` (Eq. 5) — monotone submodular for a
  fixed reference pool, which is what buys the ``1 - 1/e`` guarantee on the
  greedy curation at handoff.
* ``g_t = F_{C_{t+1}}(S_{t+1}) - F_{C_{t+1}}(S_t)`` — the Relay Gain of a block
  (Eq. 7): *both* banks are scored against the **same, updated** pool, so the
  gain measures improvement of the handoff population and not drift in the
  reference pool.

Normalised quality ``q(x) ∈ [0, 1]`` is a min–max normalisation of the raw
fitness over the current pool, so the objective is scale-free across
benchmarks whose fitness ranges from ~2.5 (circle packing) to ~4000 (TXN
scheduling).
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from specevo_baselines.search.relay.embedding import CandidateEmbedder

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


def _code_key(solution: str) -> str:
    """Whitespace-insensitive fingerprint used to deduplicate the pool."""
    return hashlib.sha1(_WS_RE.sub(" ", solution or "").strip().encode()).hexdigest()


@dataclass
class Candidate:
    """One evaluated program that may be handed off to the strong model."""

    id: str
    solution: str
    score: float
    text: str = ""
    trajectory: int = -1
    iteration: int = 0
    code_vec: Optional[np.ndarray] = None
    text_vec: Optional[np.ndarray] = None
    q: float = 0.0
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Objective terms
# ---------------------------------------------------------------------------


def top_r_quality(qualities: Sequence[float], r: int) -> float:
    """``Q_r`` (Eq. 2): mean of the r largest qualities, zero-padded."""
    if r <= 0:
        return 0.0
    top = sorted(qualities, reverse=True)[:r]
    return sum(top) / r


def relay_objective(
    q_values: Sequence[float],
    coverage: float,
    lam: float,
    r: int,
) -> float:
    """``F_C(S)`` (Eq. 5) from the bank's qualities and its coverage term."""
    return lam * top_r_quality(q_values, r) + (1.0 - lam) * coverage


# ---------------------------------------------------------------------------
# Relay bank
# ---------------------------------------------------------------------------


class RelayBank:
    """Deduplicated candidate pool + the online quality-diverse bank on top."""

    def __init__(
        self,
        embedder: CandidateEmbedder,
        k: int = 8,
        r: int = 3,
        lam: float = 0.5,
        eta: float = 0.7,
        epsilon_f: float = 1e-3,
        max_pool: int = 600,
    ):
        self.embedder = embedder
        self.k = max(1, int(k))
        self.r = max(1, int(r))
        self.lam = float(lam)
        self.eta = float(eta)
        self.epsilon_f = float(epsilon_f)
        self.max_pool = int(max_pool)

        self.pool: List[Candidate] = []
        self._by_code: Dict[str, str] = {}
        self._index: Dict[str, int] = {}
        self._code_mat: Optional[np.ndarray] = None
        self._text_mat: Optional[np.ndarray] = None
        self._sim_mat: Optional[np.ndarray] = None
        self._q: Optional[np.ndarray] = None
        self.bank: List[str] = []

    # ------------------------------------------------------------------
    # Pool maintenance
    # ------------------------------------------------------------------

    def add_to_pool(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """``Dedup(C_t ∪ X_t)`` (Eq. 6). Returns the candidates actually added."""
        fresh: List[Candidate] = []
        for cand in candidates:
            key = _code_key(cand.solution)
            if key in self._by_code:
                continue
            self._by_code[key] = cand.id
            fresh.append(cand)

        if not fresh:
            return []

        texts_code = [c.solution for c in fresh]
        texts_meta = [c.text or "" for c in fresh]
        code_vecs = self.embedder.embed_batch(texts_code)
        text_vecs = self.embedder.embed_batch(texts_meta)
        for cand, cv, tv in zip(fresh, code_vecs, text_vecs):
            cand.code_vec = cv
            cand.text_vec = tv
            self._index[cand.id] = len(self.pool)
            self.pool.append(cand)

        self._trim_pool()
        self._rebuild_matrices()
        return fresh

    def _trim_pool(self) -> None:
        """Bound the pool so the O(|C|·k) objective stays cheap late in a run.

        Bank members are never dropped, and the survivors are the highest
        scoring candidates — exactly the mass ``D^q`` weights anyway.
        """
        if len(self.pool) <= self.max_pool:
            return
        keep_ids = set(self.bank)
        ranked = sorted(self.pool, key=lambda c: c.score, reverse=True)
        kept: List[Candidate] = []
        for cand in ranked:
            if len(kept) < self.max_pool or cand.id in keep_ids:
                kept.append(cand)
        kept_ids = {c.id for c in kept}
        self.pool = [c for c in self.pool if c.id in kept_ids]
        self._index = {c.id: i for i, c in enumerate(self.pool)}
        self._by_code = {_code_key(c.solution): c.id for c in self.pool}

    def _rebuild_matrices(self) -> None:
        if not self.pool:
            self._code_mat = self._text_mat = self._q = self._sim_mat = None
            return
        self._index = {c.id: i for i, c in enumerate(self.pool)}
        self._code_mat = np.stack([c.code_vec for c in self.pool])
        self._text_mat = np.stack([c.text_vec for c in self.pool])
        # Full pairwise sim(v, x) (Eq. 3) in one matmul. The objective is
        # evaluated thousands of times per curation pass, so caching this turns
        # each evaluation into a max-reduce over |C| x |S| floats.
        self._sim_mat = np.clip(
            self.eta * (self._code_mat @ self._code_mat.T)
            + (1.0 - self.eta) * (self._text_mat @ self._text_mat.T),
            0.0,
            1.0,
        )
        self._renormalize_quality()

    def _renormalize_quality(self) -> None:
        """Min–max normalise raw fitness over the pool into ``q ∈ [0, 1]``."""
        scores = np.array([c.score for c in self.pool], dtype=np.float64)
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            for cand in self.pool:
                cand.q = 0.0
            self._q = np.zeros(len(self.pool))
            return
        lo, hi = float(finite.min()), float(finite.max())
        span = hi - lo
        for cand in self.pool:
            raw = cand.score if math.isfinite(cand.score) else lo
            cand.q = 1.0 if span <= 0 else (raw - lo) / span
        self._q = np.array([c.q for c in self.pool], dtype=np.float64)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def objective(self, bank_ids: Sequence[str]) -> float:
        """``F_C(S)`` for the current reference pool."""
        if not bank_ids or self._q is None or self._sim_mat is None:
            return 0.0
        slots = [self._index[i] for i in bank_ids if i in self._index]
        if not slots:
            return 0.0
        q_values = [self.pool[i].q for i in slots]

        total_q = float(self._q.sum())
        if total_q <= 0:
            coverage = 0.0
        else:
            covered = self._sim_mat[:, slots].max(axis=1)
            coverage = float((self._q * covered).sum() / total_q)

        return relay_objective(q_values, coverage, self.lam, self.r)

    # ------------------------------------------------------------------
    # Online update + Relay Gain
    # ------------------------------------------------------------------

    def update_block(self, block_candidates: Sequence[Candidate]) -> Tuple[float, float]:
        """Fold one block's candidates in and return ``(g_t, ρ_t)``.

        ``g_t`` is the absolute Relay Gain (Eq. 7); ``ρ_t`` is its bounded
        relative form (Eq. 8), which is what the scheduler and the handoff
        rule consume.
        """
        self.add_to_pool(block_candidates)
        if self._q is None:
            return 0.0, 0.0

        base_bank = [i for i in self.bank if i in self._index]
        self.bank = base_bank
        f_before = self.objective(base_bank)

        # Stream the block's candidates in generation order.
        for cand in sorted(block_candidates, key=lambda c: (c.iteration, c.id)):
            if cand.id not in self._index or cand.id in self.bank:
                continue
            if len(self.bank) < self.k:
                self.bank.append(cand.id)
                continue
            current = self.objective(self.bank)
            best_value, best_slot = current, None
            for slot in range(len(self.bank)):
                trial = list(self.bank)
                trial[slot] = cand.id
                value = self.objective(trial)
                if value > best_value:
                    best_value, best_slot = value, slot
            if best_slot is not None:
                self.bank[best_slot] = cand.id

        f_after = self.objective(self.bank)
        gain = max(0.0, f_after - f_before)
        denom = max(f_before, self.epsilon_f)
        rel = min(1.0, max(0.0, gain / denom))
        return gain, rel


# ---------------------------------------------------------------------------
# Offline curation at handoff (Eq. 11-13)
# ---------------------------------------------------------------------------


def _greedy_select(bank: RelayBank, k: int) -> List[str]:
    """``GreedySelect`` — standard (1 - 1/e) greedy on a monotone submodular F."""
    selected: List[str] = []
    remaining = [c.id for c in bank.pool]
    while len(selected) < k and remaining:
        best_id, best_value = None, bank.objective(selected)
        for cand_id in remaining:
            value = bank.objective(selected + [cand_id])
            if value > best_value:
                best_value, best_id = value, cand_id
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


def _local_search(bank: RelayBank, seeds: Sequence[str], max_passes: int = 3) -> List[str]:
    """Objective-improving single-element swaps; never decreases ``F``."""
    current = [s for s in seeds if s in bank._index]
    pool_ids = [c.id for c in bank.pool]
    for _ in range(max_passes):
        improved = False
        for slot in range(len(current)):
            base = bank.objective(current)
            best_value, best_id = base, None
            for cand_id in pool_ids:
                if cand_id in current:
                    continue
                trial = list(current)
                trial[slot] = cand_id
                value = bank.objective(trial)
                if value > best_value:
                    best_value, best_id = value, cand_id
            if best_id is not None:
                current[slot] = best_id
                improved = True
        if not improved:
            break
    return current


def curate_seed_population(bank: RelayBank, k: Optional[int] = None) -> List[Candidate]:
    """``S* = argmax_{S ∈ {S_g, S_o}} F_{C_τ}(S)`` (Eq. 11-13).

    The greedy start removes the arrival-order dependence baked into the
    streaming bank; the online start keeps whatever the streaming policy
    accumulated.  Both are polished by local search, and the better of the
    two wins — neither step can lower ``F``, so the greedy guarantee holds.
    """
    if not bank.pool:
        return []
    k = bank.k if k is None else max(1, int(k))

    greedy = _local_search(bank, _greedy_select(bank, k))
    online = _local_search(bank, bank.bank[:k])

    best = greedy if bank.objective(greedy) >= bank.objective(online) else online
    logger.info(
        "Relay curation: greedy F=%.4f, online F=%.4f → chose %s (k=%d, pool=%d)",
        bank.objective(greedy),
        bank.objective(online),
        "greedy" if best is greedy else "online",
        len(best),
        len(bank.pool),
    )
    return [bank.pool[bank._index[i]] for i in best if i in bank._index]
