"""ClusterArchive — adaptive MAP-Elites with hybrid AST + description-embedding
behavior signatures.

This module replaces the over-engineered Pool from prior SpecEvo versions.
The design rests on three ideas, kept deliberately simple:

1. **Hybrid behavior signature.** Every program receives one fixed-length
   feature vector built by concatenating:

       - 14 hand-crafted AST counts (depth, cyclomatic complexity, loop
         nesting, comprehension count, etc.) — the framework's original behavior
         space; captures structural shape.
       - A PCA-reduced description embedding (text-embedding-3-small,
         1536-d → ``embedding_dim`` ≈ 8). Captures the model's own
         judgement of *what the program is doing*. Critical for
         distinguishing paradigms that share AST shape (e.g. gradient
         descent vs simulated annealing both have one outer loop +
         one if).

   Either half can be ablated independently. The two halves are
   z-score-standardised online (Welford) before concatenation so neither
   dominates the cosine.

2. **Adaptive cell re-clustering.** Instead of pre-allocating 1000 CVT
   centroids that mostly stay empty, the archive **re-runs KMeans every
   K evaluations** on the live population. Centroids therefore *follow*
   the search trajectory: when the population concentrates around one
   paradigm, cells shrink and split within that region; when a paradigm
   shift produces a brand-new direction, the next re-cluster opens a
   cell for it. Centroids are warm-started from the previous iteration
   so cell identity is continuous across re-clusters.

3. **No heuristics.** A program is admitted iff its score is *strictly
   greater* than the current cell incumbent. There is no grace period,
   no priority boost, no Hall-of-Fame, no quota. Diversity is the
   *output* of clustering, not an input forced via thresholds.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

from .ast_features import N_AST_FEATURES, compute_ast_features

logger = logging.getLogger(__name__)


#: Coarse producer labels. The orchestrator tags Speculator output more
#: finely (``mutate_<template>``, ``crossover_<template>``, ``mutate_targeted``)
#: and ``snaplog.classify_producer`` folds those back onto these buckets.
Source = Literal[
    "init",
    "init_seed",
    "mutate",
    "crossover",
    "repair",
    "navigator",
    "navigator_variant",
]


@dataclass
class Program:
    code: str
    description: str
    score: float
    embedding: np.ndarray  # raw description embedding from LLM
    source: Source = "mutate"
    created_at_eval: int = 0
    uses_count: int = 0
    cell_id: int = -1  # assigned by archive
    ast_vec: np.ndarray | None = None  # 14-d, lazy-filled
    behavior_vec: np.ndarray | None = None  # standardised hybrid, filled by archive
    metrics: dict = field(default_factory=dict)  # full score_fn dict (dev/test/overall, ...)

    def short_repr(self) -> str:
        desc = self.description.replace("\n", " ")
        if len(desc) > 80:
            desc = desc[:77] + "…"
        return f"<Program score={self.score:.3f} src={self.source} desc='{desc}'>"


@dataclass
class ArchiveConfig:
    """Configuration for :class:`ClusterArchive`.

    Knobs are intentionally few. Three are exposed as paper ablations:

    * :attr:`use_ast` (ablation A1: turn off → embedding-only behavior)
    * :attr:`use_embedding` (ablation A2: turn off → AST-only behavior)
    * :attr:`adaptive_recluster` (ablation A3: turn off → KMeans once,
      then fixed)
    """

    n_cells: int = 50
    """Target number of cells (paradigm niches), held **fixed** across
    the run. KMeans always uses ``k = n_cells`` (no ``min(n_cells, n)``
    capping). Before enough programs exist to fit KMeans, each admit
    gets its own cell id, but the *target* is always ``n_cells``. After
    the first fit, all ``n_cells`` centroids are kept alive — even those
    whose Voronoi region is currently empty — so the archive can grow
    *toward* ``n_cells`` instead of collapsing to ``num_occupied_cells``.
    The runtime invariant is therefore ``num_occupied_cells ≤ n_cells``
    and ``num_occupied_cells → n_cells`` as the search fills the space."""

    embedding_dim: int = 8
    """PCA target dimension for the description embedding half. 8 is
    enough to separate paradigm classes empirically (gradient/SA/greedy
    sit on the first 3-4 principal components) and small enough not to
    swamp the 14 AST features."""

    recluster_every: int = 30
    """Re-run KMeans every N admissions (counting only accepts, not
    rejects, so the cadence scales with the speed of progress).
    Centroids are warm-started from the previous run via
    ``init=previous_centroids`` so cell IDs stay continuous across
    re-clusters — important for the rank-sampler's per-cell statistics."""

    min_admits_before_cluster: int = 16
    """Below this many admitted programs, the archive uses a degenerate
    'each program is its own cell' assignment (cell_id = index). Above
    this floor, KMeans is fit; when ``n < n_cells`` we still ask KMeans
    for ``k = n_cells`` clusters by *padding* the population with
    synthetic noise points around the existing programs — this keeps
    the centroid grid at exactly ``n_cells`` from the very first fit
    so the rest of the run can fill cells lazily rather than starting
    with ``min(n_cells, n)`` and never recovering."""

    use_ast: bool = True
    """Component A1. Include the 14-d AST count features in the behavior
    signature. Turn off for the ablation 'embedding-only' run."""

    use_embedding: bool = True
    """Component A2. Include the PCA-reduced description embedding in
    the behavior signature. Turn off for the ablation 'AST-only' run."""

    adaptive_recluster: bool = True
    """Component A3. Re-run KMeans periodically. Turn off to fit KMeans
    exactly once (at ``min_admits_before_cluster``) and freeze cells
    thereafter — closer to CVT-MAP-Elites' static centroid grid."""


class _WelfordStats:
    """Online mean/std per feature dimension. Used to z-score-standardise
    AST counts and embedding components so the two halves of the behavior
    vector contribute on the same scale to KMeans."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def standardise(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return x - self.mean
        var = self.M2 / max(self.n - 1, 1)
        std = np.sqrt(np.maximum(var, 1e-9))
        return (x - self.mean) / std


class ClusterArchive:
    """Adaptive MAP-Elites archive with hybrid behavior signature.

    Thread-safe via a single ``threading.RLock``. The public surface is
    intentionally small: ``add``, ``programs``, ``cells``, ``best``,
    ``cell_of``.
    """

    def __init__(self, config: ArchiveConfig | None = None) -> None:
        self.config = config or ArchiveConfig()
        self._programs: list[Program] = []
        self._lock = threading.RLock()
        self._admits_since_recluster: int = 0
        # KMeans state — lazy. ``_centroids`` is None until the first
        # cluster operation; before that, each program is its own cell.
        self._centroids: np.ndarray | None = None
        # PCA basis for the embedding half (fitted on first recluster
        # call when ``use_embedding`` is on).
        self._pca_basis: np.ndarray | None = None
        self._pca_mean: np.ndarray | None = None
        # Welford stats for the standardisation step, one per half.
        self._ast_stats = _WelfordStats(N_AST_FEATURES)
        # Embedding stats sized lazily on first PCA fit.
        self._emb_stats: _WelfordStats | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._programs)

    def programs(self) -> list[Program]:
        with self._lock:
            return list(self._programs)

    def best(self) -> Program | None:
        with self._lock:
            if not self._programs:
                return None
            return max(self._programs, key=lambda p: p.score)

    def cells(self) -> dict[int, Program]:
        """Mapping ``cell_id -> top-score program in that cell``."""
        with self._lock:
            out: dict[int, Program] = {}
            for p in self._programs:
                cur = out.get(p.cell_id)
                if cur is None or p.score > cur.score:
                    out[p.cell_id] = p
            return out

    def num_occupied_cells(self) -> int:
        with self._lock:
            if not self._programs:
                return 0
            return len({p.cell_id for p in self._programs})

    def mark_used(self, program: Program) -> None:
        with self._lock:
            for p in self._programs:
                if p is program:
                    p.uses_count += 1
                    return

    def add(self, program: Program) -> tuple[bool, str]:
        """Admit *program* iff its score beats the incumbent of its cell.

        Returns ``(accepted, reason)``. Reasons:

        - ``"added"`` — first program in this cell.
        - ``"replaced"`` — beat the cell incumbent.
        - ``"dropped_worse"`` — did not beat the cell incumbent.
        - ``"no_embedding"`` — refused; needs an embedding to be located.
        """
        with self._lock:
            if program.embedding is None or program.embedding.size == 0:
                return False, "no_embedding"

            # Compute the AST half once; the behavior vector itself is
            # filled inside ``_assign_cell`` because it depends on the
            # current Welford stats and (optionally) PCA basis.
            if program.ast_vec is None:
                program.ast_vec = compute_ast_features(program.code)

            # Update online stats *before* assigning the cell so the
            # newcomer participates in its own standardisation. This
            # avoids the first few admits being squashed to all-zeros.
            self._ast_stats.update(program.ast_vec.astype(np.float64))
            if self._emb_stats is not None and self.config.use_embedding:
                emb = program.embedding.astype(np.float64)
                # Project through current PCA basis before stat update.
                if self._pca_basis is not None and self._pca_mean is not None:
                    proj = (emb - self._pca_mean) @ self._pca_basis
                    self._emb_stats.update(proj)

            # Assign cell.
            program.behavior_vec = self._make_behavior_vec(program)
            program.cell_id = self._assign_cell(program.behavior_vec)

            # Cell incumbent comparison.
            incumbent = self._cell_incumbent(program.cell_id)
            if incumbent is None:
                self._programs.append(program)
                self._admits_since_recluster += 1
                self._maybe_recluster()
                return True, "added"
            if program.score > incumbent.score:
                # Replace incumbent. Keep both records in `_programs`?
                # No — to stay tight, we drop the old incumbent and let
                # `_admits_since_recluster` decide when to re-evaluate
                # cell boundaries.
                self._programs[:] = [p for p in self._programs if p is not incumbent]
                self._programs.append(program)
                self._admits_since_recluster += 1
                self._maybe_recluster()
                return True, "replaced"
            return False, "dropped_worse"

    def force_add(self, program: Program) -> tuple[bool, str]:
        """Forcefully admit *program*, dropping the weakest program in the
        archive if normal :meth:`add` rejected it.

        Intended for Navigator reframe escapes — when the search is deeply
        stagnated and a frontier seed deserves an archive slot even if its
        score is below its cell's incumbent. Behaviour:

        - First try a normal :meth:`add`. If it returns ``"added"`` or
          ``"replaced"``, just propagate that.
        - Else (``"dropped_worse"``): remove the single lowest-score
          program in the entire archive, then retry the insertion.
        - If the archive is empty (no weakest to drop), or the embedding
          is missing, fall back to the original failure reason.

        Returns ``(accepted, reason)`` with reasons mirroring :meth:`add`
        plus ``"forced"`` when the weakest-drop path was taken.
        """
        accepted, reason = self.add(program)
        if accepted:
            return accepted, reason
        if reason != "dropped_worse":
            # ``no_embedding`` or any other non-cell rejection — don't
            # paper over a real bug by force-inserting.
            return accepted, reason

        with self._lock:
            if not self._programs:
                return False, reason
            weakest = min(self._programs, key=lambda p: p.score)
            self._programs[:] = [p for p in self._programs if p is not weakest]
            # Re-run admit. Cell assignment uses the existing centroids,
            # which is what we want — we are not re-clustering just
            # because we dropped one program. The dropped slot is now
            # either (a) empty if it was a 1-program cell, in which case
            # ``add`` will return "added", or (b) refilled by the next
            # cell incumbent search; if our new program lives in a
            # different cell, it still gets in via "added"/"replaced".
            accepted2, reason2 = self.add(program)
            if accepted2:
                return True, "forced"
            # Extremely unlikely: even after dropping the weakest, the
            # new program still lost to a stronger incumbent in its cell.
            # Put the weakest back to avoid losing a slot for nothing.
            self._programs.append(weakest)
            return False, reason2

    def cell_of(self, program: Program) -> int:
        """Cell id of *program* under the *current* centroids. Useful
        for logging and for the sampler to recompute cell-membership
        after a re-cluster without re-touching the program list."""
        with self._lock:
            if program.behavior_vec is None:
                return -1
            return self._assign_cell(program.behavior_vec)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cell_incumbent(self, cell_id: int) -> Program | None:
        members = [p for p in self._programs if p.cell_id == cell_id]
        if not members:
            return None
        return max(members, key=lambda p: p.score)

    def _make_behavior_vec(self, program: Program) -> np.ndarray:
        """Build the standardised hybrid signature for one program.

        Layout: ``[ast_standardised | emb_pca_standardised]``. Either
        half is replaced with an empty array when its toggle is off, so
        the dimensionality is consistent within one archive instance."""
        parts: list[np.ndarray] = []
        if self.config.use_ast:
            ast_std = self._ast_stats.standardise(program.ast_vec.astype(np.float64))
            parts.append(ast_std.astype(np.float32))
        if self.config.use_embedding and self._pca_basis is not None and self._pca_mean is not None:
            emb = program.embedding.astype(np.float64)
            proj = (emb - self._pca_mean) @ self._pca_basis  # (embedding_dim,)
            if self._emb_stats is not None and self._emb_stats.n >= 2:
                proj = self._emb_stats.standardise(proj)
            parts.append(proj.astype(np.float32))
        if not parts:
            # Both halves disabled — degenerate, but support it for
            # internal correctness. Returns a 1-d zero vector so KMeans
            # has *something* to cluster (each program lands in cell 0).
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def _assign_cell(self, behavior_vec: np.ndarray) -> int:
        """Return the cell id (centroid index) for *behavior_vec*.

        Pre-cluster phase (centroids = None): each admit gets a fresh
        cell id equal to its admit order. This means the early phase
        behaves like 'every program is its own niche' — exactly what
        we want before there are enough points for KMeans to be stable.
        """
        if self._centroids is None:
            # Pre-cluster degenerate mode: fresh cell id per program.
            return len(self._programs)
        if behavior_vec.shape[0] != self._centroids.shape[1]:
            # Dimension mismatch (happens for the very first admit after
            # PCA is fitted but before the program's vec was recomputed).
            # Pad/truncate defensively rather than crashing.
            d = self._centroids.shape[1]
            if behavior_vec.shape[0] < d:
                behavior_vec = np.concatenate(
                    [behavior_vec, np.zeros(d - behavior_vec.shape[0], dtype=np.float32)]
                )
            else:
                behavior_vec = behavior_vec[:d]
        # Nearest centroid (Euclidean — KMeans cells are Voronoi).
        dists = np.linalg.norm(self._centroids - behavior_vec, axis=1)
        return int(np.argmin(dists))

    def _maybe_recluster(self) -> None:
        cfg = self.config
        n = len(self._programs)
        # New invariant: only fit KMeans when we have at least
        # ``n_cells`` programs. Below that, every program keeps its own
        # cell id (set by ``_assign_cell`` in pre-cluster mode), so the
        # archive grows freely toward ``n_cells`` without coalescing.
        if n < max(cfg.min_admits_before_cluster, cfg.n_cells):
            return
        first_time = self._centroids is None
        if not first_time:
            if not cfg.adaptive_recluster:
                return
            if self._admits_since_recluster < cfg.recluster_every:
                return
        self._recluster()
        self._admits_since_recluster = 0

    def _recluster(self) -> None:
        """Fit / re-fit PCA on embeddings (if used) and KMeans on the
        resulting hybrid behavior matrix. Centroids are warm-started
        from the previous run when available (KMeans ``init``).

        Invariant: this method is only called when
        ``len(self._programs) >= n_cells``, so KMeans is always asked
        for exactly ``k = n_cells`` clusters (no ``min(n_cells, n)``).
        The number of occupied cells after coalescing can still be
        ``< n_cells`` if KMeans assigns no points to some centroids
        (empty Voronoi regions), but the centroid grid itself stays at
        ``n_cells`` so subsequent admits can land in those empty cells
        and grow the archive toward the full target.
        """
        cfg = self.config
        n = len(self._programs)
        if n < max(cfg.min_admits_before_cluster, cfg.n_cells):
            return

        # ---- (Re-)fit PCA on the description embeddings ----
        if cfg.use_embedding:
            embs = np.stack([p.embedding for p in self._programs]).astype(np.float64)
            self._pca_mean = embs.mean(axis=0)
            centered = embs - self._pca_mean
            # SVD-based PCA; numpy handles the under-sampled case
            # (n_samples < dim) by returning min(n_samples, dim) basis
            # vectors. We take the top ``embedding_dim``.
            dim = min(cfg.embedding_dim, centered.shape[1], centered.shape[0])
            try:
                _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
                self._pca_basis = vt[:dim].T  # shape (embed_raw_dim, dim)
            except np.linalg.LinAlgError:
                logger.warning("[Archive] PCA SVD failed; falling back to identity slice")
                self._pca_basis = np.eye(centered.shape[1], dim)
            # Reset/re-fit embedding standardisation stats on the new
            # projection so subsequent z-scoring is well-calibrated.
            self._emb_stats = _WelfordStats(self._pca_basis.shape[1])
            for p in self._programs:
                proj = (p.embedding.astype(np.float64) - self._pca_mean) @ self._pca_basis
                self._emb_stats.update(proj)

        # Refresh AST stats too (cheap, and the existing Welford may be
        # biased by an admit order that favoured one paradigm early).
        self._ast_stats = _WelfordStats(N_AST_FEATURES)
        for p in self._programs:
            self._ast_stats.update(p.ast_vec.astype(np.float64))

        # ---- Rebuild every program's behavior_vec under the fresh stats/PCA ----
        for p in self._programs:
            p.behavior_vec = self._make_behavior_vec(p)

        # ---- Fit KMeans on the standardised vectors ----
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("[Archive] sklearn missing; cell ids stay as admit-order")
            return

        X = np.stack([p.behavior_vec for p in self._programs]).astype(np.float64)
        k = cfg.n_cells  # FIXED across the run — no min(n_cells, n).
        # Warm-start from previous centroids when shapes line up.
        init: str | np.ndarray = "k-means++"
        if (
            self._centroids is not None
            and self._centroids.shape == (k, X.shape[1])
        ):
            init = self._centroids
            n_init = 1
        else:
            n_init = 3
        try:
            kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=50, random_state=None)
            labels = kmeans.fit_predict(X)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("[Archive] KMeans failed (%s); keeping previous centroids", e)
            return
        self._centroids = kmeans.cluster_centers_.astype(np.float32)

        # ---- Assign cell ids + enforce one-program-per-cell ----
        for p, lbl in zip(self._programs, labels):
            p.cell_id = int(lbl)
        # Coalesce: keep only the top-score program per cell. This is
        # the MAP-Elites contract — the side-effect is that the
        # population shrinks back to ``num_occupied_cells`` after each
        # re-cluster, which is what we want for downstream sampling.
        best_per_cell: dict[int, Program] = {}
        for p in self._programs:
            cur = best_per_cell.get(p.cell_id)
            if cur is None or p.score > cur.score:
                best_per_cell[p.cell_id] = p
        self._programs[:] = list(best_per_cell.values())
        logger.info(
            "[Archive] recluster: n=%d → %d/%d occupied cells "
            "(admits since last recluster=%d)",
            len(self._programs),
            len(best_per_cell),
            k,
            cfg.recluster_every,
        )

    # ------------------------------------------------------------------
    # Iteration support
    # ------------------------------------------------------------------

    def iter_programs(self) -> Iterable[Program]:
        with self._lock:
            yield from list(self._programs)
