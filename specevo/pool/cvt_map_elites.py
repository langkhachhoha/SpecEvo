"""
CVT-MAP-Elites Pool with Multi-Strategy Sampling.

Single shared archive with multiple sampling strategies:
- UCB (Upper Confidence Bound) - exploration/exploitation balance
- AdaptiveRank - parameter-free Zipfian rank sampling driven by stagnation
- Uniform - random sampling for exploration
- Per-subscore - sample best performers on individual metrics
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from ..behavior import BehaviorExtractor, FeatureVector
from ..clients.base import ClientSpec
from ..core import EvaluationResult, Program
from .protocol import SampleResult


@dataclass
class Elite:
    """An elite program occupying a cell."""

    program: Program
    result: EvaluationResult
    behavior: FeatureVector
    raw_behavior: Optional[dict] = None  # Raw feature values for cross-island migration


@dataclass
class CellStats:
    """Statistics for a cell used by samplers."""

    n_samples: int = 0  # Times this cell was sampled
    n_successes: int = 0  # Times sampling led to accepted offspring

    def success_rate(self) -> float:
        if self.n_samples == 0:
            return 0.5  # Prior
        return self.n_successes / self.n_samples

    def ucb_score(self, total_samples: int, c: float = 2.0) -> float:
        """UCB1 score: exploitation (success rate) + exploration bonus.

        Uses acceptance rate rather than raw scores to avoid bias toward
        high-scoring cells that produce rejected clones. This encourages
        sampling cells that actually improve the archive.
        """
        if self.n_samples == 0:
            return float("inf")  # Unexplored cells have infinite priority
        exploitation = self.success_rate()
        exploration = c * math.sqrt(math.log(total_samples + 1) / self.n_samples)
        return exploitation + exploration


class Sampler(ABC):
    """Abstract base class for sampling strategies."""

    def __init__(self, name: str):
        self.name = name
        self.cell_stats: dict[int, CellStats] = {}

    def get_or_create_stats(self, cell: int) -> CellStats:
        if cell not in self.cell_stats:
            self.cell_stats[cell] = CellStats()
        return self.cell_stats[cell]

    def update(self, cell: int, success: bool) -> None:
        """Update statistics after observing outcome."""
        stats = self.get_or_create_stats(cell)
        stats.n_samples += 1
        if success:
            stats.n_successes += 1

    @abstractmethod
    def select_cells(self, elites: dict[int, Elite], n: int, context: Optional[dict] = None) -> list[int]:
        """Select n cells from the archive."""
        pass

    def get_stats_summary(self) -> dict:
        if not self.cell_stats:
            return {"n_cells": 0}
        rates = [s.success_rate() for s in self.cell_stats.values()]
        return {
            "n_cells": len(self.cell_stats),
            "total_samples": sum(s.n_samples for s in self.cell_stats.values()),
            "mean_success_rate": sum(rates) / len(rates) if rates else 0.0,
        }


class UCBSampler(Sampler):
    """Upper Confidence Bound sampling - balances exploration and exploitation."""

    def __init__(self, c: float = 2.0):
        super().__init__("ucb")
        self.c = c
        self._total_samples = 0

    def select_cells(self, elites: dict[int, Elite], n: int, context: Optional[dict] = None) -> list[int]:
        if not elites:
            return []
        self._total_samples += 1
        cells = list(elites.keys())

        # Compute UCB scores
        scores = []
        for cell in cells:
            stats = self.get_or_create_stats(cell)
            scores.append((stats.ucb_score(self._total_samples, self.c), cell))

        # Sort by UCB score descending, take top n
        scores.sort(reverse=True, key=lambda x: x[0])
        return [cell for _, cell in scores[: min(n, len(cells))]]


class AdaptiveRankSampler(Sampler):
    """Parameter-free Zipfian rank sampler driven by stagnation.

    Selection rule (per draw, without replacement):

        rank r(c)        ← 0-based index after sorting cells by primary score
                           descending
        β(t)             ← max(β_min, β_max · (1 - s(t)))   where s(t) is the
                           Posterior-Plateau Stagnation passed via context
        P(c) ∝ (r+1)^{-β}

    Why this is preferable to softmax-with-temperature:
      * Score-scale-invariant — the Zipfian distribution depends only on
        the rank, not on raw score gaps, so it does not collapse to near-
        deterministic mode when the archive's best cell is far ahead.
      * Single, derived knob β(t). It is *not* exposed to the bandit arms:
        the same sampler can act exploitative early (β large) and
        exploratory under stagnation (β small) without producing multiple
        arm variants. This removes the (sampler, softmax-T) cross product
        that previously confounded the joint bandit.
      * Reduces to a uniform sampler when β → 0 and to argmax when β → ∞,
        so AdaptiveRank covers both ends of the explore/exploit spectrum
        without dedicated "uniform" / "elitist" variants.
    """

    def __init__(self, beta_max: float = 2.0, beta_min: float = 0.2):
        super().__init__("adaptive_rank")
        self.beta_max = float(beta_max)
        self.beta_min = float(beta_min)
        self._last_beta: float = float(beta_max)

    def _compute_beta(self, stagnation: float) -> float:
        s = max(0.0, min(1.0, float(stagnation)))
        beta = self.beta_max * (1.0 - s)
        return max(self.beta_min, beta)

    def select_cells(self, elites: dict[int, Elite], n: int, context: Optional[dict] = None) -> list[int]:
        if not elites:
            return []

        # Stagnation defaults to 0 when caller doesn't pass it (e.g. PE
        # sampling, init phase). Producer always passes the live s(t).
        stagnation = 0.0
        if context is not None and "stagnation" in context:
            stagnation = context["stagnation"]
        beta = self._compute_beta(stagnation)
        self._last_beta = beta

        # Rank by primary score (descending); break ties deterministically
        # on cell index so behaviour is stable across runs.
        cells = list(elites.keys())
        cells.sort(key=lambda c: (-elites[c].result.primary_score, c))

        n_cells = len(cells)
        ranks = np.arange(1, n_cells + 1, dtype=float)
        weights = ranks ** (-beta)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            probs = np.full(n_cells, 1.0 / n_cells)
        else:
            probs = weights / total

        # Sampling without replacement; np.random.choice handles n>=k by
        # falling back to whatever we have.
        k = min(n, n_cells)
        idx = np.random.choice(n_cells, size=k, replace=False, p=probs)
        return [cells[int(i)] for i in idx]

    def get_stats_summary(self) -> dict:
        stats = super().get_stats_summary()
        stats["last_beta"] = self._last_beta
        stats["beta_max"] = self.beta_max
        stats["beta_min"] = self.beta_min
        return stats


class UniformSampler(Sampler):
    """Uniform random sampling for pure exploration."""

    def __init__(self):
        super().__init__("uniform")

    def select_cells(self, elites: dict[int, Elite], n: int, context: Optional[dict] = None) -> list[int]:
        if not elites:
            return []
        cells = list(elites.keys())
        return random.sample(cells, min(n, len(cells)))


class SubscoreSampler(Sampler):
    """Sample cells weighted by a specific subscore metric using softmax."""

    def __init__(self, subscore_key: str, display_name: str, temperature: float = 1.0):
        super().__init__(f"subscore_{subscore_key}")
        self.subscore_key = subscore_key
        self.display_name = display_name
        self.temperature = temperature

    def select_cells(self, elites: dict[int, Elite], n: int, context: Optional[dict] = None) -> list[int]:
        if not elites:
            return []

        # Get cells and their subscore values
        cells = list(elites.keys())
        scores = [elites[c].program.metadata.get(self.subscore_key, 0.0) for c in cells]

        # Softmax weighting by subscore
        max_s = max(scores) if scores else 0
        exp_s = [math.exp((s - max_s) / self.temperature) for s in scores]
        total = sum(exp_s)
        weights = [e / total for e in exp_s] if total > 0 else [1.0 / len(cells)] * len(cells)

        # Weighted sampling without replacement
        selected = []
        remaining_cells = cells.copy()
        remaining_weights = weights.copy()

        for _ in range(min(n, len(cells))):
            if not remaining_cells:
                break
            w_sum = sum(remaining_weights)
            if w_sum == 0:
                break
            probs = [w / w_sum for w in remaining_weights]
            idx = np.random.choice(len(remaining_cells), p=probs)
            selected.append(remaining_cells[idx])
            remaining_cells.pop(idx)
            remaining_weights.pop(idx)

        return selected


@dataclass
class SamplerModelConfig:
    sampler_name: str
    model: ClientSpec
    weight: float = 1.0

    # Joint bandit-arm dimensions for the prompt bank. None on either field
    # means "legacy arm", and update_bandit() must match with None on both.
    mutation_prompt_id: Optional[str] = None
    llm_temperature: Optional[float] = None

    # --- SAL Cơ chế D — Thompson Beta-Bernoulli bandit state ---
    # Posterior parameters over the accept-rate of this arm. Updated
    # incrementally by `update_bandit(...)` whenever an offspring produced by
    # this arm is evaluated. Reward = accept_indicator ∈ {0, 1}.
    alpha: float = 1.0  # successes + alpha_prior
    beta: float = 1.0  # failures + beta_prior
    # NEW BEST count for this arm (multiplicative bonus in the final weight).
    new_best_count: int = 0


class CVTMAPElitesPool:
    """
    CVT-MAP-Elites Pool with Multi-Strategy Sampling.

    Single shared archive with multiple sampling strategies.
    Each strategy can be associated with different LLM models.
    """

    def __init__(
        self,
        behavior_extractor: BehaviorExtractor,
        n_centroids: int = 1000,
        bounds_padding: float = 0.1,
        subscore_keys: Optional[list[str]] = None,
        data_driven_centroids: bool = False,
    ) -> None:
        self._extractor = behavior_extractor
        self._n_centroids = n_centroids
        self._feature_names = behavior_extractor.features
        self._n_dims = len(self._feature_names)

        # Adaptive bounds
        self._mins: Optional[np.ndarray] = None
        self._maxs: Optional[np.ndarray] = None
        self._ranges: Optional[np.ndarray] = None

        # Initialize centroids: uniform upfront tiling, or defer for data-driven init via set_centroids_from_data
        self._centroids: Optional[np.ndarray] = None
        if not data_driven_centroids:
            self._centroids = self._init_cvt_centroids()

        # Single shared archive
        self._elites: dict[int, Elite] = {}
        self._best_score: float = float("-inf")
        self._generation = 0

        # Initialize samplers. AdaptiveRank is the default parent-selector;
        # UCB and Uniform are retained as alternative arms for ablation.
        self._samplers: dict[str, Sampler] = {
            "ucb": UCBSampler(c=2.0),
            "adaptive_rank": AdaptiveRankSampler(),
            "uniform": UniformSampler(),
        }

        # Add per-subscore samplers
        if subscore_keys:
            for key in subscore_keys:
                sampler = SubscoreSampler(key, key)
                self._samplers[f"subscore_{key}"] = sampler

        # Sampler-model pairs for weighted selection
        self._sampler_model_pairs: list[SamplerModelConfig] = []
        self._total_weight: float = 0.0

    def _init_cvt_centroids(self) -> np.ndarray:
        """Initialize CVT centroids using k-means++ in normalized space."""
        n_dims = len(self._feature_names)
        n_samples = max(10000, self._n_centroids * 10)
        samples = np.random.uniform(0, 1, size=(n_samples, n_dims))

        kmeans = KMeans(n_clusters=self._n_centroids, init="k-means++", n_init=1, max_iter=100, random_state=42)
        kmeans.fit(samples)
        return kmeans.cluster_centers_

    def set_centroids_from_data(
        self,
        behavior_vectors: list[np.ndarray],
        n_centroids: int = 50,
    ) -> tuple[int, np.ndarray]:
        """
        Set centroids from behavior data using k-means clustering.

        Args:
            behavior_vectors: List of behavior vectors (already normalized via z-score+sigmoid)
            n_centroids: Number of centroids to create via k-means

        Returns:
            Tuple of (number of centroids created, labels array for each input vector)
        """
        if not behavior_vectors:
            raise ValueError("Need at least 1 behavior vector to build centroids")

        data = np.array(behavior_vectors)
        actual_n_centroids = min(n_centroids, len(data))

        kmeans = KMeans(n_clusters=actual_n_centroids, init="k-means++", n_init=3, max_iter=100, random_state=42)
        kmeans.fit(data)
        self._centroids = kmeans.cluster_centers_
        self._n_centroids = actual_n_centroids

        # No extra normalization - data is already [0,1] from z-score+sigmoid
        self._mins = np.zeros(self._n_dims)
        self._maxs = np.ones(self._n_dims)
        self._ranges = np.ones(self._n_dims)

        return self._n_centroids, kmeans.labels_

    @staticmethod
    def select_most_diverse(
        behavior_vectors: list[np.ndarray],
        k: int,
    ) -> list[int]:
        """
        Select k most diverse indices using farthest-first traversal.

        This guarantees maximum spread in behavior space, not just highest scores.

        Args:
            behavior_vectors: List of behavior vectors
            k: Number of diverse items to select

        Returns:
            List of indices of the k most diverse items
        """
        n = len(behavior_vectors)
        if n <= k:
            return list(range(n))

        behaviors = np.array(behavior_vectors)

        # Normalize for fair distance computation
        mins = behaviors.min(axis=0)
        maxs = behaviors.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        normalized = (behaviors - mins) / ranges

        # Farthest-first traversal
        selected = [0]  # Start with first item
        min_distances = np.full(n, np.inf)

        for _ in range(k - 1):
            # Update min distances to selected set
            last_selected = normalized[selected[-1]]
            for i in range(n):
                dist = np.linalg.norm(normalized[i] - last_selected)
                min_distances[i] = min(min_distances[i], dist)

            # Exclude already selected
            min_distances[selected] = -np.inf

            # Pick farthest from selected set
            next_idx = int(np.argmax(min_distances))
            selected.append(next_idx)

        return selected

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize a feature vector to [0, 1] range."""
        if self._mins is None:
            return np.full(self._n_dims, 0.5)
        normalized = (vec - self._mins) / self._ranges
        return np.clip(normalized, 0, 1)

    def _behavior_to_normalized_vector(self, behavior: FeatureVector) -> np.ndarray:
        """Convert FeatureVector to normalized numpy array."""
        raw = np.array([behavior[f] for f in self._feature_names])
        return self._normalize(raw)

    def _find_nearest_centroid(self, behavior: FeatureVector) -> int:
        """Find nearest centroid in normalized space."""
        vec = self._behavior_to_normalized_vector(behavior)
        distances = np.sum((self._centroids - vec) ** 2, axis=1)
        return int(np.argmin(distances))

    def preview_cell(self, program: Program, evaluation_scores: Optional[dict] = None) -> int:
        """Predict which cell a program would map to without mutating the archive."""
        behavior = self._extractor.extract(program, evaluation_scores)
        return self._find_nearest_centroid(behavior)

    def add(self, program: Program, evaluation_result: EvaluationResult) -> tuple[bool, int]:
        """Add program to archive. Returns (accepted, cell_index)."""
        if not evaluation_result.is_valid:
            return False, -1

        behavior = self._extractor.extract(program, evaluation_result.scores)
        raw_behavior = behavior.values.copy()  # Store for cross-island migration
        cell_index = self._find_nearest_centroid(behavior)
        new_score = evaluation_result.primary_score

        if cell_index not in self._elites:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior)
            self._best_score = max(self._best_score, new_score)
            return True, cell_index

        if new_score > self._elites[cell_index].result.primary_score:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior)
            self._best_score = max(self._best_score, new_score)
            return True, cell_index

        return False, cell_index

    def add_with_raw_behavior(
        self,
        program: Program,
        evaluation_result: EvaluationResult,
        raw_behavior: dict[str, float],
    ) -> bool:
        """
        Add a migrant program using raw behavior values for re-normalization.

        Used for cross-island migration where the source island has different
        adaptive bounds. The raw_behavior contains pre-normalized feature values
        that will be re-normalized using this island's bounds.
        """
        if not evaluation_result.is_valid:
            return False

        # Create FeatureVector from raw behavior
        behavior = FeatureVector(raw_behavior.copy())
        cell_index = self._find_nearest_centroid(behavior)
        new_score = evaluation_result.primary_score

        if cell_index not in self._elites:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior.copy())
            self._best_score = max(self._best_score, new_score)
            return True

        if new_score > self._elites[cell_index].result.primary_score:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior.copy())
            self._best_score = max(self._best_score, new_score)
            return True

        return False

    def add_at_cell(
        self,
        cell_index: int,
        program: Program,
        evaluation_result: EvaluationResult,
        behavior: FeatureVector,
    ) -> bool:
        """
        Add elite directly at a specific cell index (no re-extraction).

        Used during init when we already have behavior vectors and k-means labels.
        Only accepts if cell is empty or new score beats existing.
        """
        if not evaluation_result.is_valid:
            return False

        new_score = evaluation_result.primary_score
        raw_behavior = behavior.values.copy()

        if cell_index not in self._elites:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior)
            self._best_score = max(self._best_score, new_score)
            return True

        if new_score > self._elites[cell_index].result.primary_score:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior)
            self._best_score = max(self._best_score, new_score)
            return True

        return False

    def get_elites(self) -> dict[int, Elite]:
        """Get all elites in the archive."""
        return self._elites

    def get_elite(self, cell_index: int) -> Optional[Elite]:
        """Get elite at a specific cell index."""
        return self._elites.get(cell_index)

    def sample(
        self,
        sampler_name: str,
        n_parents: int = 4,
        context: Optional[dict] = None,
    ) -> SampleResult:
        """Sample from archive using specified strategy."""
        if sampler_name not in self._samplers:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        sampler = self._samplers[sampler_name]

        if not self._elites:
            raise ValueError("Archive is empty")

        cells = sampler.select_cells(self._elites, n_parents, context)

        if not cells:
            # Fallback to uniform if sampler returns nothing
            cells = random.sample(list(self._elites.keys()), min(n_parents, len(self._elites)))

        return SampleResult(
            parent=self._elites[cells[0]].program,
            inspirations=[self._elites[c].program for c in cells[1:]],
            metadata={
                "sampler": sampler_name,
                "source_cell": cells[0],
            },
        )

    def update_sampler(self, sampler_name: str, cell: int, success: bool) -> None:
        """Update sampler statistics after observing outcome."""
        if sampler_name in self._samplers:
            self._samplers[sampler_name].update(cell, success)

    def get_sampler_names(self) -> list[str]:
        """Get list of all sampler names."""
        return list(self._samplers.keys())

    def get_sampler(self, name: str) -> Sampler:
        """Get sampler by name."""
        return self._samplers[name]

    def register_sampler_model_pair(
        self,
        sampler_name: str,
        model: ClientSpec,
        weight: float = 1.0,
        mutation_prompt_id: Optional[str] = None,
        llm_temperature: Optional[float] = None,
    ) -> None:
        """Register a (sampler, model[, prompt_id, llm_temperature]) arm.

        Sampler-specific temperatures and cycle counts are no longer part
        of the arm space: AdaptiveRankSampler tunes itself from the live
        stagnation signal. The bandit's arm dimensions are now exactly
        (model, prompt_id, llm_temperature) so Thompson posteriors are not
        fragmented across spurious softmax-T variants.
        """
        if weight <= 0:
            raise ValueError("Weight must be positive")

        if sampler_name not in self._samplers:
            raise ValueError(
                f"Unknown sampler: {sampler_name}. Available: {list(self._samplers.keys())}"
            )

        self._sampler_model_pairs.append(
            SamplerModelConfig(
                sampler_name,
                model,
                weight,
                mutation_prompt_id=mutation_prompt_id,
                llm_temperature=llm_temperature,
            )
        )
        self._total_weight += weight

    def get_weighted_sampler_config(
        self,
        *,
        stagnation: Optional[float] = None,
        bandit_w_min: float = 0.05,
        bandit_new_best_gamma: float = 0.5,
    ) -> tuple[str, ClientSpec, Optional[str], Optional[float]]:
        """Pick the next (sampler, model) pair.

        Behaviour:
        - When ``stagnation`` is None we keep historical behaviour: choose by
          the static ``weight`` field on each pair (proportional roulette).
        - When ``stagnation`` is provided we use a Thompson-sampling Beta-
          Bernoulli bandit over the accept-rate of each arm, biased toward
          arms with NEW BEST history. Concretely:

              θ_i  ~  Beta(α_i, β_i)             # posterior sample
              raw_i = θ_i · (1 + γ · nb_i)^(1+s)
              w_i  = w_min + (1 - w_min) · raw_i / Σ raw_j

          and we then sample proportionally to w_i. Floor ``w_min`` keeps every
          arm playable so the bandit cannot collapse onto one combination.

        Args:
            stagnation: s(t) ∈ [0,1]; if None, bandit is disabled.
            bandit_w_min: weight floor per arm.
            bandit_new_best_gamma: γ — strength of NEW BEST bias.
        """
        if not self._sampler_model_pairs:
            raise ValueError("No sampler-model pairs registered. Call register_sampler_model_pair() first.")

        if stagnation is None:
            r = random.random() * self._total_weight
            cumulative = 0.0
            for pair in self._sampler_model_pairs:
                cumulative += pair.weight
                if r <= cumulative:
                    return pair.sampler_name, pair.model, pair.mutation_prompt_id, pair.llm_temperature
            last = self._sampler_model_pairs[-1]
            return last.sampler_name, last.model, last.mutation_prompt_id, last.llm_temperature

        # SAL Cơ chế D — Thompson Beta-Bernoulli + NEW BEST multiplicative bonus
        s = max(0.0, min(1.0, float(stagnation)))
        exponent = 1.0 + s
        raw = np.empty(len(self._sampler_model_pairs), dtype=float)
        for i, pair in enumerate(self._sampler_model_pairs):
            theta = float(np.random.beta(max(pair.alpha, 1e-6), max(pair.beta, 1e-6)))
            bonus = (1.0 + bandit_new_best_gamma * pair.new_best_count) ** exponent
            raw[i] = theta * bonus
        total = float(raw.sum())
        if total <= 0.0 or not np.isfinite(total):
            normalized = np.full(len(self._sampler_model_pairs), 1.0 / len(self._sampler_model_pairs))
        else:
            normalized = raw / total

        # Mix with floor: w_i = w_min + (1 - N*w_min) * p_i  guarantees
        # w_i >= w_min and Σ w_i = 1 when 0 ≤ N*w_min ≤ 1. Clamp w_min if a
        # caller picks a value that would make the floor exceed 1/N.
        n_arms = len(normalized)
        w_min = float(max(0.0, min(bandit_w_min, 1.0 / max(n_arms, 1))))
        weights = w_min + (1.0 - w_min * n_arms) * normalized
        weights = np.clip(weights, 1e-9, None)
        weights /= weights.sum()

        idx = int(np.random.choice(len(self._sampler_model_pairs), p=weights))
        chosen = self._sampler_model_pairs[idx]
        return chosen.sampler_name, chosen.model, chosen.mutation_prompt_id, chosen.llm_temperature

    # ------------------------------------------------------------------
    # SAL Cơ chế D — bandit posterior updates
    # ------------------------------------------------------------------

    def update_bandit(
        self,
        sampler_name: str,
        model: ClientSpec,
        *,
        accepted: bool,
        is_new_best: bool = False,
        mutation_prompt_id: Optional[str] = None,
        llm_temperature: Optional[float] = None,
    ) -> None:
        """Update Beta posterior and NEW BEST counter for the matching arm.

        Looks up the (sampler, model, prompt_id, llm_temperature) arm and
        increments α / β. Idempotent on unknown arms (silently no-ops) so
        callers don't need to special-case bundle or paradigm-shift
        evaluations that don't correspond to a bandit arm.

        When ``mutation_prompt_id`` or ``llm_temperature`` is None on a call,
        we match arms that also have None on those fields — preserving legacy
        behaviour when the prompt bank is disabled.
        """
        from ..clients.base import client_name as _client_name

        target_model = _client_name(model)
        for pair in self._sampler_model_pairs:
            if pair.sampler_name != sampler_name:
                continue
            if _client_name(pair.model) != target_model:
                continue
            if pair.mutation_prompt_id != mutation_prompt_id:
                continue
            # Compare llm_temperature with float tolerance to avoid mismatches
            # from JSON / arithmetic float drift.
            if pair.llm_temperature is None or llm_temperature is None:
                if pair.llm_temperature is not llm_temperature:
                    continue
            else:
                if abs(pair.llm_temperature - llm_temperature) > 1e-9:
                    continue
            if accepted:
                pair.alpha += 1.0
            else:
                pair.beta += 1.0
            if is_new_best:
                pair.new_best_count += 1
            return

    def get_bandit_stats(self) -> list[dict]:
        """Return per-arm bandit posterior summary (for logging / snapshot)."""
        stats = []
        for pair in self._sampler_model_pairs:
            total = pair.alpha + pair.beta
            mean = pair.alpha / total if total > 0 else 0.5
            stats.append(
                {
                    "sampler": pair.sampler_name,
                    "model": str(pair.model) if not hasattr(pair.model, "model") else pair.model.model,
                    "mutation_prompt_id": pair.mutation_prompt_id,
                    "llm_temperature": pair.llm_temperature,
                    "alpha": pair.alpha,
                    "beta": pair.beta,
                    "posterior_mean": mean,
                    "new_best_count": pair.new_best_count,
                }
            )
        return stats

    def best(self, metric: str = "score") -> Program:
        """Return best program in archive."""
        if not self._elites:
            raise ValueError("Archive is empty")

        best_elite = max(self._elites.values(), key=lambda e: e.result.primary_score)
        return best_elite.program

    def size(self) -> int:
        return len(self._elites)

    def clear(self) -> int:
        """
        Clear all elites from the archive.

        Returns the number of elites removed.
        """
        n_removed = len(self._elites)
        self._elites.clear()
        self._best_score = float("-inf")
        return n_removed

    def get_top_elites(self, n: int) -> list[Elite]:
        """
        Get top n elites by score.

        Returns list of Elite objects sorted by score descending.
        """
        if not self._elites:
            return []

        sorted_elites = sorted(self._elites.values(), key=lambda e: e.result.primary_score, reverse=True)
        return sorted_elites[:n]

    def on_generation_complete(self) -> None:
        self._generation += 1

    # ------------------------------------------------------------------
    # Adaptive Island Expansion — open a brand-new cell at the
    # candidate's own behaviour vector. Used by Punctuated Equilibrium
    # when a candidate fails standard admission under high stagnation.
    # ------------------------------------------------------------------

    def add_as_new_cell(
        self,
        program: Program,
        evaluation_result: EvaluationResult,
        behavior: FeatureVector,
    ) -> Optional[int]:
        """Append a new centroid at this candidate's behaviour vector and
        seed the new cell with the candidate.

        Returns the new cell index (a fresh integer ≥ existing
        ``n_centroids``), or ``None`` when the operation cannot proceed
        (invalid result, missing centroid table, non-finite behaviour
        vector).
        """
        if not evaluation_result.is_valid:
            return None
        if self._centroids is None:
            return None
        try:
            vec = self._behavior_to_normalized_vector(behavior)
        except Exception:
            return None
        if vec.shape != (self._centroids.shape[1],):
            return None
        if not np.all(np.isfinite(vec)):
            return None

        self._centroids = np.vstack([self._centroids, vec[None, :]])
        new_idx = int(self._centroids.shape[0] - 1)
        self._n_centroids = new_idx + 1

        raw_behavior = behavior.values.copy()
        self._elites[new_idx] = Elite(program, evaluation_result, behavior, raw_behavior)
        new_score = evaluation_result.primary_score
        self._best_score = max(self._best_score, new_score)
        return new_idx

    # ------------------------------------------------------------------
    # SAL Cơ chế B — behaviorally-far elite for contrastive context
    # ------------------------------------------------------------------

    def select_diverse_elite_from(self, parent_cell: int) -> Optional[Elite]:
        """Return the elite whose behaviour vector is farthest from `parent_cell`.

        Used when stagnation depth s(t) ≥ context_threshold: the producer
        appends this elite as a contrasting inspiration so the mutation prompt
        sees not just "parent + nearby elites" but also "a structurally
        different solution".

        Returns None when the archive has fewer than 2 elites or the cell is
        missing from the archive.
        """
        if not self._elites or parent_cell not in self._elites or len(self._elites) < 2:
            return None
        parent_behavior = self._elites[parent_cell].behavior
        try:
            parent_vec = np.array([parent_behavior[f] for f in self._feature_names], dtype=float)
        except Exception:
            return None

        best_idx: Optional[int] = None
        best_dist: float = -1.0
        for cell_idx, elite in self._elites.items():
            if cell_idx == parent_cell:
                continue
            try:
                vec = np.array([elite.behavior[f] for f in self._feature_names], dtype=float)
            except Exception:
                continue
            dist = float(np.linalg.norm(vec - parent_vec))
            if dist > best_dist:
                best_dist = dist
                best_idx = cell_idx

        if best_idx is None:
            return None
        return self._elites[best_idx]

    def best_elite(self) -> Optional[Elite]:
        """Return the elite with the highest primary score (or None)."""
        if not self._elites:
            return None
        return max(self._elites.values(), key=lambda e: e.result.primary_score)

    def get_stats(self) -> dict:
        stats = {
            "archive_size": self.size(),
            "n_centroids": self._n_centroids,
            "best_score": self._best_score,
            "generation": self._generation,
            "samplers": {name: sampler.get_stats_summary() for name, sampler in self._samplers.items()},
        }
        if self._mins is not None:
            stats["learned_bounds"] = {
                f: (float(self._mins[i]), float(self._maxs[i])) for i, f in enumerate(self._feature_names)
            }
        return stats

    def get_archive_snapshot(self) -> dict:
        """
        Get a JSON-serializable snapshot of the entire archive state.

        Returns a dict containing:
        - metadata: archive stats and configuration
        - elites: list of all elite programs with their scores and behaviors
        - sampler_stats: per-sampler statistics
        """
        elites_data = []
        for cell_idx, elite in self._elites.items():
            elite_data = {
                "cell_index": cell_idx,
                "program_id": str(elite.program.id),
                "content": elite.program.content,
                "code": elite.program.content,
                "scores": elite.result.scores,
                "primary_score": elite.result.primary_score,
                "behavior": elite.behavior.values,
                "metadata": elite.program.metadata,
                "created_at": elite.program.created_at.isoformat() if elite.program.created_at else None,
            }
            elites_data.append(elite_data)

        # Sort by primary score descending
        elites_data.sort(key=lambda x: x["primary_score"], reverse=True)

        sampler_stats = {}
        for name, sampler in self._samplers.items():
            stats = sampler.get_stats_summary()
            # Add per-cell stats
            cell_stats = {}
            for cell, cs in sampler.cell_stats.items():
                cell_stats[str(cell)] = {
                    "n_samples": cs.n_samples,
                    "n_successes": cs.n_successes,
                    "success_rate": cs.success_rate(),
                }
            stats["cell_stats"] = cell_stats
            sampler_stats[name] = stats

        snapshot = {
            "metadata": {
                "archive_size": self.size(),
                "n_centroids": self._n_centroids,
                "best_score": self._best_score,
                "generation": self._generation,
                "feature_names": self._feature_names,
                "centroids": self._centroids.tolist() if self._centroids is not None else None,
                "normalization": {
                    "mins": self._mins.tolist() if self._mins is not None else None,
                    "maxs": self._maxs.tolist() if self._maxs is not None else None,
                    "ranges": self._ranges.tolist() if self._ranges is not None else None,
                },
            },
            "elites": elites_data,
            "sampler_stats": sampler_stats,
        }

        if self._mins is not None:
            snapshot["metadata"]["learned_bounds"] = {
                f: {"min": float(self._mins[i]), "max": float(self._maxs[i])} for i, f in enumerate(self._feature_names)
            }

        return snapshot
