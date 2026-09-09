"""
Punctuated Equilibrium: Periodic paradigm shifts in evolutionary search.

Inspired by the biological theory that evolution proceeds in bursts of rapid
change separated by long periods of stasis. This module implements periodic
"paradigm shift" events that inject fundamentally new solutions into the archive.
"""

import asyncio
import logging
import math
import random

import numpy as np
from sklearn.cluster import KMeans

from ..artifacts import ArtifactAdapter
from ..clients.base import client_name
from ..config import LeviConfig
from ..core import EvaluationResult
from ..pipeline.state import (
    BudgetLimitReached,
    PipelineState,
    StrategyRecord,
    coerce_finite_float,
)
from ..pool import CVTMAPElitesPool
from ..pool.cvt_map_elites import Elite
from ..prompts import PromptBundle
from ..selection import ComponentSelector, make_component_selector
from ..utils import ResilientProcessPool, coerce_score
from .prompts import get_budget_stage

logger = logging.getLogger(__name__)


# Patterns OpenRouter / LiteLLM use when the request's max_tokens exceeds
# what the account can pre-authorize. Matched case-insensitively against
# the stringified exception.
_TOKEN_CAP_ERROR_MARKERS = (
    "fewer max_tokens",
    "requires more credits",
    "max_tokens",
    "afford",
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
)

_TOKEN_RETRY_MIN = 1024
"""Stop halving when max_tokens would drop below this — at that point the
output budget is so small the call is unlikely to produce useful code."""


def _looks_like_token_cap_error(exc: BaseException) -> bool:
    """Heuristic: did the call fail because max_tokens was too high?

    We match on the exception's stringified message rather than a specific
    exception class because LiteLLM normalises provider errors into a
    generic ``APIError`` whose discriminator lives in the message body.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _TOKEN_CAP_ERROR_MARKERS)


def _extract_affordable_tokens(exc: BaseException) -> int | None:
    """Pull the "but can only afford N" hint out of an OpenRouter error.

    Returning the number lets us jump directly to a known-affordable
    max_tokens instead of blindly halving and probably overshooting again
    on the next try.
    """
    import re

    msg = str(exc)
    m = re.search(r"can only afford\s+(\d+)", msg, re.IGNORECASE)
    if m:
        try:
            return max(_TOKEN_RETRY_MIN, int(m.group(1)))
        except ValueError:
            return None
    return None


async def _acompletion_with_token_retry(
    state: PipelineState,
    *,
    model,
    prompt,
    temperature,
    timeout: float,
    initial_max_tokens: int | None = None,
    label: str = "PE",
    **extras,
):
    """Call ``state.acompletion`` with progressive ``max_tokens`` halving.

    Default behaviour: do NOT set ``max_tokens`` (let the model use its
    own ceiling). If the provider rejects the request with a token-cap
    error (OpenRouter "fewer max_tokens / afford N"), drop to the
    affordable number it suggested and halve from there on each retry
    until either the call succeeds or ``max_tokens`` falls below
    :data:`_TOKEN_RETRY_MIN`.

    Args:
        initial_max_tokens: pre-set ``max_tokens`` for the first call,
            or None to call without the parameter.

    Raises:
        BudgetLimitReached: propagated unchanged so callers can shut
            down the pipeline cleanly.
        Exception: the *last* observed provider error when retries are
            exhausted or the failure does not look like a token-cap
            issue (so the existing higher-level handlers can decide
            whether to skip the PE event or fall back).
    """
    current_max_tokens = initial_max_tokens
    attempt = 0
    last_exc: Exception | None = None
    while True:
        attempt += 1
        kwargs = dict(extras)
        if current_max_tokens is not None:
            kwargs["max_tokens"] = current_max_tokens
        try:
            return await state.acompletion(
                model,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )
        except BudgetLimitReached:
            raise
        except Exception as e:
            last_exc = e
            if not _looks_like_token_cap_error(e):
                # Some other failure (network, server-side bug, etc.) —
                # we don't know how to recover, let the caller decide.
                raise

            # First time we see a token-cap error: use the suggested
            # affordable number if present, otherwise start from a
            # conservative 32k cap (half of common 65k defaults).
            if attempt == 1:
                hint = _extract_affordable_tokens(e)
                if hint is not None:
                    # Provider tells us exactly what fits — go just under
                    # it (round to nearest 1024) so a small concurrent
                    # spend can't tip us back over the line.
                    safe = max(_TOKEN_RETRY_MIN, (hint // 1024) * 1024)
                    current_max_tokens = safe
                else:
                    current_max_tokens = 32768
                logger.warning(
                    f"[{label}] Token-cap error on attempt {attempt}; "
                    f"retrying with max_tokens={current_max_tokens}. "
                    f"Cause: {str(e)[-200:]}"
                )
                continue

            # Subsequent retries: halve.
            if current_max_tokens is None:
                current_max_tokens = 32768
            next_max_tokens = current_max_tokens // 2
            if next_max_tokens < _TOKEN_RETRY_MIN:
                logger.warning(
                    f"[{label}] Token-cap error on attempt {attempt}; "
                    f"giving up (next max_tokens={next_max_tokens} < {_TOKEN_RETRY_MIN}). "
                    f"Cause: {str(e)[-200:]}"
                )
                break
            current_max_tokens = next_max_tokens
            logger.warning(
                f"[{label}] Token-cap error on attempt {attempt}; "
                f"halving to max_tokens={current_max_tokens}. "
                f"Cause: {str(e)[-200:]}"
            )
    assert last_exc is not None
    raise last_exc


class PunctuatedEquilibrium:
    """
    Implements punctuated equilibrium for CVT-MAP-Elites.

    Periodically:
    1. Clusters occupied centroids into behavioral regions
    2. Selects best elite from each cluster as representative
    3. Generates paradigm shift solution using heavy model
    4. Generates variants using lighter models
    5. Inserts solutions into the archive
    """

    def __init__(
        self,
        config: LeviConfig,
        pool: CVTMAPElitesPool,
        executor: ResilientProcessPool,
        artifact_adapter: ArtifactAdapter,
        archive_lock: asyncio.Lock,
        state: PipelineState,
        main_component_selector: ComponentSelector | None = None,
    ):
        self.config = config
        self.pool = pool
        self.executor = executor
        self.artifact_adapter = artifact_adapter
        self.archive_lock = archive_lock
        self.state = state
        self.pe_config = config.punctuated_equilibrium
        self.main_component_selector = main_component_selector

        self.pe_component_selector: ComponentSelector | None = None
        self._is_bundle = getattr(artifact_adapter, "is_bundle_artifact", False)
        if self._is_bundle:
            self.pe_component_selector = make_component_selector(self.pe_config.component_selector)

    def _cluster_occupied_centroids(self, n_clusters_override: int | None = None) -> dict[int, list[int]]:
        """
        Cluster occupied centroids and return mapping of cluster_id -> cell_indices.

        Returns empty dict if not enough occupied cells for clustering.

        When `n_clusters_override` is provided (SAL Cơ chế E — Hard-PE) we use
        that value instead of `self.pe_config.n_clusters`, allowing a heavier
        partition of the archive on the rare "we're really stuck" trigger.
        """
        elites = self.pool.get_elites()
        target_n = n_clusters_override if n_clusters_override is not None else self.pe_config.n_clusters
        if len(elites) < target_n:
            logger.info(f"[PE] Not enough elites ({len(elites)}) for {target_n} clusters")
            return {}

        cell_indices = list(elites.keys())
        centroids = self.pool._centroids

        if centroids is None:
            logger.warning("[PE] Centroids not initialized")
            return {}

        # Get centroid vectors for occupied cells only
        occupied_centroids = centroids[cell_indices]

        n_clusters = min(target_n, len(cell_indices))
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=3,
            random_state=None,  # Different clustering each time
        )
        labels = kmeans.fit_predict(occupied_centroids)

        # Build cluster -> cell_indices mapping
        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            label_int = int(label)
            if label_int not in clusters:
                clusters[label_int] = []
            clusters[label_int].append(cell_indices[idx])

        return clusters

    def _select_cluster_representatives(
        self,
        clusters: dict[int, list[int]],
        *,
        farthest_first: bool = False,
    ) -> list[tuple[int, Elite]]:
        """
        Select a representative elite from each cluster.

        Default: per-cluster max-score (the legacy behaviour).

        SAL Cơ chế E `farthest_first=True`: from each cluster pick the elite
        whose behaviour vector is farthest from the centroid of all elites
        already picked. That diversifies the prompt context — the heavy
        model sees structurally distinct anchors, not score-similar ones.
        """
        representatives: list[tuple[int, Elite]] = []
        elites = self.pool.get_elites()

        if not farthest_first:
            for cluster_id, cell_indices in clusters.items():
                cluster_elites = [(idx, elites[idx]) for idx in cell_indices]
                _best_idx, best_elite = max(
                    cluster_elites, key=lambda x: x[1].result.primary_score
                )
                representatives.append((cluster_id, best_elite))
            return representatives

        # Farthest-first per-cluster selection (SAL Hard-PE).
        picked_vecs: list[np.ndarray] = []
        feature_names = self.pool._feature_names

        def _vec(elite: Elite) -> np.ndarray:
            try:
                return np.array([elite.behavior[f] for f in feature_names], dtype=float)
            except Exception:
                return np.zeros(len(feature_names), dtype=float)

        for cluster_id, cell_indices in clusters.items():
            cluster_elites = [(idx, elites[idx]) for idx in cell_indices]
            if not picked_vecs:
                # First cluster — fall back to max-score so we keep a strong
                # anchor in the prompt.
                _, best_elite = max(cluster_elites, key=lambda x: x[1].result.primary_score)
                representatives.append((cluster_id, best_elite))
                picked_vecs.append(_vec(best_elite))
                continue

            anchor_centroid = np.mean(np.stack(picked_vecs, axis=0), axis=0)
            best_dist = -1.0
            best_elite_here = cluster_elites[0][1]
            for _, elite in cluster_elites:
                v = _vec(elite)
                d = float(np.linalg.norm(v - anchor_centroid))
                if d > best_dist:
                    best_dist = d
                    best_elite_here = elite
            representatives.append((cluster_id, best_elite_here))
            picked_vecs.append(_vec(best_elite_here))

        return representatives

    def _should_fire_hard_pe(self) -> bool:
        """SAL Cơ chế E gate — fire a heavier PE when *really* stuck.

        Conditions (all must hold):
          - SAL enabled, mechanism E enabled.
          - We haven't burned the per-run hard-PE budget yet.
          - The last 2 PE triggers in a row produced no NEW BEST.
          - Stagnation depth s(t) ≥ hard_pe_threshold.
        """
        sal = self.config.sal
        if not (sal.enabled and sal.enable_e_hard_pe):
            return False
        if self.state.hard_pe_count >= sal.hard_pe_max_per_run:
            return False
        if self.state.consecutive_pe_no_best < 2:
            return False
        s = self.state.stagnation_depth(sal.tau)
        return s >= sal.hard_pe_threshold

    def _build_paradigm_shift_prompt(
        self,
        representatives: list[tuple[int, Elite]],
        n_evaluations: int,
        budget_progress: float = 0.0,
        target: str | None = None,
    ) -> str:
        if target is not None and self._is_bundle:
            return self.artifact_adapter.build_component_paradigm_shift_prompt(
                target,
                representatives,
                n_evaluations=n_evaluations,
                budget_progress=budget_progress,
            )

        # SAL Cơ chế A: pass stagnation + trajectory context to the code adapter.
        sal = self.config.sal
        stagnation: float | None = None
        best_score: float | None = None
        evals_since_best: int | None = None
        sal_thresholds: tuple[float, float] | None = None
        if sal.enabled and sal.enable_a_pe_staging:
            stagnation = self.state.stagnation_depth(sal.tau)
            evals_since_best = self.state.evals_since_best()
            best_score = (
                self.state.best_score_so_far
                if self.state.best_score_so_far != float("-inf")
                else None
            )
            sal_thresholds = (sal.pe_staging_mid_threshold, sal.pe_staging_late_threshold)

        strategy_log_block = ""
        if self.config.strategy_log.enabled:
            strategy_log_block = self.state.format_strategy_log(
                max_entries=self.config.strategy_log.max_entries
            )
            if strategy_log_block.strip():
                n_recent = min(
                    len(self.state.strategy_history),
                    self.config.strategy_log.max_entries,
                )
                logger.info(
                    f"[Strategy-Log] Injecting {n_recent} prior PE record(s) into heavy prompt "
                    f"({len(strategy_log_block)} chars, of {len(self.state.strategy_history)} total)"
                )
            else:
                logger.info(
                    "[Strategy-Log] No prior PE records yet; heavy prompt has no strategy block"
                )

        return self.artifact_adapter.build_paradigm_shift_prompt(
            representatives,
            n_evaluations=n_evaluations,
            budget_progress=budget_progress,
            stagnation=stagnation,
            best_score=best_score,
            evals_since_best=evals_since_best,
            sal_thresholds=sal_thresholds,
            strategy_log_block=strategy_log_block,
        )

    def _build_variant_prompt(
        self,
        base_code: str,
        base_score: float,
        target: str | None = None,
    ) -> str:
        if target is not None and self._is_bundle:
            base_bundle = PromptBundle.deserialize_loose(base_code)
            return self.artifact_adapter.build_component_variant_prompt(target, base_bundle, base_score)
        return self.artifact_adapter.build_variant_prompt(base_code, base_score)

    # ------------------------------------------------------------------
    # Strategy History — light-model summariser
    # ------------------------------------------------------------------

    async def _summarize_strategy(self, code: str) -> tuple[str, float]:
        """Return ``(summary, cost)`` from the light summariser model.

        ``summary`` is "" when summarisation is disabled, the adapter does
        not implement it, or the API call failed. ``cost`` is best-effort
        and may be 0.0 on failure. Emits an INFO log at each decision
        point so the strategy-log pipeline is observable in run.txt.
        """
        cfg = self.config.strategy_log
        if not cfg.enabled:
            logger.info("[Strategy-Log] disabled by config; skipping summarisation")
            return "", 0.0
        if not hasattr(self.artifact_adapter, "build_strategy_summary_prompt"):
            logger.info(
                "[Strategy-Log] adapter has no build_strategy_summary_prompt; skipping"
            )
            return "", 0.0
        if cfg.summariser_model is None:
            logger.info(
                "[Strategy-Log] summariser_model is None (mutation_models[0] not set?); skipping"
            )
            return "", 0.0
        prompt = self.artifact_adapter.build_strategy_summary_prompt(code)
        logger.info(
            f"[Strategy-Log] Summarising paradigm shift "
            f"(model={client_name(cfg.summariser_model)}, "
            f"max_tokens={cfg.summariser_max_tokens}, "
            f"temperature={cfg.summariser_temperature})"
        )
        try:
            response = await _acompletion_with_token_retry(
                self.state,
                model=cfg.summariser_model,
                prompt=[{"role": "user", "content": prompt}],
                temperature=cfg.summariser_temperature,
                initial_max_tokens=cfg.summariser_max_tokens,
                timeout=60,
                label="Strategy-Log",
            )
        except BudgetLimitReached:
            raise
        except Exception as e:
            logger.warning(f"[Strategy-Log] Summarisation failed: {e}")
            return "", 0.0
        cost = float(getattr(response, "cost", 0.0) or 0.0)
        text = (response.text or "").strip()
        # Keep all non-empty lines — the IDEA/QUALITY template is two
        # lines and the heavy prompt renders them nested. We just strip
        # leading/trailing blank lines and join the rest.
        kept_lines = [ln for ln in (line.rstrip() for line in text.splitlines()) if ln.strip()]
        summary = "\n".join(kept_lines)
        if summary:
            preview = summary.replace("\n", " | ")
            if len(preview) > 200:
                preview = preview[:197] + "..."
            logger.info(
                f"[Strategy-Log] Summary ok (cost=${cost:.4f}, "
                f"lines={len(kept_lines)}): {preview}"
            )
        else:
            logger.info(f"[Strategy-Log] Summary empty (cost=${cost:.4f})")
        return summary, cost

    # ------------------------------------------------------------------
    # Adaptive Island Expansion — open a new cell at the candidate's
    # own behaviour vector when stuck. Replaces both the old stagnation
    # rescue (which mutated the incumbent in-place) and the old
    # behaviour-buffer-driven archive growth: one mechanism, one trigger,
    # bounded by ``max_per_run`` and ``max_total_centroids``.
    # ------------------------------------------------------------------

    def _try_island_expansion(
        self,
        program,
        eval_result,
        *,
        stagnation: float,
    ) -> tuple[bool, int | None]:
        """Open a brand-new cell at this candidate's behaviour vector.

        Caller invokes this only when the standard ``pool.add(...)`` test
        already rejected the candidate. Returns
        ``(expanded, new_cell_index)``. When ``expanded`` is False the
        candidate stays out of the archive (callers should not retry).
        """
        cfg = self.config.adaptive_island
        if not cfg.enabled or self._is_bundle:
            return False, None
        if stagnation < cfg.stagnation_threshold:
            return False, None
        if self.state.island_expansion_count >= cfg.max_per_run:
            return False, None
        if self.pool._n_centroids >= cfg.max_total_centroids:
            return False, None

        behavior = self.pool._extractor.extract(program, eval_result.scores)
        new_cell = self.pool.add_as_new_cell(program, eval_result, behavior)
        if new_cell is None:
            return False, None
        self.state.island_expansion_count += 1
        new_score = eval_result.primary_score
        logger.info(
            f"[PE] Adaptive Island expansion: opened cell {new_cell} "
            f"(score={new_score if math.isfinite(new_score) else 'n/a'}, "
            f"s={stagnation:.2f}, expansions_used="
            f"{self.state.island_expansion_count}/{cfg.max_per_run}, "
            f"n_centroids={self.pool._n_centroids})"
        )
        return True, new_cell

    def _pick_pe_component(self) -> str | None:
        if not self._is_bundle or self.pe_component_selector is None:
            return None
        seed_bundle = getattr(self.artifact_adapter, "seed_bundle", None)
        if seed_bundle is None:
            return None
        context: dict = {}
        if self.pe_config.share_main_selector_stats and self.main_component_selector is not None:
            context["main_stats"] = self.main_component_selector.stats()
        return self.pe_component_selector.select(
            list(seed_bundle.editable_targets), context=context or None
        )

    async def _evaluate(self, code: str) -> dict:
        """Evaluate code using the executor."""
        return await self.artifact_adapter.evaluate(self.executor, code)

    async def trigger(self, n_evaluations: int, budget_progress: float = 0.0) -> dict:
        """
        Trigger a punctuated equilibrium event.

        Args:
            n_evaluations: Current evaluation count (for prompt context)
            budget_progress: Fraction of budget consumed (0-1)

        Returns:
            Dict with statistics about the PE event:
            {
                "triggered": bool,
                "paradigm_generated": bool,
                "paradigm_score": Optional[float],
                "paradigm_accepted": bool,
                "paradigm_cell": Optional[int],
                "variants_generated": int,
                "variants_accepted": int,
                "total_cost": float,
            }
        """
        stats = {
            "triggered": True,
            "paradigm_generated": False,
            "paradigm_score": None,
            "paradigm_accepted": False,
            "paradigm_cell": None,
            "variants_generated": 0,
            "variants_accepted": 0,
            "variant_cells": [],
            "total_cost": 0.0,
            "evaluations": [],
        }
        pe_evals_started = 0

        def can_start_pe_eval() -> bool:
            if self.state.budget_exhausted:
                return False
            if self.state.budget.evaluations is None:
                return True
            eval_limit = int(coerce_finite_float(self.state.budget.evaluations, default=0.0))
            if eval_limit <= 0:
                return False
            return (self.state.eval_count + pe_evals_started) < eval_limit

        # SAL Cơ chế E — check whether this trigger should be a Hard-PE
        # (heavier clustering, farthest-first reps, forced reasoning effort).
        sal = self.config.sal
        is_hard_pe = self._should_fire_hard_pe()
        hard_n_clusters = sal.hard_pe_n_clusters if is_hard_pe else None
        if is_hard_pe:
            self.state.hard_pe_count += 1
            logger.info(
                f"[PE] HARD-PE #{self.state.hard_pe_count}/{sal.hard_pe_max_per_run} "
                f"(s={self.state.stagnation_depth(sal.tau):.2f}, "
                f"consecutive_no_best={self.state.consecutive_pe_no_best})"
            )
            stats["hard_pe"] = True

        # Step 1: Cluster occupied centroids
        async with self.archive_lock:
            clusters = self._cluster_occupied_centroids(n_clusters_override=hard_n_clusters)

        if not clusters:
            stats["triggered"] = False
            return stats

        logger.info(f"[PE] Clustered {len(self.pool.get_elites())} elites into {len(clusters)} clusters")

        # Step 2: Select cluster representatives
        async with self.archive_lock:
            representatives = self._select_cluster_representatives(
                clusters, farthest_first=is_hard_pe
            )

        for cluster_id, elite in representatives:
            logger.info(f"[PE] Cluster {cluster_id} rep: score={elite.result.primary_score:.17g}")

        # Step 3: Generate paradigm shift solution
        heavy_models = self.pe_config.heavy_models
        if not heavy_models:
            heavy_models = [self.config.sampler_model_pairs[0].model]
        heavy_model = random.choice(heavy_models)

        pe_target = self._pick_pe_component()
        if pe_target is not None:
            logger.info(f"[PE] Selected component for paradigm shift: {pe_target}")
            stats["pe_target"] = pe_target

        ref_model = heavy_model
        prompt = self._build_paradigm_shift_prompt(
            representatives, n_evaluations, budget_progress, target=pe_target
        )
        ref_role = "paradigm_shift"

        # Determine the prompt stage that fired (early/mid/late) for logging
        # into strategy_history. The same routing is used inside the adapter,
        # so we recompute it here rather than threading the value out.
        sal = self.config.sal
        stage_stagnation: float | None = None
        if sal.enabled and sal.enable_a_pe_staging:
            stage_stagnation = self.state.stagnation_depth(sal.tau)
        pe_stage = get_budget_stage(
            budget_progress,
            stagnation=stage_stagnation,
            mid_threshold=sal.pe_staging_mid_threshold,
            late_threshold=sal.pe_staging_late_threshold,
        )
        best_before = (
            self.state.best_score_so_far
            if math.isfinite(self.state.best_score_so_far)
            else float("-inf")
        )

        try:
            extras = {}
            # SAL Cơ chế E — force higher reasoning effort on Hard-PE.
            effective_reasoning_effort = self.pe_config.reasoning_effort
            if is_hard_pe:
                effective_reasoning_effort = sal.hard_pe_reasoning_effort

            # Add reasoning_effort for DeepSeek models if configured
            if effective_reasoning_effort:
                if effective_reasoning_effort == "disabled":
                    # Disable reasoning entirely (e.g., for GLM models)
                    extras["extra_body"] = {"reasoning": {"enabled": False}}
                    logger.info(f"[PE] Reasoning disabled for {ref_role}")
                else:
                    extras["reasoning_effort"] = effective_reasoning_effort
                    logger.info(f"[PE] Using reasoning_effort={effective_reasoning_effort} for {ref_role}")

            response = await _acompletion_with_token_retry(
                self.state,
                model=ref_model,
                prompt=[{"role": "user", "content": prompt}],
                temperature=self.pe_config.temperature,
                timeout=300,
                label=f"PE/{ref_role}",
                **extras,
            )
            content = response.text
            cost = response.cost
            stats["total_cost"] += cost
        except BudgetLimitReached:
            logger.info("[PE] Budget exhausted before paradigm shift generation")
            return stats
        except Exception as e:
            logger.warning(f"[PE] {ref_role} generation failed (after retries): {e}")
            content = None

        # Elite-as-paradigm fallback — when the heavy model is unreachable
        # (credit exhausted, repeated token-cap rejection, transient
        # outage…), we still want the PE event to do useful work. Reuse
        # the best cluster representative as the "paradigm" so the variant
        # step can still mutate it. The strategy log records this with
        # ``stage`` annotated so post-hoc analysis can tell rescued PEs
        # apart from real heavy-model paradigms.
        paradigm_code: str | None = None
        used_elite_fallback = False
        if content is not None:
            if pe_target is not None and representatives:
                anchor_elite = max(representatives, key=lambda item: item[1].result.primary_score)[1]
                paradigm_code = self.artifact_adapter.extract_candidate(
                    content,
                    parent_content=anchor_elite.program.content,
                    target=pe_target,
                )
            else:
                paradigm_code = self.artifact_adapter.extract_candidate(content)
            if not paradigm_code:
                logger.warning("[PE] Failed to extract paradigm shift code; trying elite fallback")

        if not paradigm_code and representatives:
            fallback_elite = max(
                representatives, key=lambda item: item[1].result.primary_score
            )[1]
            paradigm_code = fallback_elite.program.content
            used_elite_fallback = True
            stats["paradigm_score"] = fallback_elite.result.primary_score
            stats["fallback_elite_paradigm"] = True
            logger.info(
                f"[PE] Using best elite as paradigm "
                f"(score={fallback_elite.result.primary_score:.4g}); "
                f"variants will mutate it as a rescue path."
            )

        if not paradigm_code:
            # No representatives + no heavy output — we have nothing to mutate.
            logger.warning("[PE] No paradigm code available even after elite fallback; aborting PE")
            return stats

        stats["paradigm_generated"] = not used_elite_fallback
        if used_elite_fallback:
            stats["paradigm_source"] = "elite_fallback"
        logger.debug("[PE] Paradigm shift code generated (%d chars)", len(paradigm_code))

        # Step 4: Evaluate paradigm shift solution
        if used_elite_fallback:
            # Reusing an archive elite — its score is already known, the
            # archive already contains it, so skip the evaluation entirely
            # and head straight to the variant step. ``paradigm_score`` is
            # set during the fallback assignment above.
            logger.info(
                "[PE] Elite-fallback path: skipping paradigm eval, going "
                "straight to variants."
            )
            score = stats["paradigm_score"]
            result = {"score": score, "skipped_eval": True}
            stats["evaluations"].append(
                {
                    "source": "paradigm_shift",
                    "model": "elite_fallback",
                    "score": score,
                    "accepted": False,  # already in archive
                    "skipped_eval": True,
                    "archive_size": self.pool.size(),
                }
            )
            # Fall through to the rest of the function, but flag so we
            # skip the archive insert block below.
        elif not can_start_pe_eval():
            logger.info("[PE] Skipping paradigm shift evaluation (budget exhausted)")
            stats["evaluations"].append(
                {
                    "source": "paradigm_shift",
                    "model": client_name(ref_model),
                    "error": "Budget exhausted",
                    "archive_size": self.pool.size(),
                }
            )
            return stats
        else:
            try:
                pe_evals_started += 1
                result = await self._evaluate(paradigm_code)
            except Exception as e:
                logger.warning(f"[PE] Paradigm shift evaluation failed: {e}")
                result = {"error": str(e)}

        if used_elite_fallback:
            # Skip the archive insertion + result-score parsing block —
            # the elite is already in the archive and its score is the
            # one we stored in stats["paradigm_score"] above. Use the
            # known score to drive the variant step below.
            score = stats["paradigm_score"]
        else:
            if "error" not in result:
                score, score_error = coerce_score(result)
                if score_error is not None:
                    logger.warning(f"[PE] Paradigm shift invalid score: {score_error}")
                    result = {"error": score_error}
                else:
                    result = dict(result)
                    result["score"] = score

            if "error" not in result:
                stats["paradigm_score"] = score

                program = self.artifact_adapter.make_program(
                    paradigm_code,
                    metadata={
                        "source": "punctuated_equilibrium",
                        "pe_type": "paradigm_shift",
                    },
                )
                eval_result = EvaluationResult(
                    scores=result,
                    is_valid=True,
                )

                async with self.archive_lock:
                    accepted, cell_idx = self.pool.add(program, eval_result)
                    island_opened = False
                    # Adaptive Island Expansion — when standard admission
                    # fails under high stagnation, open a brand-new cell at
                    # the candidate's own behaviour vector instead of
                    # discarding the work or evicting the incumbent.
                    if not accepted:
                        s_for_island = (
                            self.state.stagnation_depth(sal.tau)
                            if sal.enabled
                            else 0.0
                        )
                        island_opened, new_cell = self._try_island_expansion(
                            program, eval_result, stagnation=s_for_island
                        )
                        if island_opened:
                            accepted = True
                            cell_idx = new_cell

                stats["paradigm_accepted"] = accepted
                stats["paradigm_cell"] = cell_idx
                if island_opened:
                    stats["paradigm_island_expanded"] = True
                if pe_target is not None and self.pe_component_selector is not None:
                    self.pe_component_selector.update(pe_target, accepted=accepted)
                stats["evaluations"].append(
                    {
                        "source": "paradigm_shift",
                        "model": client_name(ref_model),
                        "score": score,
                        "accepted": accepted,
                        "island_expanded": island_opened,
                        "cell_index": cell_idx,
                        "archive_size": self.pool.size(),
                    }
                )

                logger.info(
                    f"[PE] Paradigm shift: score={score:.17g}, accepted={accepted}, "
                    f"cell={cell_idx}{', new island' if island_opened else ''}"
                )
            else:
                error_message = str(result.get("error", "unknown"))
                logger.info(f"[PE] Paradigm shift eval error: {error_message[:100]}")
                stats["evaluations"].append(
                    {
                        "source": "paradigm_shift",
                        "model": client_name(ref_model),
                        "error": error_message,
                        "archive_size": self.pool.size(),
                    }
                )
                paradigm_code = None  # Can't generate variants

        # Step 5: Generate variants (only if paradigm was valid)
        if paradigm_code and stats["paradigm_score"] is not None:
            variant_models = self.pe_config.variant_models
            if not variant_models:
                # Use lighter models from config, cycle through sampler models
                variant_models = [p.model for p in self.config.sampler_model_pairs[:3]]
                if not variant_models:
                    variant_models = [heavy_model]

            logger.info(
                f"[PE] Generating {self.pe_config.n_variants} variants using models: "
                f"{[client_name(model) for model in variant_models[:3]]}..."
            )

            variant_prompt = self._build_variant_prompt(
                paradigm_code,
                stats["paradigm_score"],
                target=pe_target,
            )

            async def generate_variant(model, idx: int):
                try:
                    response = await _acompletion_with_token_retry(
                        self.state,
                        model=model,
                        prompt=[{"role": "user", "content": variant_prompt}],
                        temperature=self.pe_config.temperature,
                        timeout=300,
                        label=f"PE/variant#{idx}",
                    )
                    return {
                        "idx": idx,
                        "content": response.text,
                        "cost": response.cost,
                        "model": client_name(model),
                    }
                except BudgetLimitReached:
                    return {"idx": idx, "error": "Budget exhausted", "model": client_name(model)}
                except Exception as e:
                    return {"idx": idx, "error": str(e), "model": client_name(model)}

            # Generate variants in parallel
            tasks = [
                generate_variant(variant_models[i % len(variant_models)], i) for i in range(self.pe_config.n_variants)
            ]
            variant_results = await asyncio.gather(*tasks)

            # Evaluate and insert variants
            for vr in variant_results:
                if "error" in vr:
                    logger.warning(
                        f"[PE] Variant {vr['idx']} generation failed ({vr.get('model', '?')}): {vr['error'][:100]}"
                    )
                    continue

                stats["total_cost"] += vr["cost"]

                if pe_target is not None:
                    variant_code = self.artifact_adapter.extract_candidate(
                        vr["content"],
                        parent_content=paradigm_code,
                        target=pe_target,
                    )
                else:
                    variant_code = self.artifact_adapter.extract_candidate(vr["content"])
                if not variant_code:
                    logger.warning(f"[PE] Variant {vr['idx']} code extraction failed ({vr.get('model', '?')})")
                    continue

                stats["variants_generated"] += 1

                if not can_start_pe_eval():
                    logger.info(f"[PE] Skipping variant {vr['idx']} evaluation (budget exhausted)")
                    stats["evaluations"].append(
                        {
                            "source": "variant",
                            "model": vr.get("model", "unknown"),
                            "error": "Budget exhausted",
                            "archive_size": self.pool.size(),
                        }
                    )
                    continue

                try:
                    pe_evals_started += 1
                    result = await self._evaluate(variant_code)
                except Exception as e:
                    logger.warning(f"[PE] Variant {vr['idx']} evaluation exception: {e}")
                    result = {"error": str(e)}

                if "error" not in result:
                    score, score_error = coerce_score(result)
                    if score_error is not None:
                        logger.warning(f"[PE] Variant {vr['idx']} invalid score: {score_error}")
                        result = {"error": score_error}
                    else:
                        result = dict(result)
                        result["score"] = score

                if "error" not in result:
                    program = self.artifact_adapter.make_program(
                        variant_code,
                        metadata={
                            "source": "punctuated_equilibrium",
                            "pe_type": "variant",
                        },
                    )
                    eval_result = EvaluationResult(
                        scores=result,
                        is_valid=True,
                    )

                    async with self.archive_lock:
                        accepted, cell_idx = self.pool.add(program, eval_result)
                        island_opened = False
                        if not accepted:
                            s_for_island = (
                                self.state.stagnation_depth(sal.tau)
                                if sal.enabled
                                else 0.0
                            )
                            island_opened, new_cell = self._try_island_expansion(
                                program, eval_result, stagnation=s_for_island
                            )
                            if island_opened:
                                accepted = True
                                cell_idx = new_cell

                    if accepted:
                        stats["variants_accepted"] += 1
                        stats["variant_cells"].append(cell_idx)
                    if pe_target is not None and self.pe_component_selector is not None:
                        self.pe_component_selector.update(pe_target, accepted=accepted)
                    stats["evaluations"].append(
                        {
                            "source": "variant",
                            "model": vr.get("model", "unknown"),
                            "score": score,
                            "accepted": accepted,
                            "island_expanded": island_opened,
                            "cell_index": cell_idx,
                            "archive_size": self.pool.size(),
                        }
                    )

                    logger.info(
                        f"[PE] Variant {vr['idx']}: score={score:.17g}, accepted={accepted}"
                        f"{', new island' if island_opened else ''}"
                    )
                else:
                    error_message = str(result.get("error", "unknown"))
                    logger.warning(f"[PE] Variant {vr['idx']} eval error: {error_message[:100]}")
                    stats["evaluations"].append(
                        {
                            "source": "variant",
                            "model": vr.get("model", "unknown"),
                            "error": error_message,
                            "archive_size": self.pool.size(),
                        }
                    )

        # Strategy log — let a light model summarise this PE's paradigm
        # shift so the next PE prompt can see what was already tried. The
        # summarisation call is gated on actually having a paradigm code
        # (no value in summarising garbage) and is skipped under heavy
        # budget pressure.
        if (
            paradigm_code is not None
            and not self.state.budget_exhausted
            and self.config.strategy_log.enabled
        ):
            try:
                summary, sum_cost = await self._summarize_strategy(paradigm_code)
            except BudgetLimitReached:
                summary, sum_cost = "", 0.0
            stats["total_cost"] += sum_cost
            best_after = (
                self.state.best_score_so_far
                if math.isfinite(self.state.best_score_so_far)
                else best_before
            )
            delta = best_after - best_before if math.isfinite(best_before) else 0.0
            paradigm_score_val = stats.get("paradigm_score")
            record_accepted = bool(stats.get("paradigm_accepted")) or stats.get("variants_accepted", 0) > 0
            self.state.append_strategy_record(
                StrategyRecord(
                    pe_event_id=self.state.pe_trigger_count,
                    stage=pe_stage,
                    summary=summary,
                    best_before=best_before if math.isfinite(best_before) else float("nan"),
                    paradigm_score=(
                        float(paradigm_score_val)
                        if paradigm_score_val is not None
                        else float("nan")
                    ),
                    delta_score=delta,
                    accepted=record_accepted,
                )
            )
            logger.info(
                f"[Strategy-Log] Appended record PE#{self.state.pe_trigger_count} "
                f"[{pe_stage}] Δ={delta:+.4g}, "
                f"{'accepted' if record_accepted else 'rejected'} "
                f"(history size {len(self.state.strategy_history)}/"
                f"{self.state.strategy_history.maxlen})"
            )
            stats["strategy_summary"] = summary
            stats["strategy_delta"] = delta

        logger.info(
            f"[PE] Complete: paradigm_accepted={stats['paradigm_accepted']}, "
            f"variants={stats['variants_accepted']}/{stats['variants_generated']}, "
            f"cost=${stats['total_cost']:.3f}"
        )

        return stats
