"""Shared pipeline state for producer-consumer coordination."""

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

import numpy as np

from ..clients.base import ClientInput, ClientSpec
from ..clients.lm import DEFAULT_TIMEOUT, _LMResolver
from ..config import BudgetConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy History — short post-mortems of past Punctuated Equilibrium events
# ---------------------------------------------------------------------------


@dataclass
class StrategyRecord:
    """A single algorithmic strategy that was already tried by Punctuated
    Equilibrium.

    Fields:
      pe_event_id: monotonically increasing PE trigger number.
      stage: which prompt template fired (``early`` / ``mid`` / ``late``).
      summary: one-sentence description of the approach, produced by a light
               summariser model from the paradigm-shift code.
      best_before: best score in the archive just before this PE.
      paradigm_score: the reference paradigm-shift candidate's score (NaN
               when the heavy candidate failed to evaluate).
      delta_score: best_after - best_before across the whole PE event
               (paradigm + variants).
      accepted: True when at least one PE candidate entered the archive.
    """

    pe_event_id: int
    stage: str
    summary: str
    best_before: float
    paradigm_score: float
    delta_score: float
    accepted: bool

    def format_line(self) -> str:
        """Render this record as one Markdown block for the heavy prompt.

        Supports both legacy single-line summaries and the structured
        four-label format produced by the new summariser. Multi-line
        summaries are indented under the PE header so the heavy model
        sees a clean nested layout.
        """
        outcome = "accepted" if self.accepted else "rejected"
        delta = f"{self.delta_score:+.4g}"
        score = (
            f"{self.paradigm_score:.4g}"
            if math.isfinite(self.paradigm_score)
            else "n/a"
        )
        body = (self.summary or "").strip()
        header = f"### PE #{self.pe_event_id} [{self.stage}] — Δ={delta}, score={score}, {outcome}"
        if not body:
            return header + "\n  (no summary)\n"
        # Indent multi-line summary by 2 spaces so it nests under the header
        # in the heavy prompt without breaking Markdown bullet logic.
        indented = "\n".join("  " + line if line else "" for line in body.splitlines())
        return header + "\n" + indented + "\n"


# ---------------------------------------------------------------------------
# Code Error Repair — bounded buffer of recently broken candidates
# ---------------------------------------------------------------------------


@dataclass
class ErrorRecord:
    """A candidate that failed evaluation (syntax / runtime / score error)."""

    code: str
    parent_score: float
    error_msg: str
    parent_cell: Optional[int]
    source: str  # "main_loop", "pe_paradigm", "pe_variant", "repair", ...


class BudgetLimitReached(RuntimeError):
    """Raised when a new operation cannot start due to exhausted budget."""


# ---------------------------------------------------------------------------
# Module-level utilities (previously static methods on PipelineState)
# ---------------------------------------------------------------------------


def coerce_finite_float(value: object, *, default: float) -> float:
    """Best-effort numeric coercion with a safe fallback."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _coerce_positive_limit(value: object) -> float | None:
    """Normalize a budget limit.  Returns *None* for unlimited."""
    if value is None:
        return None
    return coerce_finite_float(value, default=0.0)


# ---------------------------------------------------------------------------
# Score history
# ---------------------------------------------------------------------------


@dataclass
class ScoreHistoryEntry:
    """A single entry in the score history."""

    eval_number: int
    score: float
    best_score: float
    timestamp: float
    accepted: bool
    sampler: str
    archive_size: int
    cell_index: int | None = None  # Which cell this evaluation fell into
    is_punctuated_equilibrium: bool = False  # Whether from PE
    cumulative_cost: float = 0.0
    # Prompt-bank arm identifiers — captured so snapshot.json shows exactly
    # which (prompt_id, llm_temperature) produced this candidate, not just
    # the parent-selection sampler name. None for non-prompt-bank candidates
    # (init seeds, PE-paradigm, PE-variant, repair).
    mutation_prompt_id: str | None = None
    llm_temperature: float | None = None


# ---------------------------------------------------------------------------
# BudgetTracker – budget limits, cost accounting, eval reservation
# ---------------------------------------------------------------------------


@dataclass
class BudgetTracker:
    """Tracks budget consumption and enforces limits.

    Owns every counter that participates in the budget-exhaustion decision
    (dollars, evaluations, seconds) plus the async lock that protects
    atomic reserve-if-budget-permits operations.
    """

    budget: BudgetConfig
    start_time: float = field(default_factory=time.time)

    # Cost tracking
    total_cost: float = 0.0

    # Eval / client counters
    eval_count: int = 0
    eval_in_flight: int = 0
    client_in_flight: int = 0

    # Async lock – protects counter mutations that must be atomic with
    # budget-exhaustion checks (eval reservation, client gating).
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # Client cost EMA – used to decide when the budget is tight enough to serialize requests.
    _client_cost_ema: float = 0.0
    _client_cost_samples: int = 0

    # ---- budget queries ----

    @property
    def exhausted(self) -> bool:
        dollars_limit = _coerce_positive_limit(self.budget.dollars)
        if dollars_limit is not None:
            if dollars_limit <= 0.0:
                return True
            total_cost = coerce_finite_float(self.total_cost, default=float("inf"))
            if total_cost >= dollars_limit:
                return True

        if self.budget.evaluations is not None:
            eval_limit = int(coerce_finite_float(self.budget.evaluations, default=0.0))
            if eval_limit <= 0:
                return True
            eval_used = int(
                coerce_finite_float(
                    self.eval_count + self.eval_in_flight,
                    default=float("inf"),
                )
            )
            if eval_used >= eval_limit:
                return True

        seconds_limit = _coerce_positive_limit(self.budget.seconds)
        if seconds_limit is not None:
            if seconds_limit <= 0.0:
                return True
            if self.elapsed_seconds >= seconds_limit:
                return True

        return False

    @property
    def elapsed_seconds(self) -> float:
        start = coerce_finite_float(self.start_time, default=float("-inf"))
        elapsed = time.time() - start
        if not math.isfinite(elapsed):
            return float("inf")
        return max(0.0, elapsed)

    @property
    def progress(self) -> float:
        """Return progress as fraction (0-1) based on primary budget type."""
        dollars_limit = _coerce_positive_limit(self.budget.dollars)
        if dollars_limit is not None:
            if dollars_limit <= 0.0:
                return 1.0
            total_cost = coerce_finite_float(self.total_cost, default=float("inf"))
            return max(0.0, min(1.0, total_cost / dollars_limit))

        if self.budget.evaluations is not None:
            eval_limit = int(coerce_finite_float(self.budget.evaluations, default=0.0))
            if eval_limit <= 0:
                return 1.0
            eval_used = int(
                coerce_finite_float(
                    self.eval_count + self.eval_in_flight,
                    default=float("inf"),
                )
            )
            return max(0.0, min(1.0, eval_used / eval_limit))

        seconds_limit = _coerce_positive_limit(self.budget.seconds)
        if seconds_limit is not None:
            if seconds_limit <= 0.0:
                return 1.0
            return max(0.0, min(1.0, self.elapsed_seconds / seconds_limit))

        return 0.0

    # ---- cost recording ----

    def add_cost(self, cost: float) -> None:
        normalized = coerce_finite_float(cost, default=0.0)
        if normalized < 0.0:
            logger.warning("[Budget] Ignoring negative cost update: %r", cost)
            return
        self.total_cost += normalized

    def record_client_cost(self, cost: object) -> None:
        """Record a generation call cost (updates total_cost and the rolling EMA)."""
        normalized = coerce_finite_float(cost, default=0.0)
        if normalized < 0.0:
            return
        self.total_cost += normalized
        if self._client_cost_samples == 0:
            self._client_cost_ema = normalized
        else:
            self._client_cost_ema = (0.8 * self._client_cost_ema) + (0.2 * normalized)
        self._client_cost_samples += 1

    # ---- eval reservation ----

    async def try_start_evaluation(self) -> bool:
        """Reserve one evaluation slot if budget permits."""
        async with self._lock:
            if self.exhausted:
                return False
            self.eval_in_flight += 1
            return True

    async def finish_evaluation(self) -> None:
        """Release one reserved evaluation slot."""
        async with self._lock:
            if self.eval_in_flight > 0:
                self.eval_in_flight -= 1

    # ---- serial-mode decision helpers (used by ClientGate) ----

    def remaining_dollars(self) -> float | None:
        dollars_limit = _coerce_positive_limit(self.budget.dollars)
        if dollars_limit is None:
            return None
        return dollars_limit - coerce_finite_float(self.total_cost, default=float("inf"))

    def _client_serial_threshold(self, dollars_limit: float) -> float:
        if self._client_cost_samples > 0:
            ema = self._client_cost_ema
        else:
            ema = max(0.01 * dollars_limit, 0.05)
        return max(3.0 * ema, 0.03 * dollars_limit, 0.05)

    def should_use_serial_mode(self) -> bool:
        dollars_limit = _coerce_positive_limit(self.budget.dollars)
        if dollars_limit is not None:
            remaining = self.remaining_dollars()
            if remaining is not None and remaining <= self._client_serial_threshold(dollars_limit):
                return True

        if self.budget.evaluations is not None:
            eval_limit = int(coerce_finite_float(self.budget.evaluations, default=0.0))
            eval_remaining = eval_limit - (self.eval_count + self.eval_in_flight)
            if eval_remaining <= 2:
                return True

        seconds_limit = _coerce_positive_limit(self.budget.seconds)
        if seconds_limit is not None:
            time_remaining = seconds_limit - self.elapsed_seconds
            if time_remaining <= 15.0:
                return True

        return False


# ---------------------------------------------------------------------------
# ClientGate – concurrency control and budget enforcement for client calls
# ---------------------------------------------------------------------------


class ClientGate:
    """Concurrency and budget gate for text-generation client calls.

    Wraps every ``client.acompletion`` call with:
    * semaphore-based concurrency limiting
    * budget-exhaustion check (raises ``BudgetLimitReached``)
    * automatic serial mode when budget is tight
    * cost extraction and accounting
    """

    def __init__(self, tracker: BudgetTracker, resolver: _LMResolver) -> None:
        self._tracker = tracker
        self._resolver = resolver
        self._semaphore = asyncio.Semaphore(1)
        self._serial_lock = asyncio.Lock()

    def configure_concurrency(self, max_in_flight: int) -> None:
        """Set global max concurrent generation requests for this run."""
        limit = int(coerce_finite_float(max_in_flight, default=1.0))
        if limit <= 0:
            limit = 1
        self._semaphore = asyncio.Semaphore(limit)

    async def acompletion(
        self,
        client_spec: ClientSpec,
        *,
        prompt: ClientInput,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **extras: Any,
    ) -> Any:
        """Budget/concurrency gate around ``client.acompletion``."""
        tracker = self._tracker
        client = self._resolver.resolve(client_spec)
        async with self._semaphore:
            async with tracker._lock:
                if tracker.exhausted:
                    raise BudgetLimitReached("Budget exhausted before client call")
                tracker.client_in_flight += 1
                use_serial = tracker.should_use_serial_mode()

            try:
                if use_serial:
                    async with self._serial_lock:
                        response = await client.acompletion(
                            prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=timeout,
                            **extras,
                        )
                else:
                    response = await client.acompletion(
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        **extras,
                    )
            finally:
                async with tracker._lock:
                    tracker.client_in_flight = max(0, tracker.client_in_flight - 1)

            async with tracker._lock:
                tracker.record_client_cost(getattr(response, "cost", 0.0))

            return response


# ---------------------------------------------------------------------------
# PipelineState – thin coordinator that composes the above
# ---------------------------------------------------------------------------


class PipelineState:
    """Shared pipeline state for producer-consumer coordination.

    Composes :class:`BudgetTracker` (budget/cost enforcement) and
    :class:`ClientGate` (generation concurrency control) with domain-specific
    pipeline state (eval metrics, meta-advice, score history).
    """

    def __init__(self, budget: BudgetConfig, start_time: float | None = None):
        st = start_time if start_time is not None else time.time()
        self.budget_tracker = BudgetTracker(budget, start_time=st)
        self.client_resolver = _LMResolver(timeout=DEFAULT_TIMEOUT)
        self.client_gate = ClientGate(self.budget_tracker, self.client_resolver)

        # Eval outcome counters (not budget-relevant, so kept here)
        self.accept_count: int = 0
        self.error_count: int = 0

        # Meta-advice tracking
        self.current_meta_advice: str = ""
        self.previous_meta_advice: str = ""
        self.meta_advice_eval_count: int = 0

        # Period metrics for meta-advice generation
        self.period_errors: int = 0
        self.period_acceptances: int = 0
        self.period_rejections: int = 0
        self.period_error_messages: set = set()
        self.all_error_counts: dict = {}

        # Score history tracking
        self.score_history: list = []
        self.best_score_so_far: float = float("-inf")

        # Punctuated Equilibrium tracking
        self.last_pe_eval_count: int = 0
        self.pe_trigger_count: int = 0

        # SAL tracking (Stagnation-Adaptive Loop)
        self.sal_sigma_0: float = 0.0
        """Baseline std of accepted scores (set after init phase)."""

        self.sal_init_finished: bool = False
        """True once the init phase has populated score_history."""

        self.hard_pe_count: int = 0
        """Count of hard-PE events fired so far (Cơ chế E cooldown)."""

        self.consecutive_pe_no_best: int = 0
        """Consecutive PE triggers that produced no NEW BEST."""

        self.eval_count_at_last_best: int = 0
        """eval_count snapshot at the last strict NEW BEST. O(1) cache for s(t)."""

        self.best_score_at_last_pe: float = float("-inf")
        """best_score_so_far at the moment of the most recent PE trigger.

        Used by the runner to decide whether a PE produced a NEW BEST (so we
        can update consecutive_pe_no_best and gate Hard-PE)."""

        # ------------------------------------------------------------------
        # PPS — Posterior-Plateau Stagnation
        # Bounded sliding window of (eval_count, total_cost) tuples captured
        # at the moment of each strict NEW BEST. Used by stagnation_depth()
        # to estimate the empirical NEW BEST hazard rate per unit cost.
        # ------------------------------------------------------------------
        self.new_best_history: Deque[tuple[int, float]] = deque(maxlen=32)

        # ------------------------------------------------------------------
        # Strategy History — one record per Punctuated Equilibrium event.
        # Light model produces a 1-sentence summary of each paradigm-shift
        # solution; the next PE's heavy prompt receives the formatted log so
        # the model can explicitly avoid re-trying approaches that didn't
        # work.
        # ------------------------------------------------------------------
        self.strategy_history: Deque[StrategyRecord] = deque(maxlen=12)

        # ------------------------------------------------------------------
        # Code Error Repair — bounded buffer of recently failed candidates.
        # Producer pops one at a time (rank-by-parent-score) and asks a
        # light model for a one-shot fix. No retries beyond the single
        # repair call.
        # ------------------------------------------------------------------
        self.error_buffer: Deque[ErrorRecord] = deque(maxlen=64)
        self.repair_attempt_count: int = 0
        self.repair_success_count: int = 0
        self.last_repair_at_eval: int = -10**9

        # ------------------------------------------------------------------
        # Adaptive Island Expansion bookkeeping.
        # Total number of times a PE candidate has opened a new cell at
        # its own behaviour vector (per-run budget cap).
        # ------------------------------------------------------------------
        self.island_expansion_count: int = 0

    # ------------------------------------------------------------------
    # Delegation: BudgetTracker
    # ------------------------------------------------------------------

    @property
    def budget(self) -> BudgetConfig:
        return self.budget_tracker.budget

    @property
    def start_time(self) -> float:
        return self.budget_tracker.start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        self.budget_tracker.start_time = value

    @property
    def total_cost(self) -> float:
        return self.budget_tracker.total_cost

    @total_cost.setter
    def total_cost(self, value: float) -> None:
        self.budget_tracker.total_cost = value

    @property
    def eval_count(self) -> int:
        return self.budget_tracker.eval_count

    @eval_count.setter
    def eval_count(self, value: int) -> None:
        self.budget_tracker.eval_count = value

    @property
    def eval_in_flight(self) -> int:
        return self.budget_tracker.eval_in_flight

    @property
    def client_in_flight(self) -> int:
        return self.budget_tracker.client_in_flight

    @property
    def budget_exhausted(self) -> bool:
        if self.budget_tracker.exhausted:
            return True
        target = self.budget_tracker.budget.target_score
        if target is not None and self.best_score_so_far >= target:
            return True
        return False

    @property
    def elapsed_seconds(self) -> float:
        return self.budget_tracker.elapsed_seconds

    @property
    def budget_progress(self) -> float:
        return self.budget_tracker.progress

    def add_cost(self, cost: float) -> None:
        self.budget_tracker.add_cost(cost)

    async def try_start_evaluation(self) -> bool:
        return await self.budget_tracker.try_start_evaluation()

    async def finish_evaluation(self) -> None:
        await self.budget_tracker.finish_evaluation()

    # ------------------------------------------------------------------
    # Delegation: ClientGate
    # ------------------------------------------------------------------

    def configure_client_defaults(
        self,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.client_resolver.configure(
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def configure_client_concurrency(self, max_in_flight: int) -> None:
        self.client_gate.configure_concurrency(max_in_flight)

    async def acompletion(
        self,
        client_spec: ClientSpec,
        *,
        prompt: ClientInput,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **extras: Any,
    ) -> Any:
        return await self.client_gate.acompletion(
            client_spec,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **extras,
        )

    async def close_clients(self) -> None:
        await self.client_resolver.close()

    # ------------------------------------------------------------------
    # Domain: eval outcome recording
    # ------------------------------------------------------------------

    def record_accept(self) -> None:
        self.budget_tracker.eval_count += 1
        self.accept_count += 1
        self.period_acceptances += 1

    def record_reject(self) -> None:
        self.budget_tracker.eval_count += 1
        self.period_rejections += 1

    def record_error(self, error: str) -> None:
        self.budget_tracker.eval_count += 1
        self.error_count += 1
        self.period_errors += 1
        short_error = error[:100]
        if len(self.period_error_messages) < 10:
            self.period_error_messages.add(short_error)
        self.all_error_counts[short_error] = self.all_error_counts.get(short_error, 0) + 1

    # ------------------------------------------------------------------
    # Domain: meta-advice
    # ------------------------------------------------------------------

    def should_generate_meta_advice(self, interval: int) -> bool:
        if interval <= 0:
            return False
        return (
            self.eval_count > 0 and self.eval_count % interval == 0 and self.eval_count != self.meta_advice_eval_count
        )

    def reset_period_metrics(self) -> dict:
        top_errors = sorted(self.all_error_counts.items(), key=lambda x: -x[1])[:10]
        metrics = {
            "errors": self.period_errors,
            "acceptances": self.period_acceptances,
            "rejections": self.period_rejections,
            "error_messages": set(self.period_error_messages),
            "top_errors": top_errors,
        }
        self.period_errors = 0
        self.period_acceptances = 0
        self.period_rejections = 0
        self.period_error_messages.clear()
        self.meta_advice_eval_count = self.eval_count
        return metrics

    # ------------------------------------------------------------------
    # Domain: score history
    # ------------------------------------------------------------------

    def record_score(
        self,
        score: float,
        accepted: bool,
        sampler: str,
        archive_size: int,
        cell_index: int | None = None,
        is_punctuated_equilibrium: bool = False,
        mutation_prompt_id: str | None = None,
        llm_temperature: float | None = None,
    ) -> None:
        """Record a score in the history."""
        if score > self.best_score_so_far:
            self.best_score_so_far = score

        entry = ScoreHistoryEntry(
            eval_number=self.eval_count,
            score=score,
            best_score=self.best_score_so_far,
            timestamp=time.time() - self.start_time,
            accepted=accepted,
            sampler=sampler,
            archive_size=archive_size,
            cell_index=cell_index,
            is_punctuated_equilibrium=is_punctuated_equilibrium,
            cumulative_cost=self.total_cost,
            mutation_prompt_id=mutation_prompt_id,
            llm_temperature=llm_temperature,
        )
        self.score_history.append(entry)

    def get_score_history_list(self) -> list[float]:
        """Get just the best scores over time for the result."""
        return [entry.best_score for entry in self.score_history]

    # ------------------------------------------------------------------
    # Domain: SAL (Stagnation-Adaptive Loop)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # PPS — Posterior-Plateau Stagnation
    # ------------------------------------------------------------------

    def record_new_best(self) -> None:
        """Mark the current evaluation as a strict NEW BEST.

        Snapshots ``(eval_count, total_cost)`` into the bounded history that
        powers the empirical NEW BEST hazard rate used by PPS. Also resets
        the O(1) plateau cache. Idempotent in the sense that two calls in a
        single eval just produce a duplicate entry; callers should gate on a
        score-vs-best comparison.
        """
        self.eval_count_at_last_best = self.eval_count
        self.new_best_history.append(
            (
                self.budget_tracker.eval_count,
                coerce_finite_float(self.budget_tracker.total_cost, default=0.0),
            )
        )

    def _budget_progress_components(self) -> tuple[float, float, float]:
        """Return ``(b, B_used, B_total)`` ∈ [0,1] × cost × cost.

        b is the dominant budget-consumed fraction across all defined caps
        (max over dollars / evals / seconds), used as the PPS confidence
        weight. B_used / B_total are reported in the *same* unit as the
        dominant cap so the hazard rate is interpretable. When no caps are
        defined we fall back to a synthetic "evals so far / max(evals,1)"
        signal so PPS still returns something meaningful.
        """
        budget = self.budget_tracker.budget

        dollars_limit = _coerce_positive_limit(budget.dollars)
        seconds_limit = _coerce_positive_limit(budget.seconds)
        evals_limit_raw = budget.evaluations

        candidates: list[tuple[str, float, float, float]] = []

        if dollars_limit is not None and dollars_limit > 0.0:
            used = coerce_finite_float(self.budget_tracker.total_cost, default=0.0)
            candidates.append(("dollars", used / dollars_limit, used, dollars_limit))

        if evals_limit_raw is not None:
            evals_limit = float(coerce_finite_float(evals_limit_raw, default=0.0))
            if evals_limit > 0:
                used = float(self.budget_tracker.eval_count + self.budget_tracker.eval_in_flight)
                candidates.append(("evals", used / evals_limit, used, evals_limit))

        if seconds_limit is not None and seconds_limit > 0.0:
            used = self.budget_tracker.elapsed_seconds
            candidates.append(("seconds", used / seconds_limit, used, seconds_limit))

        if not candidates:
            used = float(self.budget_tracker.eval_count)
            total = max(used + 1.0, 1.0)
            return (0.0, used, total)

        candidates.sort(key=lambda c: c[1], reverse=True)
        _, ratio, used, total = candidates[0]
        return (max(0.0, min(1.0, ratio)), used, total)

    def stagnation_depth(self, tau: int) -> float:
        """Posterior-Plateau Stagnation s(t) ∈ [0, 1].

        Replaces the legacy ``max(plateau, budget_ratios)`` with a survival-
        style estimate of "no further NEW BEST in remaining budget":

            p(t) = min(1, n_since_best / tau)              # plateau term
            b(t) ∈ [0,1]                                    # dominant budget consumed
            B_rem = (1 - b(t)) · B_total                    # remaining budget (raw)
            λ̂(t) = (k_W + 1) / (B_W + ε)                   # Laplace-smoothed hazard
                                                            #   k_W: NEW BEST in window
                                                            #   B_W: budget consumed since
                                                            #        first window entry
            posterior_stuck = p(t) · exp(-λ̂(t) · B_rem)    # P(no progress | history)
            α(t)  = b(t)^2                                  # confidence in hazard estimate
            s(t)  = (1 - α) · p(t) + α · posterior_stuck

        Properties:
          * Early run (α≈0): s(t) ≈ p(t) — legacy plateau, no noisy hazard.
          * Late run (α≈1) with no NEW BEST: B_rem · λ̂ → 0, so
            posterior_stuck → p(t) and s(t) → p(t) — but p(t) is already
            saturating, so SAL mechanisms fire as before.
          * Late run with frequent NEW BEST: λ̂ · B_rem is large,
            posterior_stuck → 0 even if p(t) > 0 — we're improving so don't
            panic-trigger PE / Hard-PE.
          * Late run with sparse NEW BEST and large B_rem: posterior_stuck
            stays close to p(t) so plateau pressure dominates.

        The formulation is a Poisson-hazard survival estimate, which is
        well-grounded statistically and treats stagnation as a *posterior
        belief about exhaustion of the archive* conditional on the
        improvement-per-budget trajectory we have observed.

        Args:
            tau: plateau length at which p(t) saturates to 1.0.
        """
        # --- plateau term p(t) ------------------------------------------
        if tau > 0:
            n_since_best = max(0, self.eval_count - self.eval_count_at_last_best)
            p = min(1.0, n_since_best / float(tau))
        else:
            p = 0.0

        # --- budget components ------------------------------------------
        b, B_used, B_total = self._budget_progress_components()
        B_rem = max(0.0, B_total - B_used)

        # --- empirical NEW BEST hazard λ̂(t) -----------------------------
        history = self.new_best_history
        if history:
            # Window covers everything since the oldest tracked NEW BEST.
            # Laplace smoothing: (k+1)/(B_W+ε) keeps λ̂ > 0 even if the
            # window saw no improvements yet (k=0 → λ̂ = 1/B_W).
            window_start_used = history[0][1]
            B_window = max(0.0, B_used - window_start_used)
            k_window = len(history)
        else:
            B_window = B_used
            k_window = 0

        eps = max(1e-9, 1e-3 * B_total)  # numerically safe floor
        lam_hat = (k_window + 1.0) / (B_window + eps)

        # Survival probability of "no NEW BEST in remaining budget" under a
        # Poisson process with rate λ̂. Multiplied by plateau term so that
        # if we're not even on a plateau, posterior_stuck contributes
        # nothing.
        survival = math.exp(-lam_hat * B_rem)
        posterior_stuck = p * survival

        # --- confidence-weighted blend ----------------------------------
        alpha = b * b
        s = (1.0 - alpha) * p + alpha * posterior_stuck

        # Safety floor: PPS should never *under-report* a strict end-of-run
        # plateau (b≈1, p≈1) regardless of hazard noise. Clamp so very-late-
        # run stagnation is always ≥ plateau term.
        if b >= 0.95 and p >= 0.95:
            s = max(s, p)

        return max(0.0, min(1.0, s))

    def evals_since_best(self) -> int:
        """Count evals since the last strict NEW BEST (O(1) via cache)."""
        return max(0, self.eval_count - self.eval_count_at_last_best)

    def recent_score_std(self, window: int) -> float:
        """Std of accepted scores in the last `window` history entries."""
        if window <= 1 or not self.score_history:
            return 0.0
        recent = self.score_history[-window:]
        scores = [e.score for e in recent if e.accepted]
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        return var ** 0.5

    def finalize_init_baseline(self, sigma_window: int) -> None:
        """Capture σ₀ from init-phase score history.

        Called once at the boundary between init and main-loop.
        """
        if self.sal_init_finished:
            return
        self.sal_sigma_0 = self.recent_score_std(window=max(sigma_window, len(self.score_history)))
        self.sal_init_finished = True

    # ------------------------------------------------------------------
    # Strategy History helpers
    # ------------------------------------------------------------------

    def append_strategy_record(self, record: StrategyRecord) -> None:
        """Append a post-mortem of a Punctuated Equilibrium event.

        Records are summarised in the next heavy prompt so the model can
        explicitly avoid retracing approaches it has already tried.
        """
        self.strategy_history.append(record)

    def format_strategy_log(self, max_entries: int = 8) -> str:
        """Render the strategy history as a ``## Strategy Log`` section.

        Returns an empty string when the history is empty so callers can
        splice it unconditionally and get a clean prompt in early runs.
        """
        if not self.strategy_history:
            return ""
        recent = list(self.strategy_history)[-max_entries:]
        lines = ["", "## Strategy Log (already tried in this run)"]
        lines.extend(r.format_line() for r in recent)
        lines.append(
            "\nDo NOT propose an approach whose summary appears above with "
            "Δ ≤ 0. Aim for something structurally different from every "
            "rejected entry."
        )
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Code Error Repair helpers
    # ------------------------------------------------------------------

    def fire_repair_if_due(self, code_repair_cfg) -> Optional[ErrorRecord]:
        """Atomically (caller-protected) check whether we should fire a repair.

        Returns the chosen :class:`ErrorRecord` or ``None`` when no repair is
        due. The caller MUST already hold whatever lock protects pipeline
        state (typically ``archive_lock``) so two producers cannot fire the
        same repair concurrently. The state mutation (counter increment,
        ``last_repair_at_eval`` advance) happens inside this method.
        """
        if code_repair_cfg is None or not getattr(code_repair_cfg, "enabled", False):
            return None
        if not self.error_buffer:
            return None
        if self.repair_attempt_count >= code_repair_cfg.max_per_run:
            return None
        every_n = max(1, int(code_repair_cfg.repair_every_n))
        if (self.eval_count - self.last_repair_at_eval) < every_n:
            return None
        record = self.sample_error_for_repair(beta=float(code_repair_cfg.beta))
        if record is None:
            return None
        self.last_repair_at_eval = self.eval_count
        self.repair_attempt_count += 1
        return record

    def push_error_record(self, record: ErrorRecord) -> None:
        """Append a failed candidate to the bounded repair buffer.

        Deduplicated by the (truncated) code key so an avalanche of
        identical errors does not crowd out informative variety.
        """
        signature = record.code[:200]
        for existing in self.error_buffer:
            if existing.code[:200] == signature:
                return
        self.error_buffer.append(record)

    def sample_error_for_repair(self, beta: float = 1.5) -> Optional[ErrorRecord]:
        """Pop one error record using Zipfian rank-by-parent-score.

        Mirrors AdaptiveRankSampler's selection rule but uses ``parent_score``
        (the score of the broken candidate's PARENT) as the ranking signal.
        A higher β favours errors whose parent was strong, on the intuition
        that fixing a near-elite is more valuable than fixing noise. The
        returned record is removed from the buffer — repair is one-shot.
        """
        if not self.error_buffer:
            return None
        entries = list(self.error_buffer)
        # Sort by parent_score descending, ties broken by insertion order.
        order = sorted(
            range(len(entries)),
            key=lambda i: (
                -coerce_finite_float(entries[i].parent_score, default=float("-inf")),
                i,
            ),
        )
        ranks = np.arange(1, len(order) + 1, dtype=float)
        weights = ranks ** (-max(0.0, float(beta)))
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            probs = np.full(len(order), 1.0 / len(order))
        else:
            probs = weights / total
        rank_idx = int(np.random.choice(len(order), p=probs))
        chosen_idx = order[rank_idx]
        chosen = entries[chosen_idx]
        # Remove (one-shot — no retries).
        # deque does not support O(1) deletion by index, so rebuild it.
        new_buf: Deque[ErrorRecord] = deque(maxlen=self.error_buffer.maxlen)
        for i, e in enumerate(entries):
            if i != chosen_idx:
                new_buf.append(e)
        self.error_buffer = new_buf
        return chosen

    # ------------------------------------------------------------------
    # Adaptive-CVT bookkeeping
    # ------------------------------------------------------------------

    def record_behavior_vector(self, vec: np.ndarray) -> None:
        """Append a normalised behaviour vector to the recent window.

        Stored vectors must be in the same space as ``pool._centroids``
        (i.e. already passed through the pool's bound-aware normaliser).
        """
        try:
            arr = np.asarray(vec, dtype=float)
        except Exception:
            return
        if arr.ndim != 1 or arr.size == 0:
            return
        if not np.all(np.isfinite(arr)):
            return
        self.recent_behaviors.append(arr.copy())
