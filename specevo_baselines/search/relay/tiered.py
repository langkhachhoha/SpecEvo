"""Shared machinery for every cheap/strong controller in this package.

``TieredController`` adds three things to SpecEvo Baselines's ``DiscoveryController``:

1. **Two LLM pools.**  ``config.llm.models`` is the *cheap* model and
   ``config.llm.guide_models`` is the *strong* model.  Which one an iteration
   uses is carried in a :mod:`contextvars` variable, so the choice is
   per-asyncio-task and therefore safe under the parallel loop.

2. **A parallel block runner.**  ``run_block()`` puts up to
   ``max_parallel_iterations`` iterations of the *same* tier in flight at once
   and returns the programs they produced, which is the unit both RelayEvolve
   (a Grow/Deepen block) and the routing baselines (a batch of generations)
   are built out of.

3. **Budget and progress accounting.**  Spend is read back from
   ``specevo_baselines.llm.cost_tracker`` (OpenRouter's own ``usage.cost``), split
   per tier at the phase boundary, and appended to ``relay_progress.jsonl``
   so a cost-vs-score curve can be plotted without re-parsing checkpoints.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from specevo_baselines.llm.base import LLMResponse
from specevo_baselines.llm.cost_tracker import budget_usd, get_totals_snapshot
from specevo_baselines.llm.llm_pool import LLMPool
from specevo_baselines.search.base_database import Program
from specevo_baselines.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from specevo_baselines.search.utils.discovery_utils import SerializableResult
from specevo_baselines.search.utils.eval_log import EvalCodeLog
from specevo_baselines.utils.metrics import get_score

logger = logging.getLogger(__name__)

CHEAP = "cheap"
STRONG = "strong"

_TIER: contextvars.ContextVar[str] = contextvars.ContextVar("relay_model_tier", default=CHEAP)


class TieredController(DiscoveryController):
    """A discovery controller that can issue calls to two model tiers."""

    #: Overridden by subclasses; recorded in ``relay_summary.json``.
    method_name = "tiered"

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        self.cheap_llms: LLMPool = self.llms
        # LLMConfig.__post_init__ copies `models` into `guide_models` when the
        # latter is unset, so a single-model config degrades to "both tiers are
        # the same model" instead of crashing.
        self.strong_llms: LLMPool = self.guide_llms

        self.cheap_model_name = _pool_name(self.cheap_llms)
        self.strong_model_name = _pool_name(self.strong_llms)

        opts = self.config.search.database
        self.max_parallel = max(1, int(self.config.max_parallel_iterations))
        # Attempts per generation. Defaults to 1: retries inside an iteration
        # hide model calls from the generation budget and serialise the worker,
        # so a failed generation is counted and dropped instead.
        self.retry_times = max(1, int(getattr(opts, "retry_times", 1)))
        self.total_budget_usd: Optional[float] = budget_usd()
        self.strong_reserve = float(getattr(opts, "strong_reserve", 0.85))

        self.phase = CHEAP
        self.handoff_iteration: Optional[int] = None
        self.requested_iterations: Optional[int] = None
        self._start_time = time.time()
        self._tier_calls: Dict[str, int] = {CHEAP: 0, STRONG: 0}
        self._phase_cost_marks: List[Dict[str, Any]] = []
        self.best_program: Optional[Program] = None
        self.best_score: float = float("-inf")
        self.iterations_used = 0

        self._progress_path = (
            os.path.join(self.output_dir, "relay_progress.jsonl") if self.output_dir else None
        )
        self._summary_path = (
            os.path.join(self.output_dir, "relay_summary.json") if self.output_dir else None
        )

        # Optional: keep the source code of every generation, valid or not.
        # relay_progress.jsonl says what each generation scored; this says what
        # it actually wrote, including the ones that never scored at all.
        self.eval_log: Optional[EvalCodeLog] = None
        if self.output_dir and bool(getattr(opts, "save_eval_code", False)):
            self.eval_log = EvalCodeLog(
                os.path.join(self.output_dir, "eval_code_log.jsonl"),
                meta={
                    "method": self.method_name,
                    "cheap_model": self.cheap_model_name,
                    "strong_model": self.strong_model_name,
                    "output_dir": self.output_dir,
                },
            )
            logger.info("Saving every generation's code to %s", self.eval_log.path)

        if self.cheap_model_name == self.strong_model_name:
            logger.warning(
                "Cheap and strong tiers resolve to the same model (%s) — pass "
                "--cheap-model / --strong-model to separate them.",
                self.cheap_model_name,
            )
        logger.info(
            "%s: cheap=%s, strong=%s, workers=%d, budget=%s",
            self.method_name,
            self.cheap_model_name,
            self.strong_model_name,
            self.max_parallel,
            f"${self.total_budget_usd:.2f}" if self.total_budget_usd else "unbounded",
        )
        if self.retry_times == 1:
            logger.info("Retries disabled: a failed generation spends its slot and is dropped.")

    # ------------------------------------------------------------------
    # Tier routing
    # ------------------------------------------------------------------

    async def _call_llm(self, system_message: str, user_message: str, **kwargs) -> LLMResponse:
        tier = _TIER.get()
        pool = self.strong_llms if tier == STRONG else self.cheap_llms
        self._tier_calls[tier] = self._tier_calls.get(tier, 0) + 1
        return await pool.generate(
            system_message, [{"role": "user", "content": user_message}], **kwargs
        )

    async def _run_iteration_tier(
        self, iteration: int, tier: str, retry_times: int = 3
    ) -> SerializableResult:
        """Run one full generate-evaluate iteration pinned to ``tier``."""
        _TIER.set(tier)
        result = await self._run_iteration(iteration, retry_times=retry_times)
        if result is not None and result.child_program_dict is not None:
            meta = result.child_program_dict.setdefault("metadata", {}) or {}
            meta["model_tier"] = tier
            meta["relay_phase"] = self.phase
            result.child_program_dict["metadata"] = meta
        self._log_eval_code(iteration, tier, result)
        return result

    def _log_eval_code(
        self, iteration: int, tier: str, result: Optional[SerializableResult]
    ) -> None:
        """Record this generation's code — however far it got."""
        if self.eval_log is None:
            return
        child = (result.child_program_dict if result is not None else None) or {}
        metrics = child.get("metrics") or {}
        if result is None:
            status = "no_result"
        elif not result.error:
            status = "ok"
        elif child:
            status = "evaluation_failed"
        else:
            status = "no_program"
        self.eval_log.record(
            iteration=iteration,
            method=self.method_name,
            tier=tier,
            model=self.strong_model_name if tier == STRONG else self.cheap_model_name,
            phase=self.phase,
            status=status,
            score=get_score(metrics) if metrics else None,
            metrics=metrics,
            error=result.error if result is not None else "iteration returned nothing",
            program_id=child.get("id"),
            parent_id=result.parent_id if result is not None else None,
            language=child.get("language"),
            attempts_used=result.attempts_used if result is not None else None,
            eval_time_s=round(result.eval_time, 3) if result is not None else None,
            llm_time_s=round(result.llm_generation_time, 3) if result is not None else None,
            code=child.get("solution"),
            llm_response=result.llm_response if result is not None else None,
        )

    # ------------------------------------------------------------------
    # Parallel block execution
    # ------------------------------------------------------------------

    async def run_block(
        self,
        start_iteration: int,
        count: int,
        tier: str,
        retry_times: int = 3,
        checkpoint_callback: Optional[Any] = None,
        max_parallel: Optional[int] = None,
        tier_of: Optional[Any] = None,
    ) -> List[Program]:
        """Run ``count`` iterations concurrently and return the new programs.

        ``tier_of(iteration)`` overrides ``tier`` per iteration — that is how
        the Random and Bandit baselines mix tiers inside one batch while still
        keeping every worker busy.
        """
        parallel = max(1, int(max_parallel or self.max_parallel))
        sem = asyncio.Semaphore(parallel)
        produced: List[Program] = []

        async def one(iteration: int) -> None:
            async with sem:
                if self.shutdown_event.is_set():
                    return
                iteration_tier = (tier_of(iteration) if tier_of else None) or tier
                try:
                    result = await self._run_iteration_tier(
                        iteration, iteration_tier, retry_times=retry_times
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Iteration %d failed: %s", iteration, exc)
                    return

            self.iterations_used += 1
            if result is None:
                return
            if result.error:
                logger.warning("Iteration %d failed: %s", iteration, result.error)
                self._log_progress(iteration, iteration_tier, None, error=result.error)
                return

            before = self.database.best_program_id
            self._process_iteration_result(result, iteration, checkpoint_callback)
            program = self.database.programs.get(result.child_program_dict["id"])
            if program is not None:
                produced.append(program)
                self._track_best(program)
                self._log_progress(
                    iteration,
                    iteration_tier,
                    program,
                    improved=self.database.best_program_id != before,
                )

        await asyncio.gather(*(one(i) for i in range(start_iteration, start_iteration + count)))
        return produced

    # ------------------------------------------------------------------
    # Budget / progress accounting
    # ------------------------------------------------------------------

    @staticmethod
    def spent_usd() -> float:
        return float(get_totals_snapshot().get("total_cost_usd") or 0.0)

    def budget_remaining(self) -> Optional[float]:
        if self.total_budget_usd is None:
            return None
        return max(0.0, self.total_budget_usd - self.spent_usd())

    def cheap_stage_budget(self) -> Optional[float]:
        """``B_c`` — the slice of the dollar budget the cheap phase may use."""
        if self.total_budget_usd is None:
            return None
        return self.total_budget_usd * (1.0 - self.strong_reserve)

    def mark_phase(self, phase: str, iteration: int) -> None:
        snap = get_totals_snapshot()
        self._phase_cost_marks.append(
            {
                "phase": phase,
                "iteration": iteration,
                "cost_usd": snap.get("total_cost_usd"),
                "llm_calls": snap.get("calls"),
                "prompt_tokens": snap.get("total_prompt_tokens"),
                "completion_tokens": snap.get("total_completion_tokens"),
                "elapsed_s": round(time.time() - self._start_time, 2),
            }
        )

    def stop_reason(self) -> str:
        """Why the run ended — the same vocabulary for every method.

        Every controller here can end three ways, and a run that stopped on
        its dollar cap looks exactly like a completed one unless it says so.
        """
        if self.budget_stop_triggered:
            return "budget_exhausted"
        if self.shutdown_event.is_set():
            return "interrupted"
        if (
            self.requested_iterations is not None
            and self.iterations_used < self.requested_iterations
        ):
            return "generations_ended_early"
        return "generation_cap"

    def _track_best(self, program: Program) -> None:
        score = get_score(program.metrics or {})
        if score > self.best_score:
            self.best_score = score
            self.best_program = program

    def _log_progress(
        self,
        iteration: int,
        tier: str,
        program: Optional[Program],
        improved: bool = False,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._progress_path:
            return
        snap = get_totals_snapshot()
        record = {
            "iteration": iteration,
            "phase": self.phase,
            "tier": tier,
            "method": self.method_name,
            "score": get_score(program.metrics or {}) if program else None,
            "best_so_far": self.best_score if self.best_score > float("-inf") else None,
            "new_best": bool(improved),
            "cost_usd": snap.get("total_cost_usd"),
            "llm_calls": snap.get("calls"),
            "prompt_tokens": snap.get("total_prompt_tokens"),
            "completion_tokens": snap.get("total_completion_tokens"),
            "elapsed_s": round(time.time() - self._start_time, 2),
            "error": error,
        }
        if extra:
            record.update(extra)
        try:
            with open(self._progress_path, "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.debug("Could not append to %s", self._progress_path, exc_info=True)

    def write_summary(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self._summary_path:
            return
        snap = get_totals_snapshot()
        summary: Dict[str, Any] = {
            "method": self.method_name,
            "cheap_model": self.cheap_model_name,
            "strong_model": self.strong_model_name,
            "max_parallel_iterations": self.max_parallel,
            "budget_usd": self.total_budget_usd,
            "strong_reserve": self.strong_reserve,
            "iterations_used": self.iterations_used,
            "requested_iterations": self.requested_iterations,
            "stop_reason": self.stop_reason(),
            "handoff_iteration": self.handoff_iteration,
            "best_score": None if self.best_score == float("-inf") else self.best_score,
            "llm_calls_by_tier": dict(self._tier_calls),
            "phase_marks": self._phase_cost_marks,
            "budget_stop_triggered": self.budget_stop_triggered,
            "totals": snap,
            "wall_clock_s": round(time.time() - self._start_time, 2),
        }
        if extra:
            summary.update(extra)
        try:
            with open(self._summary_path, "w") as fh:
                json.dump(summary, fh, indent=2, default=str)
        except OSError:
            logger.debug("Could not write %s", self._summary_path, exc_info=True)

        # The run is over either way — fold the code log into its single JSON.
        if self.eval_log is not None:
            self.eval_log.finalize(summary)

    # ------------------------------------------------------------------

    def stop_requested(self) -> bool:
        return self.shutdown_event.is_set()


def _pool_name(pool: LLMPool) -> str:
    try:
        return ",".join(str(cfg.name) for cfg in pool.models_cfg)
    except Exception:  # noqa: BLE001
        return "unknown"


def program_text_view(program: Program) -> str:
    """Textual metadata view of a candidate (``e_text`` in Eq. 3).

    Uses what the harness already records about *why* a program exists — the
    change summary the LLM wrote plus its metric fingerprint — so the text
    view carries semantics the raw source does not.
    """
    meta = program.metadata or {}
    parts: List[str] = []
    changes = meta.get("changes")
    if isinstance(changes, str) and changes.strip():
        parts.append(changes.strip())
    metrics = program.metrics or {}
    metric_bits = [
        f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
        for k, v in list(metrics.items())[:8]
    ]
    if metric_bits:
        parts.append("metrics: " + ", ".join(metric_bits))
    return "\n".join(parts) or "no description"
