"""Eval consumers: evaluate code and update archive."""

import asyncio
import logging
import math
from collections.abc import Callable
from typing import Optional

from ..artifacts import ArtifactAdapter
from ..clients.base import client_name, short_client_name
from ..config import LeviConfig
from ..core import EvaluationResult
from ..equilibrium.equilibrium import _acompletion_with_token_retry
from ..pool import CVTMAPElitesPool
from ..selection import ComponentSelector
from ..utils import ResilientProcessPool, coerce_score
from .state import BudgetLimitReached, ErrorRecord, PipelineState


_ERROR_LOG_TAIL_CHARS = 400
"""Tail length used when echoing an error into the human-readable log.

The full traceback always reaches the repair model via the error buffer
(``_ERROR_MSG_MAX_CHARS``); the log line just needs enough of the tail
to recognise which failure mode this is.
"""


def _format_error_for_log(error_msg: str) -> str:
    """Render an error for a single ``[Eval #N] ERROR:`` log line.

    The full message can be thousands of characters (a process-pool
    traceback), so we keep the *tail* — that is where the exception
    type and final line live. Newlines are collapsed to ``\\n`` so the
    line stays single-line and greppable.
    """
    msg = (error_msg or "").strip()
    if not msg:
        return "(empty)"
    if len(msg) > _ERROR_LOG_TAIL_CHARS:
        msg = "…" + msg[-_ERROR_LOG_TAIL_CHARS:]
    return msg.replace("\n", "\\n")


_ERROR_MSG_MAX_CHARS = 4000
"""Upper bound on error context handed to the repair model.

We keep the *tail* of the message because Python tracebacks put the
actual exception type and message at the bottom — that's the line the
repair model needs to see. Heading frames are useful context but
disposable when we run out of room.
"""


def _truncate_error_tail(error_msg: str, *, limit: int = _ERROR_MSG_MAX_CHARS) -> str:
    """Return the last ``limit`` chars of ``error_msg`` with a header marker.

    Python tracebacks end with the exception type and message
    (``ZeroDivisionError: division by zero``); that suffix is what
    actionable fix-it information lives in. Truncating the *head*
    preserves it and explicitly tells the repair model that the upper
    part of the stack was dropped, so it doesn't hallucinate context
    that isn't there.
    """
    msg = error_msg or ""
    if len(msg) <= limit:
        return msg
    return "[…truncated head; full traceback was longer…]\n" + msg[-limit:]


def _maybe_push_error(
    config: LeviConfig,
    state: PipelineState,
    item: dict,
    error_msg: str,
) -> None:
    """Push the broken code into ``state.error_buffer`` if eligible.

    Repair candidates that errored again are NOT re-queued — they have
    already had their single retry.
    """
    if not config.code_repair.enabled:
        return
    if item.get("is_repair"):
        return
    code = item.get("content")
    if not code:
        return
    parent_score = item.get("parent_score")
    try:
        parent_score = float(parent_score) if parent_score is not None else float("-inf")
    except (TypeError, ValueError):
        parent_score = float("-inf")
    if not math.isfinite(parent_score):
        parent_score = float("-inf")
    state.push_error_record(
        ErrorRecord(
            code=code,
            parent_score=parent_score,
            error_msg=_truncate_error_tail(error_msg),
            parent_cell=item.get("source_cell"),
            source="repair_candidate" if item.get("is_repair") else "main_loop",
        )
    )



SNAPSHOT_INTERVAL = 10  # Save snapshot every N evaluations

logger = logging.getLogger(__name__)


def _short_model(model: str) -> str:
    return short_client_name(model)


def _model_label(item: dict) -> str:
    model = _short_model(item.get("model", "unknown"))
    sampler = item.get("sampler", "")
    # Surface the repair branch so [Eval #N] lines are immediately
    # distinguishable from regular mutation evaluations in run.txt.
    if item.get("is_repair") or sampler == "repair":
        return f"repair/{model}"
    if "_T" in sampler:
        return f"{model}{sampler[sampler.index('_T') :]}"
    return model


META_ADVISOR_PROMPT = """You are a lessons-learned advisor for an evolutionary code optimization system.

## Your Role
Analyze FAILURES from recent evaluations. Your lessons get injected into LLM prompts to help future solutions avoid the same mistakes.

## What You're Given
- **Failure count**: How many candidates failed (crashes, invalid code, timeouts, etc.)
- **Error patterns**: Specific error messages encountered (including timeouts)
- **Previous lessons**: What you advised last time (learn from what worked/didn't work)

## Your Task: Write Concise Lessons (150-200 words max)

### Focus ONLY on Failure Prevention
You do NOT see successful solutions. Your job is purely defensive:
1. **Identify error patterns** - What mistakes are being made repeatedly?
2. **Explain root causes** - Why are these errors happening?
3. **Give specific fixes** - Exactly how to avoid each error type
4. **Learn from previous advice** - If similar errors persist, strengthen the warning. If errors reduced, that advice worked.

### For Each Error Pattern:
- Quote the error briefly
- Explain what causes it
- Give a specific fix

## Output Format
Keep it SHORT and DIRECT:

**Avoid These Errors:**
- [Error pattern]: [How to fix]
- [Error pattern]: [How to fix]

---

{metrics_data}
"""


# SAL Cơ chế C — Offensive meta-advice prompt. Used in place of the defensive
# template when stagnation depth s(t) ≥ context_threshold. The model is asked
# to produce STRATEGIC suggestions (not bug fixes) given the run's trajectory.
META_ADVISOR_OFFENSIVE_PROMPT = """You are a search strategist for an evolutionary code optimization system.

## Why you are being called now
The archive has STAGNATED. The recent window shows no improvement to the best
score. Your advice is injected into the next batch of mutation prompts; the
goal is to PUSH past the plateau, not just to avoid bugs.

## What you're given
- The trajectory of the best score so far.
- Stagnation depth s(t) ∈ [0,1] (0 = healthy, 1 = fully stuck).
- Per-sampler accept counts in the recent window (which samplers actually
  produced improvements).
- The top recurring error modes (so you can mention them in passing — but
  bug-fix advice is NOT the focus here).

## Your Task: Write Strategic Lessons (150–200 words max)

Propose 3 concrete *algorithmic levers* the mutators should explore next.
Each lever must be:
1. **Concrete** — name a technique, parameter, or structural change.
2. **Actionable in code** — could be tried in the next mutation.
3. **Different from what already worked** — avoid retracing the current best.

If applicable, also mention which sampler/temperature has been producing the
recent improvements so subsequent calls can lean on it.

## Output Format
**Strategic moves to explore (do not just bug-fix):**
- [Lever 1]: …
- [Lever 2]: …
- [Lever 3]: …

If error rate is high, append ONE sentence on the dominant failure mode.

---

{metrics_data}
"""


META_ADVISOR_OFFENSIVE_PROMPT = """You are a strategist for an evolutionary code optimization system.

## Your Role
The search has STAGNATED — many evaluations have produced no NEW BEST score.
Your task is to recommend **strategic moves** (not bug fixes) that future mutations should try.

## What You're Given
- **Best score so far** and how long it has been stuck
- **Stagnation depth s(t)** ∈ [0,1] — current plateau pressure
- **Recent accept patterns** — which samplers / models / cells produced accepts
- **Top failure modes** of the current best (per-example weak spots)
- **Previous lessons** (so you can refine, not repeat)

## Your Task: Write 150–200 words of *offensive* guidance
1. **Diagnose the plateau** — what is limiting further improvement on the best solution?
2. **Suggest 3 concrete algorithmic levers** to try next. Examples:
   - "Try a different optimizer (e.g. interior-point instead of gradient ascent)"
   - "Add a final polish phase that refines borderline constraints"
   - "Explore a structurally different initialization"
3. **Keep it actionable** — name techniques, parameter ranges, library functions if useful.
4. **Avoid generic platitudes** ("be careful", "test thoroughly"). Speak in concrete moves.

## Output Format
**Strategic Moves to Try:**
- [Lever 1]: [Why it might break the plateau]
- [Lever 2]: [Why it might break the plateau]
- [Lever 3]: [Why it might break the plateau]

---

{metrics_data}
"""


async def eval_consumer(
    worker_id: int,
    code_queue: asyncio.Queue,
    pool: CVTMAPElitesPool,
    archive_lock: asyncio.Lock,
    executor: ResilientProcessPool,
    config: LeviConfig,
    artifact_adapter: ArtifactAdapter,
    state: PipelineState,
    stop_event: asyncio.Event,
    snapshot_callback: Optional[Callable[[], None]] = None,
    component_selector: Optional[ComponentSelector] = None,
) -> None:
    while not stop_event.is_set() or not code_queue.empty():
        try:
            item = await asyncio.wait_for(code_queue.get(), timeout=2.0)
        except TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        try:
            if not await state.try_start_evaluation():
                stop_event.set()
                break
            try:
                cascade = config.cascade
                quick_score = None
                if cascade.enabled and cascade.quick_inputs:
                    quick_result = await artifact_adapter.evaluate(
                        executor,
                        item["content"],
                        inputs=cascade.quick_inputs,
                        timeout=cascade.quick_timeout,
                    )
                    if "error" in quick_result:
                        result = quick_result
                    else:
                        quick_score = quick_result.get("score", 0)
                        preview_program = artifact_adapter.make_program(item["content"])
                        target_cell = pool.preview_cell(preview_program, quick_result)
                        incumbent = pool.get_elite(target_cell)
                        incumbent_quick_score = None
                        if incumbent is not None:
                            incumbent_quick_score = incumbent.result.scores.get("quick_score")

                        threshold = None
                        if incumbent_quick_score is not None:
                            threshold = incumbent_quick_score * cascade.min_score_ratio

                        if threshold is not None and quick_score < threshold:
                            result = {
                                "cascade_rejected": True,
                                "quick_score": quick_score,
                                "threshold": threshold,
                                "target_cell": target_cell,
                            }
                        else:
                            result = await artifact_adapter.evaluate(executor, item["content"])
                else:
                    result = await artifact_adapter.evaluate(executor, item["content"])
            except TimeoutError:
                result = {"error": "Timeout"}
            except Exception as e:
                result = {"error": str(e)}

            sal_bandit_active = (
                config.sal.enabled and config.sal.enable_d_thompson
            )

            async with archive_lock:
                if "cascade_rejected" in result:
                    pool.update_sampler(item["sampler"], item["source_cell"], success=False)
                    if sal_bandit_active:
                        pool.update_bandit(
                            item["sampler"],
                            item["model"],
                            accepted=False,
                            is_new_best=False,
                            mutation_prompt_id=item.get("mutation_prompt_id"),
                            llm_temperature=item.get("llm_temperature"),
                        )
                    state.record_reject()
                    if component_selector is not None and item.get("target") is not None:
                        component_selector.update(item["target"], accepted=False)
                    label = _model_label(item)
                    logger.info(
                        f"[Eval #{state.eval_count}] {label:30s} "
                        f"CASCADE SKIP  | quick: {result['quick_score']:.17g} < {result['threshold']:.17g}"
                    )
                elif "error" not in result:
                    score, score_error = coerce_score(result)
                    if score_error is not None:
                        pool.update_sampler(item["sampler"], item["source_cell"], success=False)
                        if sal_bandit_active:
                            pool.update_bandit(
                                item["sampler"],
                                item["model"],
                                accepted=False,
                                is_new_best=False,
                                mutation_prompt_id=item.get("mutation_prompt_id"),
                                llm_temperature=item.get("llm_temperature"),
                            )
                        state.record_error(score_error)
                        _maybe_push_error(config, state, item, score_error)
                        label = _model_label(item)
                        logger.info(
                            f"[Eval #{state.eval_count}] {label:30s} ERROR: "
                            f"{_format_error_for_log(score_error)}"
                        )
                    else:
                        result = dict(result)
                        result["score"] = score
                        if quick_score is not None:
                            result["quick_score"] = quick_score

                        program = artifact_adapter.make_program(item["content"])
                        eval_result = EvaluationResult(
                            scores=result,
                            is_valid=True,
                        )
                        accepted, cell_index = pool.add(program, eval_result)
                        pool.update_sampler(item["sampler"], item["source_cell"], success=accepted)

                        if component_selector is not None and item.get("target") is not None:
                            component_selector.update(item["target"], accepted=accepted)

                        # Code-repair telemetry — count one repair success when
                        # the repaired candidate enters the archive.
                        if item.get("is_repair"):
                            if accepted:
                                state.repair_success_count += 1
                            logger.info(
                                f"[Repair] eval #{state.eval_count + 1} "
                                f"{'ACCEPTED' if accepted else 'rejected'} | "
                                f"score: {score:.4g} | "
                                f"successes: {state.repair_success_count}/"
                                f"{state.repair_attempt_count}"
                            )

                        if accepted:
                            state.record_accept()
                        else:
                            state.record_reject()

                        is_new_best = score > state.best_score_so_far

                        # SAL Cơ chế D — update Thompson Beta posterior + NEW BEST
                        # counter on the (sampler, model) arm that produced this
                        # offspring. Disabled-by-default arms are silently
                        # skipped inside the pool.
                        if config.sal.enabled and config.sal.enable_d_thompson:
                            pool.update_bandit(
                                item["sampler"],
                                item["model"],
                                accepted=accepted,
                                is_new_best=is_new_best,
                                mutation_prompt_id=item.get("mutation_prompt_id"),
                                llm_temperature=item.get("llm_temperature"),
                            )

                        # SAL + PPS — keep state.eval_count_at_last_best
                        # fresh so the plateau term in stagnation_depth is
                        # computed cheaply; also snapshot the (eval, cost)
                        # of this NEW BEST into the hazard history that
                        # powers PPS's posterior survival estimate.
                        if is_new_best:
                            state.record_new_best()

                        state.record_score(
                            score=score,
                            accepted=accepted,
                            sampler=item["sampler"],
                            archive_size=pool.size(),
                            cell_index=cell_index,
                            mutation_prompt_id=item.get("mutation_prompt_id"),
                            llm_temperature=item.get("llm_temperature"),
                        )

                        if is_new_best:
                            status = "NEW BEST ★"
                        elif accepted:
                            status = "accepted"
                        else:
                            status = "rejected"

                        label = _model_label(item)
                        logger.info(
                            f"[Eval #{state.eval_count}] {label:30s} {status:12s} | "
                            f"score: {score:.17g} | best: {state.best_score_so_far:.17g} | "
                            f"${state.total_cost:.3f}"
                        )
                else:
                    pool.update_sampler(item["sampler"], item["source_cell"], success=False)
                    if config.sal.enabled and config.sal.enable_d_thompson:
                        pool.update_bandit(
                            item["sampler"],
                            item["model"],
                            accepted=False,
                            is_new_best=False,
                            mutation_prompt_id=item.get("mutation_prompt_id"),
                            llm_temperature=item.get("llm_temperature"),
                        )
                    state.record_error(result["error"])
                    _maybe_push_error(config, state, item, result["error"])
                    label = _model_label(item)
                    logger.info(
                        f"[Eval #{state.eval_count}] {label:30s} ERROR: "
                        f"{_format_error_for_log(result['error'])}"
                    )

            if config.meta_advice.enabled and state.should_generate_meta_advice(config.meta_advice.interval):
                asyncio.create_task(_generate_meta_advice(config, state))

            # Save snapshot every N evaluations
            if snapshot_callback and state.eval_count % SNAPSHOT_INTERVAL == 0:
                try:
                    snapshot_callback()
                except Exception as e:
                    logger.warning(f"[Snapshot] Failed to save: {e}")
            await state.finish_evaluation()
        except asyncio.CancelledError:
            await state.finish_evaluation()
            break
        except Exception as e:
            await state.finish_evaluation()
            logger.error(f"[Eval-{worker_id}] Unexpected error (continuing): {e}", exc_info=True)


def _format_metrics_for_llm(
    metrics: dict,
    previous_advice: str,
    progress_pct: float,
    problem_description: str = "",
    function_signature: str = "",
    *,
    sal_extras: dict | None = None,
) -> str:
    total = metrics.get("acceptances", 0) + metrics.get("rejections", 0) + metrics.get("errors", 0)
    error_count = metrics.get("errors", 0)
    top_errors = metrics.get("top_errors", [])

    data = ""
    if problem_description:
        data += f"## Problem\n{problem_description}\n\n"
    if function_signature:
        data += f"## Function Signature\n```python\n{function_signature}\n```\n\n"

    data += f"""## Progress: {progress_pct:.0f}% of budget consumed

## Recent Results ({total} candidates evaluated this period):
- Acceptances: {metrics.get("acceptances", 0)}
- Rejections: {metrics.get("rejections", 0)}
- Errors/Failures: {error_count}"""

    # SAL Cơ chế C — offensive context block when stagnant.
    if sal_extras:
        data += "\n\n## Search Trajectory"
        best_score = sal_extras.get("best_score")
        if best_score is not None:
            data += f"\n- Current best score: {best_score:.17g}"
        evals_since_best = sal_extras.get("evals_since_best")
        if evals_since_best is not None:
            data += f"\n- Evaluations since last NEW BEST: {evals_since_best}"
        stagnation = sal_extras.get("stagnation")
        if stagnation is not None:
            data += f"\n- Stagnation depth s(t): {stagnation:.2f}"
        recent_best_delta = sal_extras.get("recent_best_delta")
        if recent_best_delta is not None:
            data += f"\n- Best score Δ in recent window: {recent_best_delta:+.6g}"

        per_sampler = sal_extras.get("per_sampler_accepts") or {}
        if per_sampler:
            data += "\n\n## Per-sampler accepts in recent window"
            for sampler_name, count in sorted(per_sampler.items(), key=lambda x: -x[1]):
                data += f"\n- {sampler_name}: {count}"

    if top_errors:
        data += "\n\n## Most Common Errors (across entire run):\n"
        for err, count in top_errors:
            data += f"- ({count}x) {err}\n"

    if previous_advice:
        data += f"\n\n## Your Previous Lessons:\n{previous_advice}"

    return data


def _gather_sal_meta_extras(state: PipelineState, window: int = 50) -> dict:
    """Collect trajectory data for the offensive meta-advice prompt."""
    history = state.score_history
    if not history:
        return {}

    recent = history[-window:]
    per_sampler: dict[str, int] = {}
    for entry in recent:
        if entry.accepted and entry.sampler:
            per_sampler[entry.sampler] = per_sampler.get(entry.sampler, 0) + 1

    recent_best_delta = None
    if len(recent) >= 2:
        recent_best_delta = recent[-1].best_score - recent[0].best_score

    return {
        "best_score": state.best_score_so_far if math.isfinite(state.best_score_so_far) else None,
        "evals_since_best": state.evals_since_best(),
        "recent_best_delta": recent_best_delta,
        "per_sampler_accepts": per_sampler,
    }


async def _generate_meta_advice(config: LeviConfig, state: PipelineState) -> None:
    if not config.meta_advice.model:
        return

    metrics = state.reset_period_metrics()
    progress_pct = 0.0
    if config.budget.dollars:
        progress_pct = (state.total_cost / config.budget.dollars) * 100

    # SAL Cơ chế C — pick offensive prompt when stagnation depth is high.
    sal = config.sal
    offensive_mode = False
    sal_extras: dict | None = None
    if sal.enabled and sal.enable_c_meta_advice:
        s = state.stagnation_depth(sal.tau)
        if s >= sal.context_threshold:
            offensive_mode = True
            sal_extras = _gather_sal_meta_extras(state)
            sal_extras["stagnation"] = s

    metrics_data = _format_metrics_for_llm(
        metrics,
        state.previous_meta_advice,
        progress_pct,
        config.problem_description,
        config.function_signature,
        sal_extras=sal_extras,
    )
    template = META_ADVISOR_OFFENSIVE_PROMPT if offensive_mode else META_ADVISOR_PROMPT
    prompt = template.format(metrics_data=metrics_data)

    try:
        extras = {}
        if "deepseek" in client_name(config.meta_advice.model).lower():
            extras["reasoning"] = {"enabled": True}

        response = await _acompletion_with_token_retry(
            state,
            model=config.meta_advice.model,
            prompt=[{"role": "user", "content": prompt}],
            temperature=config.meta_advice.temperature,
            initial_max_tokens=config.meta_advice.max_tokens,
            timeout=60,
            label="Meta-Advice",
            **extras,
        )
        advice = response.text.strip()
        cost = response.cost
    except BudgetLimitReached:
        return
    except Exception as e:
        logger.warning(f"[Meta-Advice] Failed to generate: {e}")
        return

    try:
        state.previous_meta_advice = state.current_meta_advice
        state.current_meta_advice = advice

        logger.info(f"[Meta-Advice] Generated new advice (${cost:.4f})")
    except Exception as e:
        logger.warning(f"[Meta-Advice] Failed to update state: {e}")
