"""LLM producers: sample from archive and call LLM."""

import asyncio
import logging
import math
import random

from ..artifacts import ArtifactAdapter, apply_diff as _apply_diff
from ..clients.base import client_name
from ..config import LeviConfig
from ..equilibrium.equilibrium import _acompletion_with_token_retry
from ..pool import CVTMAPElitesPool
from ..prompts import ProgramWithScore
from ..selection import ComponentSelector
from .state import BudgetLimitReached, PipelineState

logger = logging.getLogger(__name__)

# Backwards-compatible re-export used by existing tests and callers.
apply_diff = _apply_diff

FEEDBACK_MAX_FAILURES = 3


def _extract_failure_feedback(elite) -> list[str] | None:
    """Random-sample up to FEEDBACK_MAX_FAILURES failure-feedback strings.

    Uniform random over examples that scored below 1.0 (failures), so that
    repeated mutations of the same parent see different failure modes
    across calls instead of always the lowest-indexed three.
    """
    if elite is None:
        return None
    scores = elite.result.scores if elite.result is not None else None
    if not isinstance(scores, dict):
        return None
    fpe = scores.get("feedback_per_example")
    pes = scores.get("per_example_scores")
    if not fpe or not pes:
        return None
    failure_idx = [i for i, s in enumerate(pes) if s < 1.0 and i < len(fpe) and fpe[i]]
    if not failure_idx:
        return None
    k = min(FEEDBACK_MAX_FAILURES, len(failure_idx))
    return [fpe[i] for i in random.sample(failure_idx, k)]


async def llm_producer(
    worker_id: int,
    code_queue: asyncio.Queue,
    pool: CVTMAPElitesPool,
    archive_lock: asyncio.Lock,
    config: LeviConfig,
    artifact_adapter: ArtifactAdapter,
    state: PipelineState,
    stop_event: asyncio.Event,
    component_selector: ComponentSelector | None = None,
) -> None:
    # Pre-load prompt-bank registry once per worker so we don't hit disk on
    # every sample. Empty dict when the bank is disabled.
    prompt_bank_registry: dict[str, str] = {}
    if config.prompt_bank.enabled:
        prompt_bank_registry = config.prompt_bank.load_prompts()
    while not stop_event.is_set():
        if state.budget_exhausted:
            break

        try:
            # Code-repair branch — pull one broken candidate from the error
            # buffer using Zipfian rank-by-parent-score and ask a light model
            # for the minimal fix. Single retry: if the repair also errors
            # the record is discarded. We piggyback on archive_lock so two
            # producers cannot fire the same repair concurrently.
            repair_record = None
            async with archive_lock:
                if pool.size() == 0:
                    logger.error(f"[LLM-{worker_id}] Archive is empty; stopping pipeline")
                    stop_event.set()
                    break
                repair_record = state.fire_repair_if_due(config.code_repair)

            if repair_record is not None:
                cfg = config.code_repair
                if not hasattr(artifact_adapter, "build_code_repair_prompt"):
                    # Adapter doesn't support repair (e.g. bundle/prompt
                    # mode); skip and let the next iteration draw a normal
                    # mutation instead.
                    logger.info(
                        f"[Repair-{worker_id}] adapter has no build_code_repair_prompt; skipping"
                    )
                    continue
                parent_score_str = (
                    f"{repair_record.parent_score:.4g}"
                    if math.isfinite(repair_record.parent_score)
                    else "n/a"
                )
                repair_model = cfg.model
                # Preview the tail of the error in the producer log line —
                # the head is usually framework boilerplate, the tail is the
                # actual exception/message. We keep this short because the
                # full traceback already lives inside repair_record.error_msg
                # and is handed verbatim to the repair model below.
                err_preview = repair_record.error_msg.strip().splitlines()[-1] if repair_record.error_msg.strip() else ""
                if len(err_preview) > 200:
                    err_preview = "…" + err_preview[-200:]
                logger.info(
                    f"[Repair-{worker_id}] FIRE attempt #{state.repair_attempt_count} "
                    f"(buffer={len(state.error_buffer)}, parent_score={parent_score_str}, "
                    f"parent_cell={repair_record.parent_cell}, "
                    f"model={client_name(repair_model)}, error={err_preview!r}, "
                    f"full_error_chars={len(repair_record.error_msg)})"
                )
                repair_prompt = artifact_adapter.build_code_repair_prompt(
                    repair_record.code,
                    error_msg=repair_record.error_msg,
                    parent_score=repair_record.parent_score,
                )
                try:
                    repair_temp = cfg.temperature if cfg.temperature is not None else config.pipeline.temperature
                    response = await _acompletion_with_token_retry(
                        state,
                        model=repair_model,
                        prompt=[{"role": "user", "content": repair_prompt}],
                        temperature=repair_temp,
                        initial_max_tokens=cfg.max_tokens,
                        timeout=300,
                        label=f"Repair-{worker_id}",
                    )
                    repair_content = response.text
                except BudgetLimitReached:
                    stop_event.set()
                    break
                except Exception as e:
                    logger.warning(f"[Repair-{worker_id}] generation failed: {e}")
                    continue

                candidate = artifact_adapter.extract_candidate(repair_content)
                if not candidate:
                    logger.info(
                        f"[Repair-{worker_id}] candidate extraction failed; discarding repair"
                    )
                    continue

                logger.info(
                    f"[Repair-{worker_id}] candidate enqueued (parent_score={parent_score_str}, "
                    f"parent_cell={repair_record.parent_cell})"
                )
                await code_queue.put(
                    {
                        "content": candidate,
                        "sampler": "repair",
                        "source_cell": repair_record.parent_cell,
                        "model": client_name(repair_model),
                        "target": None,
                        "mutation_prompt_id": None,
                        "llm_temperature": None,
                        "is_repair": True,
                        "parent_score": repair_record.parent_score,
                    }
                )
                continue

            async with archive_lock:
                sampler_name, model, mutation_prompt_id, arm_llm_temperature = pool.get_weighted_sampler_config(
                    stagnation=state.stagnation_depth(config.sal.tau) if config.sal.enabled else None,
                )
                n_parents = config.pipeline.n_parents + config.pipeline.n_inspirations
                # AdaptiveRankSampler reads ``stagnation`` from context to
                # derive its β. Other samplers ignore it. budget_progress
                # is kept for any sampler that still uses it.
                live_stagnation = (
                    state.stagnation_depth(config.sal.tau) if config.sal.enabled else 0.0
                )
                context = {
                    "budget_progress": state.budget_progress,
                    "stagnation": live_stagnation,
                }
                sample = pool.sample(sampler_name, n_parents=n_parents, context=context)

                # SAL Cơ chế B — when stagnant, augment parents with global-best
                # and a behaviorally-far elite so the mutation prompt sees a
                # gold standard and a contrasting strategy, not just look-alikes.
                extra_inspirations: list = []
                if config.sal.enabled and config.sal.enable_b_mutation_ctx:
                    s = state.stagnation_depth(config.sal.tau)
                    if s >= config.sal.context_threshold:
                        parent_cell = sample.metadata.get("source_cell")
                        best_elite = pool.best_elite()
                        if best_elite is not None and best_elite.program is not sample.parent:
                            extra_inspirations.append(best_elite.program)
                        if parent_cell is not None:
                            far_elite = pool.select_diverse_elite_from(parent_cell)
                            if (
                                far_elite is not None
                                and far_elite.program is not sample.parent
                                and (not extra_inspirations or far_elite.program is not extra_inspirations[0])
                            ):
                                extra_inspirations.append(far_elite.program)

            parent = sample.parent
            inspirations = [p for p in sample.inspirations if random.random() < 0.8]
            # Append SAL contrast inspirations after the sampler ones, but cap
            # total context size to keep prompts compact.
            inspirations.extend(extra_inspirations)
            inspirations = inspirations[:3]
            parents = [parent] + inspirations

            # Determine output mode from config
            use_diff = config.pipeline.output_mode == "diff"
            model_key = client_name(model)

            is_bundle = getattr(artifact_adapter, "is_bundle_artifact", False)
            target: str | None = None
            if is_bundle and component_selector is not None:
                seed_bundle = getattr(artifact_adapter, "seed_bundle", None)
                if seed_bundle is not None:
                    async with archive_lock:
                        target = component_selector.select(list(seed_bundle.editable_targets))

            parent_elite = pool.get_elite(sample.metadata.get("source_cell"))
            feedback = _extract_failure_feedback(parent_elite)

            base_meta_advice = (
                state.current_meta_advice if state.current_meta_advice and random.random() < 0.8 else None
            )

            mutation_kwargs = {
                "meta_advice": base_meta_advice,
                "model": model,
                "use_diff": use_diff,
            }
            if target is not None:
                mutation_kwargs["target"] = target
            if feedback:
                mutation_kwargs["feedback"] = feedback

            # SAL Cơ chế A.2 — pass trajectory context only for bare CodeAdapter
            # (PromptAdapter / bundle adapter have a different signature).
            if (
                config.sal.enabled
                and config.sal.enable_a_pe_staging
                and not is_bundle
            ):
                s = state.stagnation_depth(config.sal.tau)
                top_failures = [err for err, _ in sorted(
                    state.all_error_counts.items(), key=lambda x: -x[1]
                )[:3]]
                mutation_kwargs["best_score"] = (
                    state.best_score_so_far if math.isfinite(state.best_score_so_far) else None
                )
                mutation_kwargs["evals_since_best"] = state.evals_since_best()
                mutation_kwargs["stagnation"] = s
                mutation_kwargs["top_failures"] = top_failures or None

            # Prompt-bank takes over the prompt construction when an arm
            # carries a mutation_prompt_id. We strip kwargs that build_mutation_prompt
            # accepts but the template builder does not (model, use_diff, target)
            # because they map to PromptBuilder-specific knobs.
            if (
                mutation_prompt_id is not None
                and prompt_bank_registry
                and not is_bundle
                and hasattr(artifact_adapter, "build_mutation_prompt_from_template")
            ):
                template_text = prompt_bank_registry.get(mutation_prompt_id)
                if template_text is None:
                    logger.warning(
                        f"[LLM-{worker_id}] prompt_bank: id {mutation_prompt_id!r} not in registry; "
                        "falling back to default mutation prompt"
                    )
                    prompt = artifact_adapter.build_mutation_prompt(
                        [ProgramWithScore(p, None) for p in parents],
                        **mutation_kwargs,
                    )
                else:
                    template_kwargs = {
                        k: v
                        for k, v in mutation_kwargs.items()
                        if k not in ("model", "use_diff", "target")
                    }
                    prompt = artifact_adapter.build_mutation_prompt_from_template(
                        [ProgramWithScore(p, None) for p in parents],
                        template_text,
                        **template_kwargs,
                    )
            else:
                prompt = artifact_adapter.build_mutation_prompt(
                    [ProgramWithScore(p, None) for p in parents],
                    **mutation_kwargs,
                )

            # Per-arm LLM temperature (from prompt-bank) wins over the pipeline default.
            call_temperature = (
                arm_llm_temperature if arm_llm_temperature is not None else config.pipeline.temperature
            )

            try:
                response = await _acompletion_with_token_retry(
                    state,
                    model=model,
                    prompt=[{"role": "user", "content": prompt}],
                    temperature=call_temperature,
                    initial_max_tokens=config.pipeline.max_tokens,
                    timeout=300,
                    label=f"LLM-{worker_id}",
                )
                content = response.text
            except BudgetLimitReached:
                stop_event.set()
                break
            except Exception as e:
                logger.warning(f"[LLM-{worker_id}] [{model_key}] Error: {e}")
                await asyncio.sleep(1.0)
                continue

            # state.acompletion already accounts for cost centrally.

            if state.budget_exhausted:
                stop_event.set()
                break

            extract_kwargs: dict = {
                "parent_content": parent.content if (use_diff or target is not None) else None,
                "use_diff": use_diff,
            }
            if target is not None:
                extract_kwargs["target"] = target

            candidate_content = artifact_adapter.extract_candidate(content, **extract_kwargs)
            if not candidate_content:
                continue

            await code_queue.put(
                {
                    "content": candidate_content,
                    "sampler": sampler_name,
                    "source_cell": sample.metadata.get("source_cell"),
                    "model": model_key,
                    "target": target,
                    "mutation_prompt_id": mutation_prompt_id,
                    "llm_temperature": arm_llm_temperature,
                    "parent_score": (
                        parent_elite.result.primary_score if parent_elite is not None else float("nan")
                    ),
                }
            )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[LLM-{worker_id}] Unexpected error: {e}")
            await asyncio.sleep(1.0)
