"""SpecEvo orchestrator.

Async event loop that drives a speculator-model pool plus periodic
frontier Navigator interventions. The control flow is intentionally simple —
the only sophistication lives inside :class:`ClusterArchive` (adaptive
hybrid-behavior MAP-Elites) and :class:`RankSampler` (Zipfian rank).

Phases:

1. **Bootstrap phase 1** — frontier model writes ``n_diverse_seeds``
   Navigator seeds sequentially, each one shown the seeds already
   accepted so the model is pushed toward fresh Navigator interventions.

2. **Bootstrap phase 2** — mutation model fans out
   ``n_variants_per_seed`` variants per seed in parallel.

3. **Main loop** — up to ``n_workers`` mutate/crossover workers running
   concurrently. Every ``navigator_interval`` evaluations a
   background task fires a Navigator intervention: frontier writes one seed
   informed by the current cell representatives (one program per cell,
   ranked by score, capped at ``navigator_n_anchors``), then the
   mutation model fans out ``n_navigator_variants`` variants.

The orchestrator deliberately has **no** Hall-of-Fame, **no** grace
period, **no** selector boost, **no** quota cap. A program enters the
archive only if it beats the incumbent of its cell, and once in, it is
treated like any other program. The archive's cells are re-clustered
periodically so the niche structure follows the search rather than
being imposed up-front.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..clients import LM
from ..clients.base import BaseClient
from ..simple import (
    ArchiveConfig,
    ClusterArchive,
    DescriptionEmbedder,
    EmbedderConfig,
    Monitor,
    MonitorConfig,
    OutputParser,
    ParserConfig,
    Program,
    RankSampler,
    RankSamplerConfig,
)
from ..simple.parser import fallback_summarize
from ..utils.evaluation import evaluate_code
from ..utils.resilient_pool import ResilientProcessPool
from .prompts import (
    PromptSampler,
    build_analysis_prompt,
    build_crossover_prompt,
    build_diverse_seed_prompt,
    build_init_variant_prompt,
    build_advisor_prompt,
    build_mutate_prompt,
    build_reframe_prompt,
    build_navigator_variant_prompt,
    build_surgical_exploit_prompt,
    build_synthesis_prompt,
    build_targeted_mutate_prompt,
    error_signature,
)
from .evallog import EvalCodeLog
from .snaplog import SnapLog

logger = logging.getLogger(__name__)

# Upper bound on the number of distinct failure-mode signatures the Advisor
# keeps in its accumulated error-knowledge base. The advice block only ever
# surfaces the few most-recurrent modes, so the table is merged by signature
# and capped here — recurring modes survive, one-off errors cycle out.
_ERROR_KNOWLEDGE_MAX = 24


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SpecEvoConfig:
    """End-to-end configuration for one SpecEvo run.

    Compared to prior versions, this dataclass has shrunk by ~40 fields.
    The only paper-facing ablation knobs are the three components of
    :class:`ArchiveConfig` (``use_ast``, ``use_embedding``,
    ``adaptive_recluster``).
    """

    # Problem
    problem_description: str
    function_signature: str
    score_fn: Callable[..., dict]
    fn_name: str
    inputs: list[Any] | None = None
    seed_program: str | None = None

    # Models
    speculator_model: str = "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
    navigator_model: str = "openrouter/openai/gpt-5"
    embedding_model: str = "openrouter/openai/text-embedding-3-small"

    # Budget (any one triggers stop)
    budget_dollars: float | None = None
    budget_evals: int | None = None
    budget_seconds: float | None = None
    target_score: float | None = None

    # Concurrency
    n_workers: int = 4
    n_eval_processes: int = 4
    eval_timeout: float = 120.0
    llm_temperature: float = 0.8
    llm_max_tokens: int | None = None
    navigator_max_tokens: int | None = None

    # Loop schedule
    navigator_interval: int = 50
    """Frontier Navigator intervention fires every N completed evaluations."""

    navigator_min_archive_size: int = 5
    """Skip Navigator intervention if the archive has fewer occupied cells than
    this — not enough representatives to make the prompt useful."""

    navigator_n_anchors: int = 4
    """How many cell representatives (full code + description + score)
    to send to the frontier model. Capped by the number of occupied
    cells. 4 is the default and matches what GPT-5 handles
    comfortably in a single prompt without truncation."""

    navigator_n_inspirations: int = 5
    """Description-only inspirations sent alongside the anchors. These
    widen the frontier's view of the archive without inflating tokens."""

    # Initial population
    n_diverse_seeds: int = 5
    n_variants_per_seed: int = 20
    init_diversity_temperature: float = 0.8
    init_variant_temperature: float = 0.9

    # Navigator intervention fanout
    n_navigator_variants: int = 4
    navigator_temperature: float = 0.8
    navigator_variant_temperature: float = 0.85

    # Operator mix
    p_crossover: float = 0.35
    """Static probability of choosing crossover over mutate in the main
    loop. The rank sampler already adapts its β to stagnation, which
    covers the explore/exploit trade-off; an extra 'healthy vs stuck'
    switch produced no measurable benefit in prior versions."""

    # Operator-prompt ablation (A6).
    single_prompt_operators: bool = False
    """When True, the :class:`PromptSampler` stops drawing mutate /
    crossover templates at random and always uses the single simplest
    template per operator (``MUTATE_PROMPT_GENERAL`` for mutate,
    ``CROSSOVER_PROMPT_STRUCTURAL`` for crossover). This is the A6
    ablation — it isolates the contribution of prompt-template
    diversity from the rest of the search. The targeted-mutate prompt
    is orthogonal and controlled by ``enable_targeted_mutate`` (A5)."""

    # Targeted-mutate analyzer (Đề xuất 1 — analyse-then-mutate)
    enable_targeted_mutate: bool = True
    """Master toggle for the LLM-generated parent analysis pipeline.
    When on, the orchestrator periodically asks the mutation model for
    a short 'strengths / weaknesses / suggested changes' review of the
    top-ranked programs, caches it, and (with probability
    ``p_targeted_mutate``) injects it into a separate
    :data:`TARGETED_MUTATE_PROMPT` instead of the standard mutate
    prompt."""

    analyzer_interval: int = 30
    """Refresh cached analyses every N evaluations."""

    analyzer_top_k: int = 3
    """How many *new* programs to analyse on each refresh.

    Each cycle the analyser walks the archive in score-descending order
    and picks the first ``analyzer_top_k`` programs that do NOT yet
    have a cached analysis. Programs that already have a cache entry
    are skipped (their analysis is still valid). Cache entries are
    only evicted when the underlying program leaves the archive (cell
    incumbent kicked it out, or the program was never re-inserted by
    a re-cluster) — never just because the program slipped out of
    today's top-K.

    Behavioural consequence: when the top-K is frozen (best score
    plateaued, no churn), the cache grows monotonically — refresh #1
    analyses ranks 1-3, refresh #2 analyses ranks 4-6, and so on,
    until eventually every program in the archive has an analysis.
    From then on the analyser is a no-op until a fresh program is
    admitted, at which point exactly one new analysis is produced.

    This replaces the prior "top-K only, evict everything else"
    policy that re-analysed the same three frozen parents every
    cycle and starved targeted-mutate of analysis-level diversity."""

    analyzer_temperature: float = 0.3
    analyzer_max_tokens: int = 500

    p_targeted_mutate: float = 0.5
    """Probability of choosing :data:`TARGETED_MUTATE_PROMPT` when the
    chosen parent has a cached analysis. When no cached analysis is
    available the orchestrator falls back to the standard mutate
    sampler regardless of this probability. Lowered from 0.7 → 0.5
    because the prior value crowded out the prompt-template diversity
    provided by :class:`PromptSampler` whenever a top-K parent was
    picked, which happens most of the time under the Zipfian sampler."""

    # Navigator intervention mode thresholds (three-mode shift).
    #
    # Mapping (rationale):
    #   stagnation ≤ navigator_synthesis_max_stagnation  →  synthesis
    #   stagnation ≤ navigator_surgical_max_stagnation   →  surgical
    #   stagnation  >  navigator_surgical_max_stagnation →  shift
    #
    # The earlier version routed *high* stagnation to ``surgical`` and
    # mid stagnation to ``reframe``. Empirically that left the search
    # trapped: when the champion has plateaued, asking the frontier for
    # another local fix to the same code rarely escapes. The fix that
    # actually broke prior plateaus in our runs was the ``reframe`` mode
    # (a genuinely new Navigator class). We therefore route the deepest
    # stagnation to ``reframe`` and reserve ``surgical`` for the
    # mid-stagnation band, where the champion still has some momentum
    # and a careful local polish can compound.
    navigator_synthesis_max_stagnation: float = 0.4
    """At or below this stagnation level the Navigator intervention picks the
    ``synthesis`` mode (combine 2-3 close-in-score anchors)."""

    navigator_surgical_max_stagnation: float = 0.7
    """Above ``navigator_synthesis_max_stagnation`` and at or below this
    threshold the Navigator intervention picks ``surgical`` (deep tune of the
    champion). Above this threshold the shift flips to ``reframe`` mode
    (propose a fundamentally new Navigator class) — when the search is
    deeply stuck, jumping Navigator interventions is more likely to unlock progress
    than further sanding the same local optimum."""

    navigator_synthesis_n_anchors: int = 3
    """Number of anchors surfaced in synthesis mode (2 or 3)."""

    navigator_reframe_n_anchors: int = 2
    """Number of anchors surfaced in shift mode."""

    navigator_surgical_n_inspirations: int = 5
    """Description-only inspirations passed to surgical mode in
    addition to the single (champion) anchor."""

    # Navigator-prompt ablation (A8).
    navigator_force_mode: str | None = None
    """When set to one of ``"synthesis"``, ``"surgical"`` or
    ``"reframe"``, every Navigator intervention uses that single mode regardless
    of stagnation, instead of routing across the three modes via
    ``_pick_navigator_mode``. This is the A8 ablation — it keeps the
    frontier Navigator intervention loop ON (unlike A7, which disables it
    entirely) but collapses the three-prompt repertoire down to one,
    isolating the contribution of mode diversity. ``None`` (default)
    keeps the adaptive three-mode routing."""

    # Meta-advisor — periodically writes a short prescriptive note that
    # future mutation prompts include verbatim.
    enable_advisor: bool = True
    advisor_interval: int = 50
    advisor_inject_p: float = 0.35
    """Probability of injecting the current advice block into a given
    mutation / crossover prompt.

    Lowered from 0.7 → 0.35. At 0.7 nearly every mutation prompt saw
    the same advice text, which heavily biased the mutation model
    toward whichever direction the last advisor cycle endorsed and
    blunted the diversity provided by the prompt-template sampler.
    0.35 means ~1 in 3 prompts is advice-guided, which still gives the
    advisor a meaningful nudge across a Navigator intervention window (50
    evals × 0.35 ≈ 17 advice-guided attempts) without saturating it.
    """

    advisor_temperature: float = 0.4
    advisor_max_tokens: int = 400
    advisor_mode: str = "rich"
    """Either ``"rich"`` (default) or ``"errors_only"``.

    ``rich`` feeds the Advisor with region-based trajectory signals — the
    leading niches, the niches whose frontier advanced this window
    (IMPROVING), niches that keep attracting attempts without their
    frontier moving (SATURATED), under-explored niches, and an accumulated
    failure-knowledge base — so the advice block can name what's working,
    what's saturated, what to try next, and what to avoid. Credit is
    assigned by behavioural niche, not by which prompt-template produced a
    program; "improving vs saturated" is decided by whether a niche's
    frontier moved, with no score-delta threshold. The saturation signal
    is inspired by SeaEvo's Strategic Landscape Navigation, which tracks
    effective / saturated / underexplored strategy families at the
    population level rather than reasoning about individual programs in
    isolation. ``errors_only`` deliberately drops the success-side signals
    and is the ablation flag for "old-style" defensive advice.
    """

    # ------------------------------------------------------------------
    # Error-rate ablation instrumentation.
    #
    # Every field below defaults to OFF, so a run that does not opt in
    # (any existing driver) behaves exactly
    # as before — no extra counters are touched, no files are written and
    # the advisor / budget schedules are untouched.
    # ------------------------------------------------------------------
    post_init_budget_evals: int | None = None
    """Cap evaluations at ``<evals at end of bootstrap> + N``.

    ``budget_evals`` counts every evaluation including the init phase
    (``n_diverse_seeds`` frontier seeds + ``n_variants_per_seed`` variants
    each, ~105 evals by default), which makes "run the evolutionary loop
    for exactly N iterations" impossible to express. This field is
    resolved into ``budget_evals`` once the bootstrap finishes, so the
    main loop gets exactly N evaluations of its own. ``None`` (default)
    leaves ``budget_evals`` alone."""

    ablation_window_evals: int = 0
    """When > 0, close an instrumentation window every N post-init
    evaluations: record the window's error rate and (if
    ``ablation_checkpoint_population``) dump the whole live population.
    0 (default) disables the instrumentation entirely."""

    ablation_checkpoint_population: bool = False
    """Write ``checkpoints/checkpoint_<NN>.json`` (full population code +
    descriptions + scores) at every window close."""

    align_advisor_to_post_init: bool = False
    """Restart the advisor's cadence clock at the end of the bootstrap,
    the way ``navigator_interval`` already does.

    By default ``last_advisor_eval_count`` stays at 0 while the init
    phase burns ~105 evals, so the advisor fires immediately on its first
    poll and its windows are offset from the post-init origin by an
    amount that depends on how many init candidates survived. For the
    error-rate ablation we want advisor cycle *k* and measurement window
    *k* to share one origin. They still will not coincide to the exact
    evaluation — the advisor polls on a 2 s timer, so it fires within a
    few evaluations of each boundary rather than on it."""

    save_eval_code: bool = False
    """Write ``eval_code_log.jsonl`` (+ ``eval_code_log.json`` at the end):
    the source of every candidate the run produced, including the ones that
    failed to parse, raised, scored invalid or timed out. Off by default, so
    an existing driver behaves exactly as before."""

    # Output
    output_dir: str | Path = "runs/specevo"

    # Safety
    llm_call_timeout: float = 600.0
    shutdown_grace_seconds: float = 15.0

    # Subsystem overrides (advanced)
    archive_config: ArchiveConfig = field(default_factory=ArchiveConfig)
    monitor_config: MonitorConfig = field(default_factory=MonitorConfig)
    sampler_config: RankSamplerConfig = field(default_factory=RankSamplerConfig)
    parser_config: ParserConfig = field(default_factory=ParserConfig)
    embedder_config: EmbedderConfig | None = None

    seed: int = 0


@dataclass
class NavigatorTrial:
    trial_idx: int
    description: str
    accepted: bool
    score: float | None
    delta_vs_prev_best: float | None

    def render(self, *, max_desc_chars: int = 280) -> str:
        score_str = f"{self.score:.4f}" if self.score is not None else "n/a"
        delta = self.delta_vs_prev_best
        delta_str = f"Δ={delta:+.4f}" if delta is not None else "Δ=n/a"
        accepted_str = "✓" if self.accepted else "✗"
        desc = self.description.strip().replace("\n", " ")
        if len(desc) > max_desc_chars:
            desc = desc[: max_desc_chars - 1] + "…"
        return f"[#{self.trial_idx}] {accepted_str} score={score_str} {delta_str} :: {desc}"


@dataclass
class SpecEvoResult:
    best_program: str
    best_score: float
    total_evaluations: int
    total_cost: float
    archive_size: int
    runtime_seconds: float
    output_dir: str
    navigator_trials: list[NavigatorTrial] = field(default_factory=list)
    best_metrics: dict = field(default_factory=dict)
    error_rate_windows: list[dict] = field(default_factory=list)
    """One entry per closed instrumentation window; empty unless
    ``SpecEvoConfig.ablation_window_evals`` was set."""
    total_llm_calls: int = 0
    total_prompt_tokens: int = 0
    """Input tokens over every LLM call of the run, provider-reported."""
    total_completion_tokens: int = 0
    """Output tokens over every LLM call of the run, provider-reported."""
    init_usage: dict = field(default_factory=dict)
    """Token/cost/call totals for the initialization phase alone — phase 1
    (diverse seeds) plus phase 2 (variants per seed), measured the moment
    bootstrap returns. Same keys as :meth:`_CallLog.snapshot`, plus
    ``evaluations``."""


# ---------------------------------------------------------------------------
# Cost log
# ---------------------------------------------------------------------------


@dataclass
class _CallLog:
    cost: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record(self, cost: float, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.cost += float(cost or 0.0)
        self.calls += 1
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)

    def snapshot(self) -> dict[str, float | int]:
        """Running totals as a plain dict, for summaries and phase splits."""
        return {
            "llm_calls": self.calls,
            "cost_usd": round(self.cost, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SpecEvoOrchestrator:
    def __init__(self, config: SpecEvoConfig) -> None:
        self.config = config
        random.seed(config.seed)
        self._rng = random.Random(config.seed)

        embed_cfg = config.embedder_config or EmbedderConfig(model=config.embedding_model)
        self.embedder = DescriptionEmbedder(embed_cfg)
        self.parser = OutputParser(config.parser_config)
        self.archive = ClusterArchive(config.archive_config)
        self.monitor = Monitor(config.monitor_config)
        self.sampler = RankSampler(config.sampler_config)

        self.speculator_lm: BaseClient = LM(config.speculator_model)
        self.navigator_lm: BaseClient = LM(config.navigator_model)

        self.cost = _CallLog()
        # Frozen copy of ``self.cost`` taken when the init phase ends, so the
        # bootstrap's share of tokens/spend stays separable from the search's.
        self.init_usage: dict[str, float | int] = {}
        self.navigator_trials: list[NavigatorTrial] = []
        self.recent_trials: deque[str] = deque(maxlen=6)
        # Signal streams the Advisor reads each cycle. Attempts are counted
        # per behavioural niche (archive cell) over the current window and
        # reset each cycle; the failure-knowledge base accumulates over the
        # whole run, keyed by error signature.
        self._advisor_attempts_by_cell: dict[int, int] = {}
        self._error_knowledge: dict[str, dict[str, object]] = {}

        self.start_time: float = 0.0
        self.stop_event = asyncio.Event()

        self.last_pe_eval_count: int = 0
        self.pe_trigger_count: int = 0
        self._pe_lock = asyncio.Lock()

        self.current_advisor_note: str | None = None
        self.last_advisor_eval_count: int = 0
        self.advisor_trigger_count: int = 0
        self._advisor_lock = asyncio.Lock()

        # Targeted-mutate analyzer state (Đề xuất 1).
        self.prompt_sampler = PromptSampler(
            single_prompt=config.single_prompt_operators
        )
        self._analysis_cache: dict[int, str] = {}
        self._analysis_lock = asyncio.Lock()
        self.last_analyzer_eval_count: int = 0
        self.analyzer_trigger_count: int = 0
        self.targeted_mutate_count: int = 0

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Live search trace (``snap.json``) — always on, no flag. Records
        # every Navigator call (with its routed mode) and every new best,
        # code included, flushed after each event.
        self.snap = SnapLog(
            self.output_dir / "snap.json",
            run={
                "method": "specevo",
                "speculator_model": config.speculator_model,
                "navigator_model": config.navigator_model,
                "embedding_model": config.embedding_model,
                "navigator_interval": config.navigator_interval,
                "n_workers": config.n_workers,
                "n_diverse_seeds": config.n_diverse_seeds,
                "n_variants_per_seed": config.n_variants_per_seed,
                "n_navigator_variants": config.n_navigator_variants,
                "p_crossover": config.p_crossover,
                "p_targeted_mutate": config.p_targeted_mutate,
                "navigator_force_mode": config.navigator_force_mode,
                "seed": config.seed,
            },
        )

        # Every candidate's code, valid or not. Opt-in (``--save-eval-code``).
        self.eval_log: EvalCodeLog | None = None
        if config.save_eval_code:
            self.eval_log = EvalCodeLog(
                self.output_dir / "eval_code_log.jsonl",
                run={
                    "method": "specevo",
                    "speculator_model": config.speculator_model,
                    "navigator_model": config.navigator_model,
                    "budget_evals": config.budget_evals,
                    "budget_dollars": config.budget_dollars,
                    "budget_seconds": config.budget_seconds,
                    "n_workers": config.n_workers,
                    "eval_timeout": config.eval_timeout,
                    "seed": config.seed,
                },
            )
            logger.info("[SpecEvo] saving every candidate's code to %s", self.eval_log.path)

        self._eval_processes: ResilientProcessPool | None = None
        self._semaphore = asyncio.Semaphore(config.n_workers)

        self._client_in_flight: int = 0
        self._eval_in_flight: int = 0

        # Error-rate ablation state. Inert unless
        # ``config.ablation_window_evals > 0``.
        self._post_init_eval_count: int = 0
        self._error_total: int = 0
        self._llm_failure_total: int = 0
        self._window_mark: dict[str, Any] = {}
        self._error_rate_windows: list[dict] = []

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def _budget_exhausted(self) -> bool:
        cfg = self.config
        if cfg.budget_dollars is not None and self.cost.cost >= cfg.budget_dollars:
            return True
        if cfg.budget_evals is not None and self.monitor.eval_count >= cfg.budget_evals:
            return True
        if cfg.budget_seconds is not None and (time.time() - self.start_time) >= cfg.budget_seconds:
            return True
        if cfg.target_score is not None and self.monitor.best_score >= cfg.target_score:
            return True
        return False

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    _DEFAULT_MAX_TOKENS = object()

    async def _call(
        self,
        client: BaseClient,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None | object = _DEFAULT_MAX_TOKENS,
    ) -> str:
        if max_tokens is self._DEFAULT_MAX_TOKENS:
            effective_max_tokens: int | None = self.config.llm_max_tokens
        else:
            effective_max_tokens = max_tokens  # type: ignore[assignment]
        self._client_in_flight += 1
        try:
            call_coro = client.acompletion(
                prompt, temperature=temperature, max_tokens=effective_max_tokens,
            )
            timeout = self.config.llm_call_timeout
            if timeout and timeout > 0:
                result = await asyncio.wait_for(call_coro, timeout=timeout)
            else:
                result = await call_coro
        finally:
            self._client_in_flight -= 1
        self.cost.record(
            result.cost,
            getattr(result, "prompt_tokens", 0),
            getattr(result, "completion_tokens", 0),
        )
        return result.text or ""

    async def _summarize_if_needed(self, code: str, description: str) -> str:
        if description and len(description) >= self.config.parser_config.min_description_chars:
            return description

        async def _runner(prompt: str) -> str:
            return await self._call(self.speculator_lm, prompt, temperature=0.3)

        try:
            return (await fallback_summarize(code, completion_fn=_runner)).strip()
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("[SpecEvo] fallback summary failed: %s", e)
            return description or ""

    # ------------------------------------------------------------------
    # Evaluate + admit
    # ------------------------------------------------------------------

    async def _evaluate_code(
        self, code: str, source: str = "unknown"
    ) -> tuple[float, dict, str | None]:
        outcome = await self._evaluate_code_inner(code)
        # Log here rather than at the call sites: this is the one point every
        # candidate that reaches the evaluator passes through, so the log
        # cannot drift out of sync with the search.
        if self.eval_log is not None:
            score, metrics, err = outcome
            self.eval_log.record(
                source=source,
                code=code,
                score=score,
                metrics=metrics,
                error=err,
                model=self._model_label(source),
                eval_count=self.monitor.eval_count,
                cost_usd=round(self.cost.cost, 6),
            )
        return outcome

    async def _evaluate_code_inner(self, code: str) -> tuple[float, dict, str | None]:
        assert self._eval_processes is not None
        self._eval_in_flight += 1
        try:
            try:
                result = await self._eval_processes.run(
                    evaluate_code,
                    code,
                    self.config.score_fn,
                    self.config.inputs,
                    self.config.fn_name,
                    timeout=self.config.eval_timeout,
                )
            except Exception as e:
                return float("-inf"), {}, f"executor error: {e}"
        finally:
            self._eval_in_flight -= 1
        if not isinstance(result, dict):
            return float("-inf"), {}, f"non-dict result: {type(result).__name__}"
        if "error" in result and result.get("error"):
            return float("-inf"), result, str(result["error"])
        score = float(result.get("score", 0.0))
        return score, result, None

    def _model_label(self, source: str) -> str:
        # Anything that came out of the frontier Navigator call uses the
        # Navigator model. Currently those sources are: "Navigator" (the
        # canonical Program.source for an admitted Navigator seed) and
        # "navigator_<mode>" (the per-mode label we record on rejects so
        # the run log keeps each mode's failure rate separable).
        is_navigator = source == "Navigator" or source.startswith("navigator_")
        # Variant fanout runs on the mutation model — exclude that path.
        if source.startswith("navigator_variant"):
            is_navigator = False
        model_id = (
            self.config.navigator_model if is_navigator else self.config.speculator_model
        )
        return model_id.rsplit("/", 1)[-1]

    def _record_reject(self, *, source: str, score: float = float("-inf"), error_msg: str | None = None, metrics: dict | None = None, evaluated: bool = False, llm_response: str | None = None) -> None:
        self.monitor.record_eval(score=score, accepted=False)
        if error_msg:
            self._record_error_knowledge(error_msg)
        self._log_eval(source=source, score=score, accepted=False, is_new_best=False, error_msg=error_msg, metrics=metrics)
        # ``evaluated=True`` means _evaluate_code already logged this candidate
        # with its code; the rest never produced a program to log.
        if self.eval_log is not None and not evaluated:
            self.eval_log.record(
                source=source,
                code=None,
                error=error_msg,
                model=self._model_label(source),
                eval_count=self.monitor.eval_count,
                llm_response=llm_response,
                cost_usd=round(self.cost.cost, 6),
            )
        self._ablation_after_eval(error_msg)

    def _record_error_knowledge(self, error_msg: str) -> None:
        """Merge one failure into the bounded failure-knowledge base.

        Errors are keyed by a domain-agnostic :func:`error_signature` so the
        same failure mode (the same error with different indices/sizes) merges
        into a single counted entry. The base persists across cycles (it is
        not reset per window) but is kept **bounded** at
        :data:`_ERROR_KNOWLEDGE_MAX`: when a brand-new mode arrives and the
        table is full, the least-recurrent entry (oldest among ties, via dict
        insertion order) is evicted. Recurring modes therefore survive and
        accumulate their counts while one-off errors cycle out — there is no
        point hoarding a long dict when the advice block only ever surfaces
        the few most-recurrent modes."""
        sig = error_signature(error_msg)
        clean = error_msg.strip().replace("\n", " ")[:200]
        entry = self._error_knowledge.get(sig)
        if entry is not None:
            entry["count"] = int(entry["count"]) + 1
            entry["example"] = clean
            return
        if len(self._error_knowledge) >= _ERROR_KNOWLEDGE_MAX:
            victim = min(
                self._error_knowledge,
                key=lambda k: int(self._error_knowledge[k]["count"]),
            )
            del self._error_knowledge[victim]
        self._error_knowledge[sig] = {"count": 1, "example": clean}

    # ------------------------------------------------------------------
    # Error-rate ablation instrumentation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_llm_failure(error_msg: str) -> bool:
        """True when the LLM call itself failed, so no candidate ever existed.

        The error rate is deliberately a *single* number — every failure
        counts, whatever its shape: the model returned prose with no code
        block, the code raised, the solution was invalid, the evaluator
        died. All of those are attempts that produced nothing usable, which
        is exactly what the Advisor is supposed to reduce.

        The one exclusion is a failed LLM call (API error, timeout, or a
        worker that blew up before it could evaluate anything). That is
        infrastructure noise — no program was generated, so there is no
        attempt to judge. Those evaluations are dropped from *both* sides
        of the ratio rather than counted as successes, and the count is
        reported separately per window as ``n_excluded``.
        """
        m = error_msg.strip()
        return m.startswith("LLM error") or m.startswith("worker exception")

    def _ablation_after_eval(self, error_msg: str | None) -> None:
        """Tally one evaluation and close the window if it just filled.

        Called from the two (and only two) sites that advance
        ``monitor.eval_count``, so exactly one call lands per evaluation.
        Returns immediately when the instrumentation is off.
        """
        if self.config.ablation_window_evals <= 0:
            return
        if error_msg is not None:
            if self._is_llm_failure(error_msg):
                self._llm_failure_total += 1
            else:
                self._error_total += 1
        if not self._window_mark:
            # Bootstrap is still running — windows only start once the
            # main loop's origin has been marked in ``run()``.
            return
        window = self.config.ablation_window_evals
        if self.monitor.eval_count - int(self._window_mark["eval_count"]) >= window:
            self._close_ablation_window(final=False)

    def _close_ablation_window(self, *, final: bool) -> None:
        """Record error rate (and optionally a population dump) for the
        evaluations completed since the previous window closed."""
        mark = self._window_mark
        ec = self.monitor.eval_count
        n_evals = ec - int(mark["eval_count"])
        if n_evals <= 0:
            return
        n_errors = self._error_total - int(mark["errors"])
        # Failed LLM calls produced no candidate, so they leave the ratio
        # entirely instead of padding the denominator as pseudo-successes.
        n_excluded = self._llm_failure_total - int(mark["excluded"])
        n_scored = n_evals - n_excluded
        idx = len(self._error_rate_windows) + 1
        best = self.monitor.best_score
        sample = {
            "window": idx,
            "final_partial": final,
            "eval_start": int(mark["eval_count"]),
            "eval_end": ec,
            "post_init_evals": ec - self._post_init_eval_count,
            "n_evaluations": n_evals,
            "n_scored": n_scored,
            "n_excluded": n_excluded,
            "n_errors": n_errors,
            "error_rate": (n_errors / n_scored) if n_scored > 0 else None,
            "best_score": best if best != float("-inf") else None,
            "archive_size": len(self.archive),
            "num_occupied_cells": self.archive.num_occupied_cells(),
            "total_cost": self.cost.cost,
            "elapsed_seconds": time.time() - self.start_time,
            "advisor_triggers": self.advisor_trigger_count,
        }
        self._error_rate_windows.append(sample)
        self._window_mark = {
            "eval_count": ec,
            "errors": self._error_total,
            "excluded": self._llm_failure_total,
        }
        rate = sample["error_rate"]
        logger.info(
            "[SpecEvo ablation] window %d closed at eval=%d (post-init %d) | "
            "error_rate=%s (%d/%d) | llm_failures excluded=%d",
            idx, ec, sample["post_init_evals"],
            f"{rate:.3f}" if rate is not None else "n/a",
            n_errors, n_scored, n_excluded,
        )
        if self.config.ablation_checkpoint_population:
            try:
                self._write_population_checkpoint(idx, sample)
            except Exception:  # pragma: no cover — never kill a run over a dump
                logger.exception("[SpecEvo ablation] checkpoint %d failed", idx)

    def _write_population_checkpoint(self, idx: int, sample: dict) -> None:
        """Dump every program currently in the archive, code included."""
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        population = [
            {
                "cell_id": p.cell_id,
                "score": p.score,
                "source": p.source,
                "created_at_eval": p.created_at_eval,
                "uses_count": p.uses_count,
                "description": p.description,
                "metrics": p.metrics,
                "code": p.code,
            }
            for p in self.archive.programs()
        ]
        population.sort(key=lambda d: -d["score"])
        payload = {
            "checkpoint": idx,
            "window": sample,
            "advisor": self.current_advisor_note,
            "population_size": len(population),
            "population": population,
        }
        path = ckpt_dir / f"checkpoint_{idx:02d}.json"
        path.write_text(json.dumps(payload, indent=2))
        logger.info(
            "[SpecEvo ablation] checkpoint %d written — %d programs → %s",
            idx, len(population), path,
        )

    def _report_error_rate(self) -> None:
        """Write ``error_rate_report.json``.

        The human-readable table is *not* printed here — the caller
        (``scripts/run_specevo.py``) renders it via
        :func:`format_error_rate_table` as the very last thing the run
        prints, so the ten numbers land at the bottom of the CI log
        instead of being buried above the summary.
        """
        windows = self._error_rate_windows
        if not windows:
            return
        total_evals = sum(w["n_evaluations"] for w in windows)
        total_scored = sum(w["n_scored"] for w in windows)
        total_excluded = sum(w["n_excluded"] for w in windows)
        total_errors = sum(w["n_errors"] for w in windows)

        report = {
            "window_evals": self.config.ablation_window_evals,
            "post_init_eval_count": self._post_init_eval_count,
            "enable_advisor": self.config.enable_advisor,
            "advisor_interval": self.config.advisor_interval,
            "single_prompt_operators": self.config.single_prompt_operators,
            "n_windows": len(windows),
            "total_evaluations": total_evals,
            "total_scored": total_scored,
            "total_excluded_llm_failures": total_excluded,
            "total_errors": total_errors,
            "overall_error_rate": (total_errors / total_scored) if total_scored else None,
            "windows": windows,
        }
        (self.output_dir / "error_rate_report.json").write_text(json.dumps(report, indent=2))

    @staticmethod
    def _format_problem_metrics(metrics: dict | None) -> str:
        """Render the problem-specific metrics tail, matching how the
        baselines report a run (some tasks surface ``total_cost`` /
        ``avg_cost`` / config counts rather than the bare fitness ``score``).
        Returns "" when no such metrics are present, so problems that only
        expose ``score`` log exactly as before.
        """
        if not isinstance(metrics, dict):
            return ""
        parts: list[str] = []
        # Known numeric fields, in specevo_baselines's reporting order. Only the
        # ones the problem actually returns are shown.
        # CO-Bench dev/test split: the search optimises ``dev_score`` (== the
        # fitness ``score``); ``test_score`` is the held-out tail, reported for
        # generalisation but never optimised. Surface both so a run's log/summary
        # separates them the same way the baselines do.
        if "dev_score" in metrics:
            parts.append(f"dev_score={float(metrics['dev_score']):.4f}")
        if "test_score" in metrics:
            parts.append(f"test_score={float(metrics['test_score']):.4f}")
        if "overall_score" in metrics:
            parts.append(f"overall_score={float(metrics['overall_score']):.4f}")
        if "normalized_score" in metrics:
            parts.append(f"normalized_score={float(metrics['normalized_score']):.2f}")
        if "total_cost" in metrics:
            parts.append(f"total_cost={float(metrics['total_cost']):.4f}")
        if "avg_cost" in metrics:
            parts.append(f"avg_cost={float(metrics['avg_cost']):.4f}")
        if "successful_configs" in metrics:
            parts.append(f"successful_configs={int(metrics['successful_configs'])}")
        if "failed_configs" in metrics:
            parts.append(f"failed_configs={int(metrics['failed_configs'])}")
        if "total_time" in metrics:
            parts.append(f"total_time={float(metrics['total_time']):.4f}")
        return (" | " + ", ".join(parts)) if parts else ""

    def _log_eval(
        self,
        *,
        source: str,
        score: float,
        accepted: bool,
        is_new_best: bool,
        error_msg: str | None = None,
        metrics: dict | None = None,
    ) -> None:
        model = self._model_label(source)
        if error_msg is not None:
            logger.info(
                "[Eval #%d] %-27s ERROR (%s): %s",
                self.monitor.eval_count, model, source, error_msg[:80],
            )
            return
        if is_new_best:
            status = "NEW BEST ★"
        elif accepted:
            status = "accepted"
        else:
            status = "rejected"
        best = self.monitor.best_score
        best_str = f"{best:.6f}" if best != float("-inf") else "n/a"
        score_str = f"{score:.6f}" if score != float("-inf") else "n/a"
        logger.info(
            "[Eval #%d] %-27s %-12s | source: %-18s | score: %s | best: %s | $%.3f%s",
            self.monitor.eval_count, model, status, source, score_str, best_str,
            self.cost.cost, self._format_problem_metrics(metrics),
        )

    async def _admit(
        self,
        *,
        code: str,
        description: str,
        score: float,
        source: str,
        parent_score: float | None,
        force_on_drop: bool = False,
        metrics: dict | None = None,
        navigator_mode: str | None = None,
    ) -> tuple[bool, str, bool]:
        """Admit a candidate into the archive.

        ``force_on_drop=True`` is the Navigator intervention escape valve: if the
        archive would otherwise reject the program with ``dropped_worse``
        (its score didn't beat the cell incumbent), we evict the single
        lowest-score program currently in the archive and retry. Used only
        for frontier ``Navigator`` seeds in mode=``reframe`` so a deeply
        stalled search can still inject a fresh Navigator.

        ``navigator_mode`` is carried through purely for the ``snap.json``
        trace: ``source`` collapses every frontier seed to ``"Navigator"``,
        so the routed mode has to be passed in explicitly to attribute a
        new best to synthesis / surgical / shift.

        Returns ``(accepted, reason, is_new_best)``.
        """
        embedding = await asyncio.to_thread(self.embedder.embed, description)
        program = Program(
            code=code,
            description=description,
            score=score,
            embedding=embedding,
            source=source,  # type: ignore[arg-type]
            created_at_eval=self.monitor.eval_count + 1,
            metrics=metrics if isinstance(metrics, dict) else {},
        )
        prev_best = self.monitor.best_score
        prev_best_eval = self.monitor.last_best_eval
        stagnation_before = self.monitor.stagnation_level()
        if force_on_drop:
            accepted, reason = self.archive.force_add(program)
        else:
            accepted, reason = self.archive.add(program)
        self.monitor.record_eval(score=score, accepted=accepted)
        # Count this evaluated attempt against the behavioural niche it landed
        # in. ``cell_id`` is assigned by ``archive.add`` even when the program
        # is dropped for losing to the cell incumbent, so near-misses count
        # too — this is the "busy" half of the Advisor's busy-but-stale
        # saturation signal and avoids the survivorship bias of looking only
        # at admitted programs.
        if program.cell_id >= 0:
            self._advisor_attempts_by_cell[program.cell_id] = (
                self._advisor_attempts_by_cell.get(program.cell_id, 0) + 1
            )
        is_new_best = accepted and score > prev_best
        if is_new_best:
            self.snap.new_best(
                at_eval=self.monitor.eval_count,
                source=source,
                navigator_mode=navigator_mode,
                model=self._model_label(source),
                score=score,
                prev_best=prev_best,
                evals_since_prev_best=self.monitor.eval_count - prev_best_eval,
                stagnation_before=stagnation_before,
                cell_id=program.cell_id,
                description=description,
                code=code,
                metrics=metrics,
                cost_usd=self.cost.cost,
            )
        self._log_eval(source=source, score=score, accepted=accepted, is_new_best=is_new_best, metrics=metrics)
        self._ablation_after_eval(None)
        if not accepted and reason == "dropped_worse":
            logger.debug("[SpecEvo] dropped score=%.4f did not beat cell incumbent", score)
        if accepted and reason == "forced":
            logger.info(
                "[SpecEvo PE] force-admitted Navigator seed (score=%.4f) by evicting weakest program",
                score,
            )
        return accepted, reason, is_new_best

    # ------------------------------------------------------------------
    # Worker steps
    # ------------------------------------------------------------------

    def _pick_inspirations(self, exclude: list[Program]) -> list[tuple[str, float]]:
        insps = self.sampler.select_inspirations(
            self.archive.programs(),
            exclude=exclude,
            stagnation=self.monitor.stagnation_level(),
            rng=self._rng,
        )
        for p in insps:
            self.archive.mark_used(p)
        return [(p.description, p.score) for p in insps]

    def _pick_advisor_note(self) -> str | None:
        if not self.config.enable_advisor or not self.current_advisor_note:
            return None
        if self._rng.random() >= self.config.advisor_inject_p:
            return None
        return self.current_advisor_note

    async def _generate_one(self) -> None:
        async with self._semaphore:
            if self._budget_exhausted():
                self.stop_event.set()
                return
            programs = self.archive.programs()
            if not programs:
                logger.warning("[SpecEvo] empty archive inside worker; re-bootstrapping")
                await self._bootstrap_population()
                if not self.archive.programs():
                    self._record_reject(source="bootstrap_failed", error_msg="empty archive after rebootstrap")
                return

            stagnation = self.monitor.stagnation_level()
            op = "crossover" if self._rng.random() < self.config.p_crossover and len(programs) >= 2 else "mutate"
            try:
                if op == "crossover":
                    pair = self.sampler.select_two_parents(programs, stagnation=stagnation, rng=self._rng)
                    assert pair is not None
                    p_a, p_b = pair
                    self.archive.mark_used(p_a)
                    self.archive.mark_used(p_b)
                    insps = self._pick_inspirations([p_a, p_b])
                    _label, template = self.prompt_sampler.pick_crossover(self._rng)
                    op = f"crossover_{_label}"
                    prompt = build_crossover_prompt(
                        problem_description=self.config.problem_description,
                        function_signature=self.config.function_signature,
                        parent_a_code=p_a.code, parent_a_score=p_a.score,
                        parent_b_code=p_b.code, parent_b_score=p_b.score,
                        inspirations=insps,
                        advisor=self._pick_advisor_note(),
                        template=template,
                    )
                    parent_score = max(p_a.score, p_b.score)
                else:
                    parent = self.sampler.select_parent(programs, stagnation=stagnation, rng=self._rng)
                    assert parent is not None
                    self.archive.mark_used(parent)
                    insps = self._pick_inspirations([parent])
                    # Targeted-mutate path: only if we have a cached
                    # analysis for this parent AND the coin flip lands.
                    cached_analysis = self._analysis_cache.get(id(parent))
                    use_targeted = (
                        self.config.enable_targeted_mutate
                        and cached_analysis is not None
                        and self._rng.random() < self.config.p_targeted_mutate
                    )
                    if use_targeted:
                        self.targeted_mutate_count += 1
                        op = "mutate_targeted"
                        prompt = build_targeted_mutate_prompt(
                            problem_description=self.config.problem_description,
                            function_signature=self.config.function_signature,
                            parent_code=parent.code, parent_score=parent.score,
                            analysis=cached_analysis,
                            inspirations=insps,
                            advisor=self._pick_advisor_note(),
                        )
                    else:
                        _label, template = self.prompt_sampler.pick_mutate(self._rng)
                        op = f"mutate_{_label}"
                        prompt = build_mutate_prompt(
                            problem_description=self.config.problem_description,
                            function_signature=self.config.function_signature,
                            parent_code=parent.code, parent_score=parent.score,
                            inspirations=insps,
                            advisor=self._pick_advisor_note(),
                            template=template,
                        )
                    parent_score = parent.score

                raw = await self._call(self.speculator_lm, prompt, temperature=self.config.llm_temperature)
                parsed = self.parser.parse(raw)
                if not parsed.has_code:
                    self._record_reject(source=op, error_msg="parse_miss (no code in output)", llm_response=raw)
                    return
                score, scores_dict, err = await self._evaluate_code(parsed.code, source=op)
                if err is not None:
                    self._record_reject(source=op, score=score, error_msg=err, evaluated=True)
                    return
                description = await self._summarize_if_needed(parsed.code, parsed.description)
                await self._admit(
                    code=parsed.code, description=description, score=score,
                    source=op, parent_score=parent_score, metrics=scores_dict,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[SpecEvo] worker step failed; counting as reject")
                self._record_reject(source=op, error_msg=f"worker exception: {e}")

    # ------------------------------------------------------------------
    # Analyzer — accumulating review of as-yet-unanalysed parents
    # ------------------------------------------------------------------

    async def _analyze_parent(self, parent: Program) -> str | None:
        """Generate (or fetch from cache) a short LLM review of *parent*.

        Cached by ``id(parent)`` — re-runs the analysis only when the
        parent object is brand new in memory. Failed calls return
        ``None`` and are not cached so a future refresh can retry."""
        key = id(parent)
        async with self._analysis_lock:
            cached = self._analysis_cache.get(key)
            if cached is not None:
                return cached
        prompt = build_analysis_prompt(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            parent_code=parent.code,
            parent_score=parent.score,
            parent_description=parent.description,
        )
        try:
            text = await self._call(
                self.speculator_lm, prompt,
                temperature=self.config.analyzer_temperature,
                max_tokens=self.config.analyzer_max_tokens,
            )
        except Exception:
            logger.exception("[SpecEvo analyzer] LLM call failed; skipping this parent")
            return None
        text = (text or "").strip()
        if not text:
            return None
        async with self._analysis_lock:
            self._analysis_cache[key] = text
        # No per-entry cap: the cache is bounded above by the archive
        # size (cells × in-flight). Eviction is driven exclusively by
        # ``_refresh_analyses`` when a program leaves the archive, so
        # we never silently drop a still-valid analysis for an arbitrary
        # "oldest entry" — that's the bug the prior cap could cause.
        return text

    async def _refresh_analyses(self) -> None:
        """Refresh cached analyses with an accumulating, churn-driven policy.

        Each refresh:
          1. Evicts cache entries for programs that no longer live in
             the archive (kicked out by a better cell incumbent or
             dropped during re-cluster). The cache is keyed by
             ``id(program)`` so the only valid entries are ones whose
             program object is still reachable from the archive.
          2. Walks the archive in score-descending order and analyses
             the first ``analyzer_top_k`` programs that do *not* yet
             have a cache entry. Programs already in the cache are
             skipped — their analysis is unchanged.

        Concretely: refresh #1 analyses ranks 1-3; refresh #2, given a
        frozen top-3, analyses ranks 4-6; refresh #3 analyses ranks
        7-9; … until every program in the archive has an analysis,
        after which refreshes are no-ops until a new program enters
        the archive (cell churn → exactly one new analysis next
        cycle). Top-K analyses survive across cycles unless their
        parent program is actually displaced.
        """
        cfg = self.config
        if not cfg.enable_targeted_mutate:
            return
        programs = self.archive.programs()
        if not programs:
            return

        live_ids = {id(p) for p in programs}
        async with self._analysis_lock:
            stale = [k for k in self._analysis_cache if k not in live_ids]
            for k in stale:
                self._analysis_cache.pop(k, None)
            cached_ids = set(self._analysis_cache.keys())

        ranked = sorted(programs, key=lambda p: -p.score)
        to_analyze: list[Program] = []
        for p in ranked:
            if len(to_analyze) >= cfg.analyzer_top_k:
                break
            if id(p) not in cached_ids:
                to_analyze.append(p)

        if not to_analyze:
            return
        await asyncio.gather(*(self._analyze_parent(p) for p in to_analyze))

    async def _analyzer_monitor(self) -> None:
        cfg = self.config
        if not cfg.enable_targeted_mutate or cfg.analyzer_interval <= 0:
            logger.info(
                "[SpecEvo analyzer] disabled (enable=%s, interval=%d)",
                cfg.enable_targeted_mutate, cfg.analyzer_interval,
            )
            return
        try:
            while not self.stop_event.is_set() and not self._budget_exhausted():
                await asyncio.sleep(2.0)
                if self.stop_event.is_set() or self._budget_exhausted():
                    break
                ec = self.monitor.eval_count
                if ec > 0 and ec >= self.last_analyzer_eval_count + cfg.analyzer_interval:
                    self.last_analyzer_eval_count = ec
                    self.analyzer_trigger_count += 1
                    logger.info(
                        "[SpecEvo analyzer] trigger #%d at eval=%d "
                        "(analysing up to %d uncached programs; cache=%d)",
                        self.analyzer_trigger_count, ec,
                        cfg.analyzer_top_k, len(self._analysis_cache),
                    )
                    try:
                        await self._refresh_analyses()
                    except Exception:
                        logger.exception("[SpecEvo analyzer] refresh errored")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[SpecEvo analyzer] monitor crashed")

    # ------------------------------------------------------------------
    # Navigator intervention
    # ------------------------------------------------------------------

    def _navigator_anchors(self, n: int) -> list[Program]:
        """Top-*n* cell representatives by score, descending."""
        cells = self.archive.cells()
        return sorted(cells.values(), key=lambda p: -p.score)[: max(0, n)]

    def _pick_navigator_mode(self) -> str:
        """Decide which Navigator intervention mode to fire.

        - synthesis: low stagnation, many close-in-score anchors.
        - surgical:  mid stagnation, champion still has momentum to polish.
        - shift:     high stagnation, the current Navigator family is
                     exhausted and a fresh Navigator is the most likely
                     way out.

        A8 ablation: when ``cfg.navigator_force_mode`` is set, that mode
        is returned unconditionally (no stagnation routing).
        """
        cfg = self.config
        if cfg.navigator_force_mode is not None:
            return cfg.navigator_force_mode
        s = self.monitor.stagnation_level()
        if s <= cfg.navigator_synthesis_max_stagnation:
            return "synthesis"
        if s <= cfg.navigator_surgical_max_stagnation:
            return "surgical"
        return "reframe"

    def _build_navigator_prompt_for_mode(
        self, mode: str
    ) -> tuple[str, list[Program], list[tuple[str, float]]]:
        """Assemble (prompt, anchors_used, inspiration_pairs) for *mode*."""
        cfg = self.config
        all_cells = list(self.archive.cells().values())
        all_cells.sort(key=lambda p: -p.score)

        if mode == "synthesis":
            n_anchors = cfg.navigator_synthesis_n_anchors
            anchors = all_cells[:n_anchors]
            anchor_triples = [(p.code, p.description, p.score) for p in anchors]
            anchor_ids = {id(p) for p in anchors}
            insps = [
                (p.description, p.score)
                for p in all_cells if id(p) not in anchor_ids
            ][: cfg.navigator_n_inspirations]
            prompt = build_synthesis_prompt(
                problem_description=cfg.problem_description,
                function_signature=cfg.function_signature,
                n_evaluations=self.monitor.eval_count,
                n_cells=self.archive.num_occupied_cells(),
                anchors=anchor_triples,
                inspirations=insps,
                recent_trials=list(self.recent_trials),
                stagnation=self.monitor.stagnation_level(),
            )
            return prompt, anchors, insps

        if mode == "reframe":
            n_anchors = cfg.navigator_reframe_n_anchors
            anchors = all_cells[:n_anchors]
            anchor_triples = [(p.code, p.description, p.score) for p in anchors]
            anchor_ids = {id(p) for p in anchors}
            insps = [
                (p.description, p.score)
                for p in all_cells if id(p) not in anchor_ids
            ][: cfg.navigator_n_inspirations]
            prompt = build_reframe_prompt(
                problem_description=cfg.problem_description,
                function_signature=cfg.function_signature,
                n_evaluations=self.monitor.eval_count,
                n_cells=self.archive.num_occupied_cells(),
                anchors=anchor_triples,
                inspirations=insps,
                recent_trials=list(self.recent_trials),
                stagnation=self.monitor.stagnation_level(),
            )
            return prompt, anchors, insps

        # surgical: 1 anchor (champion), top descriptions as inspiration.
        anchors = all_cells[:1]
        anchor_triples = [(p.code, p.description, p.score) for p in anchors]
        anchor_ids = {id(p) for p in anchors}
        insps = [
            (p.description, p.score)
            for p in all_cells if id(p) not in anchor_ids
        ][: cfg.navigator_surgical_n_inspirations]
        prompt = build_surgical_exploit_prompt(
            problem_description=cfg.problem_description,
            function_signature=cfg.function_signature,
            n_evaluations=self.monitor.eval_count,
            n_cells=self.archive.num_occupied_cells(),
            anchors=anchor_triples,
            inspirations=insps,
            recent_trials=list(self.recent_trials),
            stagnation=self.monitor.stagnation_level(),
        )
        return prompt, anchors, insps

    async def _navigator_intervention(self) -> None:
        cfg = self.config
        if self.archive.num_occupied_cells() < cfg.navigator_min_archive_size:
            return

        mode = self._pick_navigator_mode()
        prompt, anchors, _insps = self._build_navigator_prompt_for_mode(mode)
        for p in anchors:
            self.archive.mark_used(p)

        prev_best = self.monitor.best_score
        logger.info(
            "[SpecEvo PE] mode=%s anchors=%d stagnation=%.2f best=%.4f",
            mode, len(anchors), self.monitor.stagnation_level(),
            prev_best if prev_best != float("-inf") else float("nan"),
        )

        source_label = f"navigator_{mode}"
        snap_event = self.snap.navigator_call(
            trigger=self.pe_trigger_count,
            mode=mode,
            forced=cfg.navigator_force_mode is not None,
            at_eval=self.monitor.eval_count,
            stagnation=self.monitor.stagnation_level(),
            global_stagnation=self.monitor.global_stagnation(),
            local_stagnation=self.monitor.local_stagnation(),
            prev_best=prev_best,
            occupied_cells=self.archive.num_occupied_cells(),
            archive_size=len(self.archive),
            anchors=[
                {"cell_id": p.cell_id, "score": p.score, "source": p.source,
                 "description": p.description}
                for p in anchors
            ],
            n_inspirations=len(_insps),
            cost_usd=self.cost.cost,
        )

        # ---- frontier Navigator seed ----
        try:
            raw = await self._call(
                self.navigator_lm, prompt,
                temperature=cfg.navigator_temperature,
                max_tokens=cfg.navigator_max_tokens,
            )
        except Exception as e:
            logger.exception("[SpecEvo PE] frontier call failed; counting as reject")
            self._record_reject(source=source_label, error_msg=f"LLM error: {e}")
            self.snap.navigator_result(
                snap_event, outcome="llm_error", error=f"LLM error: {e}",
                eval_index=self.monitor.eval_count, cost_usd=self.cost.cost,
            )
            return

        parsed = self.parser.parse(raw)
        if not parsed.has_code:
            trial = NavigatorTrial(
                trial_idx=len(self.navigator_trials) + 1,
                description=parsed.description or "(no description)",
                accepted=False, score=None, delta_vs_prev_best=None,
            )
            self.navigator_trials.append(trial)
            self.recent_trials.append(trial.render())
            self._record_reject(source=source_label, error_msg="parse_miss (no code in output)", llm_response=raw)
            self.snap.navigator_result(
                snap_event, outcome="parse_miss",
                error="parse_miss (no code in output)",
                description=parsed.description,
                eval_index=self.monitor.eval_count, cost_usd=self.cost.cost,
            )
            return

        score, scores, err = await self._evaluate_code(parsed.code, source=source_label)
        description = await self._summarize_if_needed(parsed.code, parsed.description)
        accepted = False
        seed_is_new_best = False
        if err is None:
            # mode=="reframe" fires only when stagnation is high; the
            # frontier seed is the search's escape attempt. If it would
            # otherwise be dropped by its cell incumbent, evict the
            # weakest program in the archive and inject the seed anyway —
            # losing one weak slot is cheaper than missing the chance to
            # break out of the stuck Navigator family.
            force_on_drop = (mode == "reframe")
            accepted, _reason, seed_is_new_best = await self._admit(
                code=parsed.code, description=description, score=score,
                source="Navigator",
                parent_score=prev_best if prev_best != float("-inf") else None,
                force_on_drop=force_on_drop, metrics=scores,
                navigator_mode=mode,
            )
        else:
            self._record_reject(source=source_label, score=score, error_msg=err, evaluated=True)

        delta = None
        if accepted and prev_best != float("-inf"):
            delta = score - prev_best
        elif accepted:
            delta = 0.0
        self.snap.navigator_result(
            snap_event,
            outcome=("eval_error" if err is not None
                     else ("accepted" if accepted else "rejected")),
            eval_index=self.monitor.eval_count,
            score=score if err is None else None,
            delta_vs_prev_best=delta,
            is_new_best=seed_is_new_best,
            description=description or parsed.description,
            code=parsed.code,
            error=err,
            cost_usd=self.cost.cost,
        )
        trial = NavigatorTrial(
            trial_idx=len(self.navigator_trials) + 1,
            description=description or parsed.description,
            accepted=accepted, score=score if err is None else None,
            delta_vs_prev_best=delta,
        )
        self.navigator_trials.append(trial)
        self.recent_trials.append(trial.render())

        # ---- mutation fanout ----
        if err is not None or not parsed.has_code or cfg.n_navigator_variants <= 0:
            return
        if self._budget_exhausted():
            return

        base_code = parsed.code
        base_score = score
        n_variants = cfg.n_navigator_variants
        logger.info(
            "[SpecEvo PE] fanout: %d variants from Navigator seed (score=%.4f, accepted=%s)",
            n_variants, base_score, accepted,
        )

        # Stagger sampling temperature across the sibling fanout so the set
        # spans exploit → explore in one pass instead of all variants
        # sampling at the same temperature with the same prompt (which was
        # the previous failure mode — every sibling tweaked the same few
        # constants).
        center_temp = cfg.navigator_variant_temperature
        if n_variants <= 1:
            variant_temps = [center_temp]
        else:
            # Spread evenly across [center - 0.25, center + 0.15], clipped
            # to a safe LLM range. Asymmetric range biases slightly toward
            # exploration.
            lo, hi = max(0.2, center_temp - 0.25), min(1.4, center_temp + 0.15)
            step = (hi - lo) / (n_variants - 1)
            variant_temps = [lo + step * i for i in range(n_variants)]

        async def _one_navigator_variant(v_idx: int) -> dict[str, Any]:
            """Run one cheap variant; return its outcome for the snap trace.

            The fanout stats are aggregated from these return values rather
            than read off the global best, because the main-loop workers keep
            running concurrently — a best-score change during the fanout is
            not necessarily the fanout's doing.
            """
            if self._budget_exhausted():
                return {"outcome": "skipped_budget"}
            v_temp = variant_temps[v_idx]
            v_prompt = build_navigator_variant_prompt(
                problem_description=cfg.problem_description,
                function_signature=cfg.function_signature,
                base_code=base_code, base_score=base_score,
            )
            try:
                raw_v = await self._call(self.speculator_lm, v_prompt, temperature=v_temp)
            except Exception as e:
                logger.exception("[SpecEvo PE] variant LLM call failed; counting as reject")
                self._record_reject(source="navigator_variant", error_msg=f"LLM error: {e}")
                return {"outcome": "llm_error"}
            parsed_v = self.parser.parse(raw_v)
            if not parsed_v.has_code:
                self._record_reject(source="navigator_variant", error_msg="parse_miss (no code in output)", llm_response=raw_v)
                return {"outcome": "parse_miss"}
            v_score, v_scores, v_err = await self._evaluate_code(parsed_v.code, source="navigator_variant")
            if v_err is not None:
                self._record_reject(source="navigator_variant", score=v_score, error_msg=v_err, evaluated=True)
                return {"outcome": "eval_error"}
            v_description = await self._summarize_if_needed(parsed_v.code, parsed_v.description)
            v_accepted, _v_reason, v_new_best = await self._admit(
                code=parsed_v.code, description=v_description, score=v_score,
                source="navigator_variant", parent_score=base_score, metrics=v_scores,
                navigator_mode=mode,
            )
            return {
                "outcome": "accepted" if v_accepted else "rejected",
                "score": v_score,
                "is_new_best": v_new_best,
            }

        fanout = await asyncio.gather(*(_one_navigator_variant(i) for i in range(n_variants)))
        scored = [r["score"] for r in fanout if r.get("score") is not None]
        self.snap.navigator_fanout(snap_event, {
            "n_variants": len(fanout),
            "n_accepted": sum(1 for r in fanout if r["outcome"] == "accepted"),
            "n_failed": sum(
                1 for r in fanout
                if r["outcome"] in ("llm_error", "parse_miss", "eval_error")
            ),
            "n_new_best": sum(1 for r in fanout if r.get("is_new_best")),
            "best_variant_score": max(scored) if scored else None,
            "seed_score": base_score,
            "outcomes": [r["outcome"] for r in fanout],
        })

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def _bootstrap_population(self) -> None:
        cfg = self.config
        diverse_seeds: list[tuple[str, float, str]] = []

        if cfg.seed_program:
            score, scores, err = await self._evaluate_code(cfg.seed_program, source="init_seed")
            if err is None:
                description = await self._summarize_if_needed(cfg.seed_program, "")
                await self._admit(
                    code=cfg.seed_program, description=description, score=score,
                    source="init", parent_score=None, metrics=scores,
                )
                diverse_seeds.append((cfg.seed_program, score, description))
                logger.info("[SpecEvo init] seed_program admitted (score=%.4f)", score)
            else:
                self._record_reject(source="init", score=score, error_msg=err, evaluated=True)

        n_seeds = cfg.n_diverse_seeds + (0 if cfg.seed_program else 1)
        max_retries = 3
        for i in range(n_seeds):
            if self._budget_exhausted():
                logger.info("[SpecEvo init] phase 1 stop (budget exhausted)")
                break
            success = False
            for attempt in range(max_retries):
                if self._budget_exhausted():
                    break
                tag = f"[seed {i + 1}]" if attempt == 0 else f"[seed {i + 1} retry {attempt}]"
                prompt = build_diverse_seed_prompt(
                    problem_description=cfg.problem_description,
                    function_signature=cfg.function_signature,
                    existing_seeds=[(c, s) for c, s, _ in diverse_seeds],
                )
                logger.info("[SpecEvo init] %s requesting diverse seed from %s",
                            tag, self.config.navigator_model.rsplit("/", 1)[-1])
                try:
                    raw = await self._call(
                        self.navigator_lm, prompt,
                        temperature=cfg.init_diversity_temperature,
                        max_tokens=cfg.navigator_max_tokens,
                    )
                except Exception as e:
                    logger.exception("[SpecEvo init] %s frontier call failed", tag)
                    self._record_reject(source="init", error_msg=f"LLM error: {e}")
                    continue
                parsed = self.parser.parse(raw)
                if not parsed.has_code:
                    self._record_reject(source="init", error_msg="parse_miss (no code in output)", llm_response=raw)
                    continue
                score, scores, err = await self._evaluate_code(parsed.code, source="init")
                if err is not None:
                    self._record_reject(source="init", score=score, error_msg=err, evaluated=True)
                    continue
                description = await self._summarize_if_needed(parsed.code, parsed.description)
                await self._admit(
                    code=parsed.code, description=description, score=score,
                    source="init", parent_score=None, metrics=scores,
                )
                diverse_seeds.append((parsed.code, score, description))
                logger.info("[SpecEvo init] %s OK (score=%.4f)", tag, score)
                success = True
                break
            if not success:
                logger.warning("[SpecEvo init] seed %d gave up after %d retries", i + 1, max_retries)

        logger.info("[SpecEvo init] phase 1 done: %d seeds admitted", len(diverse_seeds))
        if not diverse_seeds:
            logger.error("[SpecEvo init] phase 1 produced no usable seeds — skipping phase 2")
            return

        if cfg.n_variants_per_seed <= 0:
            return

        n_variants = cfg.n_variants_per_seed * len(diverse_seeds)
        logger.info(
            "[SpecEvo init] phase 2 generating %d variants (%d × %d) in parallel",
            n_variants, cfg.n_variants_per_seed, len(diverse_seeds),
        )
        n_inspirations = min(2, len(diverse_seeds))
        prompts: list[str] = []
        for _seed_code, _seed_score, _ in diverse_seeds:
            for _ in range(cfg.n_variants_per_seed):
                insps = self._rng.sample(diverse_seeds, n_inspirations)
                prompts.append(
                    build_init_variant_prompt(
                        problem_description=cfg.problem_description,
                        function_signature=cfg.function_signature,
                        inspirations=[(c, s) for c, s, _ in insps],
                    )
                )

        async def _one_variant(prompt: str) -> None:
            if self._budget_exhausted():
                return
            try:
                raw = await self._call(self.speculator_lm, prompt, temperature=cfg.init_variant_temperature)
            except Exception as e:
                logger.exception("[SpecEvo init] variant LLM call failed; counting as reject")
                self._record_reject(source="init", error_msg=f"LLM error: {e}")
                return
            parsed = self.parser.parse(raw)
            if not parsed.has_code:
                self._record_reject(source="init", error_msg="parse_miss (no code in output)", llm_response=raw)
                return
            score, scores, err = await self._evaluate_code(parsed.code, source="init")
            if err is not None:
                self._record_reject(source="init", score=score, error_msg=err, evaluated=True)
                return
            description = await self._summarize_if_needed(parsed.code, parsed.description)
            await self._admit(
                code=parsed.code, description=description, score=score,
                source="init", parent_score=None, metrics=scores,
            )

        await asyncio.gather(*(_one_variant(p) for p in prompts))
        logger.info("[SpecEvo init] phase 2 done — archive size now %d", len(self.archive))

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self) -> SpecEvoResult:
        self.start_time = time.time()
        self._eval_processes = ResilientProcessPool(max_workers=self.config.n_eval_processes)
        cfg = self.config
        logger.info(
            "[SpecEvo] starting — budget: $%s evals=%s seconds=%s target=%s",
            cfg.budget_dollars, cfg.budget_evals, cfg.budget_seconds, cfg.target_score,
        )
        logger.info(
            "[SpecEvo] models: mutation=%s | Navigator=%s | embedding=%s",
            cfg.speculator_model, cfg.navigator_model, cfg.embedding_model,
        )
        logger.info(
            "[SpecEvo] archive: n_cells=%d use_ast=%s use_embedding=%s adaptive_recluster=%s",
            cfg.archive_config.n_cells,
            cfg.archive_config.use_ast,
            cfg.archive_config.use_embedding,
            cfg.archive_config.adaptive_recluster,
        )
        logger.info(
            "[SpecEvo] knobs: workers=%d navigator_interval=%d n_diverse_seeds=%d n_variants/seed=%d n_navigator_var=%d",
            cfg.n_workers, cfg.navigator_interval, cfg.n_diverse_seeds,
            cfg.n_variants_per_seed, cfg.n_navigator_variants,
        )
        status_task = asyncio.create_task(self._status_monitor())
        try:
            logger.info("[SpecEvo init] phase 1 starting — %d diverse seeds (sequential, frontier model)",
                        cfg.n_diverse_seeds)
            await self._bootstrap_population()
            # Everything the init phase spent — both bootstrap phases, their
            # retries and their description summaries — is exactly what the
            # counters hold right now, before any monitor task exists.
            self.init_usage = {
                **self.cost.snapshot(),
                "evaluations": self.monitor.eval_count,
            }
            logger.info("[SpecEvo] bootstrap complete — archive=%d best=%.6f cost=$%.3f evals=%d",
                        len(self.archive),
                        self.monitor.best_score if self.monitor.best_score != float("-inf") else float("nan"),
                        self.cost.cost, self.monitor.eval_count)
            logger.info(
                "[SpecEvo init] usage — llm_calls=%d input_tokens=%d output_tokens=%d cost=$%.4f",
                self.init_usage["llm_calls"], self.init_usage["prompt_tokens"],
                self.init_usage["completion_tokens"], self.init_usage["cost_usd"],
            )
            # SpecEvo PE only counts eval-cadence from the end of the init
            # phase. Init evals (phase 1 frontier seeds + phase 2 mutation
            # variants) must not push PE toward its first trigger.
            self.last_pe_eval_count = self.monitor.eval_count
            self._post_init_eval_count = self.monitor.eval_count

            # Error-rate ablation: the "iteration 0" origin for the budget,
            # the measurement windows and (optionally) the advisor cadence
            # is the end of the init phase, not the first evaluation.
            if cfg.align_advisor_to_post_init:
                self.last_advisor_eval_count = self._post_init_eval_count
            if cfg.post_init_budget_evals is not None:
                resolved = self._post_init_eval_count + cfg.post_init_budget_evals
                if cfg.budget_evals is not None:
                    logger.warning(
                        "[SpecEvo] post_init_budget_evals=%d overrides budget_evals=%s",
                        cfg.post_init_budget_evals, cfg.budget_evals,
                    )
                cfg.budget_evals = resolved
                logger.info(
                    "[SpecEvo] post-init eval budget: %d iterations from eval=%d "
                    "→ stopping at eval=%d",
                    cfg.post_init_budget_evals, self._post_init_eval_count, resolved,
                )
            if cfg.ablation_window_evals > 0:
                self._window_mark = {
                    "eval_count": self._post_init_eval_count,
                    "errors": self._error_total,
                    "excluded": self._llm_failure_total,
                }
                logger.info(
                    "[SpecEvo ablation] measuring error rate every %d post-init "
                    "evaluations (checkpoints=%s)",
                    cfg.ablation_window_evals, cfg.ablation_checkpoint_population,
                )

            pe_monitor_task = asyncio.create_task(self._pe_monitor())
            advisor_task = asyncio.create_task(self._advisor_monitor())
            analyzer_task = asyncio.create_task(self._analyzer_monitor())
            try:
                logger.info("[SpecEvo] entering evolutionary main loop (n_workers=%d)", cfg.n_workers)
                await self._main_loop()
            finally:
                self.stop_event.set()
                await self._cancel_with_grace(
                    [pe_monitor_task, advisor_task, analyzer_task],
                    label="pe/advisor/analyzer",
                )
        finally:
            self.stop_event.set()
            await self._cancel_with_grace([status_task], label="status")
            try:
                self._eval_processes.shutdown()
            except Exception:  # pragma: no cover — defensive
                logger.exception("[SpecEvo] failed to shut down process pool cleanly")

        self._flush_ablation_windows()

        elapsed = time.time() - self.start_time
        best = self.archive.best()
        result = SpecEvoResult(
            best_program=best.code if best else (self.config.seed_program or ""),
            best_score=best.score if best else float("-inf"),
            best_metrics=dict(best.metrics) if best and best.metrics else {},
            total_evaluations=self.monitor.eval_count,
            total_cost=self.cost.cost,
            archive_size=len(self.archive),
            runtime_seconds=elapsed,
            output_dir=str(self.output_dir),
            navigator_trials=list(self.navigator_trials),
            error_rate_windows=list(self._error_rate_windows),
            total_llm_calls=self.cost.calls,
            total_prompt_tokens=self.cost.prompt_tokens,
            total_completion_tokens=self.cost.completion_tokens,
            init_usage=dict(self.init_usage),
        )
        self._save_snapshot(result)
        self._save_snap_trace(result)
        self._save_eval_code_log(result)
        self._report_error_rate()
        return result

    def _flush_ablation_windows(self) -> None:
        """Close the trailing partial window, if one is worth reporting.

        Workers already in flight when the budget trips still finish, so a
        run of 500 post-init evals typically lands a handful past the cap.
        Those stragglers belong to the last full window, not to a new one:
        when the budget is an exact multiple of the window size we stop at
        that many windows and let the overshoot fall on the floor.
        """
        cfg = self.config
        if cfg.ablation_window_evals <= 0 or not self._window_mark:
            return
        if cfg.post_init_budget_evals is not None:
            max_windows = cfg.post_init_budget_evals // cfg.ablation_window_evals
            if len(self._error_rate_windows) >= max_windows:
                return
        self._close_ablation_window(final=True)

    async def _cancel_with_grace(self, tasks: list[asyncio.Task], *, label: str) -> None:
        live = [t for t in tasks if t is not None and not t.done()]
        for t in live:
            t.cancel()
        if not live:
            return
        grace = self.config.shutdown_grace_seconds
        try:
            await asyncio.wait_for(asyncio.gather(*live, return_exceptions=True), timeout=grace)
        except (TimeoutError, asyncio.TimeoutError):
            stuck = [t for t in live if not t.done()]
            logger.warning(
                "[SpecEvo] %s task(s) did not honor cancel within %.1fs (%d stuck); abandoning",
                label, grace, len(stuck),
            )

    async def _main_loop(self) -> None:
        cfg = self.config
        in_flight: set[asyncio.Task] = set()
        while not self.stop_event.is_set() and not self._budget_exhausted():
            while len(in_flight) < cfg.n_workers and not self._budget_exhausted():
                in_flight.add(asyncio.create_task(self._generate_one()))
            if not in_flight:
                await asyncio.sleep(0.05)
                continue
            done, _pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED, timeout=2.0)
            for t in done:
                exc = t.exception()
                if exc is not None:
                    logger.error("[SpecEvo] background task error: %s", exc)
            in_flight = {t for t in in_flight if not t.done()}

        leftovers: list[asyncio.Task] = list(in_flight)
        for t in leftovers:
            t.cancel()
        if leftovers:
            grace = self.config.shutdown_grace_seconds
            try:
                await asyncio.wait_for(asyncio.gather(*leftovers, return_exceptions=True), timeout=grace)
            except (TimeoutError, asyncio.TimeoutError):
                stuck = [t for t in leftovers if not t.done()]
                logger.warning(
                    "[SpecEvo] %d worker task(s) did not honor cancel within %.1fs; abandoning",
                    len(stuck), grace,
                )

    # ------------------------------------------------------------------
    # Status + cron monitors
    # ------------------------------------------------------------------

    async def _status_monitor(self) -> None:
        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(30.0)
                if self.stop_event.is_set():
                    break
                best = self.monitor.best_score
                best_str = f"{best:.6f}" if best != float("-inf") else "n/a"
                elapsed = time.time() - self.start_time
                logger.info(
                    "[Status] Cost: $%.3f | Evals: %d | Clients in-flight: %d | "
                    "Eval in-flight: %d | Archive: %d (cells %d) | Best: %s | Elapsed: %.0fs",
                    self.cost.cost, self.monitor.eval_count,
                    self._client_in_flight, self._eval_in_flight,
                    len(self.archive), self.archive.num_occupied_cells(),
                    best_str, elapsed,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[Status] monitor crashed")

    async def _pe_monitor(self) -> None:
        cfg = self.config
        if cfg.navigator_interval <= 0:
            logger.info("[SpecEvo PE] disabled (navigator_interval <= 0)")
            return
        try:
            while not self.stop_event.is_set() and not self._budget_exhausted():
                await asyncio.sleep(2.0)
                if self.stop_event.is_set() or self._budget_exhausted():
                    break
                ec = self.monitor.eval_count
                if ec > 0 and ec >= self.last_pe_eval_count + cfg.navigator_interval:
                    self.last_pe_eval_count = ec
                    async with self._pe_lock:
                        self.pe_trigger_count += 1
                        best = self.monitor.best_score
                        best_str = f"{best:.6f}" if best != float("-inf") else "n/a"
                        logger.info(
                            "[SpecEvo PE] trigger #%d at eval=%d | stagnation=%.2f "
                            "(global=%.2f local=%.2f) | best=%s | cells=%d",
                            self.pe_trigger_count, ec,
                            self.monitor.stagnation_level(),
                            self.monitor.global_stagnation(),
                            self.monitor.local_stagnation(),
                            best_str, self.archive.num_occupied_cells(),
                        )
                        try:
                            await self._navigator_intervention()
                        except Exception:
                            logger.exception("[SpecEvo PE] Navigator intervention errored")
                        self.last_pe_eval_count = max(self.last_pe_eval_count, self.monitor.eval_count)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[SpecEvo PE] monitor crashed")

    # ------------------------------------------------------------------
    # Meta-advisor
    # ------------------------------------------------------------------

    def _advisor_region_signals(
        self,
    ) -> tuple[
        list[tuple[str, float]],
        list[tuple[str, float]],
        list[tuple[str, float, int]],
        list[tuple[str, float]],
    ]:
        """Region-based Advisor signals derived from the live archive.

        Returns ``(leaders, improving, saturated, under_explored)``. Each
        niche is identified by its cell incumbent. A niche is *improving*
        when its incumbent was admitted during the current advisor window
        (its frontier moved); *saturated* when its frontier did not move but
        it kept attracting evaluated attempts (busy-but-stale, i.e. mined
        out); *under-explored* when it attracted few attempts (room to push).
        Credit is purely by niche — no per-template attribution, no
        score-delta threshold.
        """
        cells = self.archive.cells()  # cell_id -> best Program in that cell
        incumbents = sorted(cells.values(), key=lambda p: p.score, reverse=True)
        if not incumbents:
            return [], [], [], []
        now = self.monitor.eval_count
        window_start = max(0, now - max(1, self.config.advisor_interval))
        attempts = dict(self._advisor_attempts_by_cell)

        leaders = [(p.description, float(p.score)) for p in incumbents[:3]]

        # Improving: niches whose incumbent (the cell's frontier) was admitted
        # during this window — i.e. the frontier moved.
        improving_progs = [p for p in incumbents if p.created_at_eval > window_start]
        improving = [(p.description, float(p.score)) for p in improving_progs[:4]]
        improving_ids = {id(p) for p in improving_progs}
        leader_ids = {id(p) for p in incumbents[:3]}

        # Saturated: niches whose frontier did NOT move this window but which
        # still attracted evaluated attempts, most-attempted first.
        stale = [p for p in incumbents if id(p) not in improving_ids]
        stale_busy = sorted(stale, key=lambda p: attempts.get(p.cell_id, 0), reverse=True)
        saturated = [
            (p.description, float(p.score), attempts.get(p.cell_id, 0))
            for p in stale_busy
            if attempts.get(p.cell_id, 0) >= 1
        ][:3]

        # Under-explored: occupied niches (neither leader nor improving) that
        # attracted the fewest attempts this window.
        under_candidates = [
            p for p in incumbents
            if id(p) not in leader_ids and id(p) not in improving_ids
        ]
        under_sorted = sorted(under_candidates, key=lambda p: attempts.get(p.cell_id, 0))
        under_explored = [(p.description, float(p.score)) for p in under_sorted[:3]]

        return leaders, improving, saturated, under_explored

    def _error_knowledge_rows(self) -> list[tuple[str, int, str]]:
        """Accumulated failure knowledge as ``[(signature, count, example)]``."""
        return [
            (sig, int(e["count"]), str(e["example"]))
            for sig, e in self._error_knowledge.items()
        ]

    async def _generate_advisor_note(self) -> None:
        cfg = self.config
        # ``errors_only`` ablation: drop every success-side (region) signal
        # and keep only the accumulated failure knowledge. Reproduces the old
        # defensive-only advisor for measuring the success-side contribution.
        if getattr(cfg, "advisor_mode", "rich") == "errors_only":
            leaders: list[tuple[str, float]] = []
            improving: list[tuple[str, float]] = []
            saturated: list[tuple[str, float, int]] = []
            under_explored: list[tuple[str, float]] = []
        else:
            leaders, improving, saturated, under_explored = self._advisor_region_signals()
        error_rows = self._error_knowledge_rows()
        prompt = build_advisor_prompt(
            problem_description=cfg.problem_description,
            function_signature=cfg.function_signature,
            best_score=self.monitor.best_score,
            n_evaluations=self.monitor.eval_count,
            accept_rate=self.monitor.acceptance_rate(),
            stagnation_level=self.monitor.stagnation_level(),
            leaders=leaders,
            improving=improving,
            saturated=saturated,
            under_explored=under_explored,
            error_knowledge=error_rows,
            previous_advice=self.current_advisor_note,
        )
        # The window's per-niche attempt counts have now been consumed; reset
        # them so the next cycle measures a fresh window. The failure
        # knowledge base is intentionally NOT cleared — it accumulates over
        # the whole run.
        self._advisor_attempts_by_cell.clear()
        try:
            raw = await self._call(
                self.speculator_lm, prompt,
                temperature=cfg.advisor_temperature,
                max_tokens=cfg.advisor_max_tokens,
            )
        except Exception:
            logger.exception("[SpecEvo advisor] LLM call failed; keeping previous advice")
            return
        text = (raw or "").strip()
        if not text:
            logger.info("[SpecEvo advisor] empty response; keeping previous advice")
            return
        if len(text) > 1200:
            text = text[:1200].rstrip() + "…"
        self.current_advisor_note = text
        logger.info("[SpecEvo advisor] new advice (%d chars) at eval=%d", len(text), self.monitor.eval_count)

    async def _advisor_monitor(self) -> None:
        cfg = self.config
        if not cfg.enable_advisor or cfg.advisor_interval <= 0:
            logger.info(
                "[SpecEvo advisor] disabled (enable_advisor=%s, interval=%d)",
                cfg.enable_advisor, cfg.advisor_interval,
            )
            return
        try:
            while not self.stop_event.is_set() and not self._budget_exhausted():
                await asyncio.sleep(2.0)
                if self.stop_event.is_set() or self._budget_exhausted():
                    break
                ec = self.monitor.eval_count
                if ec > 0 and ec >= self.last_advisor_eval_count + cfg.advisor_interval:
                    self.last_advisor_eval_count = ec
                    async with self._advisor_lock:
                        self.advisor_trigger_count += 1
                        logger.info(
                            "[SpecEvo advisor] trigger #%d at eval=%d",
                            self.advisor_trigger_count, ec,
                        )
                        await self._generate_advisor_note()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[SpecEvo advisor] monitor crashed")

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _save_snapshot(self, result: SpecEvoResult) -> None:
        cells_dump = []
        for cell_id, prog in self.archive.cells().items():
            cells_dump.append({
                "cell_id": cell_id,
                "score": prog.score,
                "source": prog.source,
                "created_at_eval": prog.created_at_eval,
                "uses_count": prog.uses_count,
                "description": prog.description,
                "content": prog.code,
            })
        cells_dump.sort(key=lambda d: -d["score"])

        snap = {
            "method": "specevo",
            "best_score": result.best_score,
            "best_metrics": result.best_metrics,
            "total_evaluations": result.total_evaluations,
            "total_cost": result.total_cost,
            "total_llm_calls": result.total_llm_calls,
            "total_prompt_tokens": result.total_prompt_tokens,
            "total_completion_tokens": result.total_completion_tokens,
            "init_usage": result.init_usage,
            "archive_size": result.archive_size,
            "num_occupied_cells": self.archive.num_occupied_cells(),
            "runtime_seconds": result.runtime_seconds,
            "monitor": self.monitor.snapshot(),
            "advisor": {
                "enabled": self.config.enable_advisor,
                "interval": self.config.advisor_interval,
                "mode": self.config.advisor_mode,
                "inject_p": self.config.advisor_inject_p,
                "trigger_count": self.advisor_trigger_count,
                "current": self.current_advisor_note,
            },
            "analyzer": {
                "enabled": self.config.enable_targeted_mutate,
                "interval": self.config.analyzer_interval,
                "top_k": self.config.analyzer_top_k,
                "trigger_count": self.analyzer_trigger_count,
                "targeted_mutate_count": self.targeted_mutate_count,
                "p_targeted_mutate": self.config.p_targeted_mutate,
                "cache_size": len(self._analysis_cache),
            },
            "navigator_modes": {
                "synthesis_max_stagnation": self.config.navigator_synthesis_max_stagnation,
                "surgical_max_stagnation": self.config.navigator_surgical_max_stagnation,
                "synthesis_n_anchors": self.config.navigator_synthesis_n_anchors,
                "reframe_n_anchors": self.config.navigator_reframe_n_anchors,
                "surgical_n_inspirations": self.config.navigator_surgical_n_inspirations,
                "force_mode": self.config.navigator_force_mode,
            },
            "navigator_trials": [
                {
                    "idx": t.trial_idx,
                    "accepted": t.accepted,
                    "score": t.score,
                    "delta": t.delta_vs_prev_best,
                    "description": t.description,
                }
                for t in result.navigator_trials
            ],
            "cells": cells_dump,
            "ablation": {
                "use_ast": self.config.archive_config.use_ast,
                "use_embedding": self.config.archive_config.use_embedding,
                "adaptive_recluster": self.config.archive_config.adaptive_recluster,
                "n_cells": self.config.archive_config.n_cells,
                "embedding_dim": self.config.archive_config.embedding_dim,
                "recluster_every": self.config.archive_config.recluster_every,
                "single_prompt_operators": self.config.single_prompt_operators,
                "navigator_force_mode": self.config.navigator_force_mode,
            },
        }
        if self.config.ablation_window_evals > 0:
            snap["error_rate_windows"] = self._error_rate_windows
            snap["post_init_eval_count"] = self._post_init_eval_count
        (self.output_dir / "snapshot.json").write_text(json.dumps(snap, indent=2))
        if result.best_program:
            (self.output_dir / "best.py").write_text(result.best_program)

    def _save_eval_code_log(self, result: SpecEvoResult) -> None:
        """Fold ``eval_code_log.jsonl`` into its single-JSON view.

        The JSONL is already complete on disk — each candidate flushed as it
        was evaluated — so a run killed before this point still has every
        record; ``scripts/eval_log_to_json.py`` converts one of those.
        """
        if self.eval_log is None:
            return
        self.eval_log.finalize({
            "best_score": result.best_score if result.best_score != float("-inf") else None,
            "total_evaluations": result.total_evaluations,
            "total_cost": result.total_cost,
            "runtime_seconds": result.runtime_seconds,
            "output_dir": str(self.output_dir),
        })

    def _save_snap_trace(self, result: SpecEvoResult) -> None:
        """Close out ``snap.json`` with run totals and log the headline split.

        The trace is already complete on disk at this point — every event
        flushed as it happened. This only appends the ``final`` block and
        echoes the attribution to the run log so the split is visible
        without opening the file.
        """
        self.snap.finalize({
            "best_score": result.best_score if result.best_score != float("-inf") else None,
            "best_metrics": result.best_metrics,
            "total_evaluations": result.total_evaluations,
            "post_init_eval_count": self._post_init_eval_count,
            "total_cost": result.total_cost,
            "total_llm_calls": result.total_llm_calls,
            "total_prompt_tokens": result.total_prompt_tokens,
            "total_completion_tokens": result.total_completion_tokens,
            "init_usage": result.init_usage,
            "runtime_seconds": result.runtime_seconds,
            "archive_size": result.archive_size,
            "num_occupied_cells": self.archive.num_occupied_cells(),
            "pe_triggers": self.pe_trigger_count,
            "advisor_triggers": self.advisor_trigger_count,
            "analyzer_triggers": self.analyzer_trigger_count,
            "targeted_mutate_count": self.targeted_mutate_count,
        })
        s = self.snap.summary()
        nav, nb = s["navigator"], s["new_best"]
        logger.info(
            "[SpecEvo snap] navigator calls=%d %s | new bests=%d %s → %s",
            nav["calls"],
            {k: v for k, v in nav["by_mode"].items() if v},
            nb["count"],
            {k: v for k, v in nb["by_producer"].items() if v},
            self.output_dir / "snap.json",
        )


# ---------------------------------------------------------------------------
# Error-rate ablation reporting
# ---------------------------------------------------------------------------


def format_error_rate_table(windows: list[dict], *, interval: int) -> str:
    """Render the per-window error-rate table.

    Kept module-level (and out of :meth:`SpecEvoOrchestrator.run`) so the
    driver can print it as the last thing a run emits, once every other
    summary line is already on screen.
    """
    if not windows:
        return ""
    total_scored = sum(w["n_scored"] for w in windows)
    total_excluded = sum(w["n_excluded"] for w in windows)
    total_errors = sum(w["n_errors"] for w in windows)
    overall = (total_errors / total_scored) if total_scored else None

    lines = [
        "",
        "=" * 78,
        f"ERROR RATE — {len(windows)} window(s) × {interval} post-init evaluations",
        "=" * 78,
        f"{'win':>3}  {'evals':>13}  {'scored':>7}  {'err':>5}  {'rate':>7}  "
        f"{'skip':>5}  {'best':>12}",
        "-" * 78,
    ]
    for w in windows:
        best = w["best_score"]
        best_str = f"{best:.6f}" if best is not None else "n/a"
        rate = w["error_rate"]
        rate_str = f"{rate:.1%}" if rate is not None else "n/a"
        span = f"{w['eval_start']}->{w['eval_end']}"
        if w["final_partial"]:
            span += "*"
        lines.append(
            f"{w['window']:>3}  {span:>13}  {w['n_scored']:>7}  "
            f"{w['n_errors']:>5}  {rate_str:>7}  {w['n_excluded']:>5}  {best_str:>12}"
        )
    overall_str = f"{overall:.1%}" if overall is not None else "n/a"
    lines += [
        "-" * 78,
        f"overall: {total_errors}/{total_scored} = {overall_str}",
        f"skipped: {total_excluded} failed LLM call(s) — no candidate produced, "
        "excluded from the ratio",
        "=" * 78,
    ]
    if any(w["final_partial"] for w in windows):
        lines.append("* partial window (run stopped before it filled)")
    return "\n".join(lines)
