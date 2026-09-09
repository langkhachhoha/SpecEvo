import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ..clients.base import ClientSpec


class SamplerModelPair(BaseModel):
    """A single bandit arm: (sampler, model, prompt_id, llm_temperature).

    The sampler is the parent-selection strategy (default: AdaptiveRank).
    No sampler-internal hyperparameter (softmax temperature, annealing
    cycle count, etc.) appears as an arm dimension — AdaptiveRankSampler
    derives its β from the live stagnation signal, so adding such knobs
    only fragments the bandit's posterior without informational gain.
    """

    sampler: str = "adaptive_rank"
    model: ClientSpec
    weight: float = 1.0

    # Prompt-bank dimensions (joint bandit arm with sampler/model).
    # None on both means: no prompt selection at sample time (default arm).
    mutation_prompt_id: Optional[str] = None
    llm_temperature: Optional[float] = None  # LLM sampling temperature

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("weight")
    @classmethod
    def weight_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("weight must be positive")
        return v


class BudgetConfig(BaseModel):
    dollars: Optional[float] = None
    evaluations: Optional[int] = None
    seconds: Optional[float] = None
    target_score: Optional[float] = None


class CVTConfig(BaseModel):
    n_centroids: int = 50
    data_driven_centroids: bool = True


class InitConfig(BaseModel):
    enabled: bool = True
    n_diverse_seeds: int = 4
    n_variants_per_seed: int = 20
    diversity_model: Optional[ClientSpec] = None
    variant_models: Optional[list[ClientSpec]] = None
    temperature: Optional[float] = None
    diversity_prompt: Optional[str] = None  # Custom prompt for diverse seed generation
    diversity_llm_kwargs: dict = Field(
        default_factory=dict
    )  # Extra kwargs passed to diversity LLM calls (e.g. reasoning_effort, max_tokens)

    model_config = {"arbitrary_types_allowed": True}


class MetaAdviceConfig(BaseModel):
    enabled: bool = True
    interval: int = 50
    model: Optional[ClientSpec] = None
    max_tokens: int = 400
    temperature: Optional[float] = None

    model_config = {"arbitrary_types_allowed": True}


class BehaviorConfig(BaseModel):
    ast_features: list[str] = Field(default=["loop_count", "branch_count", "math_operators", "loop_nesting_max"])
    score_keys: list[str] = Field(default_factory=list)

    # Custom extractors: Callable[[Program], float]. Unlike built-in AST extractors,
    # these receive only the Program, making them usable for non-code content types.
    custom_extractors: dict[str, Callable] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class CascadeConfig(BaseModel):
    enabled: bool = True
    quick_inputs: list[Any] = Field(default_factory=list)
    min_score_ratio: float = 0.8
    quick_timeout: float = 30.0


class PunctuatedEquilibriumConfig(BaseModel):
    """Configuration for Punctuated Equilibrium feature.

    Periodically triggers paradigm-shift generation using a heavy model
    (early/mid/late prompt template chosen by PPS stagnation depth),
    creating fundamentally new solutions to escape local optima. Light
    variant models then produce a handful of nearby variants of the
    accepted paradigm shift.

    A 1-sentence summary of each paradigm-shift code is logged into
    ``state.strategy_history`` by a *light* summariser model so the next
    PE event can be told which approaches have already been tried.
    """

    enabled: bool = True
    interval: int = 10
    n_clusters: int = 3
    n_variants: int = 3
    heavy_models: Optional[list[ClientSpec]] = None
    variant_models: Optional[list[ClientSpec]] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    component_selector: Any = "stagnation"
    share_main_selector_stats: bool = True

    model_config = {"arbitrary_types_allowed": True}


class StrategyLogConfig(BaseModel):
    """Strategy-history logging (post-mortem of each PE event).

    A *light* model summarises each Punctuated Equilibrium paradigm-shift
    code into a one-sentence description of its algorithmic approach. The
    rolling history is splined into the next heavy prompt so the heavy
    model can avoid retracing approaches that already failed.

    The summary call is cheap (small model, ~80 tokens output) and only
    fires when a paradigm-shift candidate was successfully extracted —
    there is no value in summarising garbage.
    """

    enabled: bool = True
    max_entries: int = 8
    """How many of the most recent strategy records are rendered into the
    heavy prompt. The deque itself stores more (12) so older entries are
    still visible to telemetry / snapshot dumps."""

    summariser_model: Optional[ClientSpec] = None
    """When None we fall back to ``mutation_models[0]`` (cheap & fast)."""

    summariser_max_tokens: int = 400
    """Output-token cap for the strategy summary. Sized to fit the
    two-line IDEA / QUALITY template (≤ 60 words) with a small safety
    margin; keeps the summariser cost negligible per PE."""

    summariser_temperature: float = 0.2

    model_config = {"arbitrary_types_allowed": True}


class CodeRepairConfig(BaseModel):
    """One-shot code-repair using a light model.

    When a candidate fails evaluation (syntax error, runtime exception,
    invalid score), it is pushed into a bounded buffer in pipeline state.
    Every ``repair_every_n`` producer iterations, a worker pulls one
    broken candidate from the buffer using a rank-by-parent-score Zipfian
    distribution (same shape as :class:`AdaptiveRankSampler`) and asks a
    light model for the minimal fix. The repair is one-shot: if the fix
    also fails, the record is discarded.
    """

    enabled: bool = True
    buffer_size: int = 64
    repair_every_n: int = 8
    """Run a repair attempt every ``repair_every_n`` main-loop offspring."""

    beta: float = 1.5
    """Zipfian exponent for rank-by-parent-score selection from the
    error buffer. Higher β favours errors whose parent was strong."""

    max_per_run: int = 100
    """Hard cap on total repair attempts to avoid budget runaway."""

    model: Optional[ClientSpec] = None
    """Repair model (defaults to ``mutation_models[0]``)."""

    max_tokens: int = 4000
    temperature: Optional[float] = None

    model_config = {"arbitrary_types_allowed": True}


class AdaptiveIslandConfig(BaseModel):
    """Adaptive Island Expansion (AdaEvolve-style archive growth).

    Unifies the previous "stagnation rescue" and "adaptive CVT growth"
    mechanisms into a single, more principled idea:

        When the search is stuck (s(t) ≥ threshold) AND a Punctuated
        Equilibrium candidate (paradigm-shift or variant) fails the
        strict MAP-Elites admission test in its natural cell, do NOT
        replace the incumbent — instead, append the candidate's own
        normalised behaviour vector as a brand-new centroid and seed
        that new cell with the candidate.

    The archive thus grows *organically*: each rescued PE candidate
    opens a new "island" in behaviour space exactly where it lives,
    rather than relying on a separate buffer of recent behaviours or on
    a relaxed admission threshold that risks evicting good incumbents.

    The mechanism only fires for PE candidates (not main-loop offspring)
    and is bounded by ``max_per_run`` plus the hard ceiling
    ``max_total_centroids`` so a bad PE batch cannot inflate the
    archive without bound.
    """

    enabled: bool = True
    stagnation_threshold: float = 0.7
    """Minimum stagnation depth s(t) at which expansion may fire."""

    max_per_run: int = 16
    """Maximum island expansions per run (hard cap)."""

    max_total_centroids: int = 200
    """Hard ceiling on total centroids — protects against runaway growth."""


class PromptBankConfig(BaseModel):
    """Prompt-bank + temperature-bank for joint Thompson-bandit mutation.

    When enabled, every (sampler, model) is cross-product expanded with every
    (prompt_id, llm_temperature) pair from the two banks. Each combination
    becomes an independent bandit arm (α/β/new_best tracked in pool).

    Prompts in this bank are *full templates* — they replace the standard
    PromptBuilder output. Supported placeholders (all optional, missing ones
    render as empty string):

      {problem_description}       raw problem statement
      {function_signature}        target signature, no ```python wrap
      {parents_block}             v1/v2/.. parent blocks with score + code fence
      {search_trajectory_block}   SAL trajectory section (header + bullets), or ""
      {feedback_block}            per-example failure bullets, or ""
      {advisor_block}         meta-advice section, or ""

    Bank fully overrides ``prompt_opt`` (DSPy MIPROv2). If both are enabled,
    ``prompt_opt`` is skipped with a warning.
    """

    enabled: bool = False

    # Either point to JSON files on disk OR pass inline lists. Inline takes
    # precedence when both are set (useful for tests).
    prompts_file: Optional[str] = None
    temperatures_file: Optional[str] = None

    # Inline overrides. Each prompt: {"id": str, "text": str}.
    prompts: list[dict[str, str]] = Field(default_factory=list)
    temperatures: list[float] = Field(default_factory=list)

    # Replace any default auto-generated sampler_model_pairs with the cross
    # product? When False, the bank arms are *appended* (rare; usually you
    # want True so the bank is the only source of arms).
    replace_default_pairs: bool = True

    model_config = {"arbitrary_types_allowed": True}

    def load_prompts(self) -> dict[str, str]:
        """Return {prompt_id: text}, merging file and inline definitions.

        Inline entries win on id collision.
        """
        merged: dict[str, str] = {}
        if self.prompts_file:
            path = Path(self.prompts_file)
            if not path.is_file():
                raise FileNotFoundError(f"prompts_file does not exist: {self.prompts_file}")
            payload = json.loads(path.read_text())
            if not isinstance(payload, list):
                raise ValueError(f"prompts_file must contain a JSON list: {self.prompts_file}")
            for entry in payload:
                if not isinstance(entry, dict) or "id" not in entry or "text" not in entry:
                    raise ValueError(f"each prompt entry must be {{'id', 'text'}}; got {entry!r}")
                merged[str(entry["id"])] = str(entry["text"])
        for entry in self.prompts:
            if "id" not in entry or "text" not in entry:
                raise ValueError(f"inline prompt entry must be {{'id', 'text'}}; got {entry!r}")
            merged[str(entry["id"])] = str(entry["text"])
        return merged

    def load_temperatures(self) -> list[float]:
        if self.temperatures:
            return [float(t) for t in self.temperatures]
        if self.temperatures_file:
            path = Path(self.temperatures_file)
            if not path.is_file():
                raise FileNotFoundError(f"temperatures_file does not exist: {self.temperatures_file}")
            payload = json.loads(path.read_text())
            if not isinstance(payload, list):
                raise ValueError(f"temperatures_file must contain a JSON list: {self.temperatures_file}")
            return [float(t) for t in payload]
        return []


class PromptOptConfig(BaseModel):
    enabled: bool = False
    teacher_model: Optional[ClientSpec] = None  # Model for MIPROv2 instruction proposals; None = paradigm_models[0]
    n_trials: int = 12
    num_candidates: int = 4
    num_threads: int = 4
    init_temperature: float = 1.2
    optimize_mutation: bool = True
    optimize_paradigm_shift: bool = True  # Only runs if PE is enabled
    cache_dir: Optional[str] = None  # None = output_dir or cwd
    force: bool = False  # Re-optimize even if cached

    model_config = {"arbitrary_types_allowed": True}


class ProxyBenchmarkConfig(BaseModel):
    """Learn a smaller evaluation subset from init-stage per-problem scores."""

    enabled: bool = False
    discovery_inputs: list[Any] = Field(default_factory=list)
    matrix_key: str = "problem_scores"
    subset_size: int = 15
    selected_indices: list[int] = Field(default_factory=list)
    debug_logging: bool = False
    debug_top_k: int = 5


class SalConfig(BaseModel):
    """Stagnation-Adaptive Loop (SAL).

    A coordinated bundle of small enhancements that all read from one signal —
    the *stagnation depth* s(t) ∈ [0,1] — to escape local optima without
    breaking existing API.

    Mechanisms (each can be toggled independently for ablation):
      A. PE prompt staging: pick early / mid / late prompt by s(t).
      B. Mutation context: when s ≥ context_threshold, augment prompt with the
         global-best elite and a behaviorally-far elite (farthest-first).
      C. Meta-advice dual-mode: defensive when s low, offensive when s high.
      D. Thompson Beta-Bernoulli bandit over sampler-model pairs.
         Reward = accept_indicator; multiplicative bonus from NEW BEST count.
      E. Hard-PE: when s ≥ hard_pe_threshold and consecutive PE failed, fire a
         heavier PE (more clusters, farthest-first reps, reasoning_effort=high).
    """

    enabled: bool = True

    # --- Stagnation signal ---
    tau: int = 80
    """Plateau length (evals) at which s(t) saturates to 1.0."""

    sigma_window: int = 30
    """Window size for tracking accepted-score std (diagnostic only)."""

    # --- Per-mechanism toggles (all on by default when SAL is enabled) ---
    enable_a_pe_staging: bool = True
    enable_b_mutation_ctx: bool = True
    enable_c_meta_advice: bool = True
    enable_d_thompson: bool = True
    enable_e_hard_pe: bool = True

    # --- Cơ chế A ---
    pe_staging_mid_threshold: float = 0.3
    pe_staging_late_threshold: float = 0.7

    # --- Cơ chế B ---
    context_threshold: float = 0.5
    """s(t) above which mutation context gets global-best + farthest-elite."""

    # --- Cơ chế D ---
    bandit_w_min: float = 0.05
    """Floor weight per arm; ensures every arm keeps a chance."""

    bandit_new_best_bonus: float = 0.5
    """γ in weight ∝ θ × (1 + γ·nb)^(1+s). Higher = stronger NEW BEST bias."""

    bandit_alpha_prior: float = 1.0
    bandit_beta_prior: float = 1.0

    # --- Cơ chế E ---
    hard_pe_threshold: float = 0.8
    hard_pe_max_per_run: int = 2
    hard_pe_n_clusters: int = 6
    hard_pe_reasoning_effort: str = "high"


class PipelineConfig(BaseModel):
    n_llm_workers: int = 4
    n_eval_processes: int = 4
    eval_timeout: float = 60.0
    temperature: Optional[float] = None
    max_tokens: int = 16384
    n_parents: int = 1
    n_inspirations: int = 1
    output_mode: str = "full"


class LeviConfig(BaseModel):
    # Required
    problem_description: str
    function_signature: str
    seed_program: str | None = None
    inputs: Optional[list[Any]] = None
    score_fn: Callable[..., dict]
    budget: BudgetConfig

    # Core model config
    paradigm_models: ClientSpec | list[ClientSpec] = "openai/gpt-4o"
    mutation_models: ClientSpec | list[ClientSpec] = "openai/gpt-4o-mini"

    # Auto-generated from mutation_models if not provided.
    # Pass explicitly to override (e.g. for custom sampler/temperature combos).
    sampler_model_pairs: list[SamplerModelPair] = Field(default_factory=list)

    # Optional with defaults
    cvt: CVTConfig = Field(default_factory=CVTConfig)
    init: InitConfig = Field(default_factory=InitConfig)
    meta_advice: MetaAdviceConfig = Field(default_factory=MetaAdviceConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    punctuated_equilibrium: PunctuatedEquilibriumConfig = Field(default_factory=PunctuatedEquilibriumConfig)
    strategy_log: StrategyLogConfig = Field(default_factory=StrategyLogConfig)
    code_repair: CodeRepairConfig = Field(default_factory=CodeRepairConfig)
    adaptive_island: AdaptiveIslandConfig = Field(default_factory=AdaptiveIslandConfig)
    prompt_opt: PromptOptConfig = Field(default_factory=PromptOptConfig)
    prompt_bank: PromptBankConfig = Field(default_factory=PromptBankConfig)
    proxy_benchmark: ProxyBenchmarkConfig = Field(default_factory=ProxyBenchmarkConfig)
    sal: SalConfig = Field(default_factory=SalConfig)

    output_dir: Optional[str] = None  # Directory for snapshots

    # Prompt overrides from DSPy optimization
    prompt_overrides: dict[str, Any] = Field(default_factory=dict)

    # Per-component mutation selector for prompt bundles. String key
    # ("ucb", "round_robin", "stagnation") or a ComponentSelector instance.
    component_selector: Any = "ucb"

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _auto_wire_models(self) -> "LeviConfig":
        # 1. Coerce single client spec to list[ClientSpec]
        if not isinstance(self.paradigm_models, list):
            self.paradigm_models = [self.paradigm_models]
        if not isinstance(self.mutation_models, list):
            self.mutation_models = [self.mutation_models]

        # 2. Auto-generate sampler_model_pairs if not provided.
        #    With AdaptiveRankSampler the bandit arm space is just
        #    (model[, prompt_id, llm_temperature]). Each mutation model
        #    becomes exactly one base arm; the prompt bank multiplies
        #    that by (prompt × llm_temperature) if enabled.
        if not self.sampler_model_pairs:
            self.sampler_model_pairs = [
                SamplerModelPair(sampler="adaptive_rank", model=model, weight=1.0)
                for model in self.mutation_models
            ]

        # 2b. Cross-product expansion with the prompt bank (joint bandit arm).
        # Each base (sampler, model) becomes (sampler, model, prompt_id, llm_temperature)
        # for every (prompt_id, llm_temperature) pair in the bank.
        if self.prompt_bank.enabled:
            prompts_map = self.prompt_bank.load_prompts()
            temps = self.prompt_bank.load_temperatures()
            if not prompts_map:
                raise ValueError("prompt_bank.enabled=True but the prompts pool is empty")
            if not temps:
                raise ValueError("prompt_bank.enabled=True but the temperatures pool is empty")

            base_pairs = self.sampler_model_pairs
            expanded: list[SamplerModelPair] = []
            for base in base_pairs:
                for pid in prompts_map.keys():
                    for t in temps:
                        expanded.append(
                            SamplerModelPair(
                                sampler=base.sampler,
                                model=base.model,
                                weight=base.weight,
                                mutation_prompt_id=pid,
                                llm_temperature=float(t),
                            )
                        )

            if self.prompt_bank.replace_default_pairs:
                self.sampler_model_pairs = expanded
            else:
                self.sampler_model_pairs = base_pairs + expanded

        if not self.sampler_model_pairs:
            raise ValueError(
                "must have at least one sampler_model_pair (provide mutation_models or sampler_model_pairs)"
            )

        # 3. Auto-fill None model fields in sub-configs
        if self.init.diversity_model is None:
            self.init.diversity_model = self.paradigm_models[0]
        if self.init.variant_models is None:
            self.init.variant_models = list(self.mutation_models)

        if self.meta_advice.model is None:
            self.meta_advice.model = self.mutation_models[0]

        if self.punctuated_equilibrium.heavy_models is None:
            self.punctuated_equilibrium.heavy_models = list(self.paradigm_models)
        if self.punctuated_equilibrium.variant_models is None:
            self.punctuated_equilibrium.variant_models = list(self.mutation_models)

        # Light-model defaults for strategy summarisation + code repair.
        if self.strategy_log.summariser_model is None:
            self.strategy_log.summariser_model = self.mutation_models[0]
        if self.code_repair.model is None:
            self.code_repair.model = self.mutation_models[0]

        if self.prompt_opt.teacher_model is None:
            self.prompt_opt.teacher_model = self.paradigm_models[0]
        if self.prompt_opt.cache_dir is None and self.output_dir:
            self.prompt_opt.cache_dir = self.output_dir

        # 4. Auto-generate output_dir if not set
        if self.output_dir is None:
            self.output_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return self


class LeviResult(BaseModel):
    best_program: str
    best_score: float
    total_evaluations: int
    total_cost: float
    archive_size: int
    runtime_seconds: float
    score_history: Optional[list[float]] = None
    component_selector_stats: Optional[dict] = None
    pe_component_selector_stats: Optional[dict] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def best_prompt(self) -> str:
        return self.best_program

    @property
    def best_bundle(self) -> Optional[dict[str, str]]:
        """Return best prompt as a component dict when it's a bundle payload."""
        from ..prompts import PromptBundle

        if not self.best_program:
            return None
        if not PromptBundle.is_bundle_payload(self.best_program):
            return None
        return PromptBundle.from_serialized(self.best_program).as_dict()
