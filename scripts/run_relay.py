#!/usr/bin/env python3
"""Driver for RelayEvolve and its cheap/strong routing baselines.

Every method here runs on the same OpenEvolve backend (island MAP-Elites),
the same evaluator, the same prompt template, the same generation cap and the
same dollar cap — so the only thing that varies between rows of a results
table is *which model* produced each mutation, and *when*.

    python scripts/run_relay.py --method relayevolve \
        --benchmark-dir benchmarks/math/circle_packing \
        --iterations 300 --dollars 2 --workers 8 --seed 1

Methods:

    relayevolve    cheap multi-trajectory exploration → Relay-Gain handoff →
                   shared strong-model refinement
    all_cheap      cheap model throughout
    all_strong     strong model throughout
    fixed_switch   cheap prefix, then strong
    random         independent coin flip per generation
    bandit         two-armed UCB on realized best-so-far improvement
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from specevo_baselines.config import (  # noqa: E402
    LLMModelConfig,
    _parse_model_spec,
    _resolve_api_key_from_env,
    load_config,
)

METHOD_TO_SEARCH = {
    "relayevolve": "relayevolve",
    "all_cheap": "relay_all_cheap",
    "all_strong": "relay_all_strong",
    "fixed_switch": "relay_fixed_switch",
    "random": "relay_random",
    "bandit": "relay_bandit",
}

DEFAULT_CHEAP = "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
DEFAULT_STRONG = "openrouter/moonshotai/kimi-k2"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--method", required=True, choices=sorted(METHOD_TO_SEARCH))
    p.add_argument(
        "--benchmark-dir",
        required=True,
        help="Benchmark directory holding initial_program.py, config.yaml and an evaluator.",
    )
    p.add_argument("--cheap-model", default=DEFAULT_CHEAP, help=f"(default: {DEFAULT_CHEAP})")
    p.add_argument("--strong-model", default=DEFAULT_STRONG, help=f"(default: {DEFAULT_STRONG})")
    p.add_argument("--iterations", type=int, default=300, help="Generation cap N (default: 300)")
    p.add_argument(
        "--dollars",
        default="2",
        help="Total LLM spend cap in USD; the run stops gracefully once reached. "
        "0 or blank disables it. (default: 2)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Generations in flight at once (max_parallel_iterations). (default: 8)",
    )
    p.add_argument(
        "--eval-timeout",
        type=int,
        default=150,
        help="Per-candidate evaluation timeout, seconds. (default: 150)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="LLM attempts per generation. 1 (default) = no retry: a generation that "
        "produces an invalid program spends its slot and the search moves on, which "
        "is faster and keeps one generation == one model call.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default=None, help="Output directory.")
    p.add_argument("--checkpoint", default=None, help="Resume from this checkpoint directory.")
    p.add_argument("--log-level", default="INFO")

    # --- relay knobs ------------------------------------------------------
    p.add_argument(
        "--strong-reserve",
        type=float,
        default=None,
        help="Fraction of the dollar budget reserved for the strong model (0.85).",
    )
    p.add_argument("--block-size", type=int, default=None, help="Generations per block h (5).")
    p.add_argument("--max-trajectories", type=int, default=None)
    p.add_argument("--trajectory-horizon", type=int, default=None)
    p.add_argument("--bank-size", type=int, default=None, help="Relay bank / seed count k (8).")
    p.add_argument(
        "--relay-lambda",
        type=float,
        default=None,
        help="Quality vs coverage weight in F_C(S) (0.5).",
    )
    p.add_argument(
        "--epsilon-rel", type=float, default=None, help="Relay-Gain saturation threshold (0.02)."
    )
    p.add_argument(
        "--patience", type=int, default=None, help="Consecutive low-gain blocks before handoff (3)."
    )
    p.add_argument(
        "--curation",
        choices=["full", "quality", "diversity", "random"],
        default=None,
        help="Curation ablation: quality => lambda=1, diversity => lambda=0.",
    )
    p.add_argument(
        "--relay-control",
        choices=["full", "random", "no_stop", "random_no_stop"],
        default=None,
        help="Relay-mechanism ablation: replace Grow/Deepen with random choices "
        "and/or drop the Relay-Gain stopping rule.",
    )
    p.add_argument(
        "--switch-fraction",
        type=float,
        default=None,
        help="Fixed-switch handover point as a fraction of budget (0.5).",
    )
    p.add_argument(
        "--p-strong",
        type=float,
        default=None,
        help="Random baseline: P(strong) per generation (0.5).",
    )
    p.add_argument("--embedding-backend", choices=["hash", "api"], default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument(
        "--save-eval-code",
        action="store_true",
        help=(
            "Write every generation's source code to eval_code_log.jsonl and, at "
            "the end, eval_code_log.json — failed, unparseable and timed-out ones "
            "included."
        ),
    )
    p.add_argument(
        "--advanced-options",
        default=None,
        help="JSON of extra search.database overrides, e.g. '{\"ucb_c\":0.8}'.",
    )
    return p.parse_args()


def _parse_dollars(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "off", "0"}:
        return None
    amount = float(text)
    return amount if amount > 0 else None


def _resolve_evaluator(benchmark_dir: Path) -> Path:
    for candidate in (
        benchmark_dir / "evaluator.py",
        benchmark_dir / "evaluator" / "evaluator.py",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit(f"No evaluator found under {benchmark_dir}")


def _model_config(spec: str) -> LLMModelConfig:
    provider, model_name, api_base, env_vars = _parse_model_spec(spec)
    if api_base is None:
        raise SystemExit(f"Provider '{provider}' needs an explicit api_base (model '{spec}')")
    return LLMModelConfig(
        name=model_name,
        api_base=api_base,
        api_key=_resolve_api_key_from_env(env_vars),
    )


def _apply_overrides(db_config: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Push CLI knobs onto the search.database config; return what changed."""
    applied: Dict[str, Any] = {}

    def put(field: str, value: Any) -> None:
        if value is None:
            return
        setattr(db_config, field, value)
        applied[field] = value

    put("random_seed", args.seed)
    put("retry_times", args.retries)
    put("strong_reserve", args.strong_reserve)
    put("block_size", args.block_size)
    put("max_trajectories", args.max_trajectories)
    put("trajectory_horizon", args.trajectory_horizon)
    put("bank_size", args.bank_size)
    put("relay_lambda", args.relay_lambda)
    put("epsilon_rel", args.epsilon_rel)
    put("patience", args.patience)
    put("switch_fraction", args.switch_fraction)
    put("p_strong", args.p_strong)
    put("embedding_backend", args.embedding_backend)
    put("embedding_model", args.embedding_model)
    if args.save_eval_code:
        put("save_eval_code", True)

    if args.curation == "quality":
        put("relay_lambda", 1.0)
    elif args.curation == "diversity":
        put("relay_lambda", 0.0)
    elif args.curation == "random":
        put("curation_random", True)

    if args.relay_control in ("random", "random_no_stop"):
        put("random_allocation", True)
    if args.relay_control in ("no_stop", "random_no_stop"):
        put("disable_relay_stop", True)

    if args.advanced_options:
        try:
            extras = json.loads(args.advanced_options)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--advanced-options is not valid JSON: {exc}") from exc
        if not isinstance(extras, dict):
            raise SystemExit("--advanced-options must be a JSON object")
        for key, value in extras.items():
            put(key, value)

    return applied


async def _main_async() -> int:
    args = _parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_absolute():
        benchmark_dir = (REPO_ROOT / benchmark_dir).resolve()
    initial_program = benchmark_dir / "initial_program.py"
    config_path = benchmark_dir / "config.yaml"
    for required in (initial_program, config_path):
        if not required.exists():
            raise SystemExit(f"Missing {required}")
    evaluator = _resolve_evaluator(benchmark_dir)

    search_type = METHOD_TO_SEARCH[args.method]
    output_dir = Path(
        args.output
        or REPO_ROOT
        / "outputs"
        / "relay"
        / benchmark_dir.name
        / f"{args.method}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cost accounting must be armed before the first LLM call: the tracker
    # reads both variables from the environment on every call.
    os.environ.setdefault("SPECEVO_COST_LOG", str(output_dir / "cost_log.jsonl"))
    budget = _parse_dollars(args.dollars)
    if budget is not None:
        os.environ["SPECEVO_MAX_COST_USD"] = f"{budget}"

    random.seed(args.seed)

    config = load_config(str(config_path))
    config.search.type = search_type
    # Re-derive the database config so relay-only fields exist even when the
    # benchmark's YAML was written for another search type.
    from specevo_baselines.config import _DB_CONFIG_BY_TYPE

    db_cls = _DB_CONFIG_BY_TYPE[search_type]
    old_db = config.search.database
    new_db = db_cls()
    for field_name in ("db_path", "log_prompts"):
        setattr(new_db, field_name, getattr(old_db, field_name, getattr(new_db, field_name)))
    config.search.database = new_db
    applied = _apply_overrides(new_db, args)

    cheap = _model_config(args.cheap_model)
    strong = _model_config(args.strong_model)
    config.llm.api_base = cheap.api_base
    if cheap.api_key:
        config.llm.api_key = cheap.api_key
    config.llm.models = [cheap]
    config.llm.guide_models = [copy.deepcopy(strong)]
    config.llm.evaluator_models = [copy.deepcopy(cheap)]
    # __post_init__ already ran on the YAML's models; re-run the shared-param
    # propagation so timeout/max_tokens reach the two pools we just installed.
    config.llm.update_model_params(
        {
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
            "timeout": config.llm.timeout,
            "retries": config.llm.retries,
            "retry_delay": config.llm.retry_delay,
        }
    )

    config.max_parallel_iterations = max(1, args.workers)
    config.max_iterations = args.iterations
    config.evaluator.timeout = args.eval_timeout
    config.log_level = args.log_level
    config.monitor.enabled = False
    config.human_feedback_enabled = False

    run_meta = {
        "method": args.method,
        "search_type": search_type,
        "benchmark": (
            str(benchmark_dir.relative_to(REPO_ROOT))
            if str(benchmark_dir).startswith(str(REPO_ROOT))
            else str(benchmark_dir)
        ),
        "cheap_model": args.cheap_model,
        "strong_model": args.strong_model,
        "iterations": args.iterations,
        "dollars": budget,
        "workers": config.max_parallel_iterations,
        "eval_timeout": args.eval_timeout,
        "retries": args.retries,
        "seed": args.seed,
        "curation": args.curation,
        "relay_control": args.relay_control,
        "save_eval_code": bool(args.save_eval_code),
        "overrides": applied,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2, default=str))

    print("=" * 72)
    for key, value in run_meta.items():
        print(f"  {key:<14}: {value}")
    print(f"  output_dir    : {output_dir}")
    print("=" * 72, flush=True)

    from specevo_baselines import Runner

    runner = Runner(
        initial_program_path=str(initial_program),
        evaluation_file=str(evaluator),
        config=config,
        output_dir=str(output_dir),
    )
    best = await runner.run(iterations=args.iterations, checkpoint_path=args.checkpoint)

    print_finish_banner(args, output_dir, best, budget)
    return 0


STOP_REASONS = {
    "budget_exhausted": "dollar budget reached",
    "generation_cap": "generation budget spent",
    "generations_ended_early": "stopped before the generation cap",
    "interrupted": "interrupted (signal or shutdown request)",
}


def _hms(seconds: Optional[float]) -> str:
    if not seconds:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def print_finish_banner(args, output_dir: Path, best, budget: Optional[float]) -> None:
    """One closing report, identical in shape for every method.

    A run that stopped because it ran out of money used to look exactly like a
    run that finished its generations — and for the routing baselines the tail
    printed "handoff at generation None", which reads like a failure. This
    always says which of the two happened, and never prints a field that does
    not apply to the method that just ran.
    """
    summary = {}
    summary_path = output_dir / "relay_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}

    totals = summary.get("totals") or {}
    tiers = summary.get("llm_calls_by_tier") or {}
    reason_key = summary.get("stop_reason") or "unknown"
    reason = STOP_REASONS.get(reason_key, reason_key)

    spent = totals.get("total_cost_usd")
    if reason_key == "budget_exhausted" and spent is not None and budget:
        reason = f"dollar budget reached (${spent:.4f} of ${budget:.2f})"

    score = None
    if best is not None:
        score = best.metrics.get("test_combined_score", best.metrics.get("combined_score"))

    status = "RUN FINISHED" if best is not None else "RUN FINISHED — NO VALID PROGRAM"
    mark = "OK" if best is not None else "!!"
    title = (
        f" [{mark}] {status} — {args.method} on "
        f"{Path(args.benchmark_dir).name} (seed {args.seed})"
    )

    lines = [f" stopped because : {reason}"]
    if score is not None:
        kind = "test-mode" if "test_combined_score" in (best.metrics or {}) else "search-time"
        lines.append(f" best score      : {score:.6f}   ({kind})")
    else:
        lines.append(
            " best score      : none — every generation failed to produce a " "valid program"
        )
    used, asked = summary.get("iterations_used"), summary.get("requested_iterations")
    lines.append(f" generations     : {used} of {asked}" if asked else f" generations     : {used}")
    lines.append(
        f" llm calls       : cheap={tiers.get('cheap', 0)}  strong={tiers.get('strong', 0)}"
    )
    if spent is not None:
        cap = f" of ${budget:.2f}" if budget else " (no cap)"
        lines.append(f" cost            : ${spent:.4f}{cap}")
    lines.append(
        f" tokens          : in={totals.get('total_prompt_tokens', 0):,}  "
        f"out={totals.get('total_completion_tokens', 0):,}"
    )
    lines.append(f" wall clock      : {_hms(summary.get('wall_clock_s'))}")

    # Only the methods that actually hand over report a handoff.
    if summary.get("handoff_iteration") is not None:
        detail = summary.get("handoff_reason") or summary.get("switch_reason") or ""
        seeds = summary.get("seeds")
        extra = f", {len(seeds)} seeds" if seeds else ""
        lines.append(
            f" handoff         : generation {summary['handoff_iteration']}"
            f"{f' ({detail})' if detail else ''}{extra}"
        )
        if summary.get("cheap_iterations") is not None:
            lines.append(
                f" cheap / strong  : {summary['cheap_iterations']} / "
                f"{summary.get('strong_iterations')} generations"
            )
    lines.append(f" results in      : {output_dir}")

    width = max(len(title), max(len(line) for line in lines)) + 2
    bar = "=" * width
    print(f"\n{bar}\n{title}\n{'-' * width}")
    for line in lines:
        print(line)
    print(f"{bar}", flush=True)


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    sys.exit(main())
