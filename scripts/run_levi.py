#!/usr/bin/env python3
"""Generic driver for any self-contained LEVI example.

Every example under ``tasks/<name>/`` follows the same shape: there
is a ``problem.py`` that exports ``PROBLEM_DESCRIPTION``,
``FUNCTION_SIGNATURE``, ``score_fn``, and optionally ``SEED_PROGRAM`` /
``INPUTS`` / ``get_lazy_inputs`` / ``BEHAVIOR_SCORE_KEYS`` (list/tuple of keys
present in ``score_fn`` dicts for ``BehaviorConfig.score_keys``). This script
picks one of those directories
and runs LEVI on it, exposing every budget and pipeline knob as a flag so
the GitHub Actions reusable workflow can wire them straight from inputs.

The trick that makes pickling work (LEVI ships ``score_fn`` to a process
pool): we insert the example directory on ``sys.path`` and then
``importlib.import_module("problem")``. The module name on disk is
``problem`` — not a synthetic ``spec_from_file_location`` name — so worker
processes can re-import it and unpickle the closure cleanly.

Usage::

    uv run python scripts/run_levi.py \\
        --task-dir tasks/circle_packing \\
        --evals 100
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _opt_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() == "none":
        return None
    return int(value)


def _opt_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() == "none":
        return None
    return float(value)


def _parse_csv_keys(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolve_behavior_score_keys(cli_value: str | None, problem: object) -> list[str] | None:
    """CLI overrides problem.BEHAVIOR_SCORE_KEYS; omit flag → use problem constant if defined."""
    if cli_value is not None:
        keys = _parse_csv_keys(cli_value)
        return keys if keys else None
    raw = getattr(problem, "BEHAVIOR_SCORE_KEYS", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        keys = _parse_csv_keys(raw)
        return keys if keys else None
    if isinstance(raw, (list, tuple)):
        keys = [str(k).strip() for k in raw if str(k).strip()]
        return keys if keys else None
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task-dir",
        required=True,
        help="Path to a LEVI example directory (e.g. tasks/circle_packing).",
    )
    p.add_argument(
        "--problem-module",
        default="problem",
        help="Name of the Python module to import from --task-dir (default: problem).",
    )
    p.add_argument(
        "--small-model",
        default="openrouter/qwen/qwen3-30b-a3b-instruct-2507",
        help="OpenRouter id of the cheap mutation model.",
    )
    p.add_argument(
        "--large-model",
        default="openrouter/openai/gpt-5",
        help="OpenRouter id of the stronger paradigm-shift model.",
    )
    # All budgets are optional strings — blank / "none" disables that knob.
    p.add_argument("--evals", default="100", help="Max evaluations (default: 100; '' to disable).")
    p.add_argument("--dollars", default="", help="Max USD spend (default: unset).")
    p.add_argument("--seconds", default="", help="Wall-clock cap in seconds (default: unset).")
    p.add_argument("--target-score", default="", help="Stop early at this score (default: unset).")
    p.add_argument("--workers", default="4", help="Concurrent LLM workers (default: 4).")
    p.add_argument(
        "--eval-processes", default="4", help="Concurrent evaluator processes (default: 4)."
    )
    p.add_argument(
        "--eval-timeout", default="600", help="Per-candidate evaluation timeout (seconds)."
    )
    p.add_argument(
        "--init-diverse-seeds",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Init phase: how many algorithmically diverse seeds to aim for (Levi default 4; "
            "often +1 when no seed_program). Omit to use Levi defaults."
        ),
    )
    p.add_argument(
        "--init-variants-per-seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Init phase: variants per seed after diversity phase (Levi default 20). "
            "Total variant LLM calls ≈ this × (number of diverse seeds). Omit for Levi default."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to drop snapshot.json + summary.json (default: outputs/levi/<example>/<ts>).",
    )
    p.add_argument(
        "--n-centroids",
        type=int,
        default=None,
        metavar="N",
        help=(
            "CVT-MAP-Elites archive size (number of behavior cells). "
            "Levi default is 50; omit to keep default."
        ),
    )
    p.add_argument(
        "--behavior-score-keys",
        default=None,
        metavar="KEYS",
        help=(
            "Comma-separated keys appended to BehaviorExtractor via BehaviorConfig.score_keys "
            "(must be numeric fields on score_fn results). Overrides problem.BEHAVIOR_SCORE_KEYS when set."
        ),
    )
    p.add_argument(
        "--prompt-bank",
        action="store_true",
        help=(
            "Enable the mutation prompt-bank + temperature-bank (joint bandit). "
            "Off by default; when on, every (sampler, model) is cross-product expanded "
            "with every (prompt_id, llm_temperature) pair."
        ),
    )
    p.add_argument(
        "--prompt-bank-prompts-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON list of {id, text} prompts for the bank. "
            "Default: tasks/mutation_prompts.json (shared across examples)."
        ),
    )
    p.add_argument(
        "--prompt-bank-temperatures-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON list of floats for the temperature bank. "
            "Default: tasks/mutation_temperatures.json (shared across examples)."
        ),
    )

    # ------------------------------------------------------------------
    # Punctuated Equilibrium overrides — handy for ablation runs.
    # The heavy model writes a full paradigm-shift solution every interval
    # (early/mid/late prompt template chosen by stagnation depth). Light
    # variant models produce nearby variants. The strategy-log layer below
    # summarises each PE so the next heavy prompt knows what was tried.
    # ------------------------------------------------------------------
    p.add_argument(
        "--pe-interval",
        type=int,
        default=None,
        metavar="N",
        help="Trigger PE every N evaluations (Levi default 10).",
    )
    p.add_argument(
        "--pe-n-clusters",
        type=int,
        default=None,
        metavar="N",
        help="Number of behavioural clusters when picking PE anchors (default 3).",
    )
    p.add_argument(
        "--pe-n-variants",
        type=int,
        default=None,
        metavar="N",
        help="Light-model variants of the heavy paradigm shift per PE event (default 3).",
    )
    p.add_argument(
        "--no-pe",
        action="store_true",
        help="Disable Punctuated Equilibrium entirely.",
    )

    # ------------------------------------------------------------------
    # Strategy Log — one-sentence summaries of past PEs, fed into the
    # next heavy prompt so the model knows what has already been tried.
    # ------------------------------------------------------------------
    p.add_argument(
        "--no-strategy-log",
        action="store_true",
        help=(
            "Disable the post-PE light-model strategy summariser. Heavy "
            "prompts will no longer see the 'Strategy Log' section."
        ),
    )
    p.add_argument(
        "--strategy-log-entries",
        type=int,
        default=None,
        metavar="N",
        help=(
            "How many most-recent strategy records are rendered into the "
            "heavy prompt (default 8)."
        ),
    )

    # ------------------------------------------------------------------
    # Code Error Repair — one-shot light-model fix for broken candidates.
    # ------------------------------------------------------------------
    p.add_argument(
        "--no-code-repair",
        action="store_true",
        help="Disable the code-repair sub-pipeline.",
    )
    p.add_argument(
        "--code-repair-every-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Fire one repair attempt every N main-loop offspring (default 8). "
            "Smaller → more aggressive repair, but more light-model spend."
        ),
    )
    p.add_argument(
        "--code-repair-max",
        type=int,
        default=None,
        metavar="N",
        help="Hard cap on total repair attempts per run (default 100).",
    )
    p.add_argument(
        "--code-repair-beta",
        type=float,
        default=None,
        metavar="B",
        help=(
            "Zipfian exponent used to pick which broken candidate to repair "
            "(rank by parent score, higher β → favour stronger parents)."
        ),
    )
    p.add_argument(
        "--code-repair-buffer",
        type=int,
        default=None,
        metavar="N",
        help="Bounded error-buffer size (default 64).",
    )

    # ------------------------------------------------------------------
    # Adaptive Island Expansion — AdaEvolve-style archive growth.
    # When a PE candidate fails standard admission under high stagnation,
    # we open a NEW cell at the candidate's own behaviour vector instead
    # of replacing the incumbent. Unifies the previous rescue + adaptive-
    # CVT mechanisms into a single trigger.
    # ------------------------------------------------------------------
    p.add_argument(
        "--no-adaptive-island",
        action="store_true",
        help="Disable Adaptive Island Expansion (PE candidates that fail to beat the incumbent are simply discarded).",
    )
    p.add_argument(
        "--island-stagnation",
        type=float,
        default=None,
        metavar="S",
        help="Minimum stagnation depth s(t) at which island expansion may fire (default 0.7).",
    )
    p.add_argument(
        "--island-max",
        type=int,
        default=None,
        metavar="N",
        help="Maximum island expansions per run (default 16).",
    )
    p.add_argument(
        "--island-max-centroids",
        type=int,
        default=None,
        metavar="N",
        help="Hard ceiling on total centroids regardless of expansions (default 200).",
    )

    # ------------------------------------------------------------------
    # PPS — Posterior-Plateau Stagnation tuning (SAL.tau).
    # ------------------------------------------------------------------
    p.add_argument(
        "--sal-tau",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Plateau length τ (in evaluations) at which the PPS plateau "
            "term saturates to 1.0. Default 80. Smaller τ → PE / Hard-PE "
            "fire sooner."
        ),
    )
    p.add_argument(
        "--no-sal",
        action="store_true",
        help="Disable SAL/PPS entirely. Producer ignores stagnation; bandit reverts to roulette.",
    )

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 2

    task_dir = Path(args.task_dir).resolve()
    if not task_dir.is_dir():
        print(f"ERROR: --task-dir {task_dir} is not a directory.", file=sys.stderr)
        return 2

    # Make `problem` importable by its real on-disk name. Workers will
    # inherit sys.path (fork on Linux, explicit transfer on spawn) and
    # therefore be able to unpickle score_fn.
    sys.path.insert(0, str(task_dir))
    # Also make the vendored LEVI tree importable.
    sys.path.insert(0, str(REPO_ROOT / "levi"))

    problem = importlib.import_module(args.problem_module)
    import specevo  # type: ignore

    evals = _opt_int(args.evals)
    dollars = _opt_float(args.dollars)
    seconds = _opt_float(args.seconds)
    target_score = _opt_float(args.target_score)
    if evals is None and dollars is None and seconds is None:
        print(
            "ERROR: at least one of --evals / --dollars / --seconds must be set.",
            file=sys.stderr,
        )
        return 2

    workers = _opt_int(args.workers) or 4
    eval_processes = _opt_int(args.eval_processes) or 4
    eval_timeout = _opt_float(args.eval_timeout) or 600.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "outputs" / "levi" / task_dir.name / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = None
    if hasattr(problem, "INPUTS"):
        inputs = problem.INPUTS
    elif hasattr(problem, "get_lazy_inputs"):
        inputs = problem.get_lazy_inputs()

    print(f"[levi] example       = {task_dir}")
    print(f"[levi] small model   = {args.small_model}")
    print(f"[levi] large model   = {args.large_model}")
    print(f"[levi] budget evals  = {evals}")
    print(f"[levi] budget $      = {dollars}")
    print(f"[levi] budget secs   = {seconds}")
    print(f"[levi] target_score  = {target_score}")
    print(f"[levi] output_dir    = {output_dir}")
    if args.n_centroids is not None:
        print(f"[levi] n_centroids   = {args.n_centroids}")

    behavior_score_keys = _resolve_behavior_score_keys(args.behavior_score_keys, problem)
    if behavior_score_keys is not None:
        print(f"[levi] behavior score_keys = {behavior_score_keys}")

    init_updates: dict = {}
    if args.init_diverse_seeds is not None:
        init_updates["n_diverse_seeds"] = args.init_diverse_seeds
    if args.init_variants_per_seed is not None:
        init_updates["n_variants_per_seed"] = args.init_variants_per_seed
    base_init = levi.InitConfig()
    effective_init = base_init.model_copy(update=init_updates) if init_updates else base_init
    if init_updates:
        evolve_init = effective_init
        print(f"[levi] init diverse seeds      = {effective_init.n_diverse_seeds}")
        print(f"[levi] init variants per seed  = {effective_init.n_variants_per_seed}")
    else:
        evolve_init = None

    # --- SAL / PPS ---
    sal_kwargs: dict = {"enabled": not args.no_sal}
    if args.sal_tau is not None:
        sal_kwargs["tau"] = args.sal_tau

    # --- Punctuated Equilibrium ---
    pe_kwargs: dict = {"enabled": not args.no_pe}
    if args.pe_interval is not None:
        pe_kwargs["interval"] = args.pe_interval
    if args.pe_n_clusters is not None:
        pe_kwargs["n_clusters"] = args.pe_n_clusters
    if args.pe_n_variants is not None:
        pe_kwargs["n_variants"] = args.pe_n_variants

    # --- Strategy Log ---
    strategy_kwargs: dict = {"enabled": not args.no_strategy_log}
    if args.strategy_log_entries is not None:
        strategy_kwargs["max_entries"] = args.strategy_log_entries

    # --- Code Error Repair ---
    repair_kwargs: dict = {"enabled": not args.no_code_repair}
    if args.code_repair_every_n is not None:
        repair_kwargs["repair_every_n"] = args.code_repair_every_n
    if args.code_repair_max is not None:
        repair_kwargs["max_per_run"] = args.code_repair_max
    if args.code_repair_beta is not None:
        repair_kwargs["beta"] = args.code_repair_beta
    if args.code_repair_buffer is not None:
        repair_kwargs["buffer_size"] = args.code_repair_buffer

    # --- Adaptive Island Expansion ---
    island_kwargs: dict = {"enabled": not args.no_adaptive_island}
    if args.island_stagnation is not None:
        island_kwargs["stagnation_threshold"] = args.island_stagnation
    if args.island_max is not None:
        island_kwargs["max_per_run"] = args.island_max
    if args.island_max_centroids is not None:
        island_kwargs["max_total_centroids"] = args.island_max_centroids

    print(f"[levi] SAL/PPS = {'on' if not args.no_sal else 'OFF (ablation)'}")
    if not args.no_sal and args.sal_tau is not None:
        print(f"[levi]   τ (plateau length)  = {args.sal_tau}")
    print(f"[levi] Punctuated Equilibrium = {'on' if not args.no_pe else 'OFF'}")
    print(f"[levi] Strategy log = {'on' if not args.no_strategy_log else 'OFF'}")
    print(f"[levi] Code repair  = {'on' if not args.no_code_repair else 'OFF'}")
    print(f"[levi] Adaptive Island = {'on' if not args.no_adaptive_island else 'OFF'}")

    evolve_kw: dict = {
        "problem_description": problem.PROBLEM_DESCRIPTION,
        "function_signature": problem.FUNCTION_SIGNATURE,
        "seed_program": getattr(problem, "SEED_PROGRAM", None),
        "score_fn": problem.score_fn,
        "inputs": inputs,
        "paradigm_model": args.large_model,
        "mutation_model": args.small_model,
        "budget_evals": evals,
        "budget_dollars": dollars,
        "budget_seconds": seconds,
        "target_score": target_score,
        "pipeline": levi.PipelineConfig(
            n_llm_workers=workers,
            n_eval_processes=eval_processes,
            eval_timeout=eval_timeout,
        ),
        "output_dir": str(output_dir),
        "sal": levi.SalConfig(**sal_kwargs),
        "punctuated_equilibrium": levi.PunctuatedEquilibriumConfig(**pe_kwargs),
        "strategy_log": levi.StrategyLogConfig(**strategy_kwargs),
        "code_repair": levi.CodeRepairConfig(**repair_kwargs),
        "adaptive_island": levi.AdaptiveIslandConfig(**island_kwargs),
    }
    if evolve_init is not None:
        evolve_kw["init"] = evolve_init
    if args.n_centroids is not None:
        evolve_kw["cvt"] = levi.CVTConfig(n_centroids=args.n_centroids)
    if behavior_score_keys is not None:
        evolve_kw["behavior"] = levi.BehaviorConfig(score_keys=behavior_score_keys)

    if args.prompt_bank:
        # Shared defaults live one level *above* each example dir
        # (tasks/mutation_prompts.json + mutation_temperatures.json)
        # so they're not duplicated per example.
        shared_dir = REPO_ROOT / "levi" / "examples"
        prompts_file = args.prompt_bank_prompts_file
        if prompts_file is None:
            default_prompts = shared_dir / "mutation_prompts.json"
            prompts_file = str(default_prompts) if default_prompts.is_file() else None
        temperatures_file = args.prompt_bank_temperatures_file
        if temperatures_file is None:
            default_temps = shared_dir / "mutation_temperatures.json"
            temperatures_file = str(default_temps) if default_temps.is_file() else None
        if not prompts_file or not temperatures_file:
            print(
                "ERROR: --prompt-bank requires both a prompts file and a temperatures file. "
                "Provide --prompt-bank-prompts-file / --prompt-bank-temperatures-file or place "
                f"mutation_prompts.json and mutation_temperatures.json under {shared_dir}.",
                file=sys.stderr,
            )
            return 2
        print(f"[levi] prompt_bank     = enabled")
        print(f"[levi]   prompts_file        = {prompts_file}")
        print(f"[levi]   temperatures_file   = {temperatures_file}")
        evolve_kw["prompt_bank"] = levi.PromptBankConfig(
            enabled=True,
            prompts_file=prompts_file,
            temperatures_file=temperatures_file,
        )
    else:
        evolve_kw["prompt_bank"] = levi.PromptBankConfig(enabled=False)

    result = specevo.evolve_code(**evolve_kw)

    summary = {
        "task_dir": str(task_dir),
        "best_score": result.best_score,
        "total_evaluations": result.total_evaluations,
        "total_cost": result.total_cost,
        "archive_size": result.archive_size,
        "runtime_seconds": result.runtime_seconds,
        "budget": {
            "evaluations": evals,
            "dollars": dollars,
            "seconds": seconds,
            "target_score": target_score,
        },
        "small_model": args.small_model,
        "large_model": args.large_model,
        "n_centroids": args.n_centroids,
        "behavior_score_keys": behavior_score_keys,
        "init": {
            "n_diverse_seeds": effective_init.n_diverse_seeds,
            "n_variants_per_seed": effective_init.n_variants_per_seed,
        },
        "sal": {
            "enabled": not args.no_sal,
            "tau": args.sal_tau,
        },
        "punctuated_equilibrium": {
            "enabled": not args.no_pe,
            "interval": args.pe_interval,
            "n_clusters": args.pe_n_clusters,
            "n_variants": args.pe_n_variants,
        },
        "strategy_log": {
            "enabled": not args.no_strategy_log,
            "max_entries": args.strategy_log_entries,
        },
        "code_repair": {
            "enabled": not args.no_code_repair,
            "repair_every_n": args.code_repair_every_n,
            "max_per_run": args.code_repair_max,
            "beta": args.code_repair_beta,
            "buffer_size": args.code_repair_buffer,
        },
        "adaptive_island": {
            "enabled": not args.no_adaptive_island,
            "stagnation_threshold": args.island_stagnation,
            "max_per_run": args.island_max,
            "max_total_centroids": args.island_max_centroids,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "best_program.py").write_text(result.best_program or "")

    print()
    print(f"Best score        : {result.best_score:.6f}")
    print(f"Evaluations used  : {result.total_evaluations}")
    print(f"Total cost        : ${result.total_cost:.4f}")
    print(f"Archive size      : {result.archive_size}")
    print(f"Runtime           : {result.runtime_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
