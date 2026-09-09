#!/usr/bin/env python3
"""Generic driver for running SpecEvo on any self-contained SpecEvo task.

Mirrors ``scripts/run_levi.py`` but invokes :func:`specevo.run_specevo`.
Every example under ``tasks/<name>/`` exporting
``PROBLEM_DESCRIPTION``, ``FUNCTION_SIGNATURE``, ``score_fn`` (and
optionally ``SEED_PROGRAM`` / ``INPUTS``) plugs in unchanged.

Usage::

    uv run python scripts/run_specevo.py \\
        --task-dir tasks/smoke_demo \\
        --evals 50

The script exposes only the few knobs that change run-to-run plus the
three paper-facing ablation toggles (--ast-only, --emb-only,
--static-cells). Lower-level knobs (e.g. number of cells, recluster
cadence, β bounds) are reachable through ``SpecEvoConfig`` if needed.
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

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _opt_int(value: str | None) -> int | None:
    if value is None:
        return None
    s = value.strip()
    if not s or s.lower() == "none":
        return None
    return int(s)


def _opt_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = value.strip()
    if not s or s.lower() == "none":
        return None
    return float(s)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task-dir",
        required=True,
        help="Path to a SpecEvo task directory (e.g. tasks/smoke_demo).",
    )
    p.add_argument(
        "--problem-module",
        default="problem",
        help="Name of the Python module to import from --task-dir (default: problem).",
    )

    # Models
    p.add_argument(
        "--speculator-model",
        default="openrouter/qwen/qwen3-30b-a3b-instruct-2507",
        help="OpenRouter id of the small mutation model.",
    )
    p.add_argument(
        "--navigator-model",
        default="openrouter/openai/gpt-5",
        help="OpenRouter id of the frontier model used for Navigator interventions.",
    )
    p.add_argument(
        "--embedding-model",
        default="openrouter/openai/text-embedding-3-small",
        help="OpenRouter id of the description-embedding model.",
    )

    # Budget
    p.add_argument("--evals", default="", help="Max evaluations (default: unset).")
    p.add_argument("--dollars", default="", help="Max USD spend (default: unset).")
    p.add_argument("--seconds", default="", help="Wall-clock cap in seconds (default: unset).")
    p.add_argument("--target-score", default="", help="Stop early at this score (default: unset).")

    # Concurrency
    p.add_argument("--workers", default="4", help="Concurrent LLM workers (default: 4).")
    p.add_argument("--eval-processes", default="4", help="Concurrent evaluator processes (default: 4).")
    p.add_argument("--eval-timeout", default="600", help="Per-candidate evaluation timeout in seconds.")
    p.add_argument(
        "--save-eval-code",
        action="store_true",
        help=(
            "Write eval_code_log.jsonl (and eval_code_log.json at the end): the "
            "source of every candidate the run produced, including the ones that "
            "failed to parse, raised, scored invalid or timed out."
        ),
    )

    # SpecEvo knobs
    p.add_argument(
        "--navigator-interval", default="50",
        help="Frontier Navigator intervention cadence (every N evaluations; default 50).",
    )
    p.add_argument(
        "--n-diverse-seeds", type=int, default=5, metavar="N",
        help="Phase-1 diverse seeds (sequential, frontier model). Default 5.",
    )
    p.add_argument(
        "--n-variants-per-seed", type=int, default=20, metavar="N",
        help="Phase-2 variants per seed (parallel, mutation model). Default 20.",
    )
    p.add_argument(
        "--n-navigator-variants", type=int, default=4, metavar="K",
        help="Navigator intervention fanout (parallel mutation variants of the new seed).",
    )
    p.add_argument(
        "--navigator-n-anchors", type=int, default=None, metavar="N",
        help="Cell representatives sent to the frontier each Navigator intervention (default 4).",
    )
    p.add_argument(
        "--navigator-n-inspirations", type=int, default=None, metavar="N",
        help="Description-only inspirations alongside the anchors (default 5).",
    )
    p.add_argument(
        "--no-advisor", action="store_true",
        help="Disable the SpecEvo's lessons-learnt advisor.",
    )
    p.add_argument(
        "--advisor-interval", type=int, default=50, metavar="N",
        help="Refresh advisor every N evaluations (default: 50).",
    )
    p.add_argument(
        "--advisor-mode", choices=("rich", "errors_only"), default="rich",
        help="`rich` (default) feeds the advisor top-K descriptions + recent "
        "admits split into IMPROVING / SATURATED buckets + typed error "
        "taxonomy. `errors_only` is the paper ablation — drops the "
        "success-side signals.",
    )
    p.add_argument(
        "--advisor-inject-p", type=float, default=None, metavar="P",
        help="Probability that any given mutate / crossover prompt is "
        "prefixed with the current advice block (default 0.35).",
    )

    # ------------------------------------------------------------------
    # Targeted-mutate analyzer (Đề xuất 1) — review of top-ranked parents.
    # ------------------------------------------------------------------
    p.add_argument(
        "--no-targeted-mutate", action="store_true",
        help="Disable the LLM-generated parent analysis + TARGETED_MUTATE_PROMPT.",
    )
    p.add_argument(
        "--analyzer-interval", type=int, default=None, metavar="N",
        help="Refresh cached parent analyses every N evaluations (default 30).",
    )
    p.add_argument(
        "--analyzer-top-k", type=int, default=None, metavar="K",
        help="How many top-ranked programs to analyse each refresh (default 3).",
    )
    p.add_argument(
        "--p-targeted-mutate", type=float, default=None, metavar="P",
        help="Probability of using TARGETED_MUTATE_PROMPT when an analysis is "
        "cached for the chosen parent (default 0.5).",
    )

    # ------------------------------------------------------------------
    # Operator-mix ablation (paper).
    # ------------------------------------------------------------------
    p.add_argument(
        "--no-crossover", action="store_true",
        help="Ablation: drop crossover entirely (sets p_crossover=0.0).",
    )
    p.add_argument(
        "--p-crossover", type=float, default=None, metavar="P",
        help="Override crossover probability (default 0.35).",
    )

    # ------------------------------------------------------------------
    # Operator-prompt ablation (A6) — collapse the mutate / crossover
    # template repertoire down to a single simplest prompt each.
    # ------------------------------------------------------------------
    p.add_argument(
        "--single-prompt-operators", action="store_true",
        help="Ablation A6: stop sampling mutate / crossover templates at "
        "random; always use the single simplest template per operator "
        "(general improvement for mutate, structural hybrid for "
        "crossover).",
    )

    # ------------------------------------------------------------------
    # Navigator-prompt ablation (A8) — keep the Navigator intervention loop ON
    # but force a single Navigator mode instead of routing across three.
    # ------------------------------------------------------------------
    p.add_argument(
        "--navigator-force-mode", choices=("synthesis", "surgical", "reframe"),
        default=None,
        help="Ablation A8: keep the frontier Navigator intervention loop enabled "
        "but force every shift to use this single mode (default: adaptive "
        "three-mode routing by stagnation).",
    )

    # ------------------------------------------------------------------
    # Three-mode Navigator intervention thresholds (Đề xuất 8).
    # ------------------------------------------------------------------
    p.add_argument(
        "--navigator-synthesis-max-stagnation", type=float, default=None, metavar="S",
        help="At or below this stagnation level the shift uses synthesis mode (default 0.4).",
    )
    p.add_argument(
        "--navigator-surgical-max-stagnation", type=float, default=None, metavar="S",
        help="Below this stagnation level (and above synthesis cap) the shift "
        "uses surgical mode (default 0.7); above this it flips to the "
        "'new Navigator' (shift) mode.",
    )
    p.add_argument(
        "--navigator-synthesis-n-anchors", type=int, default=None, metavar="N",
        help="Number of anchors surfaced in synthesis mode (default 3).",
    )
    p.add_argument(
        "--navigator-reframe-n-anchors", type=int, default=None, metavar="N",
        help="Number of anchors surfaced in shift mode (default 2).",
    )
    p.add_argument(
        "--navigator-surgical-n-inspirations", type=int, default=None, metavar="N",
        help="Description-only inspirations passed to surgical mode (default 5).",
    )

    # ------------------------------------------------------------------
    # Error-rate ablation instrumentation. All off by default — omitting
    # these flags leaves a run byte-identical to before.
    # ------------------------------------------------------------------
    p.add_argument(
        "--post-init-evals", type=int, default=None, metavar="N",
        help="Stop after N evaluations counted from the END of the init "
        "phase (i.e. N iterations of the evolutionary main loop). Unlike "
        "--evals, the ~105 bootstrap evaluations do not eat into it.",
    )
    p.add_argument(
        "--error-rate-interval", type=int, default=0, metavar="N",
        help="Measure the error rate every N post-init evaluations. Every "
        "failed attempt counts except a failed LLM call (which produced no "
        "candidate and is excluded from the ratio). The per-window table is "
        "printed once at the end of the run and written to "
        "error_rate_report.json. 0 (default) = off.",
    )
    p.add_argument(
        "--checkpoint-population", action="store_true",
        help="With --error-rate-interval: dump the full population (all "
        "code + descriptions + scores) to checkpoints/checkpoint_<NN>.json "
        "at every window close.",
    )
    p.add_argument(
        "--align-advisor-post-init", action="store_true",
        help="Restart the advisor cadence clock at the end of the init "
        "phase (as navigator-interval already does) so advisor cycle k and "
        "error-rate window k share an origin.",
    )

    # ------------------------------------------------------------------
    # Ablation toggles (the three paper-facing knobs)
    # ------------------------------------------------------------------
    p.add_argument(
        "--ast-only", action="store_true",
        help="Ablation A2: use AST features only for behavior signature "
        "(disable the description-embedding half of the hybrid).",
    )
    p.add_argument(
        "--emb-only", action="store_true",
        help="Ablation A1: use description embedding only for behavior "
        "signature (disable the AST half of the hybrid).",
    )
    p.add_argument(
        "--static-cells", action="store_true",
        help="Ablation A3: fit KMeans once and freeze cells thereafter "
        "(disable adaptive re-clustering).",
    )

    # Archive low-level overrides (rarely needed; exposed for sweeps)
    p.add_argument(
        "--n-cells", type=int, default=None, metavar="N",
        help="Target number of cells in the archive (default 32).",
    )
    p.add_argument(
        "--recluster-every", type=int, default=None, metavar="N",
        help="Re-fit KMeans every N admits (default 30).",
    )
    p.add_argument(
        "--embedding-dim", type=int, default=None, metavar="N",
        help="PCA target dimension for the description-embedding half (default 8).",
    )

    p.add_argument(
        "--output-dir", default=None,
        help="Where to drop snapshot.json + summary.json (default: outputs/specevo/<example>/<ts>).",
    )

    return p.parse_args()


def _load_repo_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)


def _ensure_openrouter_env() -> None:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key.startswith("sk-or-") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = key


def main() -> int:
    args = _parse_args()
    _load_repo_env()
    _ensure_openrouter_env()

    task_dir = (REPO_ROOT / args.task_dir).resolve() if not Path(args.task_dir).is_absolute() else Path(args.task_dir)
    if not task_dir.is_dir():
        print(f"ERROR: --task-dir does not exist: {task_dir}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(task_dir))
    problem = importlib.import_module(args.problem_module)

    try:
        problem_description = problem.PROBLEM_DESCRIPTION
        function_signature = problem.FUNCTION_SIGNATURE
        score_fn = problem.score_fn
    except AttributeError as e:
        print(f"ERROR: example module is missing required attribute: {e}", file=sys.stderr)
        return 2

    seed_program = getattr(problem, "SEED_PROGRAM", None)
    inputs = getattr(problem, "INPUTS", None)
    # Some examples (e.g. ADRS/eplb) keep INPUTS = None as a placeholder and
    # load the real inputs lazily via get_lazy_inputs()/get_inputs() — the
    # run.py driver calls that explicitly. Mirror it here so SpecEvo receives
    # the actual inputs instead of None (which would score the seed against
    # an empty list and divide by zero).
    if inputs is None:
        loader = getattr(problem, "get_lazy_inputs", None) or getattr(problem, "get_inputs", None)
        if callable(loader):
            inputs = loader()

    evals = _opt_int(args.evals)
    dollars = _opt_float(args.dollars)
    seconds = _opt_float(args.seconds)
    target_score = _opt_float(args.target_score)
    workers = int(args.workers)
    eval_processes = int(args.eval_processes)
    eval_timeout = float(args.eval_timeout)
    navigator_interval = _opt_int(args.navigator_interval) or 50

    if args.output_dir:
        out_arg = Path(args.output_dir)
        output_dir = out_arg if out_arg.is_absolute() else (REPO_ROOT / out_arg).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = REPO_ROOT / "outputs" / "specevo" / task_dir.name / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check on ablation flags.
    if args.ast_only and args.emb_only:
        print("ERROR: --ast-only and --emb-only are mutually exclusive", file=sys.stderr)
        return 2

    overrides: dict = {}
    from specevo.simple import ArchiveConfig

    arch_kwargs = {}
    if args.ast_only:
        arch_kwargs["use_embedding"] = False
    if args.emb_only:
        arch_kwargs["use_ast"] = False
    if args.static_cells:
        arch_kwargs["adaptive_recluster"] = False
    if args.n_cells is not None:
        arch_kwargs["n_cells"] = args.n_cells
    if args.recluster_every is not None:
        arch_kwargs["recluster_every"] = args.recluster_every
    if args.embedding_dim is not None:
        arch_kwargs["embedding_dim"] = args.embedding_dim
    if arch_kwargs:
        overrides["archive_config"] = ArchiveConfig(**arch_kwargs)

    if args.navigator_n_anchors is not None:
        overrides["navigator_n_anchors"] = args.navigator_n_anchors
    if args.navigator_n_inspirations is not None:
        overrides["navigator_n_inspirations"] = args.navigator_n_inspirations

    if args.no_targeted_mutate:
        overrides["enable_targeted_mutate"] = False
    if args.analyzer_interval is not None:
        overrides["analyzer_interval"] = args.analyzer_interval
    if args.analyzer_top_k is not None:
        overrides["analyzer_top_k"] = args.analyzer_top_k
    if args.p_targeted_mutate is not None:
        overrides["p_targeted_mutate"] = args.p_targeted_mutate

    overrides["advisor_mode"] = args.advisor_mode
    if args.advisor_inject_p is not None:
        overrides["advisor_inject_p"] = args.advisor_inject_p
    if args.no_crossover:
        overrides["p_crossover"] = 0.0
    elif args.p_crossover is not None:
        overrides["p_crossover"] = args.p_crossover

    if args.post_init_evals is not None:
        overrides["post_init_budget_evals"] = args.post_init_evals
    if args.error_rate_interval > 0:
        overrides["ablation_window_evals"] = args.error_rate_interval
        overrides["ablation_checkpoint_population"] = args.checkpoint_population
    if args.align_advisor_post_init:
        overrides["align_advisor_to_post_init"] = True

    if args.save_eval_code:
        overrides["save_eval_code"] = True

    if args.single_prompt_operators:
        overrides["single_prompt_operators"] = True
    if args.navigator_force_mode is not None:
        overrides["navigator_force_mode"] = args.navigator_force_mode

    if args.navigator_synthesis_max_stagnation is not None:
        overrides["navigator_synthesis_max_stagnation"] = args.navigator_synthesis_max_stagnation
    if args.navigator_surgical_max_stagnation is not None:
        overrides["navigator_surgical_max_stagnation"] = args.navigator_surgical_max_stagnation
    if args.navigator_synthesis_n_anchors is not None:
        overrides["navigator_synthesis_n_anchors"] = args.navigator_synthesis_n_anchors
    if args.navigator_reframe_n_anchors is not None:
        overrides["navigator_reframe_n_anchors"] = args.navigator_reframe_n_anchors
    if args.navigator_surgical_n_inspirations is not None:
        overrides["navigator_surgical_n_inspirations"] = args.navigator_surgical_n_inspirations

    import specevo

    print(f"[specevo] task_dir       = {task_dir}")
    print(f"[specevo] speculator_model    = {args.speculator_model}")
    print(f"[specevo] navigator_model    = {args.navigator_model}")
    print(f"[specevo] embedding_model   = {args.embedding_model}")
    print(f"[specevo] budget evals      = {evals}")
    print(f"[specevo] budget dollars    = {dollars}")
    print(f"[specevo] budget seconds    = {seconds}")
    print(f"[specevo] navigator_interval  = {navigator_interval}")
    print(f"[specevo] n_diverse_seeds   = {args.n_diverse_seeds}")
    print(f"[specevo] n_variants/seed   = {args.n_variants_per_seed}")
    print(f"[specevo] n_navigator_var    = {args.n_navigator_variants}")
    print(f"[specevo] ablation_ast_only = {args.ast_only}")
    print(f"[specevo] ablation_emb_only = {args.emb_only}")
    print(f"[specevo] ablation_static   = {args.static_cells}")
    print(f"[specevo] post_init_evals   = {args.post_init_evals}")
    print(f"[specevo] error_rate_ivl    = {args.error_rate_interval}")
    print(f"[specevo] checkpoint_pop    = {args.checkpoint_population}")
    print(f"[specevo] advisor       = {not args.no_advisor} (interval {args.advisor_interval})")
    print(f"[specevo] single_prompt_ops = {args.single_prompt_operators}")
    print(f"[specevo] save_eval_code    = {args.save_eval_code}")
    print(f"[specevo] workers           = {workers}")
    print(f"[specevo] output_dir        = {output_dir}")

    result = specevo.run_specevo(
        problem_description,
        function_signature=function_signature,
        score_fn=score_fn,
        seed_program=seed_program,
        inputs=inputs,
        speculator_model=args.speculator_model,
        navigator_model=args.navigator_model,
        embedding_model=args.embedding_model,
        budget_evals=evals,
        budget_dollars=dollars,
        budget_seconds=seconds,
        target_score=target_score,
        n_workers=workers,
        n_eval_processes=eval_processes,
        eval_timeout=eval_timeout,
        navigator_interval=navigator_interval,
        n_diverse_seeds=args.n_diverse_seeds,
        n_variants_per_seed=args.n_variants_per_seed,
        n_navigator_variants=args.n_navigator_variants,
        enable_advisor=not args.no_advisor,
        advisor_interval=args.advisor_interval,
        output_dir=output_dir,
        **overrides,
    )

    summary = {
        "method": "specevo",
        "task_dir": str(task_dir),
        "speculator_model": args.speculator_model,
        "navigator_model": args.navigator_model,
        "embedding_model": args.embedding_model,
        "best_score": result.best_score,
        "best_metrics": result.best_metrics,
        "total_evaluations": result.total_evaluations,
        "total_cost": result.total_cost,
        # Token accounting, same shape the baselines report in
        # cost_log.totals.json: whole-run totals, plus the init phase
        # (diverse seeds + variants) broken out on its own.
        "total_llm_calls": result.total_llm_calls,
        "total_prompt_tokens": result.total_prompt_tokens,
        "total_completion_tokens": result.total_completion_tokens,
        "total_tokens": result.total_prompt_tokens + result.total_completion_tokens,
        "init_usage": result.init_usage,
        "archive_size": result.archive_size,
        "runtime_seconds": result.runtime_seconds,
        "n_navigator_trials": len(result.navigator_trials),
        "ablation": {
            "ast_only": args.ast_only,
            "emb_only": args.emb_only,
            "static_cells": args.static_cells,
            "no_advisor": args.no_advisor,
            "advisor_mode": args.advisor_mode,
            "no_targeted_mutate": args.no_targeted_mutate,
            "no_crossover": args.no_crossover,
            "p_crossover": args.p_crossover,
            "single_prompt_operators": args.single_prompt_operators,
            "navigator_force_mode": args.navigator_force_mode,
        },
        "budget": {
            "evaluations": evals,
            "dollars": dollars,
            "seconds": seconds,
            "target_score": target_score,
            "post_init_evaluations": args.post_init_evals,
        },
    }
    if args.error_rate_interval > 0:
        windows = result.error_rate_windows
        n_scored = sum(w["n_scored"] for w in windows)
        n_err = sum(w["n_errors"] for w in windows)
        summary["error_rate"] = {
            "interval": args.error_rate_interval,
            "n_windows": len(windows),
            "overall": (n_err / n_scored) if n_scored else None,
            "excluded_llm_failures": sum(w["n_excluded"] for w in windows),
            "per_window": [w["error_rate"] for w in windows],
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "best_program.py").write_text(result.best_program or "")

    print()
    print(f"Best score        : {result.best_score:.6f}")
    # CO-Bench (and any problem returning a dev/test split) reports the held-out
    # test score alongside the optimised dev score. ``best_score`` == dev_score.
    _m = result.best_metrics or {}
    if "dev_score" in _m:
        print(f"Dev score         : {float(_m['dev_score']):.6f}")
    if "test_score" in _m:
        print(f"Test score        : {float(_m['test_score']):.6f}  (held out, not optimised)")
    if "overall_score" in _m:
        print(f"Overall score     : {float(_m['overall_score']):.6f}")
    print(f"Evaluations used  : {result.total_evaluations}")
    print(f"Total cost        : ${result.total_cost:.4f}")
    print(f"LLM calls         : {result.total_llm_calls}")
    print(f"Input tokens      : {result.total_prompt_tokens}")
    print(f"Output tokens     : {result.total_completion_tokens}")
    print(f"Total tokens      : {result.total_prompt_tokens + result.total_completion_tokens}")
    _init = result.init_usage or {}
    if _init:
        print(
            f"Init input tokens : {_init.get('prompt_tokens', 0)}"
            f"  (init phase: {_init.get('llm_calls', 0)} calls,"
            f" {_init.get('evaluations', 0)} evals, ${float(_init.get('cost_usd', 0.0)):.4f})"
        )
        print(f"Init output tokens: {_init.get('completion_tokens', 0)}")
    print(f"Archive size      : {result.archive_size}")
    print(f"Navigator trials  : {len(result.navigator_trials)}")
    print(f"Runtime           : {result.runtime_seconds:.1f}s")
    print(f"Output dir        : {output_dir}")
    if args.save_eval_code:
        eval_log_json = output_dir / "eval_code_log.json"
        if eval_log_json.exists():
            try:
                n = json.loads(eval_log_json.read_text()).get("n_records")
            except Exception:  # pragma: no cover - never fail a run on a summary line
                n = "?"
            print(f"Eval code log     : {n} candidates -> {eval_log_json}")
        else:
            print(f"Eval code log     : {output_dir / 'eval_code_log.jsonl'} (json not written)")

    # Search trace: which Navigator mode fired when, and who produced each
    # new best. Written incrementally during the run by the orchestrator.
    snap_path = output_dir / "snap.json"
    if snap_path.exists():
        try:
            snap = json.loads(snap_path.read_text())
            nav = snap["summary"]["navigator"]
            nb = snap["summary"]["new_best"]
            modes = ", ".join(f"{k}={v}" for k, v in nav["by_mode"].items() if v)
            producers = ", ".join(f"{k}={v}" for k, v in nb["by_producer"].items() if v)
            print(f"Navigator calls   : {nav['calls']}" + (f"  ({modes})" if modes else ""))
            print(f"New bests         : {nb['count']}" + (f"  ({producers})" if producers else ""))
            print(f"Search trace      : {snap_path}")
        except Exception as e:  # pragma: no cover — never fail a run on a summary line
            print(f"Search trace      : {snap_path}  (unreadable: {e})")

    # Last thing on screen: the ablation's ten error-rate measurements.
    if args.error_rate_interval > 0:
        from specevo.engine import format_error_rate_table

        print(format_error_rate_table(
            result.error_rate_windows, interval=args.error_rate_interval,
        ))
        print(f"Report            : {output_dir / 'error_rate_report.json'}")
        if args.checkpoint_population:
            ckpts = sorted((output_dir / "checkpoints").glob("checkpoint_*.json"))
            print(f"Checkpoints       : {len(ckpts)} in {output_dir / 'checkpoints'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
