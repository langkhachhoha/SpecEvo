#!/usr/bin/env python3
"""Generate the per-problem benchmark directories for LSR-Synth.

LSR-Synth is a *set* of independent equation-discovery problems, and both
runners in this repo evolve one program against one scoring function, so each
problem gets its own directory — the same layout CO-Bench uses.

A per-domain layout is not possible: LSR-Synth only exposes the variables that
actually occur in the target equation, so ``phys_osc`` mixes ``(x, t, v)``,
``(x, t)`` and ``(t, v)`` problems and the evolved function's signature differs
between them.

Written for each problem::

    benchmarks/llm_srbench/<domain>/<pid>/initial_program.py    seed skeleton
    benchmarks/llm_srbench/<domain>/<pid>/config.yaml           baseline config
    benchmarks/llm_srbench/<domain>/<pid>/evaluator.py          -> lsr_eval
    benchmarks/llm_srbench/<domain>/<pid>/download_dataset.sh   data provisioning
    tasks/llm_srbench/<domain>/<pid>/problem.py         SpecEvo / SpecEvo

Everything substantive lives in ``benchmarks/llm_srbench/lsr_eval.py``; these
files only name a problem and delegate, so regenerating them after an engine
change is safe.

Usage::

    python benchmarks/llm_srbench/generate_dirs.py --limit 10      # reduced set
    python benchmarks/llm_srbench/generate_dirs.py                 # all problems
    python benchmarks/llm_srbench/generate_dirs.py --domain matsci --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lsr_eval as L  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[1]
TASK_DIR = REPO_ROOT / "tasks" / "llm_srbench"

DEFAULT_MODEL = "openrouter/qwen/qwen3-30b-a3b-instruct-2507"

# Comment prefix in config.yaml recording the LSR_SCORE_MODE it was generated
# under. Kept as a module constant so launchers can grep for it.
SCORE_MODE_STAMP = "# score-mode:"

# Bumped whenever ``config_yaml`` starts emitting something a previously
# generated config.yaml does not have. Launchers grep for the current
# value and regenerates any problem directory that predates it, the same way it
# already does for the score-mode stamp — otherwise a committed directory would
# silently keep running under the old settings.
CONFIG_REV_STAMP = "# config-rev:"
CONFIG_REV = 2


# --------------------------------------------------------------------------- #
# File bodies
# --------------------------------------------------------------------------- #
class _BlockDumper:
    """yaml.safe_dump that writes multi-line strings as ``|`` block scalars.

    The system message is a page of markdown; without this it comes out as one
    escaped line and nobody can review the prompt the model actually receives.
    """

    @staticmethod
    def dump(data: dict) -> str:
        import yaml

        class Dumper(yaml.SafeDumper):
            pass

        def _str(dumper, value):  # noqa: ANN001
            style = "|" if "\n" in value else None
            return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

        Dumper.add_representer(str, _str)
        return yaml.dump(data, Dumper=Dumper, sort_keys=False, width=100, allow_unicode=True)


def config_yaml(domain: str, pid: str, *, iterations: int, model: str) -> str:
    cfg = {
        "language": "python",
        "diff_based_generation": True,
        "max_iterations": iterations,
        "checkpoint_interval": 10,
        "max_solution_length": 60000,
        "llm": {
            "api_base": "https://openrouter.ai/api/v1",
            "models": [{"name": model, "weight": 1.0}],
            "timeout": 600,
        },
        "prompt": {
            "system_message": L.system_message(domain, pid),
            # `feedback` is a human-readable restatement of numbers that are
            # already in the metric dict as floats (train/ID/OOD NMSE, R2,
            # Acc(0.1), within-0.1). The default prompt renders the metric dict
            # once per context program, once per previous attempt, once in the
            # header and once on the current program, so the same 315-character
            # paragraph was being paid for nine times on every single call —
            # 20% of the input tokens of a 500-iteration run, carrying nothing
            # the model could not read off the floats beside it. It still lands
            # in results.jsonl and in every checkpoint; it just no longer rides
            # along in the context. Ignored by SpecEvo, which never renders a
            # metric dict into its prompts.
            "exclude_metrics": ["feedback"],
        },
        # SpecEvo runs the LSR-Synth searches with --workers 4; the baselines
        # default to one iteration in flight, which is why a 500-iteration
        # baseline problem takes ~3.5h against SpecEvo's ~40min on identical
        # hardware and an identical model.
        #
        # READ THIS BEFORE PUTTING WALL-CLOCK IN A TABLE: only openevolve_native
        # honours this. AdaEvolveController, GEPANativeController and
        # CoEvolutionController each override run_discovery with their own
        # sequential loop and never look at max_parallel_iterations, so those
        # three still run one iteration at a time.
        #
        # They are not simply unwired — each has real per-iteration state that
        # concurrency would corrupt: AdaEvolve closes every iteration with
        # database.end_iteration(), which picks the next island by UCB and
        # drives migration; GEPA's loop is a state machine over _merge_due,
        # _iterations_without_improvement and _best_score_seen, with acceptance
        # gating comparing child to parent inline; EvoX swaps its whole database
        # when it evolves its search strategy. Widening them means changing the
        # algorithms, at which point they stop being the published baselines.
        #
        # Left on for openevolve_native deliberately (decision 2026-08-01): it
        # cuts the re-run time and costs nothing, at the price of OE finishing
        # sooner than the other three for reasons that have nothing to do with
        # its search. Cost per call is unaffected by width, so the $ columns
        # stay comparable across all five methods either way.
        "max_parallel_iterations": 4,
        # EvoX co-evolves its own *search strategy* alongside the solution, and
        # that meta level loads specevo_baselines/search/evox/config/search.yaml, which
        # pins openai/gpt-5 (and gpt-5-mini for its guide). --model only rewrites
        # the config below, so without share_llm EvoX would be the one baseline
        # steered by a frontier model — ~50x the price per output token and not a
        # like-for-like comparison against the other four methods. Ignored by
        # every search type except evox.
        "search": {"share_llm": True},
        # Our own per-hypothesis cap (LSR_EVAL_TIMEOUT, default 30s per the
        # paper) is enforced inside the evaluator; this outer timeout only has
        # to be comfortably larger so it never pre-empts the graded result.
        "evaluator": {"timeout": 600, "cascade_evaluation": False},
        "monitor": {"enabled": False},
        "human_feedback_enabled": False,
    }
    header = (
        f"# LLM-SRBench / LSR-Synth :: {domain} / {pid}\n"
        f"# Generated by benchmarks/llm_srbench/generate_dirs.py — do not hand-edit.\n"
        f"# Usage: specevo-baselines-run initial_program.py evaluator.py -c config.yaml -s <strategy>\n"
        # The system message quotes the active scoring rule, so the file is only
        # valid for the LSR_SCORE_MODE it was written under. A launcher
        # reads this stamp and regenerates when the mode changes; the SpecEvo path
        # needs no stamp because it builds its prompt at import time.
        f"{SCORE_MODE_STAMP} {L.score_mode()}\n"
        f"{CONFIG_REV_STAMP} {CONFIG_REV}\n"
    )
    return header + _BlockDumper.dump(cfg)


EVALUATOR_TEMPLATE = '''\
"""Baseline evaluator for LLM-SRBench / LSR-Synth :: {domain} / {pid}.

Generated by benchmarks/llm_srbench/generate_dirs.py — do not hand-edit.

Delegates to the shared engine (benchmarks/llm_srbench/lsr_eval.py), which fits
the candidate's `params` on the training split with BFGS and then measures the
held-out in-domain (ID) and out-of-domain (OOD) test splits.

`combined_score` is a monotone transform of the training NMSE (see
LSR_SCORE_MODE in lsr_eval.py) — only the training fit drives the search;
`id_*` and `ood_*` are recorded but never optimised.

Runtime knobs (env): LSR_SCORE_MODE (log_nmse), LSR_EVAL_TIMEOUT (default 30s
per hypothesis, as in the paper), LSR_MAX_NPARAMS (10), LSR_DATA_DIR.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lsr_eval as L  # noqa: E402

DOMAIN = "{domain}"
PROBLEM = "{pid}"


def evaluate(program_path, **_):
    source = L.read_program_source(program_path)
    return L.baseline_metrics(L.evaluate_source(DOMAIN, PROBLEM, source))


if __name__ == "__main__":
    try:
        from wrapper import run

        run(evaluate)
    except Exception:
        import json

        print(json.dumps(evaluate(sys.argv[1])))
'''


DOWNLOAD_TEMPLATE = """\
#!/usr/bin/env bash
# Provision the LSR-Synth dataset for {domain}/{pid}.
# Generated by benchmarks/llm_srbench/generate_dirs.py — do not hand-edit.
#
# Idempotent: re-running only verifies the already-materialised .npz files.
set -euo pipefail
exec python "$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)/prepare_data.py" \\
    --domain {domain}
"""


PROBLEM_TEMPLATE = '''\
"""SpecEvo example for LLM-SRBench / LSR-Synth :: {domain} / {pid}.

Generated by benchmarks/llm_srbench/generate_dirs.py — do not hand-edit.

Exports PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM and score_fn so
`scripts/run_specevo.py` can evolve the equation skeleton. Scoring reuses the same
engine as the baseline path (benchmarks/llm_srbench/lsr_eval.py): `params` are
fitted on the training split with BFGS, then the held-out ID and OOD test splits
are measured. The score is a monotone transform of the training NMSE (see
LSR_SCORE_MODE in lsr_eval.py); higher is better, and a hypothesis that raises /
diverges / times out scores 0.
"""
import sys
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(_REPO / "benchmarks" / "llm_srbench"))
import lsr_eval as L  # noqa: E402

DOMAIN = "{domain}"
PROBLEM = "{pid}"

PROBLEM_DESCRIPTION = L.problem_description(DOMAIN, PROBLEM)
FUNCTION_SIGNATURE = L.equation_signature(DOMAIN, PROBLEM)
SEED_PROGRAM = L.seed_program(DOMAIN, PROBLEM)

INPUTS = None


def score_fn(equation_fn, _inputs=None) -> dict:
    return L.evaluate_callable(DOMAIN, PROBLEM, equation_fn)
'''


# --------------------------------------------------------------------------- #
def generate(domain: str, pid: str, *, iterations: int, model: str) -> None:
    bdir = BENCH_DIR / domain / pid
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "initial_program.py").write_text(L.initial_program_text(domain, pid))
    (bdir / "config.yaml").write_text(config_yaml(domain, pid, iterations=iterations, model=model))
    (bdir / "evaluator.py").write_text(EVALUATOR_TEMPLATE.format(domain=domain, pid=pid))
    dl = bdir / "download_dataset.sh"
    dl.write_text(DOWNLOAD_TEMPLATE.format(domain=domain, pid=pid))
    dl.chmod(0o755)

    ldir = TASK_DIR / domain / pid
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "problem.py").write_text(PROBLEM_TEMPLATE.format(domain=domain, pid=pid))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Only the first N problems per domain (default: every problem "
                         "present in data/problems.json).")
    ap.add_argument("--domain", action="append", default=None,
                    help="Restrict to one domain (repeatable).")
    ap.add_argument("--problem", action="append", default=None, metavar="PID",
                    help="Generate only these problem ids (repeatable). Overrides "
                         "--limit; requires a single --domain.")
    ap.add_argument("--iterations", type=int, default=500, metavar="N",
                    help="max_iterations written into config.yaml (default 500). The "
                         "runner's --iterations overrides it per run.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Default model in config.yaml (default: {DEFAULT_MODEL}). "
                         "The runner's --model overrides it per run.")
    args = ap.parse_args()

    wanted = args.domain or L.domains()
    if args.problem and len(wanted) != 1:
        ap.error("--problem requires exactly one --domain")
    total = 0
    for domain in wanted:
        known = L.problem_ids(domain)
        if args.problem:
            unknown = [p for p in args.problem if p not in known]
            if unknown:
                ap.error(f"unknown problem(s) in {domain}: {', '.join(unknown)}")
            pids = list(args.problem)
        else:
            pids = known[: args.limit or None]
        for pid in pids:
            generate(domain, pid, iterations=args.iterations, model=args.model)
        total += len(pids)
        print(f"  {domain:16s} {len(pids):3d} problems -> "
              f"benchmarks/llm_srbench/{domain}/<pid>/ + tasks/llm_srbench/{domain}/<pid>/")
    print(f"Generated {total} problem directories (max_iterations={args.iterations}, model={args.model}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
