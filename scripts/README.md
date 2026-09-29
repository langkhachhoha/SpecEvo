# Script guide

Run commands from the repository root after following the
[setup guide](../docs/SETUP.md). Use `--help` on the runners and LSR-Synth tools
for their complete options.

## Setup and local checks

| Script | Purpose |
|---|---|
| [check_tasks.py](check_tasks.py) | Import task adapters and check their required exports; no model calls |
| [download_benchmark_data.sh](download_benchmark_data.sh) | Fetch LLM-SQL, EPLB, or LSR-Synth data and generate LSR-Synth task directories |
| [install_benchmark_requirements.py](install_benchmark_requirements.py) | Install evaluator requirements for a benchmark path; supports `--dry-run` |

```bash
python scripts/check_tasks.py --suite math
python scripts/install_benchmark_requirements.py benchmarks/math/circle_packing --dry-run
bash scripts/download_benchmark_data.sh lsr_synth
```

The downloader accepts `llm_sql`, `eplb`, or `lsr_synth`; omitting the suite
fetches all three. LSR-Synth preparation requires the `lsr` dependency extra.

## Search and live API checks

| Script | Purpose |
|---|---|
| [run_specevo.py](run_specevo.py) | Run SpecEvo on a `tasks/` adapter |
| [run_levi.py](run_levi.py) | Run LEVI on a `tasks/` adapter |
| [run_relay.py](run_relay.py) | Run RelayEvolve or a controlled cheap/frontier allocation strategy on a `benchmarks/` directory |
| [reproduce/](reproduce/README.md) | Launch runs for an entire benchmark suite |
| [test_openrouter_key.py](test_openrouter_key.py) | Check authentication and request a short completion |
| [smoke_specevo_prompts.py](smoke_specevo_prompts.py) | Check prompt response structure using live model completions |

These scripts make model calls and can incur API charges. The search runners
accept `--dollars` to set an API-cost budget. See
[running SpecEvo](../docs/RUNNING.md), [baselines](../docs/BASELINES.md), and
[reproduction](../docs/REPRODUCING.md) for examples and budget settings.

## LSR-Synth results

| Script | Purpose |
|---|---|
| [lsr_finalize.py](lsr_finalize.py) | Re-evaluate a run's saved best program and append an ID/OOD result to `results.jsonl` |
| [lsr_summarize.py](lsr_summarize.py) | Aggregate `results.jsonl` files by method and domain; supports CSV and JSON export |
| [lsr_resume_plan.py](lsr_resume_plan.py) | Inspect a baseline checkpoint and print the remaining iteration budget; does not resume the search itself |
| [lsr_symbolic_accuracy.py](lsr_symbolic_accuracy.py) | Judge symbolic equivalence with an LLM; `--dry-run` prepares prompts without API calls |

```bash
python scripts/lsr_summarize.py outputs/repro/specevo --csv outputs/lsr_summary.csv
python scripts/lsr_symbolic_accuracy.py outputs/repro/specevo --dry-run
```

Finalization executes the saved candidate locally. Symbolic-accuracy judging
uses paid model calls unless `--dry-run` is set. These tools summarize newly
generated runs; the repository's [example programs](../result/README.md) do
not include full experimental logs.
