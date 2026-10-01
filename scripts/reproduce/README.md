# Reproduction scripts

Each script runs one benchmark suite end to end. They are thin loops over
`scripts/run_specevo.py` (SpecEvo) and `scripts/run_levi.py` / the
`specevo-baselines-run` CLI (baselines) — read them before launching, they
spend real API credit.

| Script | Suite | Tasks |
|---|---|---|
| `math.sh`      | Mathematical discovery | 7 |
| `adrs.sh`      | Systems optimization (ADRS) | 4 |
| `co_bench.sh`  | CO-Bench combinatorial optimization | 36 |
| `lsr_synth.sh` | LSR-Synth equation discovery | 129 |

Every script honours the same environment overrides:

```bash
METHOD=specevo        # specevo | levi | openevolve_native | gepa_native | adaevolve | evox | relayevolve | ...
SPECULATOR_MODEL=...  # lightweight model  (default: qwen3-30b-a3b-instruct-2507)
NAVIGATOR_MODEL=...   # frontier model     (default: gpt-5)
EVALS=500             # evaluation budget
DOLLARS=10            # API-cost budget in USD
SECONDS_CAP=10800     # wall-clock budget (3h)
WORKERS=4             # concurrent Speculators / native baseline iterations
SEED=1                # local search seed (also recorded in the run folder)
PY=.venv/bin/python   # optional interpreter override
OUT=outputs/repro     # output root
```

`SEED` controls local search sampling. Remote model responses, asynchronous worker
completion order, and evaluator internals can still vary, so matching seeds do
not guarantee identical trajectories. SpecEvo and LEVI summaries record the seed;
native baselines record it in `run_settings.json`.

Dollar and wall-clock limits are graceful stopping thresholds. In-flight work may
finish after a threshold is reached. Reproduction scripts support SpecEvo, LEVI,
and native baselines; external `openevolve`, `shinkaevolve`, and `gepa` runners
must be configured directly because they manage their own budgets and workers.

Credentials may be configured in `.env` or exported in the environment. For
OpenRouter, use `OPENROUTER_API_KEY` (or its `OPENAI_API_KEY` compatibility alias).
LSR-Synth requires data preparation and generated task directories; see
[`benchmarks/llm_srbench`](../../benchmarks/llm_srbench/README.md).

Native baselines require a positive integer `EVALS`, since their schedules use
that limit. `EVALS=none` is only supported by the SpecEvo and LEVI task runners.
For `relayevolve` and `relay_*`, `SPECULATOR_MODEL` selects the cheap model and
`NAVIGATOR_MODEL` selects the strong model; other native baselines use
`NAVIGATOR_MODEL`. `WORKERS` is the maximum native iteration concurrency;
methods with sequential controllers (such as AdaEvolve and EvoX) remain sequential.

After each LSR-Synth search, `lsr_synth.sh` evaluates the saved best program on
held-out ID/OOD data and appends a record to
`$OUT/$METHOD/seed$SEED/results.jsonl`. The printed summarization command then
aggregates these finalized records.
