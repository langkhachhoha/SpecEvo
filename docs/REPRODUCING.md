# Reproducing the paper

[Home](../README.md) · [Setup](SETUP.md) · [Results and artifacts](RESULTS.md)

Run commands below from the repository root. These are experiment launchers; a complete rerun requires the benchmark dependencies, downloaded data and an API key.

One script per suite, in [`scripts/reproduce/`](../scripts/reproduce/):

```bash
bash scripts/reproduce/math.sh          #   7 tasks
bash scripts/reproduce/adrs.sh          #   4 tasks
bash scripts/reproduce/co_bench.sh      #  36 tasks
bash scripts/reproduce/lsr_synth.sh     # 129 tasks
```

All four read the same environment overrides (defaults shown):

| Variable | Default | Meaning |
|:--|:--|:--|
| `METHOD` | `specevo` | `specevo`, `levi`, or a native baseline search key |
| `SPECULATOR_MODEL` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | lightweight model (LEVI's `--small-model`) |
| `NAVIGATOR_MODEL` | `openrouter/openai/gpt-5` | frontier model (also the baselines' `-m`) |
| `EVALS` | `500` | evaluation cap |
| `DOLLARS` | `10` | USD stopping threshold |
| `SECONDS_CAP` | `10800` | wall-clock stopping threshold |
| `WORKERS` | `4` | Speculator concurrency / native baseline maximum concurrency |
| `SEED` | `1` | local search seed (also used in the output path) |
| `OUT` | `outputs/repro` | output root: `$OUT/$METHOD/<task>/seed$SEED` |

```bash
METHOD=specevo EVALS=500 DOLLARS=10 SECONDS_CAP=10800 SEED=1 \
OUT=outputs/repro \
bash scripts/reproduce/math.sh
```

The scripts dispatch to `run_specevo.py` for `METHOD=specevo`, `run_levi.py` for `METHOD=levi`, and
`python -m specevo_baselines.cli -s $METHOD` for everything else. To sweep a table row, loop over
methods and seeds:

```bash
for m in specevo levi relayevolve openevolve_native gepa_native adaevolve evox; do
  for s in 1 2 3; do METHOD=$m SEED=$s bash scripts/reproduce/math.sh; done
done
```

> [!WARNING]
> These scripts spend real API credit across every task in a suite — `lsr_synth.sh` alone covers 129
> problems. Read them before launching, and start with a small `DOLLARS` value.

## Budget regimes and reproducibility

The suite launchers enable all three limits by default. A run stops when the
first limit is reached; this is a convenient bounded run, not automatically the
same as each of the paper's separate budget settings.

For an individual SpecEvo task, enable just the desired budget:

```bash
# Three independent examples; each starts a paid run.
python scripts/run_specevo.py --task-dir tasks/circle_packing --seconds 10800
python scripts/run_specevo.py --task-dir tasks/circle_packing --dollars 10
python scripts/run_specevo.py --task-dir tasks/circle_packing --evals 500
```

For SpecEvo/LEVI suite runs, set unused limits to `none`, for example
`EVALS=none DOLLARS=none SECONDS_CAP=10800`. Native baseline launchers require
numeric limits and use an iteration horizon for scheduling. Their default
`EVALS=500` is an iteration limit, which is not necessarily 500 candidate
evaluations. Select method-specific settings before comparing paper tables.
External library wrappers are run directly; suite launchers reject them because
the shared budget controls do not cover those backends.

`SEED` is passed into local search randomness and recorded with run settings.
It does not make remote model responses or concurrent execution deterministic.
Cost and wall-clock stops allow work already in flight to finish.
Some native baseline controllers (including GEPA, AdaEvolve and EvoX) run
sequentially even when a larger worker maximum is supplied.
The `PY` variable selects an interpreter; `PY_OVERRIDE` takes precedence if set.

## Aggregating LSR-Synth results

```bash
python scripts/lsr_summarize.py outputs/repro/specevo          # NMSE / Acc0.1 per domain, ID & OOD
python scripts/lsr_symbolic_accuracy.py outputs/repro/specevo  # LLM equivalence judge (spends credit)
```

`lsr_summarize.py` reads every `results.jsonl` under the paths given and takes `--csv` / `--json` to
export. Individual problems are finalized by `scripts/lsr_finalize.py`, which re-evaluates the program
left on disk rather than trusting the search's own metric dict.

> [!WARNING]
> **Hardware sensitivity.** Wall-clock results depend on the parallelism available to the Speculator
> pool. The paper's runs used a dual-socket server with two 16-core Intel Xeon Silver 4314 CPUs and
> `W=4`. API-cost and evaluation-budget comparisons reduce sensitivity to hardware, but evaluator timeouts, concurrency and stochastic model responses can still affect results.
