# Baselines

[Home](../README.md) · [Setup](SETUP.md) · [Reproduction](REPRODUCING.md)

Every baseline runs on the same task specification, evaluator, budget accounting and prompt scaffold —
only the search logic differs. See [`specevo_baselines/README.md`](../specevo_baselines/README.md) for the
shared engine.

### Heterogeneous-model methods

```bash
# LEVI — diversity preservation + role-aware routing
python scripts/run_levi.py --task-dir tasks/circle_packing --evals 500 --dollars 10

# RelayEvolve — cheap multi-trajectory exploration, one-time handoff to strong refinement
python scripts/run_relay.py --method relayevolve \
    --benchmark-dir benchmarks/math/circle_packing \
    --iterations 500 --dollars 10 --workers 8 --seed 1
```

`run_levi.py` names its models `--small-model` / `--large-model` rather than by paper role.

### Controlled model-allocation strategies

These isolate the effect of *coordination* from the mere availability of two differently priced models.
They share one evolutionary backend and differ only in which model gets each call:

| `--method` | Allocation rule |
|:--|:--|
| `relayevolve` | Cheap multi-trajectory exploration → Relay-Gain handoff → strong refinement |
| `all_cheap` | Every call to the lightweight model |
| `all_strong` | Every call to the frontier model |
| `fixed_switch` | Lightweight prefix, then a one-time switch (`--switch-fraction`) |
| `random` | Independent coin flip per generation (`--p-strong`) |
| `bandit` | Two-armed UCB on realized best-so-far improvement |

```bash
python scripts/run_relay.py --method bandit \
    --benchmark-dir benchmarks/math/circle_packing \
    --cheap-model openrouter/qwen/qwen3-30b-a3b-instruct-2507 \
    --strong-model openrouter/openai/gpt-5 \
    --iterations 500 --dollars 10
```

Defaults are `--cheap-model openrouter/qwen/qwen3-30b-a3b-instruct-2507`,
`--strong-model openrouter/moonshotai/kimi-k2`, `--iterations 300`, `--dollars 2`, `--workers 8`.

### Single-backbone evolutionary frameworks

```bash
python -m specevo_baselines.cli \
    benchmarks/math/circle_packing/initial_program.py \
    benchmarks/math/circle_packing/evaluator.py \
    -c benchmarks/math/circle_packing/config.yaml \
    -s openevolve_native -m openrouter/openai/gpt-5 \
    -i 500 --dollars 10 -o outputs/oe/circle_packing
```

| `-s` value | Method |
|:--|:--|
| `openevolve_native` | **OpenEvolve** — island MAP-Elites with evaluator feedback |
| `gepa_native` | **GEPA** — reflective mutation from execution traces, Pareto selection + merge |
| `adaevolve` | **AdaEvolve** — bandit-scheduled islands, meta-level tactics under stagnation |
| `evox` | **EvoX** — co-evolves the candidate program and the search strategy |

Also available: `best_of_n`, `beam_search`, `topk`, `relay*`, and the `openevolve` / `gepa` /
`shinkaevolve` wrappers around the upstream libraries (install the `external` extra).

Shared flags: `-i/--iterations`, `--dollars`, `-m/--model`, `-c/--config`, `-o/--output`,
`--seconds` (graceful time stop), `--workers`, `--seed`, `--checkpoint` (resume), `-l/--log-level`, `--api-base`, `--agentic`. Starter configs live in
[`configs/`](../configs/) — see [`configs/README.md`](../configs/README.md).
