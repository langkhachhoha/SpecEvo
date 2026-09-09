# RelayEvolve

Training-free **population handoff** for cost-efficient LLM-driven evolution,
plus the cheap/strong **routing baselines** it is compared against — all on
SpecEvo Baselines's `openevolve_native` backend (island MAP-Elites), all running the
parallel discovery loop.

Reference: *Relay, Don't Route: Adaptive Population Handoff for Cost-Efficient
LLM-Driven Evolution.*

## The idea

Model calls in evolutionary search are **state-coupled**: each proposal changes
the population the next call inherits. So the useful question is not "which
model should make the next call" (routing) but "has the cheap phase produced a
population worth handing to the strong model" (relaying).

```
       ┌──────────────── cheap phase ────────────────┐   handoff   ┌── strong phase ──┐
       │  traj 0  ██▁▁██▁▁                           │             │                  │
init ──┤  traj 1  ▁▁██▁▁██   ← Grow/Deepen bandit    ├── S* ──────▶│  one shared      │──▶ best
       │  traj 2  ▁▁▁▁██▁▁     reward = Relay Gain   │  k seeds    │  population      │
       └──────────┬──────────────────────────────────┘             └──────────────────┘
                  │ after each block
            online relay bank  S_t ⊆ C_t,  |S_t| ≤ k
            F_C(S) = λ·Q_r(S) + (1-λ)·D^q_C(S)
```

* **Relay bank** — a compact set balancing top-`r` quality `Q_r` against
  quality-weighted coverage `D^q_C` of the candidate pool. Monotone submodular
  for a fixed pool.
* **Relay Gain** `g_t = F_{C_{t+1}}(S_{t+1}) − F_{C_{t+1}}(S_t)` — the marginal
  improvement of that bank. Both banks are scored against the *same, updated*
  pool, so the gain measures the handoff population improving, not the
  reference pool drifting.
* **Grow–Deepen bandit** — recent-window UCB over one shared `Grow` arm and one
  `Deepen(i)` arm per trajectory, rewarded by the bounded relative gain `ρ_t`.
* **Adaptive handoff** — fires when `ρ` stays under `epsilon_rel` for
  `patience` consecutive blocks, or when the cheap stage budget `B_c` is spent.
* **Curation** — greedy submodular selection plus local search over the whole
  terminal pool, compared against the polished online bank; the better wins, so
  the `1 − 1/e` guarantee survives.

## Search types

| `--search` | Method |
|---|---|
| `relayevolve` | The full method above |
| `relay_all_cheap` | Cheap model throughout |
| `relay_all_strong` | Strong model throughout |
| `relay_fixed_switch` | Cheap prefix of the budget, then strong |
| `relay_random` | Independent coin flip per generation |
| `relay_bandit` | Two-armed UCB rewarded by best-so-far improvement |

The cheap model is `llm.models`; the strong model is `llm.guide_models`.

## Running

```bash
python scripts/run_relay.py --method relayevolve \
    --benchmark-dir benchmarks/math/circle_packing \
    --iterations 300 --dollars 2 --workers 8 --seed 1
```

On a server, wrap the same command in `tmux` or `nohup` so the run survives
an SSH disconnect.

## Files

| File | What lives there |
|---|---|
| `bank.py` | `Q_r`, `D^q_C`, `F_C`, the online bank update, Relay Gain, greedy + local-search curation |
| `scheduler.py` | `GrowDeepenScheduler` (Eq. 9) and the baselines' `TwoArmedBandit` |
| `tiered.py` | Two LLM pools, per-task tier routing, the parallel block runner, budget and progress accounting |
| `controller.py` | The two-phase RelayEvolve orchestration |
| `baselines.py` | All-cheap / All-strong / Fixed-switch / Random / Bandit |
| `embedding.py` | Local hashing embedder (default, free) or an OpenAI-compatible `/embeddings` endpoint |
| `database.py` | Thin `openevolve_native` subclasses so each method can be named |

## What a run writes

| File | Contents |
|---|---|
| `relay_summary.json` | Method, models, handoff generation and reason, curated seeds, per-tier call counts, per-block Relay Gains, token and dollar totals |
| `relay_progress.jsonl` | One record per generation: tier, phase, score, best-so-far, cumulative cost — enough for a cost-vs-score curve |
| `cost_log.jsonl` / `.totals.json` | Per-call spend as reported by OpenRouter |

## Measured throughput

`benchmarks/math/circle_packing`, 8 workers, `--retries 1`, OpenRouter:

| Model | Latency / generation | Cost / generation |
|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | 31 s (19–62 s) | $0.0007 |
| `moonshotai/kimi-k2` | 74 s (37–179 s) | $0.0099 |

So 300 generations takes roughly 20–30 min all-cheap and 30–50 min for the
mixed methods, and a $2 cap binds only for All-strong (it buys ~200 strong
generations). Raising `--workers` scales the wall clock down roughly linearly
until the provider rate-limits.

## Knowing a run is done

Every method closes with the same block, whichever of its three caps it hit:

```
 [OK] RUN FINISHED — all_strong on circle_packing (seed 1)
 stopped because : dollar budget reached ($2.0143 of $2.00)
 best score      : 2.445300   (test-mode)
 generations     : 198 of 300
 ...
```

`stopped because` is the field that matters: a run that ran out of money at
generation 198 is not the same result as one that finished 300, and without
that line the two are indistinguishable from the score alone. The `handoff`
line appears only for the methods that actually hand over.

## Reading results back

Run directories are self-describing, so results survive forgetting the tmux
session name:

```bash
python scripts/relay_summarize.py                 # table of every run
python scripts/relay_summarize.py --agg           # mean +/- std over seeds
python scripts/relay_summarize.py --csv out.csv
python scripts/relay_summarize.py --path outputs/server/<run-id>   # one run in full
```

## Budget

`--dollars` is enforced by `specevo_baselines.llm.cost_tracker`: it stops the run
gracefully at the next generation boundary once the provider-reported total
reaches the cap, keeping the best program, the checkpoints and the final test
evaluation. It is a *soft* cap by one generation's worth of in-flight requests,
so a $2 run can land at $2.03. Add `--timeout` for a hard wall-clock stop.

## Tests

```bash
python -m pytest tests/search/test_relay.py -q
```

The LLM is stubbed, so the suite runs offline and costs nothing while still
driving the real controllers, populations, evaluator and parallel loop.
