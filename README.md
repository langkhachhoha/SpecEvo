<div align="center">

## SpecEvo: Speculative Evolution with Large Language Models for Cost-Efficient Scientific Discovery

**Speculate → Consolidate.** Many cheap **Speculators** explore in parallel; an **Advisor** distills
the whole trajectory — failures included — into reusable guidance; a frontier **Navigator** is woken
only at the hard junctures.

[![Python](https://img.shields.io/badge/python-3.10%20–%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Tasks](https://img.shields.io/badge/benchmark%20tasks-176-orange.svg)](#benchmarks)
[![Tests](https://img.shields.io/badge/tests-607%20passing-brightgreen.svg)](#tests)

[Overview](#overview) · [Install](#installation) · [Quickstart](#quickstart) ·
[Running SpecEvo](#running-specevo) · [Baselines](#baselines) · [Benchmarks](#benchmarks) ·
[Reproducing](#reproducing-the-paper) · [Layout](#repository-layout) · [Tests](#tests)

</div>

---

## Overview

LLM-driven evolutionary search is a promising paradigm for scientific discovery, but existing methods remain costly and slow, often applying expensive reasoning uniformly while underusing evidence from failed, redundant, and saturated trials. Recent cost-aware approaches allocate models through predefined roles or one-way handoffs, but do not continually adapt this coordination as search evidence accumulates. We introduce **Speculative Evolution** (*SpecEvo*), an LLM-powered framework for efficient and fast scientific discovery that makes model coordination history-dependent and self-correcting. Lightweight Speculators perform parallel exploration, while an Advisor revises persistent guidance from trajectory-wide successes, failures, and saturated directions. At sparse checkpoints, a frontier-model Navigator inspects the evolving population and selects among synthesis, surgical refinement, and reframing. This *speculate-then-consolidate* workflow amortizes costly reasoning across many subsequent evaluations. Across 175 tasks in four scientific discovery domains, SpecEvo matches or outperforms frontier-model baselines while reducing API cost by up to 80\%, converges up to *1.5x* faster than the cost-efficient baselines on equation discovery, and improves over the strongest CO-Bench baseline by 3.8%. These results demonstrate the effectiveness of trajectory-conditioned model coordination for efficient scientific discovery.

| Role | Plays | Cadence | Model | What it does |
|:--|:--|:--|:--|:--|
| **Speculator** | junior researchers | continuous, `W=4` in parallel | lightweight | Draws a parent from the behavioral archive under a stagnation-adaptive Zipf distribution, applies one of six variation prompts, executes the candidate. Handles the large majority of all calls. |
| **Advisor** | senior labmate | every `Δ_A` evaluations | lightweight | Reads the whole trajectory — working niches, saturated niches, recurring execution failures — and rewrites a short guidance note. Injected into each Speculator prompt with probability `ρ_A`. |
| **Navigator** | principal investigator | every `Δ_N` evaluations | frontier | Sees a two-resolution view of the archive and makes one strategic intervention, escalating with the stagnation signal. |

The Navigator's intervention mode is chosen by the scalar stagnation signal `s_t`:

| `s_t` | Mode | Context it receives | Requested move |
|:--|:--|:--|:--|
| `≤ γ₁` | **Synthesis** | top 3 niche elites (full code) + 5 descriptions | Combine mechanisms across anchors |
| `γ₁ < s_t ≤ γ₂` | **Surgical** | global champion only + 5 descriptions | One local structural correction |
| `> γ₂` | **Reframe** | top 2 niche elites + 5 descriptions | Introduce an absent strategy family |

Expensive reasoning is therefore *deferred* until enough evidence has accumulated for one intervention
to influence many subsequent low-cost evaluations.

---

## Installation

### Prerequisites

- Python **3.10 – 3.13** (the checked-in environment uses 3.11)
- An [OpenRouter](https://openrouter.ai/) API key — every model in the paper (GPT-5, Kimi-K2, the
  Qwen3 family) is reached through one endpoint
- ~110 MB of benchmark data fetched at setup time

### 1. Get the code and an environment

<details open>
<summary><b>With <code>uv</code> (recommended)</b></summary>

```bash
git clone <repository-url> SpecEvo
cd SpecEvo

uv venv --python 3.11
uv pip install -e ".[dev]"
uv pip install -e ".[math,adrs,lsr]"   # benchmark evaluator dependencies
```

</details>

<details>
<summary><b>With <code>pip</code> / <code>venv</code></b></summary>

```bash
git clone <repository-url> SpecEvo
cd SpecEvo

python3.11 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev,math,adrs,lsr]"
```

### 2. Set your API key

```bash
cp .env.example .env
$EDITOR .env                           # set OPENAI_API_KEY=sk-or-v1-...
```

Both runners load `.env` from the repository root automatically. An OpenRouter key placed in
`OPENAI_API_KEY` is mirrored into `OPENROUTER_API_KEY` for you.

Verify it works before spending anything:

```bash
python scripts/test_openrouter_key.py
```

<details>
<summary>Expected output</summary>

```
Key: sk-or-v1...abcd
Model test: openrouter/openai/gpt-4o-mini

[1/2] GET /auth/key ...
OK — label='sk-or-v1-...', usage=..., limit=...
[2/2] POST /chat/completions (max_tokens=8) ...
OK — model='openai/gpt-4o-mini', reply='OK'
```

</details>

### 3. Fetch the benchmark data

The large CSV/JSON/parquet payloads are not committed. One command fetches all of them:

```bash
bash scripts/download_benchmark_data.sh           # all suites (~110 MB)
bash scripts/download_benchmark_data.sh llm_sql   # or just one: llm_sql | eplb | lsr_synth
```

| Suite | What it fetches | Needed for |
|:--|:--|:--|
| `llm_sql` | 5 CSVs (~69 MB) | ADRS / LLM-SQL |
| `eplb` | `expert-load.json` MoE workload | ADRS / EPLB |
| `lsr_synth` | LSR-Synth splits from the Hub, then regenerates the 129 per-problem task directories (needs the `lsr` extra) | LSR-Synth |

Some benchmark evaluators ship their own `requirements.txt`. Install them per benchmark directory:

```bash
python scripts/install_benchmark_requirements.py benchmarks/math/circle_packing
python scripts/install_benchmark_requirements.py benchmarks/math/circle_packing --dry-run   # preview
```

### 4. Verify the checkout

Import-checks all 176 tasks and makes no API calls — the fastest way to confirm setup is complete:

```bash
python scripts/check_tasks.py
```

```
Checking 176 tasks across 4 suite(s)

  Systems optimization (ADRS)        4 /   4
  CO-Bench                          36 /  36
  LSR-Synth                        129 / 129
  Mathematical discovery             7 /   7

  TOTAL                            176 / 176

All tasks load correctly.
```

---

## Quickstart

A few-minute run on a toy task, using a cheap model in **both** roles:

```bash
python scripts/run_specevo.py \
    --task-dir tasks/smoke_demo \
    --speculator-model openrouter/openai/gpt-4o-mini \
    --navigator-model  openrouter/openai/gpt-4o-mini \
    --evals 12 --n-diverse-seeds 2 --n-variants-per-seed 2 \
    --navigator-interval 5 --advisor-interval 5 \
    --workers 2 --eval-processes 2 --eval-timeout 30 \
    --output-dir outputs/smoke
```

<details>
<summary>What you should see</summary>

```
[SpecEvo init] phase 1 done: 3 seeds admitted
[SpecEvo init] phase 2 generating 6 variants (2 × 3) in parallel
[SpecEvo] bootstrap complete — archive=9 best=0.999997 cost=$0.003 evals=10
[SpecEvo] entering evolutionary main loop (n_workers=2)
[SpecEvo advisor] trigger #1 at eval=10
[SpecEvo advisor] new advice (1147 chars) at eval=10
[Eval #11] gpt-4o-mini   accepted  | source: mutate_mechanism_swap | score: 0.999993 | ...

Best score        : 0.999997
Evaluations used  : 12
Total cost        : $0.0039
Navigator calls   : 0
```

</details>

Then run something real:

```bash
python scripts/run_specevo.py --task-dir tasks/circle_packing --evals 500 --dollars 10
```

> [!WARNING]
> Every command from here down spends real API credit. Always pass `--dollars` (SpecEvo / LEVI /
> baselines) as a hard stop while you are finding your footing.

---

## Running SpecEvo

```bash
python scripts/run_specevo.py --task-dir <task> [options]
```

A task directory is any folder whose `problem.py` exports `PROBLEM_DESCRIPTION`,
`FUNCTION_SIGNATURE` and `score_fn` (optionally `SEED_PROGRAM` / `INPUTS`). All 176 benchmark tasks
under [`tasks/`](tasks/) follow this contract — see [`tasks/README.md`](tasks/README.md) — and your
own problems plug in unchanged. Use `--problem-module` if your module is not named `problem`.

### Models

| Flag | Default | Paper role |
|:--|:--|:--|
| `--speculator-model` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | `M_S` — lightweight explorer |
| `--navigator-model` | `openrouter/openai/gpt-5` | `M_N` — frontier intervention |
| `--embedding-model` | `openrouter/openai/text-embedding-3-small` | description embeddings for the archive |

The Advisor `M_A` runs on the Speculator model. Swap the backbone family with, e.g.,
`--navigator-model openrouter/moonshotai/kimi-k2-thinking`.

### Budget — the run stops at whichever trips first

| Flag | Meaning |
|:--|:--|
| `--evals N` | Evaluation budget (the paper's 500-call setting) |
| `--dollars N` | API-cost budget in USD (the paper's \$10 setting) |
| `--seconds N` | Wall-clock budget (the paper's 3-hour setting = `10800`) |
| `--target-score X` | Stop early once a candidate reaches `X` |
| `--post-init-evals N` | Budget counted from the *end* of initialization, so the ~105 bootstrap evaluations do not eat into it |

All four are unset by default.

### Concurrency

| Flag | Default | Meaning |
|:--|:--|:--|
| `--workers` | `4` | Parallel Speculators (`W` in the paper) |
| `--eval-processes` | `4` | Parallel evaluator processes |
| `--eval-timeout` | `600` | Per-candidate evaluation timeout (seconds) |

### Search structure

| Flag | Default | Paper symbol |
|:--|:--|:--|
| `--n-diverse-seeds` | `5` | `M` — frontier-generated seed strategies |
| `--n-variants-per-seed` | `20` | `V` — lightweight variants per seed |
| `--n-cells` | `50` | `K` — behavioral-archive clusters |
| `--recluster-every` | `30` | `Δ_C` — re-fit k-means every N admits |
| `--embedding-dim` | `8` | PCA dimension of the description-embedding half |

### Navigator

| Flag | Default | Meaning |
|:--|:--|:--|
| `--navigator-interval` | `50` | `Δ_N` — fire the Navigator every N completed evaluations |
| `--n-navigator-variants` | `4` | `J` — Speculator fan-out after each intervention |
| `--navigator-n-anchors` | `4` | Full-code anchors sent to the frontier |
| `--navigator-n-inspirations` | `5` | Description-only inspirations alongside the anchors |
| `--navigator-synthesis-max-stagnation` | `0.4` | `γ₁` — synthesis/surgical boundary |
| `--navigator-surgical-max-stagnation` | `0.7` | `γ₂` — surgical/reframe boundary |
| `--navigator-synthesis-n-anchors` | `3` | Anchors in Synthesis mode |
| `--navigator-reframe-n-anchors` | `2` | Anchors in Reframe mode |
| `--navigator-surgical-n-inspirations` | `5` | Inspirations in Surgical mode |
| `--navigator-force-mode` | adaptive | Pin one of `synthesis` / `surgical` / `reframe` (ablation) |

### Advisor

| Flag | Default | Meaning |
|:--|:--|:--|
| `--advisor-interval` | `50` | `Δ_A` — rewrite the guidance note every N evaluations |
| `--advisor-inject-p` | `0.35` | `ρ_A` — probability a Speculator prompt carries the note |
| `--advisor-mode` | `rich` | `rich` uses success + saturation + error signals; `errors_only` is the ablation |
| `--no-advisor` | off | Disable the Advisor entirely |

### Parent analysis and crossover

| Flag | Default | Meaning |
|:--|:--|:--|
| `--analyzer-interval` | `30` | Refresh cached parent analyses every N evaluations |
| `--analyzer-top-k` | `3` | Top-ranked programs analysed per refresh |
| `--p-targeted-mutate` | `0.5` | Probability of the targeted-mutate prompt when an analysis is cached |
| `--p-crossover` | `0.35` | Crossover probability |

### Ablation switches

| Flag | Removes |
|:--|:--|
| `--ast-only` | The description-embedding half of the archive descriptor (A2) |
| `--emb-only` | The AST-features half of the archive descriptor (A1) |
| `--static-cells` | Periodic re-clustering (k-means fit once, then frozen) (A3) |
| `--single-prompt-operators` | The six specialized variation prompts (collapse to one per operator) (A6) |
| `--no-crossover` | Crossover (sets `p_crossover = 0`) |
| `--no-targeted-mutate` | The LLM parent-analysis pipeline |
| `--no-advisor` | Consolidation into persistent guidance |
| `--navigator-force-mode M` | Adaptive mode routing (A8) |

### Instrumentation

| Flag | Writes |
|:--|:--|
| `--save-eval-code` | `eval_code_log.jsonl` — the source of **every** candidate, including those that failed to parse, raised, scored invalid or timed out |
| `--error-rate-interval N` | Execution-error rate per N-evaluation window |
| `--checkpoint-population` | `checkpoints/checkpoint_<NN>.json` — the full population at each window close (needs `--error-rate-interval`) |
| `--align-advisor-post-init` | Restart the Advisor clock at the end of init, so Advisor cycle *k* and error-rate window *k* share an origin |

### Run outputs

Each run writes to `--output-dir` (default `outputs/specevo/<task>/<timestamp>/`):

```
best_program.py        the highest-scoring program found
summary.json           score, cost, token accounting, budget, ablation flags
snap.json              search trace: every Navigator call with its routed mode,
                       every new best with its producer (init / speculator /
                       navigator / navigator_variant)
eval_code_log.jsonl    every candidate, with --save-eval-code
```

---

## Baselines

Every baseline runs on the same task specification, evaluator, budget accounting and prompt scaffold —
only the search logic differs. See [`specevo_baselines/README.md`](specevo_baselines/README.md) for the
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
`--checkpoint` (resume), `-l/--log-level`, `--api-base`, `--agentic`. Starter configs live in
[`configs/`](configs/) — see [`configs/README.md`](configs/README.md).

---

## Benchmarks

**176 tasks across four suites.** Each task appears twice: as a `problem.py` adapter under
[`tasks/`](tasks/) (used by SpecEvo and LEVI) and as an `initial_program.py` + evaluator +
`config.yaml` triple under [`benchmarks/`](benchmarks/) (used by the baselines framework). Both score
through the same evaluator, so numbers are comparable across every method.

| Suite | Tasks | Directory | Objective |
|:--|--:|:--|:--|
| **Mathematical discovery** | 7 | `tasks/` · `benchmarks/math/` | Geometric packing, extremal point configurations, signal processing |
| **Systems optimization (ADRS)** | 4 | `tasks/ADRS/` · `benchmarks/ADRS/` | EPLB, PRISM, LLM-SQL, TXN scheduling |
| **CO-Bench** | 36 | `tasks/co_bench/` · `benchmarks/co_bench/` | Combinatorial optimization across 8 categories |
| **LSR-Synth** | 129 | `tasks/llm_srbench/` · `benchmarks/llm_srbench/` | Scientific equation discovery in 4 domains |

<details>
<summary><b>Mathematical discovery — 7 tasks</b></summary>

| Task directory | Configuration | Objective |
|:--|:--|:--|
| `circle_packing` | `N = 26` | Pack disjoint circles in a unit square, maximize sum of radii |
| `circle_packing_rect` | `N = 21` | Same, in a rectangle of perimeter 4 |
| `heilbronn_triangle` | `N = 11` | Maximize the minimum triangle area over points in a unit triangle |
| `heilbronn_convex_13` | `N = 13` | Same, over a convex region |
| `minmax_distance_2` | `(N, d) = (16, 2)` | Maximize `(d_min / d_max)²` over pairwise distances |
| `minmax_distance_3` | `(N, d) = (14, 3)` | Same, in three dimensions |
| `signal_processing` | — | Continuous signal-processing objective under noisy evaluation |

</details>

<details>
<summary><b>Systems optimization (ADRS) — 4 tasks</b></summary>

| Task | Objective |
|:--|:--|
| `eplb` | Expert Parallelism Load Balancer — place MoE expert replicas across GPUs to reduce skew |
| `prism` | Model Placement — assign models to GPUs under memory and capacity constraints |
| `llm_sql` | Reorder SQL table rows/columns to maximize prefix-cache reuse |
| `txn_scheduling` | Order dependent database transactions to minimize makespan |

</details>

<details>
<summary><b>CO-Bench — 36 problems in 8 categories</b></summary>

| Category | # | Problems |
|:--|--:|:--|
| Packing | 9 | Bin packing; multi-demand multidimensional knapsack; multidimensional knapsack; container loading (+ weight restrictions); packing unequal circles / rectangles (number & area) |
| Scheduling | 7 | Aircraft landing; crew scheduling; common due date; flow shop; hybrid reentrant; job shop; open shop |
| Graph and set | 5 | Maximal independent set; graph colouring; equitable partitioning; set partitioning; set covering |
| Cutting | 4 | Assortment; constrained / unconstrained guillotine; constrained non-guillotine |
| Facility location | 4 | Capacitated / uncapacitated warehouse location; capacitated / uncapacitated p-median |
| Routing | 3 | TSP; period vehicle routing; resource-constrained shortest path |
| Assignment | 2 | Constrained / unconstrained assignment |
| Tree | 2 | Euclidean Steiner; corporate structuring |

</details>

<details>
<summary><b>LSR-Synth — 129 equation-discovery problems</b></summary>

| Domain | Problems |
|:--|--:|
| `phys_osc` — physics (oscillators) | 44 |
| `chem_react` — chemistry (reaction kinetics) | 36 |
| `matsci` — material science | 25 |
| `bio_pop_growth` — biology (population growth) | 24 |

Regenerate the per-problem directories after fetching the data:

```bash
python benchmarks/llm_srbench/generate_dirs.py                  # all
python benchmarks/llm_srbench/generate_dirs.py --domain matsci --limit 5
```

`generate_dirs.py` also accepts `--problem PID`, `--iterations N` and `--model MODEL`.

</details>


---

## Reproducing the paper

One script per suite, in [`scripts/reproduce/`](scripts/reproduce/):

```bash
bash scripts/reproduce/math.sh          #   7 tasks
bash scripts/reproduce/adrs.sh          #   4 tasks
bash scripts/reproduce/co_bench.sh      #  36 tasks
bash scripts/reproduce/lsr_synth.sh     # 129 tasks
```

All four read the same environment overrides (defaults shown):

| Variable | Default | Meaning |
|:--|:--|:--|
| `METHOD` | `specevo` | `specevo`, `levi`, or any baseline search key |
| `SPECULATOR_MODEL` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | lightweight model (LEVI's `--small-model`) |
| `NAVIGATOR_MODEL` | `openrouter/openai/gpt-5` | frontier model (also the baselines' `-m`) |
| `EVALS` | `500` | evaluation cap |
| `DOLLARS` | `10` | USD cap |
| `SECONDS_CAP` | `10800` | wall-clock cap |
| `WORKERS` | `4` | parallel workers |
| `SEED` | `1` | run seed (used in the output path) |
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

### The three budget regimes

| Regime | Setting | Paper |
|:--|:--|:--|
| Wall-clock | `SECONDS_CAP=10800` | 3-hour budget |
| API cost | `DOLLARS=10` | \$10 budget |
| Evaluations | `EVALS=500` | 500-call budget |

### Aggregating LSR-Synth results

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
> `W=4`. API-cost and evaluation-budget results are hardware-independent; wall-clock results are not.

---


<!-- ## Citation

The accompanying paper is under double-blind review at ICLR 2027, so the entry is anonymous:

```bibtex
@inproceedings{specevo,
  title     = {SpecEvo: Speculative Evolution with Large Language Models
               for Cost-Efficient Scientific Discovery},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2027},
  note      = {Under review}
}
``` -->