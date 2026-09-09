<div align="center">

# SpecEvo

### Speculative Evolution with Large Language Models for Cost-Efficient Scientific Discovery

**Speculate → Consolidate.** Many cheap **Speculators** explore in parallel; an **Advisor** distills
the whole trajectory — failures included — into reusable guidance; a frontier **Navigator** is woken
only at the hard junctures.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Tasks](https://img.shields.io/badge/benchmark%20tasks-176-orange.svg)](#-benchmarks)
[![Tests](https://img.shields.io/badge/tests-607%20passing-brightgreen.svg)](#running-the-tests)

[Overview](#-overview) · [Install](#-installation) · [Quickstart](#-quickstart) ·
[Running SpecEvo](#-running-specevo) · [Baselines](#-baselines) · [Benchmarks](#-benchmarks) ·
[Reproducing](#-reproducing-the-paper) · [Layout](#-repository-layout)

</div>

---

## 📖 Overview

LLM-driven evolutionary search is powerful but costly: most frameworks ask a single frontier model to
carry *every* step — routine edits, invalid-program repair, and major strategic changes alike. SpecEvo
splits that work the way a resource-constrained research lab does, and makes the split adapt to the
search state rather than fixing it in advance.

| Role | Plays | Cadence | Model | What it does |
|:--|:--|:--|:--|:--|
| 🔬 **Speculator** | junior researchers | continuous, `W=4` in parallel | lightweight | Draws a parent from the behavioral archive under a stagnation-adaptive Zipf distribution, applies one of six variation prompts, executes the candidate. Handles ~97% of all calls. |
| 📋 **Advisor** | senior labmate | every `Δ_A` evaluations | lightweight | Reads the whole trajectory — working niches, saturated niches, recurring execution failures — and rewrites a ≤300-word guidance note. Injected into each Speculator prompt with probability `ρ_A`. |
| 🧭 **Navigator** | principal investigator | every `Δ_N` evaluations | frontier | Sees a two-resolution view of the archive and makes one strategic intervention, escalating with the stagnation signal. |

The Navigator's intervention mode is chosen by the scalar stagnation signal `s_t`:

| `s_t` | Mode | Context it receives | Requested move |
|:--|:--|:--|:--|
| `≤ γ₁` | **Synthesis** | top 3 niche elites (full code) + 5 descriptions | Combine mechanisms across anchors |
| `γ₁ < s_t ≤ γ₂` | **Surgical** | global champion only + 5 descriptions | One local structural correction |
| `> γ₂` | **Reframe** | top 2 niche elites + 5 descriptions | Introduce an absent strategy family |

Expensive reasoning is therefore *deferred* until enough evidence has accumulated for one intervention
to influence many subsequent low-cost evaluations.

> [!NOTE]
> **Terminology.** The code uses the paper's vocabulary throughout — `speculator_*`, `navigator_*`,
> `advisor_*`, and the modes `synthesis` / `surgical` / `reframe`. The one deliberate exception is the
> body of the prompt templates in [`specevo/engine/prompts.py`](specevo/engine/prompts.py), which is
> kept byte-for-byte as it was sent to the models for the paper's experiments (it still says
> "paradigm shift" in places). Renaming that text would change model behaviour, so it is left alone.

---

## 🚀 Installation

### Requirements

- Python **3.11** or **3.12**
- An [OpenRouter](https://openrouter.ai/) API key — every model in the paper (GPT-5, Kimi-K2, the Qwen3
  family) is reached through one endpoint
- ~92 MB for the repo, plus ~110 MB of benchmark data fetched at setup

### 1. Get the code and an environment

<details open>
<summary><b>With <code>uv</code> (recommended)</b></summary>

```bash
git clone <your-new-repo-url> SpecEvo
cd SpecEvo

uv venv --python 3.11
uv pip install -e ".[dev]"
uv pip install -e ".[math,adrs,lsr]"  # benchmark evaluator dependencies
```

</details>

<details>
<summary><b>With <code>pip</code> / <code>venv</code></b></summary>

```bash
git clone <your-new-repo-url> SpecEvo
cd SpecEvo

python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,math,adrs,lsr]"
```

</details>

### 2. Set your API key

```bash
cp .env.example .env
$EDITOR .env                          # set OPENAI_API_KEY=sk-or-v1-...
```

Both runners load `.env` from the repo root automatically. An OpenRouter key placed in
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

The large CSV/JSON payloads are not committed. One command fetches all of them:

```bash
bash scripts/download_benchmark_data.sh          # all suites (~110 MB)
bash scripts/download_benchmark_data.sh llm_sql  # or just one: llm_sql | eplb | lsr_synth
```

| Suite | What it fetches | Needed for |
|:--|:--|:--|
| `llm_sql` | 5 CSVs (~69 MB) from HuggingFace | ADRS / LLM-SQL |
| `eplb` | `expert-load.json` MoE workload | ADRS / EPLB |
| `lsr_synth` | LSR-Synth splits from the Hub, then regenerates the 129 per-problem task directories (needs the `lsr` extra) | LSR-Synth |

Some benchmarks need extra Python packages; install them per suite:

```bash
python scripts/install_benchmark_requirements.py
```

### 4. Verify the checkout

Import-checks all 176 tasks and makes no API calls — the fastest way to confirm
setup is complete:

```bash
python scripts/check_tasks.py
```

```
  Mathematical discovery             7 /   7
  Systems optimization (ADRS)        4 /   4
  CO-Bench                          36 /  36
  LSR-Synth                        129 / 129

  TOTAL                            176 / 176
```

---

## ⚡ Quickstart

A ~$0.01, few-minute run on a toy task, using a cheap model in **both** roles:

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

---

## 🔧 Running SpecEvo

```bash
python scripts/run_specevo.py --task-dir <task> [options]
```

A task directory is any folder exporting `PROBLEM_DESCRIPTION`, `FUNCTION_SIGNATURE` and `score_fn`
from `problem.py` (optionally `SEED_PROGRAM` / `INPUTS`). All 176 benchmark tasks under
[`tasks/`](tasks/) follow this contract, and your own problems plug in unchanged.

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
| `--navigator-force-mode` | adaptive | Pin one of `synthesis` / `surgical` / `reframe` (ablation) |

### Advisor

| Flag | Default | Meaning |
|:--|:--|:--|
| `--advisor-interval` | `50` | `Δ_A` — rewrite the guidance note every N evaluations |
| `--advisor-inject-p` | `0.35` | `ρ_A` — probability a Speculator prompt carries the note |
| `--advisor-mode` | `rich` | `rich` uses success + saturation + error signals; `errors_only` is the ablation |
| `--no-advisor` | off | Disable the Advisor entirely |

### Ablation switches

| Flag | Removes |
|:--|:--|
| `--ast-only` | The description-embedding half of the archive descriptor |
| `--emb-only` | The AST-features half of the archive descriptor |
| `--static-cells` | Periodic re-clustering (k-means fit once, then frozen) |
| `--single-prompt-operators` | The six specialized variation prompts (collapse to one per operator) |
| `--no-crossover` | Crossover (`p_crossover = 0`) |
| `--no-targeted-mutate` | The LLM parent-analysis pipeline |
| `--no-advisor` | Consolidation into persistent guidance |
| `--navigator-force-mode M` | Adaptive mode routing |

### Instrumentation

| Flag | Writes |
|:--|:--|
| `--save-eval-code` | `eval_code_log.jsonl` — the source of **every** candidate, including those that failed to parse, raised, scored invalid or timed out |
| `--error-rate-interval N` | `error_rate_report.json` — execution-error rate per N-evaluation window |
| `--checkpoint-population` | `checkpoints/checkpoint_NN.json` — the full population at each window close |

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

## 📊 Baselines

Every baseline runs on the same task specification, evaluator, budget accounting and prompt scaffold —
only the search logic differs.

### Heterogeneous-model methods

```bash
# LEVI — diversity preservation + role-aware routing
python scripts/run_levi.py --task-dir tasks/circle_packing --evals 500 --dollars 10

# RelayEvolve — cheap multi-trajectory exploration, one-time handoff to strong refinement
python scripts/run_relay.py --method relayevolve \
    --benchmark-dir benchmarks/math/circle_packing \
    --iterations 500 --dollars 10 --workers 8 --seed 1
```

### Controlled model-allocation strategies

These isolate the effect of *coordination* from the mere availability of two differently priced models.
They share one asynchronous evolutionary backend and differ only in which model gets each call:

| `--method` | Allocation rule |
|:--|:--|
| `all_cheap` | Every call to the lightweight model |
| `all_strong` | Every call to the frontier model |
| `fixed_switch` | Lightweight for the first `B/2` calls, then a one-time switch |
| `random` | Independent coin flip per call |
| `bandit` | UCB1 over the two models, rewarded by global-best improvement |

```bash
python scripts/run_relay.py --method bandit \
    --benchmark-dir benchmarks/math/circle_packing \
    --cheap-model openrouter/qwen/qwen3-30b-a3b-instruct-2507 \
    --strong-model openrouter/openai/gpt-5 \
    --iterations 500 --dollars 10
```

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

Shared flags: `-i/--iterations`, `--dollars`, `-m/--model`, `-c/--config`, `-o/--output`,
`--checkpoint` (resume), `-l/--log-level`. Starter configs live in [`configs/`](configs/).

---

## 🧪 Benchmarks

**176 tasks across four suites.** Each task appears twice: as a `problem.py` adapter under
[`tasks/`](tasks/) (used by SpecEvo and LEVI) and as an `initial_program.py` + `evaluator.py` +
`config.yaml` triple under [`benchmarks/`](benchmarks/) (used by the baselines framework). Both score
through the same evaluator.

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
| `heilbronn_triangle` | `N = 11` | Maximize the minimum triangle area over points in a unit-area triangle |
| `heilbronn_convex_13` | `N = 13` | Same, over a convex region |
| `minmax_distance_2` | `(N, d) = (16, 2)` | Maximize the min/max pairwise-distance ratio |
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
python benchmarks/llm_srbench/generate_dirs.py                 # all
python benchmarks/llm_srbench/generate_dirs.py --domain matsci --limit 5
```

</details>

> [!IMPORTANT]
> The paper reports **175** tasks with 43 physics problems; the shipped LSR-Synth split contains **44**,
> giving 176 here. The extra problem is included rather than silently dropped — pass `--limit` to
> `generate_dirs.py` if you need an exact subset.

---

## 🔁 Reproducing the paper

One script per suite, in [`scripts/reproduce/`](scripts/reproduce/):

```bash
bash scripts/reproduce/math.sh          #   7 tasks
bash scripts/reproduce/adrs.sh          #   4 tasks
bash scripts/reproduce/co_bench.sh      #  36 tasks
bash scripts/reproduce/lsr_synth.sh     # 129 tasks
```

All four take the same environment overrides:

```bash
METHOD=specevo \
SPECULATOR_MODEL=openrouter/qwen/qwen3-30b-a3b-instruct-2507 \
NAVIGATOR_MODEL=openrouter/openai/gpt-5 \
EVALS=500 DOLLARS=10 SECONDS_CAP=10800 WORKERS=4 SEED=1 \
OUT=outputs/repro \
bash scripts/reproduce/math.sh
```

`METHOD` accepts `specevo`, `levi`, or any baseline search key
(`openevolve_native`, `gepa_native`, `adaevolve`, `evox`, `relayevolve`, …); the script dispatches to
the right runner automatically. To sweep a table row, loop over methods and seeds:

```bash
for m in specevo levi relayevolve openevolve_native gepa_native adaevolve evox; do
  for s in 1 2 3; do METHOD=$m SEED=$s bash scripts/reproduce/math.sh; done
done
```

### The three budget regimes

| Regime | Setting | Paper |
|:--|:--|:--|
| Wall-clock | `SECONDS_CAP=10800` | 3-hour budget (Figure 3) |
| API cost | `DOLLARS=10` | \$10 budget (Figure 6) |
| Evaluations | `EVALS=500` | 500-call budget (Table 1) |

### Aggregating LSR-Synth results

```bash
python scripts/lsr_summarize.py outputs/repro/specevo      # NMSE per domain/split
python scripts/lsr_symbolic_accuracy.py outputs/repro/specevo
```

> [!WARNING]
> **Hardware sensitivity.** Wall-clock results depend on the parallelism available to the Speculator
> pool. The paper's runs used a dual-socket server with two 16-core Intel Xeon Silver 4314 CPUs and
> `W=4`. API-cost and evaluation-budget results are hardware-independent; wall-clock results are not.

<details>
<summary><b>Default hyperparameters vs. the paper's appendix</b> — read before reproducing</summary>

Most defaults match Appendix E exactly:

| Parameter | Code default | Paper |
|:--|--:|--:|
| Speculator workers `W` | 4 | 4 |
| Archive clusters `K` | 50 | 50 |
| Init seeds `M` × variants `V` | 5 × 20 | 5 × 20 |
| Advisor interval `Δ_A` | 50 | 50 |
| Advisor injection `ρ_A` | 0.35 | 0.35 |
| Advisor temperature | 0.4 | 0.4 |
| Mode thresholds `(γ₁, γ₂)` | (0.4, 0.7) | (0.4, 0.7) |
| Zipf bounds `(β_min, β_max)` | (0.3, 2.0) | (0.3, 2.0) |
| Global horizon `H_g` | 100 | 100 |

Four defaults differ from the appendix and must be set explicitly for an exact reproduction:

| Parameter | Code default | Paper | Flag |
|:--|--:|--:|:--|
| Navigator interval `Δ_N` | 50 | **20** | `--navigator-interval 20` |
| Navigator fan-out `J` | 4 | **5** | `--n-navigator-variants 5` |
| Recluster interval `Δ_C` | 30 | **50** | `--recluster-every 50` |
| Speculator temperature | 0.8 | **0.7** | — (library default) |

The appendix also specifies per-mode Navigator temperatures (0.5 synthesis, 0.5 surgical, 0.8 reframe);
the implementation uses a single `navigator_temperature = 0.8` for all three modes. These defaults are
left as-published rather than silently changed — set the flags above if you want the appendix values.

</details>

---

## 📁 Repository layout

```
SpecEvo/
├── specevo/                    SpecEvo implementation + the LEVI baseline
│   ├── engine/                 ⭐ the SpecEvo engine
│   │   ├── orchestrator.py       async speculate-then-consolidate loop,
│   │   │                         Navigator routing, Advisor cycle, budgets
│   │   ├── prompts.py            all prompt templates (verbatim, see note above)
│   │   ├── snaplog.py            search trace: Navigator modes, new-best producers
│   │   └── evallog.py            per-candidate log, failures included
│   ├── simple/                 behavioral archive, AST features, embeddings,
│   │                           stagnation monitor, rank sampler
│   ├── methods/
│   │   ├── specevo.py            run_specevo() entry point
│   │   └── levi.py               the LEVI baseline
│   ├── clients/                LLM backends + cost/token accounting
│   └── pipeline/ pool/ …       shared evolutionary infrastructure
│
├── specevo_baselines/          OpenEvolve / GEPA / AdaEvolve / EvoX / RelayEvolve
│   ├── search/                 one package per search method
│   ├── evaluation/             sandboxed evaluator execution
│   └── cli.py                  the baselines command line
│
├── tasks/                      176 task adapters (problem.py) for SpecEvo & LEVI
├── benchmarks/                 the same 176 tasks as evaluator + config for baselines
├── configs/                    starter YAML configs per baseline
├── scripts/
│   ├── run_specevo.py          ⭐ SpecEvo runner
│   ├── run_levi.py             LEVI baseline runner
│   ├── run_relay.py            RelayEvolve + the 4 allocation controls
│   ├── reproduce/              one script per benchmark suite
│   ├── download_benchmark_data.sh
│   ├── check_tasks.py          import-check all 176 tasks (no API calls)
│   ├── test_openrouter_key.py  verify the API key
│   └── lsr_*.py                LSR-Synth result aggregation
│
├── best_programs/              highest-scoring programs per method (paper appendix)
├── tests/                      607 tests (specevo/ and baselines)
└── paper.pdf                   the paper this repository implements
```

### Running the tests

```bash
pytest tests/                   # 607 tests, ~5 min, no API calls (LLMs are mocked)
pytest tests/specevo -q         # SpecEvo engine only
pytest tests/ -m "not slow"     # skip the slow ones
python scripts/check_tasks.py   # all 176 tasks import and expose the contract
```

> [!NOTE]
> The `slow` tests in `tests/specevo/test_integration.py::TestFullRun` run a full
> evolutionary loop against a mocked LLM and evaluate candidates in subprocesses
> under a **5-second** timeout. On a loaded machine, or when that class is run on
> its own from cold, the timeout can fire and the assertions fail — this is
> environment sensitivity, not a broken build. Run the whole suite, or use
> `-m "not slow"`, if you hit it.

Two optional scripts do make live API calls:

```bash
python scripts/smoke_specevo_prompts.py            # render every prompt template
python tests/specevo/engine/format_compliance_live.py   # check model output-format compliance
```

---

## 📄 Citation

```bibtex
@inproceedings{specevo,
  title     = {SpecEvo: Speculative Evolution with Large Language Models
               for Cost-Efficient Scientific Discovery},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2027}
}
```

## 📜 License

[Apache-2.0](LICENSE). Benchmark data retains the license of its original source; see the README in
each benchmark directory.
