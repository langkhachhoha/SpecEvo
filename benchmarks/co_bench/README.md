# CO-Bench in SpecEvo Baselines

This directory integrates [**CO-Bench**](https://github.com/sunnweiwei/CO-Bench)
(Sun et al., 2025 — *Benchmarking Language Model Agents in Algorithm Search for
Combinatorial Optimization*) into SpecEvo Baselines so that CO-Bench problems can be
run by **both** discovery paths in this repo:

- the **baselines** (`openevolve_native`, `gepa_native`, `adaevolve`, `evox`)
  via `.github/workflows/baseline.yml` / `specevo-baselines-run`; and
- **SpecEvo / SpecEvo** via `.github/workflows/specevo.yml` / `scripts/run_specevo.py`.

The evaluation protocol follows the paper: each problem instance is solved under
a **hard 10-second per-instance time limit**, the raw objective is **normalized
against the best-known solution** (score `1.0` = best-known / optimal, higher is
better), and any program error, constraint violation, or timeout scores **0.0**
for that instance.

## Integrated problems

We integrate the **complete CO-Bench suite: all 36 problems across 8
categories.**

| Slug | CO-Bench task | Category |
|------|---------------|----------|
| `bin_packing_1d`            | Bin packing - one-dimensional              | Packing |
| `mdmkp`                     | Multi-Demand Multidimensional Knapsack problem | Packing |
| `mkp`                       | Multidimensional knapsack problem          | Packing |
| `container_loading`         | Container loading                          | Packing |
| `container_loading_weight`  | Container loading with weight restrictions | Packing |
| `packing_circles`           | Packing unequal circles                    | Packing |
| `packing_circles_area`      | Packing unequal circles area               | Packing |
| `packing_rectangles`        | Packing unequal rectangles and squares     | Packing |
| `packing_rectangles_area`   | Packing unequal rectangles and squares area | Packing |
| `non_guillotine_cutting`    | Constrained non-guillotine cutting         | Cutting |
| `assortment`                | Assortment problem                         | Cutting |
| `constrained_guillotine`    | Constrained guillotine cutting             | Cutting |
| `unconstrained_guillotine`  | Unconstrained guillotine cutting           | Cutting |
| `warehouse_location_uncap`  | Uncapacitated warehouse location           | Facility location |
| `warehouse_location_cap`    | Capacitated warehouse location             | Facility location |
| `pmedian_cap`               | p-median - capacitated                     | Facility location |
| `pmedian_uncap`             | p-median - uncapacitated                   | Facility location |
| `flow_shop`                 | Flow shop scheduling                       | Scheduling |
| `aircraft_landing`          | Aircraft landing                           | Scheduling |
| `crew_scheduling`           | Crew scheduling                            | Scheduling |
| `common_due_date`           | Common due date scheduling                 | Scheduling |
| `hybrid_reentrant`          | Hybrid Reentrant Shop Scheduling           | Scheduling |
| `job_shop`                  | Job shop scheduling                        | Scheduling |
| `open_shop`                 | Open shop scheduling                       | Scheduling |
| `tsp`                       | Travelling salesman problem                | Routing |
| `period_vrp`                | Vehicle routing: period routing            | Routing |
| `rcsp`                      | Resource constrained shortest path         | Routing |
| `gap`                       | Generalised assignment problem             | Assignment |
| `assignment`                | Assignment problem                         | Assignment |
| `steiner`                   | Euclidean Steiner problem                  | Tree |
| `corporate_structuring`     | Corporate structuring                      | Tree |
| `graph_coloring`            | Graph colouring                            | Graph & set |
| `mis`                       | Maximal independent set                    | Graph & set |
| `equitable_partitioning`    | Equitable partitioning problem             | Graph & set |
| `set_covering`              | Set covering                               | Graph & set |
| `set_partitioning`          | Set partitioning                           | Graph & set |

**Notes on vendored data.** For most tasks all OR-Library instance files are
vendored. For three tasks whose full sets are very large (Maximal independent
set ≈1 GB, Set covering ≈380 MB, Set partitioning ≈46 MB) we vendor a **bounded
subset of the smallest instances** — enough for the default run; the
alphabetically-first files (which the default `MAX_CASES=10` selects) are the
small scored instances. `mis` also carries each case as a **sub-directory** of
`.gpickle` graphs (e.g. `data/Maximal independent set/er_test/`) and requires
**networkx** (declared in `benchmarks/co_bench/mis/requirements.txt`); the
engine's `list_test_cases` lists such sub-directories as cases.

## Layout

```
benchmarks/co_bench/
├── cobench_eval.py          # shared evaluation engine (used by BOTH paths)
├── data/<CO-Bench task>/    # vendored config.py (load_data/eval_func/norm_score/get_dev) + instance files
└── <slug>/                  # baseline benchmark dir, one per problem
    ├── initial_program.py   #   seed `solve` inside an EVOLVE-BLOCK + problem description
    ├── config.yaml          #   specevo-baselines-run config
    ├── evaluator.py         #   evaluate(program_path) -> {combined_score, ...}
    └── requirements.txt

tasks/co_bench/<slug>/
└── problem.py               # SpecEvo/SpecEvo task: PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, score_fn
```

Both paths import the single engine `cobench_eval.py`, so a candidate `solve`
gets scored **identically** whether it is discovered by a baseline or by SpecEvo.

### Dependencies

Beyond `numpy`, CO-Bench needs `scipy` (the `assignment` seed uses
`scipy.optimize.linear_sum_assignment`) and `networkx` (the `mis` task loads
networkx `.gpickle` graphs). These are wired for **both** workflows:

- **baseline.yml** runs `uv sync` (both are in the root `pyproject.toml` core
  deps) **and** installs each task's `requirements.txt`
  (`assignment` → `scipy`, `mis` → `networkx`).
- The SpecEvo and LEVI runners share the repo-root dependency set, whose core deps now
  include `scipy` + `networkx`, so the spawned (daemon) eval workers import them
  cleanly.

Both execution paths are verified end-to-end: the baseline evaluator (fork
subprocess) and the SpecEvo worker (spawn + daemon → in-process `SIGALRM`) produce
identical scores on every task.

### How the engine works (`cobench_eval.py`)

- Loads the vendored task `config.py` and iterates its test-case files. For each
  file, `load_data()` yields instances; each instance runs `solve(**instance)`
  then `eval_func(**instance, **solution)`.
- **Per-instance isolation + timeout.** Instances are evaluated **sequentially**
  (one at a time). Non-daemon callers (the baseline evaluator) run each instance
  in a **forked subprocess** and hard-kill it at the limit; daemon callers (SpecEvo
  workers, which may not spawn child processes) run each instance in-process under
  **`SIGALRM`**. Both enforce the same 10s per-instance limit; a runaway `solve`
  scores 0 and never hangs the search. (Note: with sequential evaluation a
  *valid-but-slow* candidate costs up to `instances × timeout`.)
- Applies the task's `norm_score` (normalize vs. best-known), then splits the
  evaluated instances into a **dev** set (the search signal) and a **disjoint
  test** set (held out). The split is deterministic: flatten every instance in
  file order then instance order and take the first `COBENCH_DEV_FRAC` (7/10 by
  default) as dev, the remaining tail as test. So `combined_score` = `dev_score`
  = mean over the **dev** split (the only number the search optimises);
  `test_score` = mean over the **held-out test** tail (reported for
  generalisation, never optimised); `overall_score` = mean over **every**
  instance (dev + test). Also returns `valid_rate`, `num_dev`, `num_test`. The
  split is uniform across all tasks (the vendored `get_dev` is ignored).

## Runtime knobs (env vars)

| Variable | Meaning | Default |
|----------|---------|---------|
| `COBENCH_TIMEOUT` | per-instance time limit, seconds (paper: 10) | `10` |
| `COBENCH_MAX_CASES` | max test-case files per evaluation (`0` = all) | `10` |
| `COBENCH_MAX_INSTANCES` | max instances per file (`0` = all) | `3` |
| `COBENCH_DEV_FRAC` | fraction of instances in the dev (search) split; rest is held-out test | `0.7` |

**Default = up to 10 files × 3 instances/file** — ≤ 30 instances per problem
(TSP has only 2 files ⇒ 6). Each iteration is usually **dominated by the LLM
call**; evaluation is fast when solves finish or fail quickly, but since
instances run **sequentially**, a *valid-but-slow* candidate can cost up to
`instances × 10s`. To run the **full CO-Bench test set** (faithful to the paper,
slower), set `COBENCH_MAX_CASES=0` and `COBENCH_MAX_INSTANCES=0`.

## Running locally

Set your OpenRouter key (already in `.env`) and route OpenAI-style calls to
OpenRouter:

```bash
set -a; . ./.env; set +a
export OPENAI_API_BASE=https://openrouter.ai/api/v1
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export COBENCH_MAX_CASES=1 COBENCH_MAX_INSTANCES=2   # quick smoke; unset for defaults
```

### Baselines

```bash
specevo-baselines-run \
  benchmarks/co_bench/tsp/initial_program.py \
  benchmarks/co_bench/tsp/evaluator.py \
  --config benchmarks/co_bench/tsp/config.yaml \
  --search openevolve_native \
  --model openrouter/openai/gpt-5 \
  --iterations 100 \
  --output outputs/cobench/tsp
```

Swap `tsp` for any slug and `--search` for any of
`openevolve_native | gepa_native | adaevolve | evox`.

### SpecEvo / SpecEvo

```bash
python scripts/run_specevo.py \
  --example-dir tasks/co_bench/tsp \
  --mutation-model openrouter/qwen/qwen3-30b-a3b-instruct-2507 \
  --paradigm-model openrouter/openai/gpt-5 \
  --evals 64 \
  --output-dir outputs/specevo/cobench_tsp
```

## Running via GitHub Actions

Both workflows accept CO-Bench directly — no code changes needed:

- **Baselines** (`Baselines Smoke`): set `benchmark_dir` to
  `benchmarks/co_bench/<slug>` and pick a `baseline`.
- **SpecEvo** (`SpecEvo`): set `benchmark` to `tasks/co_bench/<slug>`.

Both expose `cobench_timeout`, `cobench_max_cases`, `cobench_max_instances`
inputs (they map to the env vars above; blank = the defaults). They are ignored
by non-CO-Bench benchmarks.

## Seed sanity check (no LLM)

Each problem ships a simple but **feasible** seed `solve` (Hungarian for the
assignment problem, greedy min-degree for MIS, nearest-neighbour for TSP,
first-fit for bin packing, greedy set-cover, …) — the starting point the search
improves on. Approximate seed dev scores (default sample, higher = closer to
best-known; 1.0 = optimal):

| Problem | Seed dev | Seed strategy |
|---------|:--:|-------|
| assignment               | ~1.00 | Hungarian (scipy) — optimal |
| packing_circles          | ~1.00 | greedy grid placement, prefix order |
| bin_packing_1d           | ~0.99 | first-fit decreasing |
| pmedian_uncap            | ~0.99 | greedy facility-location (numpy) |
| mkp                      | ~0.96 | profit/consumption ratio greedy |
| corporate_structuring    | ~0.94 | star tree of profitable countries |
| packing_circles_area     | ~0.94 | greedy grid, largest-first |
| hybrid_reentrant         | ~0.94 | best of a few server permutations |
| packing_rectangles_area  | ~0.95 | greedy AABB grid, largest-area-first |
| packing_rectangles       | ~0.93 | greedy AABB grid, smallest-first |
| warehouse_location_uncap | ~0.90 | assign each customer to cheapest warehouse |
| set_covering             | ~0.90 | greedy cost/coverage set cover |
| gap                      | ~0.86 | least-consumption + overflow repair |
| flow_shop                | ~0.82 | identity permutation |
| aircraft_landing         | ~0.81 | greedy runway packing near target time |
| mis                      | ~0.81 | greedy minimum-degree independent set |
| tsp                      | ~0.80 | nearest-neighbour |
| unconstrained_guillotine | ~0.78 | shelf (next-fit) packing |
| common_due_date          | ~0.77 | best of identity / SPT / LPT / V-shape |
| rcsp                     | ~0.77 | resource-bounded label-setting shortest path |
| graph_coloring           | ~0.75 | greedy largest-first |
| job_shop                 | ~0.64 | list scheduling (job order) |
| warehouse_location_cap   | ~0.63 | open cheapest warehouses + greedy split assign |
| pmedian_cap              | ~0.63 | farthest-first medians + capacity-aware assign |
| mdmkp                    | ~0.61 | greedy demand-satisfaction then profit fill |
| open_shop                | ~0.60 | list scheduling |
| set_partitioning         | ~0.59 | greedy non-overlapping exact cover (~0.7 valid) |
| period_vrp               | ~0.52 | balanced schedule choice + capacity bin-packing |
| container_loading        | ~0.45 | best single box type, uniform 3D grid |
| container_loading_weight | ~0.45 | best single box type, load-aware column stacking |
| constrained_guillotine   | ~0.43 | uniform single-piece guillotine grid |
| crew_scheduling          | ~0.29 | greedy arc-chaining (~0.43 valid — hard feasibility) |
| equitable_partitioning   | ~0.27 | balanced greedy 8-way split (perfect instances hard) |
| assortment               | ~0.19 | smallest-fitting stock + shelf packing |
| steiner                  | ~0.11 | no Steiner points (MST baseline) |
| non_guillotine_cutting   | 0.00  | places nothing — genuinely hard feasibility, for the search to crack |

(`assignment` needs **scipy**; `mis` needs **networkx** — both declared in the
task's `requirements.txt`.) Reproduce:

```bash
python - <<'PY'
import sys; sys.path.insert(0, "benchmarks/co_bench")
import cobench_eval as ce
for slug, task in ce.TASKS.items():
    # baseline seed lives in the initial_program.py of each slug
    src = open(f"benchmarks/co_bench/{slug}/initial_program.py").read()
    r = ce.evaluate_source(task, src, max_cases=3, max_instances=5)
    print(f"{slug:26s} dev={r['score']:.3f} test={r['test_score']:.3f} valid={r['valid_rate']:.2f}")
PY
```

## Adding more CO-Bench problems

1. Download the task from the [CO-Bench HF dataset](https://huggingface.co/datasets/CO-Bench/CO-Bench)
   into `benchmarks/co_bench/data/<task>/` (it must contain `config.py` + instance files).
2. Add a `slug -> task` entry to `TASKS` (and `CATEGORY`) in `cobench_eval.py`.
3. Create `benchmarks/co_bench/<slug>/` (baseline) and
   `tasks/co_bench/<slug>/problem.py` (SpecEvo), mirroring an existing
   problem — only the `TASK` name and the seed `solve` change.
