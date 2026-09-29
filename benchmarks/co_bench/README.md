# CO-Bench in SpecEvo Baselines

This directory integrates [**CO-Bench**](https://github.com/sunnweiwei/CO-Bench)
(*Benchmarking Language Model Agents in Algorithm Search for
Combinatorial Optimization*) into SpecEvo Baselines so that CO-Bench problems can be
run by **both** discovery paths in this repo:

- the **baselines** (`openevolve_native`, `gepa_native`, `adaevolve`, `evox`)
  via `python -m specevo_baselines.cli`; and
- **SpecEvo and LEVI** via `scripts/run_specevo.py` and `scripts/run_levi.py`.

The evaluation protocol follows the paper: each problem instance is solved under
a **10-second per-instance time limit**, the raw objective is **normalized
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
└── problem.py               # SpecEvo/LEVI task: PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, score_fn
```

Both paths import the single engine `cobench_eval.py`, so a candidate `solve`
gets scored **identically** whether it is discovered by a baseline or by SpecEvo.

### Dependencies

Beyond `numpy`, CO-Bench needs `scipy` (the `assignment` seed uses
`scipy.optimize.linear_sum_assignment`) and `networkx` (the `mis` task loads
networkx `.gpickle` graphs). Both are core dependencies installed with:

```bash
python -m pip install -e .
```

For task-specific dependencies, inspect the task's `requirements.txt`, or use
`python scripts/install_benchmark_requirements.py benchmarks/co_bench/<slug>`.

### How the engine works (`cobench_eval.py`)

- Loads the vendored task `config.py` and iterates its test-case files. For each
  file, `load_data()` yields instances; each instance runs `solve(**instance)`
  then `eval_func(**instance, **solution)`.
- **Per-instance isolation + timeout.** Instances are evaluated **sequentially**
  (one at a time). Non-daemon callers (the baseline evaluator) run each instance
  in a **forked subprocess** and hard-kill it at the limit; daemon callers (SpecEvo
  workers, which may not spawn child processes) run each instance in-process under
  **`SIGALRM`**. A daemon caller outside the main thread uses a soft thread
  timeout that cannot terminate the candidate thread. Timed-out instances score
  zero. With sequential evaluation, a slow candidate can consume up to
  `instances × timeout` in the normal process/signal paths.
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
`instances × 10s`. Set `COBENCH_MAX_CASES=0` and
`COBENCH_MAX_INSTANCES=0` to evaluate all **locally available** instances.
For tasks with bundled subsets, obtain the remaining upstream data before
comparing against a run that used the complete dataset.

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
python -m specevo_baselines.cli \
  benchmarks/co_bench/tsp/initial_program.py \
  benchmarks/co_bench/tsp/evaluator.py \
  --config benchmarks/co_bench/tsp/config.yaml \
  --search openevolve_native \
  --model openrouter/openai/gpt-5 \
  --iterations 100 --dollars 1 \
  --output outputs/cobench/tsp
```

Swap `tsp` for any slug and `--search` for any of
`openevolve_native | gepa_native | adaevolve | evox`.

### SpecEvo

```bash
python scripts/run_specevo.py \
  --task-dir tasks/co_bench/tsp \
  --speculator-model openrouter/qwen/qwen3-30b-a3b-instruct-2507 \
  --navigator-model openrouter/openai/gpt-5 \
  --n-diverse-seeds 2 --n-variants-per-seed 2 \
  --evals 64 --dollars 1 \
  --output-dir outputs/specevo/cobench_tsp
```

These runs make paid model calls. See the
[reproduction guide](../../docs/REPRODUCING.md) for suite-level runs and the
[baseline guide](../../docs/BASELINES.md) for LEVI and other search methods.

## Seed sanity check (no LLM)

Each problem ships a seed `solve` (Hungarian for the
assignment problem, greedy min-degree for MIS, nearest-neighbour for TSP,
first-fit for bin packing, greedy set-cover, …) — the starting point the search
improves on. Some seeds are infeasible on some instances. The table lists their
strategies; use the evaluation command below to measure scores for your local
instance selection.

| Problem | Seed strategy |
|---------|-------|
| assignment               | Hungarian (scipy) — optimal |
| packing_circles          | greedy grid placement, prefix order |
| bin_packing_1d           | first-fit decreasing |
| pmedian_uncap            | greedy facility-location (numpy) |
| mkp                      | profit/consumption ratio greedy |
| corporate_structuring    | star tree of profitable countries |
| packing_circles_area     | greedy grid, largest-first |
| hybrid_reentrant         | best of a few server permutations |
| packing_rectangles_area  | greedy AABB grid, largest-area-first |
| packing_rectangles       | greedy AABB grid, smallest-first |
| warehouse_location_uncap | assign each customer to cheapest warehouse |
| set_covering             | greedy cost/coverage set cover |
| gap                      | least-consumption + overflow repair |
| flow_shop                | identity permutation |
| aircraft_landing         | greedy runway packing near target time |
| mis                      | greedy minimum-degree independent set |
| tsp                      | nearest-neighbour |
| unconstrained_guillotine | shelf (next-fit) packing |
| common_due_date          | best of identity / SPT / LPT / V-shape |
| rcsp                     | resource-bounded label-setting shortest path |
| graph_coloring           | greedy largest-first |
| job_shop                 | list scheduling (job order) |
| warehouse_location_cap   | open cheapest warehouses + greedy split assign |
| pmedian_cap              | farthest-first medians + capacity-aware assign |
| mdmkp                    | greedy demand-satisfaction then profit fill |
| open_shop                | list scheduling |
| set_partitioning         | greedy non-overlapping exact cover (may be infeasible) |
| period_vrp               | balanced schedule choice + capacity bin-packing |
| container_loading        | best single box type, uniform 3D grid |
| container_loading_weight | best single box type, load-aware column stacking |
| constrained_guillotine   | uniform single-piece guillotine grid |
| crew_scheduling          | greedy arc-chaining (may be infeasible) |
| equitable_partitioning   | balanced greedy 8-way split (perfect instances hard) |
| assortment               | smallest-fitting stock + shelf packing |
| steiner                  | no Steiner points (MST baseline) |
| non_guillotine_cutting   | places nothing — requires the search to construct a feasible solution |

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
