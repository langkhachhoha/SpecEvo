# Benchmarks

[Home](../README.md) · [Setup and data](SETUP.md)

**176 tasks across four suites.** Each task appears twice: as a `problem.py` adapter under
[`tasks/`](../tasks/) (used by SpecEvo and LEVI) and as an `initial_program.py` + evaluator +
`config.yaml` triple under [`benchmarks/`](../benchmarks/) (used by the baselines framework). Both score
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
