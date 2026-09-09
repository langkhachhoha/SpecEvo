# Systems optimization (ADRS)

The four systems-optimization tasks used in the paper, drawn from the
[ADRS Leaderboard](https://ucbskyadrs.github.io/leaderboard/) — a benchmark for
LLM-guided algorithm discovery on real systems problems.

Some objectives are naturally minimizations; the task evaluators convert them to
scalar scores where **larger is better**, consistently across every method.

| Task | Objective | Problem |
|:--|:--|:--|
| `eplb/` | Balance expert-parallel workloads across GPUs | Given expert load statistics for a mixture-of-experts model, choose the number and placement of expert replicas across physical GPU slots to reduce computational skew and stragglers, while satisfying placement constraints and preserving inference throughput. |
| `prism/` | Minimize model-to-GPU placement cost | Assign multiple models or model components to a set of GPUs under memory and capacity constraints, minimizing global execution cost or peak resource pressure while remaining feasible and meeting service levels. |
| `llm_sql/` | Maximize prefix-cache efficiency | Reorder table rows, columns, or fields so serialized SQL inputs share longer prefixes and obtain higher prefix-cache reuse — without changing table semantics or query results. |
| `txn_scheduling/` | Minimize transaction-execution makespan | Determine an execution order for dependent or conflicting database transactions that reduces total completion time while respecting dependencies and avoiding excessive conflicts, aborts, or waiting time. |

## Data

`llm_sql` and `eplb` need payloads that are not committed to the repository:

```bash
bash scripts/download_benchmark_data.sh llm_sql
bash scripts/download_benchmark_data.sh eplb
```

`prism` and `txn_scheduling` are self-contained.

## Running

```bash
python scripts/run_specevo.py --task-dir tasks/ADRS/eplb --evals 500 --dollars 10
```

Or the whole suite, for any method:

```bash
bash scripts/reproduce/adrs.sh
METHOD=openevolve_native bash scripts/reproduce/adrs.sh
```

Extra Python dependencies (`torch` for EPLB, `pandas` for LLM-SQL) come from the
`adrs` extra:

```bash
pip install -e ".[adrs]"
```

See the root [README](../../README.md#-running-specevo) for the full flag
reference and the model defaults.
