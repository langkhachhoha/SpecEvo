# Tasks

The 176 benchmark tasks in SpecEvo's own format. Each directory holds a
`problem.py` exporting the contract both runners rely on:

| Symbol | Required | Meaning |
|:--|:--:|:--|
| `PROBLEM_DESCRIPTION` | ✅ | Natural-language specification shown to the models |
| `FUNCTION_SIGNATURE` | ✅ | Signature the evolved program must implement |
| `score_fn` | ✅ | Callable returning a metrics dict; `score` is the objective |
| `SEED_PROGRAM` | — | Starting program; omitted, the models write one from scratch |
| `INPUTS` | — | Fixed evaluation inputs (or `get_inputs()` / `get_lazy_inputs()` for lazy loading) |

Anything satisfying this contract can be evolved — your own problems included.

## Contents

| Path | Tasks | Suite |
|:--|--:|:--|
| `circle_packing`, `circle_packing_rect`, `heilbronn_triangle`, `heilbronn_convex_13`, `minmax_distance_2`, `minmax_distance_3`, `signal_processing` | 7 | Mathematical discovery |
| [`ADRS/`](ADRS/) | 4 | Systems optimization |
| [`co_bench/`](co_bench/) | 36 | Combinatorial optimization |
| [`llm_srbench/`](llm_srbench/) | 129 | Scientific equation discovery |
| `smoke_demo` | — | Tiny coin-change task for a cheap end-to-end smoke test |

The same tasks appear under [`benchmarks/`](../benchmarks/) as
`initial_program.py` + `evaluator.py` + `config.yaml`, which is the form the
baselines framework consumes. Both forms score through the same evaluator.

## Running one

There is no per-task driver; one generic runner serves every task:

```bash
python scripts/run_specevo.py --task-dir tasks/circle_packing --evals 500 --dollars 10
```

See the root [README](../README.md#-running-specevo) for the full flag reference.

## Verifying

```bash
python scripts/check_tasks.py              # import-check all 176, no API calls
python scripts/check_tasks.py --suite math
```

A failure here almost always means missing benchmark data — run
`bash scripts/download_benchmark_data.sh`.
