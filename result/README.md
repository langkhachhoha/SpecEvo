# Selected generated programs

These six programs illustrate solutions found for SpecEvo's benchmark tasks.
They are function definitions for the corresponding evaluators, rather than
standalone experiment runners.

| File | Task adapter | Entry point |
|---|---|---|
| [circle_packing.py](circle_packing.py) | [Circle packing](../tasks/circle_packing/) | `run_packing()` |
| [circle_packing_rect.py](circle_packing_rect.py) | [Rectangle packing](../tasks/circle_packing_rect/) | `circle_packing21()` |
| [llm-sql.py](llm-sql.py) | [LLM-SQL](../tasks/ADRS/llm_sql/) | `solve(df, ...)` |
| [mis.py](mis.py) | [Maximum independent set](../tasks/co_bench/mis/) | `solve(**kwargs)` |
| [set_covering.py](set_covering.py) | [Set covering](../tasks/co_bench/set_covering/) | `solve(**kwargs)` |
| [signal_processing.py](signal_processing.py) | [Signal processing](../tasks/signal_processing/) | `run_signal_processing(noisy_signal, window_size)` |

The geometry programs contain their own search loops with a 600-second time
budget. Some programs use randomness, so evaluate them under a recorded seed
and the same evaluator settings when comparing outputs.

Per-run scores and experiment provenance are not bundled with these examples.
See [paper results and their scope](../docs/RESULTS.md) and the
[reproduction guide](../docs/REPRODUCING.md) for the evaluation protocol.
