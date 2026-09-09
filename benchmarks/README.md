# Benchmarks

The 176 evaluation tasks, in the form the **baselines** framework consumes:
`initial_program.py` + an evaluator + `config.yaml`. The same tasks appear under
[`tasks/`](../tasks/) as `problem.py` adapters for SpecEvo and LEVI. Both forms
score through the same evaluator, so numbers are comparable across every method.

| Suite | Tasks | Domain | What it tests |
|:--|--:|:--|:--|
| [`math/`](math/) | 7 | Mathematical discovery | Circle packing, Heilbronn configurations, min-max point distributions, signal processing |
| [`ADRS/`](ADRS/) | 4 | Systems optimization | MoE load balancing, model placement, SQL column reordering, transaction scheduling |
| [`co_bench/`](co_bench/) | 36 | Combinatorial optimization | Packing, cutting, scheduling, routing, facility location, assignment, tree, graph & set |
| [`llm_srbench/`](llm_srbench/) | 129 | Scientific equation discovery | LSR-Synth across chemistry, biology, physics, material science |

## Setup

```bash
pip install -e ".[math,adrs,lsr]"               # evaluator dependencies
bash scripts/download_benchmark_data.sh         # payloads not kept in git
```

Some task directories carry their own `requirements.txt`:

```bash
pip install -r benchmarks/co_bench/tsp/requirements.txt
python scripts/install_benchmark_requirements.py   # or install them all
```

Verify the result — this imports all 176 tasks and makes no API calls:

```bash
python scripts/check_tasks.py
```

## Running

```bash
# A baseline, plain Python evaluator
python -m specevo_baselines.cli \
  benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  -c benchmarks/math/circle_packing/config.yaml \
  -s openevolve_native -i 500 --dollars 10

# A baseline, containerized evaluator (point at the directory, not the .py)
python -m specevo_baselines.cli \
  benchmarks/math/circle_packing_rect/initial_program.py \
  benchmarks/math/circle_packing_rect/evaluator \
  -c benchmarks/math/circle_packing_rect/config.yaml \
  -s gepa_native -i 500 --dollars 10
```

See the root [README](../README.md#-baselines) for every method and flag.

## Task layout

Two layouts are supported. The framework auto-detects which one a task uses.

### Plain Python evaluator

Runs on the host. Fine for pure-Python tasks with no system dependencies.

```
<task>/
├── initial_program.py   # seed solution
├── evaluator.py         # evaluate(program_path) -> dict, must include combined_score
└── config.yaml          # system prompt + search/evaluator settings
```

```python
def evaluate(program_path: str) -> dict:
    # load and run the program, compute a score
    return {"combined_score": 0.73, ...}
```

On failure, return `{"combined_score": 0.0, "error": "..."}` rather than raising.

### Containerized evaluator

Used where the evaluator needs system packages or bundled data (all of ADRS).
The `evaluator/` directory is the Docker build context.

```
<task>/
├── initial_program.py
├── config.yaml
└── evaluator/
    ├── Dockerfile
    ├── evaluate.sh          # entrypoint: receives <program_path> <mode>
    ├── evaluator.py         # scoring logic
    └── requirements.txt
```

`evaluate.sh` receives the candidate path and a mode — `train` (fast, called
every iteration) or `test` (authoritative, called once on the best program) —
and writes **a single JSON object to stdout**:

```json
{
  "status": "success",
  "combined_score": 0.73,
  "metrics": {"combined_score": 0.73, "accuracy": 0.85},
  "artifacts": {"error": "", "details": ""}
}
```

| Field | Type | Meaning |
|:--|:--|:--|
| `combined_score` | float, required | The primary optimization target |
| `metrics` | dict[str, float] | All numeric scores; must include `combined_score` |
| `artifacts` | dict[str, str] | Non-numeric diagnostics surfaced back to the models |
| `status` | str | `success`, `error`, or `timeout` |

Anything on **stderr** is captured for debugging and ignored by scoring — send
debug output there, never to stdout, or it will corrupt the JSON.

An existing `evaluate(program_path) -> dict` can be wrapped: copy
`specevo_baselines/evaluation/wrapper.py` into `evaluator/` and append

```python
if __name__ == "__main__":
    from wrapper import run
    run(evaluate)
```

Smallest containerized example to copy:
[`math/heilbronn_triangle/`](math/heilbronn_triangle/).

## Seed programs

`initial_program.py` marks the region the search may rewrite:

```python
# EVOLVE-BLOCK-START
def solve(input_data):
    return input_data   # the search will improve this
# EVOLVE-BLOCK-END
```

SpecEvo does not require a seed at all — with no `SEED_PROGRAM`, the Navigator
writes the initial strategies from the problem description alone.
