# Circle Packing Example (Comparable Setup)

This uses the common benchmark setup:
- `n = 26` circles
- unit square `[0,1] x [0,1]`
- objective: maximize `sum(radii)`

Candidate function API:

```python
def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # returns (centers, radii, sum_radii)
```

The evaluator enforces boundary and non-overlap constraints. Invalid packings
get score `0`.

## Run

```bash
python scripts/run_specevo.py --task-dir tasks/circle_packing --evals 500 --dollars 10
```

Models default to the paper's split — `--speculator-model` (Qwen3-30B),
`--navigator-model` (GPT-5), `--embedding-model`. See the root
[documentation](../../docs/RUNNING.md) for the full flag reference.
