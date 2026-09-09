# Circle Packing Rect Example

This mirrors the benchmark setup in `benchmarks/math/circle_packing_rect`:

- `n = 21` circles
- rectangle perimeter `4`, so `width + height <= 2`
- objective: maximize `sum(circles[:, 2])`
- candidate API: `circle_packing21() -> np.ndarray` with shape `(21, 3)`

The rectangle is inferred from the returned circles' minimum circumscribing
rectangle. Invalid packings are rejected.

SpecEvo bootstraps the initial programs from the frontier model; no starter
implementation is bundled with this example.

## Run

From the repository root:

## Run

```bash
python scripts/run_specevo.py --task-dir tasks/circle_packing_rect --evals 500 --dollars 10
```

Models default to the paper's split — `--speculator-model` (Qwen3-30B),
`--navigator-model` (GPT-5), `--embedding-model`. API keys are loaded from the
repository `.env`. See the root [README](../../README.md#-running-specevo) for
the full flag reference.
