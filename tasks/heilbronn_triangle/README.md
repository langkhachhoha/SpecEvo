# Heilbronn Triangle Example (n = 11)

Place 11 points inside the unit equilateral triangle so that the smallest
triangle formed by any three of them has maximum area. Mirrors the setup in
`benchmarks/math/heilbronn_triangle`.

Candidate function API:

```python
def heilbronn_triangle11() -> np.ndarray:
    # returns points of shape (11, 2)
```

`score_fn` rejects:
- arrays not of shape `(11, 2)` or containing non-finite values,
- points outside the equilateral triangle.

Valid solutions get `score = min_triangle_area / unit_triangle_area`, and
`combined_score = score / 0.0365298898800301...` (AlphaEvolve benchmark).

## Run

```bash
python scripts/run_specevo.py --task-dir tasks/heilbronn_triangle --evals 500 --dollars 10
```

Models default to the paper's split — `--speculator-model` (Qwen3-30B),
`--navigator-model` (GPT-5), `--embedding-model`. See the root
[README](../../README.md#-running-specevo) for the full flag reference.
