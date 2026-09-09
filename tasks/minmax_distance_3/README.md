# Minimax Pairwise Distance Example (n = 14, d = 3)

Place 14 points in `R^3` so that the ratio of the smallest to the largest
pairwise Euclidean distance is as large as possible. Mirrors the setup in
`benchmarks/math/minimizing_max_min_dist/3`.

Candidate function API:

```python
def min_max_dist_dim3_14() -> np.ndarray:
    # returns points of shape (14, 3)
```

`score_fn` rejects:
- arrays not of shape `(14, 3)` or containing non-finite values,
- coincident point sets where the maximum pairwise distance is zero.

Valid solutions get `score = (d_min / d_max)^2`, and
`combined_score = score / (1 / 4.165849767)` (AlphaEvolve benchmark).
Score is scale- and translation-invariant.

## Run

```bash
python scripts/run_specevo.py --task-dir tasks/minmax_distance_3 --evals 500 --dollars 10
```

Models default to the paper's split — `--speculator-model` (Qwen3-30B),
`--navigator-model` (GPT-5), `--embedding-model`. See the root
[README](../../README.md#-running-specevo) for the full flag reference.
