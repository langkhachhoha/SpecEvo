# Mathematical discovery benchmarks

The seven mathematical-optimization tasks of the paper (Table 5). Six follow the
formulations used in [AlphaEvolve](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)
Appendix B; all evaluators are normalized so that **larger is better**.

| Directory | Config | Objective | AlphaEvolve |
|:--|:--|:--|:--|
| [`circle_packing/`](circle_packing/) | `N = 26` | Pack `N` pairwise-disjoint circles inside a unit square, maximize the sum of radii | B.12 |
| [`circle_packing_rect/`](circle_packing_rect/) | `N = 21` | Same, inside a rectangle of perimeter 4 | B.13 |
| [`heilbronn_triangle/`](heilbronn_triangle/) | `N = 11` | Place `N` points in a unit-area triangle maximizing the minimum area of any triangle they form | B.9 |
| [`heilbronn_convex/13/`](heilbronn_convex/13/) | `N = 13` | Same, over a convex region | B.10 |
| [`minimizing_max_min_dist/2/`](minimizing_max_min_dist/2/) | `(N, d) = (16, 2)` | Find `N` points in ℝ^d maximizing the ratio between minimum and maximum pairwise distance | B.8 |
| [`minimizing_max_min_dist/3/`](minimizing_max_min_dist/3/) | `(N, d) = (14, 3)` | Same, in three dimensions | B.8 |
| [`signal_processing/`](signal_processing/) | — | Optimize a continuous signal-processing objective under noisy evaluation | — |

Each directory holds `initial_program.py`, an evaluator (`evaluator.py`, and for
some tasks a containerized `evaluator/`), and `config.yaml`. Multi-configuration
problems use numbered subdirectories.

## Run

```bash
# SpecEvo (task adapters live under tasks/)
python scripts/run_specevo.py --task-dir tasks/circle_packing --evals 500 --dollars 10

# A baseline
python -m specevo_baselines.cli \
  benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  -c benchmarks/math/circle_packing/config.yaml \
  -s openevolve_native -i 500 --dollars 10

# The whole suite, any method
bash scripts/reproduce/math.sh
```

These tasks need the `math` extra:

```bash
pip install -e ".[math]"
```
