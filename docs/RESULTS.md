# Results and example programs

The accompanying manuscript evaluates SpecEvo on **176 tasks**: 7 mathematical
discovery tasks, 4 systems optimization tasks, 36 CO-Bench problems, and 129
LSR-Synth problems. The summary below reports manuscript results; running the
repository's setup checks does not reproduce these experiments.

## Main findings

| Evaluation | Reported result | Comparison setting |
|---|---|---|
| Mathematics and systems optimization | Competitive or stronger performance with up to approximately **80% lower API cost** | Three-hour wall-clock budget; frontier-heavy baselines and the API prices used in the paper |
| CO-Bench | **3.8% higher** mean normalized score than the strongest baseline; best or tied-best on **20/36** problems | SpecEvo mean 0.9133 versus LEVI mean 0.8802, rounded from the per-task table |
| LSR-Synth | Lowest NMSE in **8/8** domain/split settings against the six main evolutionary baselines | 500 candidate evaluations; chemistry, biology, physics, and materials, each evaluated on ID and OOD data |

The main LSR-Synth comparison includes OpenEvolve, GEPA, AdaEvolve, EvoX,
RelayEvolve, and LEVI. Those baselines require 715–824 evaluations to match
SpecEvo's result at 500 evaluations; RelayEvolve and LEVI require 768 and 752,
respectively. This measures **evaluation efficiency**, not elapsed-time speedup.

The expanded appendix comparison also includes LLM-SR, LaSR, and SGA. In that
comparison, SpecEvo has the lowest NMSE in **6/8** settings: LLM-SR obtains lower
NMSE on both biology splits. CO-Bench's 20 first-place results include ties
(14 strict wins). API-cost reductions vary by task and backbone; the headline
80% is not an average across all tasks, and does not describe savings relative
to LEVI.

The paper's wall-clock runs used two 16-core Intel Xeon Silver 4314 CPUs and
four Speculators. Runtime depends on hardware, parallelism, evaluator settings,
and API latency. API-cost comparisons use the model prices in the manuscript.

## Included examples

The [`result/`](../result/README.md) directory contains six selected generated
programs:

| Program | Task |
|---|---|
| [circle_packing.py](../result/circle_packing.py) | Pack 26 circles in a unit square |
| [circle_packing_rect.py](../result/circle_packing_rect.py) | Pack 21 circles in a rectangle of perimeter 4 |
| [llm-sql.py](../result/llm-sql.py) | Reorder a table for prefix-cache reuse |
| [mis.py](../result/mis.py) | Maximum independent set |
| [set_covering.py](../result/set_covering.py) | Set covering |
| [signal_processing.py](../result/signal_processing.py) | Filter a noisy signal |

These files are inspectable candidate programs. They do not include the full
experiment logs, per-run scores, model settings, or seed provenance needed to
reconstruct the paper's aggregate tables. Evaluate them with the corresponding
benchmark adapter before assigning a score to a particular file.

For new runs, see the [reproduction guide](REPRODUCING.md),
[benchmark setup](BENCHMARKS.md), and [script index](../scripts/README.md).
