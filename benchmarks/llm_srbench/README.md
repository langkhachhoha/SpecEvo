# LLM-SRBench — LSR-Synth

Scientific equation discovery from [LLM-SRBench](https://github.com/deep-symbolic-mathematics/llm-srbench)
(Shojaee et al., ICML 2025 — [arXiv:2504.10415](https://arxiv.org/abs/2504.10415)),
wired up so both families of method in this repo can run it:

* **SpecEvo** (`scripts/run_specevo.py`) via `tasks/llm_srbench/<domain>/<pid>/problem.py`
* **baselines** (`specevo-baselines-run`: `openevolve_native`, `gepa_native`, `adaevolve`, `evox`)
  via `benchmarks/llm_srbench/<domain>/<pid>/{initial_program.py, config.yaml, evaluator.py}`

Both paths score candidates through the same engine, `lsr_eval.py`, so a
comparison between them measures search, not scaffolding.

## The benchmark

LSR-Synth is the discovery-driven half of LLM-SRBench: 129 synthetic problems
whose target equations combine *known* scientific terms with *novel synthetic*
ones, so the answer cannot be recited from memory. Four domains:

| domain | problems | task | ids |
| --- | --- | --- | --- |
| `chem_react` | 36 | `dA_dt = f(t, A)` — reaction kinetics | `crk0`… |
| `bio_pop_growth` | 24 | `dP_dt = f(t, P)` — population growth | `bpg0`… |
| `phys_osc` | 44 | `dv_dt = f(x, t, v)` — damped oscillator | `po0`… |
| `matsci` | 25 | `sigma = f(epsilon, T)` — stress/strain | `matsci0`… |

Each problem ships three disjoint splits (paper App. A.2): **4000 train**, **500
in-domain test (ID)**, **500 out-of-domain test (OOD)**. OOD is drawn beyond the
training range — the last time points for the dynamical systems, the highest
temperatures and strains for stress-strain — which is what makes it a test of
whether the discovered equation captures a mechanism rather than a curve.

Not every problem in a domain has the same inputs: LSR-Synth only hands the
discoverer the variables that actually occur in its target equation, so
`phys_osc` mixes `(x, t, v)`, `(x, t)` and `(t, v)` problems. That is why each
problem gets its own directory rather than sharing one per domain.

## Scoring protocol

A hypothesis is an *equation skeleton* — a Python function whose numeric
constants live in a `params` vector:

```python
def equation(t: np.ndarray, A: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[0] * A + params[1] * A**2 + params[2]
```

Scoring one hypothesis (following the paper and LLM-SR's own `eval_spec`):

1. fit `params` (at most 10) on the **train** split — SciPy BFGS on mean squared
   error, starting from `params = [1.0] * 10`;
2. freeze those parameters and predict on **ID** and **OOD**;
3. report NMSE, R², Acc(0.1) and MAPE per split.

**Training NMSE drives the search**, and nothing else. `combined_score` is a
monotone transform of it — by default `log10(1 + 1/train_nmse)`, which reads as
decades of NMSE below 1 (3.0 = 1e-3, 7.0 = 1e-7); `LSR_SCORE_MODE=inv_nmse`
selects the older `1 / (1 + train_nmse)` scale. Either way hypotheses rank as
they do under the paper's own signal (`-MSE`, since NMSE only divides by the
constant `var(y_train)`); see `LSR_SCORE_MODE` in `lsr_eval.py` for why the default changed. **ID and OOD are measured on every evaluation but never used for
selection** — the same dev-drives-search / test-held-out discipline as the
CO-Bench engine.

A hypothesis that raises, returns the wrong shape or length, produces NaN/inf, or
exceeds the per-hypothesis time limit scores 0.0. That limit defaults to **30s**,
the paper's `T = 30s per program hypothesis` (Table 2).

### Metric names

`Acc(0.1)` is the paper's `Acc_tau`: 1.0 only if the **maximum** point-wise
relative error across the whole split is within 10%, else 0.0. It is
all-or-nothing per problem, so on problems whose target passes through zero
(`matsci` at zero strain, for example) a single near-zero point can drive it to 0
for an otherwise excellent equation. The engine therefore also reports
`frac_within` — the fraction of points inside the tolerance — and
`scripts/lsr_summarize.py` prints both side by side (`Acc` and `w/in`). Read them
together.

## Data provisioning

```bash
python benchmarks/llm_srbench/prepare_data.py            # all 129 problems
python benchmarks/llm_srbench/prepare_data.py --check     # verify what's on disk
```

This writes `data/problems.json` plus `data/<domain>/<pid>.npz` (~9 MB total,
gitignored). Needs `huggingface_hub` and `pyarrow`.

The canonical upload `nnheui/llm-srbench` is a **gated** HuggingFace dataset
(access is granted per account), which makes unattended server setup impossible,
so the default source is the ungated mirror `pkuHaowei/llm-srbench`. Before
adopting it the data was checked against the paper: substituting the published
ground-truth equations into the mirror's inputs reproduces its outputs to
NMSE ≈ 1e-13 (float32 storage noise) for `bpg0` (Fig. 14) and `matsci0`
(Fig. 16), and the problem counts match Table 4. Use the official copy instead
with `--repo nnheui/llm-srbench` if your account has access, or
`--from-local DIR` if you copied the parquet files onto an offline machine.

One caveat: a few of the mirror's `gt_expression` *strings* have mangled constant
names (`0.189…_z` where the original had `k_z`), an artefact of substituting
numeric values into symbolic constants. The strings are recorded for reference
only; nothing in the scoring path parses them, and the numeric data is intact.

## Directory layout

```text
benchmarks/llm_srbench/
├── lsr_eval.py          shared engine: data, BFGS fit, ID/OOD metrics, timeouts
├── prepare_data.py      HuggingFace -> data/<domain>/<pid>.npz + problems.json
├── generate_dirs.py     writes the per-problem directories below
├── data/                materialised dataset (gitignored)
└── <domain>/<pid>/      initial_program.py, config.yaml, evaluator.py,
                         download_dataset.sh

tasks/llm_srbench/<domain>/<pid>/problem.py
```

Everything under `<domain>/<pid>/` is generated and delegates to `lsr_eval.py`;
regenerate after an engine change with:

```bash
python benchmarks/llm_srbench/generate_dirs.py --limit 10   # first 10 per domain
python benchmarks/llm_srbench/generate_dirs.py              # every problem
```

## Verifying the integration

```bash
python scripts/check_tasks.py --suite lsr_synth
```

Imports every generated problem and checks it exposes the runner contract. No
API calls.

## Running

Use `scripts/reproduce/lsr_synth.sh` — it loops over every problem in a
domain. Set `DOMAIN=phys_osc` to restrict it to one domain, and see the root
[documentation](../../docs/REPRODUCING.md) for the other overrides.

```bash
# SpecEvo across one domain
DOMAIN=chem_react bash scripts/reproduce/lsr_synth.sh

# A baseline across one domain
METHOD=evox DOMAIN=matsci bash scripts/reproduce/lsr_synth.sh

# Aggregate ID/OOD across everything that has finished
python scripts/lsr_summarize.py outputs/repro/specevo
```

A single problem runs through the generic runners, since the layout is the
standard one:

```bash
python scripts/run_specevo.py --task-dir tasks/llm_srbench/matsci/matsci0 --evals 500

python -m specevo_baselines.cli \
  benchmarks/llm_srbench/matsci/matsci0/initial_program.py \
  benchmarks/llm_srbench/matsci/matsci0/evaluator.py \
  -c benchmarks/llm_srbench/matsci/matsci0/config.yaml \
  -s evox -i 500
```

## Environment knobs

| variable | default | meaning |
| --- | --- | --- |
| `LSR_EVAL_TIMEOUT` | `30` | seconds per equation hypothesis (paper's `T`) |
| `LSR_MAX_NPARAMS` | `10` | size of the `params` vector |
| `LSR_MAX_FIT_POINTS` | `0` (all) | subsample train before fitting — smoke tests only, it changes the benchmark |
| `LSR_BFGS_MAXITER` | `0` (SciPy default) | cap BFGS iterations |
| `LSR_DATA_DIR` | `benchmarks/llm_srbench/data` | where the `.npz` files live |

## Symbolic accuracy (SA)

The paper's other axis asks GPT-4o whether a discovered skeleton is
mathematically equivalent to the ground truth. It is scored *after* the search,
from what `results.jsonl` already holds (`gt_expression` + `best_program`), so it
never re-runs a search and can be re-run when a new baseline lands:

```bash
python scripts/lsr_symbolic_accuracy.py outputs/repro/specevo
python scripts/lsr_symbolic_accuracy.py outputs/repro/specevo --dry-run   # no API calls
```

The judge is asked the paper's question (App. B, Fig. 11) — *do there exist
constant parameter values that make the hypothesis equivalent to the ground
truth?* — which is why the program can go over as-is: its `params[i]` are free
constants. Following App. B.2 the prompt is pre-processed by stripping comments
and docstrings from the program, and by replacing the ground truth's fitted
coefficients with placeholder symbols (`--gt-constants`), so the verdict turns on
structure and not on whether a coefficient came out as 0.1899 or 0.19. Exponents
are left literal: `A(t)**2` and `A(t)**3` are different hypotheses. This also
absorbs the mangled `0.189…_z` constant names noted above — the whole token is
one placeholder.

Per-domain SA is the fraction of *attempted* problems judged equivalent;
problems that produced no usable equation stay in the denominator. Judgements
(verdict + reasoning + tokens) are cached in
`outputs/symbolic_accuracy/judgments.jsonl` and reused, so re-running is free and
an interrupted sweep resumes.

## What is not implemented

**LSR-Transform.** Only the LSR-Synth half of LLM-SRBench is wired up here; the
111 transformed-Feynman problems are not.
