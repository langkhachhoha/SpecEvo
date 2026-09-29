# Running SpecEvo

[Home](../README.md) · [Setup](SETUP.md) · [Reproduction](REPRODUCING.md)

```bash
python scripts/run_specevo.py --task-dir <task> [options]
```

A task directory is any folder whose `problem.py` exports `PROBLEM_DESCRIPTION`,
`FUNCTION_SIGNATURE` and `score_fn` (optionally `SEED_PROGRAM` / `INPUTS`). All 176 benchmark tasks
under [`tasks/`](../tasks/) follow this contract — see [`tasks/README.md`](../tasks/README.md) — and your
own problems plug in unchanged. Use `--problem-module` if your module is not named `problem`.

### Models

| Flag | Default | Paper role |
|:--|:--|:--|
| `--speculator-model` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | `M_S` — lightweight explorer |
| `--navigator-model` | `openrouter/openai/gpt-5` | `M_N` — frontier intervention |
| `--embedding-model` | `openrouter/openai/text-embedding-3-small` | description embeddings for the archive |

The Advisor `M_A` runs on the Speculator model. Swap the backbone family with, e.g.,
`--navigator-model openrouter/moonshotai/kimi-k2-thinking`.

### Budget

| Flag | Meaning |
|:--|:--|
| `--evals N` | Evaluation budget (the paper's 500-call setting) |
| `--dollars N` | API-cost budget in USD (the paper's \$10 setting) |
| `--seconds N` | Wall-clock budget (the paper's 3-hour setting = `10800`) |
| `--target-score X` | Stop early once a candidate reaches `X` |
| `--post-init-evals N` | Budget counted from the *end* of initialization, so the ~105 bootstrap evaluations do not eat into it |

These stopping conditions are unset by default. Specify a budget for every run.
Cost is recorded after requests complete, so concurrent requests already in flight can exceed a dollar threshold.

### Random seed

`--seed N` sets the local search seed (default `0`) and records it in
`summary.json`. Remote model calls and concurrent execution remain stochastic.

### Concurrency

| Flag | Default | Meaning |
|:--|:--|:--|
| `--workers` | `4` | Parallel Speculators (`W` in the paper) |
| `--eval-processes` | `4` | Parallel evaluator processes |
| `--eval-timeout` | `600` | Per-candidate evaluation timeout (seconds) |

### Search structure

| Flag | Default | Paper symbol |
|:--|:--|:--|
| `--n-diverse-seeds` | `5` | `M` — frontier-generated seed strategies |
| `--n-variants-per-seed` | `20` | `V` — lightweight variants per seed |
| `--n-cells` | `50` | `K` — behavioral-archive clusters |
| `--recluster-every` | `30` | `Δ_C` — re-fit k-means every N admits |
| `--embedding-dim` | `8` | PCA dimension of the description-embedding half |

### Navigator

| Flag | Default | Meaning |
|:--|:--|:--|
| `--navigator-interval` | `50` | `Δ_N` — fire the Navigator every N completed evaluations |
| `--n-navigator-variants` | `4` | `J` — Speculator fan-out after each intervention |
| `--navigator-n-anchors` | `4` | Full-code anchors sent to the frontier |
| `--navigator-n-inspirations` | `5` | Description-only inspirations alongside the anchors |
| `--navigator-synthesis-max-stagnation` | `0.4` | `γ₁` — synthesis/surgical boundary |
| `--navigator-surgical-max-stagnation` | `0.7` | `γ₂` — surgical/reframe boundary |
| `--navigator-synthesis-n-anchors` | `3` | Anchors in Synthesis mode |
| `--navigator-reframe-n-anchors` | `2` | Anchors in Reframe mode |
| `--navigator-surgical-n-inspirations` | `5` | Inspirations in Surgical mode |
| `--navigator-force-mode` | adaptive | Pin one of `synthesis` / `surgical` / `reframe` (ablation) |

### Advisor

| Flag | Default | Meaning |
|:--|:--|:--|
| `--advisor-interval` | `50` | `Δ_A` — rewrite the guidance note every N evaluations |
| `--advisor-inject-p` | `0.35` | `ρ_A` — probability a Speculator prompt carries the note |
| `--advisor-mode` | `rich` | `rich` uses success + saturation + error signals; `errors_only` is the ablation |
| `--no-advisor` | off | Disable the Advisor entirely |

### Parent analysis and crossover

| Flag | Default | Meaning |
|:--|:--|:--|
| `--analyzer-interval` | `30` | Refresh cached parent analyses every N evaluations |
| `--analyzer-top-k` | `3` | Top-ranked programs analysed per refresh |
| `--p-targeted-mutate` | `0.5` | Probability of the targeted-mutate prompt when an analysis is cached |
| `--p-crossover` | `0.35` | Crossover probability |

### Ablation switches

| Flag | Removes |
|:--|:--|
| `--ast-only` | The description-embedding half of the archive descriptor (A2) |
| `--emb-only` | The AST-features half of the archive descriptor (A1) |
| `--static-cells` | Periodic re-clustering (k-means fit once, then frozen) (A3) |
| `--single-prompt-operators` | The six specialized variation prompts (collapse to one per operator) (A6) |
| `--no-crossover` | Crossover (sets `p_crossover = 0`) |
| `--no-targeted-mutate` | The LLM parent-analysis pipeline |
| `--no-advisor` | Consolidation into persistent guidance |
| `--navigator-force-mode M` | Adaptive mode routing (A8) |

### Instrumentation

| Flag | Writes |
|:--|:--|
| `--save-eval-code` | `eval_code_log.jsonl` — the source of **every** candidate, including those that failed to parse, raised, scored invalid or timed out |
| `--error-rate-interval N` | Execution-error rate per N-evaluation window |
| `--checkpoint-population` | `checkpoints/checkpoint_<NN>.json` — the full population at each window close (needs `--error-rate-interval`) |
| `--align-advisor-post-init` | Restart the Advisor clock at the end of init, so Advisor cycle *k* and error-rate window *k* share an origin |

### Run outputs

Each run writes to `--output-dir` (default `outputs/specevo/<task>/<timestamp>/`):

```
best_program.py        the highest-scoring program found
summary.json           score, cost, token accounting, budget, ablation flags
snap.json              search trace: every Navigator call with its routed mode,
                       every new best with its producer (init / speculator /
                       navigator / navigator_variant)
eval_code_log.jsonl    every candidate, with --save-eval-code
```
