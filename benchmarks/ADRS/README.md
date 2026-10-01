# ADRS: AI-Driven Research for Systems

This directory contains four systems optimization benchmarks from the
[ADRS initiative](https://ucbskyadrs.github.io/leaderboard/). Each includes an
initial program, evaluator, and baseline configuration. Matching
[task adapters](../../tasks/ADRS/README.md) support SpecEvo and LEVI.

| Benchmark | Directory | Objective |
|---|---|---|
| Expert Parallelism Load Balancer | [eplb/](eplb/) | Allocate expert replicas across GPUs to balance MoE inference load |
| Model Placement | [prism/](prism/) | Place models on GPUs while reducing the maximum KV-cache pressure |
| LLM-SQL | [llm_sql/](llm_sql/) | Reorder table data to improve prefix-cache reuse |
| Transaction Scheduling | [txn_scheduling/](txn_scheduling/) | Schedule conflicting transactions to reduce completion time |

## Setup

Install the `adrs` dependency extra and fetch data for the tasks that need it:

```bash
python -m pip install -e ".[adrs]"
bash scripts/download_benchmark_data.sh eplb
bash scripts/download_benchmark_data.sh llm_sql
```

See the [EPLB](eplb/README.md) and [LLM-SQL](llm_sql/README.md) setup notes
for their evaluator-specific requirements. The general
[setup guide](../../docs/SETUP.md) covers API credentials.

## Run a benchmark

From the repository root, after setting up API credentials:

```bash
python -m specevo_baselines.cli \
  benchmarks/ADRS/prism/initial_program.py \
  benchmarks/ADRS/prism/evaluator/evaluator.py \
  -c benchmarks/ADRS/prism/config.yaml \
  -s openevolve_native -m openrouter/openai/gpt-5 \
  --api-base https://openrouter.ai/api/v1 \
  -i 100 --dollars 1 -o outputs/adrs/prism
```

This command makes paid model calls. For SpecEvo, use
`python scripts/run_specevo.py --task-dir tasks/ADRS/prism --evals 100 --dollars 1`.
See the [reproduction guide](../../docs/REPRODUCING.md) to run the whole suite.
