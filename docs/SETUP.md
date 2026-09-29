# Setup and benchmark data

[Home](../README.md) · [Run options](RUNNING.md) · [Benchmarks](BENCHMARKS.md)

Run all commands from the repository root. Python 3.10–3.13 is supported by the
package metadata; Python 3.11 is recommended for the benchmark dependencies.

## Install

```bash
git clone https://github.com/langkhachhoha/SpecEvo.git
cd SpecEvo
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows, activate with `.venv\Scripts\activate`. With `uv`, the equivalent is:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

The base package includes the search framework and the toy task dependencies.
Install evaluator dependencies for the suites you plan to run:

```bash
python -m pip install -e ".[math,adrs,lsr]"
```

`math` and `adrs` include large scientific packages such as PyTorch and JAX.
Individual evaluators may need additional packages from their own requirements:

```bash
python scripts/install_benchmark_requirements.py benchmarks/math/circle_packing --dry-run
python scripts/install_benchmark_requirements.py benchmarks/math/circle_packing
```

Containerized baseline evaluators also require Docker; see
[benchmark layouts](../benchmarks/README.md#task-layout). External baseline
wrappers use the optional `external` extra; native baselines do not require it.

## API key

```bash
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` to your OpenRouter key. The runners load
this file automatically and mirror an OpenRouter key into `OPENROUTER_API_KEY`.
You can also export `OPENROUTER_API_KEY` in your shell.

Optional live connection check:

```bash
python scripts/test_openrouter_key.py
```

This checks authentication **and sends a small paid chat request**. Offline tests
below do not need a key. Do not commit `.env` or generated logs.

## Benchmark data

CO-Bench includes task data in the repository (some large instance sets are
bounded subsets; see its [data notes](../benchmarks/co_bench/README.md)). Other
payloads are fetched separately:

```bash
bash scripts/download_benchmark_data.sh           # all three downloads
bash scripts/download_benchmark_data.sh llm_sql   # LLM-SQL CSVs
bash scripts/download_benchmark_data.sh eplb      # MoE expert-load workload
bash scripts/download_benchmark_data.sh lsr_synth # LSR-Synth splits + task generation
```

LSR-Synth requires the `lsr` extra. Its default is the public mirror documented
in [LLM-SRBench](../benchmarks/llm_srbench/README.md#data-provisioning); that page
also covers the official dataset and local-file alternatives. Downloaded data
and run outputs are gitignored.

## Check the installation

```bash
python -m pytest -q
python scripts/run_specevo.py --help
python -m specevo_baselines.cli --help
```

After installing benchmark dependencies and downloading data:

```bash
python scripts/check_tasks.py
python scripts/check_tasks.py --suite math  # check just one suite
```

The task checker imports each adapter and validates its interface. It does not
run a complete evaluation or reproduce paper scores. Missing data or evaluator
dependencies can prevent a task from importing.
