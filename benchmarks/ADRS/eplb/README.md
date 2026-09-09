# Expert Parallelism Load Balancer (EPLB)

This benchmark uses SpecEvo Baselines to optimize the Expert Parallelism Load Balancer (EPLB) algorithm for Mixture-of-Expert (MoE) models. The goal is to rearrange and replicate experts across GPUs to balance load, while keeping the rearrangement algorithm itself fast.

## Setup

1. **Install PyTorch** (required by the evaluator):

   ```bash
   uv pip install torch
   ```

2. **Download the workload file** from [Hugging Face](https://huggingface.co/datasets/abmfy/eplb-openevolve). The evaluator reads `expert-load.json` from next to `evaluator.py`, so use the bundled script (it places the file in the right spot):

   ```bash
   bash benchmarks/ADRS/eplb/evaluator/download_dataset.sh
   ```

3. **Set your API key:**

   ```bash
   export OPENAI_API_KEY=...
   ```

## Run

From the repo root:

```bash
uv run specevo-baselines-run \
  benchmarks/ADRS/eplb/initial_program.py \
  benchmarks/ADRS/eplb/evaluator.py \
  -c benchmarks/ADRS/eplb/config.yaml \
  -s [your_algorithm] \
  -i 100 \
  -o eplb_output
```

Or from this directory:

```bash
uv run specevo-baselines-run initial_program.py evaluator.py \
  -c config.yaml \
  -s [your_algorithm] \
  -i 100
```

## Evaluate a saved program

```bash
python evaluate_best_program.py
```

## Files

| File | Description |
|------|-------------|
| `initial_program.py` | Baseline `rebalance_experts` function to evolve |
| `evaluator.py` | Scores programs on load-balance quality and execution speed |
| `config.yaml` | Task-specific config (LLM, evaluator timeout, system prompt) |
| `evaluate_best_program.py` | Standalone script to evaluate a saved best program |
| `expert-load.json` | Workload data (must be downloaded — see Setup) |
