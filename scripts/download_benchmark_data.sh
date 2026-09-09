#!/usr/bin/env bash
# Fetch every benchmark payload that is not committed to git.
#
#   bash scripts/download_benchmark_data.sh            # everything (~110 MB)
#   bash scripts/download_benchmark_data.sh llm_sql    # one suite
#
# Suites: llm_sql, eplb, lsr_synth
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
[[ -x "$REPO_ROOT/.venv/bin/python" ]] && PY="$REPO_ROOT/.venv/bin/python"

WANT="${1:-all}"
want() { [[ "$WANT" == "all" || "$WANT" == "$1" ]]; }

if want llm_sql; then
    echo "==> LLM-SQL datasets (~69 MB)"
    bash benchmarks/ADRS/llm_sql/evaluator/download_dataset.sh
fi

if want eplb; then
    echo "==> EPLB workload"
    bash benchmarks/ADRS/eplb/evaluator/download_dataset.sh
fi

if want lsr_synth; then
    echo "==> LSR-Synth data + per-problem task directories"
    "$PY" benchmarks/llm_srbench/prepare_data.py
    "$PY" benchmarks/llm_srbench/generate_dirs.py
fi

echo
echo "Benchmark data ready."
