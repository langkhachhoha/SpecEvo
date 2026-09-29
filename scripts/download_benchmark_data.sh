#!/usr/bin/env bash
# Fetch every benchmark payload that is not committed to git.
#
#   bash scripts/download_benchmark_data.sh            # everything
#   bash scripts/download_benchmark_data.sh llm_sql    # one suite
#
# Suites: llm_sql, eplb, lsr_synth
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
[[ -x "$REPO_ROOT/.venv/bin/python" ]] && PY="$REPO_ROOT/.venv/bin/python"

WANT="${1:-all}"
case "$WANT" in
    all|llm_sql|eplb|lsr_synth) ;;
    *) echo "Unknown suite: $WANT. Choose all, llm_sql, eplb, or lsr_synth." >&2; exit 2 ;;
esac
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
    if ! "$PY" -c "import huggingface_hub, pyarrow" 2>/dev/null; then
        echo "ERROR: LSR-Synth needs huggingface_hub and pyarrow to read the" >&2
        echo "       published parquet splits. Install them with:" >&2
        echo "           pip install -e \".[lsr]\"" >&2
        exit 2
    fi
    "$PY" benchmarks/llm_srbench/prepare_data.py
    "$PY" benchmarks/llm_srbench/generate_dirs.py
fi

echo
echo "Benchmark data ready."
