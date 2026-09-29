#!/usr/bin/env bash
# LSR-Synth equation discovery — every problem in every domain.
#
# Regenerate the per-problem directories first if they are missing:
#   python benchmarks/llm_srbench/generate_dirs.py
#
# Set DOMAIN=phys_osc (etc.) to run a single domain.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DOMAIN="${DOMAIN:-*}"
shopt -s nullglob
task_dirs=(tasks/llm_srbench/$DOMAIN/*/)
if (( ${#task_dirs[@]} == 0 )); then
    echo "ERROR: No LSR-Synth tasks match DOMAIN=$DOMAIN. Run the data preparation and directory generation steps first." >&2
    exit 2
fi
for task in "${task_dirs[@]}"; do
    domain="$(basename "$(dirname "$task")")"
    pid="$(basename "$task")"
    name="lsr_${domain}_${pid}"
    dispatch "$task" "benchmarks/llm_srbench/$domain/$pid" "$name"
    "$PY" scripts/lsr_finalize.py \
        --domain "$domain" --problem "$pid" --method "$METHOD" --seed "$SEED" \
        --run-dir "$OUT/$METHOD/$name/seed$SEED" \
        --results "$OUT/$METHOD/seed$SEED/results.jsonl" --allow-failed-program
done

echo "Aggregate the results with: python scripts/lsr_summarize.py $OUT/$METHOD"
