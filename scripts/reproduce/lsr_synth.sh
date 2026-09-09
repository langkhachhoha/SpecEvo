#!/usr/bin/env bash
# LSR-Synth equation discovery — every problem in every domain.
#
# Regenerate the per-problem directories first if they are missing:
#   python benchmarks/llm_srbench/generate_dirs.py
#
# Set DOMAIN=phys_osc (etc.) to run a single domain.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DOMAIN="${DOMAIN:-*}"
for task in tasks/llm_srbench/$DOMAIN/*/; do
    domain="$(basename "$(dirname "$task")")"
    pid="$(basename "$task")"
    dispatch "$task" "benchmarks/llm_srbench/$domain/$pid" "lsr_${domain}_${pid}"
done

echo "Aggregate the results with: python scripts/lsr_summarize.py $OUT/$METHOD"
