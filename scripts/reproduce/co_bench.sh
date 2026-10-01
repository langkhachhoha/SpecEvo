#!/usr/bin/env bash
# CO-Bench — all 36 combinatorial optimization problems (Table 7).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

shopt -s nullglob
task_dirs=(tasks/co_bench/*/)
if (( ${#task_dirs[@]} == 0 )); then
    echo "ERROR: No CO-Bench task directories found." >&2
    exit 2
fi
for task in "${task_dirs[@]}"; do
    name="$(basename "$task")"
    dispatch "$task" "benchmarks/co_bench/$name" "co_bench_$name"
done
