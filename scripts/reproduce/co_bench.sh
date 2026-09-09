#!/usr/bin/env bash
# CO-Bench — all 36 combinatorial optimization problems (Table 7).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for task in tasks/co_bench/*/; do
    name="$(basename "$task")"
    dispatch "$task" "benchmarks/co_bench/$name" "co_bench_$name"
done
