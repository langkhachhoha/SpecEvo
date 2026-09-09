#!/usr/bin/env bash
# Systems optimization — the 4 ADRS tasks of Table 6.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

dispatch tasks/ADRS/eplb            benchmarks/ADRS/eplb            eplb
dispatch tasks/ADRS/prism           benchmarks/ADRS/prism           prism
dispatch tasks/ADRS/llm_sql         benchmarks/ADRS/llm_sql         llm_sql
dispatch tasks/ADRS/txn_scheduling  benchmarks/ADRS/txn_scheduling  txn_scheduling
