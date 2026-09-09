#!/usr/bin/env bash
# Download the workload file for the EPLB benchmark.
#
# Required file (placed next to evaluator.py, where it is read from):
#   expert-load.json   - MoE logical expert load history (~rebalance workloads)
#
# Usage:
#   bash benchmarks/ADRS/eplb/evaluator/download_dataset.sh

set -euo pipefail
cd "$(dirname "$0")"

# Portable download: prefer curl, fall back to wget. (CI uses ubuntu-latest
# which has both; macOS ships curl but not wget.)
fetch() {  # fetch <url> <out>
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors \
            -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --tries=5 --waitretry=5 --retry-on-http-error=429,500,502,503,504 \
            -O "$2" "$1"
    else
        echo "Need curl or wget to download datasets." >&2
        exit 1
    fi
}

BASE_URL="https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main"

echo "Downloading EPLB benchmark workload..."
echo "  Downloading expert-load.json..."
fetch "${BASE_URL}/expert-load.json" expert-load.json

echo ""
echo "Done. Downloaded files:"
ls -lh expert-load.json
