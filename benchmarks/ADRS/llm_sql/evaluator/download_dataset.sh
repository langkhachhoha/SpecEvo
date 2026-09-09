#!/usr/bin/env bash
# Download CSV datasets for the LLM-SQL benchmark.
#
# Required files (placed in datasets/):
#   movies.csv    - Rotten Tomatoes movie reviews (~9 MB)
#   beer.csv      - Beer review dataset (~2.5 MB)
#   BIRD.csv      - BIRD text-to-SQL dataset (~34 MB)
#   PDMX.csv      - PDMX metadata dataset (~7.4 MB)
#   products.csv  - Amazon product catalog (~16 MB)
#
# Usage:
#   bash benchmarks/ADRS/llm_sql/evaluator/download_dataset.sh

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

BASE_URL="https://huggingface.co/datasets/f20180301/adrs-data/resolve/main/llm_sql"

echo "Downloading LLM-SQL benchmark datasets..."

mkdir -p datasets
for dataset in movies.csv beer.csv BIRD.csv PDMX.csv products.csv; do
    echo "  Downloading datasets/${dataset}..."
    fetch "${BASE_URL}/datasets/${dataset}" "datasets/${dataset}"
done

echo ""
echo "Done. Downloaded files:"
ls -lh datasets/*.csv
