#!/usr/bin/env bash
# Shared setup for the reproduction scripts. Sourced, never run directly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
[[ -x "$REPO_ROOT/.venv/bin/python" ]] && PY="${PY_OVERRIDE:-$REPO_ROOT/.venv/bin/python}"

METHOD="${METHOD:-specevo}"
SPECULATOR_MODEL="${SPECULATOR_MODEL:-openrouter/qwen/qwen3-30b-a3b-instruct-2507}"
NAVIGATOR_MODEL="${NAVIGATOR_MODEL:-openrouter/openai/gpt-5}"
EVALS="${EVALS:-500}"
DOLLARS="${DOLLARS:-10}"
SECONDS_CAP="${SECONDS_CAP:-10800}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-1}"
OUT="${OUT:-outputs/repro}"

if [[ ! -f "$REPO_ROOT/.env" ]]; then
    echo "ERROR: .env not found. Copy .env.example and set OPENAI_API_KEY to your OpenRouter key." >&2
    exit 2
fi

# Run SpecEvo (or the LEVI baseline) on one task directory.
run_task() {
    local task_dir="$1" name="${2:-$(basename "$1")}"
    local out="$OUT/$METHOD/$name/seed$SEED"
    mkdir -p "$out"
    echo "== $METHOD :: $name =="
    # The two runners name their model flags differently: SpecEvo uses the
    # paper's role names, LEVI keeps its own small/large split.
    local script model_flags
    case "$METHOD" in
        specevo)
            script=scripts/run_specevo.py
            model_flags=(--speculator-model "$SPECULATOR_MODEL"
                         --navigator-model  "$NAVIGATOR_MODEL") ;;
        levi)
            script=scripts/run_levi.py
            model_flags=(--small-model "$SPECULATOR_MODEL"
                         --large-model "$NAVIGATOR_MODEL") ;;
        *)  echo "run_task only serves METHOD=specevo|levi; got '$METHOD'" >&2; return 2 ;;
    esac
    "$PY" "$script" \
        --task-dir "$task_dir" \
        "${model_flags[@]}" \
        --evals "$EVALS" --dollars "$DOLLARS" --seconds "$SECONDS_CAP" \
        --workers "$WORKERS" \
        --output-dir "$out" 2>&1 | tee "$out/run.log"
}

# Run one of the specevo_baselines search methods on a benchmark directory.
run_baseline() {
    local bench_dir="$1" name="${2:-$(basename "$1")}"
    local out="$OUT/$METHOD/$name/seed$SEED"
    mkdir -p "$out"
    echo "== $METHOD :: $name =="
    local cfg="$bench_dir/config.yaml"
    [[ -f "$bench_dir/config_${METHOD}.yaml" ]] && cfg="$bench_dir/config_${METHOD}.yaml"

    # Most suites ship evaluator.py next to the seed; ADRS keeps it one level
    # down beside its Dockerfile. Mirrors _resolve_evaluator in run_relay.py.
    local evaluator
    if [[ -f "$bench_dir/evaluator.py" ]]; then
        evaluator="$bench_dir/evaluator.py"
    elif [[ -f "$bench_dir/evaluator/evaluator.py" ]]; then
        evaluator="$bench_dir/evaluator/evaluator.py"
    else
        echo "No evaluator found under $bench_dir" >&2; return 2
    fi

    "$PY" -m specevo_baselines.cli \
        "$bench_dir/initial_program.py" "$evaluator" \
        -c "$cfg" -s "$METHOD" -m "$NAVIGATOR_MODEL" \
        -i "$EVALS" --dollars "$DOLLARS" \
        -o "$out" 2>&1 | tee "$out/run.log"
}

# Dispatch to the right runner for the active METHOD.
dispatch() {
    local task_dir="$1" bench_dir="$2" name="$3"
    case "$METHOD" in
        specevo|levi) run_task "$task_dir" "$name" ;;
        *)            run_baseline "$bench_dir" "$name" ;;
    esac
}
