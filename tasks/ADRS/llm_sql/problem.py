"""
LLM SQL (CSV Column Reordering) Problem Definition.

Based on ADRS-Leaderboard: https://ucbskyadrs.github.io/leaderboard
Paper: Optimizing LLM Queries in Relational Workloads (arXiv:2403.05821)

Matches ADRS-Leaderboard evaluator.py:
- Same parameters passed to solve function
- Same col_merge handling
- Same scoring formula
"""

from pathlib import Path
import sys

try:
    import pandas as pd
except ModuleNotFoundError as e:
    if e.name == "pandas":
        raise RuntimeError(
            "LLM SQL requires pandas. From the repo root, run:\n"
            "  python -m pip install -e \".[adrs]\""
        ) from e
    raise

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import evaluate_df_prefix_hit_cnt

# --- Prompts ---

PROBLEM_DESCRIPTION = """
# LLM SQL Problem

Optimize CSV column ordering to maximize prefix hit rate. Reorder columns so concatenated row values have maximum common prefix overlap between consecutive rows.

**Target**: Maximize prefix hit rate, minimize runtime (up to 10s threshold)

Evaluates on: movies.csv, beer.csv, BIRD.csv, PDMX.csv, products.csv

## API (matches ADRS-Leaderboard parameters)

```python
def solve(
    df: pd.DataFrame,
    early_stop: int = 100000,
    row_stop: int = 4,
    col_stop: int = 2,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    '''
    Reorder DataFrame columns to maximize prefix hit rate.

    Args:
        df: Input DataFrame (raw, columns NOT pre-merged)
        early_stop: Early stopping threshold
        row_stop: Row stopping threshold
        col_stop: Column stopping threshold
        col_merge: List of column groups to merge, e.g. [["col1", "col2"], ["col3", "col4"]]
        one_way_dep: One-way dependencies (unused, for compatibility)
        distinct_value_threshold: Threshold for distinct values
        parallel: Whether to use parallel processing

    Returns:
        DataFrame with merged columns, reordered columns and rows
    '''
```

The solve() function must:
1. Apply col_merge: merge specified column groups into single columns (concatenate values, drop originals)
2. Reorder columns and rows to maximize prefix hit rate
3. Return the merged + reordered DataFrame

## Column Merging

For col_merge=[["col1", "col2"]], merge into "col1_col2" column:
```python
df["col1_col2"] = df[["col1", "col2"]].apply(lambda x: "".join([f"{val}" for val in x]), axis=1)
df = df.drop(columns=["col1", "col2"])
```

## Scoring (matches ADRS-Leaderboard evaluator exactly)

```
avg_hit_rate = Your solution's average prefix hit rate (a fraction in [0, 1])
avg_runtime  = Average runtime per dataset (seconds)

# Raw hit rate is used directly — there is NO baseline normalization.
# The runtime term divides by 12.
final_score = 0.95 * avg_hit_rate + 0.05 * (12 - min(12, avg_runtime)) / 12
```

Higher prefix hit rate dominates the score (0.95 weight); runtime is a small
tie-breaker (0.05 weight, saturates at 12s). Losing rows or characters during
reordering => score 0.
"""

FUNCTION_SIGNATURE = """
def solve(
    df: pd.DataFrame,
    early_stop: int = 100000,
    row_stop: int = 4,
    col_stop: int = 2,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    '''
    Reorder DataFrame columns to maximize prefix hit rate.

    Args:
        df: Input DataFrame (raw, columns NOT pre-merged)
        early_stop: Early stopping threshold
        row_stop: Row stopping threshold
        col_stop: Column stopping threshold
        col_merge: List of column groups to merge
        one_way_dep: One-way dependencies (unused)
        distinct_value_threshold: Threshold for distinct values
        parallel: Whether to use parallel processing

    Returns:
        DataFrame with merged columns, reordered columns and rows
    '''
    pass
"""

SEED_PROGRAM = '''import pandas as pd

def solve(
    df: pd.DataFrame,
    early_stop: int = 100000,
    row_stop: int = 4,
    col_stop: int = 2,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    """Reorder columns to maximize prefix sharing."""
    df = df.copy()

    # Apply column merging (required by evaluator)
    if col_merge:
        for cols_to_merge in col_merge:
            if all(col in df.columns for col in cols_to_merge):
                merged_name = "_".join(cols_to_merge)
                df[merged_name] = df[cols_to_merge].apply(
                    lambda x: "".join([f"{val}" for val in x]), axis=1
                )
                df = df.drop(columns=cols_to_merge)

    # Baseline: sort rows by all columns
    df = df.sort_values(by=list(df.columns))
    return df
'''

SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# Column Reordering for Prefix Cache Optimization

Reorder DataFrame columns so consecutive rows share long common prefixes when concatenated.

## Function Signature (ADRS-Leaderboard compatible)
```python
def solve(
    df: pd.DataFrame,
    early_stop: int = 100000,
    row_stop: int = 4,
    col_stop: int = 2,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
```

## OBJECTIVE
Maximize the average prefix hit rate (fraction in [0, 1]); runtime is a small
tie-breaker. Score (matches ADRS-Leaderboard):
`0.95 * avg_hit_rate + 0.05 * (12 - min(12, avg_runtime)) / 12`.
There is NO baseline normalization — raw hit rate is used directly.

## RULES (violations = score 0)
1. MUST apply col_merge first: merge specified column groups into single columns
2. Return DataFrame with SAME rows (same count) and no fewer characters
3. After merging, only change column order and row order
4. No iterrows() or apply(axis=1) on large data - too slow

## Column Merging (REQUIRED)
```python
if col_merge:
    for cols_to_merge in col_merge:
        if all(col in df.columns for col in cols_to_merge):
            merged_name = "_".join(cols_to_merge)
            df[merged_name] = df[cols_to_merge].apply(
                lambda x: "".join([f"{val}" for val in x]), axis=1
            )
            df = df.drop(columns=cols_to_merge)
```

## Your Task
Design a DIFFERENT algorithm than the existing seeds.

## Existing Seeds:
{existing_seeds}

## Output
Output ONLY complete Python code in a ```python block.
"""

META_ADVISOR_PROMPT = """Analyze failures and provide SPECIFIC fixes. Under 100 words.

{metrics_data}

**Fixes:**"""

# --- Dataset Configuration ---

# The CSVs live once in the benchmark tree and are shared with the baseline
# evaluator; a local ``datasets/`` beside this file wins if one is present.
_LOCAL_DATASETS = Path(__file__).parent / "datasets"
_SHARED_DATASETS = (
    Path(__file__).resolve().parents[3]
    / "benchmarks" / "ADRS" / "llm_sql" / "evaluator" / "datasets"
)
DATASETS_DIR = _LOCAL_DATASETS if _LOCAL_DATASETS.is_dir() else _SHARED_DATASETS

# Dataset specs: (filename, col_merge, sample_size)
# col_merge matches ADRS-Leaderboard evaluator.py exactly
DATASET_SPECS = [
    ("movies.csv", [["movieinfo", "movietitle", "rottentomatoeslink"]], None),
    ("beer.csv", [["beer/beerId", "beer/name"]], None),
    ("BIRD.csv", [["PostId", "Body"]], None),
    ("PDMX.csv", [["path", "metadata"], ["hasmetadata", "isofficial", "isuserpublisher", "isdraft", "hasannotations", "subsetall"]], None),
    ("products.csv", [["product_title", "parent_asin"]], None),
]


def load_datasets(sample_size: int = None):
    """Load all datasets WITHOUT column merging (raw DataFrames).

    Args:
        sample_size: If provided, sample each dataset to this many rows.

    Returns:
        List of (df, filename, col_merge) tuples - col_merge passed to solver
    """
    datasets = []
    for filename, col_merge, spec_sample_size in DATASET_SPECS:
        path = DATASETS_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        # Use provided sample_size, fall back to spec_sample_size
        effective_sample = sample_size or spec_sample_size
        if effective_sample and len(df) > effective_sample:
            df = df.sample(effective_sample, random_state=42)
        # Do NOT pre-merge columns - pass raw df + col_merge to solver
        datasets.append((df, filename, col_merge))
    return datasets


class LazyDatasets:
    """Lazily load heavy CSV inputs on first use."""

    def __init__(self, sample_size: int | None = None):
        self._sample_size = sample_size
        self._datasets = None

    def _load(self):
        if self._datasets is None:
            self._datasets = load_datasets(sample_size=self._sample_size)
        return self._datasets

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def __getitem__(self, item):
        return self._load()[item]

    def __repr__(self):
        return repr(self._load())


# Full datasets for final evaluation
INPUTS = LazyDatasets()

# Sampled datasets (1.5K rows each) for quick cascade evaluation
INPUTS_SAMPLED = LazyDatasets(sample_size=1500)


# --- Score Function ---

def score_fn(solve_fn, inputs):
    """Score function matching ADRS-Leaderboard evaluator.py exactly.

    Args:
        solve_fn: Function with signature solve(df, early_stop, row_stop, col_stop, col_merge, ...)
        inputs: List of (df, filename, col_merge) tuples
    """
    import time
    import warnings
    warnings.filterwarnings("ignore")

    try:
        hit_rates = []
        runtimes = []

        for df, filename, col_merge in inputs:
            df_copy = df.copy()
            original_row_count = len(df_copy)

            # Character count of the original DataFrame, used by the canonical
            # evaluator as a data-loss guard (the reordered df must not drop
            # characters). Matches evaluator.py::evaluate total_chars_before.
            total_chars_before = (
                df_copy.astype(str).apply(lambda x: x.str.len().sum(), axis=1).sum()
            )

            # Call solve() with all parameters - matches ADRS-Leaderboard exactly
            start = time.time()
            reordered = solve_fn(
                df_copy,
                early_stop=100000,
                row_stop=4,
                col_stop=2,
                col_merge=col_merge,
                one_way_dep=[],
                distinct_value_threshold=0.7,
                parallel=True,
            )
            runtime = time.time() - start
            runtimes.append(runtime)

            # Validate return type
            if not isinstance(reordered, pd.DataFrame):
                return {"error": f"Expected DataFrame, got {type(reordered).__name__}"}

            # Validate row count (canonical: data lost / duplicated => score 0).
            if len(reordered) != original_row_count:
                diff = len(reordered) - original_row_count
                if diff < 0:
                    return {"error": f"Evaluation failed: row count decreases by {abs(diff)} rows."}
                return {"error": f"Evaluation failed: row count increases by {diff} rows."}

            # Validate character count: reordered must not lose characters.
            # Matches evaluator.py's total_chars_after >= total_chars_before gate.
            total_chars_after = (
                reordered.astype(str).apply(lambda x: x.str.len().sum(), axis=1).sum()
            )
            if total_chars_after < total_chars_before:
                char_diff_pct = (
                    (total_chars_before - total_chars_after) / total_chars_before * 100
                    if total_chars_before > 0 else 0
                )
                return {"error": f"Evaluation failed: character decreases by {char_diff_pct:.2f}%."}

            _, hit_rate = evaluate_df_prefix_hit_cnt(reordered)
            hit_rates.append(hit_rate / 100.0)

        # Canonical scoring (benchmarks/ADRS/llm_sql/evaluator/evaluator.py):
        # combined_score = 0.95 * average_hit_rate + 0.05 * (12 - min(12, rt)) / 12
        # NOTE: the canonical formula uses the RAW average hit rate (a fraction
        # in [0, 1]) directly — it does NOT normalize against a baseline, and
        # the runtime term divides by 12 (not 10). This reproduces leaderboard
        # numbers exactly. `score` is the fitness key SpecEvo/SpecEvo read
        # (utils.coerce_score / orchestrator) — keep it equal to combined_score.
        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        avg_runtime = sum(runtimes) / len(runtimes)
        score = 0.95 * avg_hit_rate + 0.05 * (12 - min(12, avg_runtime)) / 12

        return {
            "score": score,
            "combined_score": score,
            "runs_successfully": 1.0,
            "hit_rates": hit_rates,
            "hit_rate": avg_hit_rate * 100,
            "total_runtime": sum(runtimes),
            "runtime": avg_runtime,
        }
    except MemoryError:
        return {"error": "MemoryError: code used too much memory"}
    except Exception as e:
        return {"error": str(e) or type(e).__name__}
