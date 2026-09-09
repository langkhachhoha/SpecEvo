"""
PRISM (GPU Model Placement) Problem Definition.

Contains problem description, prompts, scoring function, and test inputs.
"""

import random
import time
import concurrent.futures
from dataclasses import dataclass

import numpy as np

# --- PRISM Constants ---
GPU_MEM_SIZE = 80  # GB
MIN_INT = float('-inf')


@dataclass
class Model:
    model_name: str
    model_size: int  # GB
    req_rate: int    # requests per second
    slo: int         # service level objective (latency target)
    cur_gpu_id: int  # current GPU assignment (can be ignored)


# --- Prompts ---
PROBLEM_DESCRIPTION = """
# PRISM Problem

## Problem Setting
Optimize the placement of machine learning models across GPUs to minimize the maximum KV Cache Pressure (KVPR). Given a set of models with varying sizes, request rates, and SLO requirements, determine the optimal assignment of models to GPUs while respecting memory constraints.

KVPR (KV Cache Pressure) measures how crowded a GPU is:
```
KVPR = sum(model.req_rate / model.slo for model in gpu_models) / (GPU_MEM_SIZE - sum(model.model_size for model in gpu_models))
```

Lower maximum KVPR across all GPUs is better.

## Target
- **Primary**: Minimize maximum KVPR across all GPUs (lower is better)
- **Hard Constraint**: Models must fit within GPU memory (80GB per GPU)
- **Secondary**: Maximize successful placement rate across test cases

## Scoring (0-100)
```
baseline_kvpr = Average max-KVPR using round-robin placement
optimal_kvpr = Theoretical minimum KVPR with perfect load balance
solution_kvpr = Your solution's average max-KVPR across all test cases

For each test case:
    raw_ratio = (baseline_kvpr - solution_kvpr) / (baseline_kvpr - optimal_kvpr)
    clamped_ratio = clamp(raw_ratio, 0, 1)
    test_score = 100 * sqrt(clamped_ratio)

final_score = Average of individual test scores
```

The sqrt scaling provides diminishing returns as solutions approach optimal, giving more credit for initial improvements over the baseline.

## Implementation Notes
- GPU memory: 80 GB per GPU
- Model sizes: 10-30 GB
- Request rates: 1-10 requests/second
- SLO targets: 5-10 (latency units)
- Number of models per test: 2x gpu_num
- Number of GPUs per test: 5-10
- Each test case has a 10-second timeout
- 50 test cases are evaluated
- Test cases use a fixed random seed (42) for reproducibility
"""

FUNCTION_SIGNATURE = """
from dataclasses import dataclass

GPU_MEM_SIZE = 80  # GB

@dataclass
class Model:
    model_name: str
    model_size: int   # GB
    req_rate: int     # requests per second
    slo: int          # service level objective (latency target)
    cur_gpu_id: int   # current GPU assignment (can be ignored)

def compute_model_placement(gpu_num: int, models: list[Model]) -> dict[int, list[Model]]:
    '''
    Compute optimal model placement across GPUs.

    Args:
        gpu_num: Number of available GPUs (typically 5-10)
        models: List of Model objects to place

    Returns:
        dict mapping gpu_id (int) to list of Models assigned to that GPU
        Example: {0: [model_a, model_b], 1: [model_c], 2: [model_d, model_e]}

    Constraints:
        - Each model must be assigned to exactly one GPU
        - Total model_size on each GPU must not exceed GPU_MEM_SIZE (80GB)

    Goal:
        Minimize max(KVPR) across all GPUs
    '''
    pass
"""

SEED_PROGRAM = '''from dataclasses import dataclass

GPU_MEM_SIZE = 80  # GB

@dataclass
class Model:
    model_name: str
    model_size: int
    req_rate: int
    slo: int
    cur_gpu_id: int

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # Greedy KVPR-minimizing placement based on Algorithm 1 (without τ check)
    # 1) Sort models by r_j / s_j in descending order
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU that minimizes current KVPR while fitting in memory
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')

        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_idx = gpu_id

        # Failure: if no GPU can fit, raise an error instead of overcommitting
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

    return placement
'''

SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# PRISM (GPU Model Placement) Optimization

## Problem
Optimize the placement of machine learning models across GPUs to minimize the maximum
KV Cache Pressure (KVPR). Given models with varying sizes, request rates, and SLO
requirements, determine optimal assignment while respecting memory constraints.

KVPR measures how crowded a GPU is:
```
KVPR = sum(model.req_rate / model.slo for model in gpu_models) / (GPU_MEM_SIZE - sum(model.model_size for model in gpu_models))
```

Lower maximum KVPR across all GPUs is better.

## Key Concepts
- Models must fit within GPU memory (80GB per GPU)
- Each model assigned to exactly one GPU
- Model sizes: 10-30 GB, request rates: 1-10 req/s, SLO targets: 5-10
- Number of models per test: 2x gpu_num, GPUs per test: 5-10
- Scoring uses sqrt scaling: score = 100 * sqrt((baseline - solution) / (baseline - optimal))

## Function Signature
```python
def compute_model_placement(gpu_num: int, models: list[Model]) -> dict[int, list[Model]]:
    '''
    Args:
        gpu_num: Number of available GPUs (typically 5-10)
        models: List of Model objects with model_size, req_rate, slo attributes

    Returns:
        dict mapping gpu_id (int) to list of Models assigned to that GPU
    '''
    pass
```

## You can import standard library modules.

## Your Task: ALGORITHMIC DIVERSITY

Design a solution using a **FUNDAMENTALLY DIFFERENT ALGORITHM** than the existing seeds.

**DO NOT:**
- Make minor variations or parameter tweaks
- Use the same core algorithm with different constants

**DO:**
- Analyze what paradigm each existing seed uses
- Design from first principles using a different strategy
- Consider what information the existing seeds are NOT using

## Existing Seeds:
{existing_seeds}

## Output
Output ONLY the complete Python code in a ```python block.
"""


# --- Test Case Generation ---
_TEST_CASES = None


def _generate_test_cases(num_tests=50):
    """Generate multiple test cases with different characteristics."""
    global _TEST_CASES
    if _TEST_CASES is not None:
        return _TEST_CASES

    test_cases = []
    np.random.seed(42)

    for i in range(num_tests):
        gpu_num = np.random.randint(5, 10)
        gpu_models = []
        for j in range(gpu_num * 2):
            model_size = np.random.randint(10, 30)
            req_rate = np.random.randint(1, 10)
            slo = np.random.randint(5, 10)
            gpu_models.append(Model(
                model_name=f"model_{j}",
                model_size=model_size,
                req_rate=req_rate,
                slo=slo,
                cur_gpu_id=j
            ))
        test_cases.append((gpu_num, gpu_models))

    _TEST_CASES = test_cases
    return test_cases


def get_inputs():
    """Return the test cases as inputs for the scoring function."""
    return _generate_test_cases()


# --- Helper Functions ---
def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=30):
    """Run a function with a timeout using concurrent.futures."""
    if kwargs is None:
        kwargs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def safe_float(value):
    """Convert a value to float safely."""
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def verify_gpu_mem_constraint(placement_data: dict) -> bool:
    """Verify whether models can fit into GPU memory."""
    if placement_data is None:
        return False
    for gpu_id, models in placement_data.items():
        if sum(model.model_size for model in models) > GPU_MEM_SIZE:
            return False
    return True


def calculate_kvcache_pressure(placement_data: dict) -> float:
    """Calculate the maximum KVCache pressure across all GPUs."""
    max_kvpr = MIN_INT
    for gpu_id, models in placement_data.items():
        total_model_size = sum(model.model_size for model in models)
        total_weighted_req_rate = sum(model.req_rate / model.slo for model in models)
        if GPU_MEM_SIZE - total_model_size > 0:
            kvpr = total_weighted_req_rate / (GPU_MEM_SIZE - total_model_size)
        else:
            kvpr = 1000000
        max_kvpr = max(max_kvpr, kvpr)
    return max_kvpr


def round_robin_placement(gpu_num: int, models: list) -> dict:
    """
    Baseline placement: simple round-robin assignment.
    This is the 0-point reference (naive strategy).
    """
    placement = {i: [] for i in range(gpu_num)}
    for i, model in enumerate(models):
        placement[i % gpu_num].append(model)
    return placement


def compute_theoretical_optimal_kvpr(gpu_num: int, models: list) -> float:
    """
    Compute the theoretical optimal (minimum possible) max KVPR.
    This assumes perfect load balancing where all GPUs have equal KVPR.
    This is the 100-point reference (impossible to beat).
    """
    total_weighted_req = sum(m.req_rate / m.slo for m in models)
    total_model_size = sum(m.model_size for m in models)

    # Available memory across all GPUs
    total_available_mem = gpu_num * GPU_MEM_SIZE - total_model_size

    if total_available_mem > 0:
        # Perfect balance: evenly distribute load across available memory
        optimal_kvpr = total_weighted_req / total_available_mem
    else:
        optimal_kvpr = 0.0  # Edge case: no memory available

    return optimal_kvpr


# --- Helper aliases for score function ---
def calculate_kvpr(placement):
    """Alias for calculate_kvcache_pressure."""
    return calculate_kvcache_pressure(placement)


def compute_theoretical_optimal(gpu_num, models):
    """Alias for compute_theoretical_optimal_kvpr."""
    return compute_theoretical_optimal_kvpr(gpu_num, models)


# --- Score Function (with strict validation to prevent reward hacking) ---
def score_fn(compute_model_placement, inputs):
    """Evaluate placement algorithm using the specevo_baselines scoring scheme.

    Matches benchmarks/ADRS/prism/evaluator/evaluator.py:
    - combined_score = 1/avg(max_kvpr) + success_rate  (higher is better)
    - validation failures (wrong shape / missing / duplicate / foreign model /
      memory violation) reject the whole candidate (fail-fast, anti-reward-hack)

    NOTE: this is NOT normalized to [0, 100]; it returns the raw reciprocal of
    the mean per-test max-KVPR plus the success rate, exactly like specevo_baselines.

    Reproducibility / anti-fluke fixes (see git history):

    1. RNG is seeded once per candidate.  Some candidates use the ``random``
       module without seeding it, so their placement — and therefore their
       score — was non-deterministic.  SpecEvo evaluates each candidate exactly
       once and freezes that score in the archive, so an unseeded candidate
       could be admitted on a lucky draw and never reproduce.  Seeding here
       makes every candidate's score deterministic for a fixed test set.

    2. ``avg_kvpr`` is averaged over ALL ``total_tests`` cases, with a test
       that raises (or violates a constraint) contributing a large penalty
       KVPR rather than being silently dropped.  The previous code divided the
       summed KVPR by the number of *passing* tests only: a candidate that
       crashed on 49/50 cases but happened to leave a single very-low-KVPR
       placement on the survivor scored 1/0.0233 + 0.02 ≈ 42.9 — far past the
       legitimate record of ~26.26 — purely because the lone survivor was
       averaged against itself.  Penalising failed tests in the mean closes
       this hole; a candidate that crashes on most cases can no longer be
       inflated by one lucky survivor.
    """
    # FIX 1 — determinism: seed the stdlib + numpy RNGs that candidates may use
    # without seeding themselves.  Matches the seed the test cases are built
    # with so the whole evaluation is reproducible.
    random.seed(42)
    np.random.seed(42)

    try:
        all_kvpr = []
        total_time = 0.0
        successful_runs = 0
        total_tests = len(inputs)

        for gpu_num, models in inputs:
            try:
                # Run solution
                start_time = time.perf_counter()
                result = compute_model_placement(gpu_num, models)
                total_time += time.perf_counter() - start_time
            except Exception:
                # FIX 2 — a test that raises is penalised, not skipped: it
                # contributes a large KVPR to the mean (same sentinel the KVPR
                # helper uses for an over-full GPU) so a candidate cannot be
                # rewarded for crashing on most cases.
                all_kvpr.append(1_000_000.0)
                continue

            # Validate return type
            if not isinstance(result, dict):
                return {"error": f"Expected dict, got {type(result).__name__}"}

            # Validate GPU budget (anti reward-hack): the placement must use at
            # most `gpu_num` GPUs with ids in [0, gpu_num). Without this a
            # candidate can give each model its own GPU, ignoring gpu_num, which
            # spreads load so thinly that max-KVPR collapses and 1/avg_kvpr is
            # inflated far past any legitimate solution (the bug that produced
            # best≈39 vs the real record of ~26). The specevo_baselines evaluator
            # omits this check; we add it so the score stays meaningful.
            if len(result) > gpu_num:
                return {"error": f"Used {len(result)} GPUs but only {gpu_num} allowed"}
            for gpu_id in result:
                if not isinstance(gpu_id, int) or gpu_id < 0 or gpu_id >= gpu_num:
                    return {"error": f"Invalid GPU id {gpu_id!r}; must be int in [0, {gpu_num})"}

            # Validate all models placed exactly once
            placed = []
            for gpu_id, gpu_models in result.items():
                if not isinstance(gpu_models, list):
                    return {"error": f"GPU {gpu_id} value must be list, got {type(gpu_models).__name__}"}
                placed.extend(gpu_models)

            if len(placed) != len(models):
                return {"error": f"Not all models placed: {len(placed)}/{len(models)}"}

            # Validate model uniqueness (prevent reward hacking via duplicate models)
            placed_ids = [id(m) for m in placed]
            if len(set(placed_ids)) != len(placed_ids):
                return {"error": f"Duplicate models detected: {len(placed_ids) - len(set(placed_ids))} duplicates"}
            original_ids = {id(m) for m in models}
            if set(placed_ids) != original_ids:
                return {"error": "Placed models don't match input models (missing or foreign models)"}

            # Validate memory constraints
            for gpu_id, gpu_models in result.items():
                total_size = sum(m.model_size for m in gpu_models)
                if total_size > GPU_MEM_SIZE:
                    return {"error": f"GPU {gpu_id} exceeds memory: {total_size}GB > {GPU_MEM_SIZE}GB"}

            # specevo_baselines metric: per-test max KVCache pressure
            all_kvpr.append(calculate_kvpr(result))
            successful_runs += 1

        if successful_runs == 0:
            return {"max_kvpr": 0.0, "success_rate": 0.0, "score": 0.0, "error": "All test signals failed"}

        # FIX 3 — average over ALL tests, not just the passing ones.  With the
        # penalty pushed into all_kvpr above, len(all_kvpr) == total_tests, so a
        # candidate that crashed on most cases sees its mean KVPR dominated by
        # the penalties and cannot be inflated by a lucky survivor.
        avg_kvpr = sum(all_kvpr) / total_tests
        inv_avg_kvpr = 1.0 / avg_kvpr if avg_kvpr != 0 else 0.0
        success_rate = successful_runs / total_tests
        avg_time = total_time / successful_runs

        return {
            "score": float(inv_avg_kvpr + success_rate),
            "max_kvpr": float(inv_avg_kvpr),
            "success_rate": float(success_rate),
            "num_tests": successful_runs,
            "execution_time": avg_time,
        }

    except Exception as e:
        return {"error": str(e)}


# --- Test Inputs (lazy loaded) ---
INPUTS = None


def get_lazy_inputs():
    """Get inputs, loading them lazily on first access."""
    global INPUTS
    if INPUTS is None:
        INPUTS = get_inputs()
    return INPUTS
