"""Tiny demo problem for SpecEvo smoke testing.

Coin-change: minimum number of coins to make exactly `target` using
unlimited supply of each denomination. The score rewards correct answers
and lightly penalises slow solutions, so different paradigms (DP, BFS,
greedy, branch-and-bound) all get a chance to compete.
"""

from __future__ import annotations

import time
from typing import Any

PROBLEM_DESCRIPTION = """\
# Coin change (minimum count)

Implement ``solve(coins: list[int], target: int) -> int`` that returns the
minimum number of coins from ``coins`` summing to exactly ``target`` (each
coin may be used unlimited times), or ``-1`` if impossible.

* All test inputs use small positive integer coins (≤ 25) and targets
  in [0, 200].
* The grader runs your function on a hidden batch of ~20 cases.

## Scoring

We report ``score = correct_fraction - 0.05 * mean_runtime_seconds`` so
that a textbook DP and a clever pruned search both land near the top,
while a brute-force enumeration is heavily penalised.
"""

FUNCTION_SIGNATURE = "def solve(coins: list[int], target: int) -> int:"


# Hidden grading set — varied to reward both correctness and efficiency.
_TEST_CASES: list[tuple[list[int], int, int]] = [
    ([1, 2, 5], 11, 3),
    ([2], 3, -1),
    ([1], 0, 0),
    ([1, 3, 4], 6, 2),
    ([186, 419, 83, 408], 6249, 20),
    ([7, 13, 21], 42, 2),
    ([2, 5, 10, 1], 27, 4),
    ([5, 3], 7, -1),  # impossible
    ([1, 2, 5, 10, 20, 50, 100], 199, 7),
    ([6, 7], 13, 2),
    ([2, 4, 6, 8], 10, 2),
    ([9, 6, 5, 1], 11, 2),
    ([3, 4, 5], 27, 6),
    ([2, 7, 11], 22, 2),
    ([1, 5, 25], 64, 8),
    ([2, 5], 11, 4),
    ([10, 25, 1], 30, 3),
    ([14, 17], 31, 2),
    ([2, 4], 5, -1),
    ([3, 7, 12, 25], 100, 4),
]


def _reference(coins: list[int], target: int) -> int:
    """Trustworthy DP we score against. Only used by score_fn."""
    if target <= 0:
        return 0
    INF = float("inf")
    dp = [0] + [INF] * target
    for v in range(1, target + 1):
        best = INF
        for c in coins:
            if c <= v and dp[v - c] + 1 < best:
                best = dp[v - c] + 1
        dp[v] = best
    return -1 if dp[target] == INF else dp[target]


def score_fn(fn: Any, _inputs: list[Any] | None = None) -> dict:
    correct = 0
    total = len(_TEST_CASES)
    elapsed_total = 0.0
    last_error: str | None = None

    for coins, target, expected in _TEST_CASES:
        try:
            t0 = time.perf_counter()
            out = fn(list(coins), int(target))
            elapsed_total += time.perf_counter() - t0
            if int(out) == int(expected):
                correct += 1
        except Exception as e:  # pragma: no cover — captured in score
            last_error = f"{type(e).__name__}: {e}"
            elapsed_total += 0.1  # penalise crashes lightly via time too
            break

    mean_time = elapsed_total / max(1, total)
    correct_frac = correct / total
    score = correct_frac - 0.05 * mean_time

    result: dict[str, Any] = {
        "score": float(score),
        "correct_fraction": float(correct_frac),
        "mean_seconds": float(mean_time),
        "n_correct": int(correct),
        "n_total": int(total),
    }
    if last_error:
        result["error_sample"] = last_error
    return result


# Optional starter — SpecEvo can also bootstrap one without this.
SEED_PROGRAM = """\
def solve(coins, target):
    if target == 0:
        return 0
    INF = float('inf')
    dp = [0] + [INF] * target
    for v in range(1, target + 1):
        for c in coins:
            if c <= v and dp[v - c] + 1 < dp[v]:
                dp[v] = dp[v - c] + 1
    return -1 if dp[target] == INF else dp[target]
"""
