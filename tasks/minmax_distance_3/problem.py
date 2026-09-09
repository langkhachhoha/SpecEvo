"""
Minimizing the max-to-min pairwise distance ratio in 3D — SpecEvo example.

Task: place 14 points in R^3 so that the ratio
    (min pairwise distance) / (max pairwise distance)
is maximized. Equivalently, minimize the max/min distance ratio.

Mirrors the benchmark at benchmarks/math/minimizing_max_min_dist/3.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy as sp


NUM_POINTS = 14
DIMENSION = 3
BENCHMARK = 1.0 / 4.165849767  # ~0.24004
TIMEOUT_SECONDS = 600


PROBLEM_DESCRIPTION = f"""
# Minimax Pairwise Distance Ratio (n={NUM_POINTS}, d={DIMENSION})

## Problem
Place exactly {NUM_POINTS} points in {DIMENSION}-dimensional Euclidean
space so that the ratio of the **smallest** pairwise distance to the
**largest** pairwise distance is as large as possible (equivalently, the
``d_max / d_min`` ratio is as small as possible).

This is the "well-spread point set" problem: you want the closest pair to
be only modestly closer than the farthest pair. The best known reference
ratio for ({NUM_POINTS}, {DIMENSION}) is
``d_max / d_min ≈ 4.165849767`` (so ``d_min / d_max ≈ 0.24004``).

Return a NumPy-compatible array ``points`` with shape ({NUM_POINTS}, {DIMENSION}),
where row i is the {DIMENSION}-D coordinate of the i-th point.

## Feasibility Constraints
- ``points`` must have shape ({NUM_POINTS}, {DIMENSION}) and contain only
  finite numeric values.
- At least two of the {NUM_POINTS} points must be distinct, otherwise
  ``d_max = 0`` and the score is rejected with an error.
- There are **no coordinate-range constraints**: the score is invariant
  under translation, rotation, reflection, and uniform scaling, so any
  coordinate range or centering you find convenient is fine.

## Objective
Let ``d_min`` and ``d_max`` be the minimum and maximum pairwise Euclidean
distances among the {NUM_POINTS} points. The grader computes

    score = (d_min / d_max) ** 2

(squaring keeps the score in [0, 1] and matches the AlphaEvolve
formulation). The best-known reference is
``score = {BENCHMARK:.16f}``; a ``combined_score`` of 1.0 matches that
value. Higher scores beat the reference.

## Runtime Constraint
Your function must complete within **{TIMEOUT_SECONDS} seconds**.
"""


FUNCTION_SIGNATURE = f"""
import numpy as np
import time
import random
import math

def min_max_dist_dim3_14() -> np.ndarray:
    '''
    Place {NUM_POINTS} points in R^{DIMENSION} so that the ratio of the
    minimum pairwise distance to the maximum pairwise distance is as large
    as possible (equivalently, maximize (d_min / d_max)^2).

    Returns:
        points: numpy array of shape ({NUM_POINTS}, {DIMENSION}) of
            coordinates.
    '''
    pass
"""


def _to_points(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.shape != (NUM_POINTS, DIMENSION):
        raise ValueError(
            f"points must have shape {(NUM_POINTS, DIMENSION)}, got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("points contains non-finite values")
    return arr


def score_fn(min_max_dist_dim3_14, _inputs=None) -> dict:
    """Score a candidate constructor for the 14-point R^3 minimax problem."""
    try:
        start = time.perf_counter()
        raw = min_max_dist_dim3_14()
        exec_time = time.perf_counter() - start

        points = _to_points(raw)

        pairwise = sp.spatial.distance.pdist(points)
        if pairwise.size == 0:
            return {"error": "no pairwise distances"}

        d_min = float(np.min(pairwise))
        d_max = float(np.max(pairwise))

        if d_max <= 0.0:
            return {"error": "max pairwise distance is zero (all points coincide)"}

        inv_ratio_squared = float((d_min / d_max) ** 2)

        return {
            "score": inv_ratio_squared,
            "valid": 1.0,
            "min_max_ratio": inv_ratio_squared,
            "d_min": d_min,
            "d_max": d_max,
            "combined_score": float(inv_ratio_squared / BENCHMARK),
            "execution_time": exec_time,
        }
    except Exception as e:
        return {"error": str(e)}
