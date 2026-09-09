"""
Heilbronn Convex problem (n=13) — SpecEvo example.

Task: place 13 points in the plane so that the smallest triangle formed by
any three of them, normalized by the convex-hull area of all 13 points, is
as large as possible.

Mirrors the benchmark at benchmarks/math/heilbronn_convex/13.
"""

from __future__ import annotations

import time
from math import comb
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull


NUM_POINTS = 13
BENCHMARK = 0.030936889034895654
TIMEOUT_SECONDS = 600
NUM_TRIPLES = comb(NUM_POINTS, 3)


PROBLEM_DESCRIPTION = f"""
# Heilbronn Convex Problem (n={NUM_POINTS})

## Problem
Place exactly {NUM_POINTS} points in the plane R^2 so that the smallest
triangle area formed by any 3 of them, **after normalising by the convex-
hull area of all {NUM_POINTS} points**, is as large as possible.

Unlike the fixed-region Heilbronn variant, you choose BOTH the points and
the enclosing convex shape — the convex hull of your output IS the
enclosing region.

Return a NumPy-compatible array ``points`` with shape ({NUM_POINTS}, 2),
where row i is the (x, y) coordinate of the i-th point.

## Feasibility Constraints
- ``points`` must have shape ({NUM_POINTS}, 2) and contain only finite
  numeric values.
- The {NUM_POINTS} points must be in **general position**: their convex
  hull must have strictly positive area. Collinear or fully-duplicated
  point sets are rejected with an error and score 0.
- There are **no coordinate-range constraints**: the score is invariant
  under translation, rotation, reflection, and uniform scaling, so any
  coordinate range you find convenient is fine.

## Objective
Let ``hull_area`` be the area of the convex hull of all {NUM_POINTS}
points, and let ``min_area`` be the minimum area over all {NUM_TRIPLES}
unordered 3-subsets of indices ``{{i, j, k}}`` with
``0 <= i < j < k < {NUM_POINTS}`` (i.e. every triangle formed by three of
your {NUM_POINTS} points). Maximize

    score = min_area / hull_area

The AlphaEvolve / best-known reference is
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

def heilbronn_convex13() -> np.ndarray:
    '''
    Place {NUM_POINTS} points in the plane so that the smallest triangle area
    (normalized by convex-hull area) is maximized.

    Returns:
        points: numpy array of shape ({NUM_POINTS}, 2) of (x, y) coordinates.
    '''
    pass
"""


def _to_points(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.shape != (NUM_POINTS, 2):
        raise ValueError(f"points must have shape {(NUM_POINTS, 2)}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("points contains non-finite values")
    return arr


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(
        np.abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0
    )


def _min_triangle_area(points: np.ndarray) -> float:
    n = len(points)
    best = float("inf")
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < best:
                    best = a
    return float(best)


def score_fn(heilbronn_convex13, _inputs=None) -> dict:
    """Score a candidate constructor for the n=13 Heilbronn-convex problem."""
    try:
        start = time.perf_counter()
        raw = heilbronn_convex13()
        exec_time = time.perf_counter() - start

        points = _to_points(raw)

        try:
            hull = ConvexHull(points)
            hull_area = float(hull.volume)  # `volume` is the 2D area for ConvexHull
        except Exception as e:
            return {"error": f"convex hull failed (likely degenerate): {e}"}

        if hull_area <= 0.0 or not np.isfinite(hull_area):
            return {"error": "convex hull has non-positive area"}

        min_area = _min_triangle_area(points)
        min_area_normalized = float(min_area / hull_area)

        return {
            "score": min_area_normalized,
            "valid": 1.0,
            "min_area_normalized": min_area_normalized,
            "convex_hull_area": hull_area,
            "combined_score": float(min_area_normalized / BENCHMARK),
            "execution_time": exec_time,
        }
    except Exception as e:
        return {"error": str(e)}
