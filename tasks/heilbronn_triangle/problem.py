"""
Heilbronn Triangle problem (n=11) — SpecEvo example.

Task: place 11 points inside the equilateral triangle with vertices
(0,0), (1,0), (0.5, sqrt(3)/2) so that the smallest triangle formed by any
three of them is as large as possible.

Mirrors the benchmark at benchmarks/math/heilbronn_triangle.
"""

from __future__ import annotations

import time
from math import comb
from typing import Any

import numpy as np


NUM_POINTS = 11
BENCHMARK = 0.036529889880030156
TOL = 1e-6
TIMEOUT_SECONDS = 600
NUM_TRIPLES = comb(NUM_POINTS, 3)


PROBLEM_DESCRIPTION = f"""
# Heilbronn Triangle Problem (n={NUM_POINTS})

## Problem
Place exactly {NUM_POINTS} points inside the **unit equilateral triangle**
T with vertices
    A = (0, 0),  B = (1, 0),  C = (0.5, sqrt(3)/2)
so that the smallest triangle area formed by any 3 of the chosen points is
as large as possible.

Return a NumPy-compatible array ``points`` with shape ({NUM_POINTS}, 2),
where row i is the (x, y) coordinate of the i-th point.

## Feasibility Constraints
- ``points`` must have shape ({NUM_POINTS}, 2) and contain only finite
  numeric values.
- Every point (x, y) must lie inside T (closed triangle), within a
  numerical tolerance of {TOL}. Equivalent half-plane test (all three
  must hold):
    * y >= 0                          (bottom edge AB: y = 0)
    * y <= sqrt(3) * x                (left  edge AC: line through (0,0)
                                       and (0.5, sqrt(3)/2))
    * y <= sqrt(3) * (1 - x)          (right edge BC: line through (1,0)
                                       and (0.5, sqrt(3)/2))
- Duplicate or collinear-triple points are allowed numerically, but they
  drive ``min_triangle_area`` to 0 and earn score 0.

## Objective
Let A_T = sqrt(3)/4 be the area of the unit equilateral triangle T, and let
``min_area`` be the minimum area over all {NUM_TRIPLES} unordered
3-subsets of indices ``{{i, j, k}}`` with ``0 <= i < j < k < {NUM_POINTS}``
(i.e. every triangle formed by three of your {NUM_POINTS} points).
Maximize

    score = min_area / A_T

The AlphaEvolve / best-known reference is
``score = {BENCHMARK:.16f}``; a ``combined_score`` of 1.0 matches that
value. Higher scores beat the reference.

## Runtime Constraint
Your function must complete within **{TIMEOUT_SECONDS} seconds**. The
grader executes the function once; you may use any internal
optimisation strategy that fits in this budget.
"""


FUNCTION_SIGNATURE = f"""
import numpy as np
import time
import random
import math

def heilbronn_triangle11() -> np.ndarray:
    '''
    Place {NUM_POINTS} points inside the equilateral triangle with vertices
    (0, 0), (1, 0), (0.5, sqrt(3)/2) so that the smallest triangle area
    among all triples is maximized.

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


def _inside_equilateral(points: np.ndarray, tol: float = TOL) -> tuple[bool, str]:
    sqrt3 = np.sqrt(3.0)
    for idx, (x, y) in enumerate(points):
        if not (y >= -tol):
            return False, f"point #{idx} ({x:.4f}, {y:.4f}) violates y >= 0"
        if not (sqrt3 * x <= sqrt3 - y + tol):
            return False, f"point #{idx} ({x:.4f}, {y:.4f}) violates right edge"
        if not (y <= sqrt3 * x + tol):
            return False, f"point #{idx} ({x:.4f}, {y:.4f}) violates left edge"
    return True, ""


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


def score_fn(heilbronn_triangle11, _inputs=None) -> dict:
    """Score a candidate constructor for the n=11 Heilbronn problem."""
    try:
        start = time.perf_counter()
        raw = heilbronn_triangle11()
        exec_time = time.perf_counter() - start

        points = _to_points(raw)

        valid, reason = _inside_equilateral(points)
        if not valid:
            return {"error": reason}

        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.5, np.sqrt(3.0) / 2.0])
        unit_area = _triangle_area(a, b, c)
        min_area = _min_triangle_area(points)
        min_area_normalized = float(min_area / unit_area)

        return {
            "score": min_area_normalized,
            "valid": 1.0,
            "min_area_normalized": min_area_normalized,
            "combined_score": float(min_area_normalized / BENCHMARK),
            "execution_time": exec_time,
        }
    except Exception as e:
        return {"error": str(e)}
