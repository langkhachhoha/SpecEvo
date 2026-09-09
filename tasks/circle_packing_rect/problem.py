"""
Circle packing in a rectangle of perimeter 4.

Task: place n=21 non-overlapping circles inside any rectangle whose perimeter
is at most 4, maximizing the sum of radii. The rectangle is inferred from the
minimum circumscribing rectangle of the returned circles, matching the
benchmark evaluator in benchmarks/math/circle_packing_rect.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


NUM_CIRCLES = 21
BENCHMARK_RADII_SUM = 2.3658321334167627
MAX_WIDTH_PLUS_HEIGHT = 2.0
TOL = 1e-6
TIMEOUT_SECONDS = 600


PROBLEM_DESCRIPTION = f"""
# Circle Packing in a Rectangle (n=21, perimeter=4)

## Problem
Construct a feasible packing of exactly {NUM_CIRCLES} circles inside a
rectangle of perimeter at most 4. The rectangle itself is not returned; the
grader computes the minimum circumscribing rectangle around your circles.

Return a NumPy-compatible array `circles` with shape ({NUM_CIRCLES}, 3), where
each row is `(x, y, radius)`.

## Feasibility Constraints
- `circles` must have shape ({NUM_CIRCLES}, 3), finite numeric values.
- Radii must be non-negative.
- For every pair i != j:
  `distance((x_i, y_i), (x_j, y_j)) >= r_i + r_j`.
- Let `(width, height)` be the minimum circumscribing rectangle of all returned
  circles. It must satisfy `width + height <= {MAX_WIDTH_PLUS_HEIGHT}` because
  the rectangle perimeter is `2 * (width + height) <= 4`.

Coordinates may be translated freely; only the inferred bounding rectangle and
pairwise distances matter.

## Objective
Maximize `radii_sum = sum(circles[:, 2])`.
The AlphaEvolve benchmark reference is {BENCHMARK_RADII_SUM:.16f}; a
`combined_score` of 1.0 matches that value.

## Runtime Constraint
Your solution must complete within **{TIMEOUT_SECONDS} seconds**.
"""


FUNCTION_SIGNATURE = f"""
import numpy as np
import time

def circle_packing21() -> np.ndarray:
    '''
    Places {NUM_CIRCLES} non-overlapping circles inside a rectangle of
    perimeter 4, maximizing the sum of their radii.

    Returns:
        circles: numpy array of shape ({NUM_CIRCLES}, 3), where each row
            stores (x, y, radius).
    '''
    pass
"""


def _to_circles(data: Any) -> np.ndarray:
    circles = np.asarray(data, dtype=float)
    if circles.shape != (NUM_CIRCLES, 3):
        raise ValueError(f"circles must have shape {(NUM_CIRCLES, 3)}, got {circles.shape}")
    if not np.isfinite(circles).all():
        raise ValueError("circles contains non-finite values")
    return circles


def minimum_circumscribing_rectangle(circles: np.ndarray) -> tuple[float, float]:
    """Return (width, height) of the minimum rectangle enclosing all circles."""
    min_x = float(np.min(circles[:, 0] - circles[:, 2]))
    max_x = float(np.max(circles[:, 0] + circles[:, 2]))
    min_y = float(np.min(circles[:, 1] - circles[:, 2]))
    max_y = float(np.max(circles[:, 1] + circles[:, 2]))
    return max_x - min_x, max_y - min_y


def evaluate_packing_output(circles: np.ndarray) -> tuple[bool, str]:
    """Return (is_valid, reason), matching the benchmark constraints."""
    radii = circles[:, 2]
    if np.any(radii < 0.0):
        return False, "Negative radius"

    for i in range(NUM_CIRCLES):
        for j in range(i + 1, NUM_CIRCLES):
            dist = float(np.linalg.norm(circles[i, :2] - circles[j, :2]))
            if dist + TOL < float(radii[i] + radii[j]):
                return False, f"Overlap between circles {i} and {j}"

    width, height = minimum_circumscribing_rectangle(circles)
    if width + height > MAX_WIDTH_PLUS_HEIGHT + TOL:
        return False, "Circles are not contained inside a rectangle of perimeter 4"

    return True, ""


def compute_behavior_descriptors(circles: np.ndarray) -> dict[str, float]:
    """Compute geometry descriptors useful for SpecEvo archive diversity."""
    radii = circles[:, 2]
    width, height = minimum_circumscribing_rectangle(circles)

    if min(width, height) <= 0.0:
        aspect_ratio = 0.0
    else:
        aspect_ratio = float(max(width, height) / min(width, height))

    min_x = float(np.min(circles[:, 0] - radii))
    max_x = float(np.max(circles[:, 0] + radii))
    min_y = float(np.min(circles[:, 1] - radii))
    max_y = float(np.max(circles[:, 1] + radii))
    margins = np.stack(
        [
            circles[:, 0] - radii - min_x,
            circles[:, 1] - radii - min_y,
            max_x - circles[:, 0] - radii,
            max_y - circles[:, 1] - radii,
        ],
        axis=1,
    )
    boundary_touch_fraction = float(np.mean(np.min(margins, axis=1) <= 1e-4))

    pairwise_dist = np.linalg.norm(circles[:, None, :2] - circles[None, :, :2], axis=2)
    required_dist = radii[:, None] + radii[None, :]
    gaps = pairwise_dist - required_dist
    np.fill_diagonal(gaps, np.inf)
    nn_gap = np.min(gaps, axis=1)
    nn_gap_mean = float(np.mean(nn_gap))
    nn_gap_std = float(np.std(nn_gap))
    nn_gap_cv = float(nn_gap_std / (abs(nn_gap_mean) + 1e-9))

    radii_sum = float(np.sum(radii))
    if radii_sum <= 0.0:
        radius_entropy = 0.0
    else:
        p = radii / radii_sum
        p = p[p > 0.0]
        radius_entropy = float(-np.sum(p * np.log(p)) / np.log(len(radii))) if p.size else 0.0

    return {
        "width": float(width),
        "height": float(height),
        "width_plus_height": float(width + height),
        "aspect_ratio": aspect_ratio,
        "boundary_touch_fraction": boundary_touch_fraction,
        "nn_gap_mean": nn_gap_mean,
        "nn_gap_cv": nn_gap_cv,
        "radius_entropy": radius_entropy,
    }


def score_fn(circle_packing21, _inputs=None) -> dict:
    """
    Score a candidate constructor.

    Valid packings receive `score = radii_sum`; invalid packings are rejected
    with an error, matching the SpecEvo examples' convention.
    """
    try:
        start = time.perf_counter()
        raw = circle_packing21()
        exec_time = time.perf_counter() - start

        circles = _to_circles(raw)
        radii_sum = float(np.sum(circles[:, 2]))
        descriptors = compute_behavior_descriptors(circles)

        valid, reason = evaluate_packing_output(circles)
        if not valid:
            return {
                "score": 0.0,
                "valid": 0.0,
                "radii_sum": radii_sum,
                "combined_score": float(radii_sum / BENCHMARK_RADII_SUM),
                "execution_time": exec_time,
                **descriptors,
                "error": reason,
            }

        return {
            "score": radii_sum,
            "valid": 1.0,
            "radii_sum": radii_sum,
            "combined_score": float(radii_sum / BENCHMARK_RADII_SUM),
            "execution_time": exec_time,
            **descriptors,
        }
    except Exception as e:
        return {"error": str(e)}
