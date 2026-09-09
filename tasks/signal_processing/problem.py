"""
Real-time adaptive signal-processing problem — SpecEvo example.

Task: write a filter that takes a noisy 1D signal and a sliding window
size, and returns a denoised signal. The filter is evaluated on 5 synthetic
non-stationary test signals (smooth sinusoidal with trend, multi-frequency,
non-stationary frequency, step changes, random walk + trend). The score
combines smoothness, tracking accuracy, correlation with the clean signal,
noise reduction, and reliability across the five signals.

Mirrors the benchmark at benchmarks/math/signal_processing.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from scipy.stats import pearsonr


WINDOW_SIZE = 20
NUM_TEST_SIGNALS = 5
PER_SIGNAL_TIMEOUT = 10.0
TIMEOUT_SECONDS = 120


PROBLEM_DESCRIPTION = f"""
# Real-Time Adaptive Signal Processing

## Problem
Implement a **sliding-window denoising filter** for non-stationary 1D
real-valued signals. The filter must reduce noise while preserving the
underlying signal's dynamics (trends, frequency content, step changes).

The grader evaluates your filter on {NUM_TEST_SIGNALS} synthetic test
signals with different characteristics, all using window size W =
{WINDOW_SIZE}:

  1. Smooth low-frequency sinusoid + slow linear trend
  2. Sum of 3 sinusoids (multi-frequency mixture)
  3. Single sinusoid with **non-stationary** (time-varying) frequency
  4. Piecewise-constant signal with **step changes** at 1/3 and 2/3
  5. Random walk + slow linear trend

## Function contract
```python
def run_signal_processing(noisy_signal, window_size):
    ...
    return {{"filtered_signal": filtered_signal}}
```
- ``noisy_signal``: 1D numpy array of length L.
- ``window_size``: integer sliding-window length (the grader always passes
  {WINDOW_SIZE}; you should still use the argument rather than hard-coding).
- Return a Python ``dict`` with key ``filtered_signal`` whose value is a
  1D array (or list) of length **exactly** ``L - window_size + 1``.
  Extra dict keys are allowed and ignored.

**IMPORTANT — input constraints at runtime:**
- The grader passes ONLY ``noisy_signal`` and ``window_size``. The clean
  signal is NOT available to your function — do not look it up, regenerate
  it from a known seed, or otherwise cheat.
- The output must be finite (no NaN / Inf) and of the exact expected
  length; anything else makes that test signal count as failed.

## Score components (per test signal)
- **S**: slope-change penalty — counts directional reversals in
  ``filtered_signal``. Fewer is better.
- **L_recent**: instantaneous lag error
  ``|filtered_signal[-1] - noisy_signal[delay + len(filtered_signal) - 1]|``
  where ``delay = window_size - 1``.
- **L_avg**: mean absolute error of ``filtered_signal`` against the
  aligned noisy input over the output window.
- **R**: false-reversal penalty — directional changes that appear in
  ``filtered_signal`` but NOT in the clean signal (the grader compares
  using its own copy of the clean signal; your function only sees noisy).

Per-signal composite (higher is better, in [0, 1]):

    S'      = min(S / 50, 2)
    R'      = min(R / 25, 2)
    penalty = 0.3 * S' + 0.2 * L_recent + 0.2 * L_avg + 0.3 * R'
    J       = 1 / (1 + penalty)

## Final score
Averaged across the {NUM_TEST_SIGNALS} test signals, the grader then
combines:

    score = 0.4 * avg(J)
          + 0.2 * smoothness         (= 1 / (1 + avg_slope_changes / 20))
          + 0.2 * accuracy           (= mean Pearson corr. with clean ≥ 0)
          + 0.1 * noise_reduction    (variance reduction vs. clean)
          + 0.1 * success_rate       (fraction of test signals that ran)

**Hard gate**: if average Pearson correlation between filtered and clean
signal falls below 0.1, the final score is forced to 0 (filter that drifts
or destroys the signal is discarded). So tracking the signal accurately
matters more than aggressive smoothing.

## Runtime Constraint
Per-signal soft limit: {PER_SIGNAL_TIMEOUT} s. Total problem timeout:
{TIMEOUT_SECONDS} s.
"""


FUNCTION_SIGNATURE = """
import numpy as np
import time
import random
import math

def run_signal_processing(noisy_signal, window_size):
    '''
    Filter a 1D noisy signal using a sliding window of the given size.

    Args:
        noisy_signal: 1D numpy array of real-valued samples.
        window_size: int sliding-window length.

    Returns:
        A dict with key ``filtered_signal`` containing a 1D array of length
        ``len(noisy_signal) - window_size + 1``.
    '''
    pass
"""


# ---------------------------------------------------------------------------
# Per-component metric helpers (ported from the benchmark evaluator).
# ---------------------------------------------------------------------------


def _safe_float(value: float) -> float:
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _slope_changes(signal_data: np.ndarray) -> int:
    if len(signal_data) < 3:
        return 0
    diffs = np.diff(signal_data)
    changes = 0
    for i in range(1, len(diffs)):
        if np.sign(diffs[i]) != np.sign(diffs[i - 1]) and diffs[i - 1] != 0:
            changes += 1
    return changes


def _lag_error(filtered: np.ndarray, noisy: np.ndarray, window_size: int) -> float:
    if len(filtered) == 0:
        return 1.0
    delay = window_size - 1
    if len(noisy) <= delay:
        return 1.0
    recent_filtered = filtered[-1]
    recent_original = noisy[delay + len(filtered) - 1]
    return float(abs(recent_filtered - recent_original))


def _avg_tracking_error(filtered: np.ndarray, noisy: np.ndarray, window_size: int) -> float:
    if len(filtered) == 0:
        return 1.0
    delay = window_size - 1
    if len(noisy) <= delay:
        return 1.0
    aligned_original = noisy[delay : delay + len(filtered)]
    min_length = min(len(filtered), len(aligned_original))
    if min_length == 0:
        return 1.0
    return float(np.mean(np.abs(filtered[:min_length] - aligned_original[:min_length])))


def _false_reversal_penalty(
    filtered: np.ndarray, clean: np.ndarray, window_size: int
) -> int:
    if len(filtered) < 3 or len(clean) < 3:
        return 0
    delay = window_size - 1
    if len(clean) <= delay:
        return 0
    aligned_clean = clean[delay : delay + len(filtered)]
    min_length = min(len(filtered), len(aligned_clean))
    if min_length < 3:
        return 0
    f_diffs = np.diff(filtered[:min_length])
    c_diffs = np.diff(aligned_clean[:min_length])
    false_reversals = 0
    for i in range(1, len(f_diffs)):
        f_change = np.sign(f_diffs[i]) != np.sign(f_diffs[i - 1]) and f_diffs[i - 1] != 0
        c_change = np.sign(c_diffs[i]) != np.sign(c_diffs[i - 1]) and c_diffs[i - 1] != 0
        if f_change and not c_change:
            false_reversals += 1
    return false_reversals


def _composite_score(S: float, L_recent: float, L_avg: float, R: float) -> float:
    alpha = (0.3, 0.2, 0.2, 0.3)
    S_norm = min(S / 50.0, 2.0)
    L_recent_norm = min(L_recent, 2.0)
    L_avg_norm = min(L_avg, 2.0)
    R_norm = min(R / 25.0, 2.0)
    penalty = (
        alpha[0] * S_norm
        + alpha[1] * L_recent_norm
        + alpha[2] * L_avg_norm
        + alpha[3] * R_norm
    )
    return 1.0 / (1.0 + penalty)


def _generate_test_signals() -> list[tuple[np.ndarray, np.ndarray]]:
    """5 synthetic non-stationary signals — matches the benchmark evaluator."""
    test_signals: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(NUM_TEST_SIGNALS):
        rng = np.random.default_rng(42 + i)
        length = 500 + i * 100
        noise_level = 0.2 + i * 0.1
        t = np.linspace(0, 10, length)

        if i == 0:
            clean = 2 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * t
        elif i == 1:
            clean = (
                np.sin(2 * np.pi * 0.5 * t)
                + 0.5 * np.sin(2 * np.pi * 2 * t)
                + 0.2 * np.sin(2 * np.pi * 5 * t)
            )
        elif i == 2:
            clean = np.sin(2 * np.pi * (0.5 + 0.2 * t) * t)
        elif i == 3:
            clean = np.concatenate(
                [
                    np.ones(length // 3),
                    2 * np.ones(length // 3),
                    0.5 * np.ones(length - 2 * (length // 3)),
                ]
            )
        else:
            clean = np.cumsum(rng.standard_normal(length) * 0.1) + 0.05 * t

        noise = rng.normal(0.0, noise_level, length)
        noisy = clean + noise
        test_signals.append((noisy, clean))
    return test_signals


# ---------------------------------------------------------------------------
# Public score_fn.
# ---------------------------------------------------------------------------


def score_fn(run_signal_processing: Callable[..., Any], _inputs=None) -> dict:
    """Score a candidate ``run_signal_processing`` implementation."""
    try:
        test_signals = _generate_test_signals()

        all_scores: list[float] = []
        all_metrics: list[dict[str, float]] = []
        successful_runs = 0
        total_exec = 0.0
        per_signal_errors: list[str] = []

        for i, (noisy, clean) in enumerate(test_signals):
            try:
                start = time.perf_counter()
                result = run_signal_processing(
                    noisy_signal=noisy, window_size=WINDOW_SIZE
                )
                exec_time = time.perf_counter() - start
                total_exec += exec_time

                if not isinstance(result, dict) or "filtered_signal" not in result:
                    per_signal_errors.append(f"signal {i}: missing filtered_signal")
                    continue

                filtered = np.asarray(result["filtered_signal"], dtype=float)
                if filtered.ndim != 1 or filtered.size == 0:
                    per_signal_errors.append(f"signal {i}: empty / non-1D output")
                    continue
                if not np.isfinite(filtered).all():
                    per_signal_errors.append(f"signal {i}: non-finite values in output")
                    continue

                S = _slope_changes(filtered)
                L_recent = _lag_error(filtered, noisy, WINDOW_SIZE)
                L_avg = _avg_tracking_error(filtered, noisy, WINDOW_SIZE)
                R = _false_reversal_penalty(filtered, clean, WINDOW_SIZE)
                composite = _composite_score(S, L_recent, L_avg, R)

                # Auxiliary metrics: correlation with clean, noise reduction.
                delay = WINDOW_SIZE - 1
                aligned_clean = clean[delay : delay + len(filtered)]
                aligned_noisy = noisy[delay : delay + len(filtered)]
                min_length = min(len(filtered), len(aligned_clean))

                correlation = 0.0
                noise_reduction = 0.0
                if min_length > 1:
                    try:
                        c, _ = pearsonr(filtered[:min_length], aligned_clean[:min_length])
                        correlation = float(c) if not np.isnan(c) else 0.0
                    except Exception:
                        correlation = 0.0
                    noise_before = float(np.var(aligned_noisy[:min_length] - aligned_clean[:min_length]))
                    noise_after = float(np.var(filtered[:min_length] - aligned_clean[:min_length]))
                    noise_reduction = (
                        (noise_before - noise_after) / noise_before
                        if noise_before > 0.0
                        else 0.0
                    )
                    noise_reduction = max(0.0, noise_reduction)

                all_scores.append(composite)
                all_metrics.append(
                    {
                        "slope_changes": _safe_float(S),
                        "lag_error": _safe_float(L_recent),
                        "avg_error": _safe_float(L_avg),
                        "false_reversals": _safe_float(R),
                        "correlation": _safe_float(correlation),
                        "noise_reduction": _safe_float(noise_reduction),
                    }
                )
                successful_runs += 1
            except Exception as e:
                per_signal_errors.append(f"signal {i}: {e}")
                continue

        if successful_runs == 0:
            return {
                "error": "All test signals failed: " + "; ".join(per_signal_errors[:5])
            }

        avg_composite = float(np.mean(all_scores))
        avg_slope = float(np.mean([m["slope_changes"] for m in all_metrics]))
        avg_lag = float(np.mean([m["lag_error"] for m in all_metrics]))
        avg_track = float(np.mean([m["avg_error"] for m in all_metrics]))
        avg_false = float(np.mean([m["false_reversals"] for m in all_metrics]))
        avg_corr = float(np.mean([m["correlation"] for m in all_metrics]))
        avg_noise_reduction = float(np.mean([m["noise_reduction"] for m in all_metrics]))
        success_rate = float(successful_runs / NUM_TEST_SIGNALS)

        smoothness_score = 1.0 / (1.0 + avg_slope / 20.0)
        accuracy_score = max(0.0, avg_corr)

        overall_score = (
            0.4 * avg_composite
            + 0.2 * smoothness_score
            + 0.2 * accuracy_score
            + 0.1 * avg_noise_reduction
            + 0.1 * success_rate
        )

        if accuracy_score < 0.1:
            overall_score = 0.0

        return {
            "score": _safe_float(overall_score),
            "valid": 1.0 if overall_score > 0.0 else 0.0,
            "combined_score": _safe_float(overall_score),
            "composite_score": _safe_float(avg_composite),
            "slope_changes": _safe_float(avg_slope),
            "lag_error": _safe_float(avg_lag),
            "avg_error": _safe_float(avg_track),
            "false_reversals": _safe_float(avg_false),
            "correlation": _safe_float(avg_corr),
            "noise_reduction": _safe_float(avg_noise_reduction),
            "smoothness_score": _safe_float(smoothness_score),
            "accuracy_score": _safe_float(accuracy_score),
            "success_rate": _safe_float(success_rate),
            "execution_time": _safe_float(total_exec),
        }
    except Exception as e:
        return {"error": str(e)}
