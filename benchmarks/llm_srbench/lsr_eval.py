"""Shared LLM-SRBench / LSR-Synth evaluation engine for SpecEvo Baselines + SpecEvo.

This module adapts the LSR-Synth half of LLM-SRBench (Shojaee et al., ICML 2025 —
https://github.com/deep-symbolic-mathematics/llm-srbench) so a single candidate
``equation(...)`` function can be scored identically for BOTH integration paths
in this repo:

  * the *baseline* runners (openevolve_native / gepa_native / adaevolve / evox)
    via ``benchmarks/llm_srbench/<domain>/<problem>/evaluator.py``; and
  * the *SpecEvo / SpecEvo* runner via
    ``tasks/llm_srbench/<domain>/<problem>/problem.py``.

Protocol (paper Sec. 2.3 + App. B, and LLM-SR's own ``eval_spec``)
------------------------------------------------------------------
A hypothesis is a Python function

    def equation(v1: np.ndarray, ..., vk: np.ndarray, params: np.ndarray) -> np.ndarray

i.e. an *equation skeleton* whose numeric constants live in ``params``. Scoring a
hypothesis means:

  1. fit ``params`` (at most ``MAX_NPARAMS`` = 10) on the **train** split by
     minimising mean squared error with SciPy's BFGS from ``params = [1.0]*10``;
  2. with those fitted parameters frozen, predict on the held-out **id_test**
     (in-domain) and **ood_test** (out-of-domain) splits;
  3. report NMSE / R² / Acc(0.1) / MAPE on each split.

Only the **train** fit drives the search (``combined_score``). ID and OOD test
metrics are computed and recorded on every evaluation but never used for
selection, so the reported generalisation numbers stay honest — the same
dev-drives-search / test-held-out discipline the CO-Bench engine uses.

The search signal (``LSR_SCORE_MODE``)
--------------------------------------
``combined_score`` is always a strictly decreasing function of **train NMSE** —
i.e. of the paper's own search signal (``-MSE``; NMSE only divides by the
constant ``var(y_train)``), so the *ranking* of hypotheses is the paper's ranking
under either mode. What differs is resolution:

``log_nmse`` (default)
    ``score = log10(1 + 1/train_nmse)``, capped at ``LSR_SCORE_LOG_CAP`` (16,
    the point at which NMSE is float64 round-off). For any decent fit this is
    ``-log10(train_nmse)`` to four decimals, so the score reads directly as
    *decades of NMSE below 1*: 3.0 means NMSE = 1e-3, 7.0 means 1e-7. It stays
    strictly positive for every finite NMSE, so 0.0 still means "did not run".

``inv_nmse``
    ``score = 1 / (1 + train_nmse)`` — the original scale, kept so runs already
    on disk can be reproduced. It is monotone but saturates: every NMSE below
    1e-4 renders as ``1.0000``, and both runners show the score to 3–4 decimals
    in the prompts they build (``specevo/engine/prompts.py``,
    ``specevo_baselines/context_builder/default/builder.py``), so the model cannot see
    the difference between a good equation and an excellent one. Prefer
    ``log_nmse`` for new experiments.

``train_nmse`` and ``neg_log10_train_nmse`` are reported under both modes, so the
NMSE a score came from is always in the record.

A hypothesis that raises, returns the wrong shape, produces non-finite output, or
exceeds the per-hypothesis time limit scores 0.0.

Runtime knobs (env vars, all optional)
--------------------------------------
  LSR_SCORE_MODE       log_nmse (default) | inv_nmse — see above
  LSR_SCORE_LOG_CAP    cap on the log_nmse score (default 16)
  LSR_EVAL_TIMEOUT     per-hypothesis time limit in seconds (default 30, the
                       paper's ``T = 30s per program hypothesis``)
  LSR_MAX_NPARAMS      size of the ``params`` vector (default 10, per the paper)
  LSR_MAX_FIT_POINTS   subsample the train split to at most N points before
                       fitting (default 0 = use all 4000; only for quick smoke
                       tests — it changes the benchmark)
  LSR_BFGS_MAXITER     cap BFGS iterations (default 0 = SciPy's default)
  LSR_DATA_DIR         override the materialised data directory
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import signal
import threading
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "id_test", "ood_test")

# Human-readable scientific context per domain, used to open the prompt. The
# variables themselves come from the manifest (they vary per problem).
DOMAIN_CONTEXT: dict[str, str] = {
    "chem_react": "chemistry reaction kinetics",
    "bio_pop_growth": "biological population growth",
    "phys_osc": "a non-linear damped harmonic oscillator in physics",
    "matsci": "the stress-strain response of a material in materials science",
}


def data_dir() -> Path:
    raw = os.environ.get("LSR_DATA_DIR", "").strip()
    return Path(raw).expanduser().resolve() if raw else BENCH_DIR / "data"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def max_nparams() -> int:
    return _env_int("LSR_MAX_NPARAMS", 10)


def default_timeout() -> float:
    return _env_float("LSR_EVAL_TIMEOUT", 30.0)


# --------------------------------------------------------------------------- #
# Search signal
# --------------------------------------------------------------------------- #
SCORE_MODES = ("log_nmse", "inv_nmse")
DEFAULT_SCORE_MODE = "log_nmse"


def score_mode() -> str:
    raw = os.environ.get("LSR_SCORE_MODE", "").strip().lower()
    if not raw:
        return DEFAULT_SCORE_MODE
    if raw not in SCORE_MODES:
        raise ValueError(
            f"LSR_SCORE_MODE={raw!r} is not a known mode (one of: {', '.join(SCORE_MODES)})"
        )
    return raw


def score_log_cap() -> float:
    """Ceiling for the ``log_nmse`` score.

    Default 16: an NMSE of 1e-16 is at the float64 round-off floor of these
    datasets, so hypotheses below it are numerically indistinguishable and tying
    them is honest rather than arbitrary.
    """
    return _env_float("LSR_SCORE_LOG_CAP", 16.0)


def score_from_nmse(nmse: float) -> float:
    """Search score for a hypothesis whose training NMSE is *nmse*.

    Strictly decreasing in NMSE under every mode, and strictly positive for every
    finite NMSE — 0.0 is reserved for a hypothesis that did not produce one.
    """
    if score_mode() == "inv_nmse":
        return 1.0 / (1.0 + nmse)
    cap = score_log_cap()
    if nmse <= 0.0:
        return cap
    # log10(1 + 1/nmse) rather than -log10(nmse): identical to four decimals
    # whenever nmse << 1, but it never crosses zero, so a hypothesis merely as
    # bad as predicting the mean (nmse = 1, score 0.301) still ranks above one
    # that failed to run (score 0.0).
    return min(cap, math.log10(1.0 + 1.0 / nmse))


def score_formula() -> str:
    """The active score as a formula, short enough for a per-evaluation log line."""
    if score_mode() == "inv_nmse":
        return "score = 1/(1+NMSE)"
    return "score = log10(1 + 1/NMSE)"


def score_explanation() -> str:
    """Full description of the active score, for the task prompt."""
    if score_mode() == "inv_nmse":
        return "score = 1 / (1 + train_nmse) — 1.0 is a perfect fit, higher is better"
    return (
        f"score = log10(1 + 1/train_nmse) — i.e. the number of decades of NMSE "
        f"below 1, so 3.0 means NMSE = 1e-3 and 7.0 means NMSE = 1e-7 "
        f"(capped at {score_log_cap():g}); higher is better"
    )


# --------------------------------------------------------------------------- #
# Manifest / data access
# --------------------------------------------------------------------------- #
_MANIFEST: Optional[dict] = None
_PROBLEM_CACHE: dict[tuple[str, str], dict] = {}

_PREPARE_HINT = (
    "LSR-Synth data is missing. Materialise it once with:\n"
    "    python benchmarks/llm_srbench/prepare_data.py\n"
    "(or bash benchmarks/llm_srbench/download_dataset.sh)"
)


def manifest() -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        path = data_dir() / "problems.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. {_PREPARE_HINT}")
        _MANIFEST = json.loads(path.read_text())
    return _MANIFEST


def domains() -> list[str]:
    return list(manifest()["domains"])


def problem_ids(domain: str) -> list[str]:
    return [p["id"] for p in manifest()["domains"][domain]["problems"]]


def problem_meta(domain: str, problem_id: str) -> dict:
    doms = manifest()["domains"]
    if domain not in doms:
        raise KeyError(f"unknown LSR-Synth domain {domain!r} (have: {', '.join(doms)})")
    for p in doms[domain]["problems"]:
        if p["id"] == problem_id:
            return p
    raise KeyError(
        f"unknown problem {problem_id!r} in domain {domain!r} "
        f"(have: {', '.join(problem_ids(domain))})"
    )


def load_problem(domain: str, problem_id: str) -> dict:
    """Return ``{'meta', 'train', 'id_test', 'ood_test'}`` for one problem.

    Each split is an ``(n, 1 + k)`` float64 array whose column 0 is the target
    and columns 1.. are the inputs, in ``meta['symbols'][1:]`` order.
    """
    key = (domain, problem_id)
    if key in _PROBLEM_CACHE:
        return _PROBLEM_CACHE[key]
    meta = problem_meta(domain, problem_id)
    path = data_dir() / meta["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {_PREPARE_HINT}")
    with np.load(path) as z:
        arrays = {}
        for s in SPLITS:
            arr = np.asarray(z[s], dtype=np.float64)
            # The cache outlives individual candidates. Mark it read-only so no
            # amount of in-place arithmetic anywhere can silently corrupt the
            # dataset for every hypothesis scored afterwards.
            arr.flags.writeable = False
            arrays[s] = arr
    problem = {"meta": meta, **arrays}
    _PROBLEM_CACHE[key] = problem
    return problem


def _split_xy(arr: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """``(input columns, target)`` — the call shape ``equation`` wants.

    The input columns are fresh writable copies, not views into the cached array:
    evolved code does write to its arguments in place (``x[x < 0] = 0`` and
    friends), which must be allowed but must not reach the dataset. One copy per
    scoring call is reused across every step of the parameter fit, so this costs
    tens of microseconds. Copying also makes the columns contiguous, which speeds
    up the candidate's own numpy work.
    """
    return [arr[:, i].copy() for i in range(1, arr.shape[1])], arr[:, 0]


# --------------------------------------------------------------------------- #
# Prompt / program-text builders (mirroring LLM-SR's specification)
# --------------------------------------------------------------------------- #
def _input_desc_phrase(descs: list[str]) -> str:
    """"a, b, and c" — the phrasing LLM-SR uses in its task sentence."""
    ins = descs[1:]
    if len(ins) > 1:
        return ", ".join(ins[:-1]) + ", and " + ins[-1]
    return ins[0]


def equation_signature(domain: str, problem_id: str) -> str:
    meta = problem_meta(domain, problem_id)
    args = ", ".join(f"{v}: np.ndarray" for v in meta["symbols"][1:])
    return f"def equation({args}, params: np.ndarray) -> np.ndarray:"


def equation_docstring(domain: str, problem_id: str, indent: str = "    ") -> str:
    meta = problem_meta(domain, problem_id)
    symbols, descs = meta["symbols"], meta["symbol_descs"]
    lines = [f'{indent}""" Mathematical function for {descs[0]}', ""]
    lines.append(f"{indent}Args:")
    for name, desc in zip(symbols[1:], descs[1:]):
        lines.append(f"{indent}    {name}: A numpy array representing observations of {desc}.")
    lines.append(f"{indent}    params: Array of numeric constants or parameters to be optimized")
    lines.append("")
    lines.append(f"{indent}Return:")
    lines.append(
        f"{indent}    A numpy array representing {descs[0]} as the result of "
        f"applying the mathematical function to the inputs."
    )
    lines.append(f'{indent}"""')
    return "\n".join(lines)


def seed_program(domain: str, problem_id: str) -> str:
    """LLM-SR's seed skeleton: a linear model in every input plus an offset."""
    meta = problem_meta(domain, problem_id)
    ins = meta["symbols"][1:]
    terms = " + ".join(f"params[{i}] * {name}" for i, name in enumerate(ins))
    body = f"    output = {terms} + params[{len(ins)}]\n    return output\n"
    return (
        "import numpy as np\n\n"
        f"{equation_signature(domain, problem_id)}\n"
        f"{equation_docstring(domain, problem_id)}\n"
        f"{body}"
    )


def problem_description(domain: str, problem_id: str) -> str:
    """Markdown task statement handed to the LLM (SpecEvo / SpecEvo path)."""
    meta = problem_meta(domain, problem_id)
    symbols, descs = meta["symbols"], meta["symbol_descs"]
    context = DOMAIN_CONTEXT.get(domain, domain)
    var_lines = "\n".join(
        f"- `{name}` — {desc}" for name, desc in zip(symbols[1:], descs[1:])
    )
    n_params = max_nparams()
    return f"""\
# Scientific equation discovery — LSR-Synth / {domain} / {problem_id}

Find the mathematical function skeleton that represents **{descs[0]}**, given
data on {_input_desc_phrase(descs)}.

The system under study is {context}. Use domain knowledge of the field together
with the observed data to propose an equation that is both scientifically
plausible and an accurate fit.

## Target

`{symbols[0]}` — {descs[0]}

## Input variables

{var_lines}

## What to implement

```python
{equation_signature(domain, problem_id)}
    ...
```

Return a numpy array of the same length as the inputs. Every numeric constant
must be taken from `params`, a numpy array of {n_params} float values — do NOT
hard-code fitted numbers. `params` is optimised for you (BFGS, least squares on
the training split) before your skeleton is scored, so propose the *structure*
and let the optimiser find the coefficients. Indices `params[0]` through
`params[{n_params - 1}]` are available; unused entries are harmless.

## Scoring

`params` are fitted on the training split, then the skeleton is scored by
**normalised mean squared error (NMSE) on that same split — lower NMSE is the
objective**. The reported score is a monotone transform of it:

    {score_explanation()}

Held-out in-domain (ID) and out-of-domain (OOD) test splits are also measured and
reported, but they never influence the search — an equation that captures the
true underlying mechanism will generalise to them, whereas an over-parameterised
curve fit will not.

A hypothesis that raises, returns the wrong shape, produces NaN/inf, or takes
longer than {int(default_timeout())}s to fit scores 0.0.

## Guidance

- Only `numpy` (as `np`) and the Python standard library are available.
- Prefer combinations of terms that make sense for {context}, then extend them
  with additional non-linear structure where the data demands it.
- Keep the skeleton cheap to evaluate: it is called many times inside the
  parameter fit.
"""


def system_message(domain: str, problem_id: str) -> str:
    """``prompt.system_message`` for the baseline runners.

    Deliberately the *same* task text the SpecEvo path gets via
    ``PROBLEM_DESCRIPTION``: both families of method must see identical
    information about the problem and the scoring rule, or the comparison
    between them measures prompt wording rather than search.
    """
    return (
        "You are an expert in scientific equation discovery (symbolic regression). "
        "Improve the `equation` function below so that its skeleton captures the "
        "true underlying relationship in the data.\n\n"
        + problem_description(domain, problem_id)
    )


def initial_program_text(domain: str, problem_id: str) -> str:
    """``initial_program.py`` body for the baseline runners (EVOLVE-BLOCK form)."""
    meta = problem_meta(domain, problem_id)
    symbols, descs = meta["symbols"], meta["symbol_descs"]
    ins = symbols[1:]
    terms = " + ".join(f"params[{i}] * {name}" for i, name in enumerate(ins))
    header = "\n".join(
        f"# {line}" if line else "#"
        for line in [
            f"LLM-SRBench :: LSR-Synth / {domain} / {problem_id}",
            "",
            f"Find the mathematical function skeleton that represents {descs[0]},",
            f"given data on {_input_desc_phrase(descs)}.",
            "",
            f"System under study: {DOMAIN_CONTEXT.get(domain, domain)}.",
            "",
            "Every numeric constant must come from `params` (a numpy array of "
            f"{max_nparams()} floats),",
            "which is fitted to the training data by BFGS before the skeleton is scored.",
            "Do not hard-code fitted numbers, and do not read the data from disk.",
        ]
    )
    return (
        f"{header}\n\n"
        "import numpy as np\n\n"
        "# EVOLVE-BLOCK-START\n"
        f"{equation_signature(domain, problem_id)}\n"
        f"{equation_docstring(domain, problem_id)}\n"
        f"    output = {terms} + params[{len(ins)}]\n"
        "    return output\n"
        "# EVOLVE-BLOCK-END\n"
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _finite_pair(y_pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop points whose prediction is not finite (the reference drops NaNs)."""
    mask = np.isfinite(y_pred)
    return y_pred[mask], y[mask]


def output_metrics(y_pred: np.ndarray, y: np.ndarray, tol: float = 0.1) -> dict:
    """NMSE / R² / Acc(tol) / MAPE for one split, following the paper's App. B.

    ``acc`` is the paper's ``Acc_tau``: 1.0 only when the *maximum* point-wise
    relative error over the whole split is within ``tol``, else 0.0.
    ``frac_within`` is the softer per-point version, kept for diagnostics.
    """
    yp, yt = _finite_pair(np.asarray(y_pred, dtype=np.float64), np.asarray(y, dtype=np.float64))
    n_valid = int(yp.size)
    if n_valid == 0:
        return {
            "mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"),
            "mape": float("inf"), "acc": 0.0, "frac_within": 0.0,
            "num_valid_points": 0, "num_points": int(np.size(y)),
        }
    resid = yp - yt
    sse = float(np.sum(resid**2))
    denom = float(np.sum((yt - yt.mean()) ** 2))
    nmse = sse / denom if denom > 0 else (0.0 if sse == 0 else float("inf"))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(resid) / np.abs(yt)
        rel = np.where(np.isfinite(rel), rel, np.inf)
    # A split is only "accurate to tol" if EVERY point is, and only if no point
    # was dropped for being non-finite.
    complete = n_valid == int(np.size(y))
    max_rel = float(np.max(rel)) if rel.size else float("inf")
    return {
        "mse": sse / n_valid,
        "nmse": nmse,
        "r2": 1.0 - sse / denom if denom > 0 else float("-inf"),
        "mape": float(np.mean(rel[np.isfinite(rel)])) if np.any(np.isfinite(rel)) else float("inf"),
        "acc": 1.0 if (complete and max_rel <= tol) else 0.0,
        "frac_within": float(np.mean(rel <= tol)),
        "max_rel_error": max_rel,
        "num_valid_points": n_valid,
        "num_points": int(np.size(y)),
    }


# --------------------------------------------------------------------------- #
# Core: fit params on train, then measure every split
# --------------------------------------------------------------------------- #
_PENALTY = 1e30  # finite stand-in for a failed objective, so BFGS can back off


def _predict(fn: Callable[..., Any], cols: list[np.ndarray], params: np.ndarray, n: int) -> np.ndarray:
    """Call the candidate and coerce its output to a length-n float array."""
    out = fn(*cols, params)
    arr = np.asarray(out, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    else:
        arr = arr.reshape(-1)
        if arr.size == 1:
            arr = np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"equation returned {arr.size} values, expected {n}")
    return arr


def score_callable(fn: Callable[..., Any], problem: dict, *, tol: float = 0.1) -> dict:
    """Fit + evaluate one candidate. Never raises; failures come back as errors."""
    train, id_test, ood_test = problem["train"], problem["id_test"], problem["ood_test"]
    Xtr, ytr = _split_xy(train)

    fit_cap = _env_int("LSR_MAX_FIT_POINTS", 0)
    if fit_cap > 0 and ytr.size > fit_cap:
        idx = np.linspace(0, ytr.size - 1, fit_cap).astype(int)
        Xfit, yfit = [c[idx] for c in Xtr], ytr[idx]
    else:
        Xfit, yfit = Xtr, ytr

    n_par = max_nparams()
    x0 = np.ones(n_par, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            # A candidate that cannot even run once at params = 1 is a hard fail;
            # reporting the exception is what feeds the search's error taxonomy.
            try:
                _predict(fn, Xfit, x0, yfit.size)
            except Exception as exc:  # noqa: BLE001
                return _failure(f"{type(exc).__name__}: {exc}")

            n_fit = yfit.size

            def loss(p: np.ndarray) -> float:
                try:
                    pred = _predict(fn, Xfit, np.asarray(p, dtype=np.float64), n_fit)
                except Exception:  # noqa: BLE001 — domain errors are penalised, not fatal
                    return _PENALTY
                v = float(np.mean((pred - yfit) ** 2))
                return v if math.isfinite(v) else _PENALTY

            from scipy.optimize import minimize

            opts: dict[str, Any] = {}
            maxiter = _env_int("LSR_BFGS_MAXITER", 0)
            if maxiter > 0:
                opts["maxiter"] = maxiter
            try:
                res = minimize(loss, x0, method="BFGS", options=opts or None)
                params = np.asarray(res.x, dtype=np.float64)
            except Exception:  # noqa: BLE001 — keep the x0 fit rather than failing
                params = x0

            # BFGS occasionally wanders into a worse point than it started from.
            best_params, best_loss = params, loss(params)
            loss_at_x0 = loss(x0)
            if loss_at_x0 < best_loss:
                best_params, best_loss = x0, loss_at_x0
            if not math.isfinite(best_loss) or best_loss >= _PENALTY:
                return _failure("parameter fit did not produce a finite loss")

            out: dict[str, Any] = {}
            try:
                for name, arr in (("train", train), ("id", id_test), ("ood", ood_test)):
                    cols, y = _split_xy(arr)
                    pred = _predict(fn, cols, best_params, y.size)
                    out[name] = output_metrics(pred, y, tol=tol)
            except Exception as exc:  # noqa: BLE001
                return _failure(f"evaluation failed: {type(exc).__name__}: {exc}")

    return _assemble(out, best_params, problem, tol)


def acc_key(prefix: str, tol: float = 0.1) -> str:
    return f"{prefix}_acc_{tol:g}"


def _num(value: float) -> Optional[float]:
    """Non-finite floats become ``None`` so every metric dict is strict-JSON safe.

    Both runners persist metric dicts verbatim (baseline checkpoints,
    SpecEvo's ``summary.json``); ``inf``/``nan`` there would produce JSON that only
    Python's non-strict decoder can read back.
    """
    v = float(value)
    return v if math.isfinite(v) else None


def _assemble(parts: dict[str, dict], params: np.ndarray, problem: dict, tol: float) -> dict:
    train_nmse = parts["train"]["nmse"]
    if not math.isfinite(train_nmse):
        return _failure("non-finite training NMSE")
    score = score_from_nmse(train_nmse)
    res: dict[str, Any] = {
        "combined_score": score,
        "score": score,
        "train_nmse": train_nmse,
        "neg_log10_train_nmse": -math.log10(max(train_nmse, 1e-300)),
        "id_nmse": _num(parts["id"]["nmse"]),
        "ood_nmse": _num(parts["ood"]["nmse"]),
        acc_key("id", tol): parts["id"]["acc"],
        acc_key("ood", tol): parts["ood"]["acc"],
        "id_r2": _num(parts["id"]["r2"]),
        "ood_r2": _num(parts["ood"]["r2"]),
        "id_frac_within": parts["id"]["frac_within"],
        "ood_frac_within": parts["ood"]["frac_within"],
        "n_params_fitted": int(params.size),
        "valid": 1.0,
    }
    res["feedback"] = format_feedback(res, parts, problem)
    res["fitted_params"] = [float(v) for v in params]
    res["metrics_detail"] = {k: _jsonable(v) for k, v in parts.items()}
    return res


def _jsonable(d: dict) -> dict:
    return {k: (_num(v) if isinstance(v, float) else v) for k, v in d.items()}


def _failure(message: str) -> dict:
    """A rejected hypothesis carries no measurements — only the reason why.

    Metric keys are omitted rather than filled with sentinels so that nothing
    downstream can average a failure into a real distribution by accident.
    """
    return {
        "combined_score": 0.0,
        "score": 0.0,
        "valid": 0.0,
        "error": message,
        "feedback": f"Hypothesis rejected: {message}",
    }


def format_feedback(res: dict, parts: dict[str, dict], problem: dict) -> str:
    meta = problem["meta"]
    lines = [
        f"Problem {meta['instance_id']}  ({' , '.join(meta['symbols'][1:])} -> {meta['symbols'][0]})",
        f"train: NMSE={parts['train']['nmse']:.4g}  R2={parts['train']['r2']:.6f}"
        f"   [{score_formula()} = {res['combined_score']:.4f}]",
        f"ID   : NMSE={parts['id']['nmse']:.4g}  R2={parts['id']['r2']:.6f}"
        f"  Acc(0.1)={parts['id']['acc']:.0f}  within-0.1={parts['id']['frac_within']:.1%}",
        f"OOD  : NMSE={parts['ood']['nmse']:.4g}  R2={parts['ood']['r2']:.6f}"
        f"  Acc(0.1)={parts['ood']['acc']:.0f}  within-0.1={parts['ood']['frac_within']:.1%}",
        "(ID / OOD are held out — reported only, never optimised.)",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Timeout wrappers
# --------------------------------------------------------------------------- #
class _AlarmTimeout(BaseException):
    """Raised by the SIGALRM handler.

    Deliberately a ``BaseException``: the scoring core wraps candidate calls in
    ``except Exception`` to turn domain errors into penalties, and an ``Exception``
    here would be swallowed there and reported as an ordinary failure instead of
    a timeout (and, worse, the fit would keep running past the deadline).
    """


def _run_with_sigalrm(work: Callable[[], dict], timeout: float, problem: dict) -> dict:
    def _handler(signum, frame):  # noqa: ANN001, ARG001
        raise _AlarmTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        return work()
    except _AlarmTimeout:
        return _failure(f"Timeout ({timeout:g}s)")
    except Exception as exc:  # noqa: BLE001
        return _failure(f"{type(exc).__name__}: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _run_with_thread(work: Callable[[], dict], timeout: float, problem: dict) -> dict:
    """Soft timeout for daemon, non-main-thread callers: cannot hard-kill."""
    box: dict[str, dict] = {}

    def _target() -> None:
        try:
            box["r"] = work()
        except Exception as exc:  # noqa: BLE001
            box["r"] = _failure(f"{type(exc).__name__}: {exc}")

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout + 1.0)
    if t.is_alive():
        return _failure(f"Timeout ({timeout:g}s)")
    return box.get("r", _failure("no result"))


def _subprocess_target(source: str, domain: str, problem_id: str, tol: float,
                       timeout: float, q) -> None:  # noqa: ANN001
    # The child arms its own alarm so a compute-bound candidate is reported at
    # the stated limit; the parent's join is only the backstop for a candidate
    # that manages to escape the alarm (C-level loop, fork bomb, OOM).
    try:
        problem = load_problem(domain, problem_id)
        fn = _compile(source)
        if isinstance(fn, str):
            q.put(_failure(fn))
            return
        q.put(_run_with_sigalrm(lambda: score_callable(fn, problem, tol=tol), timeout, problem))
    except Exception as exc:  # noqa: BLE001
        q.put(_failure(f"{type(exc).__name__}: {exc}"))


def _run_in_subprocess(source: str, domain: str, problem_id: str, tol: float,
                       timeout: float) -> Optional[dict]:
    """Hard-killable path used when we are allowed to fork (baseline runners).

    ``fork`` rather than ``spawn`` deliberately: the child inherits the already
    loaded dataset, so a candidate costs no re-import and no re-read of the .npz —
    with hundreds of candidates per problem that difference dominates. This
    mirrors the CO-Bench engine's choice.

    Returns ``None`` if the child could not be started at all, so the caller can
    fall back to scoring in-process.
    """
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(
        target=_subprocess_target, args=(source, domain, problem_id, tol, timeout, q)
    )
    try:
        p.start()
    except Exception:  # noqa: BLE001 — no fork available, or process/thread limits
        return None
    # Drain the queue before join(): a large payload can fill the pipe buffer and
    # deadlock a child that is blocked writing while the parent waits on exit.
    result: Optional[dict] = None
    try:
        result = q.get(timeout=timeout + 5.0)
    except Exception:  # noqa: BLE001 — queue.Empty, or the child died
        result = None
    if p.is_alive():
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            p.terminate()
    p.join(2)
    if result is None:
        return _failure(f"Timeout ({timeout:g}s) or candidate process died (crash / OOM)")
    return result


def _compile(source: str) -> Callable[..., Any] | str:
    """Exec candidate source and pull out ``equation``. Returns an error string on failure."""
    ns: dict[str, Any] = {}
    try:
        exec(compile(source, "<candidate>", "exec"), ns)  # noqa: S102 — evolving code is the point
    except Exception as exc:  # noqa: BLE001
        return f"candidate failed to compile: {type(exc).__name__}: {exc}"
    fn = ns.get("equation")
    if not callable(fn):
        return "candidate does not define a callable named `equation`"
    return fn


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def evaluate_source(
    domain: str,
    problem_id: str,
    source: str,
    *,
    timeout: Optional[float] = None,
    tol: float = 0.1,
) -> dict:
    """Score candidate *source text* (baseline path: the file defines ``equation``)."""
    timeout = default_timeout() if timeout is None else timeout
    load_problem(domain, problem_id)  # fail fast, and warm the cache before forking

    if not mp.current_process().daemon:
        result = _run_in_subprocess(source, domain, problem_id, tol, timeout)
        if result is not None:
            return result
        # Could not fork — fall through and score in this process instead.

    fn = _compile(source)
    if isinstance(fn, str):
        return _failure(fn)
    return evaluate_callable(domain, problem_id, fn, timeout=timeout, tol=tol)


def evaluate_callable(
    domain: str,
    problem_id: str,
    fn: Callable[..., Any],
    *,
    timeout: Optional[float] = None,
    tol: float = 0.1,
) -> dict:
    """Score an already-compiled candidate (SpecEvo / SpecEvo path)."""
    timeout = default_timeout() if timeout is None else timeout
    problem = load_problem(domain, problem_id)

    def work() -> dict:
        return score_callable(fn, problem, tol=tol)

    if threading.current_thread() is threading.main_thread():
        return _run_with_sigalrm(work, timeout, problem)
    return _run_with_thread(work, timeout, problem)


def read_program_source(program_path: str) -> str:
    """Candidate source for the baseline path (the whole file defines ``equation``)."""
    return Path(program_path).read_text()


def baseline_metrics(result: dict) -> dict:
    """Trim an engine result down to what a baseline evaluator should return.

    ``fitted_params`` / ``metrics_detail`` are dropped: the baseline runners log
    every metric dict into their program database and JSON-encode it, so keeping
    them would bloat every checkpoint with per-candidate arrays.
    """
    drop = {"fitted_params", "metrics_detail"}
    return {k: v for k, v in result.items() if k not in drop}
