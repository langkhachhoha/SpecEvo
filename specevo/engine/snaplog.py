"""``snap.json`` — the live search trace of a SpecEvo/SpecEvo run.

``snapshot.json`` is a *terminal* dump: it says what the archive looked
like when the run stopped. It cannot answer "how did the search get
there" — which Navigator mode fired when, whether the expensive frontier
call actually moved the needle, and which Speculator operator produced
each record. ``snap.json`` is the complementary *timeline*: an
append-only event log written incrementally (and flushed after every
event, so a killed or timed-out run still has a complete trace up to the
moment it died).

Two event kinds are recorded, both carrying the full code and
description of the program involved:

``navigator``
    One frontier (Navigator / Navigator intervention) call. Records which of the
    three modes was routed to — ``synthesis`` / ``surgical`` / ``reframe``
    — the stagnation that routed it, the anchors it was shown, what came
    back, whether it was admitted, and how its cheap variant fanout did.

``new_best``
    Every time the run's best score improves. Records which producer
    made it (``speculator`` / ``navigator`` / ``navigator_variant`` /
    ``init``), the exact operator label (e.g. ``mutate_targeted``,
    ``crossover_structural``), the improvement over the previous best,
    and how many evaluations the search had spent since the last record.

A rolling ``summary`` block sits above the events so the headline
question — how much of the progress each component is responsible for —
is answerable without parsing the timeline.

The logger is deliberately defensive: every public method swallows its
own exceptions. Instrumentation must never take down a paid run.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = "specevo-snap/1"

#: Navigator modes, in the order :meth:`SpecEvoOrchestrator._pick_navigator_mode`
#: routes them (ascending stagnation).
NAVIGATOR_MODES = ("synthesis", "surgical", "reframe")

#: Producers a ``new_best`` can be attributed to.
PRODUCERS = ("init", "speculator", "navigator", "navigator_variant")


def classify_producer(source: str) -> str:
    """Map a ``Program.source`` label onto one of :data:`PRODUCERS`.

    ``source`` is the orchestrator's operator label, which is finer
    grained than ``Program.source``'s Literal type: the main loop tags
    mutations as ``mutate_<template>`` / ``mutate_targeted`` and
    crossovers as ``crossover_<template>``.
    """
    if source == "init":
        return "init"
    if source == "navigator_variant" or source.startswith("navigator_variant"):
        return "navigator_variant"
    if source == "Navigator" or source.startswith("navigator_"):
        return "navigator"
    return "speculator"


def _num(x: Any) -> float | None:
    """JSON-safe float: ``None`` for missing / inf / nan.

    ``best_score`` starts at ``-inf`` and the stdlib would serialise that
    as bare ``-Infinity``, which strict JSON parsers reject.
    """
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sanitize(obj: Any) -> Any:
    """Recursively make *obj* strict-JSON safe.

    Applied to the whole document on every write. Two things need it:
    non-finite floats (``-inf`` best scores, ``nan`` metrics) which the
    stdlib would emit as bare ``Infinity`` / ``NaN``, and numpy scalars
    coming out of problem ``score_fn`` metric dicts.
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]
    # numpy scalars and anything else numeric-ish.
    try:
        f = float(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(obj)
    return f if math.isfinite(f) else None


def _clip(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} chars total]"


class SnapLog:
    """Incremental writer for ``<output_dir>/snap.json``.

    Always on — there is no enable flag. One instance per run, owned by
    the orchestrator.
    """

    #: Per-field caps. These are a guard against a runaway LLM answer, not
    #: a storage budget — the code is the whole point of the trace, so the
    #: cap sits far above any real program. Measured over 369 programs in
    #: past runs: mean 153 chars, max 6.4 KB. 200 KB is ~30x the largest
    #: ever observed, so a truncation here means something pathological
    #: happened, and it is marked inline when it does.
    MAX_CODE_CHARS = 200_000
    MAX_DESC_CHARS = 20_000

    def __init__(self, path: str | Path, *, run: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.run_meta: dict[str, Any] = dict(run or {})
        self.events: list[dict[str, Any]] = []
        self.final: dict[str, Any] | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._started = time.time()
        self._write()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def navigator_call(
        self,
        *,
        trigger: int,
        mode: str,
        forced: bool,
        at_eval: int,
        stagnation: float,
        global_stagnation: float,
        local_stagnation: float,
        prev_best: float,
        occupied_cells: int,
        archive_size: int,
        anchors: list[dict[str, Any]],
        n_inspirations: int,
        cost_usd: float,
    ) -> dict[str, Any] | None:
        """Open a ``navigator`` event at the moment the frontier is called.

        Returns the event dict so the caller can complete it via
        :meth:`navigator_result` / :meth:`navigator_fanout` — the outcome
        is only known several awaits later. The event is already on disk
        when this returns, so a run killed mid-frontier-call still shows
        that the Navigator woke and in which mode.
        """
        try:
            with self._lock:
                self._seq += 1
                event: dict[str, Any] = {
                    "event": "navigator",
                    "seq": self._seq,
                    "trigger": trigger,
                    "mode": mode,
                    "mode_forced": bool(forced),
                    "at_eval": at_eval,
                    "elapsed_seconds": round(time.time() - self._started, 3),
                    "cost_usd": _num(cost_usd),
                    "stagnation": _num(stagnation),
                    "global_stagnation": _num(global_stagnation),
                    "local_stagnation": _num(local_stagnation),
                    "prev_best": _num(prev_best),
                    "occupied_cells": occupied_cells,
                    "archive_size": archive_size,
                    "anchors": anchors,
                    "n_anchors": len(anchors),
                    "n_inspirations": n_inspirations,
                    # Filled in by navigator_result().
                    "outcome": "pending",
                    "eval_index": None,
                    "score": None,
                    "delta_vs_prev_best": None,
                    "is_new_best": False,
                    "description": None,
                    "code": None,
                    "error": None,
                    "fanout": None,
                }
                self.events.append(event)
                self._write_locked()
                return event
        except Exception:  # pragma: no cover — never kill a run over a trace
            logger.exception("[snap] failed to record navigator call")
            return None

    def navigator_result(
        self,
        event: dict[str, Any] | None,
        *,
        outcome: str,
        eval_index: int | None = None,
        score: float | None = None,
        delta_vs_prev_best: float | None = None,
        is_new_best: bool = False,
        description: str | None = None,
        code: str | None = None,
        error: str | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Complete a navigator event with what the frontier produced.

        ``outcome`` is one of ``accepted`` / ``rejected`` (evaluated but
        the archive turned it down) / ``eval_error`` / ``parse_miss`` /
        ``llm_error``.
        """
        if event is None:
            return
        try:
            with self._lock:
                event["outcome"] = outcome
                event["eval_index"] = eval_index
                event["score"] = _num(score)
                event["delta_vs_prev_best"] = _num(delta_vs_prev_best)
                event["is_new_best"] = bool(is_new_best)
                event["description"] = _clip(description, self.MAX_DESC_CHARS)
                event["code"] = _clip(code, self.MAX_CODE_CHARS)
                event["error"] = _clip(error, 1_000)
                if cost_usd is not None:
                    event["cost_usd_after"] = _num(cost_usd)
                self._write_locked()
        except Exception:  # pragma: no cover
            logger.exception("[snap] failed to record navigator result")

    def navigator_fanout(self, event: dict[str, Any] | None, fanout: dict[str, Any]) -> None:
        """Attach the cheap-variant fanout summary to a navigator event."""
        if event is None:
            return
        try:
            with self._lock:
                event["fanout"] = fanout
                self._write_locked()
        except Exception:  # pragma: no cover
            logger.exception("[snap] failed to record navigator fanout")

    def new_best(
        self,
        *,
        at_eval: int,
        source: str,
        navigator_mode: str | None,
        model: str,
        score: float,
        prev_best: float,
        evals_since_prev_best: int,
        stagnation_before: float,
        cell_id: int,
        description: str,
        code: str,
        metrics: dict | None,
        cost_usd: float,
    ) -> None:
        """Record one improvement of the run's best score."""
        try:
            with self._lock:
                self._seq += 1
                prev = _num(prev_best)
                cur = _num(score)
                event = {
                    "event": "new_best",
                    "seq": self._seq,
                    "at_eval": at_eval,
                    "elapsed_seconds": round(time.time() - self._started, 3),
                    "cost_usd": _num(cost_usd),
                    "producer": classify_producer(source),
                    "source": source,
                    "navigator_mode": navigator_mode,
                    "model": model,
                    "score": cur,
                    "prev_best": prev,
                    "improvement": (cur - prev) if (cur is not None and prev is not None) else None,
                    "evals_since_prev_best": evals_since_prev_best,
                    "stagnation_before": _num(stagnation_before),
                    "cell_id": cell_id,
                    "metrics": metrics if isinstance(metrics, dict) else {},
                    "description": _clip(description, self.MAX_DESC_CHARS),
                    "code": _clip(code, self.MAX_CODE_CHARS),
                }
                self.events.append(event)
                self._write_locked()
        except Exception:  # pragma: no cover
            logger.exception("[snap] failed to record new best")

    def finalize(self, final: dict[str, Any]) -> None:
        """Attach end-of-run totals and write the trace one last time."""
        try:
            with self._lock:
                self.final = final
                self._write_locked()
        except Exception:  # pragma: no cover
            logger.exception("[snap] failed to finalize")

    # ------------------------------------------------------------------
    # Summary + serialisation
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Roll the timeline up into the numbers the trace exists to answer."""
        nav = [e for e in self.events if e["event"] == "navigator"]
        bests = [e for e in self.events if e["event"] == "new_best"]

        by_mode = {m: 0 for m in NAVIGATOR_MODES}
        accepted_by_mode = {m: 0 for m in NAVIGATOR_MODES}
        new_best_by_mode = {m: 0 for m in NAVIGATOR_MODES}
        failed_by_mode = {m: 0 for m in NAVIGATOR_MODES}
        for e in nav:
            m = e["mode"]
            by_mode[m] = by_mode.get(m, 0) + 1
            if e["outcome"] == "accepted":
                accepted_by_mode[m] = accepted_by_mode.get(m, 0) + 1
            elif e["outcome"] in ("eval_error", "parse_miss", "llm_error"):
                failed_by_mode[m] = failed_by_mode.get(m, 0) + 1
            if e.get("is_new_best"):
                new_best_by_mode[m] = new_best_by_mode.get(m, 0) + 1

        by_producer = {p: 0 for p in PRODUCERS}
        gain_by_producer = {p: 0.0 for p in PRODUCERS}
        by_source: dict[str, int] = {}
        gain_by_source: dict[str, float] = {}
        for e in bests:
            p = e["producer"]
            by_producer[p] = by_producer.get(p, 0) + 1
            by_source[e["source"]] = by_source.get(e["source"], 0) + 1
            imp = e.get("improvement")
            if imp is not None:
                gain_by_producer[p] = gain_by_producer.get(p, 0.0) + imp
                gain_by_source[e["source"]] = gain_by_source.get(e["source"], 0.0) + imp

        # Fanout is the Navigator's indirect contribution: cheap variants
        # spun off a frontier seed. Counted separately from the seed itself.
        fanout_variants = sum(
            int((e.get("fanout") or {}).get("n_variants", 0)) for e in nav
        )
        fanout_accepted = sum(
            int((e.get("fanout") or {}).get("n_accepted", 0)) for e in nav
        )

        return {
            "n_events": len(self.events),
            "navigator": {
                "calls": len(nav),
                "by_mode": by_mode,
                "accepted_by_mode": accepted_by_mode,
                "failed_by_mode": failed_by_mode,
                "new_best_by_mode": new_best_by_mode,
                "fanout_variants": fanout_variants,
                "fanout_accepted": fanout_accepted,
            },
            "new_best": {
                "count": len(bests),
                "by_producer": by_producer,
                "score_gain_by_producer": {
                    k: round(v, 6) for k, v in gain_by_producer.items()
                },
                "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
                "score_gain_by_source": {
                    k: round(v, 6)
                    for k, v in sorted(gain_by_source.items(), key=lambda kv: -kv[1])
                },
                "best_score": bests[-1]["score"] if bests else None,
                "last_at_eval": bests[-1]["at_eval"] if bests else None,
            },
        }

    def _document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": SCHEMA,
            "run": self.run_meta,
            "summary": self.summary(),
            "events": self.events,
        }
        if self.final is not None:
            doc["final"] = self.final
        return doc

    def _write_locked(self) -> None:
        """Atomically rewrite the whole document. Caller holds the lock.

        Rewriting everything on each event is O(trace) but events are
        rare — a few hundred across a run — and the alternative (append
        only) would leave a truncated file unparseable if the run is
        killed mid-write. ``os.replace`` makes the swap atomic, so a
        reader always sees a complete document.
        """
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_sanitize(self._document()), indent=2))
        os.replace(tmp, self.path)

    def _write(self) -> None:
        try:
            with self._lock:
                self._write_locked()
        except Exception:  # pragma: no cover
            logger.exception("[snap] failed to write %s", self.path)
