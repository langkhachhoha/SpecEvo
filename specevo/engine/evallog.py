"""``eval_code_log.jsonl`` — the source code of every candidate a run produced.

``snap.json`` records Navigator calls and new bests, and
``checkpoints/checkpoint_NN.json`` dumps the archive at each window close, but
both only ever contain programs that *survived*. A candidate that failed to
parse, raised, scored invalid or hit the evaluator timeout leaves nothing
behind — which is exactly the material you want when the question is "what did
the models actually write, and why did most of it not work".

This writer keeps one record per candidate, whatever became of it:

``code``
    The parsed program, or ``None`` when the model never produced one (in
    which case ``llm_response`` holds what it did produce, if available).
``status``
    ``ok`` (evaluated, no error), ``eval_failed`` (evaluated, error or
    timeout) or ``no_code`` (nothing to evaluate).
``score`` / ``metrics`` / ``error``
    Exactly what the evaluator returned.

Records are appended and flushed one line at a time, so a run killed by
``tmux kill-session`` or a wall-clock cap still has a complete log.
:meth:`finalize` folds the lines into the single ``eval_code_log.json``;
``scripts/eval_log_to_json.py`` does the same for a run that never got there.

Like :mod:`specevo.engine.snaplog`, every public method swallows its own
exceptions — instrumentation must never take down a paid run.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = "eval-code-log/1"


class EvalCodeLog:
    """Thread-safe JSONL writer for per-evaluation code records."""

    def __init__(self, path: str | Path, run: dict[str, Any] | None = None):
        self.path = Path(path)
        self.json_path = self.path.with_suffix(".json")
        self.run = dict(run or {})
        self._lock = threading.Lock()
        self._count = 0
        self._start = time.time()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")
        except Exception:  # pragma: no cover - a broken log must not stop a run
            logger.warning("Could not open %s for writing", self.path, exc_info=True)

    @property
    def count(self) -> int:
        return self._count

    def record(
        self,
        *,
        source: str,
        code: str | None,
        score: float | None = None,
        metrics: dict | None = None,
        error: str | None = None,
        model: str | None = None,
        eval_count: int | None = None,
        llm_response: str | None = None,
        **extra: Any,
    ) -> None:
        """Append one candidate. Never raises."""
        try:
            if code is None:
                status = "no_code"
            elif error is not None:
                status = "eval_failed"
            else:
                status = "ok"
            with self._lock:
                self._count += 1
                record = {
                    "index": self._count,
                    "eval_count": eval_count,
                    "elapsed_s": round(time.time() - self._start, 2),
                    "source": source,
                    "model": model,
                    "status": status,
                    "score": None if score is None or score == float("-inf") else score,
                    "metrics": metrics or {},
                    "error": error,
                    "code": code,
                    "llm_response": llm_response,
                }
                record.update(extra)
                with open(self.path, "a") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
                    fh.flush()
        except Exception:  # pragma: no cover
            logger.debug("Could not append to %s", self.path, exc_info=True)

    def load(self) -> list[dict]:
        records: list[dict] = []
        if not self.path.exists():
            return records
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", self.path)
        return records

    def finalize(self, summary: dict[str, Any] | None = None) -> Path | None:
        """Write the single-JSON view next to the JSONL log."""
        try:
            records = self.load()
            by_status: dict[str, int] = {}
            for r in records:
                key = str(r.get("status"))
                by_status[key] = by_status.get(key, 0) + 1
            payload = {
                "schema": SCHEMA,
                "run": self.run,
                "summary": summary or {},
                "n_records": len(records),
                "by_status": by_status,
                "records": records,
            }
            with self._lock:
                self.json_path.write_text(json.dumps(payload, indent=2, default=str))
            logger.info(
                "[SpecEvo] eval code log: %d candidates (%s) -> %s",
                len(records),
                ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())),
                self.json_path,
            )
            return self.json_path
        except Exception:  # pragma: no cover
            logger.warning("Could not write %s", self.json_path, exc_info=True)
            return None
