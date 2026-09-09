"""Append-only log of every evaluated candidate, source code included.

``relay_progress.jsonl`` records *what happened* per generation (tier, score,
cost) but not *what was generated*, and the checkpoints only hold the programs
the archive kept — a generation whose program failed to parse, crashed, or
timed out leaves no code behind at all. This writer keeps all of it: one record
per generation, whether it produced a valid program or nothing usable.

The live file is JSONL (one record per line, flushed on write) so a run that is
killed still has a complete log up to the moment it died. :meth:`finalize`
folds it into the single ``.json`` array that is easier to load later;
``scripts/eval_log_to_json.py`` does the same for a run that never got there.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EvalCodeLog:
    """Thread-safe JSONL writer for per-evaluation code records."""

    def __init__(self, path: str | Path, meta: Optional[Dict[str, Any]] = None):
        self.path = Path(path)
        self.json_path = self.path.with_suffix(".json")
        self.meta = dict(meta or {})
        self._lock = threading.Lock()
        self._count = 0
        self._start = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh run starts a fresh log; resuming appends, so truncate here
        # rather than in record().
        self.path.write_text("")

    @property
    def count(self) -> int:
        return self._count

    def record(self, **fields: Any) -> None:
        """Append one record. Never raises — instrumentation must not kill a run."""
        try:
            with self._lock:
                self._count += 1
                record = {"index": self._count, "wall_clock_s": round(time.time() - self._start, 2)}
                record.update(fields)
                with open(self.path, "a") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
                    fh.flush()
        except Exception:  # noqa: BLE001 - a broken log must never stop the search
            logger.debug("Could not append to %s", self.path, exc_info=True)

    def load(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
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

    def finalize(self, summary: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Write the single-JSON view next to the JSONL log."""
        try:
            records = self.load()
            payload = {
                "schema": "eval-code-log/1",
                "meta": self.meta,
                "summary": summary or {},
                "n_records": len(records),
                "records": records,
            }
            with self._lock:
                self.json_path.write_text(json.dumps(payload, indent=2, default=str))
            logger.info("Eval code log: %d records -> %s", len(records), self.json_path)
            return self.json_path
        except Exception:  # noqa: BLE001
            logger.warning("Could not write %s", self.json_path, exc_info=True)
            return None
