"""Stagnation monitor for SpecEvo.

Two stagnation signals, combined by ``max`` into a single
``stagnation_level()`` consumed by the rank sampler:

* **Global stagnation**: evaluations since the global *best* score
  increased — i.e. since the last ``NEW BEST`` event. NEW BEST is
  rare (a handful of events in a multi-hour run), so this signal
  saturates only when the search has truly halted.

* **Local stagnation**: evaluations since the last *admit* — i.e. since
  the last time any cell's incumbent was replaced. Admits are far more
  frequent than NEW BEST events: every time a worker beats a cell
  incumbent (even one that is not the global best), we count it as a
  small step of local progress. When admits also dry up, the search is
  not just failing to find a new global best, it is failing to improve
  any cell — that is the regime where we want β to drop and the sampler
  to broaden.

The combined level is

    stagnation = max(plateau_steps  / plateau_max,
                     admit_gap      / admit_gap_max)

so β shrinks as soon as *either* signal saturates. Tuning them
independently (``admit_gap_max`` smaller than ``plateau_max`` by ~5×) is
intentional: admits happen ~5× more often than new-best events, so we
want the admit-side timer to fire on a tighter horizon.

``accept_rate`` is still reported for the meta-advisor prompt but does
not switch any control pathway.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    plateau_max: int = 100
    """Denominator for the **global** stagnation signal — evaluations
    since the last NEW BEST event. At ``plateau_max``, that signal
    saturates to 1.0. Tuned so one Navigator interval (default 50
    evals) corresponds to ~0.5 on the global signal."""

    admit_gap_max: int = 20
    """Denominator for the **local** stagnation signal — evaluations
    since the last admit (any cell incumbent replacement). Smaller than
    ``plateau_max`` because admits are far more frequent than new-best
    events; we want the admit timer to fire on a tighter horizon.

    The combined :meth:`Monitor.stagnation_level` is
    ``max(plateau_steps / plateau_max, admit_gap / admit_gap_max)``, so
    20 evals with **zero admits** is enough to push the rank sampler
    fully toward exploration even when the last new-best is only a few
    dozen evals old."""

    accept_window_size: int = 50
    """How many evaluations to track for the rolling accept-rate."""

    score_eps: float = 1e-9


@dataclass
class Monitor:
    config: MonitorConfig = field(default_factory=MonitorConfig)

    eval_count: int = 0
    best_score: float = -math.inf
    last_best_eval: int = 0
    last_admit_eval: int = 0
    """Last evaluation at which the archive accepted a program (admit
    semantics: the candidate either opened a new cell or replaced a cell
    incumbent). Drives the local stagnation signal."""

    accept_window: deque[bool] = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self) -> None:
        if self.accept_window.maxlen != self.config.accept_window_size:
            self.accept_window = deque(self.accept_window, maxlen=self.config.accept_window_size)

    def record_eval(self, *, score: float, accepted: bool) -> None:
        self.eval_count += 1
        self.accept_window.append(bool(accepted))
        if accepted:
            # Every admit advances the local-progress clock — even if
            # this candidate is far from the global best. The semantic
            # claim is that an admit means *some* cell improved, which
            # is the unit of progress the rank sampler should respect.
            self.last_admit_eval = self.eval_count
        if accepted and score > self.best_score + self.config.score_eps:
            self.best_score = score
            self.last_best_eval = self.eval_count

    @property
    def plateau_steps(self) -> int:
        """Evals since the last NEW BEST event (global stagnation timer)."""
        return self.eval_count - self.last_best_eval

    @property
    def admit_gap(self) -> int:
        """Evals since the last admit (local stagnation timer)."""
        return self.eval_count - self.last_admit_eval

    def global_stagnation(self) -> float:
        """Plateau-side signal in [0, 1] (1 = ``plateau_max`` evals without a new best)."""
        return min(1.0, self.plateau_steps / max(1, self.config.plateau_max))

    def local_stagnation(self) -> float:
        """Admit-side signal in [0, 1] (1 = ``admit_gap_max`` evals without an admit)."""
        return min(1.0, self.admit_gap / max(1, self.config.admit_gap_max))

    def stagnation_level(self) -> float:
        """Combined stagnation in [0, 1]: max of the global and local signals.

        Either signal saturating is enough to drive β fully toward
        exploration. In practice ``local_stagnation`` is the one that
        moves on the second-to-second timescale; ``global_stagnation``
        accumulates more slowly and pushes things over the edge when the
        search truly stalls.
        """
        return max(self.global_stagnation(), self.local_stagnation())

    def acceptance_rate(self) -> float:
        if not self.accept_window:
            return 1.0
        return sum(self.accept_window) / len(self.accept_window)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "eval_count": self.eval_count,
            "best_score": self.best_score if math.isfinite(self.best_score) else float("nan"),
            "plateau_steps": self.plateau_steps,
            "admit_gap": self.admit_gap,
            "global_stagnation": self.global_stagnation(),
            "local_stagnation": self.local_stagnation(),
            "stagnation_level": self.stagnation_level(),
            "accept_rate": self.acceptance_rate(),
        }
