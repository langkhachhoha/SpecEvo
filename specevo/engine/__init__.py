"""SpecEvo — Behavior-Latent Adaptive Discovery Engine.

SpecEvo is the production wiring of SIMPLE-EVO. It reuses the framework's frontier
prompts (3-phase Navigator intervention), error-archive self-repair, and async
producer/consumer parallelism while swapping three subsystems:

* The behavioral archive becomes a top-K description-embedding pool
  (``specevo.simple.Pool``).
* The stagnation signal becomes three sliding-window stats
  (``specevo.simple.Monitor``).
* The 4-D Thompson bandit becomes a UCB-style selector
  (``specevo.simple.Selector``).

The entry point is :func:`specevo.methods.specevo.run_specevo`, which
matches :func:`specevo.evolve_code` so existing problem definitions and
benchmarks plug in unchanged.
"""

from .orchestrator import (
    SpecEvoConfig,
    SpecEvoOrchestrator,
    SpecEvoResult,
    NavigatorTrial,
    format_error_rate_table,
)
from .snaplog import SnapLog, classify_producer

__all__ = [
    "SpecEvoConfig",
    "SpecEvoOrchestrator",
    "SpecEvoResult",
    "NavigatorTrial",
    "SnapLog",
    "classify_producer",
    "format_error_rate_table",
]
