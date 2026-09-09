"""
SpecEvo Baselines: Self-Improving Framework for LLMs
"""

from specevo_baselines._version import __version__
from specevo_baselines.api import (
    DiscoveryResult,
    discover_solution,
    run_discovery,
)
from specevo_baselines.runner import Runner

__all__ = [
    "Runner",
    "__version__",
    "run_discovery",
    "discover_solution",
    "DiscoveryResult",
]
