"""
AdaEvolve - Adaptive Evolutionary Search Algorithm

A gradient-free optimization algorithm that adapts search intensity
per island based on accumulated improvement history.

Core Concepts:
- Improvement Signal (δ): Normalized magnitude of fitness improvement
- Accumulated Signal (G): Decayed sum of squared improvements
- Search Intensity: Adaptive exploration ratio based on G
- UCB with Decay: Island selection with decayed magnitude rewards
"""

from specevo_baselines.search.adaevolve.adaptation import AdaptiveState, MultiDimensionalAdapter
from specevo_baselines.search.adaevolve.controller import AdaEvolveController
from specevo_baselines.search.adaevolve.database import (
    EXPLOIT_LABEL,
    EXPLOIT_LABEL_PROMPT_OPT,
    EXPLORE_LABEL,
    EXPLORE_LABEL_PROMPT_OPT,
    AdaEvolveDatabase,
)

__all__ = [
    "AdaptiveState",
    "MultiDimensionalAdapter",
    "AdaEvolveDatabase",
    "AdaEvolveController",
    "EXPLORE_LABEL",
    "EXPLOIT_LABEL",
    "EXPLORE_LABEL_PROMPT_OPT",
    "EXPLOIT_LABEL_PROMPT_OPT",
]
