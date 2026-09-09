"""RelayEvolve — adaptive population handoff on top of the OpenEvolve backend.

Implements "Relay, Don't Route: Adaptive Population Handoff for Cost-Efficient
LLM-Driven Evolution" with SpecEvo Baselines's ``openevolve_native`` (MAP-Elites +
islands) as the evolutionary backend instead of ShinkaEvolve, and with the
cheap/strong routing baselines the paper compares against.

Search types registered by :mod:`specevo_baselines.search.route`:

==================== ==========================================================
``relayevolve``      Cheap multi-trajectory exploration → Relay-Gain-driven
                     handoff → shared strong-model refinement.
``relay_all_cheap``  Every call uses the cheap model.
``relay_all_strong`` Every call uses the strong model.
``relay_fixed_switch`` Cheap for a fixed prefix of the budget, then strong.
``relay_random``     Independent coin flip per generation.
``relay_bandit``     Two-armed UCB over {cheap, strong}, rewarded by the
                     realized best-so-far improvement.
==================== ==========================================================

Every one of them runs the *parallel* discovery loop
(``config.max_parallel_iterations > 1``): generation and evaluation of
different iterations overlap, so a run is bounded by throughput rather than
by the latency of a single LLM call.
"""

from specevo_baselines.search.relay.bank import (
    Candidate,
    RelayBank,
    curate_seed_population,
    relay_objective,
)
from specevo_baselines.search.relay.baselines import (
    AllCheapController,
    AllStrongController,
    BanditRouteController,
    FixedSwitchController,
    RandomRouteController,
)
from specevo_baselines.search.relay.controller import RelayEvolveController
from specevo_baselines.search.relay.database import RelayEvolveDatabase
from specevo_baselines.search.relay.scheduler import GrowDeepenScheduler

__all__ = [
    "AllCheapController",
    "AllStrongController",
    "BanditRouteController",
    "Candidate",
    "FixedSwitchController",
    "GrowDeepenScheduler",
    "RandomRouteController",
    "RelayBank",
    "RelayEvolveController",
    "RelayEvolveDatabase",
    "curate_seed_population",
    "relay_objective",
]
