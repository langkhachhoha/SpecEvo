"""
Routing for search algorithms.

Maps the ``--search`` flag to the right database, controller, and program class
at runtime.  The registries and factory functions live in ``registry.py``;
this module wires up implementations and provides ``get_discovery_controller``.
"""

import logging

from specevo_baselines.search.adaevolve.controller import AdaEvolveController
from specevo_baselines.search.adaevolve.database import AdaEvolveDatabase
from specevo_baselines.search.beam_search.database import BeamSearchDatabase

# Algorithm implementations
from specevo_baselines.search.best_of_n.database import BestOfNDatabase
from specevo_baselines.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from specevo_baselines.search.evox.controller import CoEvolutionController
from specevo_baselines.search.evox.database.search_strategy_db import SearchStrategyDatabase
from specevo_baselines.search.gepa_native.controller import GEPANativeController
from specevo_baselines.search.gepa_native.database import GEPANativeDatabase
from specevo_baselines.search.openevolve_native.database import OpenEvolveNativeDatabase
from specevo_baselines.search.registry import (
    _CONTROLLER_REGISTRY,
    register_controller,
    register_database,
)
from specevo_baselines.search.relay.baselines import (
    AllCheapController,
    AllStrongController,
    BanditRouteController,
    FixedSwitchController,
    RandomRouteController,
)
from specevo_baselines.search.relay.controller import RelayEvolveController
from specevo_baselines.search.relay.database import RelayEvolveDatabase, RouterDatabase
from specevo_baselines.search.topk.database import TopKDatabase

logger = logging.getLogger(__name__)


######################### ROUTING #########################


def get_discovery_controller(controller_input: DiscoveryControllerInput) -> DiscoveryController:
    """
    Get the discovery controller for a given search type.

    Returns the registered controller class, or the default DiscoveryController
    if none is registered.
    """
    search_type = controller_input.config.search.type
    controller_class = _CONTROLLER_REGISTRY.get(search_type, DiscoveryController)
    logger.debug(f"Using controller {controller_class.__name__} for search type '{search_type}'")
    return controller_class(controller_input)


######################### AUTO-REGISTRATION #########################

register_database("best_of_n", BestOfNDatabase)
register_database("beam_search", BeamSearchDatabase)
register_database("topk", TopKDatabase)

# AdaEvolve
register_database("adaevolve", AdaEvolveDatabase)
register_controller("adaevolve", AdaEvolveController)

# OpenEvolve Native
register_database("openevolve_native", OpenEvolveNativeDatabase)

# RelayEvolve: adaptive population handoff (cheap explore -> strong refine)
register_database("relayevolve", RelayEvolveDatabase)
register_controller("relayevolve", RelayEvolveController)

# Cheap/strong routing baselines, same backend and same budget accounting
for _relay_type, _relay_controller in (
    ("relay_all_cheap", AllCheapController),
    ("relay_all_strong", AllStrongController),
    ("relay_fixed_switch", FixedSwitchController),
    ("relay_random", RandomRouteController),
    ("relay_bandit", BanditRouteController),
):
    register_database(_relay_type, RouterDatabase)
    register_controller(_relay_type, _relay_controller)

# EvoX
register_controller("evox", CoEvolutionController)
register_database("evox_meta", SearchStrategyDatabase)

# GEPA Native: guided evolution with acceptance gating and merge
register_database("gepa_native", GEPANativeDatabase)
register_controller("gepa_native", GEPANativeController)
