"""Databases for the relay search types.

The population mechanics are unchanged OpenEvolve — island MAP-Elites with
ring migration — because the paper's contribution is *when* a population moves
between models, not how it evolves inside one.  These subclasses exist only so
the ``--search`` flag can name each method and so ``RelayEvolveDatabase`` can
be told apart in logs and checkpoints.
"""

from __future__ import annotations

from specevo_baselines.search.openevolve_native.database import OpenEvolveNativeDatabase


class RelayEvolveDatabase(OpenEvolveNativeDatabase):
    """Population handed off between the cheap and strong phases."""


class RouterDatabase(OpenEvolveNativeDatabase):
    """Single population shared by every cheap/strong routing baseline."""
