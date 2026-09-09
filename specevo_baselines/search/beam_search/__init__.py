"""
Beam Search module for selecting programs from the solution database.

This module provides a beam search-based database for parent selection,
maintaining a fixed-width beam of the most promising candidates.
"""

from specevo_baselines.search.beam_search.database import BeamSearchDatabase

__all__ = ["BeamSearchDatabase"]
