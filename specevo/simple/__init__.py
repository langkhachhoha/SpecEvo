"""SpecEvo primitives.

Three components, in dependency order:

* :class:`ClusterArchive` — adaptive MAP-Elites with hybrid AST +
  description-embedding behavior signature.
* :class:`RankSampler` — Zipfian rank-based parent / inspiration draw.
* :class:`Monitor` — plateau + accept-rate diagnostics.

Plus the same :class:`OutputParser` and :class:`DescriptionEmbedder`
the orchestrator has always used.
"""

from .archive import ArchiveConfig, ClusterArchive, Program
from .ast_features import N_AST_FEATURES, compute_ast_features
from .embedder import DescriptionEmbedder, EmbedderConfig
from .monitor import Monitor, MonitorConfig
from .parser import LLMOutput, OutputParser, ParserConfig
from .rank_sampler import RankSampler, RankSamplerConfig

__all__ = [
    "ArchiveConfig",
    "ClusterArchive",
    "DescriptionEmbedder",
    "EmbedderConfig",
    "LLMOutput",
    "Monitor",
    "MonitorConfig",
    "N_AST_FEATURES",
    "OutputParser",
    "ParserConfig",
    "Program",
    "RankSampler",
    "RankSamplerConfig",
    "compute_ast_features",
]
