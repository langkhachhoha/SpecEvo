"""
External algorithm backends for SpecEvo Baselines.

Thin wrappers that delegate to external evolution packages (OpenEvolve,
ShinkaEvolve, GEPA, etc.) while returning SpecEvo Baselines's unified DiscoveryResult.

To add a new backend:
  1. Create specevo_baselines/extras/external/<name>_backend.py with an async ``run`` function
  2. Add an entry to ``_BACKENDS`` below
"""

from typing import Awaitable, Callable, Dict

# search_type -> async runner function
_REGISTRY: Dict[str, Callable[..., Awaitable]] = {}

# All backend names we know about, even if the package is not installed.
# Used to give a helpful ImportError instead of "unknown search type".
KNOWN_EXTERNAL = {"openevolve", "shinkaevolve", "gepa"}

# search_type -> actual pip package name (when they differ)
_PACKAGE_NAMES = {
    "shinkaevolve": "shinka",
    "gepa": "gepa[full]",
}


def get_package_name(search_type: str) -> str:
    """Return the pip-installable package name for a search type."""
    return _PACKAGE_NAMES.get(search_type, search_type)


_BACKENDS = [
    ("openevolve", "specevo_baselines.extras.external.openevolve_backend", "run"),
    ("shinkaevolve", "specevo_baselines.extras.external.shinkaevolve_backend", "run"),
    ("gepa", "specevo_baselines.extras.external.gepa_backend", "run"),
]


def is_external(search_type: str) -> bool:
    return search_type in _REGISTRY


def get_runner(search_type: str):
    return _REGISTRY[search_type]


def _register_backends():
    """Attempt to register each backend. Missing packages are silently skipped."""
    import importlib
    import logging

    _logger = logging.getLogger(__name__)
    for name, module_path, func_name in _BACKENDS:
        try:
            mod = importlib.import_module(module_path)
            _REGISTRY[name] = getattr(mod, func_name)
        except ImportError:
            pass  # Package not installed — expected
        except Exception as e:
            _logger.warning("Backend '%s' package is installed but failed to register: %s", name, e)


_register_backends()
