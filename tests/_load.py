"""Load this repository the way Hermes loads a directory plugin.

The repo root *is* the plugin: ``plugin.yaml`` and ``__init__.py`` sit here, and the loader imports
the directory as a package (``hermes_plugins.<slug>``) with submodule search rooted at it. Tests
need the same shape, so they import it through this helper under the package name the modules use
for their relative imports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "hermes_voicevox"


def install() -> "module":  # type: ignore[name-defined]
    """Import the plugin directory as ``hermes_voicevox`` and return the module."""
    existing = sys.modules.get(PACKAGE)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(
        PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load the plugin from {ROOT}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = PACKAGE
    module.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module
