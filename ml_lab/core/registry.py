"""Plugin registry: decorate a demo class with @register and drop the file into
ml_lab/algorithms/<anything>/ - it is discovered automatically at start-up.
Files/folders starting with '_' are ignored."""
from __future__ import annotations

import importlib
import pkgutil
import traceback

_REGISTRY: dict[str, type] = {}


def register(cls):
    _REGISTRY[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return cls


def discover(package="ml_lab.algorithms"):
    """Import every module below `package`. Returns (classes, errors)."""
    errors = []
    pkg = importlib.import_module(package)
    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".",
                                      onerror=lambda name: errors.append((name, traceback.format_exc()))):
        if any(part.startswith("_") for part in info.name.split(".")[2:]):
            continue
        try:
            importlib.import_module(info.name)
        except Exception:
            errors.append((info.name, traceback.format_exc()))
    return list(_REGISTRY.values()), errors
