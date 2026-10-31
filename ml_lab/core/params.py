"""Parameter descriptions. The GUI turns each Param into a widget automatically."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence


@dataclass
class Param:
    name: str
    default: Any
    kind: str = "float"          # float | int | choice | bool
    lo: float = 0.0
    hi: float = 1.0
    step: float | None = None
    choices: Sequence[Any] = ()
    label: str = ""
    help: str = ""
    group: str = "Algorithm"     # "Data" params regenerate the dataset when changed

    def __post_init__(self):
        if not self.label:
            self.label = self.name.replace("_", " ").capitalize()
        if self.step is None and self.kind in ("float", "int"):
            self.step = 1 if self.kind == "int" else (self.hi - self.lo) / 100


def Float(name, default, lo, hi, step=None, **kw) -> Param:
    return Param(name, float(default), "float", lo, hi, step, **kw)


def Int(name, default, lo, hi, step=1, **kw) -> Param:
    return Param(name, int(default), "int", lo, hi, step, **kw)


def Choice(name, default, choices, **kw) -> Param:
    return Param(name, default, "choice", choices=tuple(choices), **kw)


def Bool(name, default, **kw) -> Param:
    return Param(name, bool(default), "bool", **kw)


class Values(SimpleNamespace):
    """Parameter values: access as p.learning_rate or p['learning_rate']."""

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)
