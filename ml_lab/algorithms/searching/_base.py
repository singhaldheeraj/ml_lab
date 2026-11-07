"""Shared search events, reproducible data, visualization, and instrumentation."""
from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import fill
from time import perf_counter

import numpy as np

from ml_lab.core.base import AlgorithmDemo
from ml_lab.core.params import Choice, Int


@dataclass
class SearchStep:
    current: object = None
    visited: set = field(default_factory=set)
    frontier: set = field(default_factory=set)
    path: list = field(default_factory=list)
    comparisons: int = 0
    message: str = ""
    done: bool = False
    found: bool = False
    result: object = None
    cost: float | None = None
    stored: int = 1


class SearchDemo(AlgorithmDemo):
    category = "Searching algorithms"
    family = "array"
    complexity = "See explanation"
    space = "See explanation"
    requires_sorted = False

    def data_params(self):
        return [Int("size", 32, 8, 128, group="Data"),
                Choice("distribution", "uniform", ("uniform", "clustered", "duplicates"), group="Data"),
                Choice("target", "middle", ("first", "middle", "last", "absent"), group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        if p.distribution == "duplicates":
            values = np.sort(rng.integers(1, max(3, p.size // 4), p.size))
        elif p.distribution == "clustered":
            values = np.sort(rng.integers(1, 20, p.size) ** 3)
        else:
            values = np.cumsum(rng.integers(1, 10, p.size))
        index = {"first": 0, "middle": p.size // 2, "last": p.size - 1, "absent": -1}[p.target]
        target = int(values[index]) if index >= 0 else int(values[-1]) + 1
        return {"values": values.tolist(), "target": target}

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        self.ax, self.metrics_ax = fig.subplots(2, 1, height_ratios=(4, 2))

    def draw_data(self, data, p):
        self._draw(data, SearchStep(message="Ready"))
        self._metrics(SearchStep(), 0, 0, 0)

    def _draw(self, data, step):
        self.ax.clear()
        values = data["values"]
        colors = ["#16826c" if index in step.path else
                  "#d64c4c" if index == step.current else
                  "#e4b54b" if index in step.frontier else
                  "#8e9ba8" if index in step.visited else "#487eb0"
                  for index in range(len(values))]
        self.ax.bar(range(len(values)), values, color=colors)
        self.ax.axhline(data["target"], color="#d64c4c", linestyle="--", label=f"Target {data['target']}")
        self.ax.set(xlabel="Index", ylabel="Value", title=self.name)
        self.ax.legend(loc="upper left", fontsize=8)

    def _metrics(self, step, elapsed, frames, peak):
        self.metrics_ax.clear()
        self.metrics_ax.axis("off")
        width = max(36, int(self.metrics_ax.get_window_extent().width / 6.4))
        cost = "" if step.cost is None else f" | Cost: {step.cost:g}"
        self.metrics_ax.text(0, 0.98, fill(step.message + cost, width), va="top", fontsize=10,
                             transform=self.metrics_ax.transAxes)
        lines = [f"Probes / edge checks: {step.comparisons:,} | Events: {frames:,} | Peak tracked items: {peak:,}",
             f"Compute: {elapsed * 1000:.3f} ms (drawing and animation excluded)",
             f"Time: {self.complexity}", f"Auxiliary space: {self.space}"]
        self.metrics_ax.text(0, 0.70, "\n".join(fill(line, width) for line in lines),
                             va="top", fontsize=9, transform=self.metrics_ax.transAxes)

    def run(self, data, p):
        self.last_step = None
        elapsed, frames, peak = 0.0, 0, 0
        iterator = iter(self.search(data, p))
        while True:
            started = perf_counter()
            try:
                step = next(iterator)
            except StopIteration:
                break
            elapsed += perf_counter() - started
            frames += 1
            peak = max(peak, step.stored)
            self.last_step = step
            if not self.fast_forward or step.done:
                self._draw(data, step)
                self._metrics(step, elapsed, frames, peak)
            yield step.message
        self.performance = {"compute_ms": elapsed * 1000, "events": frames,
                            "comparisons": self.last_step.comparisons if self.last_step else 0,
                            "peak_items": peak}
        self.log(f"{self.name}: {self.performance}")


def array_step(index, count, **kwargs):
    return SearchStep(current=index, comparisons=count, **kwargs)


def array_result(index, count):
    return array_step(index, count, done=True, found=index is not None, result=index,
                      path=[] if index is None else [index],
                      message="Not found" if index is None else f"Found at index {index}")