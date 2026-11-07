"""Sorting state, operation counters, datasets, and shared visualization."""
from dataclasses import dataclass
from textwrap import fill
from time import perf_counter

import numpy as np

from ml_lab.core.base import AlgorithmDemo
from ml_lab.core.params import Choice, Int


@dataclass
class SortStep:
    values: tuple
    active: tuple = ()
    comparisons: int = 0
    swaps: int = 0
    writes: int = 0
    auxiliary: int = 0
    message: str = "Ready"
    done: bool = False


class SortState:
    def __init__(self, values, descending=False):
        self.values = list(values)
        self.descending = descending
        self.comparisons = 0
        self.swaps = 0
        self.writes = 0
        self.auxiliary = 0

    def compare(self, left, right):
        self.comparisons += 1
        result = int(left > right) - int(left < right)
        return -result if self.descending else result

    def swap(self, first, second):
        if first != second:
            self.values[first], self.values[second] = self.values[second], self.values[first]
            self.swaps += 1
            self.writes += 2

    def write(self, index, value):
        self.values[index] = value
        self.writes += 1

    def event(self, message, *active, done=False):
        return SortStep(tuple(self.values), tuple(active), self.comparisons, self.swaps,
                        self.writes, self.auxiliary, message, done)


class SortingDemo(AlgorithmDemo):
    category = "Sorting algorithms"
    group = "Comparison sorting"
    complexity = "See explanation"
    space = "See explanation"
    stable = False

    def data_params(self):
        return [Int("size", 24, 4, 96, group="Data"),
                Choice("distribution", "random", ("random", "sorted", "reversed", "almost sorted",
                                                    "few unique", "all equal", "signed"), group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def get_params(self):
        return self.data_params() + [Choice("direction", "ascending", ("ascending", "descending")),
                                     Int("events_per_frame", 8, 1, 128,
                                         help="Set to 1 to animate each individual trace event.")] + list(self.params)

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        values = rng.integers(1, 101, p.size)
        if p.distribution in ("sorted", "reversed", "almost sorted"):
            values.sort()
            if p.distribution == "reversed":
                values = values[::-1].copy()
            elif p.distribution == "almost sorted":
                for _ in range(max(1, p.size // 10)):
                    first, second = rng.integers(0, p.size, 2)
                    values[first], values[second] = values[second], values[first]
        elif p.distribution == "few unique":
            values = rng.integers(1, 6, p.size)
        elif p.distribution == "all equal":
            values[:] = 42
        elif p.distribution == "signed":
            values = rng.integers(-100, 101, p.size)
        return {"values": values.tolist()}

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        self.ax, self.metrics_ax = fig.subplots(2, 1, height_ratios=(4, 2.2))

    def draw_data(self, data, p):
        self._draw(SortStep(tuple(data["values"])), 0, 0, 0)

    def _draw(self, step, elapsed, events, peak):
        self.ax.clear()
        colors = ["#16826c" if step.done else "#db5b50" if index in step.active else "#487eb0"
                  for index in range(len(step.values))]
        self.ax.bar(range(len(step.values)), step.values, color=colors)
        self.ax.axhline(0, color="#697780", linewidth=0.6)
        self.ax.set(xlabel="Index", ylabel="Value", title=self.name)
        self.metrics_ax.clear()
        self.metrics_ax.axis("off")
        width = max(32, int(self.metrics_ax.get_window_extent().width / 6.8))
        self.metrics_ax.text(0, 0.99, fill(step.message, width), fontsize=10, va="top",
                             transform=self.metrics_ax.transAxes)
        lines = [f"Comparisons: {step.comparisons:,} | Swaps: {step.swaps:,} | Writes: {step.writes:,}",
                 f"Compute: {elapsed * 1000:.3f} ms | Events: {events:,} | Peak aux items: {peak:,}",
                 f"Time: {self.complexity}", f"Auxiliary space: {self.space} | Stable: {'yes' if self.stable else 'no'}"]
        self.metrics_ax.text(0, 0.72, "\n".join(fill(line, width) for line in lines), fontsize=8.5, va="top",
                             transform=self.metrics_ax.transAxes)

    def run(self, data, p):
        state = SortState(data["values"], p.direction == "descending")
        self.last_step = None
        elapsed, events, peak = 0.0, 0, 0
        iterator = iter(self.sort(state, p))
        while True:
            started = perf_counter()
            try:
                step = next(iterator)
            except StopIteration:
                elapsed += perf_counter() - started
                break
            elapsed += perf_counter() - started
            events += 1
            peak = max(peak, step.auxiliary)
            self.last_step = step
            if events % p.events_per_frame == 0:
                if not self.fast_forward:
                    self._draw(step, elapsed, events, peak)
                yield step.message
        expected = sorted(data["values"], reverse=state.descending)
        if state.values != expected:
            raise RuntimeError(f"{self.name} produced an incorrect sort")
        final = state.event(f"Sorted {len(state.values)} values ({p.direction})", done=True)
        self.last_step = final
        self.performance = {"compute_ms": elapsed * 1000, "comparisons": state.comparisons,
                            "swaps": state.swaps, "writes": state.writes, "events": events, "peak_aux_items": peak}
        self._draw(final, elapsed, events, peak)
        self.log(f"{self.name}: {self.performance}")
        yield final.message


SORT_INFO = """
    ## Configuration
    Choose array size, distribution, random seed, and ascending/descending order.
    Sorted and reversed distributions describe the numeric input, not the chosen output order.
    Events per frame controls batching; set it to 1 for individual comparisons and moves.
    ## Performance
    Comparisons are key comparisons. Swaps exchange two array slots. Writes count
    main-array assignments (a swap counts as two); auxiliary-buffer writes are excluded.
    Distribution sorts also inspect digits or buckets without key comparisons.
    Compute excludes plotting, animation delays, data generation, and final verification,
    but includes Python counters and trace snapshots. These are educational single-run
    timings, not rigorous benchmarks. Aux items are logical storage estimates, not bytes;
    visualization snapshots, the input copy, and GUI memory are excluded.
    Stability means preserving the original order of equal keys.
    """


def integer_bounds(state):
    smallest = largest = state.values[0]
    for value in state.values[1:]:
        state.comparisons += 2
        if value < smallest:
            smallest = value
        if value > largest:
            largest = value
    return smallest, largest