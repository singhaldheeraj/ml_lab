"""Exact 0/1 knapsack search using fractional upper-bound pruning."""
import numpy as np
from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register
from ._base import SearchDemo, SearchStep


@register
class BranchAndBoundSearchDemo(SearchDemo):
    name = "Branch and Bound / Knapsack"
    family = "combinatorial"
    group = "Constraint and combinatorial search"
    order = 81
    complexity = "O(n*2^n) worst in this bound implementation"
    space = "O(n^2) recursion with copied selections"
    params = [Int("max_nodes", 5000, 100, 20000, 100)]
    info = """## Branch and bound
    Include or exclude each item. A fractional-knapsack relaxation bounds the
    best value still possible; prune branches whose bound cannot beat the incumbent.
    Like choosing the most valuable tasks that fit a fixed MCU memory budget.
    ## Performance
    Exponential worst case. Sorting by value/weight strengthens the bound.
    Probes count search nodes. A completed search certifies the optimum; a budget
    cutoff reports only the best feasible candidate, not an optimality guarantee.
    """

    def data_params(self):
        return [Int("item_count", 10, 6, 16, group="Data"),
                Float("capacity_fraction", 0.4, 0.1, 0.9, 0.05, group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        weights = rng.integers(1, 15, p.item_count)
        values = rng.integers(5, 40, p.item_count)
        return {"weights": weights, "values": values, "capacity": max(1, int(weights.sum() * p.capacity_fraction))}

    def _draw(self, data, step):
        self.ax.clear()
        colors = ["#16826c" if index in step.path else "#487eb0" for index in range(len(data["values"]))]
        self.ax.bar(range(len(data["values"])), data["values"], color=colors)
        for index, value in enumerate(data["values"]):
            self.ax.text(index, value + 0.5, f"w={data['weights'][index]}", ha="center", fontsize=8)
        weight = sum(int(data["weights"][index]) for index in step.path)
        self.ax.set(xlabel="Item", ylabel="Value", title=f"{self.name} | weight {weight} / capacity {data['capacity']}")
        self.ax.set_ylim(0, max(data["values"]) + 8)

    def search(self, data, p):
        weights, values, capacity = data["weights"], data["values"], data["capacity"]
        order = sorted(range(len(values)), key=lambda index: -values[index] / weights[index])
        best_value, best_items, count, cutoff = 0, [], 0, False

        def bound(level, weight, value):
            remaining, upper = capacity - weight, float(value)
            for index in order[level:]:
                take = min(remaining, int(weights[index]))
                upper += take * float(values[index]) / int(weights[index])
                remaining -= take
                if remaining <= 0:
                    break
            return upper

        def visit(level, weight, value, selected):
            nonlocal best_value, best_items, count, cutoff
            if count >= p.max_nodes:
                cutoff = True
                return
            count += 1
            if value > best_value:
                best_value, best_items = value, selected[:]
            upper = bound(level, weight, value)
            yield SearchStep(path=selected[:], comparisons=count, stored=(level + 1) ** 2,
                             message=f"Node {count}; value {value}; upper bound {upper:.2f}; best {best_value}")
            if level == len(order) or upper < best_value + 1 - 1e-9:
                return
            index = order[level]
            if weight + int(weights[index]) <= capacity:
                yield from visit(level + 1, weight + int(weights[index]), value + int(values[index]), selected + [index])
            yield from visit(level + 1, weight, value, selected)

        yield from visit(0, 0, 0, [])
        yield SearchStep(done=True, found=True, result=best_items, path=best_items, comparisons=count,
                         stored=len(order), message=f"{'Budget exhausted; best feasible' if cutoff else 'Certified optimum'} value {best_value}")