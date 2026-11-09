"""Exact and heuristic searches for a closed Euclidean traveling-salesman tour."""
from itertools import permutations

import numpy as np

from ml_lab.core.params import Choice, Int
from ml_lab.core.registry import register
from ._base import SearchDemo, SearchStep


@register
class TravelingSalesmanDemo(SearchDemo):
    name = "Traveling Salesman Problem (TSP)"
    family = "route"
    group = "Route optimization"
    order = 82
    params = [Choice("method", "nearest neighbor + 2-opt",
                     ("nearest neighbor + 2-opt", "exhaustive (exact)"))]
    complexity = "Exact: O(n * (n-1)!); heuristic: O(n^2 + passes * n^3)"
    space = "O(n^2) distances + O(n) tour"
    info = """## Traveling salesman problem
    Visit each city exactly once, then return to city 0. Minimize total Euclidean
    distance, including the closing edge. Exhaustive search fixes city 0 and tries
    every permutation of the remaining cities; it guarantees an optimal tour.
    Nearest neighbor builds a tour greedily, then 2-opt reverses segments whenever
    that shortens the route. This faster mode can stop at a local optimum.
    Like scheduling a robot's inspection stops while minimizing travel distance.

    ## Configuration and measurement
    Adjust city count, layout, seed and solving method. Exact mode is bounded to
    at most nine cities. Probes count evaluated tours and greedy candidate edges.
    The red city is the start; the green line includes the return journey.
    """

    def data_params(self):
        return [Int("cities", 7, 3, 9, group="Data"),
                Choice("layout", "random", ("random", "circle", "clusters"), group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        if p.layout == "circle":
            angles = np.linspace(0, 2 * np.pi, p.cities, endpoint=False)
            points = np.column_stack((np.cos(angles), np.sin(angles)))
        elif p.layout == "clusters":
            centers = np.array([[0.2, 0.2], [0.8, 0.8]])
            points = centers[np.arange(p.cities) % 2] + rng.normal(0, 0.07, (p.cities, 2))
        else:
            points = rng.random((p.cities, 2))
        return {"points": points}

    def search(self, data, p):
        points = data["points"]
        size = len(points)
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        count = 0

        def length(tour):
            return float(sum(distances[previous, node] for previous, node in zip(tour, tour[1:])))

        def event(tour, cost, message, done=False):
            return SearchStep(current=tour[-1], path=list(tour), cost=cost, comparisons=count,
                              stored=size * size + 3 * size, message=message,
                              done=done, found=done, result=list(tour) if done else None)

        if p.method == "exhaustive (exact)":
            if size > 9:
                raise ValueError("Exact TSP is limited to nine cities")
            best, cost = None, float("inf")
            for ordering in permutations(range(1, size)):
                tour = [0, *ordering, 0]
                candidate_cost = length(tour)
                count += 1
                if candidate_cost < cost:
                    best, cost = tour, candidate_cost
                    yield event(best, cost, f"Improved tour after {count} candidates")
                elif count % 200 == 0:
                    yield event(best, cost, f"Checked {count} candidate tours")
            yield event(best, cost, "Optimal closed tour (exhaustive search)", done=True)
            return

        tour, remaining = [0], set(range(1, size))
        while remaining:
            current = tour[-1]
            count += len(remaining)
            following = min(remaining, key=lambda node: (distances[current, node], node))
            tour.append(following)
            remaining.remove(following)
            yield event(tour, length(tour), f"Nearest neighbor: visit city {following}")
        tour.append(0)
        cost = length(tour)
        yield event(tour, cost, "Greedy closed tour; begin 2-opt")
        improved = True
        while improved:
            improved = False
            for start in range(1, size - 1):
                for end in range(start + 1, size):
                    candidate = tour[:start] + tour[start:end + 1][::-1] + tour[end + 1:]
                    candidate_cost = length(candidate)
                    count += 1
                    if candidate_cost < cost - 1e-12:
                        tour, cost, improved = candidate, candidate_cost, True
                        yield event(tour, cost, f"2-opt: reverse cities at positions {start} to {end}")
        yield event(tour, cost, "2-opt locally optimal closed tour; global optimum not guaranteed", done=True)

    def _draw(self, data, step):
        self.ax.clear()
        points = data["points"]
        if step.path:
            route = points[step.path]
            self.ax.plot(route[:, 0], route[:, 1], color="#16826c", lw=2)
        self.ax.scatter(points[:, 0], points[:, 1], c="#487eb0", s=90, zorder=3)
        self.ax.scatter(*points[0], c="#d64c4c", s=110, zorder=4)
        for index, point in enumerate(points):
            self.ax.annotate(str(index), point, xytext=(6, 6), textcoords="offset points")
        self.ax.set(title=self.name, xlabel="X", ylabel="Y", aspect="equal")
        self.ax.margins(0.15)