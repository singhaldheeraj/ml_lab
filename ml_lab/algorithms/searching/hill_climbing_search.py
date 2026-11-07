"""Steepest-ascent hill climbing with optional random restarts."""
import numpy as np
from ml_lab.core.params import Int
from ml_lab.core.registry import register
from ._base import SearchStep
from ._local import LOCAL_INFO, LocalSearchDemo


@register
class HillClimbingSearchDemo(LocalSearchDemo):
    name = "Hill Climbing / Random Restarts"
    order = 60
    params = LocalSearchDemo.params + [Int("restarts", 0, 0, 10)]
    complexity = "O(restarts * iterations * neighbors)"
    space = "O(iterations * restarts) retained trail"
    info = """## Hill climbing
    Move to the best strictly better neighbor; stop at a local optimum or plateau.
    Optional seeded random restarts explore other basins. Like tuning a controller
    by keeping only parameter changes that improve the measured response.
    """ + LOCAL_INFO

    def search(self, data, p):
        rng = np.random.default_rng(p.seed)
        best, count, trail = data["start"], 1, []
        for restart in range(p.restarts + 1):
            node = data["start"] if restart == 0 else tuple(int(value) for value in rng.integers(0, len(data["heights"]), 2))
            count += 1
            if data["heights"][node] > data["heights"][best]:
                best = node
            for iteration in range(p.iterations):
                trail.append(node)
                neighbors = list(self.neighbors(data, node, p.step_size))
                count += len(neighbors)
                candidate = max(neighbors, key=lambda neighbor: data["heights"][neighbor], default=node)
                yield SearchStep(node, path=list(trail), comparisons=count, stored=len(trail) + len(neighbors),
                                 message=f"Restart {restart}, iteration {iteration}, score {data['heights'][node]:.5f}")
                if data["heights"][candidate] <= data["heights"][node]:
                    break
                node = candidate
                if data["heights"][node] > data["heights"][best]:
                    best = node
        yield self.result(data, best, count, trail, "Hill climbing finished", len(trail))