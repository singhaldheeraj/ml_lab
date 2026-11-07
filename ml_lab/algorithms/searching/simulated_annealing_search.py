"""Metropolis local moves with geometric temperature cooling."""
import math
import numpy as np
from ml_lab.core.params import Float
from ml_lab.core.registry import register
from ._base import SearchStep
from ._local import LOCAL_INFO, LocalSearchDemo


@register
class SimulatedAnnealingSearchDemo(LocalSearchDemo):
    name = "Simulated Annealing"
    order = 61
    params = LocalSearchDemo.params + [Float("temperature", 0.3, 0.05, 2.0, 0.05),
                                       Float("cooling", 0.98, 0.8, 1.0, 0.01)]
    complexity = "O(iterations * neighbors) for candidate generation"
    space = "O(iterations) retained trail"
    info = """## Simulated annealing
    Choose a random neighbor. Better moves are accepted; worse moves are accepted
    with probability exp(score_change / temperature). Cool geometrically.
    Like temporarily tolerating worse calibration settings to escape a local basin.
    Finite runs with this cooling schedule do not guarantee global optimality.
    """ + LOCAL_INFO

    def search(self, data, p):
        rng = np.random.default_rng(p.seed)
        node, best, count, trail = data["start"], data["start"], 1, [data["start"]]
        temperature = p.temperature
        for iteration in range(p.iterations):
            neighbors = list(self.neighbors(data, node, p.step_size))
            if not neighbors:
                break
            candidate = neighbors[int(rng.integers(len(neighbors)))]
            count += 1
            delta = float(data["heights"][candidate] - data["heights"][node])
            if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-12)):
                node = candidate
            if data["heights"][node] > data["heights"][best]:
                best = node
            trail.append(node)
            yield SearchStep(node, path=list(trail), comparisons=count, stored=len(trail) + len(neighbors),
                             message=f"Iteration {iteration + 1}; T={temperature:.4f}; score {data['heights'][node]:.5f}")
            temperature *= p.cooling
        yield self.result(data, best, count, trail, "Annealing finished", len(trail))