"""Best admissible neighbor search with a short-term tabu list."""
from collections import deque
from ml_lab.core.params import Int
from ml_lab.core.registry import register
from ._base import SearchStep
from ._local import LOCAL_INFO, LocalSearchDemo


@register
class TabuSearchDemo(LocalSearchDemo):
    name = "Tabu Search"
    order = 62
    params = LocalSearchDemo.params + [Int("tabu_tenure", 10, 1, 40)]
    complexity = "O(iterations * neighbors * tenure) in this demo"
    space = "O(iterations + tenure) with trail"
    info = """## Tabu search
    Move to the best admissible neighbor, even if it is worse. Recent states are
    temporarily forbidden; aspiration permits a tabu move that improves the global
    best seen. Like avoiding recently failed calibration settings during a search.
    A finite tabu list reduces immediate cycling but does not guarantee optimality.
    """ + LOCAL_INFO

    def search(self, data, p):
        node, best, count = data["start"], data["start"], 1
        tabu, trail = deque([node], maxlen=p.tabu_tenure), [node]
        for iteration in range(p.iterations):
            neighbors = list(self.neighbors(data, node, p.step_size))
            count += len(neighbors)
            candidates = [neighbor for neighbor in neighbors if neighbor not in tabu or
                          data["heights"][neighbor] > data["heights"][best]]
            if not candidates:
                break
            node = max(candidates, key=lambda neighbor: data["heights"][neighbor])
            if data["heights"][node] > data["heights"][best]:
                best = node
            tabu.append(node)
            trail.append(node)
            yield SearchStep(node, path=list(trail), comparisons=count,
                             stored=len(trail) + len(tabu) + len(neighbors),
                             message=f"Iteration {iteration + 1}; tabu states {len(tabu)}; score {data['heights'][node]:.5f}")
        yield self.result(data, best, count, trail, "Tabu search finished", len(trail) + len(tabu))