"""Heuristic breadth layers with a bounded beam width."""
from ml_lab.core.params import Int
from ml_lab.core.registry import register
from ._base import SearchStep
from ._graph import GRAPH_INFO, GraphSearchDemo


@register
class BeamSearchDemo(GraphSearchDemo):
    name = "Beam Search"
    order = 27
    params = GraphSearchDemo.params + [Int("beam_width", 4, 1, 128)]
    complexity = "O(V * b * log(b*w)) upper bound on this grid"
    space = "O(V + b*w), w = beam width"
    info = """## Beam search
    Keep only the best w candidates at each breadth layer by Manhattan distance.
    Like retaining a short list of likely fault diagnoses.
    Narrow beams save frontier space but can permanently discard the only route.
    Neither completeness nor optimality is guaranteed. Bookkeeping still uses O(V).
    """ + GRAPH_INFO

    def search(self, data, p):
        beam = [data["start"]]
        parents, visited, count = {data["start"]: None}, set(), 0
        while beam:
            candidates = []
            for node in beam:
                if len(visited) >= p.max_expansions:
                    yield self.result(data, {}, visited, count, len(parents) + len(candidates), "Expansion budget exhausted")
                    return
                visited.add(node)
                yield SearchStep(node, visited, set(beam), comparisons=count,
                                 stored=len(parents) + len(beam) + len(candidates), message=f"Beam expansion {node}")
                if node == data["goal"]:
                    yield self.result(data, parents, visited, count, len(parents) + len(beam))
                    return
                for neighbor, _ in self.neighbors(data, node, p):
                    count += 1
                    if neighbor not in parents:
                        parents[neighbor] = node
                        candidates.append(neighbor)
            beam = sorted(candidates, key=lambda node: self.heuristic(node, data["goal"]))[:p.beam_width]
        yield self.result(data, {}, visited, count, len(parents), "Beam exhausted; discarded routes may exist")