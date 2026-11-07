"""Uniform-cost (Dijkstra) search on positive weighted grids."""
import heapq
from ml_lab.core.registry import register
from ._base import SearchStep
from ._graph import GRAPH_INFO, GraphSearchDemo


@register
class UniformCostSearchDemo(GraphSearchDemo):
    name = "Uniform-Cost / Dijkstra Search"
    order = 24
    complexity = "O((V + E) log V)"
    space = "O(V + E) lazy priority queue"
    info = """## Uniform-cost / Dijkstra search
    Expand the lowest accumulated-cost state with a priority queue.
    Like routing through a network with different transmission costs.
    Complete and minimum-cost for nonnegative weights with sufficient budget.
    Stale heap entries are discarded. No heuristic is required.
    """ + GRAPH_INFO

    def search(self, data, p):
        heap = [(0, data["start"])]
        distances, parents = {data["start"]: 0}, {data["start"]: None}
        visited, count = set(), 0
        while heap and len(visited) < p.max_expansions:
            cost, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            yield SearchStep(node, visited, {item[1] for item in heap}, comparisons=count, cost=cost,
                             stored=len(parents) + len(heap), message=f"Expand {node}, g={cost}")
            if node == data["goal"]:
                yield self.result(data, parents, visited, count, len(parents) + len(heap))
                return
            for neighbor, weight in self.neighbors(data, node, p):
                count += 1
                candidate = cost + weight
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor], parents[neighbor] = candidate, node
                    heapq.heappush(heap, (candidate, neighbor))
        yield self.result(data, {}, visited, count, len(parents) + len(heap),
                          "Expansion budget exhausted" if heap else "No path")