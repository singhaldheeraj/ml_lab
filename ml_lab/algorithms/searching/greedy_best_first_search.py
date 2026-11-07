"""Best-first traversal ranked only by estimated goal distance."""
import heapq
from ml_lab.core.registry import register
from ._base import SearchStep
from ._graph import GRAPH_INFO, GraphSearchDemo


@register
class GreedyBestFirstSearchDemo(GraphSearchDemo):
    name = "Greedy Best-First Search"
    order = 26
    complexity = "O((V + E) log V)"
    space = "O(V)"
    info = """## Greedy best-first search
    Expand whichever discovered state looks closest to the goal, ignoring cost so far.
    Like choosing the next road only from the compass direction.
    Complete on this finite grid with sufficient budget and visited bookkeeping,
    but neither fewest moves nor cheapest cost is guaranteed.
    """ + GRAPH_INFO

    def search(self, data, p):
        heap = [(0, data["start"])]
        parents, visited, count = {data["start"]: None}, set(), 0
        while heap and len(visited) < p.max_expansions:
            _, node = heapq.heappop(heap)
            visited.add(node)
            yield SearchStep(node, visited, {item[1] for item in heap}, comparisons=count,
                             stored=len(parents) + len(heap), message=f"Greedy expansion {node}")
            if node == data["goal"]:
                yield self.result(data, parents, visited, count, len(parents) + len(heap))
                return
            for neighbor, _ in self.neighbors(data, node, p):
                count += 1
                if neighbor not in parents:
                    parents[neighbor] = node
                    heapq.heappush(heap, (self.heuristic(neighbor, data["goal"]), neighbor))
        yield self.result(data, {}, visited, count, len(parents) + len(heap),
                          "Expansion budget exhausted" if heap else "No path")