"""A-star with a configurable Manhattan heuristic weight."""
import heapq
from ml_lab.core.params import Float
from ml_lab.core.registry import register
from ._base import SearchStep
from ._graph import GRAPH_INFO, GraphSearchDemo


@register
class AStarSearchDemo(GraphSearchDemo):
    name = "A* / Weighted A* Search"
    order = 25
    params = GraphSearchDemo.params + [Float("heuristic_weight", 1.0, 0.0, 3.0, 0.25)]
    complexity = "O((V + E) log V) for weight <= 1 on this grid"
    space = "O(V + E), may grow with weighted reopenings"
    info = """## A* search
    Rank states by g + weight * Manhattan distance. Like a robot balancing
    distance already traveled against estimated distance to its destination.
    Weight 0 is Dijkstra. Weight <= 1 is admissible and consistent on this grid
    (minimum step cost 1), so the returned path is minimum-cost with sufficient budget.
    Weight > 1 is weighted A*: often faster, but optimality is not guaranteed.
    Better-cost states can reopen. Budgets include all expansions.
    """ + GRAPH_INFO

    def search(self, data, p):
        start, goal = data["start"], data["goal"]
        heap = [(p.heuristic_weight * self.heuristic(start, goal), 0, start)]
        costs, parents, visited, count, expanded = {start: 0}, {start: None}, set(), 0, 0
        while heap and expanded < p.max_expansions:
            _, cost, node = heapq.heappop(heap)
            if cost != costs[node]:
                continue
            visited.add(node)
            expanded += 1
            yield SearchStep(node, visited, {item[2] for item in heap}, comparisons=count, cost=cost,
                             stored=len(parents) + len(heap), message=f"Expand {node}, g={cost}")
            if node == goal:
                yield self.result(data, parents, visited, count, len(parents) + len(heap))
                return
            for neighbor, weight in self.neighbors(data, node, p):
                count += 1
                candidate = cost + weight
                if candidate < costs.get(neighbor, float("inf")):
                    costs[neighbor], parents[neighbor] = candidate, node
                    priority = candidate + p.heuristic_weight * self.heuristic(neighbor, goal)
                    heapq.heappush(heap, (priority, candidate, neighbor))
        yield self.result(data, {}, visited, count, len(parents) + len(heap),
                          "Expansion budget exhausted" if heap else "No path")