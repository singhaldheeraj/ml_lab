"""FIFO traversal of a graph."""
from collections import deque
from ml_lab.core.registry import register
from ._base import SearchStep
from ._graph import GRAPH_INFO, GraphSearchDemo


@register
class BreadthFirstSearchDemo(GraphSearchDemo):
    name = "Breadth-First Search (BFS)"
    order = 20
    complexity = "O(V + E)"
    space = "O(V)"
    info = """## Breadth-first search
    Expand a FIFO queue in distance layers. Like broadcasting a diagnostic request
    one network hop at a time. Complete on this finite grid with sufficient budget.
    Finds the fewest moves, but not the cheapest weighted path.
    """ + GRAPH_INFO

    def search(self, data, p):
        queue = deque([data["start"]])
        parents, visited, count = {data["start"]: None}, set(), 0
        while queue and len(visited) < p.max_expansions:
            node = queue.popleft()
            visited.add(node)
            yield SearchStep(node, visited, set(queue), comparisons=count,
                             stored=len(parents) + len(queue), message=f"Expand {node}")
            if node == data["goal"]:
                yield self.result(data, parents, visited, count, len(parents) + len(queue))
                return
            for neighbor, _ in self.neighbors(data, node, p):
                count += 1
                if neighbor not in parents:
                    parents[neighbor] = node
                    queue.append(neighbor)
        yield self.result(data, {}, visited, count, len(parents) + len(queue),
                          "Expansion budget exhausted" if queue else "No path")