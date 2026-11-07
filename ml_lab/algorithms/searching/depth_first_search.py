"""LIFO graph traversal."""
from ml_lab.core.registry import register
from ._base import SearchStep
from ._graph import GRAPH_INFO, GraphSearchDemo


@register
class DepthFirstSearchDemo(GraphSearchDemo):
    name = "Depth-First Search (DFS)"
    order = 21
    complexity = "O(V + E)"
    space = "O(V) graph bookkeeping"
    info = """## Depth-first search
    Follow a branch using a stack before returning to alternatives.
    Like tracing one wiring branch completely before inspecting the next.
    Complete on this finite graph with a visited set and sufficient budget;
    neither shortest-move nor minimum-cost paths are guaranteed.
    """ + GRAPH_INFO

    def search(self, data, p):
        stack = [data["start"]]
        parents, visited, count = {data["start"]: None}, set(), 0
        while stack and len(visited) < p.max_expansions:
            node = stack.pop()
            visited.add(node)
            yield SearchStep(node, visited, set(stack), comparisons=count,
                             stored=len(parents) + len(stack), message=f"Expand {node}")
            if node == data["goal"]:
                yield self.result(data, parents, visited, count, len(parents) + len(stack))
                return
            for neighbor, _ in reversed(list(self.neighbors(data, node, p))):
                count += 1
                if neighbor not in parents:
                    parents[neighbor] = node
                    stack.append(neighbor)
        yield self.result(data, {}, visited, count, len(parents) + len(stack),
                          "Expansion budget exhausted" if stack else "No path")