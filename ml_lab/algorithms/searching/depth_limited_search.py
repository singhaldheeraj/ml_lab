"""Depth-limited traversal, also reused by iterative deepening."""
from ml_lab.core.params import Int
from ml_lab.core.registry import register
from ._base import SearchStep
from ._graph import GRAPH_INFO, GraphSearchDemo


def depth_pass(demo, data, p, limit, budget):
    stack = [(data["start"], [data["start"]])]
    best_depth, visited, count, expanded = {data["start"]: 0}, set(), 0, 0
    while stack and expanded < budget:
        node, path = stack.pop()
        depth = len(path) - 1
        if depth > best_depth[node]:
            continue
        expanded += 1
        visited.add(node)
        stored = len(best_depth) + sum(len(route) for _, route in stack) + len(path)
        yield SearchStep(node, visited, {item[0] for item in stack}, path=path,
                         comparisons=count, stored=stored, message=f"Depth {depth} / limit {limit}")
        if node == data["goal"]:
            parents = {path[0]: None}
            parents.update(zip(path[1:], path))
            yield demo.result(data, parents, visited, count, stored)
            return expanded, True, count, True
        if depth < limit:
            for neighbor, _ in reversed(list(demo.neighbors(data, node, p))):
                count += 1
                if depth + 1 < best_depth.get(neighbor, float("inf")):
                    best_depth[neighbor] = depth + 1
                    stack.append((neighbor, path + [neighbor]))
    return expanded, not stack, count, False


@register
class DepthLimitedSearchDemo(GraphSearchDemo):
    name = "Depth-Limited Search"
    order = 22
    params = GraphSearchDemo.params + [Int("depth_limit", 20, 1, 100)]
    complexity = "O(b^d) upper bound"
    space = "O(V + b*d^2) with copied paths"
    info = """## Depth-limited search
    DFS stops extending branches at the chosen depth. A shallower revisit is
    allowed so a deep first visit cannot hide a valid short route.
    Useful for bounded diagnostic reasoning. A cutoff does not prove no path.
    Not cost-optimal. This demo stores copied paths and best-depth bookkeeping.
    """ + GRAPH_INFO

    def search(self, data, p):
        expanded, completed, count, found = yield from depth_pass(self, data, p, p.depth_limit, p.max_expansions)
        if found:
            return
        if not completed or expanded >= p.max_expansions:
            message = "Expansion budget exhausted"
        else:
            message = "No path within depth limit"
        yield self.result(data, {}, set(), count, 0, message)