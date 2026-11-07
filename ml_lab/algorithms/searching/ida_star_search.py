"""Iterative deepening on A-star f-cost thresholds."""
from ml_lab.core.registry import register
from ._base import SearchStep
from ._graph import GRAPH_INFO, GraphSearchDemo


@register
class IDAStarSearchDemo(GraphSearchDemo):
    name = "Iterative Deepening A* (IDA*)"
    order = 29
    complexity = "O(b^d) worst, repeated threshold passes"
    space = "O(V + b*d^2) with copied paths"
    info = """## IDA* search
    Run depth-first passes bounded by g + Manhattan distance, raising the threshold
    to the smallest rejected score. Like increasing an allowed route-cost budget.
    Admissible heuristic and positive costs give minimum-cost results when the
    search completes. Repeated passes may exhaust the expansion budget.
    This educational version retains best costs per pass and copied paths.
    """ + GRAPH_INFO

    def search(self, data, p):
        start, goal = data["start"], data["goal"]
        threshold = self.heuristic(start, goal)
        count, expanded = 0, 0
        while True:
            next_threshold = float("inf")
            stack, best, visited = [(start, 0, [start])], {start: 0}, set()
            while stack:
                node, cost, path = stack.pop()
                if cost > best[node]:
                    continue
                score = cost + self.heuristic(node, goal)
                if score > threshold:
                    next_threshold = min(next_threshold, score)
                    continue
                if expanded >= p.max_expansions:
                    yield self.result(data, {}, visited, count, len(best), "Expansion budget exhausted")
                    return
                expanded += 1
                visited.add(node)
                yield SearchStep(node, visited, {item[0] for item in stack}, path=path, comparisons=count,
                                 stored=len(best) + sum(len(item[2]) for item in stack) + len(path),
                                 message=f"Threshold {threshold}; f={score}", cost=cost)
                if node == goal:
                    parents = {path[0]: None}
                    parents.update(zip(path[1:], path))
                    yield self.result(data, parents, visited, count, len(best))
                    return
                for neighbor, weight in reversed(list(self.neighbors(data, node, p))):
                    count += 1
                    candidate = cost + weight
                    if candidate < best.get(neighbor, float("inf")):
                        best[neighbor] = candidate
                        stack.append((neighbor, candidate, path + [neighbor]))
            if next_threshold == float("inf"):
                yield self.result(data, {}, visited, count, len(best))
                return
            threshold = next_threshold