"""Repeat depth-limited passes with increasing limits."""
from ml_lab.core.params import Int
from ml_lab.core.registry import register
from ._graph import GRAPH_INFO, GraphSearchDemo
from .depth_limited_search import depth_pass


@register
class IterativeDeepeningSearchDemo(GraphSearchDemo):
    name = "Iterative Deepening DFS"
    order = 23
    params = GraphSearchDemo.params + [Int("depth_limit", 30, 1, 100)]
    complexity = "O(b^d) upper bound"
    space = "O(V + b*d^2) with copied paths"
    info = """## Iterative deepening
    Repeat depth-limited DFS at limits 0, 1, 2... up to the configured maximum.
    Like increasing the permitted diagnostic chain length one link at a time.
    With sufficient depth and budget it finds a fewest-move path, not necessarily
    minimum weighted cost. Repeated passes trade computation for a bounded frontier.
    """ + GRAPH_INFO

    def search(self, data, p):
        total, count = 0, 0
        for limit in range(p.depth_limit + 1):
            iterator = depth_pass(self, data, p, limit, p.max_expansions - total)
            while True:
                try:
                    step = next(iterator)
                except StopIteration as stop:
                    expanded, completed, pass_count, _ = stop.value
                    break
                step.comparisons += count
                yield step
                if step.done:
                    return
            count += pass_count
            total += expanded
            if not completed or total >= p.max_expansions:
                yield self.result(data, {}, set(), count, 0, "Expansion budget exhausted")
                return
        yield self.result(data, {}, set(), count, 0, "No path within maximum depth")