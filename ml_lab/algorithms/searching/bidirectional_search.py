"""Bidirectional BFS with forward/backward parent reconstruction."""
from ml_lab.core.registry import register
from ._base import SearchStep
from ._graph import GRAPH_INFO, GraphSearchDemo


@register
class BidirectionalSearchDemo(GraphSearchDemo):
    name = "Bidirectional BFS"
    order = 28
    complexity = "O(V + E), often O(b^(d/2)) tree work"
    space = "O(V)"
    info = """## Bidirectional breadth-first search
    Alternate complete breadth layers from start and goal until they meet.
    Like two technicians tracing a cable from opposite ends.
    Finds a fewest-move route on this undirected grid, not minimum weighted cost.
    Both endpoints and reverse connectivity must be known.
    """ + GRAPH_INFO

    def search(self, data, p):
        start, goal = data["start"], data["goal"]
        fronts, maps = [{start}, {goal}], [{start: None}, {goal: None}]
        visited, count, expanded = set(), 0, 0
        if start == goal:
            yield self.result(data, maps[0], visited, count, 2)
            return
        side = 0
        while fronts[0] and fronts[1]:
            next_front = set()
            for node in sorted(fronts[side]):
                if expanded >= p.max_expansions:
                    yield self.result(data, {}, visited, count, sum(map(len, maps)), "Expansion budget exhausted")
                    return
                expanded += 1
                visited.add(node)
                yield SearchStep(node, visited, fronts[0] | fronts[1], comparisons=count,
                                 stored=sum(map(len, maps)) + sum(map(len, fronts)), message=f"Expand side {side}: {node}")
                for neighbor, _ in self.neighbors(data, node, p):
                    count += 1
                    if neighbor not in maps[side]:
                        maps[side][neighbor] = node
                        next_front.add(neighbor)
                    if neighbor in maps[1 - side]:
                        chain, current = [], neighbor
                        while current is not None:
                            chain.append(current)
                            current = maps[0][current]
                        chain.reverse()
                        current = maps[1][neighbor]
                        while current is not None:
                            chain.append(current)
                            current = maps[1][current]
                        parents = {chain[0]: None}
                        parents.update(zip(chain[1:], chain))
                        yield self.result(data, parents, visited, count, sum(map(len, maps)))
                        return
            fronts[side] = next_front
            side = 1 - side
        yield self.result(data, {}, visited, count, sum(map(len, maps)))