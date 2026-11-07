"""Quadtree circular-range search with bounding-box pruning."""
import numpy as np
from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register
from ._base import SearchStep
from ._spatial import SpatialSearchDemo


@register
class QuadtreeSearchDemo(SpatialSearchDemo):
    name = "Quadtree Range Search"
    order = 51
    params = [Float("radius", 0.2, 0.05, 0.6, 0.05), Int("leaf_capacity", 4, 1, 16)]
    complexity = "O(n*depth) build; O(n) worst query"
    space = "O(n*depth) worst build storage"
    info = """## Quadtree range search
    Recursively divide the plane into four quadrants. Prune boxes that do not
    intersect the query circle, then test points in surviving leaves.
    Like selecting every obstacle within a robot's safety radius.
    ## Performance
    Build O(n*depth); worst query O(n), often fewer point checks with spatial pruning.
    A depth cap of 16 handles coincident points. Change radius, leaf size, and clusters.
    Compute includes construction. Probe count measures point-distance checks only.
    """

    def search(self, data, p):
        points, query = data["points"], data["query"]
        data["radius"] = p.radius
        node_count = 0

        def build(indices, low, high, depth):
            nonlocal node_count
            node_count += 1
            if len(indices) <= p.leaf_capacity or depth >= 16:
                return low, high, indices, []
            middle, children = (low + high) / 2, []
            for horizontal in (0, 1):
                for vertical in (0, 1):
                    child_low = np.array([middle[0] if horizontal else low[0], middle[1] if vertical else low[1]])
                    child_high = np.array([high[0] if horizontal else middle[0], high[1] if vertical else middle[1]])
                    selected = [index for index in indices if int(points[index, 0] >= middle[0]) == horizontal
                                and int(points[index, 1] >= middle[1]) == vertical]
                    if selected:
                        children.append(build(selected, child_low, child_high, depth + 1))
            return low, high, [], children

        root = build(list(range(len(points))), np.zeros(2), np.ones(2), 0)
        stack, visited, matches, count = [root], set(), [], 0
        while stack:
            low, high, indices, children = stack.pop()
            nearest = np.maximum(low, np.minimum(query, high))
            if np.linalg.norm(nearest - query) > p.radius:
                continue
            stack.extend(children)
            for index in indices:
                count += 1
                visited.add(index)
                if np.linalg.norm(points[index] - query) <= p.radius:
                    matches.append(index)
                yield SearchStep(index, visited, path=list(matches), comparisons=count,
                                 stored=node_count + len(points), message=f"Range test point {index}; {len(matches)} matches")
        yield SearchStep(done=True, found=bool(matches), result=matches, path=matches, visited=visited,
                         comparisons=count, stored=node_count + len(points), message=f"Range query complete: {len(matches)} points")