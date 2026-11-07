"""Balanced 2-D k-d tree nearest-neighbor search with exact backtracking."""
import numpy as np
from ml_lab.core.registry import register
from ._base import SearchStep
from ._spatial import SpatialSearchDemo


@register
class KDTreeSearchDemo(SpatialSearchDemo):
    name = "K-D Tree Nearest Neighbor"
    order = 50
    complexity = "O(n log^2 n) demo build; O(log n) typical query, O(n) worst"
    space = "O(n) tree"
    info = """## K-d tree nearest-neighbor search
    Split points alternately by x and y. Visit the near child first; inspect the far
    child only if its splitting plane could beat the best distance found so far.
    Like finding the closest LiDAR return to a predicted landmark.
    ## Performance
    This recursive-sort build is O(n log^2 n); query is typically O(log n) in 2-D,
    O(n) worst. O(n) tree space. Build is included in timing, point-distance checks
    are counted as probes. The green circle is the current best distance.
    """

    def search(self, data, p):
        points, query = data["points"], data["query"]

        def build(indices, depth):
            if not indices:
                return None
            axis = depth % 2
            indices.sort(key=lambda index: points[index, axis])
            middle = len(indices) // 2
            return (indices[middle], axis, build(indices[:middle], depth + 1), build(indices[middle + 1:], depth + 1))

        root = build(list(range(len(points))), 0)
        best_index, best_distance, count, visited = None, float("inf"), 0, set()

        def visit(node):
            nonlocal best_index, best_distance, count
            if node is None:
                return
            index, axis, left, right = node
            distance = float(np.linalg.norm(points[index] - query))
            count += 1
            visited.add(index)
            if distance < best_distance:
                best_index, best_distance = index, distance
            yield SearchStep(index, visited, path=[best_index], comparisons=count, cost=best_distance,
                             stored=len(points), message=f"Point {index}; best distance {best_distance:.4f}")
            delta = query[axis] - points[index, axis]
            near, far = (left, right) if delta < 0 else (right, left)
            yield from visit(near)
            if abs(delta) <= best_distance:
                yield from visit(far)

        yield from visit(root)
        yield SearchStep(done=True, found=best_index is not None, result=best_index,
                         path=[] if best_index is None else [best_index], visited=visited,
                         comparisons=count, stored=len(points), cost=best_distance if best_index is not None else None,
                         message="Nearest neighbor found" if best_index is not None else "No points")