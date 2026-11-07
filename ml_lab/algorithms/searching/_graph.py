"""Grid search data and path visualization shared by graph plugins."""
import numpy as np
from matplotlib.colors import ListedColormap

from ml_lab.core.params import Choice, Float, Int
from ._base import SearchDemo, SearchStep


class GraphSearchDemo(SearchDemo):
    family = "graph"
    group = "Graph and path search"
    params = [Int("max_expansions", 3000, 100, 10000, 100),
              Choice("neighbor_order", "right first", ("right first", "down first"))]

    def data_params(self):
        return [Int("grid_size", 10, 6, 24, group="Data"),
                Float("wall_density", 0.2, 0.0, 0.5, 0.05, group="Data"),
                Choice("terrain", "uniform", ("uniform", "weighted"), group="Data"),
                Choice("map_style", "barrier", ("random", "barrier", "open"), group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        walls = rng.random((p.grid_size, p.grid_size)) < p.wall_density
        if p.map_style == "open":
            walls[:] = False
        elif p.map_style == "barrier":
            walls[:] = False
            walls[:, p.grid_size // 2] = True
            walls[p.grid_size - 2, p.grid_size // 2] = False
        start, goal = (0, 0), (p.grid_size - 1, p.grid_size - 1)
        walls[start] = walls[goal] = False
        weights = rng.integers(1, 10, walls.shape) if p.terrain == "weighted" else np.ones(walls.shape, dtype=int)
        return {"walls": walls, "weights": weights, "start": start, "goal": goal}

    @staticmethod
    def neighbors(data, node, p):
        directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
        if p.neighbor_order == "down first":
            directions = directions[::-1]
        rows, columns = data["walls"].shape
        for delta_row, delta_column in directions:
            neighbor = (node[0] + delta_row, node[1] + delta_column)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < columns and not data["walls"][neighbor]:
                yield neighbor, int(data["weights"][neighbor])

    @staticmethod
    def heuristic(node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    @staticmethod
    def result(data, parents, visited, count, stored, message=None):
        goal = data["goal"]
        path = []
        if goal in parents:
            node = goal
            while node is not None:
                path.append(node)
                node = parents[node]
            path.reverse()
        cost = sum(int(data["weights"][node]) for node in path[1:]) if path else None
        return SearchStep(visited=visited, path=path, comparisons=count, stored=stored,
                          done=True, found=bool(path), result=path, cost=cost,
                          message=message or (f"Found path: {len(path) - 1} moves" if path else "No path"))

    def _draw(self, data, step):
        self.ax.clear()
        walls = data["walls"]
        image = np.zeros(walls.shape, dtype=int)
        image[walls] = 1
        for nodes, color in ((step.visited, 2), (step.frontier, 3), (step.path, 4)):
            for node in nodes:
                image[node] = color
        if step.current is not None:
            image[step.current] = 5
        self.ax.imshow(image, cmap=ListedColormap(["#f0f3f5", "#303942", "#8db4cf", "#e4b54b", "#16826c", "#d64c4c"]),
                       vmin=0, vmax=5, origin="upper")
        if walls.shape[0] <= 12:
            for row, column in np.ndindex(walls.shape):
                if not walls[row, column]:
                    self.ax.text(column, row, str(data["weights"][row, column]), ha="center", va="center", fontsize=7)
        for label, node in (("S", data["start"]), ("G", data["goal"])):
            self.ax.text(node[1], node[0], label, ha="center", va="center", fontweight="bold",
                         bbox={"facecolor": "white", "edgecolor": "#303942", "pad": 2})
        self.ax.set_title(f"{self.name} | blue: visited, gold: frontier, green: path")
        self.ax.set_xticks([])
        self.ax.set_yticks([])


GRAPH_INFO = """
    ## Grid configuration
    Four-way movement; entering a cell pays its displayed weight. Walls are blocked.
    Change map, density, terrain, seed, neighbor order, or expansion budget.
    Unreachable goals and exhausted budgets are reported separately.
    ## Performance measurement
    Edge checks count generated neighbors. Tracked items estimate algorithm storage,
    not bytes or process memory. Compute time includes Python tracing overhead but
    excludes plotting and animation. V = reachable vertices; E = reachable edges.
    """