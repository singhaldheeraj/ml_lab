"""Shared two-dimensional query datasets and rendering."""
import numpy as np
from matplotlib.patches import Circle
from ml_lab.core.params import Choice, Float, Int
from ._base import SearchDemo


class SpatialSearchDemo(SearchDemo):
    family = "spatial"
    group = "Spatial search"

    def data_params(self):
        return [Int("point_count", 80, 10, 300, 10, group="Data"),
                Choice("distribution", "uniform", ("uniform", "clusters"), group="Data"),
                Float("query_x", 0.5, 0.0, 1.0, 0.05, group="Data"),
                Float("query_y", 0.5, 0.0, 1.0, 0.05, group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        points = rng.random((p.point_count, 2))
        if p.distribution == "clusters":
            points = np.clip(rng.normal(0, 0.08, points.shape) + rng.choice([0.25, 0.75], (p.point_count, 1)), 0, 1)
        return {"points": points, "query": np.array([p.query_x, p.query_y])}

    def _draw(self, data, step):
        self.ax.clear()
        points = data["points"]
        colors = ["#16826c" if index in step.path else "#d64c4c" if index == step.current else
                  "#e4b54b" if index in step.visited else "#487eb0" for index in range(len(points))]
        self.ax.scatter(points[:, 0], points[:, 1], c=colors, s=28)
        self.ax.scatter(*data["query"], marker="*", s=180, c="#d64c4c", edgecolor="black", label="Query")
        radius = data.get("radius", step.cost)
        if radius is not None:
            self.ax.add_patch(Circle(data["query"], radius, fill=False, color="#16826c", linestyle="--"))
        self.ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), title=self.name)
        self.ax.set_aspect("equal")
        self.ax.legend(fontsize=8)