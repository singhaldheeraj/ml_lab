"""Reproducible multi-peak landscapes for local search."""
import numpy as np
from ml_lab.core.params import Choice, Float, Int
from ._base import SearchDemo, SearchStep


class LocalSearchDemo(SearchDemo):
    family = "local"
    group = "Local optimization search"
    params = [Int("iterations", 100, 10, 500, 10), Int("step_size", 1, 1, 5)]

    def data_params(self):
        return [Int("resolution", 30, 10, 60, 5, group="Data"),
                Choice("landscape", "many peaks", ("many peaks", "single peak", "noisy"), group="Data"),
                Float("start_x", 0.15, 0.0, 1.0, 0.05, group="Data"),
                Float("start_y", 0.15, 0.0, 1.0, 0.05, group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        coordinates = np.linspace(0, 1, p.resolution)
        horizontal, vertical = np.meshgrid(coordinates, coordinates)
        heights = 1.5 * np.exp(-((horizontal - 0.75) ** 2 + (vertical - 0.75) ** 2) / 0.035)
        if p.landscape != "single peak":
            heights += np.exp(-((horizontal - 0.22) ** 2 + (vertical - 0.22) ** 2) / 0.025)
            heights += 0.8 * np.exp(-((horizontal - 0.7) ** 2 + (vertical - 0.2) ** 2) / 0.02)
        if p.landscape == "noisy":
            heights += np.random.default_rng(p.seed).normal(0, 0.08, heights.shape)
        start = (round(p.start_y * (p.resolution - 1)), round(p.start_x * (p.resolution - 1)))
        return {"heights": heights, "start": start}

    @staticmethod
    def neighbors(data, node, step_size):
        for delta_row in (-step_size, 0, step_size):
            for delta_column in (-step_size, 0, step_size):
                neighbor = node[0] + delta_row, node[1] + delta_column
                if neighbor != node and all(0 <= neighbor[axis] < data["heights"].shape[axis] for axis in (0, 1)):
                    yield neighbor

    @staticmethod
    def result(data, best, count, path, message, stored):
        return SearchStep(current=best, path=path, result=best, found=True, done=True,
                          comparisons=count, stored=stored,
                          message=f"{message}; best score {data['heights'][best]:.5f} (global optimum not guaranteed)")

    def _draw(self, data, step):
        self.ax.clear()
        self.ax.imshow(data["heights"], origin="lower", cmap="viridis")
        if step.path:
            self.ax.plot([node[1] for node in step.path], [node[0] for node in step.path], color="white", lw=1.5)
        if step.current is not None:
            self.ax.scatter(step.current[1], step.current[0], s=90, c="#e55054", edgecolor="white")
        optimum = np.unravel_index(np.argmax(data["heights"]), data["heights"].shape)
        self.ax.scatter(optimum[1], optimum[0], marker="*", c="#ffd85e", s=130, label="Global optimum (reference)")
        self.ax.set_title(self.name)
        self.ax.legend(fontsize=8, loc="upper left")


LOCAL_INFO = """
    ## Configuration and measurement
    Maximize the landscape score. Adjust start, landscape, grid resolution, step size,
    iteration budget and seed. Probes count objective evaluations, not iterations.
    Candidate results are not proofs of a global optimum. The gold star is a
    visualization-only exhaustive reference, never used to guide the search.
    Tracked items include the retained visualization trail.
    """