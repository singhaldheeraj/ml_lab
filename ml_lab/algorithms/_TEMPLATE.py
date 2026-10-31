"""TEMPLATE - copy this file to ml_lab/algorithms/supervised/my_algo.py (no leading
underscore!), rename the classes and restart ML Lab. Your tab appears automatically.

Structure used by every algorithm in this project:
  1. A pure-NumPy model class  -> the maths. No plotting. Easy to port to C later.
  2. A Demo class              -> parameters + visualisation, decorated with @register.
"""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Bool, Choice, Float, Int
from ml_lab.core.registry import register


class NearestCentroid:
    """Simplest classifier imaginable: predict the class whose mean is closest."""

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = np.array([X[y == c].mean(0) for c in self.classes])
        return self

    def predict(self, X):
        d = ((X[:, None, :] - self.centroids[None]) ** 2).sum(-1)
        return self.classes[d.argmin(1)]


@register
class MyAlgorithmDemo(ClassificationDemo):
    name = "My Algorithm (template)"
    category = "My Lab"            # creates a new top-level tab
    group = "Classification"
    order = 1
    instant = True                 # re-run when a parameter changes
    rerun_on_click = True
    info = """
    ## Idea
    Write what the algorithm does here. Lines starting with '## ' become headings,
    indented lines are shown in a monospace font.
    """
    params = [
        Float("shrink", 0.0, 0.0, 1.0, 0.05, help="Pull centroids toward the origin."),
        Int("repeat", 5, 1, 20, help="Number of animation frames."),
        Choice("style", "regions", ["regions", "points only"]),
        Bool("show_centroids", True),
    ]

    def run(self, data, p):
        # p.<name> gives the current value of each parameter above.
        for i in range(p.repeat):
            model = NearestCentroid().fit(data["X"], data["y"])
            model.centroids *= 1 - p.shrink * (i + 1) / p.repeat
            if p.style == "regions":
                self.draw_model(data, model.predict, title="Nearest centroid")
            else:
                self.draw_data(data, p)
            if p.show_centroids:
                self.ax.scatter(*model.centroids.T, marker="X", s=200, c="k")
            tr, te = self.accuracy(data, model.predict)
            yield f"frame {i + 1}: train acc {tr:.1%}  test acc {te:.1%}"   # one animation frame
        self.log("Done - this message goes to the Log tab.")
