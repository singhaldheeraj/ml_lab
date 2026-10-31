"""Base classes for every algorithm demo.

A demo = (1) parameters, (2) a dataset maker, (3) a run() GENERATOR that draws on
matplotlib axes and `yield`s a status string after every animation frame.
See ml_lab/algorithms/_TEMPLATE.py for a minimal example.
"""
from __future__ import annotations

import numpy as np

from . import datasets as D
from . import plotting as P
from .params import Choice, Float, Int, Param


class AlgorithmDemo:
    name = "Unnamed algorithm"
    category = "Supervised"      # top-level tab (any new name creates a new tab)
    group = "Misc"
    order = 100                  # sort order inside the category
    info = "No description yet."
    params: list[Param] = []
    click_to_add = False         # mouse clicks on the main axes add points
    rerun_on_click = False       # re-run instantly after a click (for fast algorithms)
    instant = False              # re-run instantly when an Algorithm parameter changes
    click_help = ""
    fast_forward = False         # True while the GUI runs to the end ("Finish"): skip heavy drawing

    def __init__(self):
        self.fig = None
        self.ax = None

    # The GUI replaces this with a function that writes into the Log tab.
    def log(self, msg: str):
        print(msg)

    def data_params(self) -> list[Param]:
        return []

    def get_params(self) -> list[Param]:
        return self.data_params() + list(self.params)

    def setup_figure(self, fig):
        """Create axes. Override for multi-panel layouts."""
        fig.clear()
        self.fig = fig
        self.ax = fig.add_subplot(111)

    def make_data(self, p) -> dict:
        return {}

    def draw_data(self, data, p):
        self.ax.clear()
        if "X" in data and "y" in data:
            P.scatter_classes(self.ax, data["X"], data["y"])
        P.style(self.ax, self.name)

    def run(self, data, p):
        self.draw_data(data, p)
        yield "Nothing to run"

    def on_click(self, data, x, y, button, p):
        """Return True if data changed. Default: left=class 0, right=class 1, middle=class 2."""
        if not self.click_to_add or "X" not in data:
            return False
        cls = {1: 0, 3: 1, 2: 2}.get(button, 0)
        if cls == 2 and not getattr(self, "multiclass", False):
            return False
        data["X"] = np.vstack([data["X"], [[x, y]]])
        data["y"] = np.append(data["y"], cls)
        return True


# ======================================================================
class ClassificationDemo(AlgorithmDemo):
    category = "Supervised"
    group = "Classification"
    click_to_add = True
    multiclass = True
    datasets = D.CLASSIFICATION_SETS
    default_dataset = "moons"
    default_n = 200
    default_noise = 0.25
    click_help = "Click on plot: LEFT = add class 0, RIGHT = add class 1, MIDDLE = class 2."

    def data_params(self):
        ps = [Choice("dataset", self.default_dataset, self.datasets, group="Data",
                     help="Shape of the toy problem. Try the hard ones (xor, spiral)!")]
        if self.multiclass:
            ps.append(Int("n_classes", 2, 2, 5, group="Data", help="Used by blobs and spiral."))
        ps += [Int("n_samples", self.default_n, 20, 1000, 10, group="Data"),
               Float("noise", self.default_noise, 0.0, 1.0, 0.01, group="Data",
                     help="Like sensor noise: more noise = more overlap between classes."),
               Int("seed", 0, 0, 999, group="Data", help="Random seed - change for a new sample.")]
        return ps

    def make_data(self, p):
        k = p.get("n_classes", 2)
        X, y = D.classification(p.dataset, p.n_samples, p.noise, p.seed, k)
        Xt, yt = D.classification(p.dataset, max(p.n_samples // 2, 20), p.noise, p.seed + 1000, k)
        return {"X": X, "y": y, "Xt": Xt, "yt": yt}

    @staticmethod
    def n_classes(data):
        return int(max(data["y"].max(), data["yt"].max())) + 1

    def draw_data(self, data, p):
        self.ax.clear()
        P.scatter_classes(self.ax, data["X"], data["y"])
        P.set_limits(self.ax, data["X"])
        P.style(self.ax, f"{self.name} - training data")
        self.ax.legend(fontsize=7, loc="lower right")

    def draw_model(self, data, predict, proba1=None, title="", ax=None, res=120):
        """Decision regions (or P(class 1) map for binary problems) + the data."""
        ax = ax or self.ax
        ax.clear()
        X, y = data["X"], data["y"]
        if proba1 is not None and self.n_classes(data) == 2:
            P.plot_proba(ax, proba1, X, res)
        else:
            P.plot_regions(ax, predict, X, self.n_classes(data), res)
        P.scatter_classes(ax, X, y)
        P.set_limits(ax, X)
        P.style(ax, title or self.name)

    def accuracy(self, data, predict):
        tr = np.mean(predict(data["X"]) == data["y"])
        te = np.mean(predict(data["Xt"]) == data["yt"])
        return tr, te


# ======================================================================
class RegressionDemo(AlgorithmDemo):
    category = "Supervised"
    group = "Regression"
    default_dataset = "sine"
    default_n = 60
    default_noise = 0.2
    datasets = D.REGRESSION_SETS
    click_to_add = True
    click_help = "Click on plot: LEFT = add a training point, RIGHT = remove nearest point."

    def data_params(self):
        return [Choice("dataset", self.default_dataset, self.datasets, group="Data",
                       help="The hidden 'true' function that generated the data."),
                Int("n_samples", self.default_n, 5, 500, 1, group="Data"),
                Float("noise", self.default_noise, 0.0, 1.0, 0.01, group="Data"),
                Float("test_fraction", 0.3, 0.0, 0.8, 0.05, group="Data",
                      help="Portion held back to measure generalisation (like field testing)."),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        x, y, f = D.regression_1d(p.dataset, p.n_samples, p.noise, p.seed)
        tr, te = D.train_test_split(len(x), p.test_fraction, D.rng_from(p.seed + 7))
        return {"x": x[tr], "y": y[tr], "xt": x[te], "yt": y[te], "f": f}

    def draw_data(self, data, p, ax=None, title=None):
        ax = ax or self.ax
        ax.clear()
        xs = np.linspace(-3.2, 3.2, 300)
        ax.plot(xs, data["f"](xs), "k--", lw=1, alpha=0.5, label="true function")
        ax.scatter(data["x"], data["y"], s=22, c=P.color(0), edgecolors="k", lw=0.4,
                   label="train", zorder=3)
        if len(data["xt"]):
            ax.scatter(data["xt"], data["yt"], s=22, c="white", edgecolors=P.color(3),
                       lw=1.2, label="test", zorder=3)
        allx = np.r_[data["x"], data["xt"]]
        ally = np.r_[data["y"], data["yt"]]
        ax.set_ylim(ally.min() - 1, ally.max() + 1)
        ax.set_xlim(min(-3.3, allx.min()), max(3.3, allx.max()))
        P.style(ax, title or f"{self.name}", "x", "y")
        ax.legend(fontsize=7, loc="upper left")

    def on_click(self, data, x, y, button, p):
        if button == 1:
            data["x"] = np.append(data["x"], x)
            data["y"] = np.append(data["y"], y)
        elif len(data["x"]) > 2:
            i = np.argmin(np.hypot(data["x"] - x, data["y"] - y))
            data["x"], data["y"] = np.delete(data["x"], i), np.delete(data["y"], i)
        o = np.argsort(data["x"])
        data["x"], data["y"] = data["x"][o], data["y"][o]
        return True


# ======================================================================
class ClusteringDemo(AlgorithmDemo):
    category = "Unsupervised"
    group = "Clustering"
    datasets = D.CLUSTER_SETS
    default_dataset = "blobs"
    default_n = 300
    default_noise = 0.3
    default_k = 3
    max_n = 1500
    click_to_add = True
    click_help = "Click on plot: LEFT = add a point, RIGHT = remove nearest point."

    def data_params(self):
        return [Choice("dataset", self.default_dataset, self.datasets, group="Data"),
                Int("true_groups", self.default_k, 1, 8, group="Data",
                    help="How many groups the generator creates (blobs/anisotropic/varied/spiral)."),
                Int("n_samples", self.default_n, 20, self.max_n, 10, group="Data"),
                Float("noise", self.default_noise, 0.0, 1.0, 0.01, group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        X, y = D.clusters(p.dataset, p.n_samples, p.noise, p.seed, p.true_groups)
        return {"X": X, "y_true": y}

    def draw_data(self, data, p):
        self.ax.clear()
        self.ax.scatter(data["X"][:, 0], data["X"][:, 1], s=16, c="0.45", edgecolors="none")
        P.set_limits(self.ax, data["X"])
        P.style(self.ax, f"{self.name} - unlabeled data (the algorithm sees only this)")

    def draw_labels(self, ax, X, labels, s=18):
        ax.scatter(X[:, 0], X[:, 1], s=s, c=P.colors_for(labels), edgecolors="none", zorder=3)

    def on_click(self, data, x, y, button, p):
        X = data["X"]
        if button == 1:
            data["X"] = np.vstack([X, [[x, y]]])
            data["y_true"] = np.append(data["y_true"], -1)
        elif len(X) > 3:
            i = np.argmin(np.hypot(X[:, 0] - x, X[:, 1] - y))
            data["X"], data["y_true"] = np.delete(X, i, 0), np.delete(data["y_true"], i)
        return True
