"""k-Nearest Neighbours - the 'lazy' learner that just remembers the training data."""
import numpy as np
from matplotlib.patches import Circle

from ml_lab.core import plotting as P
from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Bool, Choice, Int
from ml_lab.core.registry import register


def pairwise_distance(A, B, metric):
    diff = A[:, None, :] - B[None, :, :]
    if metric == "euclidean":
        return np.sqrt((diff ** 2).sum(-1))
    if metric == "manhattan":
        return np.abs(diff).sum(-1)
    return np.abs(diff).max(-1)                    # chebyshev


class KNN:
    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        self.k, self.metric, self.weights = k, metric, weights

    def fit(self, X, y, n_classes=None):
        self.X, self.y = X, y                       # "training" = storing the data
        self.K = n_classes or int(y.max()) + 1
        return self

    def neighbours(self, Q):
        D = pairwise_distance(Q, self.X, self.metric)
        k = min(self.k, len(self.X))
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        return idx, np.take_along_axis(D, idx, 1)

    def votes(self, Q, chunk=1000):
        out = np.zeros((len(Q), self.K))
        for s in range(0, len(Q), chunk):
            idx, d = self.neighbours(Q[s:s + chunk])
            w = 1.0 / (d + 1e-9) if self.weights == "distance" else np.ones_like(d)
            rows = np.repeat(np.arange(len(idx)), idx.shape[1])
            np.add.at(out[s:s + chunk], (rows, self.y[idx].ravel()), w.ravel())
        return out

    def proba(self, Q):
        v = self.votes(Q)
        return v / v.sum(1, keepdims=True)

    def predict(self, Q):
        return self.votes(Q).argmax(1)


@register
class KNNDemo(ClassificationDemo):
    name = "k-Nearest Neighbours (kNN)"
    order = 21
    instant = True
    rerun_on_click = True
    click_help = ("Click mode 'query point': LEFT-click anywhere to see its k neighbours voting.  "
                  "Mode 'add points': LEFT = class 0, RIGHT = class 1, MIDDLE = class 2.")
    params = [
        Int("k", 5, 1, 50, help="How many neighbours vote. Small k = jagged, large k = smooth."),
        Choice("metric", "euclidean", ["euclidean", "manhattan", "chebyshev"],
               help="How 'distance' is measured (L2, L1, L\u221e norms)."),
        Choice("weights", "uniform", ["uniform", "distance"],
               help="'distance': closer neighbours get a bigger vote (1/d)."),
        Choice("click_mode", "query point", ["query point", "add points"]),
        Bool("animate_k", False, help="Animate k = 1 .. k"),
        Int("max_k_curve", 30, 2, 100, help="Range of the accuracy-vs-k curve."),
    ]
    info = """
    ## Idea
    No training at all: store every sample. To classify a new point, find the k closest stored
    points and let them vote. That's it.

    ## Embedded-engineer view
    It's a lookup table with interpolation-by-voting. Great accuracy for small problems, but
    inference cost is O(n) distance computations and you must keep the whole dataset in memory
    - a real constraint on an MCU (flash/RAM!). Also: features must be scaled - if one sensor
    is in mV and another in V, the mV one dominates the distance.

    ## Bias / variance
    k = 1: zero training error, very jagged boundary (over-fit, high variance).
    Large k: smooth boundary, may under-fit.  See the accuracy-vs-k plot on the right.

    ## Try this
    \u2022 k=1 on noisy 'moons' (noise 0.5) - islands appear around single noisy points.
    \u2022 Switch metric to 'chebyshev' or 'manhattan' and look at the shape of boundaries.
    \u2022 Query mode: click a point near the boundary and count the votes.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax_k = fig.add_subplot(gs[1])

    def on_click(self, data, x, y, button, p):
        if p.click_mode == "query point":
            data["query"] = np.array([[x, y]])
            return True
        return super().on_click(data, x, y, button, p)

    def run(self, data, p):
        X, y = data["X"], data["y"]
        K = self.n_classes(data)
        ks = np.arange(1, min(p.max_k_curve, len(X)) + 1)
        tr, te = [], []
        for k in ks:
            m = KNN(k, p.metric, p.weights).fit(X, y, K)
            a, b = self.accuracy(data, m.predict)
            tr.append(a)
            te.append(b)
        for k in (range(1, p.k + 1) if p.animate_k else [p.k]):
            m = KNN(k, p.metric, p.weights).fit(X, y, K)
            self.draw_model(data, m.predict, lambda G: m.proba(G)[:, 1],
                            title=f"kNN  k={k}, {p.metric}, {p.weights}")
            a, b = self.accuracy(data, m.predict)
            P.textbox(self.ax, f"k = {k}\ntrain acc {a:.1%}\ntest  acc {b:.1%}")
            if "query" in data:
                self._draw_query(m, data["query"], p)
            ax = self.ax_k
            ax.clear()
            ax.plot(ks, tr, "o-", ms=3, color=P.color(0), label="train")
            ax.plot(ks, te, "o-", ms=3, color=P.color(3), label="test")
            ax.axvline(k, color="k", ls="--", lw=1)
            ax.legend(fontsize=7)
            P.style(ax, "Accuracy vs. k", "k (neighbours)", "accuracy")
            yield f"k={k}: train acc {a:.1%}, test acc {b:.1%}"

    def _draw_query(self, m, q, p):
        ax = self.ax
        idx, d = m.neighbours(q)
        idx, d = idx[0], d[0]
        for i in idx:
            ax.plot([q[0, 0], m.X[i, 0]], [q[0, 1], m.X[i, 1]], "k-", lw=0.8, zorder=4)
        ax.scatter(m.X[idx, 0], m.X[idx, 1], s=110, facecolors="none", edgecolors="k", lw=1.5, zorder=5)
        if p.metric == "euclidean":
            ax.add_patch(Circle(q[0], d.max(), fill=False, ls="--", color="k", zorder=4))
        v = m.votes(q)[0]
        pred = int(v.argmax())
        ax.scatter(*q[0], marker="*", s=350, c=P.color(pred), edgecolors="k", zorder=6)
        txt = "  ".join(f"c{c}:{v[c]:.2g}" for c in range(len(v)))
        P.textbox(ax, f"query votes\n{txt}\n-> class {pred}", "upper right")
        self.log(f"Query ({q[0, 0]:.2f}, {q[0, 1]:.2f}) votes {txt} -> class {pred}")
