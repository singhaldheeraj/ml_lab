"""Anomaly / outlier detection: Gaussian (Mahalanobis), k-NN distance and Isolation Forest."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import AlgorithmDemo
from ml_lab.core.datasets import rng_from
from ml_lab.core.params import Choice, Float, Int
from ml_lab.core.registry import register

METHODS = ["Gaussian (Mahalanobis)", "k-NN distance", "Isolation Forest"]


class GaussianDetector:
    def fit(self, X):
        self.mu = X.mean(0)
        self.inv = np.linalg.inv(np.cov(X.T) + 1e-6 * np.eye(X.shape[1]))
        return self

    def score(self, Q):                               # squared Mahalanobis distance
        d = Q - self.mu
        return np.einsum("ij,jk,ik->i", d, self.inv, d)


class KNNDetector:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X):
        self.X = X
        return self

    def score(self, Q, chunk=2000):
        out = []
        for s in range(0, len(Q), chunk):
            D = np.sqrt(((Q[s:s + chunk, None] - self.X[None]) ** 2).sum(-1))
            out.append(np.sort(D, 1)[:, min(self.k, len(self.X) - 1)])
        return np.concatenate(out)


def c_factor(n):
    """Average path length of an unsuccessful search in a binary search tree of n nodes."""
    n = np.asarray(n, float)
    return np.where(n > 2, 2 * (np.log(np.maximum(n - 1, 1)) + 0.5772156649) - 2 * (n - 1) / np.maximum(n, 1),
                    np.where(n == 2, 1.0, 0.0))


class IsolationForest:
    """Anomalies are 'few and different' -> random splits isolate them in very few steps."""

    def __init__(self, n_trees=100, sample_size=128, rng=None):
        self.n_trees, self.psi = n_trees, sample_size
        self.rng = rng or np.random.default_rng(0)

    def _build(self, X, depth, limit):
        if depth >= limit or len(X) <= 1:
            return ("leaf", len(X))
        f = int(self.rng.integers(X.shape[1]))
        lo, hi = X[:, f].min(), X[:, f].max()
        if lo == hi:
            return ("leaf", len(X))
        s = self.rng.uniform(lo, hi)
        m = X[:, f] < s
        return (f, s, self._build(X[m], depth + 1, limit), self._build(X[~m], depth + 1, limit))

    def fit(self, X):
        psi = min(self.psi, len(X))
        limit = int(np.ceil(np.log2(max(psi, 2))))
        self.trees = [self._build(X[self.rng.choice(len(X), psi, replace=False)], 0, limit)
                      for _ in range(self.n_trees)]
        self.c = c_factor(psi)
        return self

    def _path(self, node, Q, idx, depth, out):
        if node[0] == "leaf":
            out[idx] = depth + c_factor(node[1])
            return
        f, s, left, right = node
        m = Q[idx, f] < s
        self._path(left, Q, idx[m], depth + 1, out)
        self._path(right, Q, idx[~m], depth + 1, out)

    def score(self, Q):
        total = np.zeros(len(Q))
        for t in self.trees:
            out = np.zeros(len(Q))
            self._path(t, Q, np.arange(len(Q)), 0, out)
            total += out
        return 2 ** (-(total / self.n_trees) / self.c)       # ~1 = anomaly, < 0.5 = normal


@register
class AnomalyDemo(AlgorithmDemo):
    name = "Anomaly Detection"
    category = "Unsupervised"
    group = "Anomaly Detection"
    order = 40
    instant = True

    def data_params(self):
        return [Choice("dataset", "one cluster", ["one cluster", "two clusters", "banana"], group="Data"),
                Int("n_normal", 300, 30, 1000, 10, group="Data"),
                Float("outlier_fraction", 0.05, 0.0, 0.3, 0.01, group="Data",
                      help="Fraction of injected faults (uniform random points)."),
                Float("noise", 0.3, 0.0, 1.0, 0.01, group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    params = [
        Choice("method", METHODS[0], METHODS),
        Float("contamination", 0.05, 0.005, 0.3, 0.005,
              help="Expected fraction of anomalies -> sets the alarm threshold (score quantile)."),
        Int("k", 5, 1, 30, help="k-NN method: distance to the k-th neighbour."),
        Int("n_trees", 100, 10, 300, help="Isolation Forest trees."),
        Int("sample_size", 128, 16, 512, help="Isolation Forest sub-sample per tree."),
    ]
    info = """
    ## Idea
    Learn what NORMAL looks like (no fault labels needed), then raise an alarm for anything
    that doesn't fit.  Every method produces an anomaly SCORE; the threshold is set so that
    'contamination' fraction of the training data is flagged.

    ## Methods
        Gaussian    : fit mean + covariance; score = Mahalanobis distance (elliptic envelope)
        k-NN        : score = distance to the k-th nearest normal point (works on any shape)
        Isol. Forest: random cuts; outliers get isolated after very few cuts (short paths)

    ## Embedded-engineer view
    This is condition monitoring / predictive maintenance: learn the healthy vibration or
    current signature of a motor, flag deviations.  The Gaussian method is just a mean vector
    + inverse covariance matrix - a few dozen bytes and a quadratic form per sample: runs on any
    MCU.  The Mahalanobis distance is also the core of the chi-square test used for gating in
    Kalman filters!

    ## Panels
    Left: anomaly score map, alarm contour (red), flagged points (\u00d7). Red rings = true injected
    faults.  Right: score histogram with threshold.

    ## Try this
    \u2022 'two clusters' or 'banana' with Gaussian: the single ellipse covers the empty middle.
      k-NN and Isolation Forest follow the real shape.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax_h = fig.add_subplot(gs[1])

    def make_data(self, p):
        rng = rng_from(p.seed)
        n = p.n_normal
        s = 0.3 + p.noise
        if p.dataset == "one cluster":
            Xn = rng.normal(0, 1, (n, 2)) * [1.4 * s, 0.7 * s] @ np.array([[0.8, 0.6], [-0.6, 0.8]])
        elif p.dataset == "two clusters":
            Xn = rng.normal(0, 0.6 * s, (n, 2)) + np.where(np.arange(n)[:, None] % 2, 1, -1) * [1.8, 1.2]
        else:
            t = rng.uniform(-2, 2, n)
            Xn = np.c_[t, 0.5 * t ** 2 - 1] + rng.normal(0, 0.3 * s, (n, 2))
        n_out = int(round(p.outlier_fraction * n))
        Xo = rng.uniform(-4, 4, (n_out, 2))
        X = np.r_[Xn, Xo]
        return {"X": X, "y_true": np.r_[np.zeros(n, int), np.ones(n_out, int)]}

    def draw_data(self, data, p):
        ax = self.ax
        ax.clear()
        X, yt = data["X"], data["y_true"]
        ax.scatter(*X[yt == 0].T, s=14, c="0.4")
        ax.scatter(*X[yt == 1].T, s=60, facecolors="none", edgecolors="red", lw=1.5)
        P.set_limits(ax, np.r_[X, [[-4, -4], [4, 4]]], 0.3)
        P.style(ax, "Data (red rings = injected faults, unknown to the algorithm)")

    def run(self, data, p):
        X, yt = data["X"], data["y_true"]
        rng = np.random.default_rng(p.seed)
        if p.method == METHODS[0]:
            det = GaussianDetector().fit(X)
        elif p.method == METHODS[1]:
            det = KNNDetector(p.k).fit(X)
        else:
            det = IsolationForest(p.n_trees, p.sample_size, rng).fit(X)
        s = det.score(X)
        thr = np.quantile(s, 1 - p.contamination)
        flagged = s > thr
        ax = self.ax
        ax.clear()
        box = np.r_[X, [[-4, -4], [4, 4]]]
        xx, yy, G = P.mesh(box, 90, 0.3)
        Z = det.score(G).reshape(xx.shape)
        ax.contourf(xx, yy, Z, 20, cmap="magma_r", alpha=0.55)
        ax.contour(xx, yy, Z, levels=[thr], colors="red", linewidths=2)
        ax.scatter(*X[~flagged].T, s=14, c="0.25", zorder=3)
        ax.scatter(*X[flagged].T, s=50, c="red", marker="x", zorder=4, label="flagged")
        ax.scatter(*X[yt == 1].T, s=90, facecolors="none", edgecolors="red", lw=1.2, zorder=4, label="true fault")
        ax.legend(fontsize=7, loc="lower right")
        P.set_limits(ax, box, 0.3)
        tp = int(np.sum(flagged & (yt == 1)))
        prec = tp / max(flagged.sum(), 1)
        rec = tp / max((yt == 1).sum(), 1)
        P.style(ax, f"{p.method}: anomaly score map")
        P.textbox(ax, f"flagged {int(flagged.sum())}\nprecision {prec:.1%}\nrecall    {rec:.1%}")
        h = self.ax_h
        h.clear()
        bins = np.linspace(s.min(), s.max(), 40)
        h.hist(s[yt == 0], bins=bins, color="0.5", alpha=0.7, label="normal")
        h.hist(s[yt == 1], bins=bins, color="red", alpha=0.7, label="injected faults")
        h.axvline(thr, color="k", ls="--", label="threshold")
        h.legend(fontsize=7)
        P.style(h, "Score distribution", "anomaly score", "count")
        self.log(f"{p.method}: threshold {thr:.3f}, precision {prec:.1%}, recall {rec:.1%}")
        yield f"{p.method}: {int(flagged.sum())} flagged, precision {prec:.1%}, recall {rec:.1%}"
