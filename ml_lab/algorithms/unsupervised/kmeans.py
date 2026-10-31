"""K-Means clustering (Lloyd's algorithm) with k-means++ initialisation."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import ClusteringDemo
from ml_lab.core.params import Bool, Choice, Int
from ml_lab.core.registry import register


def sq_dist(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)


class KMeans:
    def __init__(self, k=3, init="k-means++", max_iter=50, rng=None):
        self.k, self.init, self.max_iter = k, init, max_iter
        self.rng = rng or np.random.default_rng(0)

    def init_centroids(self, X):
        n = len(X)
        if self.init == "random":
            return X[self.rng.choice(n, self.k, replace=False)].copy()
        C = [X[self.rng.integers(n)]]
        for _ in range(1, self.k):                     # pick far-away points with prob ~ D^2
            d2 = sq_dist(X, np.array(C)).min(1)
            C.append(X[self.rng.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else X[self.rng.integers(n)])
        return np.array(C)

    def steps(self, X):
        """Yields ('init'|'assign'|'update', centroids, labels, inertia)."""
        C = self.init_centroids(X)
        yield "init", C, None, None
        for _ in range(self.max_iter):
            D = sq_dist(X, C)
            labels = D.argmin(1)
            inertia = float(D[np.arange(len(X)), labels].sum())
            yield "assign", C, labels, inertia
            newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else X[self.rng.integers(len(X))]
                             for j in range(self.k)])
            shift = np.abs(newC - C).max()
            C = newC
            yield "update", C, labels, inertia
            if shift < 1e-7:
                return

    def fit(self, X):
        for _, C, labels, inertia in self.steps(X):
            pass
        self.centroids, self.labels, self.inertia = C, labels, inertia
        return self


@register
class KMeansDemo(ClusteringDemo):
    name = "K-Means"
    order = 10
    params = [
        Int("k", 3, 1, 10, help="Number of clusters you ask for."),
        Choice("init", "k-means++", ["k-means++", "random"],
               help="Random init can get stuck in bad local minima; k-means++ spreads seeds out."),
        Int("max_iter", 30, 1, 200),
        Bool("show_voronoi", True, help="Shade the region closest to each centroid."),
        Bool("elbow_curve", True, help="Also compute inertia for k = 1..10 at the end."),
    ]
    info = """
    ## Idea
    Repeat two steps until nothing changes:
      1. ASSIGN  every point to its nearest centroid.
      2. UPDATE  every centroid to the mean of its points.
    It minimises the inertia = \u03a3 |x \u2212 nearest centroid|\u00b2.  The animation shows each step
    separately and the trails show how centroids move.

    ## Embedded-engineer view
    This is vector quantisation - the same idea as building a codebook for compression, or
    finding the N typical operating points of a motor from logged (current, speed) pairs.
    After training, classification is just "nearest of k stored vectors": trivial on an MCU.

    ## Limitations
    You must choose k; it assumes round, similar-size clusters (fails on moons/circles);
    the result depends on initialisation.  Use the ELBOW curve to choose k.

    ## Try this
    \u2022 init 'random', seed 3, k 5: watch it get stuck.  Then switch to k-means++.
    \u2022 'moons' or 'circles': K-Means cuts them straight - compare with DBSCAN.
    \u2022 'anisotropic': stretched clusters confuse it - compare with GMM.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_in = fig.add_subplot(gs[0, 1])
        self.ax_elbow = fig.add_subplot(gs[1, 1])

    def run(self, data, p):
        X = data["X"]
        if p.k > len(X):
            yield "k is larger than the number of points"
            return
        km = KMeans(p.k, p.init, p.max_iter, np.random.default_rng(p.seed))
        trails, inertias, it = [], [], 0
        for phase, C, labels, inertia in km.steps(X):
            trails.append(C.copy())
            if phase == "assign":
                inertias.append(inertia)
                it += 1
            ax = self.ax
            ax.clear()
            if p.show_voronoi:
                P.plot_regions(ax, lambda G: sq_dist(G, C).argmin(1), X, p.k, res=150, alpha=0.15)
            if labels is None:
                ax.scatter(X[:, 0], X[:, 1], s=16, c="0.45")
            else:
                self.draw_labels(ax, X, labels)
            T = np.array(trails)
            for j in range(p.k):
                ax.plot(T[:, j, 0], T[:, j, 1], "-", color="k", lw=1, alpha=0.6, zorder=4)
                ax.scatter(*C[j], marker="X", s=260, c=P.color(j), edgecolors="k", lw=1.5, zorder=5)
            P.set_limits(ax, X)
            P.style(ax, f"K-Means  iteration {it}  -  {phase.upper()} step")
            if inertia is not None:
                P.textbox(ax, f"iteration {it}\ninertia {inertia:.1f}")
            ia = self.ax_in
            ia.clear()
            ia.plot(np.arange(1, len(inertias) + 1), inertias, "o-", color=P.color(3), ms=4)
            P.style(ia, "Inertia (within-cluster SSE)", "iteration", "inertia")
            yield f"iteration {it}: {phase} step" + (f", inertia {inertia:.2f}" if inertia is not None else "")

        if p.elbow_curve:
            ks = np.arange(1, min(10, len(X)) + 1)
            vals = [KMeans(k, "k-means++", 100, np.random.default_rng(p.seed)).fit(X).inertia for k in ks]
            ea = self.ax_elbow
            ea.clear()
            ea.plot(ks, vals, "o-", color=P.color(0))
            ea.axvline(p.k, color="k", ls="--", lw=1)
            P.style(ea, "Elbow method: pick k at the 'knee'", "k", "final inertia")
            self.log("Elbow inertias: " + ", ".join(f"k={k}: {v:.1f}" for k, v in zip(ks, vals)))
            yield f"Converged after {it} iterations, inertia {inertias[-1]:.2f}"
