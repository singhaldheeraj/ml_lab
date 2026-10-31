"""DBSCAN - density-based clustering that finds arbitrary shapes and flags noise."""
from collections import deque

import numpy as np
from matplotlib.patches import Circle

from ml_lab.core import plotting as P
from ml_lab.core.base import ClusteringDemo
from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register

UNVISITED, NOISE = -2, -1


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps, self.min_samples = eps, min_samples

    def steps(self, X):
        """Yields (labels, current_point, core_mask) while clusters are being expanded."""
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        neigh = [np.flatnonzero(row <= self.eps) for row in D]
        core = np.array([len(nb) >= self.min_samples for nb in neigh])
        labels = np.full(len(X), UNVISITED)
        cluster = -1
        for i in range(len(X)):
            if labels[i] != UNVISITED:
                continue
            if not core[i]:
                labels[i] = NOISE                     # may later be re-labelled as border point
                continue
            cluster += 1
            labels[i] = cluster
            queue = deque([i])
            while queue:                              # breadth-first "flood fill"
                q = queue.popleft()
                for nb in neigh[q]:
                    if labels[nb] == NOISE:
                        labels[nb] = cluster          # border point
                    elif labels[nb] == UNVISITED:
                        labels[nb] = cluster
                        if core[nb]:
                            queue.append(nb)
                yield labels, q, core
        self.labels, self.core = labels, core
        yield labels, None, core


@register
class DBSCANDemo(ClusteringDemo):
    name = "DBSCAN"
    order = 12
    default_dataset = "moons"
    default_noise = 0.15
    params = [
        Float("eps", 0.45, 0.05, 2.0, 0.01, help="Neighbourhood radius."),
        Int("min_samples", 5, 1, 30, help="Neighbours (incl. itself) needed to be a CORE point."),
        Int("draw_every", 4, 1, 100, help="Points expanded per animation frame."),
    ]
    info = """
    ## Idea  (Density-Based Spatial Clustering of Applications with Noise)
    A point with at least min_samples neighbours within radius eps is a CORE point.
    Clusters grow like a flood-fill from core point to core point.  Non-core points reached by
    the flood are BORDER points.  Everything else is NOISE (black \u00d7).

    ## Why it's great
    No need to choose k; finds any shape (moons, rings, spirals); explicitly labels outliers.

    ## Embedded-engineer view
    The flood-fill is exactly connected-component labelling (like blob detection on an image or
    a LIDAR point cloud).  In robotics, DBSCAN is used to segment LIDAR scans into obstacles.

    ## Choosing eps: the k-distance plot (right)
    Sort every point's distance to its min_samples-th neighbour.  The "knee" of this curve is a
    good eps: below it = inside dense regions, above it = noise.

    ## Try this
    \u2022 'moons' / 'circles': perfect where K-Means fails.
    \u2022 'varied' density blobs: one eps can't fit all densities (DBSCAN's weak spot).
    \u2022 eps too small -> everything noise;  too big -> one giant cluster.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax_k = fig.add_subplot(gs[1])

    def run(self, data, p):
        X = data["X"]
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        k = min(p.min_samples, len(X)) - 1
        kd = np.sort(np.sort(D, 1)[:, k])
        ka = self.ax_k
        ka.clear()
        ka.plot(kd, color=P.color(0))
        ka.axhline(p.eps, color="r", ls="--", label=f"eps = {p.eps}")
        ka.legend(fontsize=8)
        P.style(ka, f"k-distance plot (k = {p.min_samples})", "points sorted by distance", "distance to k-th neighbour")

        model = DBSCAN(p.eps, p.min_samples)
        for step, (labels, q, core) in enumerate(model.steps(X)):
            if q is not None and step % p.draw_every:
                continue
            ax = self.ax
            ax.clear()
            unv = labels == UNVISITED
            noise = labels == NOISE
            cl = labels >= 0
            ax.scatter(X[unv, 0], X[unv, 1], s=10, c="0.75", zorder=2)
            ax.scatter(X[noise, 0], X[noise, 1], s=25, c="k", marker="x", zorder=3)
            ax.scatter(X[cl & core, 0], X[cl & core, 1], s=26, c=P.colors_for(labels[cl & core]), zorder=3)
            ax.scatter(X[cl & ~core, 0], X[cl & ~core, 1], s=12, c=P.colors_for(labels[cl & ~core]),
                       edgecolors="k", lw=0.4, zorder=3)
            if q is not None:
                ax.add_patch(Circle(X[q], p.eps, fill=False, color="k", lw=1.5, zorder=5))
                ax.scatter(*X[q], s=60, c="k", zorder=6)
            P.set_limits(ax, X)
            ax.set_aspect("equal", adjustable="box")
            n_cl = int(labels.max()) + 1 if cl.any() else 0
            P.style(ax, f"DBSCAN  eps={p.eps}, min_samples={p.min_samples}")
            txt = f"clusters {n_cl}\ncore {int((cl & core).sum())}\nborder {int((cl & ~core).sum())}\nnoise {int(noise.sum())}"
            P.textbox(ax, txt)
            if q is None:
                self.log(txt.replace("\n", ", "))
                yield f"Done: {n_cl} clusters, {int(noise.sum())} noise points"
            else:
                yield f"expanding cluster {labels[q]} ..."
