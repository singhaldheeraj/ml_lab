"""Agglomerative hierarchical clustering + dendrogram (single / complete / average / Ward)."""
import numpy as np
from matplotlib.collections import LineCollection

from ml_lab.core import plotting as P
from ml_lab.core.base import ClusteringDemo
from ml_lab.core.params import Choice, Int
from ml_lab.core.registry import register


class Agglomerative:
    """Bottom-up: start with every point as its own cluster, repeatedly merge the closest pair.
    Cluster-to-cluster distances are updated with the Lance-Williams formula."""

    def __init__(self, linkage="ward"):
        self.linkage = linkage

    def fit_steps(self, X):
        n = len(X)
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        if self.linkage == "ward":
            D = D ** 2
        np.fill_diagonal(D, np.inf)
        size = np.ones(n)
        cid = np.arange(n)                  # id of the cluster living in each matrix slot
        slot_of_point = np.arange(n)
        self.merges = []                    # (id_a, id_b, height, size) like scipy's linkage
        for s in range(n - 1):
            i, j = divmod(int(np.argmin(D)), n)
            i, j = min(i, j), max(i, j)
            h = D[i, j]
            ni, nj, nk = size[i], size[j], size
            if self.linkage == "single":
                new = np.minimum(D[i], D[j])
            elif self.linkage == "complete":
                new = np.maximum(D[i], D[j])
            elif self.linkage == "average":
                new = (ni * D[i] + nj * D[j]) / (ni + nj)
            else:
                new = ((ni + nk) * D[i] + (nj + nk) * D[j] - nk * h) / (ni + nj + nk)
            new[[i, j]] = np.inf
            D[i, :] = D[:, i] = new
            D[j, :] = D[:, j] = np.inf
            self.merges.append((cid[i], cid[j], float(np.sqrt(h) if self.linkage == "ward" else h), ni + nj))
            cid[i] = n + s
            size[i] = ni + nj
            size[j] = 0
            slot_of_point[slot_of_point == j] = i
            yield s, slot_of_point.copy(), size.copy(), (i, j)


def draw_dendrogram(ax, merges, n, n_clusters):
    children = {n + s: (a, b) for s, (a, b, h, c) in enumerate(merges)}
    height = {n + s: h for s, (a, b, h, c) in enumerate(merges)}
    root = n + len(merges) - 1
    merged_away = set()
    for a, b, _, _ in merges[:n - n_clusters]:
        merged_away |= {a, b}
    alive = [c for c in list(range(n)) + list(range(n, n + n - n_clusters)) if c not in merged_away]
    color_of = {c: P.color(k) for k, c in enumerate(alive)}
    segs, cols, counter = [], [], [0]

    def place(node, col):
        col = color_of.get(node, col)
        if node < n:
            x = counter[0]
            counter[0] += 1
            return x, 0.0
        a, b = children[node]
        (xa, ha), (xb, hb) = place(a, col), place(b, col)
        h = height[node]
        c = col or "0.25"
        segs.extend([[(xa, ha), (xa, h)], [(xb, hb), (xb, h)], [(xa, h), (xb, h)]])
        cols.extend([c, c, c])
        return (xa + xb) / 2, h

    place(root, None)
    ax.clear()
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=1))
    hmax = height[root]
    ax.set_xlim(-1, n)
    ax.set_ylim(0, hmax * 1.05)
    if 1 < n_clusters < n:
        cut = (merges[n - n_clusters][2] + merges[n - n_clusters - 1][2]) / 2
        ax.axhline(cut, color="r", ls="--", lw=1.2)
        ax.text(n * 0.98, cut, f" cut \u2192 {n_clusters} clusters", color="r", ha="right", va="bottom", fontsize=8)
    ax.set_xticks([])
    P.style(ax, "Dendrogram (height = merge distance)", "points", "distance", grid=False)


@register
class HierarchicalDemo(ClusteringDemo):
    name = "Hierarchical (Agglomerative)"
    order = 11
    default_n = 120
    max_n = 400
    params = [
        Choice("linkage", "ward", ["single", "complete", "average", "ward"],
               help="How the distance between two clusters is defined."),
        Int("n_clusters", 3, 1, 10, help="Where to cut the dendrogram."),
        Int("merges_per_frame", 3, 1, 50),
    ]
    info = """
    ## Idea
    Start with N clusters (every point alone).  Merge the two closest clusters.  Repeat until
    one cluster is left.  The full merge history is the DENDROGRAM - cut it at any height to get
    any number of clusters, no need to fix k beforehand.

    ## Linkage = how to measure distance between clusters
        single   : closest pair of points      (finds snakes/chains, sensitive to noise bridges)
        complete : farthest pair of points     (compact clusters)
        average  : mean of all pair distances
        ward     : increase in within-cluster variance (similar to K-Means objective)

    ## Embedded-engineer view
    Think of grouping similar fault signatures into a taxonomy: the tree tells you which faults
    are near-duplicates and which are fundamentally different.  O(N\u00b2) memory - fine for
    offline analysis of a few hundred events, not for on-device use.

    ## Try this
    \u2022 'moons' with single linkage -> perfect; with ward -> wrong.
    \u2022 Look for the largest vertical gap in the dendrogram: that's the natural number of clusters.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_dend = fig.add_subplot(gs[0, 1])
        self.ax_h = fig.add_subplot(gs[1, 1])

    def run(self, data, p):
        X = data["X"]
        n = len(X)
        k = min(p.n_clusters, n)
        model = Agglomerative(p.linkage)
        history = []
        for s, slots, size, (i, j) in model.fit_steps(X):
            if s < n - k:
                history.append((slots, size, (i, j)))
        heights = [m[2] for m in model.merges]
        for s, (slots, size, (i, j)) in enumerate(history):
            if s % p.merges_per_frame and s != len(history) - 1:
                continue
            ax = self.ax
            ax.clear()
            multi = size[slots] > 1
            uniq = {sl: c for c, sl in enumerate(np.unique(slots[multi]))}
            cols = [P.color(uniq[sl]) if m else "0.7" for sl, m in zip(slots, multi)]
            ax.scatter(X[:, 0], X[:, 1], s=18, c=cols, zorder=3)
            members = X[slots == i]
            ax.scatter(members[:, 0], members[:, 1], s=70, facecolors="none", edgecolors="k", zorder=4)
            P.set_limits(ax, X)
            n_left = n - s - 1
            P.style(ax, f"{p.linkage} linkage: {n_left} clusters left (grey = still a single point)")
            P.textbox(ax, f"merge #{s + 1}\nheight {heights[s]:.3f}")
            ha = self.ax_h
            ha.clear()
            ha.plot(np.arange(1, n), heights, color="0.8")
            ha.plot(np.arange(1, s + 2), heights[:s + 1], color=P.color(3))
            P.style(ha, "Merge distance per step (jumps = natural clusters)", "merge #", "distance")
            yield f"merge {s + 1}/{n - k}: {n_left} clusters left, merge distance {heights[s]:.3f}"
        draw_dendrogram(self.ax_dend, model.merges, n, k)
        slots = history[-1][0] if history else np.arange(n)
        labels = np.unique(slots, return_inverse=True)[1]
        self.ax.clear()
        self.draw_labels(self.ax, X, labels)
        P.set_limits(self.ax, X)
        P.style(self.ax, f"Result: {k} clusters ({p.linkage} linkage)")
        yield f"Done: {k} clusters with {p.linkage} linkage"
