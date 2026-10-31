"""Mean Shift - hill-climbing on a kernel density estimate to find its peaks (modes)."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import ClusteringDemo
from ml_lab.core.params import Choice, Float, Int
from ml_lab.core.registry import register


def kde(G, X, h):
    d2 = ((G[:, None] - X[None]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * h * h)).sum(1) / (len(X) * 2 * np.pi * h * h)


class MeanShift:
    def __init__(self, bandwidth=0.8, kernel="gaussian"):
        self.h, self.kernel = bandwidth, kernel

    def shift(self, Y, X):
        d2 = ((Y[:, None] - X[None]) ** 2).sum(-1)
        W = np.exp(-d2 / (2 * self.h ** 2)) if self.kernel == "gaussian" else (d2 <= self.h ** 2).astype(float)
        return (W @ X) / np.maximum(W.sum(1, keepdims=True), 1e-12)   # weighted mean of the window

    def steps(self, X, max_iter=100, tol=1e-4):
        Y = X.copy()
        for it in range(max_iter):
            Yn = self.shift(Y, X)
            moved = np.abs(Yn - Y).max()
            Y = Yn
            yield it, Y, moved
            if moved < tol:
                break
        self.modes, self.labels = self.group(Y)

    def group(self, Y):
        modes, labels = [], np.full(len(Y), -1)
        for i, y in enumerate(Y):
            for j, m in enumerate(modes):
                if np.linalg.norm(y - m) < self.h / 2:
                    labels[i] = j
                    break
            else:
                modes.append(y)
                labels[i] = len(modes) - 1
        return np.array(modes), labels


@register
class MeanShiftDemo(ClusteringDemo):
    name = "Mean Shift"
    order = 14
    default_n = 250
    max_n = 600
    params = [
        Float("bandwidth", 0.8, 0.1, 3.0, 0.05, help="Kernel radius h. The ONLY real parameter."),
        Choice("kernel", "gaussian", ["gaussian", "flat"]),
        Int("max_iter", 60, 1, 300),
    ]
    info = """
    ## Idea
    Put a small Gaussian "bump" on every data point and add them up: that's a kernel density
    estimate (right panel, 3-D).  Then let every point climb uphill: repeatedly move it to the
    weighted mean of the data around it.  Points that end on the same peak form one cluster.
    The number of clusters is discovered automatically - it depends only on the bandwidth.

    ## Embedded-engineer view
    Each iteration is a smoothing filter applied in space.  Mean shift is the core of the
    CamShift object tracker used in classic computer vision / robotics (track a colour blob
    by climbing the colour-probability image frame after frame).

    ## Try this
    \u2022 bandwidth 0.3 -> many small clusters;  2.0 -> everything merges into one.
    \u2022 'flat' kernel: points jump in discrete steps.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax3d = fig.add_subplot(gs[1], projection="3d")

    def run(self, data, p):
        X = data["X"]
        ms = MeanShift(p.bandwidth, p.kernel)
        xx, yy, G = P.mesh(X, 70)
        Z = kde(G, X, p.bandwidth).reshape(xx.shape)
        a3 = self.ax3d
        a3.clear()
        a3.plot_surface(xx, yy, Z, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
        a3.set_title(f"Kernel density estimate (h={p.bandwidth})", fontsize=10, fontweight="bold")
        a3.tick_params(labelsize=7)
        for it, Y, moved in ms.steps(X, p.max_iter):
            ax = self.ax
            ax.clear()
            ax.contourf(xx, yy, Z, 15, cmap="viridis", alpha=0.35)
            ax.scatter(X[:, 0], X[:, 1], s=8, c="0.6", zorder=2)
            ax.plot(np.c_[X[:, 0], Y[:, 0]].T, np.c_[X[:, 1], Y[:, 1]].T, "-", color="k", lw=0.3, alpha=0.4)
            ax.scatter(Y[:, 0], Y[:, 1], s=18, c="crimson", edgecolors="k", lw=0.3, zorder=4)
            P.set_limits(ax, X)
            P.style(ax, f"Mean shift iteration {it + 1}: points climbing to density peaks")
            P.textbox(ax, f"iteration {it + 1}\nmax move {moved:.4f}")
            yield f"iteration {it + 1}: max move {moved:.5f}"
        ax = self.ax
        ax.clear()
        ax.contourf(xx, yy, Z, 15, cmap="viridis", alpha=0.3)
        self.draw_labels(ax, X, ms.labels)
        for j, m in enumerate(ms.modes):
            ax.scatter(*m, marker="*", s=380, c=P.color(j), edgecolors="k", zorder=6)
            a3.scatter(m[0], m[1], kde(m[None], X, p.bandwidth)[0], c="red", s=60, depthshade=False)
        P.set_limits(ax, X)
        P.style(ax, f"Result: {len(ms.modes)} modes = {len(ms.modes)} clusters")
        self.log(f"Mean shift found {len(ms.modes)} modes with bandwidth {p.bandwidth}")
        yield f"Done: {len(ms.modes)} clusters found"
