"""Self-Organising Map (Kohonen) - a neural grid that unfolds over the data."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import ClusteringDemo
from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register


class SOM:
    def __init__(self, rows=10, cols=10, lr=0.5, sigma=3.0, rng=None):
        self.rows, self.cols, self.lr0, self.sigma0 = rows, cols, lr, sigma
        self.rng = rng or np.random.default_rng(0)
        gi, gj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        self.grid = np.stack([gi, gj], -1).astype(float)          # neuron positions on the map

    def steps(self, X, n_iter=3000):
        self.W = X.mean(0) + self.rng.normal(0, 0.1, (self.rows, self.cols, X.shape[1]))
        for t in range(n_iter):
            frac = t / n_iter
            lr = self.lr0 * (1 - frac) + 0.01                      # decaying learning rate
            sigma = self.sigma0 * (0.5 / self.sigma0) ** frac      # shrinking neighbourhood
            x = X[self.rng.integers(len(X))]
            d = ((self.W - x) ** 2).sum(-1)
            bmu = np.unravel_index(np.argmin(d), d.shape)          # best matching unit
            g2 = ((self.grid - np.array(bmu)) ** 2).sum(-1)
            h = np.exp(-g2 / (2 * sigma ** 2))
            self.W += lr * h[..., None] * (x - self.W)             # pull BMU + neighbours to x
            yield t, x, bmu, lr, sigma

    def quantization_error(self, X):
        flat = self.W.reshape(-1, X.shape[1])
        return float(np.sqrt(((X[:, None] - flat[None]) ** 2).sum(-1).min(1)).mean())


@register
class SOMDemo(ClusteringDemo):
    name = "Self-Organising Map (SOM)"
    order = 20
    group = "Neural (competitive learning)"
    default_dataset = "uniform"
    default_n = 500
    params = [
        Int("rows", 10, 2, 30),
        Int("cols", 10, 2, 30),
        Float("learning_rate", 0.5, 0.01, 1.0, 0.01),
        Float("sigma", 3.0, 0.5, 10.0, 0.1, help="Initial neighbourhood radius (in grid cells)."),
        Int("iterations", 4000, 100, 30000, 100),
        Int("draw_every", 50, 1, 1000),
    ]
    info = """
    ## Idea
    A 2-D grid of neurons, each holding a weight vector (a position in data space).  For every
    random sample: find the closest neuron (Best Matching Unit) and pull it AND its grid
    neighbours toward the sample.  Over time the neighbourhood shrinks and the elastic net
    unfolds to cover the data while keeping its topology (neighbours stay neighbours).

    ## Embedded-engineer view
    Online, sample-by-sample learning with only compare + add operations - suitable for
    on-device adaptation.  Used in robotics for sensor maps and path planning (a 1-D SOM on
    city coordinates gives a decent Travelling-Salesman tour!).

    ## Try this
    \u2022 'uniform' with 10\u00d710: the classic unfolding net.
    \u2022 rows = 1, cols = 30 on 'circles' or 'spiral': a 1-D chain threading through the data.
    \u2022 sigma 0.5 from the start: the net tangles (no global ordering).
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax_q = fig.add_subplot(gs[1])

    def run(self, data, p):
        X = data["X"]
        som = SOM(p.rows, p.cols, p.learning_rate, p.sigma, np.random.default_rng(p.seed))
        qe, qt = [], []
        for t, x, bmu, lr, sigma in som.steps(X, p.iterations):
            if t % p.draw_every and t != p.iterations - 1:
                continue
            qe.append(som.quantization_error(X))
            qt.append(t)
            ax = self.ax
            ax.clear()
            ax.scatter(X[:, 0], X[:, 1], s=8, c="0.75", zorder=1)
            W = som.W
            ax.plot(W[..., 0], W[..., 1], "-", color=P.color(0), lw=1, zorder=2)
            ax.plot(W[..., 0].T, W[..., 1].T, "-", color=P.color(0), lw=1, zorder=2)
            ax.scatter(W[..., 0], W[..., 1], s=14, c=P.color(3), zorder=3)
            ax.scatter(*x, marker="*", s=200, c="gold", edgecolors="k", zorder=5)
            ax.scatter(*W[bmu], s=90, facecolors="none", edgecolors="k", lw=2, zorder=5)
            P.set_limits(ax, X)
            P.style(ax, f"SOM {p.rows}\u00d7{p.cols}  step {t + 1}")
            P.textbox(ax, f"step {t + 1}\nlearning rate {lr:.3f}\nneighbourhood \u03c3 {sigma:.2f}")
            qa = self.ax_q
            qa.clear()
            qa.plot(qt, qe, color=P.color(4))
            P.style(qa, "Quantisation error (mean distance to nearest neuron)", "step", "error")
            yield f"step {t + 1}/{p.iterations}: lr {lr:.3f}, \u03c3 {sigma:.2f}, q-error {qe[-1]:.3f}"
