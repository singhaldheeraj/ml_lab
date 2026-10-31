"""t-SNE - non-linear embedding of high-dimensional data into 2-D for visualisation."""
import numpy as np
from matplotlib import colormaps

from ml_lab.core import plotting as P
from ml_lab.core.base import AlgorithmDemo
from ml_lab.core.datasets import rng_from
from ml_lab.core.params import Choice, Float, Int
from ml_lab.core.registry import register


def sq_dists(X):
    s = (X ** 2).sum(1)
    return np.maximum(s[:, None] + s[None] - 2 * X @ X.T, 0)


def joint_probabilities(X, perplexity, tol=1e-5):
    """Binary-search a Gaussian width per point so its neighbour distribution has the
    requested perplexity (~ effective number of neighbours)."""
    D = sq_dists(X)
    n = len(X)
    P_ = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        d = np.delete(D[i], i)
        beta, lo, hi = 1.0, 0.0, np.inf
        for _ in range(60):
            p = np.exp(-(d - d.min()) * beta)
            s = p.sum()
            H = np.log(s) + beta * np.sum((d - d.min()) * p) / s     # entropy of row
            if abs(H - target) < tol:
                break
            if H > target:
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = (beta + lo) / 2
        P_[i, np.arange(n) != i] = p / s
    P_ = (P_ + P_.T) / (2 * n)
    return np.maximum(P_, 1e-12)


class TSNE:
    def __init__(self, perplexity=30, learning_rate=200, exaggeration=12, rng=None):
        self.perplexity, self.lr, self.exag = perplexity, learning_rate, exaggeration
        self.rng = rng or np.random.default_rng(0)

    def steps(self, X, n_iter=500):
        n = len(X)
        Pm = joint_probabilities(X, min(self.perplexity, (n - 1) / 3))
        Y = self.rng.normal(0, 1e-4, (n, 2))
        vel, gains = np.zeros_like(Y), np.ones_like(Y)
        for it in range(n_iter):
            ex = self.exag if it < 100 else 1.0                      # early exaggeration
            num = 1 / (1 + sq_dists(Y))                             # Student-t kernel (heavy tails)
            np.fill_diagonal(num, 0)
            Q = np.maximum(num / num.sum(), 1e-12)
            PQ = (ex * Pm - Q) * num
            grad = 4 * (PQ.sum(1)[:, None] * Y - PQ @ Y)
            gains = np.where(np.sign(grad) != np.sign(vel), gains + 0.2, gains * 0.8).clip(0.01)
            mom = 0.5 if it < 250 else 0.8
            vel = mom * vel - self.lr * gains * grad
            Y = Y + vel
            Y -= Y.mean(0)
            kl = float(np.sum(Pm * np.log(Pm / Q)))
            yield it, Y, kl


@register
class TSNEDemo(AlgorithmDemo):
    name = "t-SNE"
    category = "Unsupervised"
    group = "Dimensionality Reduction"
    order = 31

    def data_params(self):
        return [Choice("dataset", "blobs 10-D", ["blobs 10-D", "swiss roll 3-D", "two rings 3-D"], group="Data"),
                Int("n_samples", 300, 50, 700, 10, group="Data", help="t-SNE is O(n\u00b2): keep it small."),
                Int("n_clusters", 5, 2, 10, group="Data", help="For 'blobs 10-D'."),
                Float("noise", 0.2, 0.0, 1.0, 0.01, group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    params = [
        Float("perplexity", 30.0, 2.0, 100.0, 1.0, help="~ number of neighbours each point cares about."),
        Float("learning_rate", 200.0, 10.0, 1000.0, 10.0),
        Int("iterations", 500, 50, 2000, 10),
        Float("early_exaggeration", 12.0, 1.0, 30.0, 0.5, help="Pulls clusters apart in the first 100 iterations."),
        Int("draw_every", 10, 1, 100),
    ]
    info = """
    ## Idea
    Make a 2-D map in which points that were neighbours in high-D stay neighbours.
      1. In high-D, turn distances into "neighbour probabilities" with a Gaussian (width set
         per point by the perplexity).
      2. In 2-D, do the same with a heavy-tailed Student-t kernel.
      3. Move the 2-D points by gradient descent to make the two distributions match
         (minimise the KL divergence).

    ## Important
    t-SNE is for VISUALISATION only.  Distances between far-apart clusters and cluster sizes in
    the map mean nothing.  Different seeds/perplexities give different pictures.

    ## Embedded-engineer view
    Use it on your PC to *look* at high-dimensional feature vectors (e.g. 64 FFT bins of
    vibration data) and see whether fault classes separate at all before you design a
    classifier for the MCU.  Left panel: PCA view of the same data (linear) for comparison.

    ## Try this
    \u2022 'swiss roll': PCA squashes the roll; t-SNE unrolls the neighbourhood structure.
    \u2022 perplexity 5 vs 50 on the blobs.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2)
        self.ax_orig = fig.add_subplot(gs[0], projection="3d")
        self.ax = fig.add_subplot(gs[1])

    def make_data(self, p):
        rng = rng_from(p.seed)
        n = p.n_samples
        if p.dataset == "blobs 10-D":
            C = rng.normal(0, 4, (p.n_clusters, 10))
            lab = np.arange(n) % p.n_clusters
            X = C[lab] + rng.normal(0, 0.5 + 2 * p.noise, (n, 10))
            return {"X": X, "c": lab, "discrete": True}
        if p.dataset == "swiss roll 3-D":
            t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
            X = np.c_[t * np.cos(t), 20 * rng.uniform(0, 1, n), t * np.sin(t)] + rng.normal(0, p.noise, (n, 3))
            return {"X": X, "c": t, "discrete": False}
        t = rng.uniform(0, 2 * np.pi, n)
        lab = np.arange(n) % 2
        X = np.where(lab[:, None] == 0, np.c_[np.cos(t), np.sin(t), np.zeros(n)],
                     np.c_[1 + np.cos(t), np.zeros(n), np.sin(t)]) + rng.normal(0, 0.1 * p.noise, (n, 3))
        return {"X": X, "c": lab, "discrete": True}

    def _colors(self, data):
        if data["discrete"]:
            return P.colors_for(data["c"])
        c = data["c"]
        return colormaps["Spectral"]((c - c.min()) / (np.ptp(c) + 1e-12))

    def draw_data(self, data, p):
        X = data["X"]
        Xc = X - X.mean(0)
        V = np.linalg.svd(Xc, full_matrices=False)[2][:3]
        Z = Xc @ V.T if X.shape[1] > 3 else X
        a = self.ax_orig
        a.clear()
        a.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=8, c=self._colors(data))
        a.set_title(f"Original data ({X.shape[1]}-D" + (", top-3 PCs shown)" if X.shape[1] > 3 else ")"),
                    fontsize=10, fontweight="bold")
        a.tick_params(labelsize=7)
        self.ax.clear()
        Y = Xc @ V[:2].T
        self.ax.scatter(Y[:, 0], Y[:, 1], s=10, c=self._colors(data))
        P.style(self.ax, "Linear PCA projection to 2-D (press Run for t-SNE)", "PC1", "PC2")

    def run(self, data, p):
        self.draw_data(data, p)
        yield "Computing neighbour probabilities (perplexity search)..."
        ts = TSNE(p.perplexity, p.learning_rate, p.early_exaggeration, np.random.default_rng(p.seed))
        cols = self._colors(data)
        for it, Y, kl in ts.steps(data["X"], p.iterations):
            if it % p.draw_every and it != p.iterations - 1:
                continue
            ax = self.ax
            ax.clear()
            ax.scatter(Y[:, 0], Y[:, 1], s=12, c=cols)
            phase = "early exaggeration" if it < 100 else "fine tuning"
            P.style(ax, f"t-SNE embedding - iteration {it + 1} ({phase})", "t-SNE 1", "t-SNE 2")
            P.textbox(ax, f"KL divergence {kl:.3f}")
            yield f"iteration {it + 1}/{p.iterations}: KL = {kl:.4f}"
