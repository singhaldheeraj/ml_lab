"""Support Vector Machine trained with (simplified) SMO, with linear / RBF / polynomial kernels."""
import warnings

import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Choice, Float, Int
from ml_lab.core.registry import register


def kernel(A, B, kind, gamma, degree):
    if kind == "linear":
        return A @ B.T
    if kind == "poly":
        return (gamma * (A @ B.T) + 1.0) ** degree
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0))                       # rbf


class SVM:
    def __init__(self, C=1.0, kind="rbf", gamma=0.5, degree=3, tol=1e-3, rng=None):
        self.C, self.kind, self.gamma, self.degree, self.tol = C, kind, gamma, degree, tol
        self.rng = rng or np.random.default_rng(0)

    def fit_steps(self, X, y01, max_iter=100):
        """Simplified SMO (Platt 1998): repeatedly pick two Lagrange multipliers and optimise
        them analytically while keeping the constraints satisfied. Yields once per sweep."""
        self.X, self.y = X, 2.0 * y01 - 1
        y, n, C = self.y, len(X), self.C
        K = kernel(X, X, self.kind, self.gamma, self.degree)
        self.alpha, self.b = np.zeros(n), 0.0
        a = self.alpha
        for it in range(max_iter):
            changed = 0
            for i in range(n):
                Ei = (a * y) @ K[:, i] + self.b - y[i]
                if (y[i] * Ei < -self.tol and a[i] < C) or (y[i] * Ei > self.tol and a[i] > 0):
                    j = int(self.rng.integers(n - 1))
                    j += j >= i
                    Ej = (a * y) @ K[:, j] + self.b - y[j]
                    ai, aj = a[i], a[j]
                    if y[i] != y[j]:
                        L, H = max(0, aj - ai), min(C, C + aj - ai)
                    else:
                        L, H = max(0, ai + aj - C), min(C, ai + aj)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if L == H or eta >= 0:
                        continue
                    a[j] = np.clip(aj - y[j] * (Ei - Ej) / eta, L, H)
                    if abs(a[j] - aj) < 1e-6:
                        continue
                    a[i] = ai + y[i] * y[j] * (aj - a[j])
                    b1 = self.b - Ei - y[i] * (a[i] - ai) * K[i, i] - y[j] * (a[j] - aj) * K[i, j]
                    b2 = self.b - Ej - y[i] * (a[i] - ai) * K[i, j] - y[j] * (a[j] - aj) * K[j, j]
                    self.b = b1 if 0 < a[i] < C else b2 if 0 < a[j] < C else (b1 + b2) / 2
                    changed += 1
            dual = a.sum() - 0.5 * (a * y) @ K @ (a * y)
            yield it, changed, dual
            if changed == 0:
                break

    def decision(self, Q):
        sv = self.alpha > 1e-8
        if not sv.any():
            return np.full(len(Q), self.b)
        return (self.alpha[sv] * self.y[sv]) @ kernel(self.X[sv], Q, self.kind, self.gamma, self.degree) + self.b

    def predict(self, Q):
        return (self.decision(Q) > 0).astype(int)


@register
class SVMDemo(ClassificationDemo):
    name = "Support Vector Machine (SVM)"
    order = 24
    multiclass = False
    default_dataset = "moons"
    default_n = 120
    params = [
        Choice("kernel", "rbf", ["linear", "rbf", "poly"],
               help="Kernel trick: implicit mapping to a higher-dimensional space."),
        Float("log10_C", 0.0, -2.0, 3.0, 0.1,
              help="C = 10^value. Small C = wide soft margin, large C = hard margin (may over-fit)."),
        Float("gamma", 0.5, 0.01, 10.0, 0.01, help="RBF width: big gamma = very local, wiggly boundary."),
        Int("degree", 3, 2, 6, help="Polynomial kernel degree."),
        Int("max_sweeps", 60, 1, 300, help="SMO passes over the data."),
        Int("draw_every", 1, 1, 20),
    ]
    info = """
    ## Idea
    Among all lines that separate the classes, pick the one with the widest "street"
    (maximum margin).  Only the points on/inside the street edges matter: the SUPPORT VECTORS
    (circled).  All other points could be deleted without changing the model.

    ## Maths
        f(x) = \u03a3 \u03b1\u1d62 y\u1d62 K(x\u1d62, x) + b          (sum only over support vectors)
        dual: maximise \u03a3\u03b1\u1d62 \u2212 \u00bd \u03a3\u03a3 \u03b1\u1d62\u03b1\u2c7c y\u1d62y\u2c7c K(x\u1d62,x\u2c7c),   0 \u2264 \u03b1\u1d62 \u2264 C
        RBF kernel: K(a,b) = exp(\u2212\u03b3 |a\u2212b|\u00b2)
    The kernel trick lets a linear method draw curved boundaries without ever computing the
    high-dimensional features.

    ## Embedded-engineer view
    The model is "a few stored reference vectors + weights" - memory scales with the number of
    support vectors, not the dataset.  Solid line: f(x)=0 (decision). Dashed: f(x)=\u00b11 (margin).

    ## Try this
    \u2022 'linear' dataset + linear kernel; lower C until the margin widens and allows errors.
    \u2022 'circles' + rbf; raise gamma to 10 -> little islands around each point (over-fit).
    \u2022 'moons' + poly degree 3.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax_obj = fig.add_subplot(gs[1])
        self.ax_nsv = self.ax_obj.twinx()

    def run(self, data, p):
        X, y = data["X"], (data["y"] > 0).astype(int)
        d = dict(data, y=y, yt=(data["yt"] > 0).astype(int))
        svm = SVM(10 ** p.log10_C, p.kernel, p.gamma, p.degree, rng=np.random.default_rng(p.seed))
        duals, nsv = [], []
        xx, yy, G = P.mesh(X, 100)
        for it, changed, dual in svm.fit_steps(X, y, p.max_sweeps):
            duals.append(dual)
            sv = svm.alpha > 1e-8
            nsv.append(int(sv.sum()))
            if it % p.draw_every and changed and it != p.max_sweeps - 1:
                continue
            ax = self.ax
            ax.clear()
            Z = svm.decision(G).reshape(xx.shape)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contourf(xx, yy, np.tanh(Z), levels=np.linspace(-1, 1, 21), cmap=P.proba_cmap(), alpha=0.55)
                ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"],
                           linewidths=[1, 2, 1])
            P.scatter_classes(ax, X, y)
            ax.scatter(X[sv, 0], X[sv, 1], s=130, facecolors="none", edgecolors="k", lw=1.4,
                       zorder=4, label="support vectors")
            ax.legend(fontsize=7, loc="lower right")
            P.set_limits(ax, X)
            a, b = self.accuracy(d, svm.predict)
            P.style(ax, f"SVM ({p.kernel}),  C={10 ** p.log10_C:.3g}")
            P.textbox(ax, f"sweep {it + 1}\nsupport vectors {nsv[-1]}\ntrain acc {a:.1%}\ntest  acc {b:.1%}")
            oa = self.ax_obj
            oa.clear()
            oa.plot(duals, color=P.color(4), label="dual objective")
            oa2 = self.ax_nsv
            oa2.clear()
            oa2.plot(nsv, color=P.color(1), ls="--", label="# support vectors")
            oa2.tick_params(labelsize=8)
            oa.legend(fontsize=7, loc="upper left")
            oa2.legend(fontsize=7, loc="lower right")
            P.style(oa, "SMO progress", "sweep", "dual objective")
            yield f"sweep {it + 1}: {changed} \u03b1-pairs changed, {nsv[-1]} SVs, train acc {a:.1%}, test acc {b:.1%}"
        self.log(f"SVM converged with {nsv[-1]} support vectors out of {len(X)} points, b = {svm.b:.3f}")
