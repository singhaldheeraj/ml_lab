"""Gaussian generative classifiers: Naive Bayes, LDA and QDA."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Bool, Choice, Float
from ml_lab.core.registry import register

MODELS = ["Gaussian Naive Bayes", "LDA (shared covariance)", "QDA (full covariance)"]


class GaussianClassifier:
    """Fit one Gaussian per class, then use Bayes' rule: p(c|x) ~ p(x|c) p(c)."""

    def __init__(self, model=MODELS[0], smoothing=1e-9):
        self.model, self.smoothing = model, smoothing

    def fit(self, X, y, n_classes=None):
        K = n_classes or int(y.max()) + 1
        d = X.shape[1]
        self.priors = np.array([np.mean(y == c) for c in range(K)])
        self.means = np.array([X[y == c].mean(0) if np.any(y == c) else np.zeros(d) for c in range(K)])
        eps = self.smoothing * X.var(0).max() + 1e-9
        covs = []
        for c in range(K):
            Xc = X[y == c]
            C = np.cov(Xc.T, bias=True) if len(Xc) > 1 else np.eye(d)
            covs.append(np.atleast_2d(C))
        covs = np.array(covs)
        if self.model.startswith("Gaussian Naive"):
            covs = np.array([np.diag(np.diag(C)) for C in covs])    # "naive": features independent
        elif self.model.startswith("LDA"):
            shared = np.tensordot(self.priors, covs, 1)
            covs = np.array([shared] * K)
        self.covs = covs + eps * np.eye(d)
        return self

    def log_joint(self, X):
        out = []
        for mu, C, pi in zip(self.means, self.covs, self.priors):
            inv = np.linalg.inv(C)
            diff = X - mu
            maha = np.einsum("ij,jk,ik->i", diff, inv, diff)
            out.append(-0.5 * (maha + np.log(np.linalg.det(C)) + len(mu) * np.log(2 * np.pi))
                       + np.log(pi + 1e-12))
        return np.array(out).T

    def proba(self, X):
        L = self.log_joint(X)
        L -= L.max(1, keepdims=True)
        E = np.exp(L)
        return E / E.sum(1, keepdims=True)

    def predict(self, X):
        return self.log_joint(X).argmax(1)


@register
class NaiveBayesDemo(ClassificationDemo):
    name = "Naive Bayes / LDA / QDA"
    order = 22
    instant = True
    rerun_on_click = True
    default_dataset = "blobs"
    default_noise = 0.4
    params = [
        Choice("model", MODELS[0], MODELS,
               help="NB: diagonal covariance per class. LDA: one shared covariance (linear boundary). "
                    "QDA: full covariance per class (quadratic boundary)."),
        Float("smoothing", 0.0, 0.0, 1.0, 0.01, help="Variance smoothing added to the diagonal."),
        Bool("show_ellipses", True, help="1\u03c3 and 2\u03c3 contours of each class Gaussian."),
        Bool("show_probability", True, help="Binary: colour = P(class 1)."),
    ]
    info = """
    ## Idea  (generative model)
    Learn how each class *generates* data: fit a Gaussian (mean + covariance) per class.
    Then classify with Bayes' rule:
        P(class | x)  \u221d  p(x | class) \u00b7 P(class)

    ## The three flavours
        Naive Bayes : features assumed independent  -> axis-aligned ellipses
        LDA         : all classes share one covariance -> straight-line boundaries
        QDA         : each class own covariance      -> curved (quadratic) boundaries

    ## Embedded-engineer view
    Each class is described by just a mean vector and a (co)variance - a handful of numbers.
    Training is a single pass computing running mean/variance (Welford's algorithm works on an
    MCU!).  Inference is a few multiply-adds per class: extremely cheap and very robust.

    ## Panels
    Left: decision regions + class Gaussians.  Right: the 1-D class-conditional densities of
    each feature - exactly what Naive Bayes multiplies together.

    ## Try this
    \u2022 Data 'blobs' -> anisotropic look: switch NB vs LDA vs QDA.
    \u2022 'circles': QDA nails it (inner class has small variance), LDA cannot.
    \u2022 'xor' fails for all - a single Gaussian per class is too simple.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_x1 = fig.add_subplot(gs[0, 1])
        self.ax_x2 = fig.add_subplot(gs[1, 1])

    def run(self, data, p):
        X, y = data["X"], data["y"]
        K = self.n_classes(data)
        m = GaussianClassifier(p.model, p.smoothing).fit(X, y, K)
        self.draw_model(data, m.predict, (lambda G: m.proba(G)[:, 1]) if p.show_probability else None,
                        title=p.model)
        if p.show_ellipses:
            for c in range(K):
                P.draw_ellipse(self.ax, m.means[c], m.covs[c], P.color(c))
                self.ax.scatter(*m.means[c], marker="X", s=140, c=P.color(c), edgecolors="k", zorder=6)
        a, b = self.accuracy(data, m.predict)
        P.textbox(self.ax, f"train acc {a:.1%}\ntest  acc {b:.1%}")

        for j, ax in enumerate((self.ax_x1, self.ax_x2)):
            ax.clear()
            lo, hi = X[:, j].min() - 1, X[:, j].max() + 1
            xs = np.linspace(lo, hi, 300)
            for c in range(K):
                mu, var = m.means[c, j], m.covs[c, j, j]
                pdf = np.exp(-(xs - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var) * m.priors[c]
                ax.hist(X[y == c, j], bins=25, range=(lo, hi), density=False, alpha=0.25,
                        color=P.color(c), weights=np.full(np.sum(y == c), 1 / (len(X) * (hi - lo) / 25)))
                ax.plot(xs, pdf, color=P.color(c), lw=2, label=f"p(x|c{c})P(c{c})")
            ax.legend(fontsize=6)
            sub = "\u2081\u2082"[j]
            P.style(ax, f"Feature x{sub}: class densities", f"x{sub}", "density")
        for c in range(K):
            self.log(f"class {c}: prior={m.priors[c]:.2f} mean={np.round(m.means[c], 2)} "
                     f"cov={np.round(m.covs[c], 2).tolist()}")
        yield f"{p.model}: train acc {a:.1%}, test acc {b:.1%}"
