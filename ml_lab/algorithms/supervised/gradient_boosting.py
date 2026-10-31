"""Gradient Boosting (regression): each new tree fits the residual errors of the ensemble."""
import numpy as np

from ml_lab.algorithms.supervised.decision_tree import DecisionTree
from ml_lab.core import plotting as P
from ml_lab.core.base import RegressionDemo
from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=2, subsample=1.0, rng=None):
        self.n, self.lr, self.depth, self.sub = n_estimators, learning_rate, max_depth, subsample
        self.rng = rng or np.random.default_rng(0)
        self.trees = []

    def fit_steps(self, x, y):
        X = x[:, None]
        self.f0 = float(np.mean(y))
        F = np.full(len(y), self.f0)
        for _ in range(self.n):
            r = y - F                         # negative gradient of squared loss = residual
            idx = self.rng.choice(len(y), max(2, int(self.sub * len(y))), replace=False)
            t = DecisionTree(self.depth, task="regression").fit(X[idx], r[idx])
            F = F + self.lr * t.predict(X)
            self.trees.append(t)
            yield t, r

    def predict(self, x):
        X = np.asarray(x)[:, None]
        return self.f0 + self.lr * sum((t.predict(X) for t in self.trees), np.zeros(len(X)))


@register
class GradientBoostingDemo(RegressionDemo):
    name = "Gradient Boosting (regression)"
    order = 33
    group = "Trees & Ensembles"
    default_dataset = "damped"
    default_n = 120
    params = [
        Int("n_estimators", 150, 1, 600, help="Number of small trees."),
        Float("learning_rate", 0.1, 0.01, 1.0, 0.01, help="Shrinkage: how much of each tree to add."),
        Int("max_depth", 2, 1, 6, help="Depth of each small tree (2 -> 4 steps)."),
        Float("subsample", 1.0, 0.2, 1.0, 0.05, help="< 1 = Stochastic Gradient Boosting."),
        Int("draw_every", 2, 1, 50),
    ]
    info = """
    ## Idea
    Start with a constant prediction (the mean).  Compute the residuals (what we still get
    wrong).  Fit a tiny tree to the residuals.  Add a small fraction (learning rate) of it.
    Repeat.  For squared loss, the residual is exactly the negative gradient - so this is
    gradient descent in *function space*.
        F\u2098(x) = F\u2098\u208b\u2081(x) + \u03b7 \u00b7 tree\u2098(x)

    ## Embedded-engineer view
    It's iterative error correction - like successive-approximation, or adding correction terms
    to a lookup-table calibration: each stage only fixes what previous stages left over.
    XGBoost / LightGBM (the winners of most tabular-data competitions) are this algorithm.

    ## Panels
    Left: ensemble prediction.  Top-right: residuals and the tree fitted to them this round.
    Bottom-right: train / test MSE vs. rounds (watch test error start to rise = over-fitting).

    ## Try this
    \u2022 learning_rate 1.0 vs 0.05: fast & jumpy vs. slow & smooth.
    \u2022 max_depth 1 (stumps) needs many more rounds.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_res = fig.add_subplot(gs[0, 1])
        self.ax_err = fig.add_subplot(gs[1, 1])

    def run(self, data, p):
        x, y, xt, yt = data["x"], data["y"], data["xt"], data["yt"]
        gb = GradientBoostingRegressor(p.n_estimators, p.learning_rate, p.max_depth, p.subsample,
                                       np.random.default_rng(p.seed))
        xs = np.linspace(-3.3, 3.3, 400)
        Fs = Ftr = Fte = None
        tr_mse, te_mse = [], []
        for i, (t, r) in enumerate(gb.fit_steps(x, y)):
            if Fs is None:
                Fs, Ftr, Fte = np.full(len(xs), gb.f0), np.full(len(x), gb.f0), np.full(len(xt), gb.f0)
            Fs = Fs + gb.lr * t.predict(xs[:, None])
            Ftr = Ftr + gb.lr * t.predict(x[:, None])
            Fte = Fte + gb.lr * t.predict(xt[:, None]) if len(xt) else Fte
            tr_mse.append(np.mean((Ftr - y) ** 2))
            te_mse.append(np.mean((Fte - yt) ** 2) if len(xt) else np.nan)
            if i % p.draw_every and i != p.n_estimators - 1:
                continue
            self.draw_data(data, p)
            self.ax.plot(xs, Fs, color=P.color(3), lw=2.5, label=f"ensemble ({i + 1} trees)")
            self.ax.legend(fontsize=7, loc="upper left")
            P.textbox(self.ax, f"round {i + 1}\ntrain MSE {tr_mse[-1]:.3f}\ntest  MSE {te_mse[-1]:.3f}",
                      "lower right")
            ra = self.ax_res
            ra.clear()
            ra.scatter(x, r, s=12, c=P.color(4), label="residual y \u2212 F")
            ra.plot(xs, t.predict(xs[:, None]), color="k", lw=2, label="tree fitted to residual")
            ra.axhline(0, color="0.6", lw=0.7)
            ra.legend(fontsize=7)
            P.style(ra, f"Round {i + 1}: what's still wrong", "x", "residual")
            ea = self.ax_err
            ea.clear()
            ea.plot(tr_mse, color=P.color(0), label="train MSE")
            if len(xt):
                ea.plot(te_mse, color=P.color(3), label="test MSE")
            ea.set_yscale("log")
            ea.legend(fontsize=7)
            P.style(ea, "MSE vs. boosting rounds", "round", "MSE")
            yield f"round {i + 1}/{p.n_estimators}: train MSE {tr_mse[-1]:.4f}, test MSE {te_mse[-1]:.4f}"
