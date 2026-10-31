"""AdaBoost - a sequence of weak learners, each focusing on the previous mistakes."""
import numpy as np

from ml_lab.algorithms.supervised.decision_tree import DecisionTree
from ml_lab.core import plotting as P
from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register


class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0, base_depth=1):
        self.n, self.lr, self.depth = n_estimators, learning_rate, base_depth
        self.learners, self.alphas = [], []

    def fit_steps(self, X, y01):
        y = 2 * y01 - 1                                   # labels -> {-1, +1}
        w = np.full(len(y), 1 / len(y))
        for _ in range(self.n):
            stump = DecisionTree(self.depth, n_classes=2).fit(X, y01, sample_weight=w)
            pred = 2 * stump.predict(X) - 1
            err = np.clip(np.sum(w * (pred != y)) / w.sum(), 1e-10, 1 - 1e-10)
            if err >= 0.5:
                break
            alpha = self.lr * 0.5 * np.log((1 - err) / err)    # say of this learner
            w = w * np.exp(-alpha * y * pred)                   # boost the mistakes
            w /= w.sum()
            self.learners.append(stump)
            self.alphas.append(alpha)
            yield stump, alpha, err, w

    def decision(self, X):
        return sum(a * (2 * t.predict(X) - 1) for a, t in zip(self.alphas, self.learners))

    def proba1(self, X):
        return 1 / (1 + np.exp(-2 * self.decision(X)))

    def predict(self, X):
        return (self.decision(X) > 0).astype(int)


@register
class AdaBoostDemo(ClassificationDemo):
    name = "AdaBoost"
    order = 32
    group = "Trees & Ensembles"
    multiclass = False
    default_dataset = "circles"
    params = [
        Int("n_estimators", 60, 1, 400, help="Number of boosting rounds."),
        Float("learning_rate", 1.0, 0.05, 2.0, 0.05, help="Shrinks each learner's vote."),
        Int("base_depth", 1, 1, 4, help="1 = decision stump (a single threshold)."),
        Int("draw_every", 1, 1, 20),
    ]
    info = """
    ## Idea  (boosting)
    Train a very weak learner (a stump: ONE threshold on ONE feature).  Increase the weight of
    the samples it got wrong.  Train the next stump on the re-weighted data.  Repeat.
    Final prediction = weighted vote:  F(x) = \u03a3 \u03b1\u2098 h\u2098(x),   \u03b1\u2098 = \u00bd ln((1\u2212err)/err)

    ## Embedded-engineer view
    Like building a precise ADC from many 1-bit comparators: each comparator alone is crude,
    but a weighted sum of many becomes a smooth, accurate decision.  Boosting reduces BIAS
    (forests reduce variance).  Inference: N comparisons + a weighted sum - MCU-friendly.

    ## Panels
    Left: ensemble confidence.  Top-right: the newest stump; point size = current sample weight
    (the hard points grow!).  Bottom-right: errors vs. rounds.

    ## Try this
    \u2022 'circles' with stumps: watch a polygon emerge from straight cuts.
    \u2022 High noise: boosting starts chasing noisy points (huge weights) -> over-fitting.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_weak = fig.add_subplot(gs[0, 1])
        self.ax_err = fig.add_subplot(gs[1, 1])

    def run(self, data, p):
        X, y = data["X"], (data["y"] > 0).astype(int)
        d = dict(data, y=y, yt=(data["yt"] > 0).astype(int))
        ab = AdaBoost(p.n_estimators, p.learning_rate, p.base_depth)
        tr_e, te_e, weak_e = [], [], []
        for i, (stump, alpha, err, w) in enumerate(ab.fit_steps(X, y)):
            a, b = self.accuracy(d, ab.predict)
            tr_e.append(1 - a)
            te_e.append(1 - b)
            weak_e.append(err)
            if i % p.draw_every and i != p.n_estimators - 1:
                continue
            self.draw_model(d, ab.predict, ab.proba1, title=f"Ensemble of {i + 1} learners", res=100)
            P.textbox(self.ax, f"round {i + 1}\ntrain acc {a:.1%}\ntest  acc {b:.1%}")
            wa = self.ax_weak
            wa.clear()
            P.plot_regions(wa, stump.predict, X, 2, res=80, alpha=0.3)
            sizes = 8 + 3000 * w
            wa.scatter(X[:, 0], X[:, 1], s=np.minimum(sizes, 400), c=P.colors_for(y),
                       edgecolors="k", lw=0.4, zorder=3)
            P.set_limits(wa, X)
            P.style(wa, f"Weak learner #{i + 1}: err={err:.2f}, \u03b1={alpha:.2f}  (size = weight)")
            ea = self.ax_err
            ea.clear()
            r = np.arange(1, i + 2)
            ea.plot(r, tr_e, color=P.color(0), label="ensemble train err")
            ea.plot(r, te_e, color=P.color(3), label="ensemble test err")
            ea.plot(r, weak_e, ".", color="0.5", ms=3, label="weak learner weighted err")
            ea.legend(fontsize=7)
            P.style(ea, "Error vs. boosting rounds", "round", "error")
            yield f"round {i + 1}: weak err {err:.3f}, \u03b1 {alpha:.2f}, ensemble train acc {a:.1%}, test acc {b:.1%}"
        if not ab.learners:
            yield "Weak learner no better than chance - nothing to boost."
