"""Random Forest = many de-correlated trees voting together (bagging)."""
import numpy as np

from ml_lab.algorithms.supervised.decision_tree import DecisionTree, draw_splits
from ml_lab.core import plotting as P
from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Bool, Float, Int
from ml_lab.core.registry import register


class RandomForest:
    def __init__(self, n_trees=30, max_depth=8, max_features=1, bootstrap=True, sample_fraction=1.0,
                 min_samples_leaf=1, n_classes=None, rng=None):
        self.n_trees, self.max_depth, self.max_features = n_trees, max_depth, max_features
        self.bootstrap, self.frac, self.min_leaf = bootstrap, sample_fraction, min_samples_leaf
        self.K, self.rng = n_classes, rng or np.random.default_rng(0)
        self.trees = []

    def fit_steps(self, X, y):
        """Generator: yields each new tree as it is added to the forest."""
        n = len(X)
        self.K = self.K or int(y.max()) + 1
        m = max(2, int(self.frac * n))
        for _ in range(self.n_trees):
            idx = self.rng.choice(n, m, replace=self.bootstrap)   # bagging: random resample
            t = DecisionTree(self.max_depth, 2, self.min_leaf, "gini", max_features=self.max_features,
                             n_classes=self.K, rng=self.rng).fit(X[idx], y[idx])
            self.trees.append(t)
            yield t

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


@register
class RandomForestDemo(ClassificationDemo):
    name = "Random Forest"
    order = 31
    group = "Trees & Ensembles"
    default_noise = 0.35
    params = [
        Int("n_trees", 40, 1, 300, help="Number of trees that vote."),
        Int("max_depth", 8, 1, 20),
        Int("max_features", 1, 1, 2, help="Features considered per split (randomness de-correlates trees)."),
        Bool("bootstrap", True, help="Sample training data with replacement for each tree."),
        Float("sample_fraction", 1.0, 0.1, 1.0, 0.05, help="Size of each tree's sample (fraction of n)."),
        Int("min_samples_leaf", 1, 1, 30),
        Int("draw_every", 1, 1, 20),
    ]
    info = """
    ## Idea  (bagging = bootstrap aggregating)
    One deep tree over-fits: it has high variance. Train many trees, each on a random resample
    of the data and with a random subset of features at every split, then average their votes.
    The individual errors are partly independent, so averaging cancels them out.

    ## Embedded-engineer view
    Exactly like averaging N noisy sensor readings: noise std falls ~ 1/\u221aN if the noise is
    uncorrelated.  max_features / bootstrap are the tricks that de-correlate the "sensors".
    A forest is still just if/else code - TinyML tools export forests to C directly.

    ## Panels
    Left: the forest's averaged probability.  Top-right: ONLY the newest tree (notice how noisy
    one tree is).  Bottom-right: test accuracy of the forest vs. a single tree as trees are added.

    ## Try this
    \u2022 Compare 1 tree vs 100 trees on noisy moons.
    \u2022 bootstrap off + max_features 2  -> all trees identical -> no gain from averaging!
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_tree = fig.add_subplot(gs[0, 1])
        self.ax_acc = fig.add_subplot(gs[1, 1])

    def run(self, data, p):
        X, y = data["X"], data["y"]
        K = self.n_classes(data)
        rf = RandomForest(p.n_trees, p.max_depth, p.max_features, p.bootstrap, p.sample_fraction,
                          p.min_samples_leaf, K, np.random.default_rng(p.seed))
        acc_forest, acc_tree = [], []
        lo, hi = X.min(0) - 0.6, X.max(0) + 0.6
        for i, t in enumerate(rf.fit_steps(X, y)):
            acc_forest.append(np.mean(rf.predict(data["Xt"]) == data["yt"]))
            acc_tree.append(np.mean(t.predict(data["Xt"]) == data["yt"]))
            if i % p.draw_every and i != p.n_trees - 1:
                continue
            self.draw_model(data, rf.predict, lambda G: rf.predict_proba(G)[:, 1],
                            title=f"Forest of {i + 1} trees", res=100)
            a, b = self.accuracy(data, rf.predict)
            P.textbox(self.ax, f"trees {i + 1}\ntrain acc {a:.1%}\ntest  acc {b:.1%}")
            self.draw_model(data, t.predict, lambda G: t.predict_proba(G)[:, 1],
                            title=f"Tree #{i + 1} alone (test acc {acc_tree[-1]:.1%})", ax=self.ax_tree, res=80)
            draw_splits(self.ax_tree, t.root, (lo[0], hi[0], lo[1], hi[1]), lw=0.6)
            ac = self.ax_acc
            ac.clear()
            ac.plot(np.arange(1, i + 2), acc_forest, color=P.color(2), lw=2, label="forest")
            ac.plot(np.arange(1, i + 2), acc_tree, ".", color="0.5", ms=4, label="individual tree")
            ac.legend(fontsize=7, loc="lower right")
            P.style(ac, "Test accuracy vs. number of trees", "trees", "accuracy")
            yield f"{i + 1}/{p.n_trees} trees: train acc {a:.1%}, test acc {b:.1%}"
        self.log(f"Forest test acc {acc_forest[-1]:.1%}; mean single-tree test acc {np.mean(acc_tree):.1%}")
