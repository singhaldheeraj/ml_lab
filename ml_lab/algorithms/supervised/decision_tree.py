"""CART decision tree (classification + regression). Re-used by Random Forest and boosting."""
import numpy as np
from matplotlib.colors import to_rgba

from ml_lab.core import plotting as P
from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Bool, Choice, Int
from ml_lab.core.registry import register


class Node:
    __slots__ = ("feature", "thr", "left", "right", "value", "n", "impurity", "depth")

    def __init__(self, value, n, impurity, depth):
        self.feature = self.thr = self.left = self.right = None
        self.value, self.n, self.impurity, self.depth = value, n, impurity, depth

    @property
    def is_leaf(self):
        return self.left is None


def gini(p):
    return 1.0 - (p ** 2).sum(-1)


def entropy(p):
    return -(p * np.log2(np.where(p > 0, p, 1.0))).sum(-1)


class DecisionTree:
    """Greedy top-down splitting: at each node try every feature and threshold and keep the
    split that reduces impurity the most."""

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1, criterion="gini",
                 task="classification", max_features=None, n_classes=None, rng=None):
        self.max_depth, self.min_split, self.min_leaf = max_depth, min_samples_split, min_samples_leaf
        self.criterion, self.task, self.max_features = criterion, task, max_features
        self.K, self.rng = n_classes, rng or np.random.default_rng(0)
        self.crit = entropy if criterion == "entropy" else gini

    @property
    def clf(self):
        return self.task == "classification"

    def fit(self, X, y, sample_weight=None):
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if self.clf:
            y = y.astype(int)
            self.K = self.K or int(y.max()) + 1
        self.root = self._grow(X, y, w, 0)
        return self

    def _value(self, y, w):
        if self.clf:
            c = np.bincount(y, weights=w, minlength=self.K)
            return c / max(c.sum(), 1e-12)
        return float(np.average(y, weights=w))

    def _impurity(self, y, w, value):
        if self.clf:
            return float(self.crit(value))
        return float(np.average((y - value) ** 2, weights=w))

    def _grow(self, X, y, w, depth):
        value = self._value(y, w)
        node = Node(value, len(y), self._impurity(y, w, value), depth)
        if depth >= self.max_depth or len(y) < self.min_split or node.impurity <= 1e-12:
            return node
        split = self._best_split(X, y, w, node.impurity)
        if split is None or split[2] <= 1e-12:
            return node
        node.feature, node.thr, _ = split
        m = X[:, node.feature] <= node.thr
        node.left = self._grow(X[m], y[m], w[m], depth + 1)
        node.right = self._grow(X[~m], y[~m], w[~m], depth + 1)
        return node

    def _best_split(self, X, y, w, parent_impurity):
        n, d = X.shape
        W = w.sum()
        feats = np.arange(d)
        if self.max_features and self.max_features < d:
            feats = self.rng.choice(d, self.max_features, replace=False)
        best, best_cost = None, np.inf
        counts = np.arange(1, n)
        for f in feats:
            o = np.argsort(X[:, f], kind="stable")
            xs, ys, ws = X[o, f], y[o], w[o]
            if self.clf:
                oh = np.zeros((n, self.K))
                oh[np.arange(n), ys] = ws
                cl = np.cumsum(oh, 0)[:-1]
                cr = oh.sum(0) - cl
                wl, wr = cl.sum(1), cr.sum(1)
                il = self.crit(cl / np.maximum(wl, 1e-12)[:, None])
                ir = self.crit(cr / np.maximum(wr, 1e-12)[:, None])
                cost = (wl * il + wr * ir) / W
            else:
                wy = ws * ys
                cw, cs, cs2 = np.cumsum(ws)[:-1], np.cumsum(wy)[:-1], np.cumsum(wy * ys)[:-1]
                tw, ts, ts2 = ws.sum(), wy.sum(), (wy * ys).sum()
                sl = cs2 - cs ** 2 / np.maximum(cw, 1e-12)
                sr = (ts2 - cs2) - (ts - cs) ** 2 / np.maximum(tw - cw, 1e-12)
                cost = (sl + sr) / W
            valid = (xs[1:] > xs[:-1]) & (counts >= self.min_leaf) & (n - counts >= self.min_leaf)
            if not valid.any():
                continue
            cost = np.where(valid, cost, np.inf)
            i = int(np.argmin(cost))
            if cost[i] < best_cost:
                best_cost, best = cost[i], (int(f), (xs[i] + xs[i + 1]) / 2)
        if best is None:
            return None
        return best[0], best[1], parent_impurity - best_cost

    def _leaf_values(self, X):
        out = np.zeros((len(X), self.K)) if self.clf else np.zeros(len(X))
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf or len(idx) == 0:
                out[idx] = node.value
                continue
            m = X[idx, node.feature] <= node.thr
            stack += [(node.left, idx[m]), (node.right, idx[~m])]
        return out

    def predict_proba(self, X):
        return self._leaf_values(X)

    def predict(self, X):
        v = self._leaf_values(X)
        return v.argmax(1) if self.clf else v

    def nodes(self):
        out, stack = [], [self.root]
        while stack:
            nd = stack.pop()
            out.append(nd)
            if not nd.is_leaf:
                stack += [nd.right, nd.left]
        return out

    def depth(self):
        return max(nd.depth for nd in self.nodes())

    def n_leaves(self):
        return sum(nd.is_leaf for nd in self.nodes())


# ---------------------------------------------------------------- drawing helpers
def draw_splits(ax, node, bounds, lw=1.0):
    """Draw the axis-aligned cut lines of a 2-D tree."""
    if node.is_leaf:
        return
    x0, x1, y0, y1 = bounds
    if node.feature == 0:
        ax.plot([node.thr, node.thr], [y0, y1], "k-", lw=lw, alpha=0.8, zorder=2)
        draw_splits(ax, node.left, (x0, node.thr, y0, y1), lw)
        draw_splits(ax, node.right, (node.thr, x1, y0, y1), lw)
    else:
        ax.plot([x0, x1], [node.thr, node.thr], "k-", lw=lw, alpha=0.8, zorder=2)
        draw_splits(ax, node.left, (x0, x1, y0, node.thr), lw)
        draw_splits(ax, node.right, (x0, x1, node.thr, y1), lw)


def draw_tree(ax, tree, names=("x\u2081", "x\u2082"), title="Tree structure"):
    ax.clear()
    ax.axis("off")
    pos, counter = {}, [0]

    def layout(nd):
        if nd.is_leaf:
            x = counter[0]
            counter[0] += 1
        else:
            x = (layout(nd.left) + layout(nd.right)) / 2
        pos[id(nd)] = (x, -nd.depth)
        return x

    layout(tree.root)
    n_leaves = counter[0]
    nodes = tree.nodes()
    max_d = max(nd.depth for nd in nodes)
    for nd in nodes:
        if not nd.is_leaf:
            x, y = pos[id(nd)]
            for ch in (nd.left, nd.right):
                cx, cy = pos[id(ch)]
                ax.plot([x, cx], [y, cy], "-", color="0.5", lw=0.8, zorder=1)
    show_text = n_leaves <= 24
    fs = max(5, min(9, 90 / max(n_leaves, 1)))
    crit = tree.criterion if tree.clf else "mse"
    for nd in nodes:
        x, y = pos[id(nd)]
        if tree.clf:
            c = int(np.argmax(nd.value))
            fc = to_rgba(P.color(c), 0.15 + 0.65 * float(np.max(nd.value))) if nd.is_leaf else "white"
        else:
            fc = "#e8f0fe" if nd.is_leaf else "white"
        if not show_text:
            ax.scatter(x, y, s=30, c=[fc], edgecolors="0.3", zorder=3)
            continue
        if nd.is_leaf:
            lab = (f"class {int(np.argmax(nd.value))}" if tree.clf else f"{nd.value:.2f}") + f"\nn={nd.n}"
        else:
            lab = f"{names[nd.feature]} \u2264 {nd.thr:.2f}\n{crit}={nd.impurity:.2f}\nn={nd.n}"
        ax.text(x, y, lab, ha="center", va="center", fontsize=fs, zorder=3,
                bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec="0.4", lw=0.7))
    ax.set_xlim(-0.7, n_leaves - 0.3)
    ax.set_ylim(-max_d - 0.6, 0.6)
    ax.set_title(f"{title}   (depth {max_d}, {n_leaves} leaves; left branch = yes)",
                 fontsize=10, fontweight="bold")


@register
class DecisionTreeDemo(ClassificationDemo):
    name = "Decision Tree (CART)"
    order = 30
    group = "Trees & Ensembles"
    instant = True
    rerun_on_click = True
    params = [
        Choice("criterion", "gini", ["gini", "entropy"], help="Impurity measure to minimise."),
        Int("max_depth", 4, 1, 15, help="Maximum number of questions on any path."),
        Int("min_samples_split", 2, 2, 60, help="Don't split nodes smaller than this."),
        Int("min_samples_leaf", 1, 1, 40, help="Each leaf must contain at least this many samples."),
        Bool("grow_animation", True, help="Animate depth 1 -> max_depth."),
    ]
    info = """
    ## Idea
    A flowchart of yes/no questions such as "x\u2081 \u2264 0.53 ?". The tree is grown greedily:
    at each node, try every feature and threshold and keep the one that makes the two children
    the purest.

    ## Impurity
        Gini    = 1 \u2212 \u03a3 p\u2096\u00b2          (probability of mislabelling a random sample)
        Entropy = \u2212\u03a3 p\u2096 log\u2082 p\u2096       (information content in bits)
        Gain    = impurity(parent) \u2212 weighted impurity(children)

    ## Embedded-engineer view
    A trained tree IS a nested if/else block - you can literally generate C code from it.
    No floating-point multiplies, tiny memory, deterministic timing.  That's why trees and
    forests are so popular for TinyML / sensor classification.
    Boundaries are always axis-aligned (one sensor threshold at a time).

    ## Try this
    \u2022 Increase max_depth on noisy moons (noise 0.4): test accuracy peaks then falls (over-fit).
    \u2022 min_samples_leaf = 10: simpler, more robust tree (pruning).
    \u2022 'circles' needs a staircase of axis-aligned cuts. Compare to Logistic Regression + quadratic.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])
        self.ax = fig.add_subplot(gs[0])
        self.ax_tree = fig.add_subplot(gs[1])

    def run(self, data, p):
        X, y = data["X"], data["y"]
        K = self.n_classes(data)
        lo, hi = X.min(0) - 0.6, X.max(0) + 0.6
        for d in (range(1, p.max_depth + 1) if p.grow_animation else [p.max_depth]):
            t = DecisionTree(d, p.min_samples_split, p.min_samples_leaf, p.criterion,
                             n_classes=K).fit(X, y)
            self.draw_model(data, t.predict, lambda G: t.predict_proba(G)[:, 1],
                            title=f"Decision regions (max_depth={d})")
            draw_splits(self.ax, t.root, (lo[0], hi[0], lo[1], hi[1]))
            a, b = self.accuracy(data, t.predict)
            P.textbox(self.ax, f"depth {t.depth()}  leaves {t.n_leaves()}\ntrain acc {a:.1%}\ntest  acc {b:.1%}")
            draw_tree(self.ax_tree, t)
            yield f"max_depth={d}: {t.n_leaves()} leaves, train acc {a:.1%}, test acc {b:.1%}"
            if t.depth() < d:
                self.log(f"Tree stopped growing at depth {t.depth()} (pure leaves or limits reached).")
                break
