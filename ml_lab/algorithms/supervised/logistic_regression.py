"""Logistic / softmax regression trained with gradient descent."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Choice, Float, Int
from ml_lab.core.registry import register


def feature_map(X, kind):
    x1, x2 = X[:, 0], X[:, 1]
    cols = [x1, x2]
    if kind in ("quadratic", "cubic"):
        cols += [x1 ** 2, x2 ** 2, x1 * x2]
    if kind == "cubic":
        cols += [x1 ** 3, x2 ** 3, x1 ** 2 * x2, x1 * x2 ** 2]
    return np.column_stack(cols)


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


class SoftmaxRegression:
    """K-class logistic regression. For K=2 this is the classic sigmoid logistic regression."""

    def __init__(self, n_classes, features="linear"):
        self.K, self.features = n_classes, features

    def phi(self, X):
        return (feature_map(X, self.features) - self.mu) / self.sd

    def proba(self, X):
        return softmax(self.phi(X) @ self.W + self.b)

    def predict(self, X):
        return self.proba(X).argmax(1)

    def fit_steps(self, X, y, lr, epochs, l2):
        F = feature_map(X, self.features)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-9      # feature scaling
        F = (F - self.mu) / self.sd
        n, d = F.shape
        self.W, self.b = np.zeros((d, self.K)), np.zeros(self.K)
        Y = np.eye(self.K)[y]
        for ep in range(epochs):
            Pr = softmax(F @ self.W + self.b)
            G = Pr - Y                                      # dLoss/dlogits - beautifully simple
            self.W -= lr * (F.T @ G / n + l2 * self.W)
            self.b -= lr * G.mean(0)
            loss = -np.mean(np.log(Pr[np.arange(n), y] + 1e-12)) + 0.5 * l2 * (self.W ** 2).sum()
            yield ep, loss


@register
class LogisticRegressionDemo(ClassificationDemo):
    name = "Logistic / Softmax Regression"
    order = 20
    default_dataset = "linear"
    params = [
        Choice("features", "linear", ["linear", "quadratic", "cubic"],
               help="Feature engineering: add x\u00b2, x\u2081x\u2082 \u2026 so a linear model can draw curves."),
        Float("learning_rate", 0.5, 0.01, 5.0, 0.01),
        Int("epochs", 300, 10, 3000, 10),
        Float("l2", 0.0, 0.0, 0.1, 0.001, help="Weight decay (L2 regularisation)."),
        Int("draw_every", 5, 1, 100),
    ]
    info = """
    ## Idea
    Compute a score z = w\u00b7x + b and squash it into a probability with the sigmoid
    \u03c3(z) = 1/(1+e^\u2212z).  Class 1 if p > 0.5.  For K classes use softmax over K scores.
    The decision boundary w\u00b7x + b = 0 is a straight line (hyperplane) - unless you add features.

    ## Maths
        p   = softmax(W\u1d40x + b)
        Loss = cross-entropy = \u2212mean(log p[true class])
        \u2207W  = X\u1d40 (P \u2212 Y) / n          <- same form as linear regression!

    ## Embedded-engineer view
    Think of a comparator with a soft edge: instead of a hard threshold you get a confidence
    (0..1). The colour intensity in the plot is exactly that confidence. Feature engineering
    (x\u00b2, x\u2081x\u2082) is like adding non-linear pre-processing before a linear stage.

    ## Try this
    \u2022 'circles' with linear features fails (~50%). Switch to 'quadratic' -> perfect circle boundary.
    \u2022 'xor' needs the x\u2081x\u2082 term -> quadratic works.
    \u2022 'blobs' with 3-5 classes shows softmax regions.
    \u2022 Large learning rate + noise=0: weights grow forever (separable data) -> use l2.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_loss = fig.add_subplot(gs[0, 1])
        self.ax_acc = self.ax_loss.twinx()
        self.ax_w = fig.add_subplot(gs[1, 1])

    def run(self, data, p):
        X, y = data["X"], data["y"]
        K = self.n_classes(data)
        m = SoftmaxRegression(K, p.features)
        losses, accs = [], []
        for ep, loss in m.fit_steps(X, y, p.learning_rate, p.epochs, p.l2):
            losses.append(loss)
            accs.append(np.mean(m.predict(X) == y))
            if ep % p.draw_every == 0 or ep == p.epochs - 1:
                self.draw_model(data, m.predict, lambda G: m.proba(G)[:, 1],
                                title=f"{K}-class softmax regression, {p.features} features")
                tr, te = self.accuracy(data, m.predict)
                P.textbox(self.ax, f"epoch {ep + 1}\ntrain acc {tr:.1%}\ntest  acc {te:.1%}")
                a = self.ax_loss
                a.clear()
                a.plot(losses, color=P.color(3), label="cross-entropy loss")
                a2 = self.ax_acc
                a2.clear()
                a2.plot(accs, color=P.color(2), label="train accuracy")
                a2.set_ylim(0, 1.05)
                a2.tick_params(labelsize=8)
                P.style(a, "Training curves", "epoch", "loss")
                a.legend(fontsize=7, loc="center right")
                w = self.ax_w
                w.clear()
                im = w.imshow(m.W.T, cmap="coolwarm", aspect="auto",
                              vmin=-np.abs(m.W).max() - 1e-9, vmax=np.abs(m.W).max() + 1e-9)
                names = ["x\u2081", "x\u2082", "x\u2081\u00b2", "x\u2082\u00b2", "x\u2081x\u2082",
                         "x\u2081\u00b3", "x\u2082\u00b3", "x\u2081\u00b2x\u2082", "x\u2081x\u2082\u00b2"]
                w.set_xticks(range(m.W.shape[0]), names[:m.W.shape[0]], fontsize=8)
                w.set_yticks(range(K), [f"class {k}" for k in range(K)], fontsize=8)
                for (i, j), v in np.ndenumerate(m.W.T):
                    w.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
                w.set_title("Weight matrix W (one row per class)", fontsize=10, fontweight="bold")
                yield f"epoch {ep + 1}/{p.epochs}  loss={loss:.4f}  train acc={tr:.1%}  test acc={te:.1%}"
        self.log(f"Final loss {losses[-1]:.4f}, accuracy {accs[-1]:.1%}")
