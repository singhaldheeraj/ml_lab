"""Rosenblatt's Perceptron (1958) - the ancestor of every neural network."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Bool, Float, Int
from ml_lab.core.registry import register


class Perceptron:
    def __init__(self, lr=1.0):
        self.lr = lr
        self.w, self.b = np.zeros(2), 0.0

    def predict(self, X):
        return (X @ self.w + self.b > 0).astype(int)

    def fit_steps(self, X, y01, epochs, shuffle, rng):
        """Yields (epoch, sample index, n_errors_so_far_in_epoch) after every weight UPDATE,
        and (epoch, None, n_errors) at the end of every epoch."""
        y = 2 * y01 - 1
        for ep in range(epochs):
            errors = 0
            order = rng.permutation(len(X)) if shuffle else np.arange(len(X))
            for i in order:
                if y[i] * (X[i] @ self.w + self.b) <= 0:        # misclassified
                    self.w = self.w + self.lr * y[i] * X[i]      # rotate the line toward the point
                    self.b = self.b + self.lr * y[i]
                    errors += 1
                    yield ep, i, errors
            yield ep, None, errors
            if errors == 0:
                return


@register
class PerceptronDemo(ClassificationDemo):
    name = "Perceptron"
    order = 40
    group = "Neural Networks"
    multiclass = False
    default_dataset = "linear"
    default_noise = 0.1
    default_n = 80
    params = [
        Float("learning_rate", 1.0, 0.01, 2.0, 0.01),
        Int("epochs", 30, 1, 200),
        Bool("shuffle", True),
        Bool("animate_each_update", True, help="Show every single weight update (otherwise per epoch)."),
    ]
    info = """
    ## Idea
    One artificial neuron:  output = step(w\u00b7x + b).  Learning rule - for every misclassified
    sample, nudge the weights toward it:
        w \u2190 w + \u03b7\u00b7y\u1d62\u00b7x\u1d62        b \u2190 b + \u03b7\u00b7y\u1d62          (y \u2208 {\u22121, +1})
    The arrow shows w - it is always perpendicular to the decision line.

    ## Embedded-engineer view
    This is a hardware-friendly comparator with trainable weights: only adds and a sign check.
    The convergence theorem guarantees a solution IF the data is linearly separable.

    ## The famous limitation
    Select dataset 'xor': no straight line can separate it, so the perceptron never converges.
    Minsky & Papert pointed this out in 1969 and it froze neural-net research for a decade -
    until multi-layer networks + back-propagation (see the MLP tab).

    ## Try this
    \u2022 'linear' with noise 0.1: converges in a few epochs.  Raise noise to overlap: it never settles.
    \u2022 'xor': eternal oscillation.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax_err = fig.add_subplot(gs[1])

    def run(self, data, p):
        X, y = data["X"], (data["y"] > 0).astype(int)
        d = dict(data, y=y, yt=(data["yt"] > 0).astype(int))
        per = Perceptron(p.learning_rate)
        ep_err, updates = [], 0
        for ep, i, errs in per.fit_steps(X, y, p.epochs, p.shuffle, np.random.default_rng(p.seed)):
            if i is None:
                ep_err.append(errs)
                if errs == 0:
                    self._draw(d, per, None, ep, ep_err)
                    yield f"\u2714 Converged after {ep + 1} epochs ({updates} updates): w={per.w.round(2)}, b={per.b:.2f}"
                    return
                if p.animate_each_update and not self.fast_forward:
                    continue
            else:
                updates += 1
                if not p.animate_each_update or self.fast_forward:
                    continue
            self._draw(d, per, i, ep, ep_err)
            yield f"epoch {ep + 1}: update #{updates} on sample {i}  w={per.w.round(2)}, b={per.b:.2f}"
        self.log(f"Not converged after {p.epochs} epochs - data probably not linearly separable.")
        yield f"Stopped after {p.epochs} epochs without converging (not linearly separable?)"

    def _draw(self, d, per, i, ep, ep_err):
        X = d["X"]
        if np.allclose(per.w, 0):
            self.draw_data(d, None)
        else:
            self.draw_model(d, per.predict, title=f"Perceptron - epoch {ep + 1}")
            c = -per.b * per.w / (per.w @ per.w)              # point on the line closest to origin
            self.ax.annotate("", xy=c + per.w / np.linalg.norm(per.w) * 1.5, xytext=c,
                             arrowprops=dict(arrowstyle="->", lw=2.5, color="k"), zorder=6)
            self.ax.text(*(c + per.w / np.linalg.norm(per.w) * 1.7), "w", fontsize=12, fontweight="bold")
        if i is not None:
            self.ax.scatter(*X[i], s=320, facecolors="none", edgecolors="red", lw=2.5, zorder=7)
        P.set_limits(self.ax, X)
        a, b = self.accuracy(d, per.predict)
        P.textbox(self.ax, f"w = [{per.w[0]:.2f}, {per.w[1]:.2f}]\nb = {per.b:.2f}\ntrain acc {a:.1%}")
        ea = self.ax_err
        ea.clear()
        if ep_err:
            ea.bar(np.arange(1, len(ep_err) + 1), ep_err, color=P.color(3))
        P.style(ea, "Mistakes per epoch (0 = converged)", "epoch", "# updates")
