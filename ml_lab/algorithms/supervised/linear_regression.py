"""Linear regression trained by gradient descent - watch the ball roll down the loss bowl."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import RegressionDemo
from ml_lab.core.params import Bool, Choice, Float, Int
from ml_lab.core.registry import register


class LinearRegressionGD:
    """y_hat = w*x + b, trained to minimise the mean squared error (MSE)."""

    def __init__(self, w=0.0, b=0.0):
        self.w, self.b = float(w), float(b)

    def predict(self, x):
        return self.w * x + self.b

    def mse(self, x, y):
        return float(np.mean((y - self.predict(x)) ** 2))

    def gradient(self, x, y):
        e = self.predict(x) - y                   # error signal
        return 2 * np.mean(e * x), 2 * np.mean(e)  # dJ/dw, dJ/db

    def fit_steps(self, x, y, lr, epochs, method, batch_size, rng):
        """Generator - yields the epoch number after every weight update."""
        n = len(x)
        for ep in range(epochs):
            if method == "Batch GD":
                batches = [np.arange(n)]
            else:
                bs = 1 if method == "Stochastic GD" else batch_size
                idx = rng.permutation(n)
                batches = [idx[i:i + bs] for i in range(0, n, bs)]
            for bi in batches:
                gw, gb = self.gradient(x[bi], y[bi])
                self.w -= lr * gw
                self.b -= lr * gb
                yield ep

    def fit_closed_form(self, x, y):
        A = np.c_[x, np.ones_like(x)]
        self.w, self.b = np.linalg.lstsq(A, y, rcond=None)[0]
        return self


@register
class LinearRegressionDemo(RegressionDemo):
    name = "Linear Regression"
    order = 10
    default_dataset = "linear"
    default_n = 40
    default_noise = 0.3
    params = [
        Choice("method", "Batch GD", ["Batch GD", "Mini-batch GD", "Stochastic GD", "Normal equation"],
               help="How much data is used for each weight update."),
        Float("learning_rate", 0.05, 0.001, 1.2, 0.001,
              help="Step size = loop gain. Try 0.3+ to see oscillation and divergence."),
        Int("epochs", 60, 1, 500, help="Passes over the full training set."),
        Int("batch_size", 8, 1, 64, help="Only for mini-batch GD."),
        Float("init_w", -2.0, -5.0, 5.0, 0.1, help="Initial slope guess."),
        Float("init_b", 3.0, -5.0, 5.0, 0.1, help="Initial intercept guess."),
        Int("draw_every", 1, 1, 50, help="Draw one frame every N updates."),
        Bool("show_residuals", True, help="Vertical lines = errors the loss squares and averages."),
    ]
    info = """
    ## Idea
    Fit a straight line  y = w\u00b7x + b  through noisy points. "Best" = the line with the smallest
    mean squared error (MSE) between prediction and measurement.

    ## Maths
        J(w,b)  = 1/n \u03a3 (w\u00b7x\u1d62 + b \u2212 y\u1d62)\u00b2
        \u2202J/\u2202w = 2/n \u03a3 e\u1d62\u00b7x\u1d62      \u2202J/\u2202b = 2/n \u03a3 e\u1d62
        w \u2190 w \u2212 \u03b7\u00b7\u2202J/\u2202w       (\u03b7 = learning rate)

    ## Embedded-engineer view
    This is exactly a 2-point sensor calibration (gain + offset) done with many noisy points.
    Gradient descent is an integrator driven by the error: the learning rate is the loop gain.
    Too small -> slow settling; too big -> overshoot / oscillation / divergence, like an unstable PID.
    The Normal equation is the closed-form least-squares solution (what you'd do offline in MATLAB).

    ## Panels
    Left: data, model line and residuals.  Top-right: loss vs update (log scale).
    Bottom-right: the loss surface J(w,b) seen from above - the orange path is the optimiser.

    ## Try this
    \u2022 learning_rate 0.05 -> 0.5 -> 1.0: watch the path zig-zag and blow up.
    \u2022 Stochastic GD: noisy path, but each update is cheap (good for streaming data on an MCU).
    \u2022 Dataset 'quadratic': a line can't fit a curve -> high bias (see Polynomial Regression).
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_loss = fig.add_subplot(gs[0, 1])
        self.ax_surf = fig.add_subplot(gs[1, 1])

    def draw_data(self, data, p):
        super().draw_data(data, p)
        self.ax_loss.clear()
        self.ax_surf.clear()
        P.style(self.ax_loss, "Loss (MSE) vs. updates", "update #", "MSE")
        P.style(self.ax_surf, "Loss surface J(w, b)", "w (slope)", "b (intercept)")

    def run(self, data, p):
        x, y = data["x"], data["y"]
        model = LinearRegressionGD(p.init_w, p.init_b)
        opt = LinearRegressionGD().fit_closed_form(x, y)
        self.draw_data(data, p)

        W = np.linspace(min(opt.w, p.init_w) - 2, max(opt.w, p.init_w) + 2, 80)
        B = np.linspace(min(opt.b, p.init_b) - 2, max(opt.b, p.init_b) + 2, 80)
        WW, BB = np.meshgrid(W, B)
        J = np.mean((y - (WW[..., None] * x + BB[..., None])) ** 2, axis=-1)
        s = self.ax_surf
        s.contourf(WW, BB, np.log10(J + 1e-12), 30, cmap="viridis")
        s.contour(WW, BB, np.log10(J + 1e-12), 12, colors="w", linewidths=0.3)
        s.plot(opt.w, opt.b, "r*", ms=14, label="optimum")
        path_line, = s.plot([], [], "o-", color="orange", ms=3, lw=1.2, label="optimiser path")
        s.set_xlim(W[0], W[-1])
        s.set_ylim(B[0], B[-1])
        s.legend(fontsize=7, loc="upper right")
        loss_line, = self.ax_loss.plot([], [], color=P.color(0))
        self.ax_loss.set_yscale("log")

        path, losses = [(model.w, model.b)], [model.mse(x, y)]
        art = (path_line, loss_line)
        if p.method == "Normal equation":
            model.fit_closed_form(x, y)
            path.append((model.w, model.b))
            losses.append(model.mse(x, y))
            self._frame(data, p, model, path, losses, art)
            yield f"Solved in one shot: w={model.w:.3f}, b={model.b:.3f}, MSE={losses[-1]:.4f}"
            return

        rng = np.random.default_rng(p.seed)
        for k, ep in enumerate(model.fit_steps(x, y, p.learning_rate, p.epochs, p.method,
                                               p.batch_size, rng)):
            path.append((model.w, model.b))
            losses.append(model.mse(x, y))
            if not np.isfinite(losses[-1]) or losses[-1] > 1e8:
                self._frame(data, p, model, path[:-1], losses[:-1], art)
                self.log(f"Diverged at update {k}: learning rate {p.learning_rate} is too high.")
                yield "\u26a0 DIVERGED - learning rate too high (like too much loop gain)"
                return
            if k % p.draw_every == 0:
                self._frame(data, p, model, path, losses, art)
                yield (f"epoch {ep + 1}/{p.epochs}   w={model.w:.3f}  b={model.b:.3f}  "
                       f"MSE={losses[-1]:.4f}   (optimum MSE={opt.mse(x, y):.4f})")
        self._frame(data, p, model, path, losses, art)
        test = f", test MSE={model.mse(data['xt'], data['yt']):.4f}" if len(data["xt"]) else ""
        self.log(f"{p.method}: w={model.w:.4f} b={model.b:.4f} train MSE={losses[-1]:.4f}{test}")
        yield f"Done: w={model.w:.3f}, b={model.b:.3f}, train MSE={losses[-1]:.4f}{test}"

    def _frame(self, data, p, model, path, losses, art):
        path_line, loss_line = art
        ax = self.ax
        RegressionDemo.draw_data(self, data, p)
        xs = np.array([-3.5, 3.5])
        ax.plot(xs, model.predict(xs), color=P.color(3), lw=2.5, label="model")
        if p.show_residuals:
            ax.vlines(data["x"], data["y"], model.predict(data["x"]), colors=P.color(3),
                      alpha=0.45, lw=1)
        P.textbox(ax, f"y = {model.w:.3f}\u00b7x + {model.b:.3f}\nMSE = {losses[-1]:.4f}",
                  loc="lower right")
        pa = np.array(path)
        path_line.set_data(pa[:, 0], pa[:, 1])
        loss_line.set_data(np.arange(len(losses)), np.array(losses) + 1e-12)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
