"""Polynomial regression with Ridge / Lasso regularisation - the bias/variance trade-off."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import RegressionDemo
from ml_lab.core.params import Bool, Choice, Float, Int
from ml_lab.core.registry import register


class PolynomialRegression:
    def __init__(self, degree, regularization="None", lam=0.0):
        self.degree, self.reg, self.lam = degree, regularization, lam

    def features(self, x):
        # Scale x to about [-1, 1] first, otherwise x**20 explodes (think fixed-point overflow).
        return np.vander(np.asarray(x) / 3.0, self.degree + 1, increasing=True)

    def fit(self, x, y):
        Phi = self.features(x)
        n, d = Phi.shape
        if self.reg == "Lasso (L1)":
            self.w = self._lasso(Phi, y, self.lam)
        elif self.reg == "Ridge (L2)":
            D = np.eye(d)
            D[0, 0] = 0                        # never penalise the bias term
            self.w = np.linalg.solve(Phi.T @ Phi + self.lam * n * D, Phi.T @ y)
        else:
            self.w = np.linalg.lstsq(Phi, y, rcond=None)[0]
        return self

    @staticmethod
    def _lasso(Phi, y, lam, iters=300, tol=1e-6):
        """Coordinate descent: minimise 1/(2n)||y - Phi w||^2 + lam*||w||_1."""
        n, d = Phi.shape
        w = np.zeros(d)
        z = (Phi ** 2).sum(0)
        r = y.copy()
        for _ in range(iters):
            max_step = 0.0
            for j in range(d):
                rho = Phi[:, j] @ r + z[j] * w[j]
                new = rho / z[j] if j == 0 else np.sign(rho) * max(abs(rho) - lam * n, 0) / z[j]
                if new != w[j]:
                    r -= Phi[:, j] * (new - w[j])
                    max_step = max(max_step, abs(new - w[j]))
                    w[j] = new
            if max_step < tol:
                break
        return w

    def predict(self, x):
        return self.features(x) @ self.w


def rmse(m, x, y):
    return float(np.sqrt(np.mean((m.predict(x) - y) ** 2))) if len(x) else np.nan


@register
class PolynomialRegressionDemo(RegressionDemo):
    name = "Polynomial Regression + Ridge/Lasso"
    order = 11
    instant = True
    rerun_on_click = True
    params = [
        Int("degree", 3, 1, 20, help="Model complexity. 1 = straight line."),
        Choice("regularization", "None", ["None", "Ridge (L2)", "Lasso (L1)"],
               help="Penalise large weights. Lasso drives some weights exactly to zero."),
        Float("log10_lambda", -3.0, -8.0, 1.0, 0.1, help="Regularisation strength \u03bb = 10^value."),
        Bool("animate_degree", True, help="Animate degree 1 -> chosen degree."),
        Int("sweep_max_degree", 15, 2, 20, help="Range of the train/test error curve."),
    ]
    info = """
    ## Idea
    Still *linear* regression - but on features [1, x, x\u00b2, \u2026, x\u1d48]. The model is linear in
    its weights, so the same least-squares maths applies.

    ## Bias vs. variance
    Low degree -> under-fits (high bias): train AND test error high.
    High degree -> over-fits (high variance): train error tiny, test error explodes; the curve
    wiggles through the noise.  Bottom-right plot shows this classic U-shaped test curve.

    ## Regularisation
        Ridge : J = MSE + \u03bb\u00b7\u03a3w\u2c7c\u00b2      (shrinks all weights, closed form)
        Lasso : J = MSE + \u03bb\u00b7\u03a3|w\u2c7c|      (sets some weights exactly 0 = feature selection)

    ## Embedded-engineer view
    Degree = filter order.  A 20th-order fit to 40 noisy samples is like a filter with so many
    taps that it reproduces the noise.  Regularisation is like adding damping / limiting gain.
    Also note the coefficient plot: without regularisation the weights become huge - in
    fixed-point arithmetic that would overflow immediately!

    ## Try this
    \u2022 degree 15, regularization None -> wild curve.  Then Ridge with log10_lambda = -3.
    \u2022 Lasso with degree 15: look how many coefficient bars become zero.
    \u2022 Reduce n_samples to 15 - over-fitting gets much worse with less data.
    \u2022 Click to add outliers and watch high-degree fits chase them.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_coef = fig.add_subplot(gs[0, 1])
        self.ax_err = fig.add_subplot(gs[1, 1])

    def run(self, data, p):
        x, y, xt, yt = data["x"], data["y"], data["xt"], data["yt"]
        lam = 10 ** p.log10_lambda
        degs = np.arange(1, p.sweep_max_degree + 1)
        tr_err, te_err = [], []
        for d in degs:
            m = PolynomialRegression(d, p.regularization, lam).fit(x, y)
            tr_err.append(rmse(m, x, y))
            te_err.append(rmse(m, xt, yt))

        for d in (range(1, p.degree + 1) if p.animate_degree else [p.degree]):
            m = PolynomialRegression(d, p.regularization, lam).fit(x, y)
            self.draw_data(data, p)
            xs = np.linspace(-3.3, 3.3, 400)
            self.ax.plot(xs, m.predict(xs), color=P.color(3), lw=2.5, label=f"degree {d}")
            self.ax.legend(fontsize=7, loc="upper left")
            a, b = rmse(m, x, y), rmse(m, xt, yt)
            P.textbox(self.ax, f"degree {d}\ntrain RMSE {a:.3f}\ntest  RMSE {b:.3f}", "lower right")

            c = self.ax_coef
            c.clear()
            c.bar(np.arange(d + 1), m.w, color=[P.color(0) if v >= 0 else P.color(3) for v in m.w])
            c.set_yscale("symlog", linthresh=0.1)
            c.axhline(0, color="k", lw=0.6)
            nz = int(np.sum(np.abs(m.w) > 1e-8))
            P.style(c, f"Weights w\u2080..w{d}   (non-zero: {nz}, max |w| = {np.abs(m.w).max():.3g})",
                    "power of x", "weight (symlog)")

            e = self.ax_err
            e.clear()
            e.plot(degs, tr_err, "o-", ms=3, color=P.color(0), label="train RMSE")
            if len(xt):
                e.plot(degs, te_err, "o-", ms=3, color=P.color(3), label="test RMSE")
            e.axvline(d, color="k", ls="--", lw=1)
            e.set_yscale("log")
            e.legend(fontsize=7)
            P.style(e, "Error vs. model complexity", "polynomial degree", "RMSE")
            yield f"degree {d}: train RMSE {a:.3f}, test RMSE {b:.3f}, {p.regularization} \u03bb={lam:.1e}"
        best = degs[int(np.nanargmin(te_err))] if len(xt) else None
        if best is not None:
            self.log(f"Best test RMSE at degree {best} ({p.regularization}, \u03bb={lam:.1e})")
