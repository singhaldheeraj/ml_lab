"""Polynomial regression with Ridge, Lasso, or Elastic Net regularization."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import RegressionDemo
from ml_lab.core.params import Choice, Float, Int
from ml_lab.core.registry import register


class RegularizedLinearModel:
    def __init__(self, degree, regularizer, alpha, l1_ratio):
        self.degree = degree
        self.regularizer = regularizer
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _design(self, x):
        features = np.column_stack([x ** power for power in range(1, self.degree + 1)])
        return np.column_stack([np.ones(len(x)), (features - self.mean) / self.scale])

    def fit(self, x, y):
        features = np.column_stack([x ** power for power in range(1, self.degree + 1)])
        self.mean = features.mean(0)
        self.scale = features.std(0) + 1e-8
        design = np.column_stack([np.ones(len(x)), (features - self.mean) / self.scale])
        self.coef = np.zeros(design.shape[1])
        spectral_norm = np.linalg.norm(design, ord=2) ** 2 / len(x)
        ridge_strength = self.alpha if self.regularizer == "ridge" else (
            self.alpha * (1 - self.l1_ratio) if self.regularizer == "elastic net" else 0)
        rate = 1 / (spectral_norm + ridge_strength + 1e-8)
        self.history = []
        for epoch in range(1, 201):
            error = design @ self.coef - y
            gradient = design.T @ error / len(x)
            gradient[1:] += ridge_strength * self.coef[1:]
            self.coef -= rate * gradient
            if self.regularizer in ("lasso", "elastic net"):
                strength = self.alpha * (self.l1_ratio if self.regularizer == "elastic net" else 1)
                self.coef[1:] = np.sign(self.coef[1:]) * np.maximum(
                    np.abs(self.coef[1:]) - rate * strength, 0)
            if epoch % 10 == 0 or epoch == 200:
                loss = float(np.mean((design @ self.coef - y) ** 2))
                self.history.append((self.coef.copy(), loss))
        return self

    def predict(self, x):
        return self._design(np.asarray(x)) @ self.coef


@register
class RegularizedRegressionDemo(RegressionDemo):
    name = "Ridge / Lasso Regression"
    group = "Regularized regression"
    order = 25
    instant = True
    params = [Choice("regularizer", "ridge", ["ridge", "lasso", "elastic net"]),
              Int("degree", 5, 1, 12),
              Float("alpha", 0.02, 0.0, 1.0, 0.01),
              Float("l1_ratio", 0.5, 0.0, 1.0, 0.05,
                    help="Elastic Net balance: 0 is Ridge-like, 1 is Lasso-like.")]
    info = """
    ## Idea
    Fit a polynomial curve while penalizing large coefficients. Ridge shrinks weights smoothly,
    Lasso can drive weights exactly to zero, and Elastic Net blends both penalties.

    ## Try this
    Use a high degree on noisy data, then raise alpha. Compare the train and test errors to see
    how regularization trades a little fit for better generalization.
    """

    def run(self, data, p):
        model = RegularizedLinearModel(p.degree, p.regularizer, p.alpha, p.l1_ratio)
        model.fit(data["x"], data["y"])
        grid = np.linspace(-3.2, 3.2, 300)
        for index, (coefficients, loss) in enumerate(model.history):
            model.coef = coefficients
            if not self.fast_forward or index == len(model.history) - 1:
                self.draw_data(data, p, title=f"{p.regularizer.title()} regression · degree {p.degree}")
                self.ax.plot(grid, model.predict(grid), color=P.color(3), linewidth=2,
                             label="regularized fit")
                train_mse = np.mean((model.predict(data["x"]) - data["y"]) ** 2)
                test_mse = (np.mean((model.predict(data["xt"]) - data["yt"]) ** 2)
                            if len(data["xt"]) else None)
                self.ax.legend(fontsize=7, loc="upper left")
                test_label = f"{test_mse:.3f}" if test_mse is not None else "n/a"
                P.textbox(self.ax, f"epoch {(index + 1) * 10}\ntrain MSE {train_mse:.3f}\ntest MSE {test_label}")
            yield f"epoch {(index + 1) * 10}/200: train MSE {loss:.4f}"