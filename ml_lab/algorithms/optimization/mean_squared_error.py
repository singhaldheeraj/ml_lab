"""Visual breakdown of squared residuals and their mean."""
import numpy as np

from ml_lab.algorithms.supervised.linear_regression import LinearRegressionGD
from ml_lab.core.base import RegressionDemo
from ml_lab.core.params import Float
from ml_lab.core.registry import register


@register
class MeanSquaredErrorDemo(RegressionDemo):
    name = "Mean Squared Error (MSE)"
    category = "Optimization"
    group = "Loss functions"
    order = 5
    default_dataset = "linear"
    default_n = 20
    params = [Float("slope", 1., -5, 5, 0.1), Float("intercept", 0., -5, 5, 0.1)]
    info = """## Mean squared error
    For fixed predictions y_hat = slope*x + intercept, compute each residual
    e = y_hat-y, square it, then average: MSE = sum(e^2)/n. Squaring prevents
    positive and negative errors from canceling and penalizes large errors.
    MSE has squared target units; RMSE = sqrt(MSE) has the original units.
    This is a loss function, not an optimizer. Change slope/intercept to compare
    candidate models; Run reveals each training residual's contribution.
    Like averaging squared calibration errors from a set of sensor captures.
    Held-out samples are displayed but do not contribute to training MSE.
    """

    @staticmethod
    def squared_errors(samples, targets, slope, intercept):
        return np.square(LinearRegressionGD(slope, intercept).predict(samples) - targets)

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        self.ax, self.error_ax = fig.subplots(1, 2)

    def draw_data(self, data, p):
        self._draw(data, p, 0)

    def _draw(self, data, p, count):
        RegressionDemo.draw_data(self, data, p)
        model = LinearRegressionGD(p.slope, p.intercept)
        coordinates = np.linspace(*self.ax.get_xlim(), 100)
        self.ax.plot(coordinates, model.predict(coordinates), color="#16826c", label="Candidate model")
        self.ax.vlines(data["x"][:count], data["y"][:count], model.predict(data["x"][:count]),
                       color="#d64c4c", lw=1.5)
        self.ax.legend(fontsize=8)
        self.error_ax.clear()
        errors = self.squared_errors(data["x"], data["y"], p.slope, p.intercept)
        self.error_ax.bar(np.arange(len(errors)), errors,
                          color=["#d64c4c" if index < count else "#c8d6df" for index in range(len(errors))])
        self.error_ax.axhline(float(np.mean(errors)), color="#16826c", linestyle="--", label="Full MSE")
        self.error_ax.set(xlabel="Training sample", ylabel="Squared residual", title="MSE contributions")
        self.error_ax.legend(fontsize=8)

    def run(self, data, p):
        errors = self.squared_errors(data["x"], data["y"], p.slope, p.intercept)
        for count in range(1, len(errors) + 1):
            if not self.fast_forward or count == len(errors):
                self._draw(data, p, count)
            yield f"Sample {count}/{len(errors)}: squared error={errors[count - 1]:.5f}; running mean={np.mean(errors[:count]):.5f}"
        self.last_result = float(np.mean(errors))
        yield f"MSE={self.last_result:.6f}; RMSE={np.sqrt(self.last_result):.6f}; samples={len(errors)}"