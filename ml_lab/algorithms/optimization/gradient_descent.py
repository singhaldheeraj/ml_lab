"""Gradient descent on an anisotropic convex quadratic."""
import numpy as np

from ml_lab.core.base import AlgorithmDemo
from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register


class QuadraticObjective:
    def __init__(self, condition):
        self.curvature = np.array([1., float(condition)])

    def value(self, point):
        return float(0.5 * np.sum(self.curvature * np.square(point)))

    def gradient(self, point):
        return self.curvature * point

    def steps(self, start, learning_rate, iterations):
        point = np.array(start, dtype=float)
        yield point.copy(), self.value(point)
        for _ in range(iterations):
            if np.linalg.norm(self.gradient(point)) < 1e-6:
                break
            point -= learning_rate * self.gradient(point)
            value = self.value(point)
            yield point.copy(), value
            if not np.isfinite(value) or value > 1e12:
                break


@register
class GradientDescentDemo(AlgorithmDemo):
    name = "Gradient Descent"
    category = "Optimization"
    group = "Gradient-based training"
    order = 10
    params = [Float("learning_rate", 0.15, 0.001, 1.0, 0.001),
              Int("iterations", 60, 1, 300), Float("condition", 5, 1, 30, 1),
              Float("start_x", -3, -5, 5, 0.1), Float("start_y", 2, -5, 5, 0.1)]
    info = """## Gradient descent
    Minimize f(x,y) = (x^2 + condition*y^2)/2 with gradient (x, condition*y).
    Update point <- point - learning_rate*gradient. The unique minimum is (0,0).
    Curvature controls how narrow the bowl is. A stable constant learning rate
    satisfies 0 < learning_rate < 2/condition. Larger steps can oscillate or
    diverge; an iteration limit is not a convergence guarantee.
    Like an error-driven integrator: high curvature requires lower loop gain.
    The contour plot shows the path; the second panel records objective values.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        self.ax, self.loss_ax = fig.subplots(1, 2)

    def draw_data(self, data, p):
        self._draw(p, [np.array([p.start_x, p.start_y])], [])

    def _draw(self, p, path, losses):
        self.ax.clear()
        self.loss_ax.clear()
        extent = max(4., abs(p.start_x) + 1, abs(p.start_y) + 1)
        coordinates = np.linspace(-extent, extent, 100)
        horizontal, vertical = np.meshgrid(coordinates, coordinates)
        values = (horizontal ** 2 + p.condition * vertical ** 2) / 2
        self.ax.contour(horizontal, vertical, values, 20, cmap="viridis")
        points = np.array(path)
        self.ax.plot(points[:, 0], points[:, 1], "o-", color="#d64c4c", ms=3)
        self.ax.scatter(0, 0, marker="*", s=120, c="#16826c", label="Minimum")
        self.ax.set(xlim=(-extent, extent), ylim=(-extent, extent),
                    xlabel="x", ylabel="y", title="Quadratic objective", aspect="equal")
        self.ax.legend(fontsize=8)
        self.loss_ax.plot(np.arange(len(losses)), np.maximum(losses, 1e-16), color="#487eb0")
        self.loss_ax.set(xlabel="Update", ylabel="Objective", yscale="log", title="Descent history")

    def run(self, data, p):
        model = QuadraticObjective(p.condition)
        path, losses = [], []
        for index, (point, loss) in enumerate(model.steps([p.start_x, p.start_y], p.learning_rate, p.iterations)):
            path.append(point)
            losses.append(loss)
            diverged = not np.isfinite(loss) or loss > 1e12
            self.last_result = {"point": point.copy(), "loss": loss, "diverged": diverged}
            if not self.fast_forward or diverged:
                self._draw(p, path, losses)
            yield f"Update {index}: objective={loss:.6g}; gradient norm={np.linalg.norm(model.gradient(point)):.6g}"
            if diverged:
                yield "Diverged: reduce the learning rate"
                return
        self._draw(p, path, losses)
        converged = np.linalg.norm(model.gradient(path[-1])) < 1e-6
        yield "Converged to gradient tolerance" if converged else "Iteration budget reached; convergence not guaranteed"