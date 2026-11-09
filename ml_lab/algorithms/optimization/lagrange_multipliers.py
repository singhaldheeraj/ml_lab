"""Lagrange multipliers for a quadratic objective and a linear equality."""
import numpy as np

from ml_lab.core.base import AlgorithmDemo
from ml_lab.core.params import Choice, Float, Int
from ml_lab.core.registry import register


class LagrangeProjection:
    @staticmethod
    def solve(target, normal, offset):
        target, normal = np.asarray(target, dtype=float), np.asarray(normal, dtype=float)
        magnitude = float(normal @ normal)
        if magnitude == 0:
            raise ValueError("Constraint normal must be nonzero")
        multiplier = 2 * float(normal @ target - offset) / magnitude
        return target - multiplier * normal / 2, multiplier


@register
class LagrangeMultipliersDemo(AlgorithmDemo):
    name = "Lagrange Multiplier Optimization"
    category = "Optimization"
    group = "Constrained optimization"
    order = 50
    params = [Float("target_x", 2, -5, 5, 0.1), Float("target_y", 2, -5, 5, 0.1),
              Choice("constraint", "x + y = c", ("x + y = c", "x - y = c", "2*x + y = c")),
              Float("offset", 1, -4, 4, 0.1), Int("frames", 20, 1, 100)]
    info = """## Lagrange multipliers
    Minimize f(z) = ||z-target||^2 subject to normal.z = c.
    L(z,lambda) = f(z) + lambda*(normal.z-c).
    Stationarity: 2*(z-target) + lambda*normal = 0.
    Feasibility: normal.z = c. Together these give
    lambda = 2*(normal.target-c)/(normal.normal), z = target-lambda*normal/2.
    The strictly convex quadratic has a unique global constrained minimum.
    At that point the objective gradient is parallel to the constraint normal.
    Like choosing the closest calibration setting while satisfying a fixed budget.
    The animation constructs the analytic projection from the unconstrained
    target onto the constraint line; intermediate points need not be feasible.
    It is not an iterative optimizer or a general nonlinear constraint solver.
    """

    @staticmethod
    def normal(p):
        return np.array({"x + y = c": [1., 1.], "x - y = c": [1., -1.],
                         "2*x + y = c": [2., 1.]}[p.constraint])

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        self.ax, self.history_ax = fig.subplots(1, 2)

    def draw_data(self, data, p):
        target = np.array([p.target_x, p.target_y])
        optimum, multiplier = LagrangeProjection.solve(target, self.normal(p), p.offset)
        self._draw(p, target, optimum, [target], multiplier)

    def _draw(self, p, target, optimum, path, multiplier):
        self.ax.clear()
        self.history_ax.clear()
        normal = self.normal(p)
        extent = max(5., float(np.max(np.abs([target, optimum]))) + 1)
        coordinates = np.linspace(-extent, extent, 100)
        horizontal, vertical = np.meshgrid(coordinates, coordinates)
        values = (horizontal - target[0]) ** 2 + (vertical - target[1]) ** 2
        self.ax.contour(horizontal, vertical, values, 18, cmap="viridis")
        self.ax.plot(coordinates, (p.offset - normal[0] * coordinates) / normal[1],
                     color="#487eb0", label=p.constraint)
        points = np.array(path)
        self.ax.plot(points[:, 0], points[:, 1], "o-", color="#d64c4c", ms=3)
        self.ax.scatter(*target, marker="x", s=90, c="#d64c4c", label="Unconstrained minimum")
        self.ax.scatter(*optimum, marker="*", s=120, c="#16826c", label="Constrained minimum")
        self.ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="x", ylabel="y",
                    title=f"Lagrange multiplier = {multiplier:.3f}", aspect="equal")
        self.ax.legend(fontsize=7)
        self.history_ax.plot(np.sum((points - target) ** 2, axis=1), color="#16826c", label="Objective")
        self.history_ax.plot(np.abs(points @ normal - p.offset), color="#d64c4c", label="Constraint violation")
        self.history_ax.set(xlabel="Construction frame", ylabel="Value", title="Projection construction")
        self.history_ax.legend(fontsize=8)

    def run(self, data, p):
        target = np.array([p.target_x, p.target_y])
        normal = self.normal(p)
        optimum, multiplier = LagrangeProjection.solve(target, normal, p.offset)
        path = []
        for frame in range(p.frames + 1):
            point = target + (optimum - target) * frame / p.frames
            path.append(point)
            if not self.fast_forward or frame == p.frames:
                self._draw(p, target, optimum, path, multiplier)
            yield f"Projection {frame}/{p.frames}; objective={np.sum((point - target) ** 2):.5f}; constraint violation={abs(normal @ point - p.offset):.5f}"
        self.last_result = {"point": optimum, "multiplier": multiplier,
                            "objective": float(np.sum((optimum - target) ** 2))}
        yield f"Constrained global minimum: ({optimum[0]:.5f}, {optimum[1]:.5f}); lambda={multiplier:.5f}"