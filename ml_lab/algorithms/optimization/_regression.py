"""Optimization-focused views of the existing regression training model."""
from dataclasses import replace

from ml_lab.algorithms.supervised.linear_regression import LinearRegressionDemo
from ml_lab.core.params import Values


class TrainingDemo(LinearRegressionDemo):
    category = "Optimization"
    group = "Gradient-based training"
    method = "Batch GD"
    info = """## Least-squares optimization
    Fit y_hat = w*x + b by minimizing mean squared error J = mean((y_hat-y)^2).
    Each update subtracts learning_rate * gradient from the slope and intercept.
    The exact least-squares solution is shown as a reference on the loss surface.
    Full batch uses every training sample; stochastic uses one shuffled sample;
    mini-batch uses a shuffled group. An epoch is one pass over the training set.
    Stochastic and mini-batch loss can rise on individual updates. A constant
    learning rate can leave noisy motion near the optimum; convergence is not
    guaranteed for arbitrary learning rates. Excessively large updates diverge.
    Like adjusting a sensor's gain and offset from all captures, one incoming
    capture, or a small buffer of captures respectively.
    """

    def get_params(self):
        parameters = []
        for param in super().get_params():
            if param.name == "method" or (param.name == "batch_size" and self.method != "Mini-batch GD"):
                continue
            if param.name == "epochs":
                param = replace(param, default=30)
            elif param.name == "draw_every":
                param = replace(param, default=10)
            parameters.append(param)
        return parameters

    def run(self, data, p):
        values = Values(**dict(vars(p), method=self.method, batch_size=p.get("batch_size", 8)))
        yield from super().run(data, values)