"""Single-sample gradient updates using the shared least-squares model."""
from ml_lab.core.registry import register
from ._regression import TrainingDemo


@register
class StochasticGradientDescentDemo(TrainingDemo):
    name = "Stochastic Gradient Descent (SGD)"
    order = 30
    method = "Stochastic GD"
    info = """## Stochastic gradient descent
    Shuffle the training set each epoch, then update from one sample at a time.
    Each update is cheap, but its gradient is noisy. The seed controls shuffling.
    Like calibrating continuously from one incoming sensor capture at a time.
    """ + TrainingDemo.info