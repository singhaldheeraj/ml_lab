"""Full-batch gradient descent using the shared least-squares model."""
from ml_lab.core.registry import register
from ._regression import TrainingDemo


@register
class BatchGradientDescentDemo(TrainingDemo):
    name = "Batch Gradient Descent"
    order = 20
    method = "Batch GD"
    info = """## Batch method
    Average gradients over the entire training set before each update.
    One epoch produces one update: deterministic, smooth, but each step reads
    every sample. Compare its loss path with stochastic and mini-batch descent.
    """ + TrainingDemo.info