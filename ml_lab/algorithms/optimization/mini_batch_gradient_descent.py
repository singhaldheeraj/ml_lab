"""Small shuffled batches using the shared least-squares model."""
from ml_lab.core.registry import register
from ._regression import TrainingDemo


@register
class MiniBatchGradientDescentDemo(TrainingDemo):
    name = "Mini-Batch Gradient Descent"
    order = 40
    method = "Mini-batch GD"
    info = """## Mini-batch gradient descent
    Average gradients over a small shuffled group before updating parameters.
    Batch size 1 is SGD; a batch at least as large as the training set is full
    batch descent. The last group may be smaller. This balances noisy cheap
    updates and smoother expensive updates, like processing a sensor buffer.
    """ + TrainingDemo.info