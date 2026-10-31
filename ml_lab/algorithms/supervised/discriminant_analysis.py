"""Linear and quadratic Gaussian discriminant analysis."""
import numpy as np

from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Float
from ml_lab.core.registry import register


class GaussianDiscriminant:
    def __init__(self, mode="lda", regularization=0.05):
        self.mode = mode
        self.regularization = regularization

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = np.array([X[y == label].mean(0) for label in self.classes])
        self.priors = np.array([(y == label).mean() for label in self.classes])
        eye = np.eye(X.shape[1]) * max(self.regularization, 1e-8)
        covariances = []
        for label in self.classes:
            points = X[y == label]
            centered = points - points.mean(0)
            covariances.append(centered.T @ centered / max(len(points), 1))
        if self.mode == "lda":
            pooled = sum(cov * np.sum(y == label) for cov, label in zip(covariances, self.classes))
            pooled /= max(len(y) - len(self.classes), 1)
            covariances = [pooled] * len(self.classes)
        self.inverses = [np.linalg.inv(cov + eye) for cov in covariances]
        self.log_determinants = [np.linalg.slogdet(cov + eye)[1] for cov in covariances]
        return self

    def predict(self, X):
        scores = []
        for mean, prior, inverse, log_det in zip(
                self.means, self.priors, self.inverses, self.log_determinants):
            delta = X - mean
            distance = np.einsum("ni,ij,nj->n", delta, inverse, delta)
            scores.append(np.log(prior + 1e-12) - 0.5 * (distance + log_det))
        return self.classes[np.argmax(np.column_stack(scores), axis=1)]


class _DiscriminantDemo(ClassificationDemo):
    instant = True
    rerun_on_click = True
    params = [Float("regularization", 0.05, 0.0, 1.0, 0.01,
                    help="Diagonal covariance stabilizer; increase for nearly-collinear features.")]
    info = """
    ## Idea
    Model each class as a Gaussian distribution, then choose the class with the highest
    posterior score. LDA shares one covariance across classes; QDA fits one covariance per class.

    ## Compare
    LDA creates linear decision boundaries and is data-efficient. QDA can model curved
    boundaries, but estimates more parameters and can overfit small datasets.
    """

    def run(self, data, p):
        model = GaussianDiscriminant(self.mode, p.regularization).fit(data["X"], data["y"])
        self.draw_model(data, model.predict, title=self.name)
        train, test = self.accuracy(data, model.predict)
        yield f"train accuracy {train:.1%}, test accuracy {test:.1%}"


@register
class LDADemo(_DiscriminantDemo):
    name = "Linear Discriminant Analysis (LDA)"
    order = 50
    mode = "lda"


@register
class QDADemo(_DiscriminantDemo):
    name = "Quadratic Discriminant Analysis (QDA)"
    order = 51
    mode = "qda"