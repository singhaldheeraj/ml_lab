"""Spectral clustering with a locally scaled nearest-neighbor graph."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import ClusteringDemo
from ml_lab.core.params import Int
from ml_lab.core.registry import register


def _kmeans(X, k, rng, max_iter=100):
    centers = [X[int(rng.integers(len(X)))]]
    for _ in range(1, k):
        distances = ((X[:, None] - np.asarray(centers)[None]) ** 2).sum(2).min(1)
        centers.append(X[int(rng.choice(len(X), p=distances / distances.sum()))]
                       if distances.sum() > 0 else X[int(rng.integers(len(X)))])
    centers = np.asarray(centers).copy()
    for _ in range(max_iter):
        labels = ((X[:, None] - centers[None]) ** 2).sum(2).argmin(1)
        updated = np.array([X[labels == i].mean(0) if np.any(labels == i)
                            else X[int(rng.integers(len(X)))] for i in range(k)])
        if np.max(np.abs(updated - centers)) < 1e-7:
            centers = updated
            break
        centers = updated
    return ((X[:, None] - centers[None]) ** 2).sum(2).argmin(1)


class SpectralClustering:
    def __init__(self, k=3, neighbors=10, seed=0):
        self.k, self.neighbors, self.seed = k, neighbors, seed

    def fit_predict(self, X):
        n = len(X)
        distances = np.sqrt(np.maximum(((X[:, None] - X[None, :]) ** 2).sum(2), 0))
        neighbors = min(self.neighbors, n - 1)
        nearest = np.argsort(distances, axis=1)[:, 1:neighbors + 1]
        scales = np.maximum(np.take_along_axis(distances, nearest[:, -1:], axis=1), 1e-8).ravel()
        affinity = np.exp(-(distances ** 2) / np.maximum(scales[:, None] * scales[None, :], 1e-12))
        mask = np.zeros((n, n), dtype=bool)
        mask[np.arange(n)[:, None], nearest] = True
        mask |= mask.T
        affinity *= mask
        np.fill_diagonal(affinity, 0)
        degree = affinity.sum(1)
        inv_sqrt = 1 / np.sqrt(np.maximum(degree, 1e-12))
        normalized = inv_sqrt[:, None] * affinity * inv_sqrt[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(normalized)
        embedding = eigenvectors[:, -self.k:]
        embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
        self.eigenvalues = eigenvalues[-self.k:][::-1]
        self.labels = _kmeans(embedding, self.k, np.random.default_rng(self.seed))
        return self.labels


@register
class SpectralClusteringDemo(ClusteringDemo):
    name = "Spectral Clustering"
    group = "Graph clustering"
    order = 65
    max_n = 500
    instant = True
    params = [Int("k", 3, 1, 10),
              Int("neighbors", 10, 2, 30,
                  help="Number of nearby points used to connect the similarity graph.")]
    info = """
    ## Idea
    Build a similarity graph, use eigenvectors of its normalized adjacency matrix to map
    points into a space where connected groups separate, then run K-Means there.

    ## When it helps
    The graph can follow curved shapes such as moons and circles, where ordinary K-Means
    struggles. The neighbor count controls how local the graph connections are.
    """

    def run(self, data, p):
        X = data["X"]
        if p.k > len(X):
            yield "k is larger than the number of points"
            return
        model = SpectralClustering(p.k, p.neighbors, p.seed)
        labels = model.fit_predict(X)
        self.ax.clear()
        self.draw_labels(self.ax, X, labels)
        P.set_limits(self.ax, X)
        P.style(self.ax, f"Spectral clustering · {p.k} groups, {p.neighbors} neighbors")
        yield f"clustered {len(X)} points; leading graph eigenvalues {model.eigenvalues[:3].round(3)}"