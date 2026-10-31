"""Toy dataset generators (pure numpy). All 2-D sets live roughly inside [-4, 4]^2."""
from __future__ import annotations

import numpy as np

CLASSIFICATION_SETS = ("blobs", "moons", "circles", "xor", "spiral", "linear")
CLUSTER_SETS = ("blobs", "anisotropic", "varied", "moons", "circles", "spiral", "uniform")
REGRESSION_SETS = ("linear", "quadratic", "cubic", "sine", "step", "damped")


def rng_from(seed) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _split2(n):
    n1 = n // 2
    return n1, n - n1


# ---------------------------------------------------------------- 2-D shapes
def blobs(n, k, std, rng, radius=3.0):
    if k == 1:
        centers = np.zeros((1, 2))
    else:
        ang = rng.uniform(0, 2 * np.pi) + np.arange(k) * 2 * np.pi / k
        centers = np.c_[np.cos(ang), np.sin(ang)] * radius
    y = np.arange(n) % k
    X = centers[y] + rng.normal(0, std, (n, 2))
    return X, y


def moons(n, noise, rng):
    n1, n2 = _split2(n)
    t1, t2 = rng.uniform(0, np.pi, n1), rng.uniform(0, np.pi, n2)
    X = np.r_[np.c_[np.cos(t1), np.sin(t1)], np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]]
    X = (X - [0.5, 0.25] + rng.normal(0, noise * 0.5, X.shape)) * 2.0
    y = np.r_[np.zeros(n1, int), np.ones(n2, int)]
    return X, y


def circles(n, noise, rng, factor=0.4):
    n1, n2 = _split2(n)
    t = rng.uniform(0, 2 * np.pi, n)
    r = np.r_[np.ones(n1), np.full(n2, factor)] * 2.5
    X = np.c_[r * np.cos(t), r * np.sin(t)] + rng.normal(0, noise * 0.8, (n, 2))
    y = np.r_[np.zeros(n1, int), np.ones(n2, int)]
    return X, y


def xor(n, noise, rng):
    X = rng.uniform(-2.5, 2.5, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X + rng.normal(0, noise * 0.6, X.shape), y


def spiral(n, noise, rng, k=2):
    y = np.arange(n) % k
    t = rng.uniform(0, 1, n)
    r = 0.3 + 2.7 * t
    theta = 3.5 * np.pi * t + y * 2 * np.pi / k
    X = np.c_[r * np.cos(theta), r * np.sin(theta)] + rng.normal(0, noise * 0.5, (n, 2))
    return X, y


def linear(n, noise, rng):
    n1, n2 = _split2(n)
    std = 0.4 + 1.4 * noise
    X = np.r_[rng.normal([-1.5, -1.0], std, (n1, 2)), rng.normal([1.5, 1.0], std, (n2, 2))]
    y = np.r_[np.zeros(n1, int), np.ones(n2, int)]
    return X, y


def classification(kind, n, noise, seed, n_classes=2):
    rng = rng_from(seed)
    if kind == "blobs":
        return blobs(n, n_classes, 0.4 + 1.6 * noise, rng)
    if kind == "moons":
        return moons(n, noise, rng)
    if kind == "circles":
        return circles(n, noise, rng)
    if kind == "xor":
        return xor(n, noise, rng)
    if kind == "spiral":
        return spiral(n, noise, rng, n_classes)
    if kind == "linear":
        return linear(n, noise, rng)
    raise ValueError(f"Unknown dataset {kind!r}")


def clusters(kind, n, noise, seed, k=3):
    """Returns X and the *true* labels (which unsupervised algorithms never see)."""
    rng = rng_from(seed)
    if kind == "blobs":
        return blobs(n, k, 0.3 + 1.2 * noise, rng)
    if kind == "anisotropic":
        X, y = blobs(n, k, 0.3 + 0.8 * noise, rng)
        return X @ np.array([[0.9, -0.5], [-0.6, 0.5]]) * 1.3, y
    if kind == "varied":
        X, y = blobs(n, k, 1.0, rng)
        stds = np.linspace(0.25, 1.2, k) * (0.5 + noise)
        centers = np.array([X[y == c].mean(0) for c in range(k)])
        return centers[y] + rng.normal(0, 1, (n, 2)) * stds[y, None], y
    if kind == "moons":
        return moons(n, noise, rng)
    if kind == "circles":
        return circles(n, noise, rng)
    if kind == "spiral":
        return spiral(n, noise, rng, k)
    if kind == "uniform":
        return rng.uniform(-3, 3, (n, 2)), np.zeros(n, int)
    raise ValueError(f"Unknown dataset {kind!r}")


# ---------------------------------------------------------------- 1-D regression
TRUE_FUNCTIONS = {
    "linear": lambda x: 1.5 * x + 0.5,
    "quadratic": lambda x: 0.6 * x ** 2 - x - 1.0,
    "cubic": lambda x: 0.25 * x ** 3 - 1.2 * x,
    "sine": lambda x: 2.0 * np.sin(1.5 * x),
    "step": lambda x: np.where(x < -1, -1.5, np.where(x < 1, 1.0, 2.5)),
    "damped": lambda x: 3.0 * np.exp(-0.3 * (x + 3)) * np.cos(3 * x),
}


def regression_1d(kind, n, noise, seed, lo=-3.0, hi=3.0):
    rng = rng_from(seed)
    f = TRUE_FUNCTIONS[kind]
    x = np.sort(rng.uniform(lo, hi, n))
    y = f(x) + rng.normal(0, noise * 2.0, n)
    return x, y, f


# ---------------------------------------------------------------- utilities
def standardize(X):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    return (X - mu) / sd, mu, sd


def train_test_split(n, test_frac, rng):
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    return idx[n_test:], idx[:n_test]
