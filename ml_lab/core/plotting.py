"""Small matplotlib helpers shared by all demos."""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def color(i):
    return COLORS[int(i) % len(COLORS)]


def colors_for(labels):
    return [color(c) if c >= 0 else "#444444" for c in labels]


def proba_cmap():
    return LinearSegmentedColormap.from_list("p01", [COLORS[0], "#ffffff", COLORS[1]])


def style(ax, title="", xlabel="x\u2081", ylabel="x\u2082", grid=True):
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    if grid:
        ax.grid(True, alpha=0.25)


def set_limits(ax, X, pad=0.6):
    lo, hi = X.min(0) - pad, X.max(0) + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])


def scatter_classes(ax, X, y, s=28, alpha=0.95, marker="o", edge="k", label=True, zorder=3):
    for c in np.unique(y):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=s, c=color(c), edgecolors=edge, linewidths=0.5,
                   marker=marker, alpha=alpha, zorder=zorder,
                   label=f"class {c}" if label else None)


def mesh(X, res=120, pad=0.6):
    lo, hi = X.min(0) - pad, X.max(0) + pad
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], res), np.linspace(lo[1], hi[1], res))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def plot_regions(ax, predict, X, n_classes, res=120, alpha=0.22):
    """Colour the plane by predicted class (predict: (m,2) -> (m,) ints)."""
    xx, yy, G = mesh(X, res)
    Z = np.asarray(predict(G)).reshape(xx.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.contourf(xx, yy, Z, levels=np.arange(n_classes + 1) - 0.5,
                    colors=COLORS[:n_classes], alpha=alpha)
        ax.contour(xx, yy, Z, levels=np.arange(n_classes - 1) + 0.5, colors="k",
                   linewidths=0.8, alpha=0.6)


def plot_proba(ax, proba1, X, res=120, alpha=0.65):
    """Binary problems: heat-map of P(class 1) with the 0.5 decision boundary."""
    xx, yy, G = mesh(X, res)
    Z = np.asarray(proba1(G)).reshape(xx.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap=proba_cmap(),
                    alpha=alpha, vmin=0, vmax=1)
        ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.6)


def plot_scalar_field(ax, fn, X, levels=None, res=120, cmap="viridis", alpha=0.6, lines=None):
    xx, yy, G = mesh(X, res)
    Z = np.asarray(fn(G)).reshape(xx.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = ax.contourf(xx, yy, Z, levels=levels or 20, cmap=cmap, alpha=alpha)
        if lines is not None:
            ax.contour(xx, yy, Z, levels=lines, colors="k", linewidths=1.2,
                       linestyles=["--", "-", "--"][:len(lines)] if len(lines) == 3 else "-")
    return cs


def draw_ellipse(ax, mean, cov, col, nstd=(1.0, 2.0), lw=1.5, fill_alpha=0.0):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    for k in np.atleast_1d(nstd):
        w, h = 2 * k * np.sqrt(vals[-1]), 2 * k * np.sqrt(vals[0])
        ax.add_patch(Ellipse(mean, w, h, angle=angle, edgecolor=col, lw=lw,
                             facecolor=col if fill_alpha else "none", alpha=max(fill_alpha, 0.9),
                             zorder=4))


def textbox(ax, text, loc="upper left", fontsize=8):
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.98, "top") if "upper" in loc else (0.02, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, zorder=10,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
