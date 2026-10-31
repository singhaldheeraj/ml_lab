"""Principal Component Analysis - find the directions of maximum variance."""
import numpy as np

from ml_lab.core import plotting as P
from ml_lab.core.base import AlgorithmDemo
from ml_lab.core.datasets import rng_from
from ml_lab.core.params import Bool, Choice, Float, Int
from ml_lab.core.registry import register


class PCA:
    def fit(self, X):
        self.mean = X.mean(0)
        C = np.cov((X - self.mean).T)                         # covariance matrix
        vals, vecs = np.linalg.eigh(C)                        # eigen-decomposition
        order = np.argsort(vals)[::-1]
        self.variances, self.components = vals[order], vecs[:, order].T
        self.ratio = self.variances / self.variances.sum()
        return self

    def transform(self, X, n=None):
        return (X - self.mean) @ self.components[:n].T

    def inverse(self, Z):
        return Z @ self.components[:Z.shape[1]] + self.mean


@register
class PCADemo(AlgorithmDemo):
    name = "PCA (Principal Component Analysis)"
    category = "Unsupervised"
    group = "Dimensionality Reduction"
    order = 30
    click_to_add = True
    click_help = "Click on plot: LEFT = add a point, RIGHT = remove nearest point."

    def data_params(self):
        return [Choice("dataset", "tilted ellipse", ["tilted ellipse", "two blobs", "curved"], group="Data"),
                Int("n_samples", 200, 10, 1000, 10, group="Data"),
                Float("angle_deg", 30.0, 0.0, 180.0, 1.0, group="Data", help="Rotation of the cloud."),
                Float("stretch", 3.0, 1.0, 8.0, 0.1, group="Data", help="Ratio of long to short axis."),
                Int("seed", 0, 0, 999, group="Data")]

    params = [
        Int("n_angles", 90, 10, 360, help="Frames of the rotating search direction."),
        Bool("show_reconstruction", True, help="Final: project onto PC1 only (2-D -> 1-D compression)."),
    ]
    info = """
    ## Idea
    Find the direction along which the data varies the most (PC1).  PC2 is the most varying
    direction perpendicular to PC1, and so on.  Mathematically: eigenvectors of the covariance
    matrix, sorted by eigenvalue (= variance along that direction).

    ## The animation
    A line rotates from 0\u00b0 to 180\u00b0.  The data is projected onto it (red dots) and the variance of
    the projection is plotted on the right.  The maximum of that curve IS the first principal
    component - PCA is literally "the rotation with the most spread".

    ## Embedded-engineer view
    \u2022 A 3-axis accelerometer mounted at an unknown angle on a vibrating machine: PC1 is the real
      vibration axis - rotate your coordinates to it and you get a 1-channel signal.
    \u2022 Compression: keep only the top components (explained-variance plot tells you how much
      you lose).  Same maths as the Karhunen-Lo\u00e8ve transform used in signal coding.
    \u2022 Standard pre-processing to decorrelate & reduce features before a small classifier.

    ## Try this
    \u2022 stretch = 1: a round cloud - all directions equal, PCA becomes meaningless.
    \u2022 'curved': PCA is linear; it cannot unroll curves (see t-SNE).
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_var = fig.add_subplot(gs[0, 1])
        self.ax_pc = fig.add_subplot(gs[1, 1])

    def make_data(self, p):
        rng = rng_from(p.seed)
        n = p.n_samples
        if p.dataset == "tilted ellipse":
            Z = rng.normal(0, 1, (n, 2)) * [p.stretch * 0.6, 0.6]
        elif p.dataset == "two blobs":
            Z = rng.normal(0, 0.5, (n, 2)) + np.where(np.arange(n)[:, None] % 2, 1, -1) * [p.stretch * 0.5, 0]
        else:
            t = rng.uniform(-2.5, 2.5, n)
            Z = np.c_[t * p.stretch / 3, 0.35 * t ** 2 - 1] + rng.normal(0, 0.15, (n, 2))
        a = np.radians(p.angle_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return {"X": Z @ R.T + [0.5, 0.3]}

    def draw_data(self, data, p):
        ax = self.ax
        ax.clear()
        ax.scatter(*data["X"].T, s=14, c=P.color(0), alpha=0.7)
        ax.set_aspect("equal", adjustable="datalim")
        P.style(ax, "Data")

    def on_click(self, data, x, y, button, p):
        X = data["X"]
        if button == 1:
            data["X"] = np.vstack([X, [[x, y]]])
        elif len(X) > 3:
            data["X"] = np.delete(X, np.argmin(np.hypot(X[:, 0] - x, X[:, 1] - y)), 0)
        return True

    def run(self, data, p):
        X = data["X"]
        pca = PCA().fit(X)
        mu = pca.mean
        R = np.abs(X - mu).max() * 1.3
        angles = np.linspace(0, 180, p.n_angles, endpoint=False)
        variances = []
        pc1_angle = np.degrees(np.arctan2(pca.components[0, 1], pca.components[0, 0])) % 180
        for a in angles:
            u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
            proj = (X - mu) @ u
            variances.append(proj.var(ddof=1))
            ax = self.ax
            self.draw_data(data, p)
            ax.plot(*np.c_[mu - R * u, mu + R * u], "k-", lw=1.5)
            pts = mu + proj[:, None] * u
            ax.plot(np.c_[X[:, 0], pts[:, 0]].T, np.c_[X[:, 1], pts[:, 1]].T, "-", color="0.6", lw=0.4)
            ax.scatter(*pts.T, s=10, c="crimson", zorder=4)
            P.textbox(ax, f"angle {a:5.1f}\u00b0\nprojected variance {variances[-1]:.3f}")
            va = self.ax_var
            va.clear()
            va.plot(angles[:len(variances)], variances, color=P.color(3))
            va.axvline(pc1_angle, color="k", ls=":", lw=1)
            va.set_xlim(0, 180)
            va.set_ylim(0, pca.variances[0] * 1.1)
            P.style(va, "Variance of the projection vs. angle", "direction angle (\u00b0)", "variance")
            yield f"angle {a:.1f}\u00b0: projected variance {variances[-1]:.3f}  (max = PC1 at {pc1_angle:.1f}\u00b0)"

        ax = self.ax
        self.draw_data(data, p)
        for i, (v, lam) in enumerate(zip(pca.components, pca.variances)):
            ax.annotate("", xy=mu + v * 2 * np.sqrt(lam), xytext=mu,
                        arrowprops=dict(arrowstyle="->", lw=3, color=P.color(3 if i == 0 else 2)))
            ax.text(*(mu + v * 2.2 * np.sqrt(lam)), f"PC{i + 1}", fontsize=11, fontweight="bold")
        if p.show_reconstruction:
            Xr = pca.inverse(pca.transform(X, 1))
            ax.scatter(*Xr.T, s=8, c="crimson", label="reconstructed from PC1 only")
            err = np.mean(np.sum((X - Xr) ** 2, 1))
            ax.legend(fontsize=7, loc="lower right")
            self.log(f"Reconstruction MSE using 1 of 2 components: {err:.4f}")
        P.textbox(ax, "eigenvalues: " + ", ".join(f"{v:.3f}" for v in pca.variances)
                  + "\nexplained:   " + ", ".join(f"{r:.1%}" for r in pca.ratio))
        Z = pca.transform(X)
        pa = self.ax_pc
        pa.clear()
        pa.scatter(Z[:, 0], Z[:, 1], s=10, c=P.color(0))
        pa.set_aspect("equal", adjustable="datalim")
        P.style(pa, f"Data in PC coordinates (decorrelated)  PC1 = {pca.ratio[0]:.1%}", "PC1", "PC2")
        self.log(f"Covariance in PC space:\n{np.round(np.cov(Z.T), 4)}")
        yield f"PC1 at {pc1_angle:.1f}\u00b0 explains {pca.ratio[0]:.1%} of the variance"
