"""Multi-Layer Perceptron (neural network) with back-propagation, written from scratch."""
import numpy as np
from matplotlib.collections import LineCollection

from ml_lab.core import plotting as P
from ml_lab.core.base import ClassificationDemo
from ml_lab.core.params import Choice, Float, Int
from ml_lab.core.registry import register

ACTIVATIONS = {
    "relu": (lambda z: np.maximum(z, 0), lambda z, a: (z > 0).astype(float)),
    "leaky_relu": (lambda z: np.where(z > 0, z, 0.05 * z), lambda z, a: np.where(z > 0, 1.0, 0.05)),
    "tanh": (np.tanh, lambda z, a: 1 - a ** 2),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z, a: a * (1 - a)),
}


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


class MLP:
    def __init__(self, sizes, activation="tanh", rng=None):
        rng = rng or np.random.default_rng(0)
        self.sizes, self.act_name = sizes, activation
        self.act, self.dact = ACTIVATIONS[activation]
        gain = 2.0 if activation.endswith("relu") else 1.0      # He / Xavier initialisation
        self.W = [rng.normal(0, np.sqrt(gain / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self._opt_state = None

    def forward(self, X):
        A, Z = [X], []
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = A[-1] @ W + b
            Z.append(z)
            A.append(softmax(z) if l == len(self.W) - 1 else self.act(z))
        return A, Z

    def proba(self, X):
        return self.forward(X)[0][-1]

    def predict(self, X):
        return self.proba(X).argmax(1)

    def backward(self, X, Y, l2):
        """Back-propagation = chain rule applied layer by layer, from output to input."""
        A, Z = self.forward(X)
        n = len(X)
        dz = (A[-1] - Y) / n                     # softmax + cross-entropy gradient
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            gW[l] = A[l].T @ dz + l2 * self.W[l]
            gb[l] = dz.sum(0)
            if l > 0:
                dz = (dz @ self.W[l].T) * self.dact(Z[l - 1], A[l])
        loss = -np.mean(np.sum(Y * np.log(A[-1] + 1e-12), 1))
        return gW, gb, loss

    def update(self, gW, gb, lr, optimizer):
        params, grads = self.W + self.b, gW + gb
        if self._opt_state is None:
            self._opt_state = ([np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params], 0)
        m, v, t = self._opt_state
        t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if optimizer == "SGD":
                p -= lr * g
            elif optimizer == "Momentum":
                m[i] = 0.9 * m[i] + g
                p -= lr * m[i]
            else:                                                # Adam
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g ** 2
                p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        self._opt_state = (m, v, t)

    def fit_steps(self, X, y, epochs, lr, batch_size, optimizer, l2, rng):
        Y = np.eye(self.sizes[-1])[y]
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            losses = []
            for s in range(0, len(X), batch_size):
                bi = idx[s:s + batch_size]
                gW, gb, loss = self.backward(X[bi], Y[bi], l2)
                self.update(gW, gb, lr, optimizer)
                losses.append(loss)
            yield ep, float(np.mean(losses))


def draw_network(ax, net, max_show=12):
    ax.clear()
    ax.axis("off")
    L = len(net.sizes)
    shown = [min(s, max_show) for s in net.sizes]
    ys = [np.linspace(-1, 1, k) if k > 1 else np.zeros(1) for k in shown]
    wmax = max(np.abs(W).max() for W in net.W) + 1e-9
    segs, cols, lws = [], [], []
    for l, W in enumerate(net.W):
        for i in range(shown[l]):
            for j in range(shown[l + 1]):
                w = W[i, j]
                segs.append([(l, ys[l][i]), (l + 1, ys[l + 1][j])])
                cols.append((0.84, 0.15, 0.16, min(1, 0.1 + abs(w) / wmax)) if w > 0
                            else (0.12, 0.47, 0.71, min(1, 0.1 + abs(w) / wmax)))
                lws.append(0.3 + 2.5 * abs(w) / wmax)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=1))
    for l in range(L):
        fc = "#9ecae1" if l == 0 else "#fdae6b" if l == L - 1 else "#d9d9d9"
        ax.scatter(np.full(shown[l], l), ys[l], s=120, c=fc, edgecolors="k", zorder=3)
        lab = "input" if l == 0 else "output" if l == L - 1 else f"hidden {l}"
        more = f"\n(+{net.sizes[l] - shown[l]})" if net.sizes[l] > shown[l] else ""
        ax.text(l, -1.25, f"{lab}\n{net.sizes[l]}{more}", ha="center", va="top", fontsize=7)
    ax.set_xlim(-0.4, L - 0.6)
    ax.set_ylim(-1.7, 1.2)
    ax.set_title("Network  (red = +w, blue = \u2212w, width = |w|)", fontsize=10, fontweight="bold")


@register
class MLPDemo(ClassificationDemo):
    name = "Neural Network (MLP)"
    order = 41
    group = "Neural Networks"
    default_dataset = "spiral"
    default_noise = 0.15
    default_n = 300
    params = [
        Int("hidden_layers", 2, 1, 4),
        Int("neurons_per_layer", 16, 1, 64),
        Choice("activation", "tanh", list(ACTIVATIONS)),
        Choice("optimizer", "Adam", ["SGD", "Momentum", "Adam"]),
        Float("learning_rate", 0.02, 0.001, 1.0, 0.001, help="Adam likes ~0.01, SGD likes ~0.1-0.5."),
        Int("epochs", 600, 10, 5000, 10),
        Int("batch_size", 32, 1, 512),
        Float("l2", 0.0, 0.0, 0.01, 0.0001, help="Weight decay."),
        Int("draw_every", 10, 1, 200),
    ]
    info = """
    ## Idea
    Stack layers of neurons: each layer = linear map + non-linear activation.
          h\u2081 = act(W\u2081x + b\u2081),  h\u2082 = act(W\u2082h\u2081 + b\u2082),  \u2026,  p = softmax(W\u2099h + b\u2099)
    Without the non-linearity the whole stack collapses into ONE linear map.  With it, an MLP
    can approximate any continuous function (universal approximation theorem).

    ## Training = back-propagation + gradient descent
    Forward pass computes the prediction; backward pass applies the chain rule layer by layer
    to get \u2202loss/\u2202W for every weight; the optimiser (SGD / Momentum / Adam) takes a step.

    ## Embedded-engineer view
    Inference is just matrix-vector multiplies + an activation lookup - exactly what CMSIS-NN /
    TFLite Micro run on Cortex-M.  Parameters count = \u03a3 (in\u00d7out + out).  ReLU is cheapest
    (a compare); tanh/sigmoid need a LUT or approximation.  Weights are usually quantised to int8.
    Momentum is like a low-pass filter on the gradient; Adam additionally normalises each
    weight's step by its gradient RMS (automatic gain control per weight!).

    ## Try this
    \u2022 'spiral' with 1 hidden layer of 2 neurons -> fails. 2 layers of 16 -> success.
    \u2022 Activation 'sigmoid' + deep (4 layers) + SGD: very slow (vanishing gradients).
    \u2022 'xor' with 1 layer of 2-4 neurons: the classic problem the perceptron could not solve.
    """

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
        self.ax = fig.add_subplot(gs[:, 0])
        self.ax_loss = fig.add_subplot(gs[0, 1])
        self.ax_acc = self.ax_loss.twinx()
        self.ax_net = fig.add_subplot(gs[1, 1])

    def run(self, data, p):
        X, y = data["X"], data["y"]
        K = self.n_classes(data)
        rng = np.random.default_rng(p.seed)
        sizes = [2] + [p.neurons_per_layer] * p.hidden_layers + [K]
        net = MLP(sizes, p.activation, rng)
        n_params = sum(W.size + b.size for W, b in zip(net.W, net.b))
        self.log(f"Network {sizes}: {n_params} parameters (~{n_params} bytes as int8)")
        losses, accs = [], []
        for ep, loss in net.fit_steps(X, y, p.epochs, p.learning_rate, p.batch_size, p.optimizer, p.l2, rng):
            losses.append(loss)
            accs.append(np.mean(net.predict(X) == y))
            if not np.isfinite(loss):
                yield "\u26a0 Loss became NaN - learning rate too high"
                return
            if ep % p.draw_every and ep != p.epochs - 1:
                continue
            self.draw_model(data, net.predict, lambda G: net.proba(G)[:, 1],
                            title=f"MLP {sizes}, {p.activation}, {p.optimizer}", res=100)
            a, b = self.accuracy(data, net.predict)
            P.textbox(self.ax, f"epoch {ep + 1}\nparams {n_params}\ntrain acc {a:.1%}\ntest  acc {b:.1%}")
            la, aa = self.ax_loss, self.ax_acc
            la.clear()
            aa.clear()
            la.plot(losses, color=P.color(3))
            la.set_yscale("log")
            aa.plot(accs, color=P.color(2))
            aa.set_ylim(0, 1.05)
            aa.tick_params(labelsize=8)
            P.style(la, "Loss (red, log) & train accuracy (green)", "epoch", "cross-entropy")
            draw_network(self.ax_net, net)
            yield f"epoch {ep + 1}/{p.epochs}: loss {loss:.4f}, train acc {a:.1%}, test acc {b:.1%}"
