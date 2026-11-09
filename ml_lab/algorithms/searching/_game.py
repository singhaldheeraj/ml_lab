"""Finite alternating MAX/MIN trees for adversarial search demos."""
import numpy as np
from ml_lab.core.params import Choice, Int
from ._base import SearchDemo


class GameSearchDemo(SearchDemo):
    category = "Miscellaneous"
    family = "game"
    group = "Adversarial game-tree search"
    params = [Choice("move_order", "left first", ("left first", "right first"))]

    def data_params(self):
        return [Int("depth", 3, 1, 4, group="Data"),
                Int("branching", 2, 2, 4, group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        children, payoffs, positions, depths, leaf_count = {}, {}, {}, {}, 0

        def build(depth):
            nonlocal leaf_count
            node = len(children)
            children[node], depths[node] = [], depth
            if depth == p.depth:
                payoffs[node] = int(rng.integers(-20, 21))
                positions[node] = (leaf_count, -depth)
                leaf_count += 1
            else:
                children[node] = [build(depth + 1) for _ in range(p.branching)]
                positions[node] = (float(np.mean([positions[child][0] for child in children[node]])), -depth)
            return node

        build(0)
        return {"children": children, "payoffs": payoffs, "positions": positions, "depths": depths, "scores": {}}

    @staticmethod
    def ordered(data, node, p):
        children = data["children"][node]
        return children if p.move_order == "left first" else children[::-1]

    def _draw(self, data, step):
        self.ax.clear()
        positions = data["positions"]
        for node, children in data["children"].items():
            for child in children:
                self.ax.plot([positions[node][0], positions[child][0]], [positions[node][1], positions[child][1]],
                             color="#d2dbe1", lw=0.7, zorder=1)
        for node, (horizontal, vertical) in positions.items():
            color = "#16826c" if node in step.path else "#d64c4c" if node == step.current else (
                "#e4b54b" if node in step.visited else "#c8d6df")
            self.ax.scatter(horizontal, vertical, s=90 if len(positions) < 70 else 12, c=color, zorder=2)
            if len(positions) < 70 or data["depths"][node] < 2:
                label = str(data["scores"].get(node, data["payoffs"].get(node,
                            "MAX" if data["depths"][node] % 2 == 0 else "MIN")))
                self.ax.annotate(label, (horizontal, vertical), xytext=(0, 8), textcoords="offset points",
                                 ha="center", fontsize=8)
        self.ax.margins(0.08, 0.2)
        self.ax.set_title(self.name)
        self.ax.axis("off")


GAME_INFO = """
    ## Configuration and performance
    Root is MAX; players alternate MAX/MIN. Leaf payoffs are seeded integers.
    Adjust depth, branching, seed and traversal order. Probe counts are evaluated
    terminal payoffs. The best line is green; visited nodes are gold.
    Minimax/alpha-beta values are exact for this finite tree; UCT values are estimates.
    Storage estimates include retained display scores and visited-node history.
    """