"""Minimax with alpha-beta pruning."""
from ml_lab.core.registry import register
from ._base import SearchStep
from ._game import GAME_INFO, GameSearchDemo


@register
class AlphaBetaSearchDemo(GameSearchDemo):
    name = "Alpha-Beta Pruning"
    order = 71
    complexity = "O(b^d) worst, O(b^(d/2)) ideal ordering"
    space = "O(b^d) retained tree/trace; O(d) recursion"
    info = """## Alpha-beta search
    Maintain MAX's lower bound alpha and MIN's upper bound beta. Prune a subtree
    once it cannot affect an ancestor's decision. Produces the same root value as
    full minimax while often examining fewer leaves. Move order strongly affects pruning.
    Like abandoning a diagnostic option once it cannot beat an already known choice.
    """ + GAME_INFO

    def search(self, data, p):
        data["scores"] = {}
        visited, count = set(), 0

        def visit(node, alpha, beta):
            nonlocal count
            visited.add(node)
            yield SearchStep(node, visited, comparisons=count, stored=len(visited),
                             message=f"Visit {node}; alpha={alpha:g}, beta={beta:g}")
            if not data["children"][node]:
                count += 1
                data["scores"][node] = data["payoffs"][node]
                return data["payoffs"][node], [node]
            maximizing = data["depths"][node] % 2 == 0
            best, line = (-float("inf") if maximizing else float("inf")), []
            for child in self.ordered(data, node, p):
                score, candidate = yield from visit(child, alpha, beta)
                if (maximizing and score > best) or (not maximizing and score < best):
                    best, line = score, candidate
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta:
                    yield SearchStep(node, visited, comparisons=count, stored=len(visited), message=f"Prune siblings: alpha {alpha:g} >= beta {beta:g}")
                    break
            data["scores"][node] = best
            return best, [node] + line

        score, line = yield from visit(0, -float("inf"), float("inf"))
        yield SearchStep(done=True, found=True, result=score, path=line, visited=visited,
                         comparisons=count, stored=len(visited), message=f"Minimax payoff {score}; {count} leaves evaluated")