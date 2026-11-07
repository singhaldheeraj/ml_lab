"""Full-width minimax on a finite alternating game tree."""
from ml_lab.core.registry import register
from ._base import SearchStep
from ._game import GAME_INFO, GameSearchDemo


@register
class MinimaxSearchDemo(GameSearchDemo):
    name = "Minimax Search"
    order = 70
    complexity = "O(b^d)"
    space = "O(b^d) with retained tree and trace; O(d) recursion"
    info = """## Minimax
    MAX selects the largest child value; MIN selects the smallest. Evaluate every
    leaf and propagate values upward. Like choosing a control action while assuming
    an adversarial disturbance always chooses its most damaging response.
    """ + GAME_INFO

    def search(self, data, p):
        data["scores"] = {}
        visited, count = set(), 0

        def visit(node):
            nonlocal count
            visited.add(node)
            yield SearchStep(node, visited, comparisons=count, stored=len(visited), message=f"Visit node {node}")
            if not data["children"][node]:
                count += 1
                data["scores"][node] = data["payoffs"][node]
                return data["payoffs"][node], [node]
            maximizing = data["depths"][node] % 2 == 0
            best, line = (-float("inf") if maximizing else float("inf")), []
            for child in self.ordered(data, node, p):
                score, candidate = yield from visit(child)
                if (maximizing and score > best) or (not maximizing and score < best):
                    best, line = score, candidate
            data["scores"][node] = best
            yield SearchStep(node, visited, comparisons=count, stored=len(visited), message=f"Node {node} = {best}")
            return best, [node] + line

        score, line = yield from visit(0)
        yield SearchStep(done=True, found=True, result=score, path=line, visited=visited,
                         comparisons=count, stored=len(visited), message=f"Minimax payoff {score}; {count} leaves evaluated")