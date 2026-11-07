"""Adversarial UCT search with seeded random terminal rollouts."""
import math
import numpy as np
from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register
from ._base import SearchStep
from ._game import GAME_INFO, GameSearchDemo


@register
class MonteCarloTreeSearchDemo(GameSearchDemo):
    name = "Monte Carlo Tree Search (UCT)"
    order = 72
    params = GameSearchDemo.params + [Int("simulations", 200, 10, 1000, 10),
                                      Float("exploration", 1.4, 0.0, 3.0, 0.1)]
    complexity = "O(simulations * depth * branching)"
    space = "O(min(simulations, b^d)) search statistics"
    info = """## Monte Carlo tree search / UCT
    Select children using an upper confidence bound, expand one new node, simulate
    random play to a leaf, then backpropagate the normalized payoff. At MIN nodes,
    invert the exploitation score so the opponent prefers low MAX payoffs.
    Like allocating more experiments to promising actions while still exploring.
    ## Result
    Recommend the root child with most visits. The reported value is its sampled
    mean payoff, not an exact minimax value. More simulations may improve estimates
    but finite budgets do not certify optimality. Probes count terminal rollouts.
    """ + GAME_INFO

    def search(self, data, p):
        rng = np.random.default_rng(p.seed)
        visits, rewards, expanded = {0: 0}, {0: 0.0}, {0: []}
        visited = set()
        data["scores"] = {}
        for simulation in range(p.simulations):
            node, trace = 0, [0]
            while data["children"][node]:
                children = self.ordered(data, node, p)
                untried = [child for child in children if child not in expanded[node]]
                if untried:
                    child = untried[0]
                    expanded[node].append(child)
                    visits[child], rewards[child], expanded[child] = 0, 0.0, []
                    node = child
                    trace.append(node)
                    break
                maximizing = data["depths"][node] % 2 == 0

                def confidence(child):
                    mean = rewards[child] / visits[child]
                    exploitation = mean if maximizing else 1 - mean
                    return exploitation + p.exploration * math.sqrt(math.log(max(1, visits[node])) / visits[child])

                node = max(expanded[node], key=confidence)
                trace.append(node)
            terminal = node
            while data["children"][terminal]:
                children = data["children"][terminal]
                terminal = children[int(rng.integers(len(children)))]
            reward = (data["payoffs"][terminal] + 20) / 40
            for ancestor in trace:
                visits[ancestor] += 1
                rewards[ancestor] += reward
                visited.add(ancestor)
                data["scores"][ancestor] = round(40 * rewards[ancestor] / visits[ancestor] - 20, 1)
            yield SearchStep(node, visited, path=trace, comparisons=simulation + 1, stored=3 * len(visits),
                             message=f"Simulation {simulation + 1}; sampled payoff {data['payoffs'][terminal]}")
        line, node = [0], 0
        while expanded[node]:
            node = max(expanded[node], key=lambda child: visits[child])
            line.append(node)
        choice = line[1] if len(line) > 1 else 0
        estimate = 40 * rewards[choice] / visits[choice] - 20
        yield SearchStep(done=True, found=True, result=estimate, path=line, visited=visited,
                         comparisons=p.simulations, stored=3 * len(visits),
                         message=f"Recommend node {choice}; sampled mean {estimate:.3f} (not certified minimax)")