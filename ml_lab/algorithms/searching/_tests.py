"""Search correctness tests, independent of Tk and drawing."""
import unittest

import heapq
import numpy as np

from ml_lab.core.params import Values
from ml_lab.core.registry import discover


class SearchTests(unittest.TestCase):
    def test_category_name(self):
        classes, errors = discover()
        self.assertFalse(errors)
        searching = [cls for cls in classes if cls.__module__.startswith("ml_lab.algorithms.searching.")]
        self.assertEqual(len(searching), 37)
        self.assertEqual({cls.category for cls in searching}, {"Searching algorithms", "Miscellaneous"})
        miscellaneous = [cls for cls in searching if cls.category == "Miscellaneous"]
        self.assertEqual({cls.name for cls in miscellaneous},
                         {"Minimax Search", "Alpha-Beta Pruning", "Monte Carlo Tree Search (UCT)"})

    def test_genetic_search(self):
        from .genetic_search import GeneticSearchDemo

        demo = GeneticSearchDemo()
        for seed in (0, 7):
            for mutation_rate in (0.0, 0.15, 1.0):
                params = Values(**{param.name: param.default for param in demo.get_params()})
                params.seed, params.mutation_rate = seed, mutation_rate
                params.generations, params.population_size = 5, 7
                data = demo.make_data(params)
                steps = list(demo.search(data, params))
                last = steps[-1]
                self.assertTrue(last.done and last.found)
                self.assertEqual(last.comparisons, 42)
                scores = [data["heights"][node] for node in last.path]
                self.assertGreaterEqual(scores[0], data["heights"][data["start"]])
                self.assertEqual(scores, sorted(scores))
                self.assertEqual(last.result, list(demo.search(data, params))[-1].result)
                for step in steps:
                    for node in step.frontier:
                        self.assertTrue(all(0 <= node[axis] < data["heights"].shape[axis] for axis in (0, 1)))

    def test_traveling_salesman(self):
        from itertools import permutations
        from .traveling_salesman import TravelingSalesmanDemo

        demo = TravelingSalesmanDemo()
        params = Values(**{param.name: param.default for param in demo.get_params()})
        params.cities = 5
        for layout in ("random", "circle", "clusters"):
            params.layout = layout
            data = demo.make_data(params)
            points = data["points"]

            def length(tour):
                return sum(float(np.linalg.norm(points[first] - points[second]))
                           for first, second in zip(tour, tour[1:]))

            optimum = min(length([0, *ordering, 0]) for ordering in permutations(range(1, 5)))
            for method in ("exhaustive (exact)", "nearest neighbor + 2-opt"):
                params.method = method
                steps = list(demo.search(data, params))
                last = steps[-1]
                self.assertTrue(last.done and last.found)
                self.assertEqual(last.result[0], 0)
                self.assertEqual(last.result[-1], 0)
                self.assertEqual(sorted(last.result[:-1]), list(range(5)))
                self.assertAlmostEqual(last.cost, length(last.result))
                if method == "exhaustive (exact)":
                    self.assertAlmostEqual(last.cost, optimum)
                    self.assertEqual(last.comparisons, 24)
                else:
                    closed_costs = [step.cost for step in steps if len(step.path) == 6]
                    self.assertEqual(closed_costs, sorted(closed_costs, reverse=True))
                repeated = list(demo.search(data, params))[-1]
                self.assertEqual(last.result, repeated.result)

    def test_game_searches(self):
        from .minimax_search import MinimaxSearchDemo
        from .alpha_beta_search import AlphaBetaSearchDemo

        for seed in range(10):
            demo = MinimaxSearchDemo()
            params = Values(**{param.name: param.default for param in demo.get_params()})
            params.seed = seed
            data = demo.make_data(params)

            def reference(node):
                children = data["children"][node]
                if not children:
                    return data["payoffs"][node]
                operation = max if data["depths"][node] % 2 == 0 else min
                return operation(reference(child) for child in children)

            full = list(demo.search(data, params))[-1]
            for order in ("left first", "right first"):
                params.move_order = order
                pruned = list(AlphaBetaSearchDemo().search(data, params))[-1]
                self.assertEqual(full.result, reference(0))
                self.assertEqual(pruned.result, full.result)
                self.assertLessEqual(pruned.comparisons, full.comparisons)

    def test_monte_carlo_tree_search(self):
        from .monte_carlo_tree_search import MonteCarloTreeSearchDemo

        demo = MonteCarloTreeSearchDemo()
        params = Values(**{param.name: param.default for param in demo.get_params()})
        data = demo.make_data(params)
        first = list(demo.search(data, params))[-1]
        second = list(demo.search(data, params))[-1]
        self.assertEqual(first.result, second.result)
        self.assertEqual(first.path, second.path)
        self.assertEqual(first.comparisons, params.simulations)
        self.assertIn(first.path[1], data["children"][0])
        self.assertTrue(-20 <= first.result <= 20)

    def test_local_and_constraint_searches(self):
        classes, _ = discover()
        for cls in classes:
            if cls.category != "Searching algorithms" or cls.family not in ("local", "constraint", "combinatorial", "trie"):
                continue
            demo = cls()
            params = Values(**{param.name: param.default for param in demo.get_params()})
            data = demo.make_data(params)
            last = list(demo.search(data, params))[-1]
            self.assertTrue(last.done)
            if cls.family == "local":
                self.assertGreaterEqual(data["heights"][last.result], data["heights"][data["start"]])
            elif cls.family == "constraint":
                self.assertEqual(len(last.result), data["size"])
                for row, column in last.result:
                    for other_row, other_column in last.result:
                        if row != other_row:
                            self.assertNotEqual(column, other_column)
                            self.assertNotEqual(abs(row - other_row), abs(column - other_column))
            elif cls.family == "combinatorial":
                expected = 0
                for mask in range(1 << len(data["weights"])):
                    selected = [index for index in range(len(data["weights"])) if mask & (1 << index)]
                    if sum(data["weights"][selected]) <= data["capacity"]:
                        expected = max(expected, sum(data["values"][selected]))
                self.assertEqual(sum(data["values"][last.result]), expected)
            else:
                for mode in ("prefix", "exact"):
                    params.mode = mode
                    for query in ("", data["words"][0], data["words"][0][:2], "ZZZ"):
                        data["query"] = query
                        last = list(demo.search(data, params))[-1]
                        expected = any(word.startswith(query) for word in data["words"]) if mode == "prefix" else query in data["words"]
                        self.assertEqual(last.found, expected)

    def test_string_searches(self):
        classes, _ = discover()
        rng = np.random.default_rng(42)
        texts = ["", "A", "AAAAAA", "ABABABABA", "BANANA"]
        texts += ["".join(rng.choice(list("ABC"), 40)) for _ in range(10)]
        for cls in classes:
            if cls.category != "Searching algorithms" or cls.family != "string":
                continue
            demo = cls()
            params = Values(**{param.name: param.default for param in demo.get_params()})
            for text in texts:
                for pattern in ("", "A", "AA", "ABA", "NANA", "Z", text, text + "X"):
                    with self.subTest(algorithm=cls.name, text=text, pattern=pattern):
                        last = list(demo.search({"text": text, "pattern": pattern}, params))[-1]
                        expected = text.find(pattern)
                        self.assertEqual(last.found, expected >= 0)
                        self.assertEqual(last.result, expected if expected >= 0 else None)

    def test_spatial_searches(self):
        classes, _ = discover()
        for cls in classes:
            if cls.category != "Searching algorithms" or cls.family != "spatial":
                continue
            demo = cls()
            params = Values(**{param.name: param.default for param in demo.get_params()})
            for seed in range(5):
                params.seed = seed
                data = demo.make_data(params)
                last = list(demo.search(data, params))[-1]
                distances = np.linalg.norm(data["points"] - data["query"], axis=1)
                if "Nearest" in cls.name:
                    self.assertAlmostEqual(distances[last.result], distances.min())
                else:
                    self.assertEqual(set(last.result), set(np.flatnonzero(distances <= params.radius)))

    def test_graph_paths_and_costs(self):
        classes, _ = discover()
        optimal = {"Uniform-Cost / Dijkstra Search", "A* / Weighted A* Search", "Iterative Deepening A* (IDA*)"}
        shortest = {"Breadth-First Search (BFS)", "Bidirectional BFS", "Iterative Deepening DFS"}
        for cls in classes:
            if cls.category != "Searching algorithms" or cls.family != "graph":
                continue
            demo = cls()
            defaults = {param.name: param.default for param in demo.get_params()}
            params = Values(**dict(defaults, depth_limit=30, beam_width=128, max_expansions=10000))
            for weighted in (False, True):
                walls = np.zeros((5, 5), dtype=bool)
                walls[:4, 2] = True
                weights = np.random.default_rng(42).integers(1, 10, (5, 5)) if weighted else np.ones((5, 5), dtype=int)
                data = {"walls": walls, "weights": weights, "start": (0, 0), "goal": (4, 4)}
                heap, best = [(0, data["start"])], {data["start"]: 0}
                while heap:
                    cost, node = heapq.heappop(heap)
                    for neighbor, weight in demo.neighbors(data, node, params):
                        if cost + weight < best.get(neighbor, float("inf")):
                            best[neighbor] = cost + weight
                            heapq.heappush(heap, (cost + weight, neighbor))
                with self.subTest(algorithm=cls.name, weighted=weighted):
                    steps = list(demo.search(data, params))
                    last = steps[-1]
                    self.assertTrue(last.done)
                    self.assertTrue(last.found, last.message)
                    self.assertEqual(last.path[0], data["start"])
                    self.assertEqual(last.path[-1], data["goal"])
                    for previous, node in zip(last.path, last.path[1:]):
                        self.assertFalse(walls[node])
                        self.assertEqual(abs(previous[0] - node[0]) + abs(previous[1] - node[1]), 1)
                    self.assertEqual(last.cost, sum(int(weights[node]) for node in last.path[1:]))
                    if cls.name in optimal:
                        self.assertEqual(last.cost, best[data["goal"]])
                    if cls.name in shortest:
                        self.assertEqual(len(last.path) - 1, 8)
                    walls[4, 2] = True
                    last = list(demo.search(data, params))[-1]
                    self.assertFalse(last.found)

    def test_array_searches(self):
        classes, errors = discover()
        self.assertFalse(errors)
        for cls in classes:
            if cls.category != "Searching algorithms" or cls.family != "array":
                continue
            demo = cls()
            defaults = {param.name: param.default for param in demo.get_params()}
            policies = ("any", "first", "last") if "match" in defaults else (None,)
            for values in ([], [3], [3, 3], [1, 2, 2, 2, 4, 9], list(range(65))):
                for target in (-1, 0, 1, 2, 3, 9, 64, 100):
                    for policy in policies:
                        with self.subTest(algorithm=cls.name, values=values, target=target, policy=policy):
                            params = Values(**dict(defaults, match=policy))
                            steps = list(demo.search({"values": values, "target": target}, params))
                            last = steps[-1]
                            self.assertTrue(last.done)
                            self.assertEqual(last.found, target in values)
                            if last.found:
                                self.assertEqual(values[last.result], target)
                                if policy == "first":
                                    self.assertEqual(last.result, values.index(target))
                                if policy == "last":
                                    self.assertEqual(last.result, len(values) - 1 - values[::-1].index(target))


if __name__ == "__main__":
    unittest.main()