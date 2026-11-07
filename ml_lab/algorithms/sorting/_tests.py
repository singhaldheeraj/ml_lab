"""Correctness and instrumentation tests for every sorting plugin."""
from collections import Counter
import inspect
import unittest

import numpy as np

from ml_lab.core.params import Values
from ml_lab.core.registry import discover
from ._base import SortState


class TaggedInt(int):
    def __new__(cls, value, tag):
        instance = super().__new__(cls, value)
        instance.tag = tag
        return instance


class SortingTests(unittest.TestCase):
    def classes(self):
        classes, errors = discover()
        self.assertFalse(errors)
        sorting = [cls for cls in classes if cls.category == "Sorting algorithms"]
        self.assertEqual(len(sorting), 20)
        return sorting

    def test_correctness_and_choices(self):
        rng = np.random.default_rng(42)
        samples = [[], [1], [2, 1], [3, 3, 3], [0, -3, 4, -3, 2, 0], list(range(17)), list(range(17))[::-1]]
        samples += [rng.integers(-20, 21, size).tolist() for size in (3, 7, 16, 23, 32)]
        for cls in self.classes():
            demo = cls()
            defaults = {param.name: param.default for param in demo.get_params()}
            variants = [defaults]
            for param in demo.params:
                if param.kind in ("choice", "bool"):
                    variants += [dict(defaults, **{param.name: value}) for value in (param.choices if param.kind == "choice" else (True, False))]
            for variant in variants:
                for descending in (False, True):
                    for values in samples:
                        with self.subTest(algorithm=cls.name, descending=descending, variant=variant, values=values):
                            state = SortState(values, descending)
                            for _ in demo.sort(state, Values(**variant)):
                                pass
                            self.assertEqual(state.values, sorted(values, reverse=descending))
                            self.assertEqual(Counter(state.values), Counter(values))

    def test_stability(self):
        for cls in self.classes():
            if not cls.stable:
                continue
            demo = cls()
            params = Values(**{param.name: param.default for param in demo.get_params()})
            values = [TaggedInt(value, index) for index, value in enumerate([3, 1, 2, 1, 3, 2, 0, 1])]
            for descending in (False, True):
                with self.subTest(algorithm=cls.name, descending=descending):
                    state = SortState(values, descending)
                    for _ in demo.sort(state, params):
                        pass
                    self.assertEqual([value.tag for value in state.values],
                                     [value.tag for value in sorted(values, reverse=descending)])

    def test_individual_source_files(self):
        classes = self.classes()
        paths = [inspect.getsourcefile(cls) for cls in classes]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all("sorting" in path for path in paths))

    def test_counters(self):
        state = SortState([3, 1])
        self.assertEqual(state.compare(3, 1), 1)
        state.swap(0, 1)
        state.write(0, 1)
        self.assertEqual((state.comparisons, state.swaps, state.writes), (1, 1, 3))


if __name__ == "__main__":
    unittest.main()