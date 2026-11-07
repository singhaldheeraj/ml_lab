"""Iterative Lomuto quicksort with configurable pivot selection."""
import random
from ml_lab.core.params import Choice
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class QuickSortDemo(SortingDemo):
    name = "Quick Sort"
    group = "Divide-and-conquer sorting"
    order = 5
    complexity = "O(n log n) expected; O(n^2) worst"
    space = "O(n) pending intervals upper bound"
    params = [Choice("pivot", "middle", ("first", "middle", "last", "random"))]
    info = """## Quick sort
    Partition around a pivot, then sort the two remaining intervals. Like splitting
    service jobs into those before and after a selected priority threshold.
    Compare pivot strategies on sorted and duplicate-heavy input. This two-way Lomuto
    partition can degrade to quadratic time, especially with many equal keys. Not stable.
    """ + SORT_INFO

    def sort(self, state, p):
        rng = random.Random(p.seed)
        stack = [(0, len(state.values) - 1)]
        while stack:
            state.auxiliary = len(stack)
            low, high = stack.pop()
            if low >= high:
                continue
            pivot_index = {"first": low, "middle": (low + high) // 2, "last": high}.get(p.pivot)
            if p.pivot == "random":
                pivot_index = rng.randint(low, high)
            state.swap(pivot_index, high)
            pivot, boundary = state.values[high], low
            for index in range(low, high):
                if state.compare(state.values[index], pivot) < 0:
                    state.swap(index, boundary)
                    boundary += 1
                yield state.event(f"Partition [{low}, {high}]; pivot {pivot}", index, high, boundary)
            state.swap(boundary, high)
            yield state.event(f"Pivot settled at {boundary}", boundary)
            stack.extend(((low, boundary - 1), (boundary + 1, high)))