"""Bidirectional bubble passes shrink both ends of the active range."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class CocktailSortDemo(SortingDemo):
    name = "Cocktail Shaker Sort"
    order = 11
    complexity = "O(n^2) worst; O(n) best"
    space = "O(1)"
    stable = True
    info = """## Cocktail shaker sort
    Alternate forward and backward bubble passes. A small value near the end can move
    to the front in one backward pass. Like straightening a queue from both ends.
    Stops when a pass makes no swaps; adjacent strict comparisons preserve stability.
    """ + SORT_INFO

    def sort(self, state, p):
        low, high = 0, len(state.values) - 1
        while low < high:
            changed = False
            for index in range(low, high):
                if state.compare(state.values[index], state.values[index + 1]) > 0:
                    state.swap(index, index + 1)
                    changed = True
                yield state.event("Forward shaker pass", index, index + 1)
            if not changed:
                return
            high -= 1
            changed = False
            for index in range(high, low, -1):
                if state.compare(state.values[index - 1], state.values[index]) > 0:
                    state.swap(index - 1, index)
                    changed = True
                yield state.event("Backward shaker pass", index - 1, index)
            low += 1
            if not changed:
                return