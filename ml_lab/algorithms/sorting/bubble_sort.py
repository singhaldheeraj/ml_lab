"""Adjacent-swap sorting with an early-exit option."""
from ml_lab.core.params import Bool
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class BubbleSortDemo(SortingDemo):
    name = "Bubble Sort"
    order = 1
    complexity = "O(n^2) worst; O(n) best with early exit"
    space = "O(1)"
    stable = True
    params = [Bool("early_exit", True)]
    info = """## Bubble sort
    Exchange adjacent out-of-order values. Each pass settles the next largest value
    at the end. Like repeatedly correcting adjacent timestamp inversions in a log.
    Early exit detects an already sorted pass. Equal keys are never exchanged.
    """ + SORT_INFO

    def sort(self, state, p):
        for end in range(len(state.values) - 1, 0, -1):
            changed = False
            for index in range(end):
                if state.compare(state.values[index], state.values[index + 1]) > 0:
                    state.swap(index, index + 1)
                    changed = True
                yield state.event(f"Bubble pass ending at {end}", index, index + 1)
            if p.early_exit and not changed:
                return