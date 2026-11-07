"""Select the smallest remaining key for each output position."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class SelectionSortDemo(SortingDemo):
    name = "Selection Sort"
    order = 2
    complexity = "O(n^2) all cases"
    space = "O(1)"
    info = """## Selection sort
    Scan the unsorted suffix for its best key, then swap it into the next position.
    Like selecting the next highest-priority job from an unsorted queue.
    Performs few swaps but always quadratic comparisons; long-distance swaps are not stable.
    """ + SORT_INFO

    def sort(self, state, p):
        for start in range(len(state.values) - 1):
            best = start
            for index in range(start + 1, len(state.values)):
                if state.compare(state.values[index], state.values[best]) < 0:
                    best = index
                yield state.event(f"Select position {start}; candidate {best}", start, index, best)
            state.swap(start, best)
            yield state.event(f"Settle position {start}", start, best)