"""Stable integer-key counting sort with signed-key normalization."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo, integer_bounds


@register
class CountingSortDemo(SortingDemo):
    name = "Counting Sort"
    group = "Integer and distribution sorting"
    order = 8
    complexity = "O(n + k), k = key range"
    space = "O(n + k)"
    stable = True
    info = """## Counting sort
    Count each integer key, accumulate bucket endpoints, then place values stably
    into an output buffer. Signed keys are shifted; descending keys are mirrored.
    Like grouping events by a small integer priority field. Excellent for narrow key
    ranges; memory grows with the numeric range, not just the input length.
    """ + SORT_INFO

    def sort(self, state, p):
        if not state.values:
            return
        smallest, largest = integer_bounds(state)
        counts = [0] * (largest - smallest + 1)
        output = [None] * len(state.values)
        state.auxiliary = len(counts) + len(output)

        def key(value):
            return largest - value if state.descending else value - smallest

        for index, value in enumerate(state.values):
            counts[key(value)] += 1
            yield state.event(f"Count key {value}", index)
        for index in range(1, len(counts)):
            counts[index] += counts[index - 1]
        for value in reversed(state.values):
            bucket = key(value)
            counts[bucket] -= 1
            output[counts[bucket]] = value
        for index, value in enumerate(output):
            state.write(index, value)
            yield state.event(f"Write stable bucket output {index}", index)