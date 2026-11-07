"""Partition a sorted sequence with Fibonacci offsets."""
from ml_lab.core.registry import register
from ._base import SearchDemo, array_result, array_step


@register
class FibonacciSearchDemo(SearchDemo):
    name = "Fibonacci Search"
    group = "Array search"
    order = 6
    requires_sorted = True
    complexity = "O(log n)"
    space = "O(1)"
    info = """## Fibonacci search
    Divide sorted data using consecutive Fibonacci numbers rather than midpoints.
    Like binary search implemented with additions and subtractions.
    ## Performance
    O(log n) probes and O(1) auxiliary space. Input must be sorted.
    """

    def search(self, data, p):
        values, target = data["values"], data["target"]
        smaller, small, large = 0, 1, 1
        while large < len(values):
            smaller, small, large = small, large, small + large
        offset, count = -1, 0
        while large > 1:
            index = min(offset + smaller, len(values) - 1)
            if index < 0:
                break
            count += 1
            yield array_step(index, count, message=f"Fibonacci partition {large}")
            if values[index] < target:
                large, small, smaller = small, smaller, small - smaller
                offset = index
            elif values[index] > target:
                large, small, smaller = smaller, small - smaller, 2 * smaller - small
            else:
                yield array_result(index, count)
                return
        index = offset + 1
        if small and index < len(values):
            count += 1
            yield array_step(index, count, message="Check remaining element")
            if values[index] == target:
                yield array_result(index, count)
                return
        yield array_result(None, count)