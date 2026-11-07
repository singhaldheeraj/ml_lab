"""Find an exponential bound and then binary-search it."""
from ml_lab.core.registry import register
from ._base import SearchDemo, array_result, array_step


@register
class ExponentialSearchDemo(SearchDemo):
    name = "Exponential Search"
    group = "Array search"
    order = 5
    requires_sorted = True
    complexity = "O(log(i + 1)), O(log n) worst"
    space = "O(1)"
    info = """## Exponential search
    Probe indices 1, 2, 4, 8... to bracket the target, then binary search.
    Sorted input is required. Useful when the target is near the beginning,
    or when an accessible sequence's upper bound is not known in advance.
    ## Performance
    O(log(i + 1)) probes for a match at index i; O(log n) worst case, O(1) space.
    """

    def search(self, data, p):
        values, target = data["values"], data["target"]
        if not values:
            yield array_result(None, 0)
            return
        count = 1
        yield array_step(0, count, message="Inspect first value")
        if values[0] == target:
            yield array_result(0, count)
            return
        bound = 1
        while bound < len(values):
            count += 1
            yield array_step(bound, count, message=f"Exponential bound {bound}")
            if values[bound] >= target:
                break
            bound *= 2
        low, high = bound // 2, min(bound, len(values) - 1)
        while low <= high:
            middle = (low + high) // 2
            count += 1
            yield array_step(middle, count, frontier={low, high}, message="Binary refinement")
            if values[middle] == target:
                yield array_result(middle, count)
                return
            if values[middle] < target:
                low = middle + 1
            else:
                high = middle - 1
        yield array_result(None, count)