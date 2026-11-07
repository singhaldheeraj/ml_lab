"""Three-way exact search on a sorted array."""
from ml_lab.core.registry import register
from ._base import SearchDemo, array_result, array_step


@register
class TernarySearchDemo(SearchDemo):
    name = "Ternary Search"
    group = "Array search"
    order = 7
    requires_sorted = True
    complexity = "O(log n)"
    space = "O(1)"
    info = """## Ternary exact search
    Split a sorted interval into three parts using two probes.
    This is exact lookup, not continuous unimodal-function optimization.
    ## Performance
    O(log n) time and O(1) space. More probes per round can make it slower than binary search.
    """

    def search(self, data, p):
        values, target = data["values"], data["target"]
        low, high, count = 0, len(values) - 1, 0
        while low <= high:
            first = low + (high - low) // 3
            second = high - (high - low) // 3
            for index in (first, second):
                count += 1
                yield array_step(index, count, frontier={low, high}, message=f"Probe third at {index}")
                if values[index] == target:
                    yield array_result(index, count)
                    return
            if target < values[first]:
                high = first - 1
            elif target > values[second]:
                low = second + 1
            else:
                low, high = first + 1, second - 1
        yield array_result(None, count)