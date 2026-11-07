"""Binary search with any, first, and last duplicate-match policies."""
from ml_lab.core.params import Choice
from ml_lab.core.registry import register
from ._base import SearchDemo, array_result, array_step


@register
class BinarySearchDemo(SearchDemo):
    name = "Binary Search"
    group = "Array search"
    order = 2
    requires_sorted = True
    complexity = "O(log n)"
    space = "O(1)"
    params = [Choice("match", "any", ("any", "first", "last"))]
    info = """## Binary search
    Halve a sorted interval after every probe. Like locating a calibration entry
    in a sorted lookup table. Sorted input is required.
    ## Performance
    O(log n) time and O(1) auxiliary space. Select first or last to resolve duplicates.
    Sorting costs O(n log n) and is dataset preparation, not search time.
    """

    def search(self, data, p):
        values, target = data["values"], data["target"]
        low, high, count, result = 0, len(values) - 1, 0, None
        while low <= high:
            middle = (low + high) // 2
            count += 1
            yield array_step(middle, count, frontier={low, high}, message=f"Interval [{low}, {high}]")
            if values[middle] == target:
                result = middle
                if p.match == "any":
                    break
                if p.match == "first":
                    high = middle - 1
                else:
                    low = middle + 1
            elif values[middle] < target:
                low = middle + 1
            else:
                high = middle - 1
        yield array_result(result, count)