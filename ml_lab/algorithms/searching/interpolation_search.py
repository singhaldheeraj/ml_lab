"""Estimate an index from the numeric value range."""
from ml_lab.core.registry import register
from ._base import SearchDemo, array_result, array_step


@register
class InterpolationSearchDemo(SearchDemo):
    name = "Interpolation Search"
    group = "Array search"
    order = 4
    requires_sorted = True
    complexity = "O(log log n) expected uniform, O(n) worst"
    space = "O(1)"
    info = """## Interpolation search
    Estimate the target position by linear interpolation between sorted endpoints.
    Like estimating a register address from a nearly uniform calibration table.
    ## Performance
    O(log log n) expected on uniform data, O(n) worst case on skewed data.
    O(1) auxiliary space. Compare uniform, clustered, and duplicate-heavy data.
    """

    def search(self, data, p):
        values, target = data["values"], data["target"]
        low, high, count = 0, len(values) - 1, 0
        while low <= high and values[low] <= target <= values[high]:
            index = low if values[low] == values[high] else (
                low + (target - values[low]) * (high - low) // (values[high] - values[low]))
            count += 1
            yield array_step(index, count, frontier={low, high}, message=f"Estimated index {index}")
            if values[index] == target:
                yield array_result(index, count)
                return
            if values[index] < target:
                low = index + 1
            else:
                high = index - 1
        yield array_result(None, count)