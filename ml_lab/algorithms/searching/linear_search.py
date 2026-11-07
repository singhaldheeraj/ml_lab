"""Sequential scan: no ordering or index is required."""
from ml_lab.core.registry import register
from ._base import SearchDemo, array_result, array_step


@register
class LinearSearchDemo(SearchDemo):
    name = "Linear Search"
    group = "Array search"
    order = 1
    complexity = "O(n) worst, O(1) best"
    space = "O(1)"
    info = """## Linear search
    Inspect each value until the target is found. Works on unsorted data too.
    Like checking sensor records one at a time without an index.
    ## Performance
    O(n) worst-case comparisons; O(1) auxiliary space. Returns the first match.
    Change size, target location, duplicates, and seed to compare workloads.
    The visualization uses sorted values for comparison with ordered searches.
    """

    def search(self, data, p):
        for index, value in enumerate(data["values"]):
            yield array_step(index, index + 1, message=f"Inspect index {index}")
            if value == data["target"]:
                yield array_result(index, index + 1)
                return
        yield array_result(None, len(data["values"]))