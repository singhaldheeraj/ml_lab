"""Linear search using an appended sentinel."""
from ml_lab.core.registry import register
from ._base import SearchDemo, array_result, array_step


@register
class SentinelSearchDemo(SearchDemo):
    name = "Sentinel Linear Search"
    group = "Array search"
    order = 8
    complexity = "O(n)"
    space = "O(n) safe copy; O(1) with mutable input"
    info = """## Sentinel search
    Append the target as a guard, removing the per-iteration bounds test.
    The guard is not a real match. Original data is preserved using a copy.
    ## Performance
    O(n) time. This safe implementation uses O(n) copy space; an in-place
    implementation can use O(1) auxiliary space. Works on unsorted input.
    """

    def search(self, data, p):
        values = data["values"] + [data["target"]]
        index = 0
        while True:
            yield array_step(index, index + 1, stored=len(values), message=f"Inspect {index} (last is sentinel)")
            if values[index] == data["target"]:
                break
            index += 1
        yield array_result(index if index < len(data["values"]) else None, index + 1)