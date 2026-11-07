"""Build a hash index, then perform an exact lookup."""
from ml_lab.core.registry import register
from ._base import SearchDemo, array_result, array_step


@register
class HashSearchDemo(SearchDemo):
    name = "Hash Table Search"
    group = "Indexed search"
    order = 9
    complexity = "O(n) build; O(1) expected lookup, O(n) worst"
    space = "O(n)"
    info = """## Hash table search
    Index values by key and retain the first position for duplicate values.
    Like a CAN-message-ID lookup table. Input need not be sorted.
    ## Performance
    Python dict supplies hashing and collision handling. O(n) build, O(1) expected
    lookup, O(n) worst-case lookup, O(n) space. Compute time includes index build.
    Probe counts show inserted records and one logical dictionary lookup,
    not internal hash-table collision probes.
    """

    def search(self, data, p):
        table = {}
        for index, value in enumerate(data["values"]):
            table.setdefault(value, index)
            yield array_step(index, index + 1, stored=len(table), message=f"Index key {value}")
        result = table.get(data["target"])
        step = array_result(result, len(data["values"]) + 1)
        step.stored = len(table)
        yield step