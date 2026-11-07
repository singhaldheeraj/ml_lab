"""Block jumping followed by a sequential scan."""
from math import isqrt
from ml_lab.core.params import Int
from ml_lab.core.registry import register
from ._base import SearchDemo, array_result, array_step


@register
class JumpSearchDemo(SearchDemo):
    name = "Jump Search"
    group = "Array search"
    order = 3
    requires_sorted = True
    complexity = "O(n / block + block), O(sqrt(n)) with auto block"
    space = "O(1)"
    params = [Int("block_size", 0, 0, 32, help="0 selects sqrt(n).")]
    info = """## Jump search
    Probe block endpoints, then scan the first block that can contain the target.
    Requires sorted input. Like skipping pages in a sorted service log.
    ## Performance
    A square-root block balances endpoint probes and the final scan: O(sqrt(n)).
    Try very small or large blocks to see the tradeoff. O(1) auxiliary space.
    """

    def search(self, data, p):
        values, target = data["values"], data["target"]
        block = p.block_size or max(1, isqrt(len(values)))
        start, count = 0, 0
        while start < len(values):
            end = min(start + block, len(values)) - 1
            count += 1
            yield array_step(end, count, message=f"Probe block [{start}, {end}]")
            if values[end] >= target:
                for index in range(start, end + 1):
                    count += 1
                    yield array_step(index, count, message=f"Scan index {index}")
                    if values[index] == target:
                        yield array_result(index, count)
                        return
                    if values[index] > target:
                        break
                break
            start += block
        yield array_result(None, count)