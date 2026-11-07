"""In-place binary heap construction and repeated root extraction."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class HeapSortDemo(SortingDemo):
    name = "Heap Sort"
    group = "Heap and tree sorting"
    order = 6
    complexity = "O(n log n) worst; O(n) heap construction"
    space = "O(1)"
    info = """## Heap sort
    Build a max-heap for ascending output (min-heap for descending), exchange its
    root with the last active slot, and repair the heap. Like repeatedly extracting
    the most urgent item from a priority scheduler. In-place, predictable, not stable.
    """ + SORT_INFO

    def sort(self, state, p):
        def sift(root, size):
            while 2 * root + 1 < size:
                child = 2 * root + 1
                if child + 1 < size and state.compare(state.values[child], state.values[child + 1]) < 0:
                    child += 1
                comparison = state.compare(state.values[root], state.values[child])
                yield state.event(f"Repair heap of size {size}", root, child)
                if comparison >= 0:
                    return
                state.swap(root, child)
                root = child

        size = len(state.values)
        for root in range(size // 2 - 1, -1, -1):
            yield from sift(root, size)
        for end in range(size - 1, 0, -1):
            state.swap(0, end)
            yield state.event(f"Extract heap root to {end}", 0, end)
            yield from sift(0, end)