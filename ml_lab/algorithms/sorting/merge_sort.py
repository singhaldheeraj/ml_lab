"""Bottom-up stable merge sort with one reusable buffer."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class MergeSortDemo(SortingDemo):
    name = "Merge Sort"
    group = "Divide-and-conquer sorting"
    order = 4
    complexity = "O(n log n) all cases"
    space = "O(n)"
    stable = True
    info = """## Merge sort
    Merge adjacent sorted runs of lengths 1, 2, 4, 8... using a reusable output buffer.
    Like combining timestamp-ordered logs from two devices. Equal keys from the left
    run are taken first, preserving stability. Predictable O(n log n) time needs O(n) memory.
    """ + SORT_INFO

    def sort(self, state, p):
        size, width = len(state.values), 1
        buffer = [None] * size
        state.auxiliary = size
        while width < size:
            for low in range(0, size, width * 2):
                middle, high = min(low + width, size), min(low + 2 * width, size)
                left, right = low, middle
                for output in range(low, high):
                    if left < middle and (right >= high or state.compare(state.values[left], state.values[right]) <= 0):
                        buffer[output] = state.values[left]
                        left += 1
                    else:
                        buffer[output] = state.values[right]
                        right += 1
                    yield state.event(f"Merge runs [{low}, {middle}) and [{middle}, {high})", left - 1, right - 1)
                for output in range(low, high):
                    state.write(output, buffer[output])
                    yield state.event(f"Write merged position {output}", output)
            width *= 2