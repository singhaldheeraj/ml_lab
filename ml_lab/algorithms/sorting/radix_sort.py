"""Stable least-significant-digit integer radix sort."""
from ml_lab.core.params import Choice
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo, integer_bounds


@register
class RadixSortDemo(SortingDemo):
    name = "Radix Sort (LSD)"
    group = "Integer and distribution sorting"
    order = 9
    complexity = "O(d*(n + b)), d = digits, b = radix"
    space = "O(n + b)"
    stable = True
    params = [Choice("radix", 10, (2, 10, 16))]
    info = """## LSD radix sort
    Make stable counting passes from the least significant digit to the most significant.
    Like ordering packet sequence numbers one byte at a time. Signed keys are shifted;
    descending keys are mirrored without reversing equal-key groups.
    Compare bases 2, 10 and 16. More buckets can reduce the number of digit passes.
    """ + SORT_INFO

    def sort(self, state, p):
        if not state.values:
            return
        smallest, largest = integer_bounds(state)
        maximum, place = largest - smallest, 1
        state.auxiliary = len(state.values) + p.radix
        while maximum // place:
            counts, output = [0] * p.radix, [None] * len(state.values)

            def digit(value):
                key = largest - value if state.descending else value - smallest
                return (key // place) % p.radix

            for index, value in enumerate(state.values):
                counts[digit(value)] += 1
                yield state.event(f"Count digit at place {place}", index)
            for index in range(1, p.radix):
                counts[index] += counts[index - 1]
            for value in reversed(state.values):
                bucket = digit(value)
                counts[bucket] -= 1
                output[counts[bucket]] = value
            for index, value in enumerate(output):
                state.write(index, value)
                yield state.event(f"Radix {p.radix}, place {place}: write {index}", index)
            place *= p.radix