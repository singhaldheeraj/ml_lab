"""Alternating disjoint adjacent-comparison phases."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class OddEvenSortDemo(SortingDemo):
    name = "Odd-Even Transposition Sort"
    group = "Exchange and network sorting"
    order = 14
    complexity = "O(n^2) sequential; O(n) best with early exit"
    space = "O(1)"
    stable = True
    info = """## Odd-even transposition sort
    Alternate disjoint pairs beginning at odd and even indices. Independent pairs
    could execute in parallel; this demo runs sequentially and counts all comparisons.
    Like neighboring processors exchanging out-of-order values in alternating phases.
    Repeats until neither phase swaps anything. Strict adjacent swaps preserve stability.
    """ + SORT_INFO

    def sort(self, state, p):
        changed = True
        while changed:
            changed = False
            for parity in (1, 0):
                for index in range(parity, len(state.values) - 1, 2):
                    if state.compare(state.values[index], state.values[index + 1]) > 0:
                        state.swap(index, index + 1)
                        changed = True
                    yield state.event(f"{'Odd' if parity else 'Even'} pair phase", index, index + 1)