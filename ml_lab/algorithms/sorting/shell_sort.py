"""Gapped insertion sort with halving or Knuth gaps."""
from ml_lab.core.params import Choice
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class ShellSortDemo(SortingDemo):
    name = "Shell Sort"
    order = 7
    complexity = "O(n^2) halving; O(n^(3/2)) Knuth worst bound"
    space = "O(1)"
    params = [Choice("gap_sequence", "Knuth", ("Knuth", "halving"))]
    info = """## Shell sort
    Insertion-sort interleaved subsequences at decreasing gaps, finishing with gap 1.
    Like correcting distant ordering errors before making a final local pass.
    Gap choice changes performance; long-distance moves mean this sort is not stable.
    """ + SORT_INFO

    def sort(self, state, p):
        size = len(state.values)
        gap = size // 2 if p.gap_sequence == "halving" else 1
        if p.gap_sequence == "Knuth":
            while gap < size // 3:
                gap = 3 * gap + 1
        while gap > 0:
            for index in range(gap, size):
                value, position = state.values[index], index
                while position >= gap:
                    comparison = state.compare(state.values[position - gap], value)
                    yield state.event(f"Gap {gap}; insert from {index}", position, position - gap)
                    if comparison <= 0:
                        break
                    state.write(position, state.values[position - gap])
                    position -= gap
                state.write(position, value)
                yield state.event(f"Gap {gap}; write {position}", position)
            gap = gap // 2 if p.gap_sequence == "halving" else (gap - 1) // 3