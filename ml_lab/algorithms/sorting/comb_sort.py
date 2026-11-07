"""Shrinking-gap exchanges followed by adjacent correction."""
from ml_lab.core.params import Float
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class CombSortDemo(SortingDemo):
    name = "Comb Sort"
    order = 12
    complexity = "O(n^2) worst"
    space = "O(1)"
    params = [Float("shrink_factor", 1.3, 1.1, 2.0, 0.1)]
    info = """## Comb sort
    Compare keys separated by a shrinking gap, then repeat gap-1 passes until settled.
    Like moving badly misplaced events closer to their destinations before local cleanup.
    The shrink factor changes the gap schedule. Long-distance swaps make it unstable.
    """ + SORT_INFO

    def sort(self, state, p):
        gap, changed = len(state.values), True
        while gap > 1 or changed:
            gap = max(1, int(gap / p.shrink_factor))
            changed = False
            for index in range(len(state.values) - gap):
                if state.compare(state.values[index], state.values[index + gap]) > 0:
                    state.swap(index, index + gap)
                    changed = True
                yield state.event(f"Comb gap {gap}", index, index + gap)