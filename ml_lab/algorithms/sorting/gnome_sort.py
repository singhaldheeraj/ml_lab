"""Walk forward through ordered neighbors and back through inversions."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class GnomeSortDemo(SortingDemo):
    name = "Gnome Sort"
    order = 13
    complexity = "O(n^2) worst; O(n) best"
    space = "O(1)"
    stable = True
    info = """## Gnome sort
    Step forward when adjacent keys are ordered. Otherwise swap and step backward.
    Like a technician checking a row of ordered parts and correcting each inversion.
    Similar to insertion sort, but uses adjacent swaps. Equal keys are never exchanged.
    """ + SORT_INFO

    def sort(self, state, p):
        index = 1
        while index < len(state.values):
            if index == 0:
                index = 1
                continue
            if state.compare(state.values[index - 1], state.values[index]) <= 0:
                yield state.event("Step forward", index - 1, index)
                index += 1
            else:
                state.swap(index - 1, index)
                yield state.event("Swap and step backward", index - 1, index)
                index -= 1