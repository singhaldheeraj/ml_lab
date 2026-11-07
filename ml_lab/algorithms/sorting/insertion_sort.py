"""Stable insertion by shifting larger predecessors."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class InsertionSortDemo(SortingDemo):
    name = "Insertion Sort"
    order = 3
    complexity = "O(n^2) worst; O(n) best"
    space = "O(1)"
    stable = True
    info = """## Insertion sort
    Insert each new value into an already sorted prefix, shifting larger keys right.
    Like keeping a short sensor-event queue ordered as samples arrive.
    Especially effective on small or nearly sorted data. Equal keys retain their order.
    """ + SORT_INFO

    def sort(self, state, p):
        for index in range(1, len(state.values)):
            value, position = state.values[index], index
            while position > 0:
                comparison = state.compare(state.values[position - 1], value)
                yield state.event(f"Insert value from {index}", position - 1, position)
                if comparison <= 0:
                    break
                state.write(position, state.values[position - 1])
                position -= 1
            state.write(position, value)
            yield state.event(f"Insertion at {position}", position)