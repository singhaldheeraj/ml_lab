"""Iterative Batcher bitonic sorting network with finite padding."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo, integer_bounds


@register
class BitonicSortDemo(SortingDemo):
    name = "Bitonic Sorting Network"
    group = "Exchange and network sorting"
    order = 17
    complexity = "O(n log^2 n) sequential work"
    space = "O(n) padding worst case"
    info = """## Bitonic sort
    Apply a fixed compare-exchange network, first constructing bitonic runs and then
    merging them. Like a hardware sorting network with predictable comparator wiring.
    This sequential demo counts every comparator; parallel circuit depth is O(log^2 n).
    Non-power-of-two inputs are padded with extreme integer keys, shown as temporary
    extra bars and removed afterward. Swaps may involve padding. Not stable.
    """ + SORT_INFO

    def sort(self, state, p):
        original_size = len(state.values)
        if original_size < 2:
            return
        size = 1
        while size < original_size:
            size *= 2
        smallest, largest = integer_bounds(state)
        padding = smallest - 1 if state.descending else largest + 1
        state.values.extend([padding] * (size - original_size))
        state.auxiliary = size - original_size
        state.writes += size - original_size
        width = 2
        while width <= size:
            distance = width // 2
            while distance:
                for index in range(size):
                    partner = index ^ distance
                    if partner <= index:
                        continue
                    comparison = state.compare(state.values[index], state.values[partner])
                    forward = (index & width) == 0
                    if (forward and comparison > 0) or (not forward and comparison < 0):
                        state.swap(index, partner)
                    yield state.event(f"Network width {width}, distance {distance}", index, partner)
                distance //= 2
            width *= 2
        del state.values[original_size:]
        yield state.event("Remove network padding")