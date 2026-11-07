"""Rotate permutation cycles to limit main-array writes."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class CycleSortDemo(SortingDemo):
    name = "Cycle Sort"
    order = 15
    complexity = "O(n^2) comparisons; O(n) writes"
    space = "O(1)"
    info = """## Cycle sort
    Count how many keys precede an item to find its destination, then rotate the
    displaced items until the cycle closes. Skip duplicate destinations safely.
    Useful when nonvolatile-memory writes are expensive. Trades quadratic comparisons
    for few main-array writes. Cyclic long-distance movements are not stable.
    """ + SORT_INFO

    def sort(self, state, p):
        size = len(state.values)
        for start in range(size - 1):
            item = state.values[start]
            position = start
            for index in range(start + 1, size):
                if state.compare(state.values[index], item) < 0:
                    position += 1
                yield state.event(f"Find destination for cycle at {start}", start, index)
            if position == start:
                continue
            while state.compare(item, state.values[position]) == 0:
                position += 1
                yield state.event("Skip duplicate destination", position)
            displaced = state.values[position]
            state.write(position, item)
            item = displaced
            yield state.event("Rotate cycle", position)
            while position != start:
                position = start
                for index in range(start + 1, size):
                    if state.compare(state.values[index], item) < 0:
                        position += 1
                    yield state.event("Find next cycle destination", index)
                while state.compare(item, state.values[position]) == 0:
                    position += 1
                    yield state.event("Skip duplicate destination", position)
                displaced = state.values[position]
                state.write(position, item)
                item = displaced
                yield state.event("Rotate cycle", position)