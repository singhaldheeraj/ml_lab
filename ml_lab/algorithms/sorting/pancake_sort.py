"""Sort by reversing prefixes to move the extreme item to each suffix."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class PancakeSortDemo(SortingDemo):
    name = "Pancake Sort"
    order = 16
    complexity = "O(n^2) comparisons and moves; O(n) prefix flips"
    space = "O(1)"
    info = """## Pancake sort
    Find the largest remaining key, flip its prefix to bring it to the front, then
    flip the active prefix to move it to the end. Like ordering a stack using only
    a tool that can reverse its top section. Prefix reversals are not stable.
    """ + SORT_INFO

    def sort(self, state, p):
        def flip(end):
            low = 0
            while low < end:
                state.swap(low, end)
                yield state.event("Reverse prefix", low, end)
                low += 1
                end -= 1

        for end in range(len(state.values) - 1, 0, -1):
            extreme = 0
            for index in range(1, end + 1):
                if state.compare(state.values[index], state.values[extreme]) > 0:
                    extreme = index
                yield state.event(f"Find extreme in [0, {end}]", index, extreme)
            if extreme != end:
                yield from flip(extreme)
                yield from flip(end)