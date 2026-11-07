"""Winner-tree extraction with stable original-index tie ordering."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class TournamentSortDemo(SortingDemo):
    name = "Tournament Sort"
    group = "Heap and tree sorting"
    order = 19
    complexity = "O(n log n); O(n) initial tournament"
    space = "O(n)"
    stable = True
    info = """## Tournament sort
    Build a tree of pairwise winners, output the root winner, then replay only its
    ancestor matches. Like a knockout competition that repeatedly selects the next best.
    Leaves retain original input positions. Equal keys choose the earlier leaf,
    preserving stability. Unlike in-place heapsort, this implementation stores an index tree.
    """ + SORT_INFO

    def sort(self, state, p):
        original = state.values[:]
        if not original:
            return
        leaves = 1
        while leaves < len(original):
            leaves *= 2
        tree = [None] * (2 * leaves)
        state.auxiliary = len(original) + len(tree)
        for index in range(len(original)):
            tree[leaves + index] = index

        def winner(first, second):
            if first is None:
                return second
            if second is None:
                return first
            return first if state.compare(original[first], original[second]) <= 0 else second

        for node in range(leaves - 1, 0, -1):
            tree[node] = winner(tree[2 * node], tree[2 * node + 1])
            yield state.event(f"Build tournament node {node}")
        for output in range(len(original)):
            champion = tree[1]
            state.write(output, original[champion])
            yield state.event(f"Output winner from original position {champion}", output)
            node = leaves + champion
            tree[node] = None
            node //= 2
            while node:
                tree[node] = winner(tree[2 * node], tree[2 * node + 1])
                yield state.event(f"Replay tournament node {node}", output)
                node //= 2