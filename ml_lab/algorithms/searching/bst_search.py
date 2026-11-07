"""Balanced binary-search-tree construction and traversal."""
from ml_lab.core.registry import register
from ._base import SearchDemo, array_result, array_step


@register
class BSTSearchDemo(SearchDemo):
    name = "Binary Search Tree Lookup"
    group = "Indexed search"
    order = 10
    requires_sorted = True
    complexity = "O(n) build from sorted input; O(log n) lookup"
    space = "O(n) tree"
    info = """## Balanced binary search tree
    Build a balanced tree from sorted values, then follow left/right child links.
    The highlighted interval endpoints are child nodes, not array scan bounds.
    Like a balanced decision index for diagnostic codes.
    ## Performance
    O(n) construction from already sorted data, O(log n) lookup, O(n) tree space.
    Includes construction in timing. An unbalanced BST can degrade to O(n) lookup.
    """

    def search(self, data, p):
        values = data["values"]
        nodes = {}

        def build(low, high):
            if low > high:
                return None
            middle = (low + high) // 2
            nodes[middle] = (build(low, middle - 1), build(middle + 1, high))
            return middle

        current = build(0, len(values) - 1)
        count = 0
        while current is not None:
            count += 1
            children = nodes[current]
            yield array_step(current, count, stored=len(nodes),
                             frontier={child for child in children if child is not None},
                             message=f"Visit tree node {current}; index build included")
            if values[current] == data["target"]:
                yield array_result(current, count)
                return
            current = children[0 if data["target"] < values[current] else 1]
        yield array_result(None, count)