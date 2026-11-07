"""Instrumented native Python adaptive stable-sort baseline."""
from functools import cmp_to_key
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class PythonTimsortDemo(SortingDemo):
    name = "Python Adaptive Sort (Timsort)"
    group = "Native library baseline"
    order = 20
    complexity = "O(n log n) worst; O(n) on ordered input"
    space = "O(n) wrappers, output, and native merge buffers"
    stable = True
    info = """## Python's native adaptive stable sort
    Call Python's built-in sorted with an instrumented comparator. CPython uses an
    adaptive natural-merge sorting implementation from the Timsort family; recent
    versions use a Powersort merge policy. The native implementation is not duplicated here.
    ## Baseline limitations
    Shows before/after, not native internal animation. Key comparisons are measured;
    native swaps and internal buffer writes are unavailable. Writes report only copying
    the final result into the demo array, and aux items are an O(n) estimate.
    Comparator wrappers add Python overhead, so this is not raw built-in sort throughput.
    Open source code opens this adapter, not CPython's C implementation.
    """ + SORT_INFO

    def sort(self, state, p):
        state.auxiliary = 3 * len(state.values)
        yield state.event("Native baseline: input (internal steps are not exposed)")
        result = sorted(state.values, key=cmp_to_key(state.compare))
        for index, value in enumerate(result):
            state.write(index, value)
        yield state.event("Native stable sort complete; output copied into the demo array")