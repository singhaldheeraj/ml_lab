"""Adaptive merging of existing nondecreasing runs."""
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo


@register
class NaturalMergeSortDemo(SortingDemo):
    name = "Natural Merge Sort"
    group = "Divide-and-conquer sorting"
    order = 18
    complexity = "O(n log r), r = initial runs; O(n) for one run"
    space = "O(n)"
    stable = True
    info = """## Natural merge sort
    Detect existing ordered runs, then repeatedly merge neighboring runs.
    Like combining batches of already timestamp-sorted telemetry.
    Adapts to presorted input. This is a simple natural merge sort, not a reimplementation
    of Timsort's run extension, galloping, or native merge policy.
    """ + SORT_INFO

    def sort(self, state, p):
        size, runs, start = len(state.values), [], 0
        while start < size:
            end = start + 1
            while end < size:
                comparison = state.compare(state.values[end - 1], state.values[end])
                state.auxiliary = len(runs)
                yield state.event(f"Detect ordered run beginning at {start}", end - 1, end)
                if comparison > 0:
                    break
                end += 1
            runs.append((start, end))
            start = end
        buffer = [None] * size
        while len(runs) > 1:
            next_runs = []
            for run_index in range(0, len(runs), 2):
                state.auxiliary = size + 2 * len(runs) + 2 * len(next_runs)
                if run_index + 1 == len(runs):
                    next_runs.append(runs[run_index])
                    continue
                low, middle = runs[run_index]
                _, high = runs[run_index + 1]
                left, right = low, middle
                for output in range(low, high):
                    if left < middle and (right >= high or state.compare(state.values[left], state.values[right]) <= 0):
                        buffer[output] = state.values[left]
                        left += 1
                    else:
                        buffer[output] = state.values[right]
                        right += 1
                    yield state.event(f"Merge natural runs [{low}, {middle}) and [{middle}, {high})", left - 1, right - 1)
                for output in range(low, high):
                    state.write(output, buffer[output])
                    yield state.event("Write natural merge result", output)
                next_runs.append((low, high))
            runs = next_runs