"""Numeric range buckets with stable insertion sort inside each bucket."""
from ml_lab.core.params import Int
from ml_lab.core.registry import register
from ._base import SORT_INFO, SortingDemo, integer_bounds


@register
class BucketSortDemo(SortingDemo):
    name = "Bucket Sort"
    group = "Integer and distribution sorting"
    order = 10
    complexity = "O(n + k) expected uniform; O(n^2 + k) worst"
    space = "O(n + k)"
    stable = True
    params = [Int("bucket_count", 8, 2, 32)]
    info = """## Bucket sort
    Distribute values across equal-width numeric intervals, insertion-sort each bucket,
    then concatenate. Like partitioning timestamps into time windows before local sorting.
    Uniform keys spread the work; clusters can concentrate it in one quadratic bucket.
    Uses stable insertion and preserves equal-key order in both output directions.
    """ + SORT_INFO

    def sort(self, state, p):
        if not state.values:
            return
        smallest, largest = integer_bounds(state)
        buckets = [[] for _ in range(p.bucket_count)]
        span = largest - smallest + 1
        state.auxiliary = len(state.values) + p.bucket_count
        for index, value in enumerate(state.values):
            bucket = min(p.bucket_count - 1, (value - smallest) * p.bucket_count // span)
            buckets[bucket].append(value)
            yield state.event(f"Distribute value to bucket {bucket}", index)
        output = 0
        order = range(p.bucket_count - 1, -1, -1) if state.descending else range(p.bucket_count)
        for bucket_index in order:
            bucket = buckets[bucket_index]
            for index in range(1, len(bucket)):
                value, position = bucket[index], index
                while position > 0:
                    comparison = state.compare(bucket[position - 1], value)
                    yield state.event(f"Insertion sort inside bucket {bucket_index}")
                    if comparison <= 0:
                        break
                    bucket[position] = bucket[position - 1]
                    position -= 1
                bucket[position] = value
            for value in bucket:
                state.write(output, value)
                yield state.event(f"Write bucket {bucket_index}", output)
                output += 1