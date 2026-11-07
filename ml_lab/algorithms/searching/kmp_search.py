"""Knuth-Morris-Pratt search with prefix-function preprocessing."""
from ml_lab.core.registry import register
from ._base import SearchStep
from ._string import StringSearchDemo, string_result


@register
class KMPSearchDemo(StringSearchDemo):
    name = "Knuth-Morris-Pratt (KMP)"
    order = 41
    complexity = "O(n + m)"
    space = "O(m)"
    info = """## Knuth-Morris-Pratt
    Build a longest-prefix/suffix table, then reuse matched prefixes after mismatches.
    Like a packet parser that remembers partial synchronization instead of restarting.
    ## Performance
    O(n + m) time, O(m) auxiliary space. Timing and checks include prefix preprocessing.
    Compare repetitive binary text with naive matching. Returns the first match.
    """

    def search(self, data, p):
        text, pattern, count = data["text"], data["pattern"], 0
        if not pattern:
            yield string_result(0, 0, pattern)
            return
        prefix = [0] * len(pattern)
        matched, index = 0, 1
        while index < len(pattern):
            count += 1
            yield SearchStep(comparisons=count, stored=len(prefix), message=f"Build prefix table: pattern index {index}")
            if pattern[index] == pattern[matched]:
                matched += 1
                prefix[index] = matched
                index += 1
            elif matched:
                matched = prefix[matched - 1]
            else:
                index += 1
        index, matched = 0, 0
        while index < len(text):
            count += 1
            yield SearchStep(index, comparisons=count, stored=len(prefix), message=f"Matched prefix length {matched}")
            if text[index] == pattern[matched]:
                index += 1
                matched += 1
                if matched == len(pattern):
                    yield string_result(index - matched, count, pattern)
                    return
            elif matched:
                matched = prefix[matched - 1]
            else:
                index += 1
        yield string_result(None, count, pattern)