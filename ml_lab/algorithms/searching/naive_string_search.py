"""Try every alignment and compare characters from left to right."""
from ml_lab.core.registry import register
from ._base import SearchStep
from ._string import StringSearchDemo, string_result


@register
class NaiveStringSearchDemo(StringSearchDemo):
    name = "Naive String Search"
    order = 40
    complexity = "O(n*m) worst"
    space = "O(1)"
    info = """## Naive substring search
    Try every pattern alignment, comparing characters left to right.
    Like recognizing a packet header at every offset in a byte stream.
    ## Performance
    O(n*m) worst-case character checks, O(1) auxiliary space.
    n is text length; m is pattern length. Returns the first match.
    """

    def search(self, data, p):
        text, pattern, count = data["text"], data["pattern"], 0
        for start in range(len(text) - len(pattern) + 1):
            for offset, character in enumerate(pattern):
                count += 1
                yield SearchStep(start + offset, comparisons=count, message=f"Alignment {start}, offset {offset}")
                if text[start + offset] != character:
                    break
            else:
                yield string_result(start, count, pattern)
                return
        yield string_result(None, count, pattern)