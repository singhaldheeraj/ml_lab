"""Right-to-left matching with Horspool bad-character shifts."""
from ml_lab.core.registry import register
from ._base import SearchStep
from ._string import StringSearchDemo, string_result


@register
class BoyerMooreHorspoolSearchDemo(StringSearchDemo):
    name = "Boyer-Moore-Horspool Search"
    order = 43
    complexity = "O(n*m) worst, often sublinear probes"
    space = "O(m) shift table"
    info = """## Boyer-Moore-Horspool
    Match right to left and skip alignments using the character at the window's end.
    This is Horspool's simplified Boyer-Moore variant, without a good-suffix table.
    Like skipping impossible packet-header positions using their last byte.
    ## Performance
    O(n*m) worst, often sublinear text probes; O(m) preprocessing and table space.
    Returns the first match. Repetitive alphabets can reduce the benefit of skips.
    """

    def search(self, data, p):
        text, pattern = data["text"], data["pattern"]
        length, count = len(pattern), 0
        if not length:
            yield string_result(0, 0, pattern)
            return
        shifts = {pattern[index]: length - index - 1 for index in range(length - 1)}
        start = 0
        while start <= len(text) - length:
            offset = length - 1
            while offset >= 0:
                count += 1
                yield SearchStep(start + offset, comparisons=count, stored=len(shifts), message=f"Right-to-left alignment {start}")
                if text[start + offset] != pattern[offset]:
                    break
                offset -= 1
            if offset < 0:
                yield string_result(start, count, pattern)
                return
            start += shifts.get(text[start + length - 1], length)
        yield string_result(None, count, pattern)