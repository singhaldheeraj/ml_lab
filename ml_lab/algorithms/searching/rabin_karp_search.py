"""Rolling hash candidate selection with collision-safe verification."""
from ml_lab.core.params import Choice
from ml_lab.core.registry import register
from ._base import SearchStep
from ._string import StringSearchDemo, string_result


@register
class RabinKarpSearchDemo(StringSearchDemo):
    name = "Rabin-Karp Search"
    order = 42
    complexity = "O(n + m) expected, O(n*m) worst"
    space = "O(1) rolling hash"
    params = [Choice("modulus", 101, (7, 101, 1009))]
    info = """## Rabin-Karp
    Slide a rolling hash over the text. Verify every hash match character by character,
    so collisions never produce false matches. Like a checksum-assisted packet scan.
    ## Performance
    O(n + m) expected, O(n*m) worst; O(1) hash state. Choose a small modulus to see
    more collisions. Probes include hash-window checks and verification characters.
    """

    def search(self, data, p):
        text, pattern = data["text"], data["pattern"]
        length, count, base = len(pattern), 0, 256
        if length == 0:
            yield string_result(0, 0, pattern)
            return
        if length > len(text):
            yield string_result(None, 0, pattern)
            return
        pattern_hash, window_hash = 0, 0
        for index in range(length):
            pattern_hash = (base * pattern_hash + ord(pattern[index])) % p.modulus
            window_hash = (base * window_hash + ord(text[index])) % p.modulus
        factor = pow(base, length - 1, p.modulus)
        for start in range(len(text) - length + 1):
            count += 1
            yield SearchStep(start, comparisons=count, message=f"Window hash {window_hash}; pattern hash {pattern_hash}")
            if window_hash == pattern_hash:
                for offset in range(length):
                    count += 1
                    yield SearchStep(start + offset, comparisons=count, message="Verify hash candidate")
                    if text[start + offset] != pattern[offset]:
                        break
                else:
                    yield string_result(start, count, pattern)
                    return
            if start + length < len(text):
                window_hash = (base * (window_hash - ord(text[start]) * factor) + ord(text[start + length])) % p.modulus
        yield string_result(None, count, pattern)