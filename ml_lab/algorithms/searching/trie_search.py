"""Character-trie exact-word and prefix lookup."""
import numpy as np
from ml_lab.core.params import Choice, Int
from ml_lab.core.registry import register
from ._base import SearchDemo, SearchStep


@register
class TrieSearchDemo(SearchDemo):
    name = "Trie / Prefix Search"
    family = "trie"
    group = "String and text search"
    order = 44
    complexity = "O(total characters) build; O(query length) lookup"
    space = "O(total characters)"
    params = [Choice("mode", "prefix", ("prefix", "exact"))]
    info = """## Trie search
    Store shared word prefixes as character edges. Follow one edge per query character.
    Like decoding hierarchical command names in a device console.
    ## Performance
    O(total word characters) build and space, O(query length) lookup.
    Timing includes index build. Prefix mode accepts a query ending inside a word;
    exact mode requires a terminal marker. Probe counts are logical character lookups.
    """

    def data_params(self):
        return [Int("word_count", 24, 8, 64, group="Data"),
                Choice("query", "prefix", ("prefix", "word", "absent"), group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        words = sorted({"".join(rng.choice(list("ABCDE"), int(rng.integers(3, 8)))) for _ in range(p.word_count)})
        word = words[len(words) // 2]
        query = {"prefix": word[:2], "word": word, "absent": "ZZZ"}[p.query]
        return {"words": words, "query": query}

    def _draw(self, data, step):
        self.ax.clear()
        for index, word in enumerate(data["words"]):
            column, row = divmod(index, 16)
            color = "#16826c" if word.startswith(data["query"]) and step.found else "#374957"
            self.ax.text(column, -row, word, color=color, fontfamily="monospace", fontsize=10)
        self.ax.set(xlim=(-0.2, 4), ylim=(-17, 1), title=f"{self.name} | Query: {data['query']!r}")
        self.ax.axis("off")

    def search(self, data, p):
        root, nodes, count = {}, 1, 0
        for word in data["words"]:
            node = root
            for character in word:
                count += 1
                if character not in node:
                    node[character] = {}
                    nodes += 1
                node = node[character]
            node[None] = True
        node = root
        for index, character in enumerate(data["query"]):
            count += 1
            yield SearchStep(comparisons=count, stored=nodes, message=f"Query character {index}: {character}")
            if character not in node:
                yield SearchStep(done=True, comparisons=count, stored=nodes, message="Query not found")
                return
            node = node[character]
        found = p.mode == "prefix" or None in node
        yield SearchStep(done=True, found=found, result=data["query"] if found else None,
                         comparisons=count, stored=nodes, message="Query found" if found else "Prefix exists, exact word absent")