"""String-search datasets and character-grid rendering."""
import numpy as np
from ml_lab.core.params import Choice, Int
from ._base import SearchDemo, SearchStep


class StringSearchDemo(SearchDemo):
    family = "string"
    group = "String and text search"

    def data_params(self):
        return [Int("text_length", 80, 16, 160, group="Data"),
                Int("pattern_length", 5, 1, 12, group="Data"),
                Choice("alphabet", "DNA", ("DNA", "binary", "letters"), group="Data"),
                Choice("target", "middle", ("first", "middle", "last", "absent"), group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        alphabet = {"DNA": "ACGT", "binary": "01", "letters": "SEARCHALGORITHM"}[p.alphabet]
        text = "".join(rng.choice(list(alphabet), p.text_length))
        start = {"first": 0, "middle": p.text_length // 2, "last": p.text_length - p.pattern_length,
                 "absent": 0}[p.target]
        pattern = text[start:start + p.pattern_length] if p.target != "absent" else "#" * p.pattern_length
        return {"text": text, "pattern": pattern}

    def _draw(self, data, step):
        self.ax.clear()
        columns = 32
        for index, character in enumerate(data["text"]):
            row, column = divmod(index, columns)
            color = "#16826c" if index in step.path else "#d64c4c" if index == step.current else "#e8eef2"
            self.ax.text(column, -row, character, ha="center", va="center", fontsize=10,
                         color="white" if color != "#e8eef2" else "#26343d",
                         bbox={"facecolor": color, "edgecolor": "white", "boxstyle": "square,pad=0.3"})
            if column == 0:
                self.ax.text(-1.3, -row, str(index), ha="right", va="center", fontsize=8)
        rows = max(1, (len(data["text"]) + columns - 1) // columns)
        self.ax.set(xlim=(-3, columns), ylim=(-rows, 1), title=f"{self.name}\nPattern: {data['pattern']!r}")
        self.ax.axis("off")


def string_result(index, count, pattern):
    return SearchStep(current=index, comparisons=count, done=True, found=index is not None, result=index,
                      path=[] if index is None else list(range(index, index + len(pattern))),
                      message="Pattern not found" if index is None else f"First match at character {index}")