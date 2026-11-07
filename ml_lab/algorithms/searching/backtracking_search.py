"""N-Queens constraint search with backtracking and optional MRV ordering."""
import numpy as np
from ml_lab.core.params import Choice, Int
from ml_lab.core.registry import register
from ._base import SearchDemo, SearchStep


@register
class BacktrackingSearchDemo(SearchDemo):
    name = "Backtracking / N-Queens (CSP)"
    family = "constraint"
    group = "Constraint and combinatorial search"
    order = 80
    complexity = "O(n!) search-tree upper bound; extra MRV checks"
    space = "O(n^2) retained per-depth candidates"
    params = [Choice("row_order", "sequential", ("sequential", "MRV")),
              Choice("column_order", "left first", ("left first", "center first"))]
    info = """## Backtracking constraint search
    Place one queen per row without sharing a column or diagonal. Undo a placement
    when the remaining rows cannot be completed. Optional minimum-remaining-values
    (MRV) chooses the most constrained row first. Like assigning scarce resources
    while immediately rejecting incompatible assignments.
    ## Performance
    Exponential worst-case search; O(n) assignment plus O(n^2) candidate storage. Probes count candidate
    legality tests, including MRV checks. Solves the first valid board, not all boards.
    """

    def data_params(self):
        return [Int("board_size", 6, 4, 10, group="Data")]

    def make_data(self, p):
        return {"size": p.board_size}

    def _draw(self, data, step):
        self.ax.clear()
        size = data["size"]
        self.ax.imshow(np.indices((size, size)).sum(axis=0) % 2, cmap="Greys", vmin=-0.5, vmax=2)
        for row, column in step.path:
            self.ax.text(column, row, "Q", ha="center", va="center", fontsize=24, color="#16826c", fontweight="bold")
        self.ax.set_title(self.name)
        self.ax.set_xticks(range(size))
        self.ax.set_yticks(range(size))

    def search(self, data, p):
        size, assignment, count = data["size"], {}, 0
        columns = list(range(size))
        if p.column_order == "center first":
            columns.sort(key=lambda column: abs(column - (size - 1) / 2))

        def legal(row):
            nonlocal count
            allowed = []
            for column in columns:
                count += 1
                if all(column != other_column and abs(row - other_row) != abs(column - other_column)
                       for other_row, other_column in assignment.items()):
                    allowed.append(column)
            return allowed

        def solve():
            if len(assignment) == size:
                return True
            remaining = [row for row in range(size) if row not in assignment]
            if p.row_order == "MRV":
                domains = {row: legal(row) for row in remaining}
                row = min(remaining, key=lambda candidate: len(domains[candidate]))
                candidates = domains[row]
                del domains
            else:
                row, candidates = remaining[0], legal(remaining[0])
            for column in candidates:
                assignment[row] = column
                yield SearchStep((row, column), path=list(assignment.items()), comparisons=count,
                                 stored=len(assignment) + len(candidates), message=f"Place queen at row {row}, column {column}")
                if (yield from solve()):
                    return True
                del assignment[row]
                yield SearchStep(path=list(assignment.items()), comparisons=count, stored=len(assignment), message=f"Backtrack row {row}")
            return False

        found = yield from solve()
        result = sorted(assignment.items())
        yield SearchStep(done=True, found=found, result=result, path=result, comparisons=count, stored=len(assignment),
                         message="Valid N-Queens arrangement found" if found else "No arrangement exists")