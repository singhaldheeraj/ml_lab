"""Population-based genetic optimization on a discrete landscape."""
import numpy as np

from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register
from ._base import SearchStep
from ._local import LOCAL_INFO, LocalSearchDemo


@register
class GeneticSearchDemo(LocalSearchDemo):
    name = "Genetic Algorithm Search"
    order = 63
    params = [Int("generations", 80, 1, 300), Int("population_size", 30, 4, 100),
              Float("mutation_rate", 0.15, 0.0, 1.0, 0.05)]
    complexity = "O(generations * population size) objective evaluations"
    space = "O(population size + generations) including retained best trail"
    info = """## Genetic algorithm
    Each individual is a row/column coordinate. Three-way tournament selection
    favors higher scores; coordinate crossover combines two parents. Mutation
    replaces each coordinate independently with a random legal coordinate.
    Elitism preserves the best individual, including the configured start point.
    Like evolving calibration candidates by keeping good settings and mixing them.
    The seed makes the run reproducible; finite runs do not guarantee the optimum.
    """ + LOCAL_INFO

    def search(self, data, p):
        rng = np.random.default_rng(p.seed)
        shape = np.array(data["heights"].shape)
        population = rng.integers(0, shape, size=(p.population_size, 2))
        population[0] = data["start"]
        trail, count = [], 0
        for generation in range(p.generations + 1):
            scores = data["heights"][population[:, 0], population[:, 1]]
            count += len(population)
            best = tuple(int(value) for value in population[int(np.argmax(scores))])
            trail.append(best)
            yield SearchStep(current=best, frontier={tuple(map(int, node)) for node in population},
                             path=list(trail), comparisons=count,
                             stored=4 * len(population) + len(trail),
                             message=f"Generation {generation}; best score {data['heights'][best]:.5f}")
            if generation == p.generations:
                break
            contestants = rng.integers(len(population), size=(2, p.population_size, 3))
            winners = np.argmax(scores[contestants], axis=2)
            parent_indices = np.take_along_axis(contestants, winners[..., None], axis=2)[..., 0]
            parents = population[parent_indices]
            crossover = rng.random(population.shape) < 0.5
            offspring = np.where(crossover, parents[0], parents[1])
            mutations = rng.random(population.shape) < p.mutation_rate
            offspring = np.where(mutations, rng.integers(0, shape, size=population.shape), offspring)
            offspring[0] = best
            population = offspring
        yield self.result(data, best, count, trail, "Genetic search finished", 4 * len(population) + len(trail))

    def _draw(self, data, step):
        super()._draw(data, step)
        if step.frontier:
            self.ax.scatter([node[1] for node in step.frontier], [node[0] for node in step.frontier],
                            s=25, facecolors="none", edgecolors="#f5f5f5", label="Population")
            self.ax.legend(fontsize=8, loc="upper left")