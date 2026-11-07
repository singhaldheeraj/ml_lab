# ML Lab Architecture

ML Lab is a desktop learning playground built with Tkinter, Matplotlib, and NumPy. Each algorithm is a self-contained plugin: a small model implementation plus a demo that connects parameters, data, explanation, and visualization to the shared GUI.

## Runtime flow

1. `run_ml_lab.py` and `ml_lab.__main__` call `ml_lab.gui.app.main()`.
2. `ml_lab.core.registry.discover()` walks packages below `ml_lab.algorithms`, imports modules, and collects classes decorated with `@register`.
3. `ml_lab.gui.app.App` groups the registered classes by `category`. Each category becomes a top-level tab, so adding a new category creates a new tab automatically.
4. The selected algorithm is constructed lazily in `ml_lab.gui.algo_tab.AlgoTab`. The tab builds its controls from the demo's parameters and embeds a Matplotlib figure.
5. The GUI creates data with `make_data()`, prepares axes with `setup_figure()`, and advances the `run()` generator to animate the algorithm. The Explanation and Log panels are shared by all demos.

## Source layout

| Path | Responsibility |
| --- | --- |
| `ml_lab/algorithms/<category>/` | Algorithm implementations and registered demos; underscore-prefixed modules are ignored. |
| `ml_lab/core/base.py` | Shared demo contracts for classification, regression, clustering, and custom demos. |
| `ml_lab/core/datasets.py` | Deterministic NumPy toy data and train/test splitting. |
| `ml_lab/core/params.py` | Typed parameter descriptors used to build controls. |
| `ml_lab/core/plotting.py` | Shared Matplotlib styling, class regions, and drawing helpers. |
| `ml_lab/core/registry.py` | Plugin decorator and package discovery. |
| `ml_lab/gui/app.py` | Main window, category tabs, startup, and Home content. |
| `ml_lab/gui/algo_tab.py` | Per-algorithm controls, figure, animation, and explanation/log views. |
| `ml_lab/gui/search_tab.py` | Shared filterable, grouped, scrollable searching/sorting catalogs with lazy demo construction. |
| `ml_lab/algorithms/searching/` | One file per searching algorithm; shared trace and visualization modules are underscore-prefixed. |
| `ml_lab/algorithms/sorting/` | One file per sorting algorithm; shared state, counters, visualization, and tests are underscore-prefixed. |
| `ml_lab/gui/widgets.py` | Reusable Tk parameter panel, tooltips, and scrollable frame. |
| `ml_lab/selftest.py` | Headless smoke tests for every registered demo and parameter choice. |

## Adding an algorithm

1. Put the module in the closest package under `ml_lab/algorithms/`. A new directory must contain `__init__.py` so package discovery can walk it.
2. Keep the mathematical model independent from the GUI where practical. Use NumPy and make random behavior reproducible with a seed.
3. Define a demo class based on `ClassificationDemo`, `RegressionDemo`, `ClusteringDemo`, or `AlgorithmDemo` from `ml_lab.core.base`.
4. Declare controls with `Float`, `Int`, `Choice`, and `Bool`. Put data-regenerating controls in the `Data` group.
5. Decorate the demo with `@register`, give it a useful `name`, and set `category`, `group`, and `order` as needed. A previously unseen category appears as a new top-level tab.
6. Implement `run(data, p)` as a generator. Draw on the axes established by `setup_figure()` and yield a short status message for each animation frame. Override `setup_figure()` for multiple panels.
7. Add a concise `info` explanation and test the module with `python -m ml_lab.selftest <module-or-name>`, then run `python -m ml_lab.selftest` for the whole catalog.

For a minimal working example, copy the pattern in `ml_lab/algorithms/_TEMPLATE.py`. For examples of custom multi-panel visualizations, see the supervised decision-tree demo and `ml_lab/algorithms/reinforcement/gridworld.py`.

## Current categories

- **Supervised:** linear and polynomial regression, regularized regression, logistic regression, perceptron, KNN, Naive Bayes, LDA/QDA, SVM, MLP, trees, forests, AdaBoost, and gradient boosting.
- **Unsupervised:** K-Means, DBSCAN, mean shift, hierarchical clustering, Gaussian mixtures, self-organizing maps, spectral clustering, PCA, t-SNE, and anomaly detection.
- **Robotics Demos:** Monte Carlo Localization (particle filter), Kalman GPS/odometry fusion, DBSCAN LiDAR segmentation, Gaussian Naive Bayes terrain recognition, Bayesian occupancy-grid mapping, and Gaussian Process actuator calibration.
- **Reinforcement Learning:** Q-learning, SARSA, Expected SARSA, Double Q-learning, Monte Carlo control, Dyna-Q, value iteration, and policy iteration in a shared grid-world environment.
- **Searching algorithms:** 35 demos spanning array/index lookup, graph/path search, string matching, spatial queries, local optimization, adversarial game trees, and constraint/combinatorial search.
- **Sorting algorithms:** 20 demos spanning comparison, divide-and-conquer, heap/tree, distribution, adaptive, and sorting-network algorithms.

## Searching catalog

All implementations live in `ml_lab/algorithms/searching/`, one algorithm per module.
The category has a filterable grouped selector, animated probes, Run/Pause/Step/Finish,
parameters, an explanation, a performance panel, and the existing Open source code /
Reload code controls. Source opens the selected algorithm's own implementation file.

| Family | Algorithms |
| --- | --- |
| Array and index | Linear, binary (any/first/last match), jump, interpolation, exponential, Fibonacci, ternary, sentinel linear, hash table, balanced BST lookup |
| Graph and path | BFS, DFS, depth-limited, iterative deepening DFS, uniform-cost/Dijkstra, A*/weighted A*, greedy best-first, beam, bidirectional BFS, IDA* |
| String and text | Naive substring, KMP, Rabin-Karp, Boyer-Moore-Horspool, trie exact/prefix lookup |
| Spatial | K-d tree nearest neighbor, quadtree circular-range search |
| Local optimization | Hill climbing/random restarts, simulated annealing, tabu search |
| Adversarial | Minimax, alpha-beta pruning, Monte Carlo tree search/UCT |
| Constraints and combinatorial | N-Queens backtracking with optional MRV, branch-and-bound 0/1 knapsack |

This covers the major families, not every published algorithm or specialized variant.
Arrays are generated sorted to support cross-algorithm comparisons; sorting/data
generation is not timed. Index construction inside each search is timed.
Graphs use four-way movement, positive entry costs, configurable obstacles and terrain.
The default barrier map has a route; random maps may be disconnected.

Performance shows algorithm-specific probes, trace events, peak tracked items,
compute milliseconds, and time/space complexity. Compute time excludes drawing and
animation delays but includes trace instrumentation and index construction. It is an
educational single-run measurement, not a statistically rigorous benchmark. Tracked
items are estimates of logical storage, not bytes or process-memory measurements.
Graph edge checks, string character checks, local objective evaluations, game leaf
evaluations and other probe definitions are explained per demo; counts across unlike
families should not be compared directly. Depth/beam/iteration/expansion limits and
heuristic searches explicitly qualify incomplete or nonoptimal results.

Focused checks:

```console
python -m unittest ml_lab.algorithms.searching._tests -v
python -m ml_lab.selftest searching
python -m ml_lab.selftest searching --choices
```

## Sorting catalog

Each algorithm has a separate file in `ml_lab/algorithms/sorting/`. The category uses
the same grouped, filterable catalog and source viewing/reloading controls as searching.

| Family | Algorithms |
| --- | --- |
| Comparison | Bubble, selection, insertion, Shell, cocktail shaker, comb, gnome, cycle, pancake |
| Divide-and-conquer | Merge, quick (first/middle/last/random pivots), natural merge |
| Heap/tree | Heap, tournament |
| Integer/distribution | Counting, LSD radix (bases 2/10/16), bucket |
| Exchange/network | Odd-even transposition, bitonic network with power-of-two padding |
| Native baseline | Python's adaptive stable sort (Timsort family; recent CPython uses a Powersort merge policy) |

Controls include size, seed, ascending/descending order, events per animation frame,
and random/sorted/reversed/almost-sorted/duplicate-heavy/equal/signed integer inputs.
Algorithm-specific settings include early exit, pivot strategy, Shell gap sequence,
Comb shrink factor, radix, and bucket count. Set events per frame to 1 for individual
trace steps. The native baseline exposes only before/after, not native internal animation;
its source button opens the instrumented adapter rather than CPython's C implementation.

Performance reports key comparisons, swaps, main-array writes, trace events, compute
time, estimated auxiliary items, time/space complexity, and stability. A swap counts
as two main-array writes; auxiliary-buffer writes are excluded. Digit/bucket operations
are not key comparisons. Timings include instrumentation and snapshot overhead but
exclude plotting, animation delays, data creation, and final correctness verification.
Auxiliary counts exclude the input copy, display snapshots, and GUI memory. These are
single-run educational measurements, not statistically rigorous benchmarks. Native
internal moves/buffer sizes are not observable and are explicitly qualified.

Correctness tests compare every implementation to `sorted()` on empty, singleton,
duplicate, signed, ordered, reversed, random and non-power-of-two inputs in both
directions. Tagged equal keys verify each declared stability guarantee.

```console
python -m unittest ml_lab.algorithms.sorting._tests ml_lab.algorithms.searching._tests -v
python -m ml_lab.selftest sorting
python -m ml_lab.selftest sorting --choices
```

## Design boundaries

- Algorithm math belongs in the algorithm module; reusable data, plotting, and parameter behavior belongs in `ml_lab/core/`.
- The GUI owns widget lifecycle and animation scheduling. Demos should not create their own Tk windows or block on input.
- Matplotlib axes should be created once in `setup_figure()` and updated during `run()`; do not create new axes for every frame.
- Add a dependency only when the feature needs it. The current model demos rely on NumPy rather than a heavyweight ML framework so their mechanics remain inspectable.