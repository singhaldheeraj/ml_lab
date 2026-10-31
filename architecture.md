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

## Design boundaries

- Algorithm math belongs in the algorithm module; reusable data, plotting, and parameter behavior belongs in `ml_lab/core/`.
- The GUI owns widget lifecycle and animation scheduling. Demos should not create their own Tk windows or block on input.
- Matplotlib axes should be created once in `setup_figure()` and updated during `run()`; do not create new axes for every frame.
- Add a dependency only when the feature needs it. The current model demos rely on NumPy rather than a heavyweight ML framework so their mechanics remain inspectable.