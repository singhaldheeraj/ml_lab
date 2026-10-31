"""Small tabular grid-world demos for foundational reinforcement learning."""
import numpy as np
from matplotlib.patches import Rectangle

from ml_lab.core import plotting as P
from ml_lab.core.base import AlgorithmDemo
from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register

ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARROWS = ("↑", "↓", "←", "→")


class GridWorld:
    def __init__(self, size):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        barrier_row = size // 2
        self.walls = {(barrier_row, col) for col in range(1, size - 1)}

    def step(self, state, action):
        dr, dc = ACTIONS[action]
        row = min(max(state[0] + dr, 0), self.size - 1)
        col = min(max(state[1] + dc, 0), self.size - 1)
        next_state = (row, col)
        if next_state in self.walls:
            next_state = state
        done = next_state == self.goal
        return next_state, 1.0 if done else -0.02, done


class _GridWorldDemo(AlgorithmDemo):
    category = "Reinforcement Learning"
    group = "Tabular control"
    params = []
    info = """
    ## Grid world
    Move an agent from the start to the goal. Each move has a small cost, reaching the
    goal earns a reward, and the dark cells are obstacles. The agent learns from reward
    feedback rather than labeled examples.

    ## Reading the display
    Arrows show the current greedy action. The colored cells show the value estimate,
    the highlighted route is the current greedy path, and the lower plot tracks returns.
    """

    def data_params(self):
        return [Int("grid_size", 5, 4, 8, group="Data"),
                Int("seed", 0, 0, 999, group="Data")]

    def make_data(self, p):
        return {"world": GridWorld(p.grid_size), "seed": p.seed}

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        self.ax = fig.add_subplot(1, 2, 1)
        self.ax_rewards = fig.add_subplot(1, 2, 2)

    def draw_data(self, data, p):
        size = data["world"].size
        self._draw(data["world"], np.zeros((size, size, len(ACTIONS))), [], [], "Untrained policy")

    @staticmethod
    def _greedy_action(q_values, state, rng):
        values = q_values[state]
        best = np.flatnonzero(np.isclose(values, values.max()))
        return int(rng.choice(best))

    def _greedy_path(self, world, q_values):
        path = [world.start]
        state = world.start
        visited = {state}
        for _ in range(world.size * world.size * 2):
            action = int(np.argmax(q_values[state]))
            state, _, done = world.step(state, action)
            path.append(state)
            if done or state in visited:
                break
            visited.add(state)
        return path

    def _draw(self, world, q_values, returns, path, title):
        size = world.size
        values = np.max(q_values, axis=2)
        self.ax.clear()
        self.ax.imshow(values, cmap="YlGn", origin="upper", vmin=-0.1, vmax=1.0,
                       interpolation="nearest")
        self.ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        self.ax.grid(which="minor", color="white", linewidth=2)
        self.ax.tick_params(which="both", bottom=False, left=False,
                            labelbottom=False, labelleft=False)
        for row, col in world.walls:
            self.ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, color="#344054"))
        for row in range(size):
            for col in range(size):
                state = (row, col)
                if state in world.walls:
                    continue
                if state == world.start:
                    label = "S"
                elif state == world.goal:
                    label = "G"
                else:
                    label = ARROWS[int(np.argmax(q_values[state]))]
                self.ax.text(col, row, label, ha="center", va="center", fontsize=12,
                             fontweight="bold", color="#172b4d")
        if path:
            self.ax.plot([c for _, c in path], [r for r, _ in path], "o-", color="#e76f51",
                         linewidth=1.6, markersize=3, alpha=0.8, zorder=4)
            self.ax.text(world.goal[1], world.goal[0], "G", ha="center", va="center",
                         fontsize=12, fontweight="bold", color="#172b4d", zorder=5)
        self.ax.set_title(title, fontsize=10, fontweight="bold")
        self.ax.set_xlim(-0.5, size - 0.5)
        self.ax.set_ylim(size - 0.5, -0.5)

        self.ax_rewards.clear()
        if returns:
            episodes = np.arange(1, len(returns) + 1)
            self.ax_rewards.plot(episodes, returns, color=P.color(0), alpha=0.35, label="episode")
            width = min(10, len(returns))
            smooth = np.convolve(returns, np.ones(width) / width, mode="valid")
            self.ax_rewards.plot(episodes[width - 1:], smooth, color=P.color(3),
                                 linewidth=1.8, label=f"{width}-episode mean")
            self.ax_rewards.legend(fontsize=8)
        P.style(self.ax_rewards, "Reward per episode", "episode", "return")


@register
class QLearningDemo(_GridWorldDemo):
    name = "Q-learning"
    order = 10
    params = [Int("episodes", 80, 10, 300, 10),
              Float("learning_rate", 0.25, 0.05, 1.0, 0.05),
              Float("discount", 0.95, 0.0, 1.0, 0.05),
              Float("exploration", 0.3, 0.0, 1.0, 0.05)]

    def run(self, data, p):
        world = data["world"]
        rng = np.random.default_rng(data["seed"])
        q_values = np.zeros((world.size, world.size, len(ACTIONS)))
        returns = []
        for episode in range(p.episodes):
            state = world.start
            path = [state]
            total = 0.0
            epsilon = p.exploration * max(0.05, 1 - episode / p.episodes)
            for _ in range(world.size * world.size * 4):
                if rng.random() < epsilon:
                    action = int(rng.integers(len(ACTIONS)))
                else:
                    action = self._greedy_action(q_values, state, rng)
                next_state, reward, done = world.step(state, action)
                target = reward + (p.discount * q_values[next_state].max() if not done else 0)
                q_values[state][action] += p.learning_rate * (target - q_values[state][action])
                state = next_state
                path.append(state)
                total += reward
                if done:
                    break
            returns.append(total)
            if not self.fast_forward or episode == p.episodes - 1:
                self._draw(world, q_values, returns, self._greedy_path(world, q_values),
                           f"Q-learning · episode {episode + 1}/{p.episodes}")
            yield f"episode {episode + 1}/{p.episodes}: return {total:.2f}"


@register
class SarsaDemo(_GridWorldDemo):
    name = "SARSA"
    order = 20
    params = [Int("episodes", 80, 10, 300, 10),
              Float("learning_rate", 0.25, 0.05, 1.0, 0.05),
              Float("discount", 0.95, 0.0, 1.0, 0.05),
              Float("exploration", 0.3, 0.0, 1.0, 0.05)]

    def run(self, data, p):
        world = data["world"]
        rng = np.random.default_rng(data["seed"])
        q_values = np.zeros((world.size, world.size, len(ACTIONS)))
        returns = []
        for episode in range(p.episodes):
            state = world.start
            path = [state]
            total = 0.0
            epsilon = p.exploration * max(0.05, 1 - episode / p.episodes)

            def choose_action(current):
                if rng.random() < epsilon:
                    return int(rng.integers(len(ACTIONS)))
                return self._greedy_action(q_values, current, rng)

            action = choose_action(state)
            for _ in range(world.size * world.size * 4):
                next_state, reward, done = world.step(state, action)
                next_action = choose_action(next_state) if not done else 0
                target = reward + (p.discount * q_values[next_state][next_action] if not done else 0)
                q_values[state][action] += p.learning_rate * (target - q_values[state][action])
                state, action = next_state, next_action
                path.append(state)
                total += reward
                if done:
                    break
            returns.append(total)
            if not self.fast_forward or episode == p.episodes - 1:
                self._draw(world, q_values, returns, self._greedy_path(world, q_values),
                           f"SARSA · episode {episode + 1}/{p.episodes}")
            yield f"episode {episode + 1}/{p.episodes}: return {total:.2f}"


@register
class ValueIterationDemo(_GridWorldDemo):
    name = "Value Iteration"
    order = 30
    group = "Dynamic programming"
    params = [Int("sweeps", 60, 1, 200),
              Float("discount", 0.95, 0.0, 1.0, 0.05),
              Float("tolerance", 0.001, 0.0001, 0.05, 0.0001)]

    def run(self, data, p):
        world = data["world"]
        size = world.size
        values = np.zeros((size, size))
        q_values = np.zeros((size, size, len(ACTIONS)))
        final_sweep = 0
        for sweep in range(1, p.sweeps + 1):
            updated = values.copy()
            for row in range(size):
                for col in range(size):
                    state = (row, col)
                    if state == world.goal or state in world.walls:
                        continue
                    for action in range(len(ACTIONS)):
                        next_state, reward, done = world.step(state, action)
                        q_values[state][action] = reward + (
                            p.discount * values[next_state] if not done else 0)
                    updated[state] = q_values[state].max()
            delta = float(np.max(np.abs(updated - values)))
            values = updated
            final_sweep = sweep
            if not self.fast_forward or delta < p.tolerance or sweep == p.sweeps:
                self._draw(world, q_values, [], self._greedy_path(world, q_values),
                           f"Value iteration · sweep {sweep}")
            yield f"sweep {sweep}/{p.sweeps}: max value change {delta:.5f}"
            if delta < p.tolerance:
                break
        self.log(f"Converged after {final_sweep} sweep(s).")


@register
class ExpectedSarsaDemo(_GridWorldDemo):
    name = "Expected SARSA"
    order = 40
    params = [Int("episodes", 80, 10, 300, 10),
              Float("learning_rate", 0.25, 0.05, 1.0, 0.05),
              Float("discount", 0.95, 0.0, 1.0, 0.05),
              Float("exploration", 0.3, 0.0, 1.0, 0.05)]
    info = """
    ## Algorithm used
    Expected SARSA: an on-policy temporal-difference update that averages the next-state
    action values under the current epsilon-greedy policy instead of sampling one action.

    ## Compare
    It often has lower update variance than SARSA while still accounting for exploration.
    """

    def run(self, data, p):
        world = data["world"]
        rng = np.random.default_rng(data["seed"])
        q_values = np.zeros((world.size, world.size, len(ACTIONS)))
        returns = []
        for episode in range(p.episodes):
            state = world.start
            total = 0.0
            epsilon = p.exploration * max(0.05, 1 - episode / p.episodes)
            for _ in range(world.size * world.size * 4):
                action = (int(rng.integers(len(ACTIONS))) if rng.random() < epsilon
                          else self._greedy_action(q_values, state, rng))
                next_state, reward, done = world.step(state, action)
                if done:
                    target = reward
                else:
                    next_values = q_values[next_state]
                    target = reward + p.discount * (
                        (1 - epsilon) * next_values.max() + epsilon * next_values.mean())
                q_values[state][action] += p.learning_rate * (target - q_values[state][action])
                state = next_state
                total += reward
                if done:
                    break
            returns.append(total)
            if not self.fast_forward or episode == p.episodes - 1:
                self._draw(world, q_values, returns, self._greedy_path(world, q_values),
                           f"Expected SARSA · episode {episode + 1}/{p.episodes}")
            yield f"episode {episode + 1}/{p.episodes}: return {total:.2f}"


@register
class DoubleQLearningDemo(_GridWorldDemo):
    name = "Double Q-learning"
    order = 50
    params = [Int("episodes", 80, 10, 300, 10),
              Float("learning_rate", 0.25, 0.05, 1.0, 0.05),
              Float("discount", 0.95, 0.0, 1.0, 0.05),
              Float("exploration", 0.3, 0.0, 1.0, 0.05)]
    info = """
    ## Algorithm used
    Double Q-learning maintains two action-value tables. One table selects the next action
    while the other evaluates it, reducing the maximization bias of ordinary Q-learning.
    """

    def run(self, data, p):
        world = data["world"]
        rng = np.random.default_rng(data["seed"])
        q_first = np.zeros((world.size, world.size, len(ACTIONS)))
        q_second = np.zeros_like(q_first)
        returns = []
        for episode in range(p.episodes):
            state = world.start
            total = 0.0
            epsilon = p.exploration * max(0.05, 1 - episode / p.episodes)
            for _ in range(world.size * world.size * 4):
                combined = q_first + q_second
                action = (int(rng.integers(len(ACTIONS))) if rng.random() < epsilon
                          else self._greedy_action(combined, state, rng))
                next_state, reward, done = world.step(state, action)
                if rng.random() < 0.5:
                    selected, evaluator = q_first, q_second
                else:
                    selected, evaluator = q_second, q_first
                best_next = int(np.argmax(selected[next_state]))
                target = reward + (p.discount * evaluator[next_state][best_next] if not done else 0)
                selected[state][action] += p.learning_rate * (target - selected[state][action])
                state = next_state
                total += reward
                if done:
                    break
            returns.append(total)
            if not self.fast_forward or episode == p.episodes - 1:
                self._draw(world, q_first + q_second, returns,
                           self._greedy_path(world, q_first + q_second),
                           f"Double Q-learning · episode {episode + 1}/{p.episodes}")
            yield f"episode {episode + 1}/{p.episodes}: return {total:.2f}"


@register
class MonteCarloControlDemo(_GridWorldDemo):
    name = "Monte Carlo Control"
    order = 60
    params = [Int("episodes", 80, 10, 300, 10),
              Float("discount", 0.95, 0.0, 1.0, 0.05),
              Float("exploration", 0.3, 0.0, 1.0, 0.05)]
    info = """
    ## Algorithm used
    First-visit Monte Carlo control: generate a complete episode, compute discounted returns,
    and average returns for each state-action pair. It learns from episodes rather than bootstrapping.

    ## Trade-off
    Updates are simple and unbiased for the sampled policy, but feedback arrives only after
    an episode ends and return estimates can have high variance.
    """

    def run(self, data, p):
        world = data["world"]
        rng = np.random.default_rng(data["seed"])
        q_values = np.zeros((world.size, world.size, len(ACTIONS)))
        counts = np.zeros_like(q_values)
        returns = []
        for episode in range(p.episodes):
            epsilon = p.exploration * max(0.05, 1 - episode / p.episodes)
            state = world.start
            trajectory = []
            total = 0.0
            for _ in range(world.size * world.size * 4):
                action = (int(rng.integers(len(ACTIONS))) if rng.random() < epsilon
                          else self._greedy_action(q_values, state, rng))
                next_state, reward, done = world.step(state, action)
                trajectory.append((state, action, reward))
                state = next_state
                total += reward
                if done:
                    break
            visited = set()
            for index, (visited_state, visited_action, _) in enumerate(trajectory):
                pair = (visited_state, visited_action)
                if pair in visited:
                    continue
                visited.add(pair)
                future = trajectory[index:]
                outcome = sum((p.discount ** offset) * item[2]
                              for offset, item in enumerate(future))
                counts[visited_state][visited_action] += 1
                q_values[visited_state][visited_action] += (
                    outcome - q_values[visited_state][visited_action]
                ) / counts[visited_state][visited_action]
            returns.append(total)
            if not self.fast_forward or episode == p.episodes - 1:
                self._draw(world, q_values, returns, self._greedy_path(world, q_values),
                           f"Monte Carlo control · episode {episode + 1}/{p.episodes}")
            yield f"episode {episode + 1}/{p.episodes}: return {total:.2f}"


@register
class DynaQDemo(_GridWorldDemo):
    name = "Dyna-Q (Model-Based RL)"
    order = 70
    params = [Int("episodes", 80, 10, 300, 10),
              Int("planning_steps", 8, 0, 30),
              Float("learning_rate", 0.25, 0.05, 1.0, 0.05),
              Float("discount", 0.95, 0.0, 1.0, 0.05),
              Float("exploration", 0.3, 0.0, 1.0, 0.05)]
    info = """
    ## Algorithm used
    Dyna-Q: perform Q-learning from real transitions, store them in a learned environment
    model, then replay simulated transitions for additional planning updates.

    ## Robotics connection
    A robot can reuse a learned action model to plan between sensor observations, reducing
    the number of costly real-world trials.
    """

    def run(self, data, p):
        world = data["world"]
        rng = np.random.default_rng(data["seed"])
        q_values = np.zeros((world.size, world.size, len(ACTIONS)))
        model = {}
        returns = []
        for episode in range(p.episodes):
            state = world.start
            total = 0.0
            epsilon = p.exploration * max(0.05, 1 - episode / p.episodes)
            for _ in range(world.size * world.size * 4):
                action = (int(rng.integers(len(ACTIONS))) if rng.random() < epsilon
                          else self._greedy_action(q_values, state, rng))
                next_state, reward, done = world.step(state, action)
                target = reward + (p.discount * q_values[next_state].max() if not done else 0)
                q_values[state][action] += p.learning_rate * (target - q_values[state][action])
                model[(state, action)] = (reward, next_state, done)
                for _ in range(p.planning_steps):
                    simulated_state, simulated_action = list(model)[int(rng.integers(len(model)))]
                    sim_reward, sim_next, sim_done = model[(simulated_state, simulated_action)]
                    sim_target = sim_reward + (p.discount * q_values[sim_next].max() if not sim_done else 0)
                    q_values[simulated_state][simulated_action] += p.learning_rate * (
                        sim_target - q_values[simulated_state][simulated_action])
                state = next_state
                total += reward
                if done:
                    break
            returns.append(total)
            if not self.fast_forward or episode == p.episodes - 1:
                self._draw(world, q_values, returns, self._greedy_path(world, q_values),
                           f"Dyna-Q · episode {episode + 1}/{p.episodes}")
            yield f"episode {episode + 1}/{p.episodes}: return {total:.2f}, model transitions {len(model)}"


@register
class PolicyIterationDemo(_GridWorldDemo):
    name = "Policy Iteration"
    order = 80
    group = "Dynamic programming"
    params = [Int("iterations", 20, 1, 100),
              Float("discount", 0.95, 0.0, 1.0, 0.05),
              Float("tolerance", 0.001, 0.0001, 0.05, 0.0001)]
    info = """
    ## Algorithm used
    Policy Iteration for a known environment model alternates two steps: evaluate the current
    policy's state values, then improve each action greedily with respect to those values.

    ## Compare
    Value Iteration combines evaluation and improvement in each sweep; Policy Iteration fully
    evaluates a policy before changing it.
    """

    def run(self, data, p):
        world = data["world"]
        size = world.size
        values = np.zeros((size, size))
        policy = np.zeros((size, size), dtype=int)
        q_values = np.zeros((size, size, len(ACTIONS)))
        stable = False
        for iteration in range(1, p.iterations + 1):
            for _ in range(size * size * 4):
                updated = values.copy()
                for row in range(size):
                    for col in range(size):
                        state = (row, col)
                        if state == world.goal or state in world.walls:
                            continue
                        next_state, reward, done = world.step(state, int(policy[state]))
                        updated[state] = reward + (p.discount * values[next_state] if not done else 0)
                delta = float(np.max(np.abs(updated - values)))
                values = updated
                if delta < p.tolerance:
                    break
            stable = True
            for row in range(size):
                for col in range(size):
                    state = (row, col)
                    if state == world.goal or state in world.walls:
                        continue
                    for action in range(len(ACTIONS)):
                        next_state, reward, done = world.step(state, action)
                        q_values[state][action] = reward + (
                            p.discount * values[next_state] if not done else 0)
                    best_action = int(np.argmax(q_values[state]))
                    if best_action != policy[state]:
                        policy[state] = best_action
                        stable = False
            self._draw(world, q_values, [], self._greedy_path(world, q_values),
                       f"Policy iteration · improvement {iteration}")
            yield f"policy improvement {iteration}/{p.iterations}: {'stable' if stable else 'updated'}"
            if stable:
                break