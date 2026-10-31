"""Machine-learning workflows inspired by mobile robotics and manipulation."""
import numpy as np
from matplotlib.patches import Circle, Rectangle

from ml_lab.algorithms.unsupervised.dbscan import DBSCAN
from ml_lab.core import plotting as P
from ml_lab.core.base import AlgorithmDemo
from ml_lab.core.params import Float, Int
from ml_lab.core.registry import register


class _RoboticsDemo(AlgorithmDemo):
    category = "Robotics Demos"
    group = "Perception and state estimation"
    instant = True


@register
class ParticleLocalizationDemo(_RoboticsDemo):
    name = "MCL: Particle-Filter Localization"
    order = 10
    params = [Int("particles", 500, 100, 1500, 100),
              Float("range_noise", 0.18, 0.03, 0.8, 0.01)]
    info = """
    ## Algorithms used
    Monte Carlo Localization (MCL): a particle filter with noisy odometry, landmark-range
    likelihood weighting, and systematic resampling.

    ## Robotics problem
    Estimate a mobile robot's 2D pose from wheel motion and ranges to known landmarks. The
    cloud shows plausible poses; the estimate and error trace show how sensor evidence narrows it.
    """

    def data_params(self):
        return [Int("seed", 3, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        angles = np.linspace(0, 2 * np.pi, 45)
        poses = np.column_stack((5 + 3.2 * np.cos(angles), 5 + 3.2 * np.sin(angles)))
        landmarks = np.array([[1, 1], [9, 1], [9, 9], [1, 9], [5, 5]])
        ranges = np.linalg.norm(poses[:, None] - landmarks[None], axis=2)
        ranges += rng.normal(0, p.range_noise, ranges.shape)
        return {"poses": poses, "landmarks": landmarks, "ranges": ranges, "seed": p.seed}

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax_error = fig.add_subplot(gs[1])

    def draw_data(self, data, p):
        self.ax.clear()
        self.ax.scatter(data["landmarks"][:, 0], data["landmarks"][:, 1], marker="*", s=150,
                        c="#e76f51", edgecolors="black", label="known landmarks")
        self.ax.plot(data["poses"][:, 0], data["poses"][:, 1], "--", color="0.5", label="true route")
        P.style(self.ax, "MCL robot localization", "x position (m)", "y position (m)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.legend(fontsize=7)
        self.ax_error.clear()
        P.style(self.ax_error, "Position error", "sensor update", "error (m)")

    def run(self, data, p):
        rng = np.random.default_rng(data["seed"] + 17)
        truth, landmarks, ranges = data["poses"], data["landmarks"], data["ranges"]
        particles = np.clip(truth[0] + rng.normal(0, 1.0, (p.particles, 2)), 0, 10)
        estimates, errors = [], []
        for step, pose in enumerate(truth):
            if step:
                motion = truth[step] - truth[step - 1]
                particles += motion + rng.normal(0, 0.07, particles.shape)
                particles = np.clip(particles, 0, 10)
            predicted = np.linalg.norm(particles[:, None] - landmarks[None], axis=2)
            residual = predicted - ranges[step]
            log_weights = -0.5 * np.sum((residual / p.range_noise) ** 2, axis=1)
            weights = np.exp(log_weights - log_weights.max())
            weights /= weights.sum()
            estimate = weights @ particles
            estimates.append(estimate)
            errors.append(np.linalg.norm(estimate - pose))
            positions = (rng.random() + np.arange(p.particles)) / p.particles
            particles = particles[np.searchsorted(np.cumsum(weights), positions)]
            if not self.fast_forward or step == len(truth) - 1:
                self.ax.clear()
                self.ax.plot(truth[:step + 1, 0], truth[:step + 1, 1], "--", color="0.4",
                             label="true route")
                self.ax.plot(*np.asarray(estimates).T, color=P.color(0), linewidth=2,
                             label="estimated route")
                self.ax.scatter(particles[:, 0], particles[:, 1], s=7, alpha=0.16,
                                color=P.color(2), label="resampled particles")
                self.ax.scatter(*landmarks.T, marker="*", s=150, c="#e76f51",
                                edgecolors="black", label="landmarks")
                self.ax.scatter(*pose, marker="s", s=55, c="black", label="true robot")
                P.set_limits(self.ax, np.vstack((truth, landmarks)), pad=0.8)
                self.ax.set_aspect("equal", adjustable="box")
                P.style(self.ax, f"MCL localization · update {step + 1}/{len(truth)}",
                        "x position (m)", "y position (m)")
                self.ax.legend(fontsize=7, loc="upper right")
                self.ax_error.clear()
                self.ax_error.plot(errors, color=P.color(3))
                P.style(self.ax_error, "Particle-filter position error", "sensor update", "error (m)")
            yield f"update {step + 1}/{len(truth)}: position error {errors[-1]:.2f} m"


@register
class KalmanFusionDemo(_RoboticsDemo):
    name = "Kalman Filter: GPS + Wheel-Odometry Fusion"
    order = 20
    params = [Float("gps_noise", 1.0, 0.1, 3.0, 0.1),
              Float("encoder_noise", 0.25, 0.03, 1.0, 0.01)]
    info = """
    ## Algorithms used
    Linear Kalman Filter with a constant-velocity state model, Gaussian process noise, and
    sequential measurement updates from GPS position and wheel-encoder velocity.

    ## Robotics problem
    Fuse sensors with complementary weaknesses: GPS gives noisy global position, while wheel
    odometry gives useful short-term velocity but drifts. The shaded band is filter uncertainty.
    """

    def data_params(self):
        return [Int("steps", 90, 30, 180, 10, group="Data"),
                Int("seed", 5, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        dt = 0.15
        times = np.arange(p.steps) * dt
        acceleration = 0.22 * np.sin(times * 0.8)
        velocity = np.zeros(p.steps)
        position = np.zeros(p.steps)
        for i in range(1, p.steps):
            velocity[i] = velocity[i - 1] + acceleration[i] * dt
            position[i] = position[i - 1] + velocity[i] * dt
        gps = position + rng.normal(0, p.gps_noise, p.steps)
        encoder = velocity + rng.normal(0, p.encoder_noise, p.steps)
        return {"time": times, "position": position, "velocity": velocity,
                "gps": gps, "encoder": encoder, "dt": dt, "seed": p.seed}

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax_error = fig.add_subplot(gs[1])

    def draw_data(self, data, p):
        self.ax.clear()
        self.ax.plot(data["time"], data["position"], color="black", label="true position")
        self.ax.scatter(data["time"], data["gps"], s=12, alpha=0.5, label="GPS")
        P.style(self.ax, "Kalman sensor fusion", "time (s)", "position (m)")
        self.ax.legend(fontsize=8)
        self.ax_error.clear()
        P.style(self.ax_error, "Position estimation error", "time (s)", "error (m)")

    def run(self, data, p):
        dt = data["dt"]
        transition = np.array([[1, dt], [0, 1]])
        process = 0.04 * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                                   [dt ** 3 / 2, dt ** 2]])
        gps_matrix = np.array([[1.0, 0.0]])
        encoder_matrix = np.array([[0.0, 1.0]])
        state = np.array([data["gps"][0], data["encoder"][0]])
        covariance = np.diag([p.gps_noise ** 2, p.encoder_noise ** 2])
        estimates, errors = [], []
        for i, time_value in enumerate(data["time"]):
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
            for measurement, matrix, variance in (
                    (data["gps"][i], gps_matrix, p.gps_noise ** 2),
                    (data["encoder"][i], encoder_matrix, p.encoder_noise ** 2)):
                innovation = measurement - (matrix @ state)[0]
                innovation_variance = (matrix @ covariance @ matrix.T)[0, 0] + variance
                gain = covariance @ matrix.T / innovation_variance
                state = state + gain[:, 0] * innovation
                covariance = (np.eye(2) - gain @ matrix) @ covariance
            estimates.append(state[0])
            errors.append(state[0] - data["position"][i])
            if not self.fast_forward or i == len(data["time"]) - 1:
                self.ax.clear()
                self.ax.plot(data["time"][:i + 1], data["position"][:i + 1],
                             color="black", label="true position")
                self.ax.scatter(data["time"][:i + 1], data["gps"][:i + 1], s=10,
                                alpha=0.4, color=P.color(3), label="GPS position")
                self.ax.plot(data["time"][:i + 1], estimates, color=P.color(0),
                             linewidth=2, label="Kalman estimate")
                uncertainty = 2 * np.sqrt(max(covariance[0, 0], 0))
                self.ax.fill_between([time_value - dt, time_value],
                                     [state[0] - uncertainty] * 2,
                                     [state[0] + uncertainty] * 2, color=P.color(0), alpha=0.18)
                P.style(self.ax, f"Kalman fusion · sample {i + 1}/{len(data['time'])}",
                        "time (s)", "position (m)")
                self.ax.legend(fontsize=8)
                self.ax_error.clear()
                self.ax_error.plot(data["time"][:i + 1], errors, color=P.color(2))
                self.ax_error.axhline(0, color="0.4", linewidth=0.8)
                P.style(self.ax_error, "Fused estimate error", "time (s)", "error (m)")
            yield f"sample {i + 1}/{len(data['time'])}: position error {errors[-1]:.2f} m"


@register
class LidarDBSCANDemo(_RoboticsDemo):
    name = "DBSCAN: 2D LiDAR Obstacle Segmentation"
    group = "Perception and mapping"
    order = 30
    params = [Float("eps", 0.38, 0.08, 1.2, 0.02),
              Int("min_samples", 5, 2, 15)]
    info = """
    ## Algorithms used
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise): radius-neighbor
    connectivity with a minimum core-point density.

    ## Robotics problem
    Segment a 2D LiDAR point cloud into nearby obstacles without specifying their count.
    Cluster colors identify objects; crosses are isolated returns or background noise.
    """

    def data_params(self):
        return [Int("returns_per_object", 65, 20, 120, 5, group="Data"),
                Float("sensor_noise", 0.08, 0.01, 0.35, 0.01, group="Data"),
                Int("seed", 12, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        centers = np.array([[2.0, 3.0], [5.0, 6.2], [8.0, 3.7]])
        clusters = [center + rng.normal(0, p.sensor_noise, (p.returns_per_object, 2))
                    for center in centers]
        background = rng.uniform([0, 0], [10, 8], (30, 2))
        return {"points": np.vstack(clusters + [background]), "truth": centers}

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax_sizes = fig.add_subplot(gs[1])

    def draw_data(self, data, p):
        self.ax.clear()
        self.ax.scatter(data["points"][:, 0], data["points"][:, 1], s=9, color="0.55")
        self.ax.scatter(0, 0, marker="^", s=100, color="black", label="robot / LiDAR")
        P.style(self.ax, "Raw LiDAR returns", "x (m)", "y (m)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.legend(fontsize=8)
        self.ax_sizes.clear()
        P.style(self.ax_sizes, "Segment sizes", "cluster", "returns")

    def run(self, data, p):
        points = data["points"]
        model = DBSCAN(p.eps, p.min_samples)
        labels = None
        core = None
        for labels, _, core in model.steps(points):
            pass
        self.ax.clear()
        for label in sorted(set(labels) - {-1}):
            mask = labels == label
            self.ax.scatter(points[mask, 0], points[mask, 1], s=18,
                            color=P.color(label), label=f"obstacle {label + 1}")
        noise = labels == -1
        self.ax.scatter(points[noise, 0], points[noise, 1], s=20, color="0.2",
                        marker="x", label="noise returns")
        self.ax.scatter(0, 0, marker="^", s=120, color="black", label="robot / LiDAR")
        P.set_limits(self.ax, np.vstack((points, [[0, 0]])), pad=0.7)
        self.ax.set_aspect("equal", adjustable="box")
        P.style(self.ax, f"DBSCAN segmentation · eps={p.eps:.2f}", "x (m)", "y (m)")
        self.ax.legend(fontsize=7)
        cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
        self.ax_sizes.clear()
        if len(cluster_ids):
            self.ax_sizes.bar([f"O{i + 1}" for i in cluster_ids], counts,
                              color=[P.color(i) for i in cluster_ids])
        self.ax_sizes.text(0.98, 0.96, f"noise: {int(noise.sum())}\ncore returns: {int(core.sum())}",
                           transform=self.ax_sizes.transAxes, ha="right", va="top", fontsize=9)
        P.style(self.ax_sizes, "DBSCAN object clusters", "detected obstacle", "returns")
        yield f"DBSCAN found {len(cluster_ids)} obstacle(s), {int(noise.sum())} noise returns"


@register
class TerrainNaiveBayesDemo(_RoboticsDemo):
    name = "Gaussian Naive Bayes: Robot Terrain Recognition"
    group = "Perception and mapping"
    order = 40
    params = [Int("samples_per_terrain", 120, 40, 300, 20),
              Float("sensor_noise", 0.18, 0.02, 0.7, 0.02)]
    info = """
    ## Algorithms used
    Gaussian Naive Bayes: per-class feature means and variances, Bayes' rule, and an
    independence assumption between sensor channels.

    ## Robotics problem
    Classify the surface under a mobile robot from simulated wheel-slip, chassis-vibration,
    and motor-current readings. Compare asphalt, gravel, and mud predictions in the confusion map.
    """

    def data_params(self):
        return [Int("seed", 22, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        means = np.array([[0.08, 0.18, 0.38], [0.28, 0.58, 0.62], [0.72, 0.32, 0.92]])
        spreads = np.array([[0.07, 0.08, 0.10], [0.12, 0.16, 0.14], [0.15, 0.13, 0.18]])
        X = np.vstack([rng.normal(means[c], spreads[c] * p.sensor_noise / 0.18,
                                  (p.samples_per_terrain, 3)) for c in range(3)])
        y = np.repeat(np.arange(3), p.samples_per_terrain)
        order = rng.permutation(len(y))
        split = int(len(y) * 0.75)
        return {"X": X[order[:split]], "y": y[order[:split]],
                "Xt": X[order[split:]], "yt": y[order[split:]]}

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])
        self.ax = fig.add_subplot(gs[0])
        self.ax_cm = fig.add_subplot(gs[1])

    def draw_data(self, data, p):
        self.ax.clear()
        for terrain in range(3):
            mask = data["y"] == terrain
            self.ax.scatter(data["X"][mask, 0], data["X"][mask, 1], s=18,
                            color=P.color(terrain), label=("asphalt", "gravel", "mud")[terrain])
        P.style(self.ax, "Training sensor signatures", "wheel slip", "chassis vibration")
        self.ax.legend(fontsize=8)
        self.ax_cm.clear()
        P.style(self.ax_cm, "Test confusion matrix", "predicted terrain", "actual terrain")

    def run(self, data, p):
        X, y = data["X"], data["y"]
        classes = np.unique(y)
        means = np.array([X[y == c].mean(0) for c in classes])
        variances = np.array([X[y == c].var(0) + 1e-5 for c in classes])
        priors = np.array([(y == c).mean() for c in classes])
        delta = data["Xt"][:, None, :] - means[None, :, :]
        scores = np.log(priors)[None, :] - 0.5 * np.sum(
            np.log(2 * np.pi * variances)[None, :, :] + delta ** 2 / variances[None, :, :], axis=2)
        predicted = classes[np.argmax(scores, axis=1)]
        confusion = np.zeros((3, 3), dtype=int)
        np.add.at(confusion, (data["yt"], predicted), 1)
        accuracy = float(np.mean(predicted == data["yt"]))
        self.ax.clear()
        for terrain in range(3):
            mask = y == terrain
            self.ax.scatter(X[mask, 0], X[mask, 1], s=16, alpha=0.55,
                            color=P.color(terrain), label=("asphalt", "gravel", "mud")[terrain])
        self.ax.set_xlabel("wheel slip ratio")
        self.ax.set_ylabel("chassis vibration")
        P.style(self.ax, "Gaussian Naive Bayes: terrain sensor model")
        self.ax.legend(fontsize=8)
        self.ax_cm.clear()
        self.ax_cm.imshow(confusion, cmap="Blues")
        for row in range(3):
            for col in range(3):
                self.ax_cm.text(col, row, str(confusion[row, col]), ha="center", va="center")
        names = ["asphalt", "gravel", "mud"]
        self.ax_cm.set_xticks(range(3), names, rotation=25, ha="right")
        self.ax_cm.set_yticks(range(3), names)
        P.style(self.ax_cm, f"Naive Bayes test accuracy {accuracy:.1%}", "predicted", "actual")
        yield f"Gaussian Naive Bayes: test accuracy {accuracy:.1%} on {len(data['yt'])} sensor windows"


@register
class OccupancyMappingDemo(_RoboticsDemo):
    name = "Bayesian Occupancy-Grid Mapping"
    group = "Perception and mapping"
    order = 50
    params = [Int("laser_beams", 61, 21, 121, 10),
              Float("range_noise", 0.04, 0.0, 0.2, 0.01)]
    info = """
    ## Algorithms used
    Inverse sensor model with Bayesian log-odds updates: free-space evidence decreases cell
    occupancy odds; obstacle returns increase them. Log odds make repeated updates additive.

    ## Robotics problem
    Build a 2D map while a robot drives a survey route using a simulated planar laser scanner.
    The display compares the known room with the map inferred from successive scans.
    """

    def data_params(self):
        return [Int("scans", 12, 4, 24, 2, group="Data"),
                Int("seed", 31, 0, 999, group="Data")]

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        poses = np.column_stack((np.linspace(1, 9, p.scans),
                                 1.2 + 0.5 * np.sin(np.linspace(0, np.pi, p.scans))))
        obstacles = np.array([[2.2, 4.1], [3.8, 6.5], [5.9, 4.8], [7.6, 7.1],
                              [8.4, 3.8], [1.5, 8.0], [5.2, 8.3]])
        return {"poses": poses, "obstacles": obstacles, "seed": p.seed}

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        gs = fig.add_gridspec(1, 2)
        self.ax = fig.add_subplot(gs[0])
        self.ax_map = fig.add_subplot(gs[1])

    def draw_data(self, data, p):
        self.ax.clear()
        self.ax.scatter(*data["obstacles"].T, marker="s", color="black", label="walls / obstacles")
        self.ax.plot(*data["poses"].T, "--", color=P.color(0), label="robot survey route")
        P.style(self.ax, "Known simulated room", "x (m)", "y (m)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.legend(fontsize=7)
        self.ax_map.clear()
        P.style(self.ax_map, "Occupancy probability", "x (m)", "y (m)")

    @staticmethod
    def _cell(x, y, size):
        col = int(np.clip(round(x / 10 * (size - 1)), 0, size - 1))
        row = size - 1 - int(np.clip(round(y / 10 * (size - 1)), 0, size - 1))
        return row, col

    def run(self, data, p):
        size = 60
        rng = np.random.default_rng(data["seed"] + 1)
        log_odds = np.zeros((size, size))
        fov = np.pi
        max_range = 9.0
        angles = np.linspace(-fov / 2, fov / 2, p.laser_beams)
        for scan, pose in enumerate(data["poses"]):
            hits = {}
            for obstacle in data["obstacles"]:
                dx, dy = obstacle - pose
                distance = np.hypot(dx, dy)
                bearing = np.arctan2(dy, dx)
                if distance > max_range or abs(bearing) > fov / 2:
                    continue
                beam = int(np.argmin(np.abs(angles - bearing)))
                if beam not in hits or distance < hits[beam][0]:
                    hits[beam] = (distance, obstacle)
            for beam, angle in enumerate(angles):
                direction = angle
                distance, endpoint = hits.get(beam, (max_range, None))
                distance = max(0.1, distance + rng.normal(0, p.range_noise))
                free_to = max(0.05, distance - 0.18) if endpoint is not None else distance
                ray = np.array([np.cos(direction), np.sin(direction)])
                for traveled in np.arange(0, free_to, 0.12):
                    point = pose + traveled * ray
                    row, col = self._cell(*point, size)
                    log_odds[row, col] -= 0.12
                if endpoint is not None:
                    measured = pose + distance * ray
                    row, col = self._cell(*measured, size)
                    log_odds[row, col] += 0.9
            np.clip(log_odds, -5, 5, out=log_odds)
            if not self.fast_forward or scan == len(data["poses"]) - 1:
                self.ax.clear()
                self.ax.scatter(*data["obstacles"].T, marker="s", color="black", label="true obstacles")
                self.ax.plot(*data["poses"][:scan + 1].T, "--", color=P.color(0), label="robot route")
                self.ax.scatter(*pose, marker="^", s=100, color=P.color(3), label="robot")
                self.ax.set_xlim(0, 10)
                self.ax.set_ylim(0, 10)
                self.ax.set_aspect("equal", adjustable="box")
                P.style(self.ax, f"Bayesian mapping · scan {scan + 1}/{len(data['poses'])}",
                        "x (m)", "y (m)")
                self.ax.legend(fontsize=7)
                probability = 1 / (1 + np.exp(-log_odds))
                self.ax_map.clear()
                self.ax_map.imshow(probability, origin="lower", extent=(0, 10, 0, 10),
                                   cmap="gray_r", vmin=0.25, vmax=0.75, interpolation="nearest")
                self.ax_map.scatter(*pose, marker="^", s=70, color=P.color(3))
                P.style(self.ax_map, "Bayesian occupancy probability", "x (m)", "y (m)")
            yield f"scan {scan + 1}/{len(data['poses'])}: updated occupancy log odds"


@register
class GaussianProcessCalibrationDemo(_RoboticsDemo):
    name = "Gaussian Process: Tactile Actuator Calibration"
    group = "Manipulation and calibration"
    order = 60
    params = [Float("length_scale", 0.18, 0.04, 0.6, 0.02),
              Float("noise_level", 0.12, 0.01, 0.5, 0.01)]
    info = """
    ## Algorithms used
    Gaussian Process Regression with an RBF (squared-exponential) kernel and Gaussian
    observation noise. The posterior gives both a prediction and uncertainty.

    ## Robotics problem
    Calibrate the relationship between actuator command and measured fingertip deflection.
    The shaded interval shows where the robot should collect more calibration measurements.
    """

    def data_params(self):
        return [Int("observations", 24, 8, 80, 2, group="Data"),
                Int("seed", 42, 0, 999, group="Data")]

    @staticmethod
    def response(command):
        return 0.7 * np.sin(command * 2.4 * np.pi) + 0.45 * command

    def make_data(self, p):
        rng = np.random.default_rng(p.seed)
        command = np.sort(rng.uniform(0, 1, p.observations))
        measured = self.response(command) + rng.normal(0, 0.12, p.observations)
        return {"command": command, "measured": measured}

    def draw_data(self, data, p):
        self.ax.clear()
        self.ax.scatter(data["command"], data["measured"], s=24, color=P.color(0),
                        edgecolors="black", label="calibration measurements")
        P.style(self.ax, "Actuator calibration measurements", "normalized command", "tip deflection")
        self.ax.legend(fontsize=8)

    def run(self, data, p):
        X, y = data["command"], data["measured"]
        grid = np.linspace(0, 1, 240)

        def kernel(a, b):
            return np.exp(-0.5 * ((a[:, None] - b[None, :]) / p.length_scale) ** 2)

        covariance = kernel(X, X) + (p.noise_level ** 2 + 1e-8) * np.eye(len(X))
        cholesky = np.linalg.cholesky(covariance)
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, y))
        cross = kernel(grid, X)
        mean = cross @ alpha
        projected = np.linalg.solve(cholesky, cross.T)
        uncertainty = np.sqrt(np.maximum(1 - np.sum(projected ** 2, axis=0), 0))
        self.ax.clear()
        self.ax.plot(grid, self.response(grid), "--", color="0.35", label="simulated true response")
        self.ax.fill_between(grid, mean - 2 * uncertainty, mean + 2 * uncertainty,
                             color=P.color(0), alpha=0.18, label="95% predictive interval")
        self.ax.plot(grid, mean, color=P.color(0), linewidth=2, label="GP posterior mean")
        self.ax.scatter(X, y, s=24, color=P.color(3), edgecolors="black", zorder=3,
                        label="calibration measurements")
        P.style(self.ax, f"Gaussian Process calibration · length scale {p.length_scale:.2f}",
                "normalized actuator command", "tip deflection")
        self.ax.legend(fontsize=7)
        yield f"fit Gaussian Process to {len(X)} calibration observations"


@register
class MicromouseDemo(_RoboticsDemo):
    name = "Micromouse: Flood-Fill Maze Solver"
    group = "Navigation and planning"
    order = 70
    info = """
    ## Algorithms used
    Depth-first exploration maps the maze one cell at a time. Flood-fill propagates
    distances from the center through the known map; after mapping, breadth-first search
    finds a shortest route for the final speed run.

    ## Robotics problem
    A micromouse uses wall sensors to explore an unknown maze, reach the center goal,
    and then exploit its map to run the shortest path. The left panel is its onboard
    map; the right panel shows the ground truth and the robot's trail.
    """

    _DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def data_params(self):
        return [Int("maze_size", 8, 6, 16, 2, group="Data"),
                Int("seed", 19, 0, 999, group="Data")]

    def make_data(self, p):
        size = p.maze_size
        rng = np.random.default_rng(p.seed)
        walls = np.ones((size, size, 4), dtype=bool)
        start = (size - 1, 0)
        stack, visited = [start], {start}
        while stack:
            row, col = stack[-1]
            options = []
            for direction, (dr, dc) in enumerate(self._DIRECTIONS):
                neighbor = (row + dr, col + dc)
                if (0 <= neighbor[0] < size and 0 <= neighbor[1] < size
                        and neighbor not in visited):
                    options.append((direction, neighbor))
            if not options:
                stack.pop()
                continue
            direction, neighbor = options[int(rng.integers(len(options)))]
            walls[row, col, direction] = False
            walls[neighbor[0], neighbor[1], (direction + 2) % 4] = False
            visited.add(neighbor)
            stack.append(neighbor)
        middle = size // 2
        goals = {(row, col) for row in (middle - 1, middle)
                 for col in (middle - 1, middle)}
        return {"walls": walls, "start": start, "goals": goals, "seed": p.seed}

    def setup_figure(self, fig):
        fig.clear()
        self.fig = fig
        self.ax = fig.add_subplot(1, 2, 1)
        self.ax_truth = fig.add_subplot(1, 2, 2)

    @staticmethod
    def _reveal(walls, known, cell):
        size = len(walls)
        row, col = cell
        known[row, col, :] = True
        for direction, (dr, dc) in enumerate(MicromouseDemo._DIRECTIONS):
            neighbor = (row + dr, col + dc)
            if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                known[neighbor[0], neighbor[1], (direction + 2) % 4] = True

    @staticmethod
    def _floodfill(walls, known, goals):
        size = len(walls)
        distances = np.full((size, size), -1, dtype=int)
        queue = list(goals)
        for row, col in queue:
            distances[row, col] = 0
        for row, col in queue:
            for direction, (dr, dc) in enumerate(MicromouseDemo._DIRECTIONS):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < size and 0 <= nc < size):
                    continue
                if known[row, col, direction] and walls[row, col, direction]:
                    continue
                if distances[nr, nc] < 0:
                    distances[nr, nc] = distances[row, col] + 1
                    queue.append((nr, nc))
        return distances

    @staticmethod
    def _shortest_path(walls, start, goals):
        size = len(walls)
        parents = {start: None}
        queue = [start]
        target = None
        for row, col in queue:
            if (row, col) in goals:
                target = (row, col)
                break
            for direction, (dr, dc) in enumerate(MicromouseDemo._DIRECTIONS):
                nr, nc = row + dr, col + dc
                neighbor = (nr, nc)
                if (0 <= nr < size and 0 <= nc < size
                        and not walls[row, col, direction] and neighbor not in parents):
                    parents[neighbor] = (row, col)
                    queue.append(neighbor)
        path = []
        while target is not None:
            path.append(target)
            target = parents[target]
        return path[::-1]

    def _draw_maze(self, ax, walls, known, position, trail, goals, title,
                   distances=None, speed_path=None, reveal_all=False):
        size = len(walls)
        ax.clear()
        if distances is not None:
            ax.imshow(distances, origin="upper", extent=(0, size, 0, size),
                      cmap="YlGnBu", alpha=0.42, interpolation="nearest",
                      vmin=0, vmax=max(1, int(distances.max())))
        for row, col in goals:
            ax.add_patch(Rectangle((col, size - row - 1), 1, 1,
                                   facecolor="#8bd3a8", edgecolor="none",
                                   alpha=0.55, zorder=1))
        ax.set_xticks(np.arange(size + 1), minor=True)
        ax.set_yticks(np.arange(size + 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.9)
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        for row in range(size):
            for col in range(size):
                for direction, (dr, dc) in enumerate(self._DIRECTIONS):
                    if not walls[row, col, direction] or not (reveal_all or known[row, col, direction]):
                        continue
                    x0, x1 = col, col + 1
                    y0, y1 = size - row, size - row - 1
                    if direction == 0:
                        y0 = y1 = size - row
                    elif direction == 1:
                        x0 = x1 = col + 1
                    elif direction == 2:
                        y0 = y1 = size - row - 1
                    else:
                        x0 = x1 = col
                    ax.plot([x0, x1], [y0, y1], color="#263238", linewidth=1.5,
                            solid_capstyle="butt", zorder=3)
        if distances is not None:
            fontsize = max(5, 9 - size // 4)
            for row in range(size):
                for col in range(size):
                    if distances[row, col] >= 0:
                        ax.text(col + 0.5, size - row - 0.5, str(distances[row, col]),
                                ha="center", va="center", fontsize=fontsize,
                                color="#263238", alpha=0.85, zorder=4)
        if trail:
            ax.plot([col + 0.5 for row, col in trail],
                    [size - row - 0.5 for row, col in trail],
                    color="#e76f51", linewidth=1.2, alpha=0.7, zorder=5)
        if speed_path:
            ax.plot([col + 0.5 for row, col in speed_path],
                    [size - row - 0.5 for row, col in speed_path],
                    color="#087f5b", linewidth=2.4, zorder=6)
        row, col = position
        ax.scatter(col + 0.5, size - row - 0.5, marker="^", s=65,
                   color="#e76f51", edgecolors="white", linewidths=0.8, zorder=7)
        P.style(ax, title, "column", "row")
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect("equal", adjustable="box")

    def draw_data(self, data, p):
        size = len(data["walls"])
        unknown = np.zeros_like(data["walls"])
        self._draw_maze(self.ax, data["walls"], unknown, data["start"], [], data["goals"],
                        "Micromouse · ready to explore")
        self._draw_maze(self.ax_truth, data["walls"], unknown, data["start"], [],
                        data["goals"], f"Generated {size} × {size} maze", reveal_all=True)

    def run(self, data, p):
        walls, start, goals = data["walls"], data["start"], data["goals"]
        size = len(walls)
        known = np.zeros_like(walls)
        visited = {start}
        self._reveal(walls, known, start)
        rng = np.random.default_rng(data["seed"] + 1)
        stack, trail = [start], [start]
        while stack:
            row, col = stack[-1]
            distances = self._floodfill(walls, known, goals)
            options = []
            for direction, (dr, dc) in enumerate(self._DIRECTIONS):
                neighbor = (row + dr, col + dc)
                if (not walls[row, col, direction]
                        and 0 <= neighbor[0] < size and 0 <= neighbor[1] < size
                        and neighbor not in visited):
                    options.append(neighbor)
            if options:
                best_distance = min(distances[cell] for cell in options)
                best_options = [cell for cell in options if distances[cell] == best_distance]
                next_cell = best_options[int(rng.integers(len(best_options)))]
                visited.add(next_cell)
                self._reveal(walls, known, next_cell)
                stack.append(next_cell)
            else:
                stack.pop()
                if not stack:
                    break
                next_cell = stack[-1]
            trail.append(next_cell)
            distances = self._floodfill(walls, known, goals)
            if not self.fast_forward:
                self._draw_maze(self.ax, walls, known, next_cell, trail, goals,
                                f"Onboard map · {len(visited)}/{size * size} cells",
                                distances=distances)
                self._draw_maze(self.ax_truth, walls, known, next_cell, trail, goals,
                                "Ground truth · exploration trail", reveal_all=True)
            yield f"exploration: mapped {len(visited)}/{size * size} cells"

        known[:] = True
        path = self._shortest_path(walls, start, goals)
        if self.fast_forward:
            self._draw_maze(self.ax, walls, known, start, trail, goals,
                            "Onboard map · shortest route", distances=self._floodfill(walls, known, goals),
                            speed_path=path)
            self._draw_maze(self.ax_truth, walls, known, start, trail, goals,
                            "Ground truth · shortest route", reveal_all=True, speed_path=path)
        for step, position in enumerate(path):
            speed_trail = path[:step + 1]
            if not self.fast_forward or step == len(path) - 1:
                self._draw_maze(self.ax, walls, known, position, trail, goals,
                                "Onboard map · shortest route",
                                distances=self._floodfill(walls, known, goals),
                                speed_path=speed_trail)
                self._draw_maze(self.ax_truth, walls, known, position, trail, goals,
                                "Ground truth · shortest speed run", reveal_all=True,
                                speed_path=speed_trail)
            yield f"speed run: step {step + 1}/{len(path)}"