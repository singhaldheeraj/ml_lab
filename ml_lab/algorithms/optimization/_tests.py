"""Headless checks for optimization math and demo contracts."""
import unittest

import numpy as np

from ml_lab.algorithms.supervised.linear_regression import LinearRegressionGD
from ml_lab.core.params import Values


class OptimizationTests(unittest.TestCase):
    def test_batch_gradient_and_update(self):
        from .batch_gradient_descent import BatchGradientDescentDemo

        demo = BatchGradientDescentDemo()
        params = Values(**{param.name: param.default for param in demo.get_params()})
        self.assertNotIn("method", vars(params))
        self.assertEqual(demo.category, "Optimization")
        model = LinearRegressionGD(0, 0)
        samples, targets = np.array([1., 2.]), np.array([2., 4.])
        self.assertEqual(model.mse(samples, targets), 10.)
        np.testing.assert_allclose(model.gradient(samples, targets), [-10., -6.])
        updates = list(model.fit_steps(samples, targets, 0.1, 1, demo.method, 1,
                                       np.random.default_rng(0)))
        self.assertEqual(updates, [0])
        np.testing.assert_allclose([model.w, model.b], [1., 0.6])

    def test_training_methods(self):
        from .batch_gradient_descent import BatchGradientDescentDemo
        from .stochastic_gradient_descent import StochasticGradientDescentDemo
        from .mini_batch_gradient_descent import MiniBatchGradientDescentDemo

        samples = np.linspace(-1, 1, 7)
        targets = 2 * samples + 0.5
        for demo_class, updates_per_epoch in ((BatchGradientDescentDemo, 1),
                                               (StochasticGradientDescentDemo, 7),
                                               (MiniBatchGradientDescentDemo, 3)):
            demo = demo_class()
            model = LinearRegressionGD()
            updates = list(model.fit_steps(samples, targets, 0.05, 30, demo.method, 3,
                                           np.random.default_rng(7)))
            self.assertEqual(len(updates), 30 * updates_per_epoch)
            self.assertLess(model.mse(samples, targets), 0.2)
            repeated = LinearRegressionGD()
            list(repeated.fit_steps(samples, targets, 0.05, 30, demo.method, 3,
                                    np.random.default_rng(7)))
            np.testing.assert_allclose([model.w, model.b], [repeated.w, repeated.b])
        batch, mini = LinearRegressionGD(), LinearRegressionGD()
        list(batch.fit_steps(samples, targets, 0.05, 1, "Batch GD", 7, np.random.default_rng(0)))
        list(mini.fit_steps(samples, targets, 0.05, 1, "Mini-batch GD", 7, np.random.default_rng(0)))
        np.testing.assert_allclose([batch.w, batch.b], [mini.w, mini.b])

    def test_mse_contributions(self):
        from .mean_squared_error import MeanSquaredErrorDemo

        samples, targets = np.array([1., 2.]), np.array([2., 4.])
        errors = MeanSquaredErrorDemo.squared_errors(samples, targets, 1., 0.)
        np.testing.assert_allclose(errors, [1., 4.])
        self.assertEqual(float(np.mean(errors)), 2.5)
        self.assertEqual(float(np.mean(errors)), LinearRegressionGD(1, 0).mse(samples, targets))
        np.testing.assert_allclose(MeanSquaredErrorDemo.squared_errors(samples, targets, 2., 0.), 0.)

    def test_training_demo_runner(self):
        from matplotlib.figure import Figure
        from .batch_gradient_descent import BatchGradientDescentDemo
        from .stochastic_gradient_descent import StochasticGradientDescentDemo
        from .mini_batch_gradient_descent import MiniBatchGradientDescentDemo

        for demo_class in (BatchGradientDescentDemo, StochasticGradientDescentDemo, MiniBatchGradientDescentDemo):
            with self.subTest(demo=demo_class.name):
                demo = demo_class()
                demo.log = lambda message: None
                params = Values(**{param.name: param.default for param in demo.get_params()})
                params.epochs = 1
                data = demo.make_data(params)
                demo.setup_figure(Figure())
                messages = list(demo.run(data, params))
                self.assertTrue(messages[-1].startswith("Done:"))
                self.assertNotIn("method", vars(params))
                if demo.method != "Mini-batch GD":
                    self.assertNotIn("batch_size", vars(params))

    def test_quadratic_descent(self):
        from .gradient_descent import QuadraticObjective

        model = QuadraticObjective(5)
        np.testing.assert_allclose(model.gradient(np.array([2., 3.])), [2., 15.])
        steps = list(model.steps([2., 3.], 0.1, 200))
        np.testing.assert_allclose(steps[1][0], [1.8, 1.5])
        losses = [value for point, value in steps]
        self.assertEqual(losses, sorted(losses, reverse=True))
        self.assertLess(losses[-1], 1e-10)
        divergent = list(model.steps([2., 3.], 1., 100))
        self.assertGreater(divergent[-1][1], 1e12)
        self.assertLess(len(divergent), 101)

    def test_lagrange_conditions(self):
        from .lagrange_multipliers import LagrangeProjection

        for target in ([2., 2.], [-3., 1.], [0., 0.]):
            for normal in ([1., 1.], [1., -1.], [2., 1.]):
                for offset in (-4., 0., 1., 4.):
                    point, multiplier = LagrangeProjection.solve(target, normal, offset)
                    normal = np.array(normal)
                    self.assertAlmostEqual(float(point @ normal), offset)
                    np.testing.assert_allclose(2 * (point - target) + multiplier * normal, 0., atol=1e-12)
                    tangent = np.array([-normal[1], normal[0]])
                    value = np.sum((point - target) ** 2)
                    self.assertLess(value, np.sum((point + tangent - target) ** 2))
        with self.assertRaises(ValueError):
            LagrangeProjection.solve([0, 0], [0, 0], 1)

    def test_registration(self):
        from ml_lab.core.registry import discover

        classes, errors = discover()
        self.assertFalse(errors)
        optimization = [cls for cls in classes if cls.category == "Optimization"]
        self.assertEqual(len(optimization), 6)
        self.assertEqual(len({cls.__module__ for cls in optimization}), 6)


if __name__ == "__main__":
    unittest.main()