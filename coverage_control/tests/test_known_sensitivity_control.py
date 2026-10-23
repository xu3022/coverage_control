#!/usr/bin/env python3
"""Regression tests for known-sensitivity centroid tracking."""

import unittest

import numpy as np


from coverage_control.config import CoverageConfig
from coverage_control.control_module import ControlModule


class KnownSensitivityControlTests(unittest.TestCase):
    def setUp(self):
        self.config = CoverageConfig()
        self.config.CONTROL_CONFIG["use_second_order_dynamics"] = False
        self.controller = ControlModule("robot1", self.config)

    def test_velocity_points_to_known_weighted_centroid(self):
        velocity = self.controller.compute_planar_velocity(
            "robot1",
            {"robot1": (1.0, 2.0)},
            centroid=(2.0, 4.0),
            coverage_gain=0.5,
        )
        np.testing.assert_allclose(velocity, (0.5, 1.0), atol=1.0e-12)

    def test_velocity_is_limited_without_changing_direction(self):
        velocity = np.asarray(self.controller.compute_planar_velocity(
            "robot1",
            {"robot1": (0.0, 0.0)},
            centroid=(6.0, 8.0),
            coverage_gain=1.0,
        ))
        self.assertAlmostEqual(np.linalg.norm(velocity), 2.0)
        np.testing.assert_allclose(velocity / 2.0, (0.6, 0.8), atol=1.0e-12)

    def test_zero_error_produces_hover_command(self):
        velocity = self.controller.compute_planar_velocity(
            "robot1",
            {"robot1": (3.0, 3.0)},
            centroid=(3.0, 3.0),
            coverage_gain=1.0,
        )
        np.testing.assert_allclose(velocity, (0.0, 0.0), atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
