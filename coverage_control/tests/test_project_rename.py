#!/usr/bin/env python3
"""Regression checks for the renamed pure-Python project interface."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coverage_control import CoverageConfig


class ProjectRenameTests(unittest.TestCase):
    def test_coverage_environment_names_are_applied(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "COVERAGE_MAX_CONTROL_TIMESTEPS": "7",
                "COVERAGE_ENABLE_PLOTTING": "0",
                "COVERAGE_SCRIPTS_DIR": directory,
            },
            clear=False,
        ):
            config = CoverageConfig()

        self.assertEqual(config.EXPERIMENT_CONFIG["max_control_timesteps"], 7)
        self.assertFalse(config.EXPERIMENT_CONFIG["enable_plotting"])
        self.assertEqual(config.EXPERIMENT_CONFIG["scripts_dir"], directory)

    def test_default_output_is_inside_project(self):
        config = CoverageConfig()
        project_root = Path(__file__).resolve().parents[1]
        expected = project_root / "scripts"
        self.assertEqual(
            Path(config.EXPERIMENT_CONFIG["scripts_dir"]).resolve(),
            expected.resolve(),
        )


if __name__ == "__main__":
    unittest.main()
