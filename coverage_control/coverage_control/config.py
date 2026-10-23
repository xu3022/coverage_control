#!/usr/bin/env python3
"""Configuration for known-sensitivity coverage control only."""

import os
from pathlib import Path

import numpy as np


class CoverageConfig:
    """Configuration for known-sensitivity coverage control."""

    def __init__(self):
        num_robots = 20
        self.ROBOT_CONFIG = {
            "num_robots": num_robots,
            "robot_names": self._generate_robot_names(num_robots),
        }
        self.CONTROL_CONFIG = {
            "control_period": 0.1,
            "voronoi_update_period": 0.1,
            "kp_linear": 0.5,
            "kp_angular": 8.0,
            "target_reached_threshold": 0.01,
            "max_linear_speed": 2.0,
            "max_angular_speed": 15.0,
            "max_jump_distance": 1.0,
            "coverage_gain": 1.0,
            "acceleration_gain": 10.0,
            "use_second_order_dynamics": False,
        }
        self.QUADROTOR_CONFIG = {
            "mass": 1.0,
            "inertia": [0.02, 0.02, 0.04],
            "arm_length": 0.20,
            "thrust_coefficient": 1.0e-5,
            "torque_coefficient": 2.0e-7,
            "motor_time_constant": 0.03,
            "min_rotor_speed": 0.0,
            "max_rotor_speed": 1000.0,
            "gravity": 9.81,
            "translational_drag": [0.10, 0.10, 0.20],
            "angular_drag": [0.02, 0.02, 0.02],
            "desired_altitude": 2.0,
            "desired_yaw": 0.0,
            "coverage_command_mode": "velocity",
            "max_horizontal_speed": 2.0,
            "max_horizontal_acceleration": 3.0,
            "max_vertical_acceleration": 4.0,
            "max_tilt_angle": np.deg2rad(30.0),
            "horizontal_velocity_gain": 2.0,
            "use_velocity_feedforward": True,
            "velocity_feedforward_filter_time_constant": 0.10,
            "altitude_position_gain": 4.0,
            "altitude_velocity_gain": 3.0,
            "attitude_gain": [6.0, 6.0, 2.5],
            "angular_rate_gain": [0.35, 0.35, 0.25],
            "integration_substeps": 10,
        }
        self.WORK_AREA_CONFIG = {
            "boundary": [0.0, 0.0, 10.0, 10.0],
            "resolution": 0.1,
        }
        self.WORK_AREA_BOUNDARY = [
            (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)
        ]
        self.DENSITY_FUNCTION_CONFIG = {
            "type": "gaussian_mixture",
            "gaussian_functions": [
                {
                    "name": "gaussian1",
                    "function_type": "sigmoid_cutoff",
                    "center_x": 5.0,
                    "center_y": 5.0,
                    "amplitude": 50.0,
                    "variance": 2.0,
                    "movable": False,
                    "cutoff_direction": 0.0,
                    "cutoff_smoothness": 0.2,
                },
                {
                    "name": "gaussian2",
                    "function_type": "donut",
                    "center_x": 7.0,
                    "center_y": 7.0,
                    "amplitude": 50.0,
                    "variance": 0.5,
                    "movable": False,
                    "ring_radius": 1.5,
                },
                {
                    "name": "gaussian3",
                    "function_type": "gaussian",
                    "center_x": 8.0,
                    "center_y": 2.0,
                    "amplitude": 100.0,
                    "variance": 1.0,
                    "movable": True,
                    "motion_type": "linear_timed",
                    "start_cycle": 50,
                    "end_cycle": 55,
                    "start_center_x": 8.0,
                    "start_center_y": 2.0,
                    "end_center_x": 2.0,
                    "end_center_y": 8.0,
                },
            ],
            "gaussian_weights": [0.3, 0.4, 0.4],
            "base_density": 1.0,
        }
        self.GRID_SENSITIVITY_CONFIG = {
            "enabled": False,
            "resolution": 400,
            "cell_size": 10.0 / 400,
        }
        self.VORONOI_CONFIG = {
            "min_distance": 0.5,
            "perturbation": 0.1,
            "grid_size": 0.1,
            "grid_resolution": 0.1,
            "margin": 0.0,
        }
        self.EXPERIMENT_CONFIG = {
            "experiment_name": "known_sensitivity_coverage",
            "distance_log_name": "known-sensitivity",
            "enable_distance_logging": True,
            "distance_log_unique_run": False,
            "max_runtime": 7200,
            "max_control_timesteps": int(
                os.environ.get("COVERAGE_MAX_CONTROL_TIMESTEPS", "200")
            ),
            "enable_plotting": (
                os.environ.get("COVERAGE_ENABLE_PLOTTING", "1") != "0"
            ),
            "save_results": True,
            "scripts_dir": os.environ.get(
                "COVERAGE_SCRIPTS_DIR",
                str(Path(__file__).resolve().parents[1] / "scripts"),
            ),
        }
        self.RANDOM_CONFIG = {"fix_random_seed": True, "random_seed": 42}
        self.ROBOT_INITIAL_POSITIONS = self._generate_initial_positions()

    @staticmethod
    def _generate_robot_names(num_robots=20):
        return [f"robot{i}" for i in range(1, num_robots + 1)]

    def _load_initial_positions_from_file(self):
        positions = {}
        override_path = os.environ.get("COVERAGE_INITIAL_POSITIONS_FILE")
        file_path = override_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "position",
            "initial_positions.txt",
        )
        try:
            with open(file_path, "r", encoding="utf-8") as stream:
                for raw_line in stream:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) == 3:
                        positions[parts[0]] = [float(parts[1]), float(parts[2])]
        except FileNotFoundError:
            if override_path:
                raise
            return self._generate_random_initial_positions()
        missing = [
            name for name in self.ROBOT_CONFIG["robot_names"]
            if name not in positions
        ]
        if missing:
            raise ValueError(f"初始位置文件缺少机器人: {', '.join(missing)}")
        return positions

    def _generate_random_initial_positions(self):
        rng = np.random.default_rng(self.RANDOM_CONFIG["random_seed"])
        min_x, min_y, max_x, max_y = self.WORK_AREA_CONFIG["boundary"]
        return {
            name: [rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)]
            for name in self.ROBOT_CONFIG["robot_names"]
        }

    def _generate_initial_positions(self):
        return self._load_initial_positions_from_file()

    def get_robot_names(self):
        return list(self.ROBOT_CONFIG["robot_names"])

    def get_num_robots(self):
        return int(self.ROBOT_CONFIG["num_robots"])

    def get_control_periods(self):
        return {
            "control": self.CONTROL_CONFIG["control_period"],
            "voronoi": self.CONTROL_CONFIG["voronoi_update_period"],
        }

    def get_pid_params(self):
        return {
            "kp_linear": self.CONTROL_CONFIG["kp_linear"],
            "kp_angular": self.CONTROL_CONFIG["kp_angular"],
            "target_threshold": self.CONTROL_CONFIG["target_reached_threshold"],
        }

    def validate_config(self):
        errors = []
        if self.ROBOT_CONFIG["num_robots"] < 3:
            errors.append("Voronoi分区至少需要3个机器人")
        if self.CONTROL_CONFIG["control_period"] <= 0:
            errors.append("控制周期必须大于0")
        if self.CONTROL_CONFIG["coverage_gain"] < 0:
            errors.append("覆盖增益不能为负")
        if self.VORONOI_CONFIG["grid_resolution"] <= 0:
            errors.append("Voronoi积分网格分辨率必须大于0")
        if len(self.DENSITY_FUNCTION_CONFIG["gaussian_weights"]) != len(
            self.DENSITY_FUNCTION_CONFIG["gaussian_functions"]
        ):
            errors.append("敏感度分量数量与权重数量不一致")
        max_steps = self.EXPERIMENT_CONFIG["max_control_timesteps"]
        if not isinstance(max_steps, int) or max_steps <= 0:
            errors.append("最大控制周期必须为正整数")
        if errors:
            raise ValueError("配置验证失败: " + "; ".join(errors))
        return True

    def print_summary(self):
        print("=== 已知敏感度覆盖控制配置 ===")
        print(f"机器人数量: {self.get_num_robots()}")
        print(f"控制周期: {self.CONTROL_CONFIG['control_period']} s")
        print(f"覆盖增益: {self.CONTROL_CONFIG['coverage_gain']}")
        print("敏感度来源: 配置中的已知解析场")


default_config = CoverageConfig()


if __name__ == "__main__":
    default_config.validate_config()
    default_config.print_summary()
