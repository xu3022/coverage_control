#!/usr/bin/env python3
"""Multi-robot coverage control using a fully known sensitivity field."""

import csv
import math
import os
import time

import numpy as np
from shapely.geometry import Polygon

try:
    from .config import default_config
    from .control_module import ControlModule
    from .plotting_module import PlottingModule
    from .robot_model import RobotManager
    from .voronoi_partition import VoronoiPartition
except ImportError:
    try:
        from coverage_control.config import default_config
        from coverage_control.control_module import ControlModule
        from coverage_control.plotting_module import PlottingModule
        from coverage_control.robot_model import RobotManager
        from coverage_control.voronoi_partition import VoronoiPartition
    except ImportError:
        from config import default_config
        from control_module import ControlModule
        from plotting_module import PlottingModule
        from robot_model import RobotManager
        from voronoi_partition import VoronoiPartition


class VoronoiCoverageController:
    """Closed-loop Lloyd coverage controller using the configured true field."""

    def __init__(self, config=None):
        self.config = config if config is not None else default_config
        self.config.validate_config()
        self.num_robots = self.config.ROBOT_CONFIG["num_robots"]
        self.robot_names = list(self.config.ROBOT_CONFIG["robot_names"])
        self.dt = float(self.config.CONTROL_CONFIG["control_period"])
        self.coverage_gain = float(
            self.config.CONTROL_CONFIG["coverage_gain"]
        )

        random_config = self.config.RANDOM_CONFIG
        if random_config.get("fix_random_seed", True):
            np.random.seed(int(random_config.get("random_seed", 42)))

        self.robot_manager = RobotManager(
            self.num_robots,
            self.config.ROBOT_INITIAL_POSITIONS,
            self.config,
        )
        self.robot_positions = self.robot_manager.get_robot_positions()
        self.control_modules = {
            name: ControlModule(name, self.config) for name in self.robot_names
        }

        self.work_area_boundary = list(self.config.WORK_AREA_BOUNDARY)
        self.work_area_polygon = Polygon(self.work_area_boundary)
        self.voronoi_partitioner = VoronoiPartition(
            self.work_area_polygon, self.config
        )
        self.plotting_module = (
            PlottingModule(self.work_area_boundary)
            if self.config.EXPERIMENT_CONFIG["enable_plotting"]
            else None
        )

        density = self.config.DENSITY_FUNCTION_CONFIG
        self.gaussian_params = [
            self._normalise_field_component(component)
            for component in density["gaussian_functions"]
        ]
        self.gaussian_weights = list(density["gaussian_weights"])
        self.base_density = float(density["base_density"])
        self.time_counter = 0
        self.start_time = time.time()
        self.robot_targets = {}

        self._grid_enabled = bool(
            self.config.GRID_SENSITIVITY_CONFIG.get("enabled", False)
        )
        self._grid_resolution = int(
            self.config.GRID_SENSITIVITY_CONFIG["resolution"]
        )
        self._grid_cell_size = float(
            self.config.GRID_SENSITIVITY_CONFIG["cell_size"]
        )
        self._sensitivity_grid = None
        if self._grid_enabled:
            self._rebuild_sensitivity_grid()

        self.scripts_dir = self.config.EXPERIMENT_CONFIG["scripts_dir"]
        os.makedirs(self.scripts_dir, exist_ok=True)
        self.cost_function_dir = os.path.join(self.scripts_dir, "cost function")
        self.distance_csv_dir = os.path.join(self.scripts_dir, "distance")
        os.makedirs(self.cost_function_dir, exist_ok=True)
        os.makedirs(self.distance_csv_dir, exist_ok=True)
        self.cost_function_csv_path = os.path.join(
            self.cost_function_dir, "known-sensitivity.csv"
        )
        self.distance_csv_path = os.path.join(
            self.distance_csv_dir, "known-sensitivity-distance.csv"
        )
        self._last_saved_distance_timestep = None
        self._initialize_output_files()
        self.initialize_voronoi_regions()

    @staticmethod
    def _normalise_field_component(component):
        params = dict(component)
        motion_type = params.get("motion_type", "static")
        if motion_type == "circular":
            params.setdefault("current_angle", params.get("circular_initial_angle", 0.0))
        elif params.get("movable", False) and motion_type == "linear":
            params.setdefault("fixed_direction", 3.0 * np.pi / 4.0)
            params.setdefault("speed", 0.1)
            params.setdefault("boundary_min", 0.0)
            params.setdefault("boundary_max", 10.0)
        return params

    def _initialize_output_files(self):
        with open(
            self.cost_function_csv_path, "w", newline="", encoding="utf-8"
        ) as stream:
            csv.writer(stream).writerow(["timestamp", "coverage_cost"])
        with open(
            self.distance_csv_path, "w", newline="", encoding="utf-8"
        ) as stream:
            csv.writer(stream).writerow(["timestep"] + self.robot_names)

    def initialize_voronoi_regions(self):
        self.robot_positions = self.robot_manager.get_robot_positions()
        regions = self.voronoi_partitioner.update_voronoi_regions(
            self.robot_positions, self.robot_names, self.num_robots
        )
        if regions is None or isinstance(regions, tuple):
            raise RuntimeError("无法根据初始位置生成Voronoi区域")
        statistics = (
            self.voronoi_partitioner
            .calculate_all_weighted_centroids_with_true_sensitivity(
                self.robot_positions
            )
        )
        self.voronoi_partitioner.voronoi_centroids = [
            statistics[name][1] for name in self.robot_names
        ]
        self.robot_targets = {
            name: statistics[name][1] for name in self.robot_names
        }
        self.save_coverage_cost_function_to_csv(
            self.compute_coverage_cost_function()
        )

    def _update_moving_gaussians(self):
        for params in self.gaussian_params:
            if not params.get("movable", False):
                continue
            motion_type = params.get("motion_type", "linear")
            if motion_type == "linear_timed":
                start = int(params.get("start_cycle", 100))
                end = int(params.get("end_cycle", 200))
                start_xy = np.array([
                    params.get("start_center_x", params["center_x"]),
                    params.get("start_center_y", params["center_y"]),
                ], dtype=float)
                end_xy = np.array([
                    params.get("end_center_x", params["center_x"]),
                    params.get("end_center_y", params["center_y"]),
                ], dtype=float)
                if self.time_counter < start:
                    center = start_xy
                elif self.time_counter <= end:
                    duration = max(1, end - start)
                    alpha = (self.time_counter - start) / duration
                    center = start_xy + alpha * (end_xy - start_xy)
                else:
                    center = end_xy
                params["center_x"], params["center_y"] = map(float, center)
            elif motion_type == "circular":
                angle = (
                    float(params.get("current_angle", 0.0))
                    + float(params.get("circular_angular_speed", 0.1)) * self.dt
                ) % (2.0 * np.pi)
                params["current_angle"] = angle
                params["center_x"] = float(
                    params.get("circular_center_x", 5.0)
                    + params.get("circular_radius", 1.0) * np.cos(angle)
                )
                params["center_y"] = float(
                    params.get("circular_center_y", 5.0)
                    + params.get("circular_radius", 1.0) * np.sin(angle)
                )
            else:
                angle = float(params.get("fixed_direction", 3.0 * np.pi / 4.0))
                speed = float(params.get("speed", 0.1))
                next_x = params["center_x"] + speed * np.cos(angle) * self.dt
                next_y = params["center_y"] + speed * np.sin(angle) * self.dt
                lower = float(params.get("boundary_min", 0.0))
                upper = float(params.get("boundary_max", 10.0))
                if not lower <= next_x <= upper or not lower <= next_y <= upper:
                    params["fixed_direction"] = angle + np.pi
                    next_x = np.clip(next_x, lower, upper)
                    next_y = np.clip(next_y, lower, upper)
                params["center_x"] = float(next_x)
                params["center_y"] = float(next_y)

    def _rebuild_sensitivity_grid(self):
        cell_size = self._grid_cell_size
        coordinates = (np.arange(self._grid_resolution) + 0.5) * cell_size
        grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
        self.voronoi_partitioner.set_gaussian_params(
            self.gaussian_params, self.gaussian_weights
        )
        previous_enabled = self.voronoi_partitioner._grid_enabled
        self.voronoi_partitioner._grid_enabled = False
        self._sensitivity_grid = self.voronoi_partitioner.sample_true_density(
            grid_x, grid_y
        )
        self.voronoi_partitioner._grid_enabled = previous_enabled
        self.voronoi_partitioner.set_sensitivity_grid(
            self._sensitivity_grid,
            self._grid_resolution,
            self._grid_cell_size,
        )

    def sample_true_density(self, x, y):
        return self.voronoi_partitioner.sample_true_density(x, y)

    def _plot_state(self):
        if self.plotting_module is None:
            return
        self.plotting_module.update_gaussian_params(self.gaussian_params)
        positions = {
            name: (self.robot_positions[name].x, self.robot_positions[name].y)
            for name in self.robot_names
        }
        self.plotting_module.plot_voronoi_regions(
            coords=[],
            robot_names=self.robot_names,
            voronoi_areas=self.voronoi_partitioner.voronoi_areas,
            robot_positions=positions,
            robot_orientations=self.robot_manager.get_robot_orientations(),
            robot_targets=self.robot_targets,
            position_received={name: True for name in self.robot_names},
            centroids=self.voronoi_partitioner.voronoi_centroids,
            timestep=self.time_counter,
        )

    def control_timer_callback(self):
        self.time_counter += 1
        self._update_moving_gaussians()
        self.voronoi_partitioner.set_gaussian_params(
            self.gaussian_params, self.gaussian_weights
        )
        if self._grid_enabled:
            self._rebuild_sensitivity_grid()

        self.robot_positions = self.robot_manager.get_robot_positions()
        regions = self.voronoi_partitioner.update_voronoi_regions(
            self.robot_positions, self.robot_names, self.num_robots
        )
        if regions is None or isinstance(regions, tuple):
            raise RuntimeError(
                f"t={self.time_counter} Voronoi区域更新失败"
            )

        statistics = (
            self.voronoi_partitioner
            .calculate_all_weighted_centroids_with_true_sensitivity(
                self.robot_positions
            )
        )
        self.robot_targets = {
            name: statistics[name][1] for name in self.robot_names
        }
        self.voronoi_partitioner.voronoi_centroids = [
            self.robot_targets[name] for name in self.robot_names
        ]

        coverage_cost = self.compute_coverage_cost_function()
        self.save_coverage_cost_function_to_csv(coverage_cost)
        self._plot_state()

        commands = {}
        for name in self.robot_names:
            control = self.control_modules[name]
            if self.config.CONTROL_CONFIG.get("use_second_order_dynamics", False):
                velocity = self.robot_manager.robots[name].velocity
                control.set_current_velocity(velocity[0], velocity[1])
            commands[name] = control.compute_planar_velocity(
                name,
                self.robot_positions,
                self.robot_targets[name],
                coverage_gain=self.coverage_gain,
            )
        self.robot_manager.update_all_robots(
            self.robot_targets,
            self.dt,
            planar_velocity_commands=commands,
        )
        return coverage_cost

    def compute_coverage_cost_function(self):
        """Return ``-sum_i integral_Vi ||p_i-q||^2 phi(q) dq``."""
        resolution = float(self.config.VORONOI_CONFIG["grid_resolution"])
        min_x, min_y, max_x, max_y = self.config.WORK_AREA_CONFIG["boundary"]
        x_values = np.arange(min_x, max_x, resolution)
        y_values = np.arange(min_y, max_y, resolution)
        grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        positions = np.asarray([
            [self.robot_positions[name].x, self.robot_positions[name].y]
            for name in self.robot_names
        ], dtype=float)
        squared_distances = np.sum(
            (points[:, None, :] - positions[None, :, :]) ** 2, axis=2
        )
        nearest_squared_distance = np.min(squared_distances, axis=1)
        density = self.sample_true_density(points[:, 0], points[:, 1])
        return -float(np.sum(
            nearest_squared_distance * density
        ) * resolution ** 2)

    def save_coverage_cost_function_to_csv(self, coverage_cost):
        with open(
            self.cost_function_csv_path, "a", newline="", encoding="utf-8"
        ) as stream:
            csv.writer(stream).writerow([self.time_counter, coverage_cost])

    def save_all_robot_distances(self):
        if not self.config.EXPERIMENT_CONFIG.get(
            "enable_distance_logging", True
        ):
            return
        timestep = int(self.time_counter)
        if self._last_saved_distance_timestep == timestep:
            return
        row = [timestep] + [
            self.control_modules[name].current_distance
            for name in self.robot_names
        ]
        with open(
            self.distance_csv_path, "a", newline="", encoding="utf-8"
        ) as stream:
            csv.writer(stream).writerow(row)
        self._last_saved_distance_timestep = timestep

    def run(self):
        print("开始运行已知敏感度覆盖控制器")
        try:
            while True:
                loop_start = time.time()
                try:
                    self.control_timer_callback()
                finally:
                    self.save_all_robot_distances()
                max_steps = int(
                    self.config.EXPERIMENT_CONFIG["max_control_timesteps"]
                )
                if self.time_counter >= max_steps:
                    break
                elapsed = time.time() - loop_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)
                if (
                    self.config.EXPERIMENT_CONFIG["max_runtime"] > 0
                    and time.time() - self.start_time
                    > self.config.EXPERIMENT_CONFIG["max_runtime"]
                ):
                    break
        except KeyboardInterrupt:
            print("收到中断信号")
        finally:
            print(f"控制器已停止，完成 {self.time_counter} 个控制周期")


def main():
    VoronoiCoverageController().run()


if __name__ == "__main__":
    main()
