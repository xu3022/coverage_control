#!/usr/bin/env python3
"""Coverage controllers for a known sensitivity field."""

import math

import numpy as np

try:
    from .config import CoverageConfig
except ImportError:
    from config import CoverageConfig


class ControlModule:
    def __init__(self, robot_name=None, config=None):
        del robot_name
        self.config = config if config is not None else CoverageConfig()
        control = self.config.CONTROL_CONFIG
        self.kp_linear = control["kp_linear"]
        self.kp_angular = control["kp_angular"]
        self.target_reached_threshold = control["target_reached_threshold"]
        self.max_linear_speed = control["max_linear_speed"]
        self.max_angular_speed = control["max_angular_speed"]
        self.max_jump_distance = control["max_jump_distance"]
        self.acceleration_gain = control.get("acceleration_gain", 4.0)
        self.use_second_order = control.get("use_second_order_dynamics", False)
        self._dt = control.get("control_period", 0.1)
        self._current_vx = 0.0
        self._current_vy = 0.0
        self.current_distance = 0.0

    @staticmethod
    def _xy(value):
        if hasattr(value, "x") and hasattr(value, "y"):
            return float(value.x), float(value.y)
        return float(value[0]), float(value[1])

    def calculate_cost(self, robot_pos, target_pos):
        robot_x, robot_y = self._xy(robot_pos)
        target_x, target_y = self._xy(target_pos)
        return math.hypot(target_x - robot_x, target_y - robot_y)

    def assign_optimal_targets(self, centroids, robot_positions, robot_names):
        """Retain the historical greedy target assignment helper."""
        if len(centroids) != len(robot_names):
            return {
                name: centroids[index]
                for index, name in enumerate(robot_names)
                if index < len(centroids)
            }
        remaining = set(range(len(centroids)))
        assignments = {}
        for name in robot_names:
            target_index = min(
                remaining,
                key=lambda index: self.calculate_cost(
                    robot_positions[name], centroids[index]
                ),
            )
            assignments[name] = centroids[target_index]
            remaining.remove(target_index)
        return assignments

    def smooth_target_transition(self, robot_name, new_target, previous_targets):
        old_target = previous_targets.get(robot_name)
        if old_target is None:
            previous_targets[robot_name] = new_target
            return new_target
        old_x, old_y = self._xy(old_target)
        new_x, new_y = self._xy(new_target)
        delta = np.array([new_x - old_x, new_y - old_y], dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance > self.max_jump_distance:
            delta *= self.max_jump_distance / distance
            result = [old_x + float(delta[0]), old_y + float(delta[1])]
        else:
            result = [new_x, new_y]
        previous_targets[robot_name] = result
        return result

    def calculate_control(
        self, robot_name, robot_positions, robot_orientations, robot_targets
    ):
        """Historical unicycle P controller, retained for compatibility."""
        command = {"linear": {"x": 0.0}, "angular": {"z": 0.0}}
        if robot_name not in robot_targets:
            return command
        target_x, target_y = self._xy(robot_targets[robot_name])
        pos_x, pos_y = self._xy(robot_positions[robot_name])
        dx, dy = target_x - pos_x, target_y - pos_y
        distance = math.hypot(dx, dy)
        self.current_distance = distance
        if distance < self.target_reached_threshold:
            return command
        angle_error = math.atan2(dy, dx) - robot_orientations[robot_name]
        angle_error = (angle_error + math.pi) % (2.0 * math.pi) - math.pi
        command["linear"]["x"] = min(
            self.kp_linear * distance, self.max_linear_speed
        )
        command["angular"]["z"] = float(np.clip(
            self.kp_angular * angle_error,
            -self.max_angular_speed,
            self.max_angular_speed,
        ))
        return command

    def set_current_velocity(self, vx, vy):
        self._current_vx = float(vx)
        self._current_vy = float(vy)

    def reset_velocity(self):
        self._current_vx = 0.0
        self._current_vy = 0.0

    def get_current_velocity(self):
        return self._current_vx, self._current_vy

    def compute_planar_velocity(
        self, robot_name, robot_positions, centroid, coverage_gain=1.0
    ):
        """Return ``k_m(c_i-p_i)`` for the known weighted centroid ``c_i``."""
        position_xy = np.asarray(self._xy(robot_positions[robot_name]), dtype=float)
        centroid_xy = np.asarray(centroid, dtype=float).reshape(-1)[:2]
        desired_velocity = float(coverage_gain) * (centroid_xy - position_xy)
        self.current_distance = float(np.linalg.norm(centroid_xy - position_xy))

        speed = float(np.linalg.norm(desired_velocity))
        if speed < self.target_reached_threshold:
            desired_velocity[:] = 0.0
        elif speed > self.max_linear_speed:
            desired_velocity *= self.max_linear_speed / speed

        if self.use_second_order:
            current_velocity = np.array(
                [self._current_vx, self._current_vy], dtype=float
            )
            acceleration = self.acceleration_gain * (
                desired_velocity - current_velocity
            )
            desired_velocity = current_velocity + acceleration * self._dt
            speed = float(np.linalg.norm(desired_velocity))
            if speed > self.max_linear_speed:
                desired_velocity *= self.max_linear_speed / speed
            self._current_vx = float(desired_velocity[0])
            self._current_vy = float(desired_velocity[1])

        return float(desired_velocity[0]), float(desired_velocity[1])
