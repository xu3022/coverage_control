#!/usr/bin/env python3
"""
固定高度二维覆盖任务使用的四旋翼刚体动力学与三级飞行控制器。

控制链路：
    覆盖控制（世界系期望水平速度）
    -> 水平速度/高度外环（期望总作用力）
    -> SO(3) 姿态内环（期望力矩）
    -> 电机分配与六自由度刚体动力学。

坐标约定：世界坐标系 z 轴向上，旋转矩阵 R 将机体系向量变换到世界系，
机体 b3 轴为旋翼总推力方向，四元数顺序为 [w, x, y, z]。
"""

import math

import numpy as np


_EPS = 1.0e-12


def _normalize_quaternion(quaternion):
    """返回归一化的 [w, x, y, z] 四元数。"""
    quaternion = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(quaternion)
    if norm < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quaternion / norm


def _quaternion_multiply(q1, q2):
    """Hamilton 四元数乘法 q1*q2。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quaternion_to_rotation_matrix(quaternion):
    """将 [w, x, y, z] 转为机体系到世界系旋转矩阵。"""
    w, x, y, z = _normalize_quaternion(quaternion)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ])


def _vee(skew_matrix):
    """将反对称矩阵映射到 R^3。"""
    return np.array([
        skew_matrix[2, 1],
        skew_matrix[0, 2],
        skew_matrix[1, 0],
    ])


class QuadrotorRobot:
    """四旋翼状态、三级飞控、电机模型与六自由度动力学。"""

    def __init__(self, robot_id, initial_position, config=None, mass=None,
                 max_speed=None):
        """
        Args:
            robot_id: 机器人名称或编号。
            initial_position: [x, y] 或 [x, y, z]。
            config: CoverageConfig 实例。
            mass: 兼容旧接口的质量覆盖值。
            max_speed: 兼容旧接口的水平速度上限覆盖值。
        """
        if config is None:
            try:
                from .config import CoverageConfig
            except ImportError:
                from config import CoverageConfig
            config = CoverageConfig()

        self.id = robot_id
        self.config = config
        quad_config = dict(config.QUADROTOR_CONFIG)
        if mass is not None:
            quad_config['mass'] = mass
        if max_speed is not None:
            quad_config['max_horizontal_speed'] = max_speed

        self.mass = float(quad_config['mass'])
        self.inertia = np.diag(np.asarray(quad_config['inertia'], dtype=float))
        self.inertia_inverse = np.linalg.inv(self.inertia)
        self.arm_length = float(quad_config['arm_length'])
        self.thrust_coefficient = float(quad_config['thrust_coefficient'])
        self.torque_coefficient = float(quad_config['torque_coefficient'])
        self.motor_time_constant = float(quad_config['motor_time_constant'])
        self.min_rotor_speed = float(quad_config['min_rotor_speed'])
        self.max_rotor_speed = float(quad_config['max_rotor_speed'])
        self.gravity = float(quad_config['gravity'])

        self.translational_drag = np.asarray(
            quad_config['translational_drag'], dtype=float
        )
        self.angular_drag = np.asarray(
            quad_config['angular_drag'], dtype=float
        )

        self.desired_altitude = float(quad_config['desired_altitude'])
        self.desired_yaw = float(quad_config['desired_yaw'])
        self.max_horizontal_speed = float(quad_config['max_horizontal_speed'])
        self.max_horizontal_acceleration = float(
            quad_config['max_horizontal_acceleration']
        )
        self.max_vertical_acceleration = float(
            quad_config['max_vertical_acceleration']
        )
        self.max_tilt_angle = float(quad_config['max_tilt_angle'])
        self.horizontal_velocity_gain = float(
            quad_config['horizontal_velocity_gain']
        )
        self.use_velocity_feedforward = bool(
            quad_config.get('use_velocity_feedforward', True)
        )
        self.velocity_feedforward_filter_time_constant = float(
            quad_config.get('velocity_feedforward_filter_time_constant', 0.10)
        )
        self.altitude_position_gain = float(
            quad_config['altitude_position_gain']
        )
        self.altitude_velocity_gain = float(
            quad_config['altitude_velocity_gain']
        )
        self.attitude_gain = np.asarray(quad_config['attitude_gain'], dtype=float)
        self.angular_rate_gain = np.asarray(
            quad_config['angular_rate_gain'], dtype=float
        )
        self.integration_substeps = int(quad_config['integration_substeps'])
        self._validate_parameters()

        # 电机顺序：+x、+y、-x、-y；偏航反扭矩符号为 +、-、+、-。
        k_f = self.thrust_coefficient
        k_m = self.torque_coefficient
        arm = self.arm_length
        self.allocation_matrix = np.array([
            [k_f, k_f, k_f, k_f],
            [0.0, arm * k_f, 0.0, -arm * k_f],
            [-arm * k_f, 0.0, arm * k_f, 0.0],
            [k_m, -k_m, k_m, -k_m],
        ])
        self.allocation_matrix_inverse = np.linalg.inv(self.allocation_matrix)

        initial_position = np.asarray(initial_position, dtype=float).reshape(-1)
        if initial_position.size < 2:
            raise ValueError('initial_position 至少需要包含 [x, y]')
        initial_z = (
            float(initial_position[2])
            if initial_position.size >= 3 else self.desired_altitude
        )
        self.position = np.array(
            [initial_position[0], initial_position[1], initial_z], dtype=float
        )
        self.velocity = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.angular_velocity = np.zeros(3)

        hover_speed = math.sqrt(
            self.mass * self.gravity / (4.0 * self.thrust_coefficient)
        )
        hover_speed = float(np.clip(
            hover_speed, self.min_rotor_speed, self.max_rotor_speed
        ))
        self.rotor_speeds = np.full(4, hover_speed)
        self.commanded_rotor_speeds = self.rotor_speeds.copy()

        self.desired_planar_velocity = np.zeros(2)
        self.desired_planar_acceleration = np.zeros(2)
        self.target = None

        # 飞控诊断量，便于后续记录和参数分析。
        self.desired_force = np.array([0.0, 0.0, self.mass * self.gravity])
        self.desired_rotation = np.eye(3)
        self.commanded_thrust = self.mass * self.gravity
        self.commanded_torque = np.zeros(3)
        self.actual_thrust = self.mass * self.gravity
        self.actual_torque = np.zeros(3)

    def _validate_parameters(self):
        if self.mass <= 0.0:
            raise ValueError('QUADROTOR_CONFIG.mass 必须大于 0')
        if np.any(np.diag(self.inertia) <= 0.0):
            raise ValueError('QUADROTOR_CONFIG.inertia 各分量必须大于 0')
        if self.arm_length <= 0.0:
            raise ValueError('QUADROTOR_CONFIG.arm_length 必须大于 0')
        if self.thrust_coefficient <= 0.0 or self.torque_coefficient <= 0.0:
            raise ValueError('旋翼推力/反扭矩系数必须大于 0')
        if self.motor_time_constant <= 0.0:
            raise ValueError('motor_time_constant 必须大于 0')
        if self.max_rotor_speed <= self.min_rotor_speed:
            raise ValueError('max_rotor_speed 必须大于 min_rotor_speed')
        if self.integration_substeps < 1:
            raise ValueError('integration_substeps 必须至少为 1')
        if self.max_horizontal_acceleration <= 0.0:
            raise ValueError('max_horizontal_acceleration 必须大于 0')
        if self.velocity_feedforward_filter_time_constant < 0.0:
            raise ValueError(
                'velocity_feedforward_filter_time_constant 不能小于 0'
            )
        if self.max_vertical_acceleration <= 0.0:
            raise ValueError('max_vertical_acceleration 必须大于 0')
        if not 0.0 < self.max_tilt_angle < 0.5 * math.pi:
            raise ValueError('max_tilt_angle 必须位于 (0, pi/2)')

    def set_target(self, target_position):
        """保存二维目标点，用于没有覆盖速度指令时的兼容回退。"""
        target = np.asarray(target_position, dtype=float).reshape(-1)
        if target.size < 2:
            raise ValueError('target_position 至少需要包含 [x, y]')
        self.target = target[:2].copy()

    def set_planar_velocity_command(
        self,
        velocity_x,
        velocity_y,
        acceleration_x=None,
        acceleration_y=None,
        command_dt=None,
    ):
        """
        设置世界系期望水平速度和可选的期望加速度前馈。

        若上层显式给出 ``acceleration_x/y``，直接采用解析前馈；否则在
        ``command_dt`` 有效时，由相邻速度指令的差分估计前馈并低通滤波。
        未提供这两类信息时前馈为零，以兼容已有调用方式。
        """
        previous_desired_velocity = self.desired_planar_velocity.copy()
        desired_velocity = np.array([velocity_x, velocity_y], dtype=float)
        speed = np.linalg.norm(desired_velocity)
        if speed > self.max_horizontal_speed and speed > _EPS:
            desired_velocity *= self.max_horizontal_speed / speed
        self.desired_planar_velocity = desired_velocity

        has_acceleration_x = acceleration_x is not None
        has_acceleration_y = acceleration_y is not None
        if has_acceleration_x != has_acceleration_y:
            raise ValueError('acceleration_x 和 acceleration_y 必须同时提供')

        if not self.use_velocity_feedforward:
            desired_acceleration = np.zeros(2)
        elif has_acceleration_x:
            desired_acceleration = np.array(
                [acceleration_x, acceleration_y], dtype=float
            )
        elif command_dt is not None:
            command_dt = float(command_dt)
            if command_dt <= 0.0:
                raise ValueError('command_dt 必须大于 0')
            raw_acceleration = (
                desired_velocity - previous_desired_velocity
            ) / command_dt
            filter_time_constant = (
                self.velocity_feedforward_filter_time_constant
            )
            filter_weight = (
                1.0
                if filter_time_constant <= _EPS
                else command_dt / (filter_time_constant + command_dt)
            )
            desired_acceleration = (
                (1.0 - filter_weight) * self.desired_planar_acceleration
                + filter_weight * raw_acceleration
            )
        else:
            desired_acceleration = np.zeros(2)

        acceleration_norm = np.linalg.norm(desired_acceleration)
        if (
            acceleration_norm > self.max_horizontal_acceleration
            and acceleration_norm > _EPS
        ):
            desired_acceleration *= (
                self.max_horizontal_acceleration / acceleration_norm
            )
        self.desired_planar_acceleration = desired_acceleration

    def set_velocity_command(self, velocity_x, velocity_y, command_dt=None):
        """旧世界系速度接口的兼容别名。"""
        self.set_planar_velocity_command(
            velocity_x, velocity_y, command_dt=command_dt
        )

    def set_body_velocity_command(
        self, linear_velocity, angular_velocity, command_dt=None
    ):
        """
        兼容旧单车接口，仅将机头方向线速度转换为世界系水平速度。

        angular_velocity 不再直接积分偏航；偏航由 desired_yaw 闭环跟踪。
        """
        del angular_velocity
        yaw = self.get_orientation()
        self.set_planar_velocity_command(
            linear_velocity * math.cos(yaw),
            linear_velocity * math.sin(yaw),
            command_dt=command_dt,
        )

    def clear_velocity_command(self):
        """清零期望水平速度，飞行器保持悬停。"""
        self.desired_planar_velocity[:] = 0.0
        self.desired_planar_acceleration[:] = 0.0

    def calculate_control(self, kp_linear=1.0, target_threshold=0.1):
        """根据二维目标点生成兼容用期望水平速度。"""
        if self.target is None:
            return 0.0, 0.0
        position_error = self.target - self.position[:2]
        if np.linalg.norm(position_error) < target_threshold:
            return 0.0, 0.0
        desired_velocity = kp_linear * position_error
        speed = np.linalg.norm(desired_velocity)
        if speed > self.max_horizontal_speed and speed > _EPS:
            desired_velocity *= self.max_horizontal_speed / speed
        return float(desired_velocity[0]), float(desired_velocity[1])

    def _unpack_state(self, state):
        return (
            state[0:3],
            state[3:6],
            _normalize_quaternion(state[6:10]),
            state[10:13],
            np.clip(state[13:17], self.min_rotor_speed, self.max_rotor_speed),
        )

    def _outer_loop(self, state):
        """水平速度/高度外环，计算期望总作用力 F_d。"""
        position, velocity, _, _, _ = self._unpack_state(state)

        horizontal_acceleration = (
            self.desired_planar_acceleration
            + self.horizontal_velocity_gain * (
                self.desired_planar_velocity - velocity[:2]
            )
        )
        horizontal_norm = np.linalg.norm(horizontal_acceleration)
        if (
            horizontal_norm > self.max_horizontal_acceleration
            and horizontal_norm > _EPS
        ):
            horizontal_acceleration *= (
                self.max_horizontal_acceleration / horizontal_norm
            )

        vertical_acceleration = (
            self.altitude_position_gain * (self.desired_altitude - position[2])
            - self.altitude_velocity_gain * velocity[2]
        )
        vertical_acceleration = float(np.clip(
            vertical_acceleration,
            -self.max_vertical_acceleration,
            self.max_vertical_acceleration,
        ))
        desired_acceleration = np.array([
            horizontal_acceleration[0],
            horizontal_acceleration[1],
            vertical_acceleration,
        ])

        # F_d=m(a_d+g*e3)，并补偿模型中的线性阻力。
        desired_force = self.mass * (
            desired_acceleration + np.array([0.0, 0.0, self.gravity])
        ) + self.translational_drag * velocity

        desired_force[2] = max(desired_force[2], 0.1 * self.mass * self.gravity)
        max_horizontal_force = desired_force[2] * math.tan(self.max_tilt_angle)
        horizontal_force_norm = np.linalg.norm(desired_force[:2])
        if horizontal_force_norm > max_horizontal_force and horizontal_force_norm > _EPS:
            desired_force[:2] *= max_horizontal_force / horizontal_force_norm
        return desired_force

    def _desired_rotation_from_force(self, desired_force):
        """由 F_d 方向和固定偏航角 psi_d 构造期望旋转矩阵 R_d。"""
        force_norm = np.linalg.norm(desired_force)
        desired_b3 = (
            desired_force / force_norm
            if force_norm > _EPS else np.array([0.0, 0.0, 1.0])
        )
        heading_b1 = np.array([
            math.cos(self.desired_yaw), math.sin(self.desired_yaw), 0.0
        ])
        desired_b2 = np.cross(desired_b3, heading_b1)
        desired_b2_norm = np.linalg.norm(desired_b2)
        if desired_b2_norm < _EPS:
            heading_b1 = np.array([
                -math.sin(self.desired_yaw), math.cos(self.desired_yaw), 0.0
            ])
            desired_b2 = np.cross(desired_b3, heading_b1)
            desired_b2_norm = np.linalg.norm(desired_b2)
        desired_b2 /= desired_b2_norm
        desired_b1 = np.cross(desired_b2, desired_b3)
        return np.column_stack((desired_b1, desired_b2, desired_b3))

    def _flight_controller(self, state, update_diagnostics=False):
        """运行外环、SO(3) 内环和电机分配。"""
        _, _, quaternion, angular_velocity, _ = self._unpack_state(state)
        rotation = _quaternion_to_rotation_matrix(quaternion)
        desired_force = self._outer_loop(state)
        desired_rotation = self._desired_rotation_from_force(desired_force)

        attitude_error_matrix = 0.5 * (
            desired_rotation.T @ rotation - rotation.T @ desired_rotation
        )
        attitude_error = _vee(attitude_error_matrix)
        commanded_torque = (
            -self.attitude_gain * attitude_error
            -self.angular_rate_gain * angular_velocity
            +np.cross(angular_velocity, self.inertia @ angular_velocity)
        )

        body_b3_world = rotation[:, 2]
        commanded_thrust = max(float(np.dot(desired_force, body_b3_world)), 0.0)
        maximum_thrust = (
            4.0 * self.thrust_coefficient * self.max_rotor_speed ** 2
        )
        commanded_thrust = min(commanded_thrust, maximum_thrust)
        rotor_speed_squared = self._allocate_rotor_speed_squared(
            commanded_thrust, commanded_torque
        )
        commanded_rotor_speeds = np.sqrt(rotor_speed_squared)

        if update_diagnostics:
            self.desired_force = desired_force.copy()
            self.desired_rotation = desired_rotation.copy()
            self.commanded_thrust = commanded_thrust
            self.commanded_torque = commanded_torque.copy()
            self.commanded_rotor_speeds = commanded_rotor_speeds.copy()
        return commanded_rotor_speeds

    def _allocate_rotor_speed_squared(self, thrust, torque):
        """
        带饱和优先级的分配：总推力 > 滚转/俯仰力矩 > 偏航力矩。

        该策略避免不可实现的偏航力矩在逐电机硬裁剪后改变总推力。
        """
        minimum_squared = self.min_rotor_speed ** 2
        maximum_squared = self.max_rotor_speed ** 2
        collective = self.allocation_matrix_inverse @ np.array([
            thrust, 0.0, 0.0, 0.0
        ])
        rotor_speed_squared = np.clip(
            collective, minimum_squared, maximum_squared
        )

        roll_pitch_delta = self.allocation_matrix_inverse @ np.array([
            0.0, torque[0], torque[1], 0.0
        ])
        rotor_speed_squared = self._add_allocation_delta_with_saturation(
            rotor_speed_squared,
            roll_pitch_delta,
            minimum_squared,
            maximum_squared,
        )

        yaw_delta = self.allocation_matrix_inverse @ np.array([
            0.0, 0.0, 0.0, torque[2]
        ])
        return self._add_allocation_delta_with_saturation(
            rotor_speed_squared,
            yaw_delta,
            minimum_squared,
            maximum_squared,
        )

    @staticmethod
    def _add_allocation_delta_with_saturation(
            base_squared, delta_squared, minimum_squared, maximum_squared):
        """沿给定分配方向取满足所有电机边界的最大比例。"""
        scale = 1.0
        for base_value, delta_value in zip(base_squared, delta_squared):
            if delta_value > _EPS:
                scale = min(
                    scale, (maximum_squared - base_value) / delta_value
                )
            elif delta_value < -_EPS:
                scale = min(
                    scale, (minimum_squared - base_value) / delta_value
                )
        scale = float(np.clip(scale, 0.0, 1.0))
        return np.clip(
            base_squared + scale * delta_squared,
            minimum_squared,
            maximum_squared,
        )

    def _derivative(self, state):
        """完整闭环四旋翼状态微分。"""
        _, velocity, quaternion, angular_velocity, rotor_speeds = (
            self._unpack_state(state)
        )
        rotation = _quaternion_to_rotation_matrix(quaternion)
        commanded_rotor_speeds = self._flight_controller(state)

        actual_wrench = self.allocation_matrix @ (rotor_speeds ** 2)
        thrust = float(actual_wrench[0])
        torque = actual_wrench[1:4]
        position_derivative = velocity
        velocity_derivative = (
            np.array([0.0, 0.0, -self.gravity])
            + (thrust / self.mass) * rotation[:, 2]
            - (self.translational_drag / self.mass) * velocity
        )
        quaternion_derivative = 0.5 * _quaternion_multiply(
            quaternion, np.concatenate(([0.0], angular_velocity))
        )
        angular_velocity_derivative = self.inertia_inverse @ (
            torque
            - np.cross(angular_velocity, self.inertia @ angular_velocity)
            - self.angular_drag * angular_velocity
        )
        rotor_speed_derivative = (
            commanded_rotor_speeds - rotor_speeds
        ) / self.motor_time_constant
        return np.concatenate((
            position_derivative,
            velocity_derivative,
            quaternion_derivative,
            angular_velocity_derivative,
            rotor_speed_derivative,
        ))

    def _state_vector(self):
        return np.concatenate((
            self.position,
            self.velocity,
            self.quaternion,
            self.angular_velocity,
            self.rotor_speeds,
        ))

    def _set_state_vector(self, state):
        self.position = state[0:3].copy()
        self.velocity = state[3:6].copy()
        self.quaternion = _normalize_quaternion(state[6:10])
        self.angular_velocity = state[10:13].copy()
        self.rotor_speeds = np.clip(
            state[13:17], self.min_rotor_speed, self.max_rotor_speed
        )

    def update_dynamics(self, dt):
        """使用固定子步长 RK4 积分完整闭环动力学。"""
        dt = float(dt)
        if dt <= 0.0:
            return
        step = dt / self.integration_substeps
        state = self._state_vector()
        for _ in range(self.integration_substeps):
            k1 = self._derivative(state)
            k2 = self._derivative(state + 0.5 * step * k1)
            k3 = self._derivative(state + 0.5 * step * k2)
            k4 = self._derivative(state + step * k3)
            state = state + (step / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            state[6:10] = _normalize_quaternion(state[6:10])
            state[13:17] = np.clip(
                state[13:17], self.min_rotor_speed, self.max_rotor_speed
            )

        self._set_state_vector(state)
        self._flight_controller(state, update_diagnostics=True)
        actual_wrench = self.allocation_matrix @ (self.rotor_speeds ** 2)
        self.actual_thrust = float(actual_wrench[0])
        self.actual_torque = actual_wrench[1:4].copy()

    def update_position(self, dt, velocity_x=None, velocity_y=None):
        """旧接口兼容实现，内部仍积分完整四旋翼动力学。"""
        if velocity_x is not None and velocity_y is not None:
            self.set_planar_velocity_command(
                velocity_x, velocity_y, command_dt=dt
            )
        self.update_dynamics(dt)

    def get_position(self):
        return self.position.copy()

    def get_orientation(self):
        """返回偏航角，兼容原二维绘图接口。"""
        rotation = _quaternion_to_rotation_matrix(self.quaternion)
        return math.atan2(rotation[1, 0], rotation[0, 0])

    def get_attitude(self):
        """返回 (roll, pitch, yaw)。"""
        rotation = _quaternion_to_rotation_matrix(self.quaternion)
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        return roll, pitch, yaw

    def get_state(self):
        """返回用于记录和诊断的完整状态快照。"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'quaternion': self.quaternion.copy(),
            'attitude': np.asarray(self.get_attitude()),
            'angular_velocity': self.angular_velocity.copy(),
            'rotor_speeds': self.rotor_speeds.copy(),
            'commanded_rotor_speeds': self.commanded_rotor_speeds.copy(),
            'desired_planar_velocity': self.desired_planar_velocity.copy(),
            'desired_planar_acceleration': (
                self.desired_planar_acceleration.copy()
            ),
            'desired_force': self.desired_force.copy(),
            'commanded_torque': self.commanded_torque.copy(),
        }

    def is_at_target(self, target_threshold=0.1):
        if self.target is None:
            return True
        return np.linalg.norm(self.target - self.position[:2]) < target_threshold


# 保留旧类名，避免项目内外已有调用方失效。
SimpleRobot = QuadrotorRobot


class RobotManager:
    """多架四旋翼管理器。"""

    def __init__(self, num_robots, initial_positions, config):
        self.robots = {}
        self.num_robots = num_robots
        self.config = config
        self.kp_linear = config.CONTROL_CONFIG['kp_linear']
        self.target_threshold = config.CONTROL_CONFIG['target_reached_threshold']

        for index in range(1, num_robots + 1):
            robot_name = f'robot{index}'
            initial_position = initial_positions.get(robot_name, [0.0, 0.0])
            self.robots[robot_name] = QuadrotorRobot(
                robot_id=robot_name,
                initial_position=initial_position,
                config=config,
            )

    def update_all_robots(self, targets, dt, velocity_commands=None,
                          body_velocity_commands=None,
                          planar_velocity_commands=None):
        """
        更新全部四旋翼。

        指令优先级：planar_velocity_commands > velocity_commands >
        body_velocity_commands > 目标点 P 控制回退。新主循环应使用
        planar_velocity_commands={robot_name: (vx_d, vy_d)}。
        """
        targets = targets or {}
        for robot_name, robot in self.robots.items():
            if robot_name in targets:
                robot.set_target(targets[robot_name])

            if (
                planar_velocity_commands is not None
                and robot_name in planar_velocity_commands
            ):
                velocity_x, velocity_y = planar_velocity_commands[robot_name]
                robot.set_planar_velocity_command(
                    velocity_x, velocity_y, command_dt=dt
                )
            elif velocity_commands is not None and robot_name in velocity_commands:
                velocity_x, velocity_y = velocity_commands[robot_name]
                robot.set_velocity_command(
                    velocity_x, velocity_y, command_dt=dt
                )
            elif (
                body_velocity_commands is not None
                and robot_name in body_velocity_commands
            ):
                linear_velocity, angular_velocity = body_velocity_commands[robot_name]
                robot.set_body_velocity_command(
                    linear_velocity, angular_velocity, command_dt=dt
                )
            else:
                velocity_x, velocity_y = robot.calculate_control(
                    kp_linear=self.kp_linear,
                    target_threshold=self.target_threshold,
                )
                robot.set_planar_velocity_command(
                    velocity_x, velocity_y, command_dt=dt
                )

            robot.update_dynamics(dt)

    def get_robot_positions(self):
        """返回含真实 x、y、z 的兼容 Point 对象。"""
        positions = {}
        for robot_name, robot in self.robots.items():
            position = robot.get_position()
            positions[robot_name] = type('Point', (), {
                'x': float(position[0]),
                'y': float(position[1]),
                'z': float(position[2]),
            })()
        return positions

    def get_robot_orientations(self):
        """返回各四旋翼偏航角，供旧绘图接口使用。"""
        return {
            robot_name: robot.get_orientation()
            for robot_name, robot in self.robots.items()
        }

    def get_robot_states(self):
        """返回各四旋翼完整状态。"""
        return {
            robot_name: robot.get_state()
            for robot_name, robot in self.robots.items()
        }

    def get_robot_at_target_status(self):
        return {
            robot_name: robot.is_at_target(self.target_threshold)
            for robot_name, robot in self.robots.items()
        }
