#!/usr/bin/env python3
import datetime
import math
import os
import re
import xml.etree.ElementTree as ET

import matplotlib
# Preserve the interactive window on desktop runs while allowing headless
# rendering in batch experiments and tests.
requested_backend = os.environ.get('MPLBACKEND')
display_is_valid = (
    bool(os.environ.get('DISPLAY'))
    and matplotlib._c_internal_utils.display_is_valid()
)
matplotlib.use(
    requested_backend or ('TkAgg' if display_is_valid else 'Agg')
)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
import numpy as np

try:
    from .config import CoverageConfig
except ImportError:  # Support direct execution from the source directory.
    from config import CoverageConfig


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
UAV_SVG_PATH = os.path.join(MODULE_DIR, "assets", "uav_marker.svg")
_SVG_TOKEN_PATTERN = re.compile(
    r"[MmLlCcSsZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
)

# Smooth light-blue field palette inspired by the reference figure.  Low-value
# regions remain blue instead of fading to white; high-value regions transition
# through cyan/green to a soft yellow.
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "coverage_field",
    [
        "#C7D9F1",
        "#B6D8F0",
        "#A8E1E0",
        "#CBE8C2",
        "#F6E7A5",
    ],
    N=256,
)

PLOT_COLORS = {
    'background': '#C7D9F1',
    'boundary': '#203746',
    'voronoi': '#425967',
    'trajectory': '#D55E4A',
    'moving_field': '#786A91',
    'sample': '#2E86AB',
    'uav_fill': '#FFFFFF',
    'uav_edge': '#17324D',
}


def _parse_svg_path_data(path_data):
    """Parse the M/L/C/S/Z subset used by the project UAV SVG."""
    tokens = _SVG_TOKEN_PATTERN.findall(path_data)
    vertices = []
    codes = []
    index = 0
    command = None
    previous_command = None
    current = np.zeros(2, dtype=float)
    subpath_start = np.zeros(2, dtype=float)
    last_cubic_control = None

    def read_point(relative):
        nonlocal index
        if index + 1 >= len(tokens):
            raise ValueError("Incomplete coordinate pair in SVG path")
        point = np.array(
            [float(tokens[index]), float(tokens[index + 1])], dtype=float
        )
        index += 2
        return current + point if relative else point

    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command.lower() == 'z':
                vertices.append(subpath_start.copy())
                codes.append(MplPath.CLOSEPOLY)
                current = subpath_start.copy()
                last_cubic_control = None
                previous_command = 'z'
                command = None
                continue

        if command is None:
            raise ValueError("SVG path data is missing a drawing command")

        relative = command.islower()
        normalized_command = command.lower()
        if normalized_command == 'm':
            current = read_point(relative)
            subpath_start = current.copy()
            vertices.append(current.copy())
            codes.append(MplPath.MOVETO)
            last_cubic_control = None
            previous_command = 'm'
            # Further coordinate pairs after M/m are implicit L/l commands.
            command = 'l' if relative else 'L'
        elif normalized_command == 'l':
            current = read_point(relative)
            vertices.append(current.copy())
            codes.append(MplPath.LINETO)
            last_cubic_control = None
            previous_command = 'l'
        elif normalized_command == 'c':
            control_1 = read_point(relative)
            control_2 = read_point(relative)
            endpoint = read_point(relative)
            vertices.extend([control_1, control_2, endpoint])
            codes.extend([MplPath.CURVE4] * 3)
            current = endpoint
            last_cubic_control = control_2
            previous_command = 'c'
        elif normalized_command == 's':
            if previous_command in {'c', 's'} and last_cubic_control is not None:
                control_1 = 2.0 * current - last_cubic_control
            else:
                control_1 = current.copy()
            control_2 = read_point(relative)
            endpoint = read_point(relative)
            vertices.extend([control_1, control_2, endpoint])
            codes.extend([MplPath.CURVE4] * 3)
            current = endpoint
            last_cubic_control = control_2
            previous_command = 's'
        else:
            raise ValueError(
                f"Unsupported SVG path command: {command}"
            )

    return MplPath(np.asarray(vertices, dtype=float), np.asarray(codes))


def _load_svg_marker(svg_path):
    """Load SVG path geometry and normalize it as a centered vector marker."""
    root = ET.parse(svg_path).getroot()
    paths = []
    for element in root.iter():
        if element.tag.rsplit('}', 1)[-1] == 'path' and element.get('d'):
            paths.append(_parse_svg_path_data(element.get('d')))
    if not paths:
        raise ValueError(f"No path geometry found in UAV SVG: {svg_path}")

    marker = (
        paths[0]
        if len(paths) == 1
        else MplPath.make_compound_path(*paths)
    )
    bounds = marker.get_extents()
    width = float(bounds.x1 - bounds.x0)
    height = float(bounds.y1 - bounds.y0)
    scale = max(width, height)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid path bounds in UAV SVG: {svg_path}")

    vertices = marker.vertices.copy()
    vertices[:, 0] = (
        vertices[:, 0] - 0.5 * (bounds.x0 + bounds.x1)
    ) / scale
    # SVG uses a downward-positive Y axis; flip it for plot coordinates.
    vertices[:, 1] = -(
        vertices[:, 1] - 0.5 * (bounds.y0 + bounds.y1)
    ) / scale
    return MplPath(vertices, marker.codes)

class PlottingModule:
    def __init__(self, work_area_boundary):
        self.work_area_boundary = work_area_boundary
        boundary_array = np.asarray(work_area_boundary, dtype=float)
        self.x_min = float(np.min(boundary_array[:, 0]))
        self.x_max = float(np.max(boundary_array[:, 0]))
        self.y_min = float(np.min(boundary_array[:, 1]))
        self.y_max = float(np.max(boundary_array[:, 1]))

        # Load the supplied quadcopter artwork once.  Using it as a Matplotlib
        # Path keeps every UAV marker as vector geometry in the saved SVG.
        self.uav_marker = _load_svg_marker(UAV_SVG_PATH)

        # One run identifier plus the control-cycle index makes frames ordered,
        # unique across runs and immune to same-second filename collisions.
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_counter = 0
        self.last_saved_timestep = None
        self.save_dir = os.path.join(os.path.dirname(MODULE_DIR), "pic")

        # 设置绘图
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(
            figsize=(8, 8), facecolor=PLOT_COLORS['background']
        )
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect('equal')
        self.ax.grid(False)
        self.ax.set_axis_off()
        self.ax.set_facecolor(PLOT_COLORS['background'])
        
        # 从配置文件读取高斯函数参数
        config = CoverageConfig()
        density_config = config.DENSITY_FUNCTION_CONFIG
        
        # 高斯函数参数
        self.gaussian_params = density_config['gaussian_functions']
        
        # 高斯函数权重
        self.gaussian_weights = density_config['gaussian_weights']
        
        # 基础密度值
        self.base_density = density_config['base_density']
        
        # 初始化机器人轨迹存储
        self.robot_trajectories = {}
        
        # 跟踪时变高斯函数（gaussian3）的中心轨迹
        self.gaussian3_trajectory = []
        
        # 从配置获取热力图分辨率
        self.heatmap_resolution = 0.05  # 提高显示精度
    
    def _gaussian_function(self, x, y, params):
        """计算单个敏感度函数的值（支持三种类型）"""
        # 获取基础参数
        center_x = params['center_x']
        center_y = params['center_y']
        amplitude = params['amplitude']
        function_type = params.get('function_type', 'gaussian')
        movable = params.get('movable', False)
        
        # 如果是圆形运动且可移动，才计算实时位置
        motion_type = params.get('motion_type', 'static')
        if motion_type == 'circular' and movable:
            current_angle = params.get('current_angle', params.get('circular_initial_angle', 0))
            circular_center_x = params.get('circular_center_x', 5.0)
            circular_center_y = params.get('circular_center_y', 5.0)
            circular_radius = params.get('circular_radius', 1.0)
            center_x = circular_center_x + circular_radius * np.cos(current_angle)
            center_y = circular_center_y + circular_radius * np.sin(current_angle)
        
        dx = x - center_x
        dy = y - center_y
        dist_sq = dx**2 + dy**2
        r = math.sqrt(dist_sq)
        
        if function_type == 'gaussian':
            # 标准高斯: A * exp(-r² / 2σ²)
            variance = params.get('variance', 1.0)
            value = amplitude * math.exp(-dist_sq / (2 * variance))
        elif function_type == 'sigmoid_cutoff':
            # Sigmoid平滑截断高斯（连续可导）
            variance = params.get('variance', 1.0)
            direction = params.get('cutoff_direction', 0.0)
            smoothness = params.get('cutoff_smoothness', 0.2)
            gauss = amplitude * math.exp(-dist_sq / (2 * variance))
            projection = dx * math.cos(direction) + dy * math.sin(direction)
            sigmoid = 1.0 / (1.0 + math.exp(-projection / smoothness))
            value = gauss * sigmoid
        elif function_type == 'donut':
            # 甜甜圈高斯: A * exp(-(r - r₀)² / 2σ²)
            variance = params.get('variance', 0.5)
            ring_radius = params.get('ring_radius', 1.5)
            value = amplitude * math.exp(-(r - ring_radius)**2 / (2 * variance))
        else:
            raise ValueError(f"不支持的函数类型: {function_type}")
        
        return value
    
    def density_function(self, x, y):
        """计算给定点的密度值（三个高斯函数的加权和）"""
        # 计算每个高斯函数的值
        gaussian_densities = []
        for params in self.gaussian_params:
            density = self._gaussian_function(x, y, params)
            gaussian_densities.append(density)
        
        # 加权求和
        total_density = 0.0
        for i, weight in enumerate(self.gaussian_weights):
            if i < len(gaussian_densities):
                total_density += weight * gaussian_densities[i]
        
        # 添加基础密度值
        measured_density = self.base_density + total_density
        
        # 确保密度值为正
        measured_density = max(measured_density, 0.0)
        
        return measured_density

    def update_gaussian_params(self, new_gaussian_params):
        """更新高斯函数参数，用于移动高斯函数的可视化"""
        self.gaussian_params = new_gaussian_params
        # 跟踪 gaussian3 的轨迹
        for params in self.gaussian_params:
            if params.get('name') == 'gaussian3' and params.get('motion_type') == 'linear_timed':
                self.gaussian3_trajectory.append((params['center_x'], params['center_y']))
    
    def generate_density_heatmap(self, resolution=None):
        """生成密度热力图数据"""
        # 如果没有提供分辨率，则使用配置文件中的分辨率
        if resolution is None:
            resolution = self.heatmap_resolution
        
        # 创建网格
        x = np.arange(self.x_min, self.x_max + resolution, resolution)
        y = np.arange(self.y_min, self.y_max + resolution, resolution)
        X, Y = np.meshgrid(x, y)
        
        # 计算每个网格点的密度值
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.density_function(X[i, j], Y[i, j])
        
        return X, Y, Z
    
    def plot_density_region(self):
        """Draw the field as a clean, publication-ready density layer."""
        # 生成热力图数据
        X, Y, Z = self.generate_density_heatmap()
        
        # Bicubic interpolation produces the same smooth, contour-free field
        # appearance as the reference visualization.
        heatmap = self.ax.imshow(
            Z,
            origin='lower',
            extent=(
                float(np.min(X)),
                float(np.max(X)),
                float(np.min(Y)),
                float(np.max(Y)),
            ),
            cmap=FIELD_CMAP,
            alpha=1.0,
            interpolation='bicubic',
            rasterized=True,
            zorder=1,
        )
        return heatmap
    
    def plot_voronoi_regions(self, coords, robot_names, voronoi_areas, robot_positions,
                             robot_orientations, robot_targets, position_received,
                             centroids=None, timestep=None):
        """Draw and save one text-free coverage snapshot for a control cycle."""
        self.ax.clear()
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect('equal')
        self.ax.grid(False)
        self.ax.set_facecolor(PLOT_COLORS['background'])
        # Axis-off removes titles, labels, ticks and numeric annotations while
        # leaving the field, boundary, Voronoi edges and UAV artwork visible.
        self.ax.set_axis_off()
        
        # 绘制密度区域
        self.plot_density_region()
        
        # 绘制时变高斯函数 gaussian3 的轨迹和当前中心
        if self.gaussian3_trajectory:
            traj_x, traj_y = zip(*self.gaussian3_trajectory)
            self.ax.plot(
                traj_x,
                traj_y,
                color=PLOT_COLORS['moving_field'],
                linestyle=(0, (1.5, 2.5)),
                linewidth=1.0,
                alpha=0.68,
                zorder=4,
            )
            self.ax.plot(
                traj_x[-1],
                traj_y[-1],
                marker='*',
                color=PLOT_COLORS['moving_field'],
                markersize=11,
                markeredgecolor='white',
                markeredgewidth=0.9,
                zorder=6,
            )
        
        # 绘制工作区域边界
        boundary_x, boundary_y = zip(*self.work_area_boundary)
        self.ax.plot(
            boundary_x + (boundary_x[0],),
            boundary_y + (boundary_y[0],),
            color=PLOT_COLORS['boundary'],
            linewidth=1.5,
            solid_joinstyle='round',
            zorder=7,
        )
        
        # 绘制Voronoi区域边界
        if voronoi_areas:
            for poly in voronoi_areas:
                # 检查poly是否为有效的多边形对象
                if hasattr(poly, 'is_valid') and hasattr(poly, 'is_empty') and poly.is_valid and not poly.is_empty:
                    x, y = poly.exterior.xy
                    self.ax.plot(
                        x,
                        y,
                        color=PLOT_COLORS['voronoi'],
                        linewidth=1.05,
                        alpha=0.84,
                        solid_capstyle='round',
                        solid_joinstyle='round',
                        zorder=5,
                    )
        
        # 绘制机器人位置
        for robot_name in robot_names:
            if position_received[robot_name]:
                # 检查robot_positions[robot_name]是否为具有x、y属性的对象，还是元组
                if hasattr(robot_positions[robot_name], 'x') and hasattr(robot_positions[robot_name], 'y'):
                    # 使用实际坐标绘制机器人位置 (对象格式)
                    x = robot_positions[robot_name].x
                    y = robot_positions[robot_name].y
                else:
                    # 使用元组格式 (x, y)
                    x = robot_positions[robot_name][0]
                    y = robot_positions[robot_name][1]
                
                # 更新机器人轨迹
                if robot_name not in self.robot_trajectories:
                    self.robot_trajectories[robot_name] = []
                self.robot_trajectories[robot_name].append((x, y))
                
                # 绘制机器人轨迹（半透明虚线）- 加粗线条
                if len(self.robot_trajectories[robot_name]) > 1:
                    traj_x, traj_y = zip(*self.robot_trajectories[robot_name])
                    self.ax.plot(
                        traj_x,
                        traj_y,
                        color=PLOT_COLORS['trajectory'],
                        linestyle=(0, (5, 3)),
                        alpha=0.72,
                        linewidth=0.95,
                        dash_capstyle='round',
                        solid_capstyle='round',
                        zorder=3,
                    )

                orientation = robot_orientations.get(robot_name, 0.0)
                if isinstance(orientation, (tuple, list, np.ndarray)):
                    yaw = float(orientation[2]) if len(orientation) >= 3 else 0.0
                else:
                    yaw = float(orientation)
                rotated_marker = self.uav_marker.transformed(
                    Affine2D().rotate(yaw)
                )
                self.ax.scatter(
                    [x],
                    [y],
                    marker=rotated_marker,
                    s=235,
                    facecolors=PLOT_COLORS['uav_fill'],
                    edgecolors=PLOT_COLORS['uav_edge'],
                    linewidths=1.0,
                    zorder=9,
                )

        # robot_targets and centroids remain in the signature for backward
        # compatibility, but their former red-cross annotations are deliberately
        # omitted from publication snapshots.
        self.fig.canvas.draw()

        # A deterministic control-cycle suffix guarantees exactly one named
        # snapshot per cycle.  The run prefix prevents a later experiment from
        # overwriting an earlier one.
        if timestep is None:
            self.frame_counter += 1
            cycle = self.frame_counter
        else:
            cycle = int(timestep)
            self.frame_counter = max(self.frame_counter, cycle)

        try:
            os.makedirs(self.save_dir, exist_ok=True)
            filename = (
                f"voronoi_coverage_{self.run_timestamp}_t{cycle:04d}.svg"
            )
            filepath = os.path.join(self.save_dir, filename)
            self.fig.savefig(
                filepath,
                format='svg',
                bbox_inches='tight',
                pad_inches=0,
                facecolor=PLOT_COLORS['background'],
            )
            self.last_saved_timestep = cycle
            print(f'Voronoi图已保存（控制周期 {cycle}）: {filepath}')
        except Exception as e:
            print(f'保存图片时出错: {str(e)}')
            return None

        if plt.isinteractive() and 'agg' not in matplotlib.get_backend().lower():
            plt.pause(0.001)
        return filepath
