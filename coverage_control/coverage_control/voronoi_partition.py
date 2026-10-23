#!/usr/bin/env python3
"""Bounded Voronoi partitioning and known-field weighted centroids."""

import numpy as np
from geovoronoi import voronoi_regions_from_coords
from shapely.geometry import Point as ShapelyPoint

try:
    from .config import CoverageConfig
except ImportError:
    from config import CoverageConfig


class VoronoiPartition:
    def __init__(self, work_area, config=None):
        self.work_area = work_area
        self.config = config if config is not None else CoverageConfig()
        self.voronoi_areas = []
        self.voronoi_centroids = []
        density = self.config.DENSITY_FUNCTION_CONFIG
        self.gaussian_params = [dict(item) for item in density["gaussian_functions"]]
        self.gaussian_weights = list(density["gaussian_weights"])
        self.base_density = float(density["base_density"])
        grid = self.config.GRID_SENSITIVITY_CONFIG
        self._grid_enabled = bool(grid.get("enabled", False))
        self._grid_resolution = int(grid.get("resolution", 400))
        self._grid_cell_size = float(grid.get("cell_size", 10.0 / 400))
        self._sensitivity_grid = None

    def update_voronoi_regions(self, robot_positions, robot_names, N=None):
        """Generate bounded Voronoi regions and preserve robot-to-cell ownership."""
        del N
        coordinates = np.asarray([
            [robot_positions[name].x, robot_positions[name].y]
            for name in robot_names
        ], dtype=float)
        if len(coordinates) < 3:
            return None, None
        if any(
            not self.work_area.contains(ShapelyPoint(float(x), float(y)))
            for x, y in coordinates
        ):
            return None, None

        try:
            region_polygons, region_points = voronoi_regions_from_coords(
                coordinates, self.work_area
            )[:2]
        except Exception as exc:
            print(f"更新Voronoi区域时出错: {exc}")
            return None, None

        regions = {name: [] for name in robot_names}
        polygons_by_robot = {}
        for region_id, point_indices in region_points.items():
            polygon = region_polygons[region_id]
            if polygon.is_empty or not polygon.is_valid:
                continue
            for point_index in point_indices:
                name = robot_names[int(point_index)]
                polygons_by_robot[name] = polygon
                vertices = list(polygon.exterior.coords)
                if len(vertices) > 1 and vertices[0] == vertices[-1]:
                    vertices = vertices[:-1]
                regions[name] = vertices

        self.voronoi_areas = [
            polygons_by_robot[name]
            for name in robot_names
            if name in polygons_by_robot
        ]
        self.voronoi_centroids = [
            (
                float(polygons_by_robot[name].centroid.x),
                float(polygons_by_robot[name].centroid.y),
            )
            for name in robot_names
            if name in polygons_by_robot
        ]
        return regions

    @staticmethod
    def _compute_component(x, y, params, weight=1.0):
        center_x = float(params["center_x"])
        center_y = float(params["center_y"])
        amplitude = float(params["amplitude"])
        dx = np.asarray(x, dtype=float) - center_x
        dy = np.asarray(y, dtype=float) - center_y
        radius_squared = dx * dx + dy * dy
        function_type = params.get("function_type", "gaussian")

        if function_type == "gaussian":
            value = amplitude * np.exp(
                -radius_squared / (2.0 * float(params.get("variance", 1.0)))
            )
        elif function_type == "sigmoid_cutoff":
            variance = float(params.get("variance", 1.0))
            direction = float(params.get("cutoff_direction", 0.0))
            smoothness = float(params.get("cutoff_smoothness", 0.2))
            gaussian = amplitude * np.exp(-radius_squared / (2.0 * variance))
            projection = dx * np.cos(direction) + dy * np.sin(direction)
            value = gaussian / (1.0 + np.exp(-projection / smoothness))
        elif function_type == "donut":
            radius = np.sqrt(radius_squared)
            ring_radius = float(params.get("ring_radius", 1.5))
            variance = float(params.get("variance", 0.5))
            value = amplitude * np.exp(
                -(radius - ring_radius) ** 2 / (2.0 * variance)
            )
        else:
            raise ValueError(f"不支持的敏感度函数类型: {function_type}")
        return float(weight) * value

    def set_gaussian_params(self, gaussian_params, gaussian_weights=None):
        self.gaussian_params = gaussian_params
        if gaussian_weights is not None:
            self.gaussian_weights = list(gaussian_weights)

    def set_sensitivity_grid(self, grid, resolution, cell_size):
        self._sensitivity_grid = np.asarray(grid, dtype=float)
        self._grid_resolution = int(resolution)
        self._grid_cell_size = float(cell_size)
        self._grid_enabled = True

    def _sample_grid_density(self, x, y):
        min_x, min_y, max_x, max_y = self.config.WORK_AREA_CONFIG["boundary"]
        clipped_x = np.clip(x, min_x, max_x - 1.0e-10)
        clipped_y = np.clip(y, min_y, max_y - 1.0e-10)
        column = min(
            int((clipped_x - min_x) / self._grid_cell_size),
            self._grid_resolution - 1,
        )
        row = min(
            int((clipped_y - min_y) / self._grid_cell_size),
            self._grid_resolution - 1,
        )
        return float(self._sensitivity_grid[row, column])

    def sample_true_density(self, x, y):
        if self._grid_enabled and self._sensitivity_grid is not None:
            return self._sample_grid_density(x, y)
        density = np.asarray(x, dtype=float) * 0.0 + self.base_density
        for params, weight in zip(self.gaussian_params, self.gaussian_weights):
            density = density + self._compute_component(x, y, params, weight)
        density = np.maximum(density, 0.0)
        return float(density) if np.ndim(density) == 0 else density

    def calculate_all_weighted_centroids_with_true_sensitivity(
        self, robot_positions
    ):
        """Compute all known-field region masses and centroids in one grid pass."""
        if not robot_positions:
            return {}
        resolution = float(self.config.VORONOI_CONFIG["grid_resolution"])
        min_x, min_y, max_x, max_y = self.config.WORK_AREA_CONFIG["boundary"]
        x_values = np.arange(min_x, max_x, resolution)
        y_values = np.arange(min_y, max_y, resolution)
        grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        names = list(robot_positions)
        positions = np.asarray([
            [robot_positions[name].x, robot_positions[name].y]
            for name in names
        ], dtype=float)
        distance_squared = np.sum(
            (points[:, None, :] - positions[None, :, :]) ** 2, axis=2
        )
        owners = np.argmin(distance_squared, axis=1)
        density_values = self.sample_true_density(points[:, 0], points[:, 1])
        area_element = resolution ** 2

        statistics = {}
        for index, name in enumerate(names):
            mask = owners == index
            weights = density_values[mask]
            weight_sum = float(np.sum(weights))
            mass = weight_sum * area_element
            if weight_sum > 1.0e-12:
                centroid = np.sum(
                    points[mask] * weights[:, None], axis=0
                ) / weight_sum
            else:
                centroid = positions[index]
            statistics[name] = (
                mass,
                [float(centroid[0]), float(centroid[1])],
            )
        return statistics

    def calculate_weighted_centroid_with_true_sensitivity(
        self, polygon, robot_name, robot_positions
    ):
        """Compatibility wrapper returning one robot's known-field centroid."""
        del polygon
        return tuple(
            self.calculate_all_weighted_centroids_with_true_sensitivity(
                robot_positions
            )[robot_name][1]
        )
