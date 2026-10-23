"""Known-sensitivity multi-robot coverage-control package."""

from .config import CoverageConfig, default_config
from .control_module import ControlModule
from .robot_model import QuadrotorRobot, RobotManager
from .voronoi_partition import VoronoiPartition
from .coverage_control import VoronoiCoverageController

__all__ = [
    "CoverageConfig",
    "default_config",
    "ControlModule",
    "QuadrotorRobot",
    "RobotManager",
    "VoronoiPartition",
    "VoronoiCoverageController",
]
