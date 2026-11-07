"""Utility functions and classes for Udaan."""

from . import assets, trajectory
from .assets import xml_model_generator
from .geometry import *
from .grasp import (
    formation_cable_attitudes,
    grasp_map,
    hover_wrench_map,
    ring_attachment_points,
)
from .hose import hose_spawn_directions, hose_vertex_positions, validate_attachments
from .logging import LoggerMixin, get_logger, setup_logging

__all__ = [
    # Logging
    "get_logger",
    "setup_logging",
    "LoggerMixin",
    # Assets
    "assets",
    "xml_model_generator",
    # Trajectory
    "trajectory",
    # Grasp and formation geometry (M3/M4)
    "formation_cable_attitudes",
    "grasp_map",
    "hover_wrench_map",
    "ring_attachment_points",
    # Hose spawn geometry (M6)
    "hose_spawn_directions",
    "hose_vertex_positions",
    "validate_attachments",
]
