"""Type definitions for Udaan library.

This module defines type aliases and enumerations used throughout the codebase
to ensure type safety and improve code readability.
"""
from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Type aliases for geometric objects
# =============================================================================

#: 3D vector (position, velocity, angular velocity, etc.)
Vec3: TypeAlias = NDArray[np.floating]  # Shape: (3,)

#: 3x3 matrix (rotation matrix, inertia tensor, etc.)
Mat3: TypeAlias = NDArray[np.floating]  # Shape: (3, 3)

#: 4x4 matrix (homogeneous transformation)
Mat4: TypeAlias = NDArray[np.floating]  # Shape: (4, 4)

#: Quaternion in [w, x, y, z] format (scalar-first convention)
Quaternion: TypeAlias = NDArray[np.floating]  # Shape: (4,)

#: Trajectory function: t -> (position, velocity, acceleration)
TrajectoryFunc: TypeAlias = Callable[[float], tuple[Vec3, Vec3, Vec3]]


# =============================================================================
# Enumerations
# =============================================================================


class InputType(Enum):
    """Input command type for quadrotor control.

    Defines what type of command the controller generates.

    Attributes:
        WRENCH: Force and torque vector [thrust, tau_x, tau_y, tau_z]
        PROP_FORCES: Individual propeller forces [f1, f2, f3, f4]
        ACCELERATION: Desired acceleration command [ax, ay, az]
    """

    WRENCH = auto()
    PROP_FORCES = auto()
    ACCELERATION = auto()

    # Aliases for backward compatibility with existing code
    CMD_WRENCH = WRENCH
    CMD_PROP_FORCES = PROP_FORCES
    CMD_ACCEL = ACCELERATION


class ForceType(Enum):
    """Force application type on quadrotor body.

    Defines how forces are applied to the physics simulation.

    Attributes:
        WRENCH: Apply net force and torque at center of mass
        PROP_FORCES: Apply individual forces at propeller locations
    """

    WRENCH = auto()
    PROP_FORCES = auto()
