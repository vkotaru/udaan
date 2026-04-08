"""Udaan: Quadrotor dynamics and control for cable-suspended payload systems.

Udaan provides geometric control on SE(3) × S² manifolds for quadrotor systems
with cable-suspended payloads, supporting multiple physics backends.
"""

from __future__ import annotations

import os
from pathlib import Path

# Package path for asset loading
_FOLDER_PATH, _FILE_PATH = os.path.split(os.path.abspath(os.path.dirname(__file__)))
PATH = _FOLDER_PATH
PACKAGE_DIR = Path(__file__).parent

# Core types and exceptions
# Submodules
from . import control, core, manif, models, utils

# Version
__version__ = "1.0.0"

__all__ = [
    "core",
    "models",
    "control",
    "utils",
    "manif",
    "PATH",
    "PACKAGE_DIR",
    "__version__",
]
