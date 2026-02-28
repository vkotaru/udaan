"""Custom exceptions for Udaan library.

Provides a hierarchy of exceptions for better error handling and debugging.
All Udaan-specific exceptions inherit from UdaanError.
"""
from __future__ import annotations


class UdaanError(Exception):
    """Base exception for all Udaan library errors.

    Catch this to handle any Udaan-specific exception.
    """


class ConfigurationError(UdaanError):
    """Raised when configuration parameters are invalid.

    Examples:
        - Negative mass
        - Non-symmetric inertia matrix
        - Invalid control gains
    """


class SingularityError(UdaanError):
    """Raised when a numerical singularity is encountered.

    This typically occurs in geometric computations when:
        - Normalizing a near-zero vector
        - Computing logarithm at 180-degree rotation
        - Dividing by near-zero thrust

    Attributes:
        operation: Name of the operation that encountered the singularity.
        value: The problematic value that caused the singularity.
    """

    def __init__(self, operation: str, value: float) -> None:
        self.operation = operation
        self.value = value
        super().__init__(f"Singularity in {operation}: value={value:.2e} is too small")


class ControllerNotInitializedError(UdaanError):
    """Raised when a controller is used before proper initialization.

    Some controllers (like L1 adaptive) require an initialization step
    before they can compute control outputs.
    """


class SimulationError(UdaanError):
    """Raised when an error occurs during simulation execution.

    Examples:
        - Physics engine failure
        - Invalid state reached
        - Timeout exceeded
    """


class PhysicsBackendError(UdaanError):
    """Raised when there's an issue with the physics backend.

    Examples:
        - MuJoCo model loading failure
        - PyBullet connection error
        - Missing physics assets
    """

    def __init__(self, backend: str, message: str) -> None:
        self.backend = backend
        super().__init__(f"[{backend}] {message}")
