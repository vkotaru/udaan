from __future__ import annotations

import numpy as np

from ..core.exceptions import ManifoldTypeError
from .utils import hat, rodrigues_expm, vee


class TSO3(np.ndarray):
    """Element of the Lie algebra so(3) — tangent vector to SO(3).

    Subclasses np.ndarray (3-vector) representing angular velocity or
    rotation error. Supports:
        w1 - w2  -> TSO3   (tangent vector difference)
        w1 + w2  -> TSO3   (tangent vector sum)
        w * s    -> TSO3   (scalar multiplication)
        w.hat()  -> 3x3    (skew-symmetric matrix in so(3))
        w.transport(R_from, R_to) -> TSO3  (frame transport)
    """

    def __new__(cls, vector=np.zeros(3)):
        obj = np.asarray(vector, dtype=float).view(cls)
        return obj

    @property
    def arr(self) -> np.ndarray:
        """Plain numpy array (strips type wrapper)."""
        return np.asarray(self)

    @property
    def vector(self) -> np.ndarray:
        """Raw 3-vector as a plain np.ndarray."""
        return np.asarray(self)

    def hat(self) -> np.ndarray:
        """Return the 3x3 skew-symmetric matrix in so(3)."""
        return hat(np.asarray(self))

    @property
    def norm(self) -> float:
        """Magnitude of the tangent vector."""
        return float(np.linalg.norm(self))

    def __sub__(self, other) -> TSO3:
        if not isinstance(other, TSO3):
            return NotImplemented
        return TSO3(np.asarray(self) - np.asarray(other))

    def __add__(self, other) -> TSO3:
        if not isinstance(other, TSO3):
            return NotImplemented
        return TSO3(np.asarray(self) + np.asarray(other))

    def transport(self, R_from: SO3, R_to: SO3) -> TSO3:
        """Transport this tangent vector from R_from's body frame to R_to's.

        Computes R_to^T @ R_from @ self, which maps an angular velocity (self)
        expressed in R_from's frame into R_to's frame.

        Example:
            eOm = Om - Omd.transport(Rd, R)  # angular velocity error
        """
        return TSO3(np.asarray(R_to).T @ np.asarray(R_from) @ np.asarray(self))

    def __repr__(self) -> str:
        return f"TSO3({np.array2string(np.asarray(self), precision=4, separator=', ')})"


class SO3(np.ndarray):
    """Rotation matrix on the Special Orthogonal group SO(3).

    Subclasses np.ndarray so it can be used directly in matrix algebra.
    Supports:
        R1 @ R2  -> SO3    (group composition, if both SO3)
        R @ v    -> ndarray (rotate a vector)
        R1 - R2  -> TSO3   (configuration error in the Lie algebra)
        R + w    -> SO3    (exponential map step, w is a TSO3)
        R.T      -> SO3    (transpose, preserves type)
    """

    def __new__(cls, R=np.eye(3)):
        obj = np.asarray(R, dtype=float).view(cls)
        return obj

    @property
    def T(self) -> SO3:
        """Transpose (same as inverse for SO3)."""
        return SO3(np.asarray(self).T)

    def __matmul__(self, other):
        """SO3 @ SO3 -> SO3, SO3 @ ndarray -> ndarray."""
        if isinstance(other, SO3):
            return SO3(np.asarray(self) @ np.asarray(other))
        return np.asarray(self) @ np.asarray(other)

    def __rmatmul__(self, other):
        """ndarray @ SO3 -> ndarray."""
        return np.asarray(other) @ np.asarray(self)

    @property
    def arr(self) -> np.ndarray:
        """Plain numpy array (strips type wrapper)."""
        return np.asarray(self)

    @staticmethod
    def from_angle_axis(eta: np.ndarray) -> SO3:
        """Construct SO3 from an angle-axis vector via the exponential map.

        Args:
            eta: 3-vector whose direction is the rotation axis and
                 magnitude is the rotation angle (radians).
        """
        return SO3(rodrigues_expm(eta))

    @staticmethod
    def from_two_vectors(b3: np.ndarray, b1: np.ndarray) -> SO3:
        """Construct SO3 from a primary axis (b3) and a heading hint (b1).

        Builds an orthonormal frame [b1', b2, b3] where b3 is preserved
        exactly, b1' is the closest vector to b1 orthogonal to b3, and
        b2 = b3 x b1'. Handles the singularity when b3 is parallel to b1.

        This is the standard construction for desired attitude from thrust
        direction (b3) and heading direction (b1).
        """
        _e1 = np.array([1.0, 0.0, 0.0])
        _e2 = np.array([0.0, 1.0, 0.0])

        b3_b1 = np.cross(b3, b1)
        norm_b3_b1 = np.linalg.norm(b3_b1)
        if norm_b3_b1 < 1e-10:
            # b3 parallel to b1, fall back to perpendicular axis
            b1 = _e2 if abs(np.dot(b3, _e1)) > 0.9 else _e1
            b3_b1 = np.cross(b3, b1)
            norm_b3_b1 = np.linalg.norm(b3_b1)
        b1 = (-1 / norm_b3_b1) * np.cross(b3, b3_b1)
        b2 = np.cross(b3, b1)
        return SO3(np.column_stack([b1, b2, b3]))

    @staticmethod
    def from_tilt_yaw(tilt: np.ndarray, yaw: float) -> SO3:
        """Construct SO3 from a tilt vector and a yaw angle.

        Args:
            tilt: 3-vector (angle-axis) applied to the upright orientation.
                  The resulting b3 axis is expm(hat(tilt)) @ e3.
            yaw: heading angle in radians. b1 hint is [cos(yaw), sin(yaw), 0].
        """
        b3 = rodrigues_expm(tilt) @ np.array([0.0, 0.0, 1.0])
        b1 = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return SO3.from_two_vectors(b3, b1)

    def inv(self) -> SO3:
        """Inverse rotation (transpose, since R^{-1} = R^T for SO(3))."""
        return SO3(np.asarray(self).T)

    def config_error(self, other: SO3) -> float:
        """Scalar configuration error: 0.5 * tr(I - other^T @ self).

        Returns 0 when self == other, approaches 2 for 180-degree error.
        """
        return 0.5 * np.trace(np.eye(3) - np.asarray(other).T @ np.asarray(self))

    def step(self, Omega_dt=np.zeros(3)):
        """Integrate angular velocity via the exponential map.

        Args:
            Omega_dt: angular velocity scaled by dt (3-vector).

        Returns a new SO3 element: R_next = R @ expm(hat(Omega_dt)).
        """
        return SO3(self @ rodrigues_expm(Omega_dt))

    def __sub__(self, other) -> TSO3:
        """Configuration error: eR = 1/2 vee(Rd^T R - R^T Rd).

        Ref: Lee, Leok, McClamroch 2010, Eq. 9.
        """
        if isinstance(other, np.ndarray) and not isinstance(other, SO3):
            # Delegate to numpy for plain ndarray operands (e.g. R @ R.T - I)
            return NotImplemented
        if not isinstance(other, SO3):
            raise ManifoldTypeError(
                f"SO3.__sub__ expects an SO3 element, got {type(other).__name__}. "
                "Use SO3(R) to wrap a rotation matrix."
            )
        # R - Rd: eR = 1/2 vee(Rd^T R - R^T Rd), where self=R, other=Rd
        err_matrix = np.asarray(other).T @ np.asarray(self) - np.asarray(self).T @ np.asarray(other)
        return TSO3(vee(err_matrix) / 2.0)

    def __add__(self, tangent) -> SO3:
        """Exponential map step: R_next = R @ expm(hat(tangent.vector)).

        Args:
            tangent: TSO3 element (e.g., TSO3(Omega * dt)).
        """
        if isinstance(tangent, np.ndarray) and not isinstance(tangent, TSO3):
            return NotImplemented
        if not isinstance(tangent, TSO3):
            raise ManifoldTypeError(
                f"SO3.__add__ expects a TSO3 tangent vector, got {type(tangent).__name__}. "
                "Use TSO3(omega * dt) to wrap an angular velocity vector."
            )
        return SO3(np.asarray(self) @ rodrigues_expm(np.asarray(tangent)))

    def __repr__(self) -> str:
        rows = [np.array2string(row, precision=4, separator=", ") for row in np.asarray(self)]
        return f"SO3([{rows[0]}, {rows[1]}, {rows[2]}])"


def Rot2Eul(R):
    phi_val = np.arctan2(R[2, 1], R[2, 2])
    theta_val = np.arcsin(-R[2, 0])
    psi_val = np.arctan2(R[1, 0], R[0, 0])

    if abs(theta_val - np.pi / 2) < 1.0e-3:
        phi_val = 0.0
        psi_val = np.arctan2(R[1, 2], R[0, 2])
    elif abs(theta_val + np.pi / 2) < 1.0e-3:
        phi_val = 0.0
        psi_val = np.arctan2(-R[1, 2], -R[0, 2])

    return np.array([phi_val, theta_val, psi_val])
