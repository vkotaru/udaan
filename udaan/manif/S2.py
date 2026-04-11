from __future__ import annotations

import numpy as np

from ..core.exceptions import ManifoldTypeError
from .utils import hat, rodrigues_expm


class TS2(np.ndarray):
    """Tangent vector to the 2-sphere S2.

    Subclasses np.ndarray (3-vector) representing angular velocity or
    configuration error on the sphere. Supports:
        v1 - v2  -> TS2   (tangent vector difference)
        v1 + v2  -> TS2   (tangent vector sum)
        v * s    -> TS2   (scalar multiplication)
        v.transport(q) -> TS2  (project onto tangent space at q)
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

    @property
    def norm(self) -> float:
        """Magnitude of the tangent vector."""
        return float(np.linalg.norm(self))

    def __sub__(self, other) -> TS2:
        if not isinstance(other, TS2):
            return NotImplemented
        return TS2(np.asarray(self) - np.asarray(other))

    def __add__(self, other) -> TS2:
        if not isinstance(other, TS2):
            return NotImplemented
        return TS2(np.asarray(self) + np.asarray(other))

    def transport(self, q: S2) -> TS2:
        """Transport this tangent vector to the tangent space at q.

        Computes -hat(q)^2 @ self, which projects self onto T_q S2
        with the correct sign so that eω = ω - ωd.transport(q), so that
        eω = ω + q × (q × ωd).
        """
        q_arr = np.asarray(q)
        return TS2(-hat(q_arr) @ hat(q_arr) @ np.asarray(self))

    def __repr__(self) -> str:
        return f"TS2({np.array2string(np.asarray(self), precision=4, separator=', ')})"


class S2(np.ndarray):
    """Unit vector on the 2-sphere S2.

    Subclasses np.ndarray so it can be used directly in vector algebra.
    Supports:
        q1 - q2  -> TS2   (configuration error as tangent vector)
        q + w    -> S2    (geodesic step via exponential map, w is a TS2)
    """

    def __new__(cls, q=np.array([0.0, 0.0, 1.0])):
        obj = np.asarray(q, dtype=float).view(cls)
        return obj

    @property
    def arr(self) -> np.ndarray:
        """Plain numpy array (strips type wrapper)."""
        return np.asarray(self)

    def step(self, omega_dt=np.zeros(3)):
        """Geodesic step on S2 via the exponential map.

        Args:
            omega_dt: angular velocity scaled by dt (3-vector).

        Returns a new S2 element: q_next = expm(hat(omega_dt)) @ q.
        """
        return S2(rodrigues_expm(omega_dt) @ self)

    def config_error(self, other) -> float:
        """Scalar configuration error: 1 - q^T q_other."""
        return 1 - np.dot(self, other)

    def error_vec(self, other, version=2) -> np.ndarray:
        """Configuration error vector on the tangent space.

        Args:
            other: The other S2 point.
            version: Error formula variant.
                2 (default): hat(q)^2 @ q_other
                1: q_other x q (cross product)
        """
        if version == 2:
            return hat(self) @ hat(self) @ other
        else:
            return np.cross(other, self)

    def __sub__(self, other) -> TS2:
        """Configuration error between two points on S2.

        Returns a TS2 tangent vector: hat(self)^2 @ other.
        """
        if isinstance(other, np.ndarray) and not isinstance(other, S2):
            return NotImplemented
        if not isinstance(other, S2):
            raise ManifoldTypeError(
                f"S2.__sub__ expects an S2 element, got {type(other).__name__}. "
                "Use S2(q) to wrap a unit vector."
            )
        return TS2(self.error_vec(other))

    def __add__(self, tangent) -> S2:
        """Geodesic step on S2: q_next = expm(hat(tangent.vector)) @ q.

        Args:
            tangent: TS2 element (e.g., TS2(omega * dt)).
        """
        if isinstance(tangent, np.ndarray) and not isinstance(tangent, TS2):
            return NotImplemented
        if not isinstance(tangent, TS2):
            raise ManifoldTypeError(
                f"S2.__add__ expects a TS2 tangent vector, got {type(tangent).__name__}. "
                "Use TS2(omega * dt) to wrap an angular velocity vector."
            )
        return S2(rodrigues_expm(np.asarray(tangent)) @ np.asarray(self))

    def __repr__(self) -> str:
        return f"S2({np.array2string(np.asarray(self), precision=4, separator=', ')})"

    @staticmethod
    def from_spherical(phi=0.0, th=0.0):
        """Point on S2 from spherical coordinates (azimuth phi, polar th)."""
        return np.array([np.cos(phi) * np.sin(th), np.sin(phi) * np.sin(th), np.cos(th)])
