from __future__ import annotations

import numpy as np

from .utils import hat, rodrigues_expm


class TS2:
    """Tangent vector to the 2-sphere S2.

    Wraps a 3-vector representing angular velocity or configuration error
    on the sphere. Supports:
        v1 - v2  -> TS2   (tangent vector difference)
        v1 + v2  -> TS2   (tangent vector sum)
        v * s    -> TS2   (scalar multiplication)
        v.transport(q) -> TS2  (project onto tangent space at q)
    """

    __slots__ = ("_data",)

    def __init__(self, vector=None):
        if vector is None:
            self._data = np.zeros(3)
        else:
            self._data = np.asarray(vector, dtype=float).copy()

    # ─── numpy interop ─────────────────────────────────────────────

    def __array__(self, dtype=None):
        return self._data if dtype is None else self._data.astype(dtype)

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return 3

    # ─── properties ────────────────────────────────────────────────

    @property
    def arr(self) -> np.ndarray:
        """Plain numpy array (copy)."""
        return self._data.copy()

    @property
    def vector(self) -> np.ndarray:
        """Raw 3-vector as a plain np.ndarray."""
        return self._data

    @property
    def norm(self) -> float:
        """Magnitude of the tangent vector."""
        return float(np.linalg.norm(self._data))

    # ─── arithmetic ────────────────────────────────────────────────

    def __sub__(self, other) -> TS2:
        if not isinstance(other, TS2):
            return NotImplemented
        return TS2(self._data - other._data)

    def __add__(self, other) -> TS2:
        if not isinstance(other, TS2):
            return NotImplemented
        return TS2(self._data + other._data)

    def __iadd__(self, other) -> TS2:
        if not isinstance(other, TS2):
            return NotImplemented
        self._data += other._data
        return self

    def __isub__(self, other) -> TS2:
        if not isinstance(other, TS2):
            return NotImplemented
        self._data -= other._data
        return self

    def __mul__(self, scalar) -> TS2:
        return TS2(self._data * scalar)

    def __rmul__(self, scalar) -> TS2:
        return TS2(scalar * self._data)

    def __neg__(self) -> TS2:
        return TS2(-self._data)

    # ─── Lie algebra ───────────────────────────────────────────────

    def transport(self, q: S2) -> TS2:
        """Transport this tangent vector to the tangent space at q.

        Computes -hat(q)^2 @ self, which projects self onto T_q S2
        with the correct sign so that ew = w - wd.transport(q).
        """
        q_arr = np.asarray(q)
        return TS2(-hat(q_arr) @ hat(q_arr) @ self._data)

    def __repr__(self) -> str:
        return f"TS2({np.array2string(self._data, precision=4, separator=', ')})"


class S2:
    """Unit vector on the 2-sphere S2.

    Wraps a 3-vector (unit norm). Supports:
        q1 - q2  -> TS2   (configuration error as tangent vector)
        q + w    -> S2    (geodesic step via exponential map, w is a TS2)
    """

    __slots__ = ("_data",)

    def __init__(self, q=None):
        if q is None:
            self._data = np.array([0.0, 0.0, 1.0])
        else:
            self._data = np.asarray(q, dtype=float).copy()

    # ─── numpy interop ─────────────────────────────────────────────

    def __array__(self, dtype=None):
        return self._data if dtype is None else self._data.astype(dtype)

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return 3

    # ─── properties ────────────────────────────────────────────────

    @property
    def arr(self) -> np.ndarray:
        """Plain numpy array (copy)."""
        return self._data.copy()

    # ─── geometry ──────────────────────────────────────────────────

    def step(self, omega_dt=None):
        """Geodesic step on S2 via the exponential map.

        Args:
            omega_dt: angular velocity scaled by dt (3-vector).

        Returns a new S2 element: q_next = expm(hat(omega_dt)) @ q.
        """
        if omega_dt is None:
            omega_dt = np.zeros(3)
        return S2(rodrigues_expm(np.asarray(omega_dt)) @ self._data)

    def config_error(self, other) -> float:
        """Scalar configuration error: 1 - q^T q_other."""
        return 1 - np.dot(self._data, np.asarray(other))

    def error_vec(self, other, version=2) -> np.ndarray:
        """Configuration error vector on the tangent space.

        Args:
            other: The other S2 point.
            version: Error formula variant.
                2 (default): hat(q)^2 @ q_other
                1: q_other x q (cross product)
        """
        if version == 2:
            return hat(self._data) @ hat(self._data) @ np.asarray(other)
        else:
            return np.cross(np.asarray(other), self._data)

    # ─── Lie group arithmetic ──────────────────────────────────────

    def __sub__(self, other) -> TS2:
        """Configuration error between two points on S2.

        Returns a TS2 tangent vector: hat(self)^2 @ other.
        """
        if not isinstance(other, S2):
            return NotImplemented
        return TS2(self.error_vec(other))

    def __add__(self, tangent) -> S2:
        """Geodesic step on S2: q_next = expm(hat(tangent.vector)) @ q.

        Args:
            tangent: TS2 element (e.g., TS2(omega * dt)).
        """
        if not isinstance(tangent, TS2):
            return NotImplemented
        return S2(rodrigues_expm(tangent._data) @ self._data)

    def __repr__(self) -> str:
        return f"S2({np.array2string(self._data, precision=4, separator=', ')})"

    @staticmethod
    def from_spherical(phi=0.0, th=0.0):
        """Point on S2 from spherical coordinates (azimuth phi, polar th)."""
        return S2(np.array([np.cos(phi) * np.sin(th), np.sin(phi) * np.sin(th), np.cos(th)]))
