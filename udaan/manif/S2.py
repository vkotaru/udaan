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


def zoh_step(q: S2, omega: TS2, omega_dot, h: float) -> tuple[S2, TS2]:
    r"""One zero-order-hold Euler substep of the ``S2`` kinematics, on the manifold.

    Every cable/link-carrying model in :mod:`udaan.models` integrates a
    direction ``q_i`` and its rate ``ω_i`` with the same two lines::

        q  <- q.step(ω h)                       # exact rotation, stays on S2
        ω  <- ω + h ω̇(q_old)                    # leaves T_q S2

    The first line is a true rotation, so ``‖q‖`` never drifts. The second is
    the problem: ``ω̇`` is built at ``q_old`` and is therefore tangent to
    ``q_old``, not to the ``q`` the first line just produced. Each substep
    leaves an ``O(h)`` residual along the *new* ``q``, and those residuals
    accumulate — halving the substep halves the accumulated ``|q·ω|`` rather
    than quartering it, which is what pinned the drift at first order.

    ``S2``'s angular velocity is only defined up to a component along ``q``:
    ``q̇ = ω × q`` annihilates it, and every model's ``ω̇`` is a cross product
    with ``q`` or a solve whose gauge row forces ``ω̇ · q = 0``. The residual is
    therefore a *gauge* quantity of the continuous dynamics, dynamically inert
    there but free to grow without bound. Projecting it out with
    :meth:`TS2.transport` — ``(I - q qᵀ) ω`` at the updated ``q`` — costs one
    outer product per link and pins ``|q·ω|`` at machine precision for good.

    Two things do *not* change **within one substep**, which is why this is
    safe to apply uniformly:

    * ``q̇ = ω × q`` is unaffected, so ``q_next``, the velocities derived from
      it and the centripetal ``‖q̇‖²`` coefficients are bit-identical for that
      substep.
    * The integrator stays first-order accurate. Removing a component that no
      right-hand side reads at that substep cannot change the local truncation
      error.

    Over a *run*, however, the trajectory does move, and it is worth being
    exact about why — "the gauge is dynamically inert" is true of the
    continuous dynamics and not of this discrete step. ``ω_next`` is what the
    next substep hands to :meth:`S2.step`, and
    ``q_next = expm(hat(ω h)) q`` is **not** gauge-invariant: a component of
    ``ω`` along ``q`` tilts the rotation axis and lengthens the angle, so the
    same ``q̇`` produces a different ``q_next`` at second order. Every
    cable-carrying model therefore integrates a slightly different trajectory
    once the drift is removed. Measured over a 3 s closed-loop tracking run,
    the payload position moves by 2.81e-04 m (M2), 9.97e-05 m (M3),
    5.83e-04 m (M4) and 4.58e-05 m (M5); M6's runs from its default spawn keep
    ``q·ω`` at exactly zero, so nothing there moves at all (< 1e-13 m).

    M5 and M6 have a second path on top of that one, and it is the reason the
    change is a *fix* rather than a wash: their LQR error states
    (``quadrotor_flexcable/payload_controller.py`` and
    ``multi_quad_hose/hose_controller.py``) build the link-rate block as
    ``ω + q × (q × ω_d)``, i.e. ``ω - (I - q qᵀ) ω_d`` — the *reference* rate
    is projected and the measured one is not. The drifting gauge component was
    therefore entering the feedback directly, as an error the controller could
    neither reduce nor make sense of. Pinning it at machine precision removes
    that term outright.

    Args:
        q: current direction on ``S2``.
        omega: current rate in ``T_q S2``.
        omega_dot: angular acceleration evaluated at the *current* ``q``.
        h: substep size.

    Returns:
        ``(q_next, omega_next)`` with ``omega_next`` exactly tangent to
        ``q_next``.
    """
    q_next = q.step(np.asarray(omega) * h)
    omega_next = TS2(np.asarray(omega) + np.asarray(omega_dot) * h).transport(q_next)
    return q_next, omega_next
