r"""LQR for quadrotors on a hose pinned to a ground anchor (M7 / M8).

M6's hose LQR (:class:`~udaan.control.multi_quad_hose.MultiQuadHoseController`)
with the anchor's two coordinate blocks removed, solved at the pinned trim, and
carried round the anchor by the system's rotational symmetry.

.. math::

    (f_k, M_k) = (f_k^d, M_k^d) + [K\,T_θ\,s]_k ,
    \qquad
    s = \bigl[\{\xi_i\},\; \{\delta\omega_i\},\; \{\eta_k\},\; \{\delta\Omega_k\}\bigr],

with ``(f^d, M^d)`` and the reference state from
:class:`~udaan.flatness.TetheredHose` — the static trim at the
:attr:`setpoint` when no trajectory is bound — and ``T_θ`` the azimuth
transport below.

The error state and the linearisation
-------------------------------------
The error blocks are M6's (``ξ_i = q_i^d × q_i``, ``δω_i``, ``η_k``,
``δΩ_k``) without ``e_{x_0}, e_{v_0}``: the anchor vertex does not move.  The
linearisation is M6's variational one on the reduced equations of motion, the
link rows of M6's system with the root row and column deleted and
``α = \ddot x_0 + g e_3`` fixed at ``g e_3``:

.. math::

    \bar M\,\dot\omega = b - \bigl[M_{0i}\hat q_i\bigr]_i\, g\, e_3 ,

where ``\bar M`` is M6's link block (the ``M_{ii} I`` gauge fix included).
The Riccati equation is solved on the tangent-reduced pair, exactly as M6
does, with M6's cost weights for the link, rate, attitude, body-rate, thrust
and moment blocks (:data:`~udaan.core.defaults.DEFAULT_HOSE_LQR_Q_LINK` and
the rest).  There is no root weight to give.

The azimuth transport
---------------------
A pinned hose's trim depends on where the formation stands round the anchor,
so a gain solved at one azimuth and held in world coordinates (as M6 holds its
gain) is wrong elsewhere.  The system is symmetric under rotation about the
anchor's vertical axis, except that every vehicle keeps ``ψ ≡ 0``: rotating a
trim by ``E = R_z(Δθ)`` gives a trim whose airframes are ``E R_k^0``, while
the reference flown is ``R_k^d``, the same thrust axis at zero yaw.  The two
differ by a rotation about the body ``z`` axis,

.. math::

    Y_k = R_k^{d\top} E\, R_k^0 ,

and the gain solved at azimuth ``θ_0`` is applied at ``θ_0 + Δθ`` to the
error rotated back into its own frame: each link block by ``E^\top``, each
vehicle's ``η_k`` and ``δΩ_k`` by ``Y_k^\top``, and each moment it returns by
``Y_k``.  ``Δθ`` is the azimuth of the reference's top vehicle about the
anchor.  The map is exact where ``Y_k`` commutes with the airframe inertia
(``Δθ = 180°``, or a yaw-symmetric airframe) and close elsewhere: the
generated airframe's ``J_x ≠ J_y`` is the difference.

One gain serves a whole flight: it is solved at the static trim at
``setpoint(0)`` (assigning a new setpoint re-solves it there), and the
transport accounts for a change of azimuth only.  A reference that changes
the standoff or height, or the whirl part of a circle, is regulated with the
gain of the start trim.
"""

from __future__ import annotations

import numpy as np

from ...core.defaults import (
    DEFAULT_HOSE_LQR_Q_ANGULAR_RATE,
    DEFAULT_HOSE_LQR_Q_ATTITUDE,
    DEFAULT_HOSE_LQR_Q_LINK,
    DEFAULT_HOSE_LQR_Q_LINK_RATE,
    DEFAULT_HOSE_LQR_R_MOMENT,
    DEFAULT_HOSE_LQR_R_THRUST,
    DEFAULT_HOSE_MIN_TENSION,
    DEFAULT_TETHER_ANCHOR_HEIGHT,
    DEFAULT_TETHER_HEIGHT_RATIO,
    DEFAULT_TETHER_LENGTH,
    DEFAULT_TETHER_LINEAR_DENSITY,
    DEFAULT_TETHER_NUM_LINKS,
    DEFAULT_TETHER_STANDOFF_RATIO,
    GRAVITY,
)
from ...flatness.multi_quad_hose import MultiQuadHoseRefState
from ...flatness.tethered_hose import TetheredHose, TetheredHoseInputs
from ...utils import hat, vee
from ..multi_quad_hose import MultiQuadHoseController
from ..multi_quad_hose.hose_controller import _tangent_basis

__all__ = ["TetheredHoseController", "default_top_position"]

_E3 = np.array([0.0, 0.0, 1.0])

#: Horizontal distance, in metres, below which the top vehicle is treated as
#: directly above the anchor: the azimuth is undefined there and the gain is
#: applied untransported.
_MIN_AZIMUTH_RADIUS = 1e-9


def default_top_position(hose_length: float, anchor_height: float = 0.0) -> np.ndarray:
    """The top vehicle's default station: ``(0.5 L, 0, h_a + 0.8 L)``.

    The two fractions are :data:`~udaan.core.defaults.DEFAULT_TETHER_STANDOFF_RATIO`
    and :data:`~udaan.core.defaults.DEFAULT_TETHER_HEIGHT_RATIO`.
    """
    L = float(hose_length)
    return np.array(
        [
            DEFAULT_TETHER_STANDOFF_RATIO * L,
            0.0,
            float(anchor_height) + DEFAULT_TETHER_HEIGHT_RATIO * L,
        ]
    )


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TetheredHoseController(MultiQuadHoseController):
    """Infinite-horizon LQR for ``nQ`` quadrotors on a hose pinned at vertex 0.

    ``compute(t, state[, reference])`` returns the ``(nQ, 4)`` wrench stack.

    The :attr:`setpoint` is the **top vehicle's** position,
    ``t -> (pos, vel, acc)``; without a bound trajectory the loop regulates
    against the static trim with the vehicles evenly on the line from the
    anchor to ``setpoint(t)[0]``.  The default setpoint is
    :func:`default_top_position`.

    Args:
        num_links: ``n``.
        attachments: vehicle vertices, strictly increasing, all ``>= 1``, the
            last ``n``.  Default ``(n,)``, one vehicle at the top.
        masses: per-vehicle airframe masses (as for M6's controller).
        inertias: per-vehicle inertia tensors.
        vertex_masses: the hose's lumped mass at each vertex.
        link_lengths: per-link lengths.
        hose_length: total length, when ``link_lengths`` is not given.
        hose_mass: total hose mass, when ``vertex_masses`` is not given;
            default 0.25 kg per metre of hose.
        gravity: ``g``.
        min_tension: floor under every derived link tension in the flat map.
        anchor_height: height of the anchor (vertex 0), which sits at
            ``(0, 0, anchor_height)``.
        q_link: M6's weight on the link-direction errors.
        q_link_rate: M6's weight on the link-rate errors.
        q_attitude: M6's weight on the airframe attitude errors.
        q_angular_rate: M6's weight on the body-rate errors.
        r_thrust: M6's weight on each thrust correction.
        r_moment: M6's weight on each moment component.

    Raises:
        ValueError: for a malformed parameter, a vehicle at vertex 0, or a
            root weight (``q_root_position``/``q_root_velocity``: the anchor
            has no position error to weight).
    """

    def __init__(
        self,
        *,
        num_links: int = DEFAULT_TETHER_NUM_LINKS,
        attachments=None,
        masses=None,
        inertias=None,
        vertex_masses=None,
        link_lengths=None,
        hose_length: float = DEFAULT_TETHER_LENGTH,
        hose_mass: float | None = None,
        gravity: float = GRAVITY,
        anchor_height: float = DEFAULT_TETHER_ANCHOR_HEIGHT,
        q_link: float = DEFAULT_HOSE_LQR_Q_LINK,
        q_link_rate: float = DEFAULT_HOSE_LQR_Q_LINK_RATE,
        q_attitude: float = DEFAULT_HOSE_LQR_Q_ATTITUDE,
        q_angular_rate: float = DEFAULT_HOSE_LQR_Q_ANGULAR_RATE,
        r_thrust: float = DEFAULT_HOSE_LQR_R_THRUST,
        r_moment: float = DEFAULT_HOSE_LQR_R_MOMENT,
        min_tension: float = DEFAULT_HOSE_MIN_TENSION,
        **kwargs,
    ):
        for name in ("q_root_position", "q_root_velocity", "sag_ratio", "horizontal_tension"):
            if name in kwargs:
                raise ValueError(
                    f"{type(self).__name__} takes no '{name}': the anchor vertex does not move, "
                    "and the trim is set by the vehicle positions (the setpoint)"
                )
        n = int(num_links)
        self._anchor: np.ndarray = np.array([0.0, 0.0, float(anchor_height)])
        if link_lengths is None:
            length = float(hose_length)
        else:
            length = float(np.sum(np.broadcast_to(np.asarray(link_lengths, dtype=float), (n,))))
        if "setpoint" not in kwargs:
            top = default_top_position(length, anchor_height)
            kwargs["setpoint"] = lambda t: (top, np.zeros(3), np.zeros(3))
        self._trim_cache: dict[tuple[float, ...], tuple] = {}
        super().__init__(
            num_links=n,
            attachments=(n,) if attachments is None else attachments,
            masses=masses,
            inertias=inertias,
            vertex_masses=vertex_masses,
            link_lengths=link_lengths,
            hose_length=hose_length,
            hose_mass=DEFAULT_TETHER_LINEAR_DENSITY * length if hose_mass is None else hose_mass,
            gravity=gravity,
            q_link=q_link,
            q_link_rate=q_link_rate,
            q_attitude=q_attitude,
            q_angular_rate=q_angular_rate,
            r_thrust=r_thrust,
            r_moment=r_moment,
            min_tension=min_tension,
            **kwargs,
        )

    # ─── Parameter plumbing ────────────────────────────────────────

    def _set_parameters(self, *, masses, inertias, vertex_masses, link_lengths) -> None:
        """M6's validation, then the pinned flat map and its trim."""
        if self._attachments[0] < 1:
            raise ValueError(
                f"vertex 0 is the anchor, so every vehicle must sit at a vertex >= 1; got "
                f"{self._attachments}"
            )
        # M6's routine builds (and this class replaces) an M6 map; until the
        # pinned map exists, _solve_trim_tensions returns a placeholder.
        self._flat_map = None  # type: ignore[assignment]
        super()._set_parameters(
            masses=masses,
            inertias=inertias,
            vertex_masses=vertex_masses,
            link_lengths=link_lengths,
        )
        self._tethered_map = TetheredHose(
            num_links=self._num_links,
            attachments=self._attachments,
            vertex_masses=self._vertex_masses.copy(),
            link_lengths=self._link_lengths.copy(),
            masses=self._masses.copy(),
            inertias=self._inertias.copy(),
            gravity=self._g,
            min_tension=self._min_tension,
            anchor=self._anchor,
        )
        self._flat_map = self._tethered_map  # type: ignore[assignment]
        self._trim_cache = {}
        self._trim_tensions = self._solve_trim_tensions()

    def _solve_trim_tensions(self) -> np.ndarray:
        """``(nQ, 3)``: the flat tensions of the static trim at ``setpoint(0)``.

        Called once by M6's parameter routine before the pinned map exists,
        where only the shape matters.
        """
        if not isinstance(getattr(self, "_flat_map", None), TetheredHose):
            return np.zeros((self._num_quadrotors, 3))
        return np.asarray(self.trim_at(self.setpoint(0.0)[0]).flat_tensions)

    @classmethod
    def from_model(cls, model, **kwargs) -> TetheredHoseController:
        """Build from a :class:`~udaan.models.tethered_hose.TetheredHoseBase`-like model."""
        kwargs.setdefault("anchor_height", float(np.asarray(model.anchor, dtype=float)[2]))
        return cls(
            num_links=model.num_links,
            attachments=model.attachments,
            masses=np.asarray(model.masses, dtype=float).copy(),
            inertias=np.asarray(model.inertias, dtype=float).copy(),
            vertex_masses=np.asarray(model.vertex_masses, dtype=float).copy(),
            link_lengths=np.asarray(model.link_lengths, dtype=float).copy(),
            **kwargs,
        )

    # ─── Properties ────────────────────────────────────────────────

    @property
    def setpoint(self):
        """The top vehicle's reference, ``t -> (pos, vel, acc, ...)``.

        Assigning it drops the cached gain and re-derives :attr:`trim_tensions`:
        both belong to the trim at the new ``setpoint(0)``.  The model's
        stability warning is issued at construction, for the default station;
        call :meth:`sampled_growth_rate` after binding a new station to check it.
        """
        return self._setpoint

    @setpoint.setter
    def setpoint(self, fn):
        self._setpoint = fn
        if hasattr(self, "_gain"):
            self._invalidate_gain()
            self._trim_tensions = self._solve_trim_tensions()

    @property
    def anchor(self) -> np.ndarray:
        """Position of the anchor vertex, ``(0, 0, anchor_height)``."""
        return np.array(self._anchor, dtype=float)

    @property
    def cost_weights(self) -> dict[str, float]:
        """The six cost weights in use (M6's, without the root blocks)."""
        return {k: v for k, v in self._weights.items() if not k.startswith("q_root")}

    @property
    def state_dim(self) -> int:
        """``6n + 6nQ`` — the embedded error :meth:`error_state` returns."""
        return 6 * self._num_links + 6 * self._num_quadrotors

    @property
    def trim_tensions(self) -> np.ndarray:
        """``(nQ, 3)`` flat tensions of the static trim at ``setpoint(0)``: ``𝛕_0`` first.

        Read-only: the trim is set by the vehicle positions, through
        :attr:`setpoint`.
        """
        return self._trim_tensions

    @trim_tensions.setter
    def trim_tensions(self, value):
        raise AttributeError(
            "a tethered hose's trim is set by its vehicle positions; assign the setpoint instead"
        )

    @property
    def flat_map(self) -> TetheredHose:  # type: ignore[override]
        """The :class:`~udaan.flatness.TetheredHose` map built from these parameters."""
        return self._tethered_map

    @property
    def jet_order(self) -> int:
        """``2n + 2`` — the anchor tension's required jet depth."""
        return self._tethered_map.jet_order

    # ─── Trim and reference ────────────────────────────────────────

    def trim_at(self, top_position):
        """The static :class:`~udaan.utils.hose.TetheredTrim` with the top vehicle at ``top_position``."""
        return self._trim_and_reference(top_position)[0]

    def _trim_and_reference(self, top_position):
        key = tuple(float(v) for v in np.asarray(top_position, dtype=float).reshape(3))
        cached = self._trim_cache.get(key)
        if cached is None:
            trim = self._tethered_map.trim(np.array(key))
            ref, inputs = self._tethered_map.recover(
                self._tethered_map.constant_flats(trim.flat_tensions)
            )
            cached = (trim, ref, inputs)
            if len(self._trim_cache) > 64:
                self._trim_cache.clear()
            self._trim_cache[key] = cached
        return cached

    def equilibrium_reference(  # type: ignore[override]
        self, t: float
    ) -> tuple[MultiQuadHoseRefState, TetheredHoseInputs]:
        """The static trim with the top vehicle at ``setpoint(t)``'s position."""
        _, ref, inputs = self._trim_and_reference(self.setpoint(t)[0])
        return ref, inputs

    def build_flats(self, root_jet):
        """Not used by a pinned hose: its flat output has no root position."""
        raise NotImplementedError(
            "a tethered hose's flat output is the anchor-link force, not a root position; "
            "use flat_map.constant_flats() or a TetherReference"
        )

    # ─── Error state ───────────────────────────────────────────────

    def error_state(self, state, reference) -> np.ndarray:
        r"""The ``(6n + 6nQ,)`` error ``[ξ_i, δω_i, η_k, δΩ_k]`` (M6's blocks, no root)."""
        n = self._num_links
        nQ = self._num_quadrotors
        if len(state.link_attitudes) != n or len(reference.link_attitudes) != n:
            raise ValueError(f"state and reference must both describe {n} links")
        if len(state.orientations) != nQ or len(reference.orientations) != nQ:
            raise ValueError(f"state and reference must both describe {nQ} vehicles")
        error = np.zeros(self.state_dim)
        for i in range(n):
            q = np.asarray(state.link_attitudes[i], dtype=float)
            omega = np.asarray(state.link_angular_velocities[i], dtype=float)
            q_d = np.asarray(reference.link_attitudes[i], dtype=float)
            omega_d = np.asarray(reference.link_angular_velocities[i], dtype=float)
            error[3 * i : 3 * i + 3] = np.cross(q_d, q)
            error[3 * n + 3 * i : 3 * n + 3 * i + 3] = omega + np.cross(q, np.cross(q, omega_d))
        base = 6 * n
        for k in range(nQ):
            R = np.asarray(state.orientations[k], dtype=float)
            Omega = np.asarray(state.angular_velocities[k], dtype=float)
            R_d = np.asarray(reference.orientations[k], dtype=float)
            Omega_d = np.asarray(reference.angular_velocities[k], dtype=float)
            error[base + 3 * k : base + 3 * k + 3] = 0.5 * vee(R_d.T @ R - R.T @ R_d)
            error[base + 3 * nQ + 3 * k : base + 3 * nQ + 3 * k + 3] = Omega - R.T @ R_d @ Omega_d
        return error

    # ─── Linearisation and gain ────────────────────────────────────

    def linearize(self, reference=None, inputs=None) -> tuple[np.ndarray, np.ndarray]:
        r"""Variational linearisation ``(A, B)`` of the pinned hose, ``ṡ = A s + B δu``.

        ``A`` is ``(6n + 6nQ)``-square in :meth:`error_state`'s block order and
        ``B`` is ``(6n + 6nQ) × 4nQ``, thrusts first, then the moments.  M6's
        linearisation of the link rows (its ``state_jac``/``input_jac``) on
        M6's link block of the mass matrix, with ``α`` fixed at ``g e_3``
        (``M_{0i}\hat q_i α`` moves to the right-hand side, where it is
        linearised at ``α_d = g e_3``).  With no arguments the reference is
        :meth:`equilibrium_reference` at ``t = 0``.
        """
        if (reference is None) != (inputs is None):
            raise ValueError("linearize() needs the reference state and its inputs together")
        if reference is None or inputs is None:
            reference, inputs = self.equilibrium_reference(0.0)

        n = self._num_links
        nQ = self._num_quadrotors
        eye = np.eye(3)
        _, M0i, Mij = self._mass_coefficients()
        lengths = self._link_lengths
        qd = np.asarray(reference.link_attitudes, dtype=float).reshape(n, 3)
        wd = np.asarray(reference.link_angular_velocities, dtype=float).reshape(n, 3)
        dwd = np.asarray(reference.link_angular_accelerations, dtype=float).reshape(n, 3)
        R_d = [np.asarray(R, dtype=float) for R in reference.orientations]
        Omega_d = [np.asarray(w, dtype=float) for w in reference.angular_velocities]
        thrust_d = np.asarray(inputs.thrusts, dtype=float).reshape(nQ)
        alpha_d = self._ge3

        qd_hat = [hat(qd[i]) for i in range(n)]
        speed_sq = np.array([float(np.dot(wd[i], wd[i])) for i in range(n)])
        vertex_force = np.zeros((n + 1, 3))
        for k, vertex in enumerate(self._attachments):
            vertex_force[vertex] = thrust_d[k] * (R_d[k] @ _E3)
        below = np.cumsum(vertex_force[::-1], axis=0)[::-1]

        dim = self.state_dim
        rate0 = 3 * n
        eta0 = 6 * n
        omega0 = eta0 + 3 * nQ

        mass = np.zeros((3 * n, 3 * n))
        for i in range(n):
            for j in range(n):
                mass[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = (
                    Mij[i, j] * eye if i == j else -Mij[i, j] * (qd_hat[i] @ qd_hat[j])
                )

        state_jac = np.zeros((3 * n, dim))
        for i in range(n):
            r = 3 * i
            block = -M0i[i] * hat(alpha_d) + lengths[i] * hat(below[i + 1])
            for k in range(n):
                if k == i:
                    continue
                block = block + Mij[i, k] * (hat(qd_hat[k] @ dwd[k]) + speed_sq[k] * qd_hat[k])
            state_jac[r : r + 3, 3 * i : 3 * i + 3] = block @ qd_hat[i]
            for j in range(n):
                if j == i:
                    continue
                state_jac[r : r + 3, 3 * j : 3 * j + 3] = (
                    Mij[i, j] * qd_hat[i] @ (hat(dwd[j]) - speed_sq[j] * eye) @ qd_hat[j]
                )
                state_jac[r : r + 3, rate0 + 3 * j : rate0 + 3 * j + 3] = (
                    2.0 * Mij[i, j] * qd_hat[i] @ np.outer(qd[j], wd[j])
                )
            for k, vertex in enumerate(self._attachments):
                if vertex <= i:
                    continue
                col = eta0 + 3 * k
                state_jac[r : r + 3, col : col + 3] = (
                    -lengths[i] * thrust_d[k] * qd_hat[i] @ R_d[k] @ hat(_E3)
                )

        input_jac = np.zeros((3 * n, self.input_dim))
        for k, vertex in enumerate(self._attachments):
            for i in range(n):
                if vertex <= i:
                    continue
                input_jac[3 * i : 3 * i + 3, k] = lengths[i] * (qd_hat[i] @ R_d[k] @ _E3)

        solved_state = np.linalg.solve(mass, state_jac)
        solved_input = np.linalg.solve(mass, input_jac)

        A = np.zeros((dim, dim))
        for i in range(n):
            r = 3 * i
            A[r : r + 3, r : r + 3] = np.outer(qd[i], qd[i]) @ hat(wd[i])
            A[r : r + 3, rate0 + r : rate0 + r + 3] = eye - np.outer(qd[i], qd[i])
        A[rate0 : rate0 + 3 * n, :] = solved_state
        for k in range(nQ):
            r = eta0 + 3 * k
            A[r : r + 3, r : r + 3] = -hat(Omega_d[k])
            A[r : r + 3, omega0 + 3 * k : omega0 + 3 * k + 3] = eye
            s = omega0 + 3 * k
            A[s : s + 3, s : s + 3] = self._inertia_invs[k] @ (
                hat(self._inertias[k] @ Omega_d[k]) - hat(Omega_d[k]) @ self._inertias[k]
            )

        B = np.zeros((dim, self.input_dim))
        B[rate0 : rate0 + 3 * n, :] = solved_input
        for k in range(nQ):
            s = omega0 + 3 * k
            B[s : s + 3, nQ + 3 * k : nQ + 3 * k + 3] = self._inertia_invs[k]
        return A, B

    def tangent_projection(self, link_directions=None) -> np.ndarray:
        """``W``, the ``(6n + 6nQ) × (4n + 6nQ)`` non-gauge basis (no root block)."""
        n = self._num_links
        nQ = self._num_quadrotors
        if link_directions is None:
            _, ref, _ = self._trim_and_reference(self.setpoint(0.0)[0])
            link_directions = ref.link_attitudes
        directions = np.asarray(link_directions, dtype=float).reshape(n, 3)
        basis = np.zeros((self.state_dim, 4 * n + 6 * nQ))
        for i in range(n):
            tangent = _tangent_basis(directions[i])
            basis[3 * i : 3 * i + 3, 2 * i : 2 * i + 2] = tangent
            basis[3 * n + 3 * i : 3 * n + 3 * i + 3, 2 * n + 2 * i : 2 * n + 2 * i + 2] = tangent
        basis[6 * n :, 4 * n :] = np.eye(6 * nQ)
        return basis

    def cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """``(Q_r, R)`` — M6's blocks for links, rates, attitudes and body rates."""
        n = self._num_links
        nQ = self._num_quadrotors
        w = self._weights
        diagonal = np.concatenate(
            [
                np.full(2 * n, w["q_link"]),
                np.full(2 * n, w["q_link_rate"]),
                np.full(3 * nQ, w["q_attitude"]),
                np.full(3 * nQ, w["q_angular_rate"]),
            ]
        )
        inputs = np.concatenate([np.full(nQ, w["r_thrust"]), np.full(3 * nQ, w["r_moment"])])
        return np.diag(diagonal), np.diag(inputs)

    def _solve_gain(self) -> None:
        """Solve the Riccati equation at the static trim at ``setpoint(0)``.

        Also records the trim's azimuth and airframe attitudes, which the
        azimuth transport in :meth:`compute` measures from.
        """
        reference, inputs = self.equilibrium_reference(0.0)
        A, B = self.linearize(reference, inputs)
        self.set_gain_from(A, B, reference.link_attitudes)
        self._gain_azimuth = self._azimuth(reference)
        self._gain_orientations = [np.asarray(R, dtype=float) for R in reference.orientations]

    def set_gain_from(self, A, B, link_directions=None, reference=None) -> None:
        """Re-solve the Riccati equation at ``(A, B)`` and adopt the gain.

        ``reference`` is the state ``(A, B)`` was linearised at; the azimuth
        transport measures from it.  Without it the transport keeps the
        azimuth and attitudes of the previous gain (the trim at
        ``setpoint(0)`` if there was none).
        """
        super().set_gain_from(A, B, link_directions)
        if reference is not None:
            self._gain_azimuth = self._azimuth(reference)
            self._gain_orientations = [np.asarray(R, dtype=float) for R in reference.orientations]
        elif not hasattr(self, "_gain_orientations"):
            ref, _ = self.equilibrium_reference(0.0)
            self._gain_azimuth = self._azimuth(ref)
            self._gain_orientations = [np.asarray(R, dtype=float) for R in ref.orientations]

    @property
    def reduced_gain(self) -> np.ndarray:
        """``K_r``, the gain on the tangent-reduced error, ``(4nQ, 4n + 6nQ)``."""
        _, ref, _ = self._trim_and_reference(self.setpoint(0.0)[0])
        return np.asarray(self.gain @ self.tangent_projection(ref.link_attitudes))

    # ─── Azimuth transport ─────────────────────────────────────────

    def _azimuth(self, reference) -> float:
        offset = np.asarray(reference.positions[-1], dtype=float) - self._anchor
        if float(np.hypot(offset[0], offset[1])) < _MIN_AZIMUTH_RADIUS:
            return float("nan")
        return float(np.arctan2(offset[1], offset[0]))

    def transport(self, reference) -> tuple[np.ndarray, list[np.ndarray]]:
        """``(E, [Y_k])`` carrying the gain from its own azimuth to ``reference``'s.

        ``E = R_z(Δθ)`` acts on the link blocks and ``Y_k = R_k^{d⊤} E R_k^0``
        on vehicle ``k``'s; see the module docstring.  The identity (and every
        ``Y_k = R_k^{d⊤} R_k^0``) where either azimuth is undefined.
        """
        self._ensure_gain()
        theta = self._azimuth(reference)
        delta = theta - self._gain_azimuth
        E = np.eye(3) if not np.isfinite(delta) else _rot_z(delta)
        Ys = [
            np.asarray(R_d, dtype=float).T @ E @ R0
            for R_d, R0 in zip(reference.orientations, self._gain_orientations, strict=True)
        ]
        return E, Ys

    # ─── Control ───────────────────────────────────────────────────

    def compute(self, *args) -> np.ndarray:
        """The ``(nQ, 4)`` wrench stack ``[f_k, M_kx, M_ky, M_kz]``.

        Args:
            args: ``(t, state)``, or ``(t, state, reference)`` with
                ``reference`` the ``(RefState, Inputs)`` pair
                :attr:`flat_map` recovers.  Without it, the static trim at the
                setpoint.
        """
        t = args[0]
        state = args[1]
        reference = args[2] if len(args) > 2 and args[2] is not None else None
        if reference is None:
            ref, feedforward = self.equilibrium_reference(t)
        else:
            ref, feedforward = reference

        n = self._num_links
        nQ = self._num_quadrotors
        gain = self._ensure_gain()
        error = self.error_state(state, ref)
        E, Ys = self.transport(ref)
        for i in range(2 * n):
            error[3 * i : 3 * i + 3] = E.T @ error[3 * i : 3 * i + 3]
        for k in range(nQ):
            for base in (6 * n, 6 * n + 3 * nQ):
                block = slice(base + 3 * k, base + 3 * k + 3)
                error[block] = Ys[k].T @ error[block]
        correction = gain @ error
        moments = correction[nQ:].reshape(nQ, 3)
        moments = np.stack([Ys[k] @ moments[k] for k in range(nQ)])
        wrench = np.zeros((nQ, 4))
        wrench[:, 0] = np.asarray(feedforward.thrusts, dtype=float) + correction[:nQ]
        wrench[:, 1:] = np.asarray(feedforward.moments, dtype=float) + moments
        return wrench

    # ─── Analysis ──────────────────────────────────────────────────

    def sampled_closed_loop(self, dt: float, substeps: int) -> np.ndarray:
        """The closed loop of the analytical model's step at the trim, on the reduced error.

        One control step holds ``δu = K s`` for ``substeps`` forward-Euler
        substeps of ``dt/substeps``: ``s⁺ = (Φ + Γ K_r) s`` with ``Φ = (I +
        hA_r)^N`` and ``Γ = Σ_j (I + hA_r)^j h B_r``.  Returns that matrix.
        """
        _, ref, inputs = self._trim_and_reference(self.setpoint(0.0)[0])
        A, B = self.linearize(ref, inputs)
        basis = self.tangent_projection(ref.link_attitudes)
        A_r = basis.T @ A @ basis
        B_r = basis.T @ B
        K_r = self.gain @ basis
        h = float(dt) / int(substeps)
        step = np.eye(A_r.shape[0]) + h * A_r
        phi = np.eye(A_r.shape[0])
        gamma = np.zeros_like(B_r)
        for _ in range(int(substeps)):
            gamma = step @ gamma + h * B_r
            phi = step @ phi
        return np.asarray(phi + gamma @ K_r)

    def sampled_growth_rate(self, dt: float, substeps: int) -> float:
        """``ln ρ / dt`` of :meth:`sampled_closed_loop`, in 1/s; positive means it diverges."""
        rho = float(np.max(np.abs(np.linalg.eigvals(self.sampled_closed_loop(dt, substeps)))))
        return float(np.log(rho) / float(dt))
