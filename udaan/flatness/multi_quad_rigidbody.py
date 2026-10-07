r"""Differential flatness for ``nQ`` quadrotors carrying one cable-suspended
**rigid-body** payload (M4).

Flat output::

    y = (x_L, R_L, {ψ_i}_{i=1..nQ}, λ)

— the payload **pose**, position *and attitude*, one yaw angle per quadrotor,
and the ``3 nQ − 6`` null-space coordinates of the wrench→tension allocation.
Counting: ``3 + 3 + nQ + 3(nQ − 6/3)`` = ``6 + nQ + 3nQ − 6 = 4 nQ = dim u``,
the dimension-matching condition (see ``docs/theory/differential-flatness/``).

This is M3 (:mod:`udaan.flatness.multi_quad_cspayload`) plus the payload
attitude.  The change is not cosmetic:

* the payload's balance is a **wrench** balance ``Φ τ = W``, not a force
  balance, so the free choice is a ``3 nQ − 6`` dimensional null space rather
  than M3's ``3(nQ − 1)`` internal-force space, and resolving it is the job of
  the separate primitive :mod:`udaan.flatness.allocation`;
* the cable hangs off a *moving, rotating* attachment point
  ``x_{P_i} = x_L + R_L r_i``, so the payload rotation enters every cable's
  chain, twice — once through ``x_{P_i}`` and once through the frame change
  ``𝐓_i = -R_L τ_i`` between the body-frame allocation and the world-frame
  Newton equations;
* the payload attitude brings its own derivative chain, and it is *deeper* than
  the position chain measured in the flat variable's own derivatives — see
  "Derivative orders" below.

Sign convention
---------------
Unchanged from M3 and :mod:`udaan.flatness.cable`: ``𝐓_i = T_i q_i`` is the
force the cable exerts **on quadrotor** ``i``, so ``q_i`` points from the
quadrotor to the payload and the force on the *payload* is ``-𝐓_i``.  The
allocation works in the payload body frame with ``τ_i := R_L^⊤(-𝐓_i)``, the
force on the *load*, which is the convention
:func:`udaan.utils.grasp.grasp_map` and
:class:`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController` use.
Hence ``𝐓_i = -R_L τ_i``.

Derivative orders required::

    x_L      through pop            (x_L, …, x_L^(6))
    Ω_L      through Ω_L^(5)        (equivalently R_L through R_L^(6))
    ψ_i      through ψ̈_i            (ψ_i, ψ̇_i, ψ̈_i)

The chain that fixes both: the moment ``M_i`` needs ``B̈_i``, hence
``x_{Q_i}^{(4)}``, hence ``q_i^{(4)}``, hence ``𝐓_i^{(4)}``, hence
``τ^{(4)} = Φ⁺ W^{(4)}``.  So **the wrench is needed through its 4th
derivative**, and the wrench is second order in the payload pose:
``W_f = m_L R_L^⊤(\ddot x_L + g e_3)`` gives ``x_L^{(6)}`` and
``W_m = J_L \dot\Omega_L + \Omega_L \times J_L \Omega_L`` gives ``Ω_L^{(5)}``.
Both channels therefore bottom out at the *sixth* derivative of the payload
pose; expressed in body rates, which is how the attitude flat output is carried
here, that reads as an order-5 ``Ω_L`` jet.  Separately, ``R_L^{(4)}`` alone is
needed for the attachment point and the frame change — a shallower requirement
that the wrench chain subsumes.  M3's table stops at ``x_L^{(6)}``; M4 adds the
whole ``Ω_L`` column, and it is the reason
:class:`MultiQuadRigidbodyFlats` validates an order-5 jet where M3 needs none.

``nQ ≤ 2``
----------
``rank Φ`` is 3 at ``nQ = 1`` and 5 at ``nQ = 2`` (default ring), so the payload
attitude is **not** a flat output there: only wrenches inside ``range Φ`` can be
produced, and :meth:`MultiQuadRigidbody.recover` raises
:class:`~udaan.flatness.WrenchAllocationError` on anything else.  Level hover
still round-trips at every ``nQ ≥ 1`` — its wrench happens to lie in the range —
but a payload-attitude manoeuvre does not.  This is a property of the
mechanism, identical to the one
:attr:`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController.grasp_rank`
reports in closed loop; the flat map refuses rather than silently returning the
least-squares reference, because that reference does not solve the equations of
motion.

References:
    T. Lee, "Geometric Control of Quadrotor UAVs Transporting a Cable-Suspended
    Rigid Body", IEEE Transactions on Control Systems Technology, 2018.
    K. Sreenath and V. Kumar, "Dynamics, Control and Planning for Cooperative
    Manipulation of Payloads Suspended by Cables from Multiple Quadrotor
    Robots", Robotics: Science and Systems (RSS), 2013 — the point-mass case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

from ..core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_NUM_QUADROTORS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    DEFAULT_RIGID_PAYLOAD_INERTIA,
    DEFAULT_RIGID_PAYLOAD_MASS,
    GRAVITY,
)
from ..core.types import Vec3
from ..manif import S2, SO3, TS2, TSO3, hat
from ..utils.grasp import ring_attachment_points
from .allocation import (
    RingTensionDistribution,
    TensionDistributionPolicy,
    WrenchAllocation,
)
from .base import Flat2State
from .cable import tension_direction_jet
from .derivatives import leibniz_matmul
from .jet import Jet
from .quadrotor import _attitude_from_thrust_vector

__all__ = [
    "MultiQuadRigidbody",
    "MultiQuadRigidbodyFlats",
    "MultiQuadRigidbodyInputs",
    "MultiQuadRigidbodyRefState",
    "rotation_jet",
]

_E3 = np.array([0.0, 0.0, 1.0])

#: Derivative orders the map consumes — see "Derivative orders" above.
XL_ORDER = 6
OMEGA_ORDER = 5
PSI_ORDER = 2

#: Highest wrench derivative the recovery needs (``q_i^(4)`` closes the chain).
_WRENCH_ORDER = 4


def _zeros3() -> Vec3:
    return np.zeros(3)


def rotation_jet(orientation, angular_velocity: list[np.ndarray], order: int) -> list[np.ndarray]:
    r"""Derivative tuple ``[R, Ṙ, …, R^(order)]`` of a rotation from its body rate.

    Differentiating ``Ṙ = R \hat\Omega`` with the Leibniz rule gives the exact
    recursion

    .. math::

        R^{(k+1)} \;=\; \sum_{j=0}^{k} \binom{k}{j}\, R^{(j)}\,
                        \widehat{\Omega^{(k-j)}},

    so ``order`` derivatives of ``R`` cost ``order - 1`` derivatives of ``Ω``.
    Carrying the payload attitude as ``(R_L, J^m \Omega_L)`` rather than as a
    jet of matrices is what keeps the flat output chart-free: no Euler angles,
    no quaternion sign, and the derivatives stay exactly on ``SO(3)``'s tangent
    structure instead of drifting off it.

    Args:
        orientation: ``(3, 3)`` rotation matrix (or anything ``np.asarray``
            turns into one, e.g. an :class:`~udaan.manif.SO3`).
        angular_velocity: ``[Ω, Ω̇, …]``, at least ``order - 1`` derivatives.
        order: highest derivative of ``R`` to return.

    Returns:
        ``order + 1`` arrays of shape ``(3, 3)``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0; got {order}")
    if len(angular_velocity) < order:
        raise ValueError(
            f"rotation_jet needs Ω through derivative {order - 1} to build R^({order}); "
            f"got {len(angular_velocity)} level(s)"
        )
    R = [np.asarray(orientation, dtype=float)]
    for k in range(order):
        R.append(
            sum(
                comb(k, j) * (R[j] @ hat(np.asarray(angular_velocity[k - j], dtype=float)))
                for j in range(k + 1)
            )
        )
    return R


# ─── Flat output / reference state / inputs ───────────────────────────


@dataclass
class MultiQuadRigidbodyFlats:
    """Flat output of ``nQ`` quadrotors sharing one cable-suspended rigid body.

    The null-space part of the flat output is carried by the map's
    :class:`~udaan.flatness.TensionDistributionPolicy` rather than by this
    struct, exactly as M3 carries its internal forces on the tension-split
    policy: the shipped policies hold it constant, which is what makes
    ``τ^{(k)} = Φ⁺ W^{(k)}`` exact for ``k ≥ 1``.

    Attributes:
        x_L: dim-3 :class:`Jet` of the payload position, derivatives through pop
             (order ≥ 6).
        R_L: ``(3, 3)`` payload attitude (an :class:`~udaan.manif.SO3` is
             accepted).  Its derivatives are *not* stored — they are rebuilt
             from ``Omega_L`` by :func:`rotation_jet`.
        Omega_L: dim-3 :class:`Jet` of the payload **body** angular velocity,
             derivatives through ``Ω_L^{(5)}`` (order ≥ 5).  This is the rung
             M3 has no analogue of; ``W_m^{(4)}`` needs it.
        psi: length-``nQ`` list of scalar :class:`Jet` s, one yaw per quadrotor,
             each with derivatives through ``ψ̈`` (order ≥ 2).
    """

    x_L: Jet
    R_L: np.ndarray
    Omega_L: Jet
    psi: list[Jet]

    def __post_init__(self):
        if self.x_L.dim != 3:
            raise ValueError(f"x_L must be a 3-D jet; got dim={self.x_L.dim}")
        if self.x_L.order < XL_ORDER:
            raise ValueError(f"x_L.order must be >= {XL_ORDER} (pop); got {self.x_L.order}")

        R = np.asarray(self.R_L, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"R_L must be a 3x3 rotation matrix; got shape {R.shape}")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-8):
            raise ValueError("R_L must be orthonormal (R_Lᵀ R_L = I)")
        if float(np.linalg.det(R)) < 0.0:
            raise ValueError("R_L must have det = +1; got a reflection")
        self.R_L = R

        if self.Omega_L.dim != 3:
            raise ValueError(f"Omega_L must be a 3-D jet; got dim={self.Omega_L.dim}")
        if self.Omega_L.order < OMEGA_ORDER:
            raise ValueError(
                f"Omega_L.order must be >= {OMEGA_ORDER}; got {self.Omega_L.order}. "
                "The payload-attitude channel is one rung deeper than M3's: the "
                "wrench is needed through its 4th derivative and W_m is first "
                "order in Ω_L."
            )

        if isinstance(self.psi, Jet):
            raise ValueError("psi must be a list of scalar jets, one per quadrotor")
        self.psi = list(self.psi)
        if not self.psi:
            raise ValueError("psi must hold at least one yaw jet")
        for i, psi_i in enumerate(self.psi):
            if not isinstance(psi_i, Jet):
                raise ValueError(f"psi[{i}] must be a Jet; got {type(psi_i).__name__}")
            if not psi_i.is_scalar:
                raise ValueError(f"psi[{i}] must be a scalar jet; got dim={psi_i.dim}")
            if psi_i.order < PSI_ORDER:
                raise ValueError(f"psi[{i}].order must be >= {PSI_ORDER}; got {psi_i.order}")

    @property
    def num_quadrotors(self) -> int:
        """Number of quadrotors, inferred from the yaw list."""
        return len(self.psi)


@dataclass
class MultiQuadRigidbodyRefState:
    """Recovered kinematic reference state for the M4 system.

    Field naming mirrors
    :class:`udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyState` where the
    two overlap; the ``*_acceleration(s)`` entries are extras the flatness map
    also produces.  Every per-quadrotor field is a list indexed by quadrotor.
    """

    # Payload pose and twist
    payload_position: Vec3 = field(default_factory=_zeros3)
    payload_velocity: Vec3 = field(default_factory=_zeros3)
    payload_acceleration: Vec3 = field(default_factory=_zeros3)
    payload_orientation: SO3 = field(default_factory=SO3)
    payload_angular_velocity: TSO3 = field(default_factory=TSO3)
    payload_angular_acceleration: Vec3 = field(default_factory=_zeros3)

    # Per-quadrotor centre-of-mass kinematics
    positions: list[Vec3] = field(default_factory=list)
    velocities: list[Vec3] = field(default_factory=list)
    accelerations: list[Vec3] = field(default_factory=list)

    # Per-cable kinematics on TS²
    cable_attitudes: list[Vec3] = field(default_factory=list)
    cable_angular_velocities: list[Vec3] = field(default_factory=list)
    cable_angular_accelerations: list[Vec3] = field(default_factory=list)

    # Per-quadrotor attitude
    orientations: list[SO3] = field(default_factory=list)
    angular_velocities: list[TSO3] = field(default_factory=list)
    angular_accelerations: list[Vec3] = field(default_factory=list)

    @property
    def num_quadrotors(self) -> int:
        return len(self.positions)

    def as_state(self):
        """Project onto the model's ``MultiQuadRigidbodyState`` shape.

        Drops the ``*_acceleration(s)`` extras and wraps the manifold fields.
        """
        from ..models.multi_quad_rigidbody.base import MultiQuadRigidbodyState

        return MultiQuadRigidbodyState(
            num_quadrotors=self.num_quadrotors,
            payload_position=np.asarray(self.payload_position).copy(),
            payload_velocity=np.asarray(self.payload_velocity).copy(),
            payload_orientation=SO3(np.asarray(self.payload_orientation)),
            payload_angular_velocity=TSO3(np.asarray(self.payload_angular_velocity)),
            positions=[np.asarray(p).copy() for p in self.positions],
            velocities=[np.asarray(v).copy() for v in self.velocities],
            orientations=[SO3(np.asarray(R)) for R in self.orientations],
            angular_velocities=[TSO3(np.asarray(w)) for w in self.angular_velocities],
            cable_attitudes=[S2(np.asarray(q)) for q in self.cable_attitudes],
            cable_angular_velocities=[TS2(np.asarray(w)) for w in self.cable_angular_velocities],
        )


@dataclass
class MultiQuadRigidbodyInputs:
    """Feedforward inputs recovered from the flat output, one entry per quadrotor.

    Attributes:
        thrusts:  collective thrust magnitudes ``f_i``.
        moments:  body-frame moments ``M_i``.
        tensions: cable tensions ``T_i = ‖𝐓_i‖``.  Positive throughout the taut
                  regime; the two-sided slack refusal lives in
                  :meth:`~udaan.flatness.WrenchAllocation.check_taut`.
        cable_forces: ``(nQ, 3)`` body-frame forces ``τ_i`` on the *load* — the
                  raw output of the allocation, kept because it is what
                  ``‖Φ τ − W‖`` is checked against.
    """

    thrusts: list[float] = field(default_factory=list)
    moments: list[Vec3] = field(default_factory=list)
    tensions: list[float] = field(default_factory=list)
    cable_forces: list[Vec3] = field(default_factory=list)

    @property
    def wrench(self) -> np.ndarray:
        """``(nQ, 4)`` stack of ``[f_i, M_i]`` — the ``input="wrench"`` shape of
        :meth:`udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase.step`."""
        return np.array([[f, *M] for f, M in zip(self.thrusts, self.moments, strict=True)])


# ─── The map ──────────────────────────────────────────────────────────


class MultiQuadRigidbody(Flat2State):
    """Differential-flatness recovery for ``nQ`` quadrotors carrying one
    cable-suspended rigid-body payload through taut, massless cables.

    Args:
        num_quadrotors: ``nQ ≥ 1``.
        masses: scalar (broadcast) or ``(nQ,)`` quadrotor masses.
        inertias: ``(3,)`` diagonal, ``(3, 3)`` tensor (both broadcast) or
            ``(nQ, 3, 3)`` stack.
        payload_mass: ``m_L``.
        payload_inertia: ``(3,)`` diagonal or ``(3, 3)`` tensor ``J_L``.
        cable_lengths: scalar (broadcast) or ``(nQ,)`` cable lengths.
        attachment_points: ``(nQ, 3)`` body-frame attachment points ``r_i``;
            defaults to :func:`~udaan.utils.grasp.ring_attachment_points`.
        gravity: ``g``.
        formation_radius: radius of the default
            :class:`~udaan.flatness.RingTensionDistribution`.
        tension_policy: an explicit
            :class:`~udaan.flatness.TensionDistributionPolicy`; defaults to the
            ring built from the arguments above.
        min_tension: cables below this tension are reported as slack.
    """

    Flats = MultiQuadRigidbodyFlats
    RefState = MultiQuadRigidbodyRefState
    Inputs = MultiQuadRigidbodyInputs

    def __init__(
        self,
        *,
        num_quadrotors: int = DEFAULT_NUM_QUADROTORS,
        masses=DEFAULT_QUAD_MASS,
        inertias=DEFAULT_QUAD_INERTIA,
        payload_mass: float = DEFAULT_RIGID_PAYLOAD_MASS,
        payload_inertia=DEFAULT_RIGID_PAYLOAD_INERTIA,
        cable_lengths=DEFAULT_CABLE_LENGTH,
        attachment_points=None,
        gravity: float = GRAVITY,
        formation_radius: float = DEFAULT_FORMATION_RADIUS,
        tension_policy: TensionDistributionPolicy | None = None,
        min_tension: float | None = None,
    ):
        nQ = int(num_quadrotors)
        if nQ < 1:
            raise ValueError(f"num_quadrotors must be >= 1, got {nQ}")
        self._num_quadrotors = nQ

        self.masses = self._broadcast(masses, "masses")
        if np.any(self.masses <= 0.0):
            raise ValueError(f"masses must be positive; got {self.masses.tolist()}")

        self.cable_lengths = self._broadcast(cable_lengths, "cable_lengths")
        if np.any(self.cable_lengths <= 0.0):
            raise ValueError(f"cable_lengths must be positive; got {self.cable_lengths.tolist()}")

        self.inertias = self._broadcast_inertias(inertias)

        self.payload_mass = float(payload_mass)
        if self.payload_mass <= 0.0:
            raise ValueError(f"payload_mass must be positive; got {self.payload_mass}")

        J_L = np.asarray(payload_inertia, dtype=float)
        if J_L.shape == (3,):
            J_L = np.diag(J_L)
        if J_L.shape != (3, 3):
            raise ValueError(f"payload_inertia must be (3,) or (3, 3); got {J_L.shape}")
        if not np.all(np.linalg.eigvalsh(0.5 * (J_L + J_L.T)) > 0.0):
            raise ValueError("payload_inertia must be positive definite")
        self.payload_inertia = J_L

        points = (
            ring_attachment_points(nQ)
            if attachment_points is None
            else np.asarray(attachment_points, dtype=float)
        )
        if points.shape != (nQ, 3):
            raise ValueError(f"attachment_points must be ({nQ}, 3); got {points.shape}")
        self.attachment_points = points.copy()

        self.gravity = float(gravity)
        self.formation_radius = float(formation_radius)

        if tension_policy is None:
            tension_policy = RingTensionDistribution(
                num_quadrotors=nQ,
                payload_mass=self.payload_mass,
                cable_lengths=self.cable_lengths,
                attachment_points=self.attachment_points,
                formation_radius=self.formation_radius,
                gravity=self.gravity,
            )
        alloc_kwargs = {} if min_tension is None else {"min_tension": float(min_tension)}
        self.allocation = WrenchAllocation(
            self.attachment_points, policy=tension_policy, **alloc_kwargs
        )

    # ─── Parameter plumbing ────────────────────────────────────────

    def _broadcast(self, value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self._num_quadrotors, float(arr))
        arr = arr.reshape(-1)
        if arr.shape[0] != self._num_quadrotors:
            raise ValueError(f"{name} must have {self._num_quadrotors} entries, got {arr.shape[0]}")
        return arr

    def _broadcast_inertias(self, value) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        nQ = self._num_quadrotors
        if arr.shape == (3,):
            return np.tile(np.diag(arr), (nQ, 1, 1))
        if arr.shape == (3, 3):
            return np.tile(arr, (nQ, 1, 1))
        if arr.shape == (nQ, 3, 3):
            return arr
        raise ValueError(f"inertias must be (3,), (3, 3) or (nQ, 3, 3); got {arr.shape}")

    @property
    def num_quadrotors(self) -> int:
        return self._num_quadrotors

    @property
    def tension_policy(self) -> TensionDistributionPolicy:
        """The null-space choice this map allocates with."""
        return self.allocation.policy

    @classmethod
    def from_model(cls, model, tension_policy: TensionDistributionPolicy | None = None):
        """Build from a
        :class:`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase`-like
        model.

        The default policy is then the ring at the model's own
        ``formation_radius`` and attachment points.  For *ring* attachment
        points that trim is exact, and the flat reference at hover coincides
        with the model's ``reset()`` configuration.  For any other grasp the
        trim is only least-squares — the recovered hover reference is still an
        exact equilibrium of the model, but it is not ``reset()``'s formation,
        and :attr:`~udaan.flatness.RingTensionDistribution.hover_residual`
        reports the gap.
        """
        return cls(
            num_quadrotors=model.num_quadrotors,
            masses=model.masses,
            inertias=model.inertias,
            payload_mass=model.payload_mass,
            payload_inertia=model.payload_inertia,
            cable_lengths=model.cable_lengths,
            attachment_points=model.attachment_points,
            formation_radius=model.formation_radius,
            tension_policy=tension_policy,
        )

    # ─── Wrench chain ──────────────────────────────────────────────

    def payload_wrench_jet(
        self, flats: MultiQuadRigidbodyFlats, rotation: list[np.ndarray] | None = None
    ) -> list[np.ndarray]:
        r"""Body-frame payload wrench jet ``[W, Ẇ, …, W^{(4)}]`` from the flat output.

        The wrench the cables must jointly apply to the load is Newton–Euler on
        the payload, read right-to-left:

        .. math::

            W = \begin{bmatrix} W_f \\ W_m \end{bmatrix}
              = \begin{bmatrix}
                    m_L R_L^\top (\ddot x_L + g e_3) \\
                    J_L \dot\Omega_L + \Omega_L \times J_L \Omega_L
                \end{bmatrix}.

        Both blocks are differentiated exactly: the force block by
        :func:`~udaan.flatness.leibniz_matmul` against the ``R_L^\top`` jet of
        :func:`rotation_jet`, the moment block by the Leibniz rule on the
        gyroscopic cross product.  The deepest terms are ``x_L^{(6)}`` and
        ``Ω_L^{(5)}``.

        Exposed separately because it is the whole payload-side content of the
        map, and testing it against the model's own ``6 × 6`` solve is a check
        that does not go through the cable chain at all.

        Args:
            flats: the flat-output jet.
            rotation: optional ``[R_L, …, R_L^{(4)}]`` from :func:`rotation_jet`.
                :meth:`recover` needs the same tuple for the frame change
                ``𝐓_i = -R_L τ_i`` and passes its own rather than paying for a
                second identical recursion; omit it and one is built here.
        """
        g = self.gravity
        mL = self.payload_mass
        J_L = self.payload_inertia

        x_L = [np.asarray(flats.x_L[k], dtype=float) for k in range(XL_ORDER + 1)]
        Omega = [np.asarray(flats.Omega_L[k], dtype=float) for k in range(OMEGA_ORDER + 1)]
        R = rotation_jet(flats.R_L, Omega, order=_WRENCH_ORDER) if rotation is None else rotation

        # α^(k) = (ẍ_L + g e₃)^(k) = x_L^(k+2) for k ≥ 1; gravity is constant.
        alpha = [x_L[2] + g * _E3]
        alpha.extend(x_L[k] for k in range(3, XL_ORDER + 1))

        force = leibniz_matmul([Rk.T for Rk in R], alpha)
        wrench: list[np.ndarray] = []
        for k in range(_WRENCH_ORDER + 1):
            moment = J_L @ Omega[k + 1]
            for j in range(k + 1):
                moment = moment + comb(k, j) * np.cross(Omega[j], J_L @ Omega[k - j])
            wrench.append(np.concatenate([mL * force[k], moment]))
        return wrench

    # ─── Recovery ──────────────────────────────────────────────────

    def recover(
        self, flats: MultiQuadRigidbodyFlats
    ) -> tuple[MultiQuadRigidbodyRefState, MultiQuadRigidbodyInputs]:
        """Map the flat-output jet to (reference state, feedforward inputs).

        Raises:
            WrenchAllocationError: the payload wrench is outside ``range Φ`` —
                the geometry cannot produce the requested payload motion (see
                "``nQ ≤ 2``" in the module docstring).
            ValueError: a cable would go slack or invert, a quadrotor is in
                free fall, or a thrust direction is aligned with its yaw hint.
        """
        nQ = self._num_quadrotors
        if flats.num_quadrotors != nQ:
            raise ValueError(
                f"flats carry {flats.num_quadrotors} yaw jets but the map has {nQ} quadrotors"
            )
        g = self.gravity

        x_L = [np.asarray(flats.x_L[k], dtype=float) for k in range(XL_ORDER + 1)]
        Omega = [np.asarray(flats.Omega_L[k], dtype=float) for k in range(OMEGA_ORDER + 1)]
        R = rotation_jet(flats.R_L, Omega, order=_WRENCH_ORDER)

        # ── Step 1: the payload wrench and its four derivatives. R is shared:
        # the wrench needs R_Lᵀ, the frame change of step 3 needs R_L.
        wrench = self.payload_wrench_jet(flats, rotation=R)

        # ── Step 2: allocate it — the free (3 nQ − 6)-dimensional choice.
        # Also where tautness and range(Φ) are enforced.
        cable_forces = self.allocation.allocate_jet(wrench)

        ref = MultiQuadRigidbodyRefState(
            payload_position=x_L[0],
            payload_velocity=x_L[1],
            payload_acceleration=x_L[2],
            payload_orientation=SO3(R[0].copy()),
            payload_angular_velocity=TSO3(Omega[0].copy()),
            payload_angular_acceleration=Omega[1].copy(),
        )
        inputs = MultiQuadRigidbodyInputs()

        for i in range(nQ):
            m_i = float(self.masses[i])
            l_i = float(self.cable_lengths[i])
            J_i = self.inertias[i]
            r_i = self.attachment_points[i]
            tau = cable_forces[i]

            # ── Step 3: into the world frame. τ_i is the force on the *load*
            # in the payload body frame; 𝐓_i = -R_L τ_i is the force on the
            # quadrotor, in the world frame, which is what the cable chain and
            # the quadrotor's Newton equation both want.
            force = [-v for v in leibniz_matmul(R, tau)]

            # ── Step 4: tension magnitude and unit-direction jets, the same
            # primitive M2/M3 use — it takes the force, so it does not care
            # that here the tension is a free rather than a determined quantity.
            tension, q = tension_direction_jet(force)

            # ── Step 5: quadrotor positions. The cable hangs off the *moving,
            # rotating* attachment point x_{P_i} = x_L + R_L r_i, which is the
            # one geometric difference from M3; at r_i = 0 this collapses to
            # x_{Q_i} = x_L − l_i q_i.
            xQ = [x_L[k] + R[k] @ r_i - l_i * q[k] for k in range(_WRENCH_ORDER + 1)]

            # ── Step 6: cable rates on T_{q_i}S². ω = q × q̇, ω̇ = q × q̈.
            omega_cable = np.cross(q[0], q[1])
            domega_cable = np.cross(q[0], q[2])

            # ── Step 7: thrust vector. Newton on quadrotor i, whose cable
            # reaction is +𝐓_i:  m_i ẍ_{Q_i} = f_i R_i e₃ − m_i g e₃ + 𝐓_i.
            B = m_i * (xQ[2] + g * _E3) - force[0]
            dB = m_i * xQ[3] - force[1]
            d2B = m_i * xQ[4] - force[2]

            if float(np.linalg.norm(B)) < 1e-6:
                raise ValueError(
                    f"MultiQuadRigidbody.recover: quadrotor {i} has a vanishing "
                    "thrust vector (free fall) — no unique attitude."
                )

            # ── Step 8: R_i, Ω_i, Ω̇_i, M_i from (B_i, ψ_i).
            psi_i = flats.psi[i]
            f, R_i, Omega_i, dOmega_i, moment = _attitude_from_thrust_vector(
                B,
                dB,
                d2B,
                float(psi_i.value),
                float(psi_i.velocity),
                float(psi_i.acceleration),
                J_i,
            )

            ref.positions.append(xQ[0])
            ref.velocities.append(xQ[1])
            ref.accelerations.append(xQ[2])
            ref.cable_attitudes.append(q[0])
            ref.cable_angular_velocities.append(omega_cable)
            ref.cable_angular_accelerations.append(domega_cable)
            ref.orientations.append(SO3(R_i))
            ref.angular_velocities.append(TSO3(Omega_i))
            ref.angular_accelerations.append(dOmega_i)

            inputs.thrusts.append(f)
            inputs.moments.append(moment)
            inputs.tensions.append(float(tension[0]))
            inputs.cable_forces.append(tau[0])

        return ref, inputs
