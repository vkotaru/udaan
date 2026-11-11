"""MultiQuadCsPayloadBase — pure dynamics and state, no rendering.

``nQ`` rigid-body quadrotors cooperatively carrying a single point-mass payload,
each through its own taut, massless cable, integrated with ZOH (zero-order hold)
Euler.

Configuration
-------------
Payload position/velocity ``x_L, v_L ∈ R³``; per cable ``i`` a direction
``q_i ∈ S²`` with angular velocity ``ω_i ∈ T_{q_i}S²`` (so ``q̇_i = ω_i × q_i``);
per quadrotor ``i`` an attitude ``R_i ∈ SO(3)`` with body rate ``Ω_i``.  As
everywhere else in this codebase (see :mod:`udaan.flatness.cable` and
:mod:`udaan.models.quadrotor_cspayload`), ``q_i`` points **from the quadrotor to
the load**, so the quadrotor position is a derived quantity::

    x_{Q_i} = x_L - l_i q_i,      v_{Q_i} = v_L - l_i q̇_i

Equations of motion
-------------------
With ``u_i = f_i R_i e₃`` the thrust vector of quadrotor ``i``, ``u_i^∥ =
q_i q_iᵀ u_i`` and ``u_i^⊥ = (I - q_i q_iᵀ) u_i``::

    (m_L I + Σ_i m_i q_i q_iᵀ)(a_L + g e₃) = Σ_i ( u_i^∥ - m_i l_i ‖q̇_i‖² q_i )

    ω̇_i = (1/l_i) q̂_i (a_L + g e₃) - (1/(m_i l_i)) q̂_i u_i^⊥

The centripetal coefficient is ``‖q̇_i‖² = ‖ω_i × q_i‖²``, which equals
``‖ω_i‖²`` only while ``ω_i`` is exactly tangent to the sphere.

    Ṙ_i = R_i Ω̂_i,    Ω̇_i = J_i⁻¹ ( M_i - Ω_i × J_i Ω_i )

The ``ω̇_i`` right-hand side is a cross product with ``q_i``, hence always
tangent to the sphere, and the mass matrix ``m_L I + Σ m_i q_i q_iᵀ`` is
positive definite for ``m_L > 0``.

Dynamics reference:
Koushil Sreenath and Vijay Kumar. "Dynamics, Control and Planning for
Cooperative Manipulation of Payloads Suspended by Cables from Multiple Quadrotor
Robots." Robotics: Science and Systems IX, 2013.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ...control.multi_quad_cspayload import MultiQuadCsPayloadController
from ...control.quadrotor import GeometricAttitudeController, PositionPDController
from ...core.defaults import (
    DEFAULT_ARM_LENGTH,
    DEFAULT_ATTITUDE_RATE_HZ,
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORCE_CONSTANT,
    DEFAULT_GENERATED_QUAD_INERTIA,
    DEFAULT_GENERATED_QUAD_MASS,
    DEFAULT_MAX_THRUST,
    DEFAULT_MAX_TORQUE,
    DEFAULT_MIN_THRUST,
    DEFAULT_MIN_TORQUE,
    DEFAULT_NUM_QUADROTORS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_PHYSICS_RATE_HZ,
    DEFAULT_PROP_MAX_FORCE,
    DEFAULT_PROP_MIN_FORCE,
    DEFAULT_TORQUE_CONSTANT,
    GRAVITY,
)
from ...core.types import ForceType, InputType, Vec3
from ...core.validation import MIN_PHYSICAL_MASS, check_positive
from ...flatness import (
    Jet,
    MultiQuadCsPayload,
    MultiQuadCsPayloadFlats,
    TensionSplitContractError,
    yaw_jet,
)
from ...flatness.feasibility import format_intervals, infeasible_intervals
from ...manif import S2, SO3, TS2, TSO3, hat, zoh_step
from ...utils.formation import check_formation, default_formation_radius
from ...utils.grasp import formation_cable_attitudes
from ...utils.logging import get_logger
from .._kwargs import check_reset_kwargs
from .._rates import attitude_updates_per_step, physics_steps_per_step
from ..quadrotor.base import _ThrustCommand

_logger = get_logger(__name__)

__all__ = [
    "MultiQuadCsPayloadBase",
    "MultiQuadCsPayloadState",
    "cable_state_from_geometry",
    "formation_cable_attitudes",
]


def cable_state_from_geometry(
    payload_position: Vec3,
    payload_velocity: Vec3,
    quad_position: Vec3,
    quad_velocity: Vec3,
) -> tuple[S2, TS2, float]:
    """Recover a cable's ``(q, ω, length)`` from measured endpoint states.

    With ``p = x_L - x_Q`` and ``q = p/‖p‖``, differentiating gives
    ``q̇ = (I - q qᵀ) ṗ / ‖p‖``, hence::

        ω = q × q̇ = q × (v_L - v_Q) / ‖p‖

    The ``1/‖p‖`` factor is **not** optional — dropping it scales ω by the
    cable length and silently corrupts every downstream cable-rate term.

    Raises:
        ValueError: if the endpoints coincide, leaving ``q`` undefined.
    """
    p = np.asarray(payload_position, dtype=float) - np.asarray(quad_position, dtype=float)
    length = float(np.linalg.norm(p))
    if length < 1e-9:
        raise ValueError("cable endpoints coincide; direction is undefined")
    q = p / length
    rel_v = np.asarray(payload_velocity, dtype=float) - np.asarray(quad_velocity, dtype=float)
    return S2(q), TS2(np.cross(q, rel_v) / length), length


@dataclass
class MultiQuadCsPayloadState:
    """State of ``num_quadrotors`` quadrotors sharing one cable-suspended payload.

    Every per-quadrotor entry is a list of length ``num_quadrotors``, indexed by
    quadrotor.  ``positions``/``velocities`` are *derived* from the payload and
    cable states in the analytical model (the cables are inextensible); the
    MuJoCo backend overwrites them with the measured body states.

    ``cable_tensions`` is the ``(nQ,)`` tension each cable carried on the last
    physics substep, in N: on the analytical model the rigid cable's force
    (negative while it pushes), on MuJoCo the tendon's limit force (zero while
    it is slack). It is ``nan`` until the first substep after a reset.
    """

    num_quadrotors: int = DEFAULT_NUM_QUADROTORS

    # Payload position and velocity
    payload_position: Vec3 = field(default_factory=lambda: np.zeros(3))
    payload_velocity: Vec3 = field(default_factory=lambda: np.zeros(3))

    # Per-quadrotor position and velocity (derived from payload + cable)
    positions: list[Vec3] = field(default_factory=list)
    velocities: list[Vec3] = field(default_factory=list)

    # Per-quadrotor orientation and body angular velocity
    orientations: list[SO3] = field(default_factory=list)
    angular_velocities: list[TSO3] = field(default_factory=list)

    # Per-cable attitude and angular velocity
    cable_attitudes: list[S2] = field(default_factory=list)
    cable_angular_velocities: list[TS2] = field(default_factory=list)

    # Per-cable tension measured on the last physics substep, in N (``nan``
    # before the first substep after a reset)
    cable_tensions: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self):
        if self.num_quadrotors < 1:
            raise ValueError(f"num_quadrotors must be >= 1, got {self.num_quadrotors}")
        if not self.positions:
            self.reset()

    def reset(self):
        """Reset payload, quadrotor *and* cable entries to their defaults."""
        nQ = self.num_quadrotors
        self.payload_position = np.zeros(3)
        self.payload_velocity = np.zeros(3)
        self.positions = [np.zeros(3) for _ in range(nQ)]
        self.velocities = [np.zeros(3) for _ in range(nQ)]
        self.orientations = [SO3() for _ in range(nQ)]
        self.angular_velocities = [TSO3() for _ in range(nQ)]
        self.cable_attitudes = [S2(np.array([0.0, 0.0, -1.0])) for _ in range(nQ)]
        self.cable_angular_velocities = [TS2() for _ in range(nQ)]
        self.cable_tensions = np.full(nQ, np.nan)

    def dq(self, i: int) -> np.ndarray:
        """Cable direction rate ``q̇_i = ω_i × q_i`` of cable ``i``."""
        return np.cross(
            np.asarray(self.cable_angular_velocities[i]),
            np.asarray(self.cable_attitudes[i]),
        )


#: The initial-condition overrides :meth:`MultiQuadCsPayloadBase.reset` acts on.
#: Anything else is rejected rather than silently discarded — see
#: :mod:`udaan.models._kwargs`.
_SUPPORTED_RESET_KWARGS = frozenset(
    {
        "payload_position",
        "payload_velocity",
        "orientations",
        "angular_velocities",
        "cable_attitudes",
        "cable_angular_velocities",
    }
)


class MultiQuadCsPayloadBase:
    """Base multi-quadrotor cable-suspended payload model — pure dynamics.

    The quadrotors are rigid bodies: the per-quadrotor input is a thrust
    magnitude and a body moment, not a free force vector.  Masses, inertias and
    cable lengths are per-index and accept a scalar (broadcast to every
    quadrotor) or a length-``num_quadrotors`` sequence.

    Usage::

        mdl = MultiQuadCsPayloadBase(num_quadrotors=4)
        mdl.cable_lengths = [1.0, 1.2, 1.0, 1.2]
        mdl.simulate(tf=5.0, payload_position=np.array([0.0, 0.0, 1.0]))

    The airframe defaults to the one the MJCF generator emits
    (:data:`~udaan.core.defaults.DEFAULT_GENERATED_QUAD_MASS` and
    :data:`~udaan.core.defaults.DEFAULT_GENERATED_QUAD_INERTIA`), so the
    analytical model and :class:`MultiQuadCsPayloadMujoco` fly the same machine.

    The ring formation is checked by
    :func:`udaan.utils.formation.check_formation` at construction and whenever
    ``formation_radius`` or ``cable_lengths`` is assigned: neighbouring
    airframes must be at least the rotor span plus a clearance apart, and the
    radius must stay below every cable length. ``formation_radius`` defaults to
    :func:`udaan.utils.formation.default_formation_radius`, which grows with
    ``num_quadrotors``.

    Three loops, at three rates, as in
    :class:`~udaan.models.quadrotor.QuadrotorBase`: the payload cascade runs
    once per :meth:`step` (``1/dt``, 200 Hz); each airframe's attitude loop runs
    ``attitude_rate_hz · dt`` times inside it (800 Hz by default); the physics
    integrates ``N = physics_rate_hz · dt`` substeps of ``dt/N`` (1600 Hz,
    ``N = 8``, by default). The clock counts substeps.

    Keyword Args:
        dt: the payload-loop period, in s (default 5 ms).
        physics_rate_hz: the physics rate, in Hz (default
            :data:`~udaan.core.defaults.DEFAULT_PHYSICS_RATE_HZ`); ``rate · dt``
            must be whole.
        attitude_rate_hz: the attitude-loop rate, in Hz (default
            :data:`~udaan.core.defaults.DEFAULT_ATTITUDE_RATE_HZ`); its updates
            per step must divide ``N``. ``None`` is once per step.
    """

    INPUT_TYPE = InputType
    FORCE_TYPE = ForceType

    def __init__(self, **kwargs):
        self._num_quadrotors = int(kwargs.get("num_quadrotors", DEFAULT_NUM_QUADROTORS))
        if self._num_quadrotors < 1:
            raise ValueError(f"num_quadrotors must be >= 1, got {self._num_quadrotors}")
        nQ = self._num_quadrotors

        # The parameter setters push their new values into the controllers, and
        # the constructor runs several of them *before* those controllers exist.
        self._controllers_ready = False

        self.state = MultiQuadCsPayloadState(num_quadrotors=nQ)
        self.t = 0.0
        self.verbose = kwargs.get("verbose", 0)

        # System parameters — per quadrotor, scalar-broadcastable
        self._masses = np.full(nQ, DEFAULT_GENERATED_QUAD_MASS)
        self._inertias = np.tile(DEFAULT_GENERATED_QUAD_INERTIA, (nQ, 1, 1))
        self._inertia_invs = np.linalg.inv(self._inertias)
        self._cable_lengths = np.full(nQ, DEFAULT_CABLE_LENGTH)
        self._payload_mass = DEFAULT_PAYLOAD_MASS
        # Routed through the property so the constructor kwarg and later
        # assignment share one validation.
        radius = kwargs.get("formation_radius")
        self.formation_radius = default_formation_radius(nQ) if radius is None else radius

        if "masses" in kwargs:
            self.masses = kwargs["masses"]
        if "inertias" in kwargs:
            self.inertias = kwargs["inertias"]
        if "cable_lengths" in kwargs:
            self.cable_lengths = kwargs["cable_lengths"]
        if "payload_mass" in kwargs:
            self.payload_mass = kwargs["payload_mass"]
        check_formation(nQ, self._formation_radius, self._cable_lengths)

        # Physical constants
        self._g = GRAVITY
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._e3 = np.array([0.0, 0.0, 1.0])

        # Actuator limits (per quadrotor)
        self._min_thrust = DEFAULT_MIN_THRUST
        self._max_thrust = DEFAULT_MAX_THRUST
        self._min_torque = DEFAULT_MIN_TORQUE.copy()
        self._max_torque = DEFAULT_MAX_TORQUE.copy()
        self._prop_min_force = DEFAULT_PROP_MIN_FORCE
        self._prop_max_force = DEFAULT_PROP_MAX_FORCE

        # Integration: a 5 ms outer loop of N substeps of h = dt/N
        # (N = physics_rate_hz · dt; 8 substeps of 0.625 ms at the default 1600 Hz).
        if "inner_loop_steps" in kwargs:
            raise TypeError(
                "inner_loop_steps is not a keyword; the substeps are set by physics_rate_hz: "
                "N = physics_rate_hz · dt substeps per step (800 Hz is 4 substeps at a 5 ms dt)"
            )
        self.dt = kwargs.get("dt", 1.0 / 200.0)
        self._physics_rate_hz = kwargs.get("physics_rate_hz", DEFAULT_PHYSICS_RATE_HZ)
        self._inner_loop_steps = physics_steps_per_step(self.dt, self._physics_rate_hz)
        self.h = self.dt / self._inner_loop_steps
        self._attitude_rate_hz = kwargs.get("attitude_rate_hz", DEFAULT_ATTITUDE_RATE_HZ)
        self._attitude_updates = attitude_updates_per_step(
            self.dt,
            self._attitude_rate_hz,
            self._inner_loop_steps,
            is_default="attitude_rate_hz" not in kwargs,
        )
        self._substeps_taken = 0

        # Input/force type
        self._input_type = InputType.ACCELERATION
        self._force_type = ForceType.WRENCH
        if "input" in kwargs:
            if kwargs["input"] == "prop_forces":
                self._input_type = InputType.PROP_FORCES
            elif kwargs["input"] == "wrench":
                self._input_type = InputType.WRENCH
        if "force" in kwargs and kwargs["force"] == "prop_forces":
            self._force_type = ForceType.PROP_FORCES

        # Allocation matrix
        self._compute_allocation_matrix()

        # Per-quadrotor station-keeping targets, refreshed by reset()
        self._formation_targets = [np.zeros(3) for _ in range(nQ)]

        # Steps on which the flat map refused the reference or recovered an
        # airframe that is not upright, and the loop fell back to
        # feedback-only; zeroed at the start of every ``simulate()``.
        self._feedforward_drops = 0

        # Substeps on which the analytical cable pushed (negative tension);
        # zeroed by ``reset()``.
        self._compressive_cable_steps = 0
        self._cable_tensions = np.zeros(nQ)

        # Feedforward yaw reference shared by every quadrotor; None is ψ ≡ 0.
        self._yaw_setpoint = None

        # Default controllers
        self._init_default_controllers()

    # ─── Controllers ───────────────────────────────────────────────

    def _init_default_controllers(self):
        nQ = self._num_quadrotors
        # The real cascade: payload PD -> tension split -> per-cable S² control
        # -> per-quadrotor thrust vector. Its time-scale design also fixes the
        # attitude gains below.
        self._payload_controller = MultiQuadCsPayloadController.from_model(self)
        design = self._payload_controller.design_gains
        self._att_controllers = [
            GeometricAttitudeController(
                inertia=self._inertias[i],
                kp=design.attitude_kp[i],
                kd=design.attitude_kd[i],
            )
            for i in range(nQ)
        ]
        self._default_att_controllers = list(self._att_controllers)
        # Each quadrotor carries its own weight plus an equal share of the
        # payload; that share is the feedforward of its position controller.
        # These are *station-keeping only* — they hold whatever formation
        # ``reset()`` left behind and know nothing about the payload. They are
        # kept because they are a useful open-loop-ish baseline and because
        # ``compute_formation_input()`` exposes them, but ``simulate()`` runs
        # the payload cascade below.
        self._pos_controllers = [
            PositionPDController(mass=self._masses[i] + self._payload_mass / nQ) for i in range(nQ)
        ]
        for i in range(nQ):
            self._pos_controllers[i].setpoint = self._formation_setpoint(i)
        self._controllers_ready = True

    def _sync_default_controllers(self):
        """Push the current physical parameters into the default controllers.

        Every controller in the cascade caches parameters at construction — the
        attitude loops hold ``J_i``, the station-keeping PDs hold their share of
        the system mass, and the payload cascade holds every mass *and* its
        tension-split policy. A setter that writes only ``self._…`` therefore
        leaves the closed loop driving a different machine than the one being
        integrated: assigning ``payload_mass`` after construction without a
        sync would leave the cascade splitting the old load's weight, and the
        payload would settle below its setpoint.

        A controller the caller has substituted for one of ours is left alone
        unless it opts in with a ``sync_from_model`` hook.
        """
        if not self._controllers_ready:
            return
        for i, att_controller in enumerate(self._att_controllers):
            if hasattr(att_controller, "inertia"):
                att_controller.inertia = self._inertias[i]
        for i, pos_controller in enumerate(self._pos_controllers):
            if hasattr(pos_controller, "mass"):
                pos_controller.mass = self._masses[i] + self._payload_mass / self._num_quadrotors
        sync = getattr(self._payload_controller, "sync_from_model", None)
        if sync is not None:
            sync(self)
        # The attitude gains follow the time-scale design, on the attitude
        # controllers this model built.
        design = getattr(self._payload_controller, "design_gains", None)
        if design is not None:
            for i, att_controller in enumerate(self._att_controllers):
                if att_controller is self._default_att_controllers[i]:
                    att_controller._gains.kp = np.array(design.attitude_kp[i])
                    att_controller._gains.kd = np.array(design.attitude_kd[i])

    def _formation_setpoint(self, i: int):
        """Station-keeping setpoint callable for quadrotor ``i``."""
        return lambda t: (self._formation_targets[i], np.zeros(3), np.zeros(3))

    @property
    def position_controllers(self) -> list:
        """Per-quadrotor station-keeping position controllers.

        Not what :meth:`simulate` drives — see :attr:`payload_controller`.
        """
        return self._pos_controllers

    @position_controllers.setter
    def position_controllers(self, controllers: list):
        self._check_length(controllers, "position_controllers")
        self._pos_controllers = list(controllers)

    @property
    def payload_controller(self):
        """The payload-tracking cascade driven by :meth:`simulate`."""
        return self._payload_controller

    @payload_controller.setter
    def payload_controller(self, controller):
        self._payload_controller = controller

    @property
    def yaw_setpoint(self):
        r"""Yaw reference for the flatness feedforward: ``ψ(t) -> (ψ, ψ̇, ψ̈)``.

        One callable, shared by every quadrotor — a formation flies one
        heading. Yaw is the only channel the recovered ``R_i`` adds on top of
        the position loop (see
        :meth:`~udaan.control.quadrotor.GeometricAttitudeController.compute`),
        so without this the feedforward runs at ψ ≡ 0 and the attitude
        composition is close to a no-op.

        ``None`` (the default) means ψ ≡ 0, so binding nothing changes nothing.
        Same protocol and same default as
        :attr:`udaan.models.quadrotor.QuadrotorBase.yaw_setpoint`.
        """
        return self._yaw_setpoint

    @yaw_setpoint.setter
    def yaw_setpoint(self, setpoint):
        if setpoint is not None and not callable(setpoint):
            raise TypeError("yaw_setpoint must be callable psi(t) -> (psi, dpsi, ddpsi), or None")
        self._yaw_setpoint = setpoint

    @property
    def attitude_controllers(self) -> list:
        """Per-quadrotor attitude controllers."""
        return self._att_controllers

    @attitude_controllers.setter
    def attitude_controllers(self, controllers: list):
        self._check_length(controllers, "attitude_controllers")
        self._att_controllers = list(controllers)

    # ─── Properties ────────────────────────────────────────────────

    def _check_length(self, value, name: str):
        if len(value) != self._num_quadrotors:
            raise ValueError(f"{name} must have {self._num_quadrotors} entries, got {len(value)}")

    def _broadcast(self, value, name: str) -> np.ndarray:
        """Scalar -> (nQ,) array, or validate a per-index sequence."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self._num_quadrotors, float(arr))
        self._check_length(arr, name)
        return arr.astype(float)

    @property
    def num_quadrotors(self) -> int:
        return self._num_quadrotors

    @property
    def masses(self) -> np.ndarray:
        """Per-quadrotor masses, shape ``(nQ,)``."""
        return self._masses

    @masses.setter
    def masses(self, value):
        masses = self._broadcast(value, "masses")
        check_positive(masses, "masses", warn_below=MIN_PHYSICAL_MASS)
        self._masses = masses
        self._sync_default_controllers()

    @property
    def inertias(self) -> np.ndarray:
        """Per-quadrotor inertia tensors, shape ``(nQ, 3, 3)``.

        Accepts a single ``(3,)`` diagonal or ``(3, 3)`` tensor, broadcast to
        every quadrotor, or a per-index ``(nQ, 3, 3)`` stack. A ``(nQ, 3)``
        stack of diagonals is deliberately *not* accepted: at ``nQ == 3`` it
        has the same shape as a single tensor. Every tensor is required to be
        symmetric positive definite, which is both physically necessary and
        what disambiguates that ``(3, 3)`` case — a stack of distinct
        diagonals is never symmetric. Build the stack yourself with
        :func:`numpy.diag`.
        """
        return self._inertias

    @inertias.setter
    def inertias(self, value):
        arr = np.asarray(value, dtype=float)
        if arr.shape == (3,):
            arr = np.tile(np.diag(arr), (self._num_quadrotors, 1, 1))
        elif arr.shape == (3, 3):
            arr = np.tile(arr, (self._num_quadrotors, 1, 1))
        elif arr.shape != (self._num_quadrotors, 3, 3):
            raise ValueError(f"inertias must be (3,), (3, 3) or (nQ, 3, 3); got {arr.shape}")
        for i in range(self._num_quadrotors):
            self._check_inertia_tensor(arr[i], f"inertias[{i}]")
        self._inertias = arr
        self._inertia_invs = np.linalg.inv(arr)
        self._sync_default_controllers()

    @staticmethod
    def _check_inertia_tensor(tensor: np.ndarray, name: str):
        """Reject anything that is not a symmetric positive-definite tensor.

        Symmetry is not only physics: it is what lets a ``(3, 3)`` argument be
        read unambiguously as *one* tensor at ``num_quadrotors == 3``, where an
        ``(nQ, 3)`` stack of per-quadrotor diagonals would otherwise be
        silently tiled to every quadrotor as a single asymmetric tensor.
        """
        scale = max(1.0, float(np.max(np.abs(tensor))))
        if not np.allclose(tensor, tensor.T, rtol=0.0, atol=1e-12 * scale):
            raise ValueError(
                f"{name} must be a symmetric tensor; got {tensor.tolist()}. "
                "An (nQ, 3) stack of diagonals is not accepted — build the "
                "(nQ, 3, 3) stack with numpy.diag."
            )
        if not np.all(np.linalg.eigvalsh(tensor) > 0.0):
            raise ValueError(f"{name} must be positive definite; got {tensor.tolist()}")

    @property
    def inertia_invs(self) -> np.ndarray:
        return self._inertia_invs

    @property
    def cable_lengths(self) -> np.ndarray:
        """Per-cable lengths, shape ``(nQ,)``."""
        return self._cable_lengths

    @cable_lengths.setter
    def cable_lengths(self, value):
        lengths = self._broadcast(value, "cable_lengths")
        check_positive(lengths, "cable_lengths")
        if self._controllers_ready:
            check_formation(self._num_quadrotors, self._formation_radius, lengths)
        self._cable_lengths = lengths
        self._sync_default_controllers()

    @property
    def formation_radius(self) -> float:
        """Radius of the default ring formation used by :meth:`reset`.

        Must be strictly positive, matching the domain
        :class:`~udaan.flatness.RingTensionSplit` accepts for the same ring —
        the two re-derivations have to agree on where the formation is defined.
        A non-positive radius passes the ``lengths <= radius`` guard in
        :func:`formation_cable_attitudes` and would surface only as a ``nan``
        cable direction out of :meth:`reset`.
        """
        return self._formation_radius

    @formation_radius.setter
    def formation_radius(self, value: float):
        radius = float(value)
        if radius <= 0.0:
            raise ValueError(
                "formation_radius must be strictly positive: at radius 0 every "
                "cable is vertical and the quadrotors stack on top of each "
                f"other, and a negative radius has no geometric meaning; got {radius}"
            )
        if self._controllers_ready:
            check_formation(self._num_quadrotors, radius, self._cable_lengths)
        self._formation_radius = radius
        self._sync_default_controllers()

    @property
    def payload_mass(self) -> float:
        r"""Payload mass ``m_L``. Must be strictly positive.

        ``m_L > 0`` is exactly the condition under which the payload mass
        matrix ``m_L I + Σ_i m_i q_i q_iᵀ`` of the module docstring is positive
        definite. At ``m_L = 0`` the rank-one sum is all that is left, and with
        parallel cables (the ``nQ = 1`` case, and the ring at radius → 0) it is
        singular — ``np.linalg.solve`` then raises out of the middle of
        :meth:`_zoh`, several frames from the assignment that caused it. At
        ``m_L < 0`` it is indefinite and the solve happily returns an
        acceleration that describes nothing.
        """
        return self._payload_mass

    @payload_mass.setter
    def payload_mass(self, value: float):
        check_positive(value, "payload_mass", warn_below=MIN_PHYSICAL_MASS)
        self._payload_mass = float(value)
        self._sync_default_controllers()

    @property
    def physics_rate_hz(self) -> float:
        """Rate at which the physics integrates, in Hz."""
        return float(self._physics_rate_hz)

    @property
    def attitude_rate_hz(self) -> float:
        """Rate at which each airframe's attitude loop closes, in Hz."""
        return float(self._attitude_updates / self.dt)

    @property
    def physics_steps(self) -> int:
        """Physics substeps per :meth:`step`, ``N = physics_rate_hz · dt``."""
        return self._inner_loop_steps

    @property
    def compressive_cable_steps(self) -> int:
        """Substeps since :meth:`reset` on which some analytical cable pushed.

        The analytical cable is a rigid rod: its tension (:meth:`cable_tensions`)
        has no sign condition, so a cable can push the payload where a real one
        would go slack. Each substep with a negative tension is counted, and the
        first one after a reset logs a warning; MuJoCo's tendons go slack instead.
        """
        return self._compressive_cable_steps

    def cable_tensions(self) -> np.ndarray:
        """``(nQ,)`` cable tensions of the last substep, in N (negative: pushing).

        From quadrotor ``i``'s motion along its cable,
        ``T_i = m_i (a_L + g e₃)·q_i + m_i l_i ‖q̇_i‖² − u_i·q_i``.
        """
        return np.array(self._cable_tensions, dtype=float)

    # ─── Dynamics ──────────────────────────────────────────────────

    def _zoh(self, thrusts: np.ndarray, torques: np.ndarray, h: float):
        """Zero-order hold Euler integration for one substep.

        Args:
            thrusts: ``(nQ,)`` thrust magnitudes ``f_i``.
            torques: ``(nQ, 3)`` body moments ``M_i``.
            h: substep size.
        """
        nQ = self._num_quadrotors
        q = [np.asarray(self.state.cable_attitudes[i]).copy() for i in range(nQ)]
        omega = [np.asarray(self.state.cable_angular_velocities[i]).copy() for i in range(nQ)]

        # Payload equation: (m_L I + Σ m_i q_i q_iᵀ)(a_L + g e3)
        #                     = Σ (q_i q_iᵀ u_i - m_i l_i ‖q̇_i‖² q_i)
        #
        # The centripetal coefficient is ‖q̇_i‖², not ‖ω_i‖²: the exact identity
        # is q_i · q̈_i = -‖q̇_i‖², and ‖q̇_i‖² = ‖ω_i‖² - (q_i·ω_i)² coincides
        # with ‖ω_i‖² only while ω_i is exactly tangent.  The integrator
        # keeps it exactly tangent (`manif.zoh_step`, see
        # test_cable_rates_stay_tangent), so the two agree *along a run* — but
        # they are still not interchangeable, because nothing stops a caller
        # handing `reset()` a rate with a component along q_i, and only the
        # ‖q̇_i‖² form is right then.  This matches the single-quadrotor model
        # ``udaan/models/quadrotor_cspayload/base.py``.
        mass_matrix = self._payload_mass * np.eye(3)
        rhs = np.zeros(3)
        u_perp: list[np.ndarray] = []
        for i in range(nQ):
            qqT = np.outer(q[i], q[i])
            mass_matrix += self._masses[i] * qqT
            u_i = thrusts[i] * (self.state.orientations[i] @ self._e3)
            u_perp.append(u_i - qqT @ u_i)
            dq_i = self.state.dq(i)
            rhs += qqT @ u_i - self._masses[i] * self._cable_lengths[i] * np.dot(dq_i, dq_i) * q[i]
        payload_accel = np.linalg.solve(mass_matrix, rhs) - self._ge3

        # payload position dynamics
        self.state.payload_position = (
            self.state.payload_position
            + self.state.payload_velocity * h
            + 0.5 * payload_accel * h**2
        )
        self.state.payload_velocity = self.state.payload_velocity + payload_accel * h

        accel_plus_g = payload_accel + self._ge3
        # Cable tensions, from each quadrotor's motion along its cable:
        # m_i (a_L - l_i q̈_i + g e₃)·q_i = u_i·q_i + T_i with q_i·q̈_i = -‖q̇_i‖².
        for i in range(nQ):
            dq_i = self.state.dq(i)
            u_i = thrusts[i] * (self.state.orientations[i] @ self._e3)
            self._cable_tensions[i] = (
                self._masses[i] * float(accel_plus_g @ q[i])
                + self._masses[i] * self._cable_lengths[i] * float(dq_i @ dq_i)
                - float(u_i @ q[i])
            )
        self.state.cable_tensions = self._cable_tensions.copy()
        if np.any(self._cable_tensions < 0.0):
            self._compressive_cable_steps += 1
            if self._compressive_cable_steps == 1:
                _logger.warning(
                    "analytical cable pushed at t=%.4f (tensions %s N): the rigid-cable "
                    "model has no slack, so this is a state a real cable cannot reach; "
                    "see compressive_cable_steps",
                    self.t,
                    np.array2string(self._cable_tensions, precision=3),
                )
        for i in range(nQ):
            l_i = self._cable_lengths[i]
            q_hat = hat(q[i])

            # cable attitude dynamics
            domega = (q_hat @ accel_plus_g) / l_i - (q_hat @ u_perp[i]) / (self._masses[i] * l_i)
            (
                self.state.cable_attitudes[i],
                self.state.cable_angular_velocities[i],
            ) = zoh_step(
                self.state.cable_attitudes[i],
                self.state.cable_angular_velocities[i],
                domega,
                h,
            )

            # quadrotor attitude dynamics
            Omega = np.asarray(self.state.angular_velocities[i]).copy()
            self.state.orientations[i] = self.state.orientations[i].step(Omega * h)
            dOmega = self._inertia_invs[i] @ (
                torques[i] - np.cross(Omega, self._inertias[i] @ Omega)
            )
            self.state.angular_velocities[i] += TSO3(dOmega * h)

            # quadrotor position & velocity derived from payload + cable
            self.state.positions[i] = self.state.payload_position - l_i * np.asarray(
                self.state.cable_attitudes[i]
            )
            self.state.velocities[i] = self.state.payload_velocity - l_i * self.state.dq(i)

    # ─── Allocation ────────────────────────────────────────────────

    def _compute_allocation_matrix(self):
        kf = DEFAULT_FORCE_CONSTANT
        km = DEFAULT_TORQUE_CONSTANT
        c = km / kf
        arm = DEFAULT_ARM_LENGTH
        ang = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
        d = [-1.0, 1.0, -1.0, 1.0]

        self._alloc = np.zeros((4, 4))
        for i in range(4):
            self._alloc[0, i] = 1.0
            self._alloc[1, i] = arm * np.sin(ang[i])
            self._alloc[2, i] = -arm * np.cos(ang[i])
            self._alloc[3, i] = c * d[i]
        self._alloc_inv = np.linalg.pinv(self._alloc)

    def _wrench_to_propforces(self, wrench: np.ndarray) -> np.ndarray:
        """``(nQ, 4)`` wrenches -> ``(nQ, 4)`` propeller forces."""
        return np.asarray(wrench @ self._alloc_inv.T)

    def _propforces_to_wrench(self, prop_forces: np.ndarray) -> np.ndarray:
        """``(nQ, 4)`` propeller forces -> ``(nQ, 4)`` wrenches."""
        return np.asarray(prop_forces @ self._alloc.T)

    # ─── Input repackaging ─────────────────────────────────────────

    def _attitude_input(self, thrust_forces, desired_att=None) -> np.ndarray:
        """Close every attitude loop on ``thrust_forces``, in this input path's representation.

        ``(nQ, 4)`` wrenches ``[f, Mx, My, Mz]`` on the WRENCH path, rotor forces
        on the PROP_FORCES path.
        """
        nQ = self._num_quadrotors
        forces = np.asarray(thrust_forces, dtype=float).reshape(nQ, 3)
        out = np.zeros((nQ, 4))
        for i in range(nQ):
            f, M = self._att_controllers[i].compute(
                self.t,
                (self.state.orientations[i], self.state.angular_velocities[i]),
                forces[i],
                desired_att=None if desired_att is None else desired_att[i],
            )
            out[i] = np.array([f, *M])
        if self._input_type == InputType.PROP_FORCES:
            out = self._wrench_to_propforces(out)
        return out

    def _repackage_input(self, u, desired_att=None) -> np.ndarray:
        """Convert user input to a ``(nQ, 4)`` stack of ``[thrust, Mx, My, Mz]``.

        Args:
            u: ``(nQ, 3)`` thrust force vectors (``ACCELERATION``), or
                ``(nQ, 4)`` wrenches / propeller forces. Flat ``(3*nQ,)`` and
                ``(4*nQ,)`` inputs are reshaped.
            desired_att: optional per-quadrotor ``(Rd, Omegad, dOmegad)``
                feedforward, one entry per quadrotor.
        """
        nQ = self._num_quadrotors
        if isinstance(u, _ThrustCommand):
            # simulate()'s WRENCH / PROP_FORCES input: close the attitude loops
            # now, then carry on as that path's wrenches or rotor forces.
            u = self._attitude_input(u.force, desired_att=desired_att)
        arr = np.asarray(u, dtype=float)

        if self._input_type == InputType.ACCELERATION:
            arr = arr.reshape(nQ, 3)
            wrench = np.zeros((nQ, 4))
            for i in range(nQ):
                f, M = self._att_controllers[i].compute(
                    self.t,
                    (self.state.orientations[i], self.state.angular_velocities[i]),
                    arr[i],
                    desired_att=None if desired_att is None else desired_att[i],
                )
                wrench[i] = np.array([f, *M])
        elif self._input_type == InputType.PROP_FORCES:
            wrench = self._propforces_to_wrench(arr.reshape(nQ, 4))
        else:
            wrench = arr.reshape(nQ, 4).copy()

        # Clip to actuator limits
        wrench[:, 0] = np.clip(wrench[:, 0], self._min_thrust, self._max_thrust)
        wrench[:, 1:] = np.clip(wrench[:, 1:], self._min_torque, self._max_torque)

        # Convert to prop forces if needed
        if self._force_type == ForceType.PROP_FORCES:
            props = self._wrench_to_propforces(wrench)
            props = np.clip(props, self._prop_min_force, self._prop_max_force)
            wrench = self._propforces_to_wrench(props)

        return wrench

    # ─── Step / Reset / Simulate ───────────────────────────────────

    def step(self, u, desired_att=None):
        """One outer step of ``dt``.

        The input is repackaged — which, on the ACCELERATION path and for
        :meth:`simulate`'s WRENCH and PROP_FORCES input, closes every attitude
        loop — :attr:`attitude_rate_hz` ``· dt`` times, each wrench held over an
        equal share of the ``N`` substeps. ``u`` and ``desired_att`` are held
        for the whole step.
        """
        hold = self._inner_loop_steps // self._attitude_updates
        for _ in range(self._attitude_updates):
            wrench = self._repackage_input(u, desired_att=desired_att)
            thrusts = wrench[:, 0]
            torques = wrench[:, 1:]
            for _ in range(hold):
                self._zoh(thrusts, torques, self.h)
            self._advance_clock(hold)

    def _advance_clock(self, substeps: int) -> None:
        """Advance :attr:`t` by ``substeps`` physics substeps, without accumulating round-off.

        The clock counts whole substeps since :meth:`reset` and divides by the
        physics rate, so ``t`` after ``k`` steps is the closest float to
        ``k · dt``.
        """
        self._substeps_taken += substeps
        self.t = self._substeps_taken / self._physics_rate_hz

    def reset(self, **kwargs):
        """Reset time and state, then apply any state overrides.

        Recognised kwargs: ``payload_position``, ``payload_velocity``,
        ``orientations``, ``angular_velocities``, ``cable_attitudes``,
        ``cable_angular_velocities``.  Per-quadrotor entries must be sequences
        of length ``num_quadrotors``.  When ``cable_attitudes`` is omitted the
        cables are placed in the default ring formation.

        Raises:
            ValueError: for any other keyword.  An unrecognised keyword is an
                error rather than a no-op so that a stale name cannot silently
                run a different initial condition.
        """
        check_reset_kwargs(self, kwargs, _SUPPORTED_RESET_KWARGS)
        self.t = 0.0
        self._substeps_taken = 0
        self._compressive_cable_steps = 0
        self._cable_tensions = np.zeros(self._num_quadrotors)
        self.state.reset()
        self.state.cable_attitudes = formation_cable_attitudes(
            self._num_quadrotors, self._cable_lengths, self._formation_radius
        )

        for key in ("payload_position", "payload_velocity"):
            if key in kwargs:
                # Copy: the integrator rebinds these, but a caller-supplied
                # array must never be aliased into the state.
                setattr(self.state, key, np.array(kwargs[key], dtype=float))

        for key, wrap in (
            ("orientations", SO3),
            ("angular_velocities", TSO3),
            ("cable_attitudes", S2),
            ("cable_angular_velocities", TS2),
        ):
            if key in kwargs:
                values = kwargs[key]
                self._check_length(values, key)
                # Rewrap every entry: the integrator mutates the tangent
                # wrappers in place, so a caller's object must not be aliased.
                setattr(self.state, key, [wrap(np.asarray(v)) for v in values])

        self._sync_quad_states()
        self._formation_targets = [p.copy() for p in self.state.positions]

    def _sync_quad_states(self):
        """Recompute quadrotor positions/velocities from payload + cable states."""
        for i in range(self._num_quadrotors):
            l_i = self._cable_lengths[i]
            self.state.positions[i] = self.state.payload_position - l_i * np.asarray(
                self.state.cable_attitudes[i]
            )
            self.state.velocities[i] = self.state.payload_velocity - l_i * self.state.dq(i)

    def compute_default_input(self) -> np.ndarray:
        """Per-quadrotor thrust force vectors from the payload cascade.

        Returns a ``(nQ, 3)`` array suitable for :meth:`step` under the
        ``ACCELERATION`` input type.
        """
        return np.asarray(self._payload_controller.compute(self.t, self.state))

    def compute_formation_input(self) -> np.ndarray:
        """Per-quadrotor thrust force vectors from the station-keeping PDs.

        The pre-payload-controller baseline: every quadrotor holds the position
        :meth:`reset` left it at, and the payload goes wherever the cables take
        it. Kept as a comparison point; :meth:`simulate` uses
        :meth:`compute_default_input`.
        """
        return np.array(
            [
                self._pos_controllers[i].compute(
                    self.t, (self.state.positions[i], self.state.velocities[i])
                )
                for i in range(self._num_quadrotors)
            ]
        )

    # ─── Flatness feedforward ──────────────────────────────────────

    def _flatness_feedforward_fn(self):
        """The bound ``get_full`` of the payload trajectory, or ``None``.

        Mirrors :meth:`udaan.models.quadrotor.QuadrotorBase.simulate`: the
        payload controller's ``setpoint`` is sniffed for a bound method whose
        owner exposes ``get_full(t) -> (pos, vel, acc, jerk, snap)``. A
        trajectory that grows higher derivatives therefore unlocks the
        feedforward on its own — no controller change required.
        """
        setpoint_fn = getattr(self._payload_controller, "setpoint", None)
        owner = getattr(setpoint_fn, "__self__", None)
        return getattr(owner, "get_full", None)

    @property
    def feedforward_drops(self) -> int:
        """Steps of the last :meth:`simulate` that flew without the airframe feed-forward.

        A step drops it where the flat map refuses the reference sample (a
        cable would go slack, an airframe in free fall) or recovers an airframe
        that is not upright
        (:attr:`~udaan.flatness.MultiQuadCsPayloadRefState.upright`). Reset by
        :meth:`simulate`, which logs one warning with the intervals at its end.
        """
        return self._feedforward_drops

    def _desired_attitudes(self, flat_map, flat_fn, t):
        """Per-quadrotor ``(R_i, Ω_i, Ω̇_i)`` recovered from the flat output.

        All three are used. The attitude loop of
        :class:`~udaan.control.quadrotor.GeometricAttitudeController` keeps
        ``b₃`` of ``R_i^d`` from the commanded thrust vector and takes from the
        recovered ``R_i`` the one remaining degree of freedom, the heading of
        ``b₁`` about ``b₃``. With the default ψ ≡ 0 that heading is the world
        ``x`` axis and the composition is close to a no-op; bind
        :attr:`yaw_setpoint` to command a heading that is not zero.

        Returns ``None`` when the flat map refuses the sample — a cable would
        go slack, or a quadrotor is in free fall — or recovers an airframe that
        is not upright. That is a legitimate outcome for an aggressive
        trajectory, and the right response is to drop the feedforward for that
        step rather than to abort the simulation; the feedback cascade is still
        stabilising. :meth:`simulate` counts it in :attr:`feedforward_drops`.

        A :class:`~udaan.flatness.TensionSplitContractError` is deliberately
        not absorbed. A policy whose split does not add up is a programming
        error, and degrading it to feedback-only control would hide it behind a
        slightly worse tracking number.
        """
        pos, vel, acc, jerk, snap = flat_fn(t)
        # MultiQuadCsPayloadFlats needs x_L through pop; trajectories stop at
        # snap, so crackle and pop are taken as zero. They enter only the
        # moment feedforward Ω̇_i, and truncating them there is exactly the
        # approximation the M1 path makes by stopping its jet at snap.
        # A trajectory that also serves `get_jet(t, order)` (CrazyTrajectory,
        # SpiralTrajectory) supplies crackle and pop exactly instead.
        get_jet = getattr(getattr(flat_fn, "__self__", None), "get_jet", None)
        if get_jet is not None:
            x_L = Jet(np.asarray(get_jet(t, 6), dtype=float))
        else:
            x_L = Jet(np.array([pos, vel, acc, jerk, snap, np.zeros(3), np.zeros(3)], dtype=float))
        psi = yaw_jet(self._yaw_setpoint, t)
        flats = MultiQuadCsPayloadFlats(
            x_L=x_L,
            psi=[psi for _ in range(self._num_quadrotors)],
        )
        try:
            ref, _ = flat_map.recover(flats)
        except TensionSplitContractError:
            # The policy itself is broken — never degrade this into a silently
            # feedback-only run.
            raise
        except ValueError as err:
            _logger.debug("flatness feedforward unavailable at t=%.4f: %s", t, err)
            return None
        if not ref.upright:
            return None
        return [
            (ref.orientations[i], ref.angular_velocities[i], ref.angular_accelerations[i])
            for i in range(self._num_quadrotors)
        ]

    def simulate(self, tf, **kwargs):
        """Run the closed loop to time ``tf`` on the payload controller's setpoint.

        Every input path flies the same law: the payload cascade's thrust-force
        vectors once per :meth:`step`, and each airframe's attitude loop closed
        on them inside :meth:`step` at :attr:`attitude_rate_hz`. On WRENCH and
        PROP_FORCES the vectors are handed to :meth:`step` wrapped, and the
        attitude loops' output goes on as wrenches or, allocated, as rotor
        forces.

        Where the reference needs an airframe that is not upright, or the flat
        map refuses the sample, the step flies on feedback alone: it is counted
        in :attr:`feedforward_drops`, and one warning at the end of the run
        gives the intervals.
        :meth:`~udaan.flatness.MultiQuadCsPayload.check_reference` finds them
        before flight.
        """
        self.reset(**kwargs)
        self._feedforward_drops = 0

        flat_fn = self._flatness_feedforward_fn()
        flat_map = None
        if flat_fn is not None:
            # The reference must be split the same way the closed loop splits
            # it, or the feedforward attitude belongs to a different formation
            # than the one the controller is driving towards. A payload
            # controller that does not expose its policy makes that agreement
            # unknowable — falling back on a default ring here would build the
            # reference at *this model's* radius while the loop allocates at
            # whatever the controller uses, which is exactly the mismatch the
            # shared policy exists to prevent. Drop the feedforward instead.
            policy = getattr(self._payload_controller, "tension_policy", None)
            if policy is None:
                _logger.warning(
                    "%s exposes no tension_policy, so the closed loop's force "
                    "split is unknown; running feedback-only rather than "
                    "feeding forward a reference built at a different split.",
                    type(self._payload_controller).__name__,
                )
            else:
                flat_map = MultiQuadCsPayload.from_model(self, tension_policy=policy)

        sample_times: list[float] = []
        sample_upright: list[bool] = []
        start_t = time.time_ns()
        while self.t < tf:
            thrust_forces = self.compute_default_input()
            desired_att = None
            if flat_map is not None and flat_fn is not None:
                desired_att = self._desired_attitudes(flat_map, flat_fn, self.t)
                sample_times.append(self.t)
                sample_upright.append(desired_att is not None)
                if desired_att is None:
                    self._feedforward_drops += 1
            if self.verbose >= 2:
                self._log_state(thrust_forces)
            u: np.ndarray | _ThrustCommand = thrust_forces
            if self._input_type != InputType.ACCELERATION:
                u = _ThrustCommand(np.asarray(thrust_forces, dtype=float))
            self.step(u, desired_att=desired_att)

        elapsed = (time.time_ns() - start_t) * 1e-9
        _logger.debug("Simulated %.2fs in %.3fs wall time", self.t, elapsed)
        if self._feedforward_drops:
            intervals = infeasible_intervals(sample_times, sample_upright, self.dt)
            _logger.warning(
                "the reference needs an airframe that is not upright (or the flat map refuses "
                "it) on %d of %d steps, %s: those steps flew without the airframe feed-forward",
                self._feedforward_drops,
                len(sample_times),
                format_intervals(intervals),
            )

    def _log_state(self, u):
        if self._input_type != InputType.ACCELERATION:
            u = _ThrustCommand(np.asarray(u, dtype=float))
        wrench = self._repackage_input(u)
        _logger.debug(
            "t=%.4f xL=%s vL=%s thrust=%s",
            self.t,
            np.array2string(self.state.payload_position, precision=4),
            np.array2string(self.state.payload_velocity, precision=4),
            np.array2string(wrench[:, 0], precision=4),
        )
