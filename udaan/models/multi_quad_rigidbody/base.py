r"""MultiQuadRigidbodyBase — pure dynamics and state, no rendering.

``nQ`` rigid-body quadrotors cooperatively carrying a single **rigid-body**
payload, each through its own taut, massless cable attached at a distinct
body-frame point of the load, integrated with ZOH (zero-order hold) Euler.

This is M4.  It is M3
(:mod:`udaan.models.multi_quad_cspayload`) plus two things: the payload has an
attitude, and the cables attach at distinct points ``r_i`` of the payload body
frame.  Those two additions are what let the team exert a *moment* on the load,
and they are also what couples payload translation to payload rotation.

.. warning::

   **This module is a derivation, not a port.**  The legacy reference
   ``phd_research/quadrotor-flexible-cables-payload/Multiple-Quad-Rigid-Load/
   main_multiquad_rigid_flexible.m::odefun_multiQuad_rigid`` builds the coupling
   blocks and then never assembles or solves them — it returns six variables
   that are never assigned and would raise at runtime.  Its Python sibling
   ``floating_models/base/multi_point_suspended_rigidload.py`` is a ``pass``
   stub.  Everything below is re-derived from scratch; the "Validation" section
   records how each line was checked.

Configuration
-------------
The configuration manifold is

.. math::

    Q = \mathbb{R}^3 \times SO(3) \times \left(S^2 \times SO(3)\right)^{n_Q},

carrying the payload position ``x_L`` and attitude ``R_L``, one cable direction
``q_i \in S^2`` and one quadrotor attitude ``R_i \in SO(3)`` per vehicle.  With
the tangent directions the state has dimension ``12 + 10 n_Q``.

* Payload: position ``x_L``, velocity ``v_L``, attitude ``R_L \in SO(3)``, body
  angular velocity ``\Omega_L`` with ``\dot R_L = R_L \hat\Omega_L``.
* Cable ``i``: direction ``q_i \in S^2``, angular velocity
  ``\omega_i \in T_{q_i}S^2`` with ``\dot q_i = \omega_i \times q_i``, length
  ``l_i``.
* Quadrotor ``i``: attitude ``R_i``, body rate ``\Omega_i``, mass ``m_i``,
  inertia ``J_i``, thrust vector ``u_i = f_i R_i e_3`` and body moment ``M_i``.
* Attachment ``i``: a fixed point ``r_i`` in the **payload body frame**.

As everywhere else in this codebase (:mod:`udaan.flatness.cable`,
:mod:`udaan.models.quadrotor_cspayload`, :mod:`udaan.models.multi_quad_cspayload`)
``q_i`` points **from the quadrotor to the load**.  The world-frame attachment
point and the quadrotor position are therefore *derived*, not independent:

.. math::

    x_{P_i} = x_L + R_L r_i, \qquad x_{Q_i} = x_L + R_L r_i - l_i q_i .

Setting every ``r_i = 0`` collapses this to M3's ``x_{Q_i} = x_L - l_i q_i``,
which is the sign check that matters: the convention is fixed by internal
consistency with M3, not by any external source.

Derivation
----------
Write ``A_i := R_L \hat r_i`` (so that ``A_i v = R_L (r_i \times v)``),
``P_i := q_i q_i^\top`` for the projection along cable ``i`` and
``P_i^\perp := I - P_i``.  Differentiating the attachment kinematics twice,

.. math::

    \dot x_{P_i} &= v_L - A_i \Omega_L, \\
    \ddot x_{P_i} &= a_L - A_i \dot\Omega_L + c_i,
    \qquad c_i := R_L \hat\Omega_L \hat\Omega_L r_i ,

where ``c_i`` is the centripetal term of the rotating attachment point.  For
the cable, differentiating ``\dot q_i = \omega_i \times q_i`` and using
``q_i \cdot \ddot q_i = -\|\dot q_i\|^2`` gives
``\ddot q_i = \dot\omega_i \times q_i - \|\dot q_i\|^2 q_i``, hence

.. math::

    a_{Q_i} = \ddot x_{P_i} - l_i (\dot\omega_i \times q_i)
              + l_i \|\dot q_i\|^2 q_i .

**Step 1 — Newton–Euler with explicit tensions.**  Let ``\mathbf{T}_i`` be the
force the cable applies *to quadrotor* ``i``.  A taut massless cable can only
pull along its own direction, so ``\mathbf{T}_i = t_i q_i`` with ``t_i \ge 0``
(the cable pulls the vehicle towards the load, along ``+q_i``); by Newton's
third law the force applied to the payload at ``x_{P_i}`` is ``-\mathbf{T}_i``.
Then

.. math::
    :label: m4-newton

    m_i a_{Q_i} &= u_i - m_i g e_3 + \mathbf{T}_i, \\
    m_L a_L &= -\sum_i \mathbf{T}_i - m_L g e_3, \\
    J_L \dot\Omega_L + \Omega_L \times J_L \Omega_L
      &= \sum_i r_i \times \bigl(R_L^\top (-\mathbf{T}_i)\bigr)
       = \sum_i A_i^\top \mathbf{T}_i ,

the last equality because ``\hat r_i R_L^\top = -(R_L \hat r_i)^\top = -A_i^\top``.
Quadrotor attitude is unchanged from M1: ``\dot R_i = R_i \hat\Omega_i`` and
``J_i \dot\Omega_i = M_i - \Omega_i \times J_i \Omega_i``.

**Step 2 — the parallel projection eliminates the tensions.**  Because
``\dot\omega_i \times q_i \perp q_i``, projecting the quadrotor equation onto
``q_i`` contains no ``\dot\omega_i``:

.. math::

    t_i = m_i q_i^\top(\ddot x_{P_i} + g e_3) + m_i l_i \|\dot q_i\|^2
          - q_i^\top u_i ,

so, writing ``\alpha := a_L + g e_3`` and
``b_i := m_i l_i \|\dot q_i\|^2 q_i - P_i u_i``,

.. math::
    :label: m4-tension

    \mathbf{T}_i = m_i P_i\bigl(\alpha - A_i \dot\Omega_L + c_i\bigr) + b_i .

This is the whole trick: ``\mathbf{T}_i`` is an *affine function of*
``(\alpha, \dot\Omega_L)`` alone.  The cable accelerations never enter it.

**Step 3 — the coupled 6×6 payload system.**  Substituting
:eq:`m4-tension` into the payload force and moment equations of
:eq:`m4-newton` gives a linear system in the six unknowns
``(\alpha, \dot\Omega_L)``:

.. math::
    :label: m4-mass-matrix

    \begin{bmatrix} M_{11} & M_{12} \\ M_{12}^\top & M_{22} \end{bmatrix}
    \begin{bmatrix} \alpha \\ \dot\Omega_L \end{bmatrix}
    =
    \begin{bmatrix} h_1 \\ h_2 \end{bmatrix},

.. math::

    M_{11} &= m_L I + \sum_i m_i P_i, &
    M_{12} &= -\sum_i m_i P_i A_i, &
    M_{22} &= J_L + \sum_i m_i A_i^\top P_i A_i, \\
    h_1 &= -\sum_i \bigl(m_i P_i c_i + b_i\bigr), &
    h_2 &= \sum_i A_i^\top \bigl(m_i P_i c_i + b_i\bigr) - \Omega_L \times J_L \Omega_L .
    & &

**Step 4 — the cable accelerations follow explicitly.**  Crossing the
quadrotor equation of :eq:`m4-newton` with ``q_i`` annihilates
``\mathbf{T}_i \parallel q_i`` and leaves

.. math::
    :label: m4-cable

    \dot\omega_i = \frac{1}{l_i}\, \hat q_i \bigl(\alpha - A_i \dot\Omega_L + c_i\bigr)
                   - \frac{1}{m_i l_i}\, \hat q_i u_i^{\perp} .

The right-hand side is a cross product with ``q_i``, hence automatically
tangent to the sphere — the tangency constraint is preserved by the continuous
dynamics, exactly as in M3.

**On the "``(6 + 3 n_Q)`` block system".**  Assembling
:eq:`m4-newton` in the unknowns ``(\alpha, \dot\Omega_L, \dot\omega_1, \dots)``
*does* give a ``(6 + 3 n_Q) \times (6 + 3 n_Q)`` symmetric system — the M3
system with a payload-rotation block bordered onto it.  Steps 2 and 4 above are
its exact block elimination: the parallel/perpendicular split of the quadrotor
equation decouples ``t_i`` from ``\dot\omega_i``, giving
:eq:`m4-mass-matrix` in closed form and the ``\dot\omega_i`` back-substitution
:eq:`m4-cable`.  Only a ``6 × 6`` solve is ever performed, at any ``n_Q`` — the
same reduction M3 makes to ``3 × 3``.

Two words of precision about that assembled system, since it is easy to say
more than is true.  Its per-cable diagonal block is
``m_i l_i^2 P_i^\perp``, which is **rank 2**: the full matrix is singular by
exactly ``n_Q``, one direction per cable, because ``\dot\omega_i`` along
``q_i`` does not appear in the dynamics at all (the ``\omega_i`` component
along ``q_i`` is a gauge freedom of the ``S^2`` parametrisation, and the
tangency constraint ``q_i \cdot \omega_i = 0`` removes it).  So "the Schur
complement onto ``(\alpha, \dot\Omega_L)``" is well defined only through the
pseudo-inverse of those blocks, or equivalently on the tangent spaces
``T_{q_i}S^2`` where they *are* invertible.  On that quotient the elimination
is exact, and it is the elimination — not any approximation of it — that
:meth:`MultiQuadRigidbodyBase.accelerations` implements.

**Invertibility.**  Each cable contributes

.. math::

    m_i \begin{bmatrix} I \\ -A_i^\top \end{bmatrix} P_i
        \begin{bmatrix} I & -A_i \end{bmatrix} \succeq 0,

so the mass matrix of :eq:`m4-mass-matrix` is
``\mathrm{diag}(m_L I, J_L)`` plus a sum of positive semi-definite terms, hence
positive definite whenever ``m_L > 0`` and ``J_L \succ 0`` — for *every* cable
configuration and *every* attachment geometry, including all-parallel cables,
collinear attachment points and ``n_Q = 1``.  Degenerate grasp geometry makes
the *static* allocation problem ill-posed (see :attr:`grasp_map`); it never
makes the dynamics singular.

Static trim
-----------
Setting every derivative to zero in :eq:`m4-newton` gives the hover
conditions, which are what :func:`hover_wrench_map` and the tests use:

.. math::

    \sum_i t_i q_i = -m_L g e_3,
    \qquad
    \sum_i t_i \bigl(r_i \times (R_L^\top q_i)\bigr) = 0,
    \qquad
    u_i = m_i g e_3 - t_i q_i .

The middle equation is the extra constraint a rigid-body payload imposes and a
point mass does not: the grasp geometry must be able to produce zero net
moment.  The ring formation built by :func:`ring_attachment_points` and
:func:`~udaan.utils.grasp.formation_cable_attitudes` satisfies
it exactly at every ``n_Q``, with a uniform tension
``t = m_L g\, l /(n_Q \Delta z)``: the vertical component of ``r_i \times q_i``
vanishes because ``r_i`` and ``q_i`` share an azimuth, and the horizontal
components rotate with ``\theta_i`` and cancel around the ring.

Validation
----------
Six independent checks, all exercised by ``tests/test_multi_quad_rigidbody_base.py``:

1. **Payload-only oracle.**  Extracting ``\mathbf{T}_i`` with
   :meth:`MultiQuadRigidbodyBase.cable_tensions` and feeding it to a
   payload-only SE(3) integrator reproduces the payload trajectory.  That
   oracle is the *corrected* form of
   ``verify_diff_flat.m::odefun_load_dynamics`` — see the note below.
2. **Reduction to M3.**  With every ``r_i = 0`` and ``\Omega_L(0) = 0`` the
   payload rotation decouples and the translation matches
   :class:`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase` to
   machine precision.  Round-off, not bitwise: the two paths solve *different*
   linear systems — M3 assembles its ``3 × 3`` inline, M4 goes through
   :meth:`MultiQuadRigidbodyBase._payload_blocks` and a ``6 × 6`` — and agree
   only to rounding.  Measured over 400 steps the payload trajectory does come
   out bit-identical at every ``n_Q`` tested, while individual cable directions
   drift by up to ``4e-17``; the test asserts ``atol=TOL_EXACT`` rather than
   equality for exactly that reason.  Sharing nothing but two geometry helpers
   is what makes the agreement evidence at all.
3. **Hover trim.**  The first-principles trim above, applied open loop, holds
   position *and* attitude.
4. **Energy.**  Conserved under zero input.
5. **Momentum.**  Linear momentum changes only under gravity; angular momentum
   about the system centre of mass is conserved.
6. **Conditioning.**  :meth:`MultiQuadRigidbodyBase.mass_matrix` stays
   positive definite for parallel cables, collinear attachments and ``n_Q = 1``.

.. note::

   ``verify_diff_flat.m`` has an internal sign inconsistency in exactly the
   channel this model adds.  Its ``odefun_load_dynamics`` accumulates
   ``sum_Torques -= r_i \times (R_L^\top B_i)`` (line 304) while its own grasp
   matrix, ``data.Phi = [I; \hat r_i]`` (line 31), defines the load wrench with
   the *opposite* sign, ``W = [\ldots; \sum \hat r_i \tau_i]``, and it is that
   ``\Phi`` which generates the very tensions the integrator consumes.  Since
   ``B_i`` is unambiguously the force *on the load* there (``vL_dot =
   \sum B_i/m_L - g e_3``), the moment must be ``+\sum r_i \times (R_L^\top B_i)``,
   which is what :eq:`m4-newton` says and what ``\Phi`` says.  Line 304 is
   wrong; the oracle used for validation here is the corrected one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ...control.multi_quad_rigidbody import MultiQuadRigidbodyController
from ...control.quadrotor import GeometricAttitudeController, PositionPDController
from ...core.defaults import (
    DEFAULT_ARM_LENGTH,
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORCE_CONSTANT,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_NUM_QUADROTORS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    DEFAULT_RIGID_PAYLOAD_INERTIA,
    DEFAULT_RIGID_PAYLOAD_MASS,
    DEFAULT_TORQUE_CONSTANT,
    GRAVITY,
)
from ...core.types import ForceType, InputType, Vec3
from ...flatness.jet import Jet
from ...flatness.multi_quad_rigidbody import MultiQuadRigidbody, MultiQuadRigidbodyFlats
from ...manif import S2, SO3, TS2, TSO3, hat
from ...utils.grasp import (
    formation_cable_attitudes,
    grasp_map,
    hover_wrench_map,
    ring_attachment_points,
)
from ...utils.logging import get_logger
from ..multi_quad_cspayload.base import cable_state_from_geometry

_logger = get_logger(__name__)

#: Below this mass (kg) a body is warned about as unphysical. The dynamics stay
#: well posed — the guard that matters is strict positivity — but a payload this
#: light is almost certainly a typo or a unit error.
_MIN_PHYSICAL_MASS = 1e-6

#: A cable tension below this (N) counts as compressive rather than as round-off
#: on a slack-but-taut cable.
_TENSION_TOL = 1e-9

__all__ = [
    "MultiQuadRigidbodyBase",
    "MultiQuadRigidbodyState",
    "cable_state_from_geometry",
    "formation_cable_attitudes",
    "grasp_map",
    "hover_wrench_map",
    "ring_attachment_points",
]


@dataclass
class MultiQuadRigidbodyState:
    """State of ``num_quadrotors`` quadrotors sharing one rigid-body payload.

    Every per-quadrotor entry is a list of length ``num_quadrotors``, indexed by
    quadrotor.  ``positions``/``velocities`` are *derived* from the payload pose
    and the cable states in the analytical model (the cables are inextensible
    and the attachment points are rigidly fixed to the load); the MuJoCo backend
    overwrites them with the measured body states.
    """

    num_quadrotors: int = DEFAULT_NUM_QUADROTORS

    # Payload pose and twist
    payload_position: Vec3 = field(default_factory=lambda: np.zeros(3))
    payload_velocity: Vec3 = field(default_factory=lambda: np.zeros(3))
    payload_orientation: SO3 = field(default_factory=SO3)
    payload_angular_velocity: TSO3 = field(default_factory=TSO3)

    # Per-quadrotor position and velocity (derived from payload pose + cable)
    positions: list[Vec3] = field(default_factory=list)
    velocities: list[Vec3] = field(default_factory=list)

    # Per-quadrotor orientation and body angular velocity
    orientations: list[SO3] = field(default_factory=list)
    angular_velocities: list[TSO3] = field(default_factory=list)

    # Per-cable attitude and angular velocity
    cable_attitudes: list[S2] = field(default_factory=list)
    cable_angular_velocities: list[TS2] = field(default_factory=list)

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
        self.payload_orientation = SO3()
        self.payload_angular_velocity = TSO3()
        self.positions = [np.zeros(3) for _ in range(nQ)]
        self.velocities = [np.zeros(3) for _ in range(nQ)]
        self.orientations = [SO3() for _ in range(nQ)]
        self.angular_velocities = [TSO3() for _ in range(nQ)]
        self.cable_attitudes = [S2(np.array([0.0, 0.0, -1.0])) for _ in range(nQ)]
        self.cable_angular_velocities = [TS2() for _ in range(nQ)]

    def dq(self, i: int) -> np.ndarray:
        """Cable direction rate ``q̇_i = ω_i × q_i`` of cable ``i``."""
        return np.cross(
            np.asarray(self.cable_angular_velocities[i]),
            np.asarray(self.cable_attitudes[i]),
        )


class MultiQuadRigidbodyBase:
    """Base multi-quadrotor rigid-body-payload model (M4) — pure dynamics.

    The quadrotors are rigid bodies: the per-quadrotor input is a thrust
    magnitude and a body moment, not a free force vector.  Masses, inertias and
    cable lengths are per-index and accept a scalar (broadcast to every
    quadrotor) or a length-``num_quadrotors`` sequence.  The attachment points
    ``r_i`` are a first-class parameter — the geometry of the grasp, not a
    hardcoded matrix.

    Usage::

        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        mdl.attachment_points = [[0.5, 0.5, 0.05], [-0.5, 0.5, 0.05],
                                 [-0.5, -0.5, 0.05], [0.5, -0.5, 0.05]]
        mdl.simulate(tf=5.0, payload_position=np.array([0.0, 0.0, 1.0]))

    Assigning a parameter after construction is equivalent to passing it to the
    constructor: every setter re-synchronises the default controllers, which
    cache the grasp and the masses (see :meth:`_sync_default_controllers`).
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
        self._compressive_cable_steps = 0

        self.state = MultiQuadRigidbodyState(num_quadrotors=nQ)
        self.t = 0.0
        self.verbose = kwargs.get("verbose", 0)

        # System parameters — per quadrotor, scalar-broadcastable
        self._masses = np.full(nQ, DEFAULT_QUAD_MASS)
        self._inertias = np.tile(DEFAULT_QUAD_INERTIA, (nQ, 1, 1))
        self._inertia_invs = np.linalg.inv(self._inertias)
        self._cable_lengths = np.full(nQ, DEFAULT_CABLE_LENGTH)
        self._payload_mass = DEFAULT_RIGID_PAYLOAD_MASS
        self._payload_inertia = np.array(DEFAULT_RIGID_PAYLOAD_INERTIA, dtype=float)
        self._payload_inertia_inv = np.linalg.inv(self._payload_inertia)
        self._attachment_points = ring_attachment_points(nQ)
        # Routed through the property so the constructor kwarg and later
        # assignment share one validation.
        self.formation_radius = kwargs.get("formation_radius", DEFAULT_FORMATION_RADIUS)

        for key in (
            "masses",
            "inertias",
            "cable_lengths",
            "payload_mass",
            "payload_inertia",
            "attachment_points",
        ):
            if key in kwargs:
                setattr(self, key, kwargs[key])

        # Physical constants
        self._g = GRAVITY
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._e3 = np.array([0.0, 0.0, 1.0])

        # Actuator limits (per quadrotor)
        self._min_thrust = 0.0
        self._max_thrust = 30.0
        self._min_torque = np.array([-3.0, -3.0, -3.0])
        self._max_torque = np.array([3.0, 3.0, 3.0])
        self._prop_min_force = 0.0
        self._prop_max_force = 10.0

        # Integration: 5ms outer loop, 4 inner substeps at 1.25ms
        self.dt = kwargs.get("dt", 1.0 / 200.0)
        self._inner_loop_steps = kwargs.get("inner_loop_steps", 4)
        self.h = self.dt / self._inner_loop_steps

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

        # Steps on which the flat map refused the reference and the loop fell
        # back to feedback-only; zeroed at the start of every ``simulate()``.
        self._feedforward_drops = 0

        # Default controllers
        self._init_default_controllers()

    # ─── Controllers ───────────────────────────────────────────────

    def _init_default_controllers(self):
        nQ = self._num_quadrotors
        self._att_controllers = [
            GeometricAttitudeController(inertia=self._inertias[i]) for i in range(nQ)
        ]
        # Station-keeping only: each quadrotor holds whatever position
        # ``reset()`` left it at, carrying its own weight plus an equal share of
        # the payload. Kept as an open-loop-ish baseline and exposed through
        # ``compute_formation_input()``; ``simulate()`` runs the payload
        # cascade below.
        self._pos_controllers = [
            PositionPDController(mass=self._masses[i] + self._payload_mass / nQ) for i in range(nQ)
        ]
        for i in range(nQ):
            self._pos_controllers[i].setpoint = self._formation_setpoint(i)

        # The real cascade: payload SE(3) PD -> grasp-map allocation -> per-cable
        # S² control -> per-quadrotor thrust vector.
        self._payload_controller = MultiQuadRigidbodyController.from_model(self)
        self._controllers_ready = True

    def _sync_default_controllers(self):
        """Push the current physical parameters into the default controllers.

        Every controller in the cascade caches parameters at construction: the
        attitude loops hold ``J_i``, the station-keeping PDs hold their share of
        the system mass, and the payload cascade caches the whole grasp — ``Φ``,
        ``Φ⁺``, the null-space bias and every mass. A setter that writes only
        ``self._…`` therefore leaves the *controller* describing the machine
        that was constructed rather than the one being simulated. That is not a
        cosmetic drift: assigning this class's own documented example geometry
        through :attr:`attachment_points` and simulating for 5 s used to leave
        the payload at 132° of attitude error, with position tracking still
        perfect, because the cascade was allocating through the default ring's
        grasp instead of the assigned one.

        A controller the caller has substituted for one of ours is left alone
        unless it opts in with a ``sync_from_model`` hook — its parameters are
        the caller's business.
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

    @staticmethod
    def _check_positive(values: float | np.ndarray, name: str):
        """Reject non-positive or non-finite physical parameters.

        The positive-definiteness of the ``6 × 6`` system (module docstring,
        "Invertibility") is conditioned on ``m_L > 0``, ``m_i > 0`` and
        ``J_L ≻ 0``; the inertia was already guarded, the masses were not. A
        negative mass makes the mass matrix indefinite and ``accelerations()``
        returns a solution to an equation that describes nothing, while a zero
        cable length divides by zero in :eq:`m4-cable` and only surfaces as a
        ``nan`` behind a ``RuntimeWarning``.
        """
        arr = np.atleast_1d(np.asarray(values, dtype=float))
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite; got {arr.tolist()}")
        if not np.all(arr > 0.0):
            raise ValueError(f"{name} must be strictly positive; got {arr.tolist()}")

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
        self._check_positive(masses, "masses")
        self._masses = masses
        self._sync_default_controllers()

    @property
    def inertias(self) -> np.ndarray:
        """Per-quadrotor inertia tensors, shape ``(nQ, 3, 3)``.

        Accepts a single ``(3,)`` diagonal or ``(3, 3)`` tensor, broadcast to
        every quadrotor, or a per-index ``(nQ, 3, 3)`` stack. A ``(nQ, 3)``
        stack of diagonals is deliberately *not* accepted: at ``nQ == 3`` it
        has the same shape as a single tensor, and symmetry is what
        disambiguates the two. Build the stack yourself with :func:`numpy.diag`.
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
        """Reject anything that is not a symmetric positive-definite tensor."""
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
        self._check_positive(lengths, "cable_lengths")
        self._cable_lengths = lengths
        self._sync_default_controllers()

    @property
    def formation_radius(self) -> float:
        """Radius of the default ring formation used by :meth:`reset`.

        Each quadrotor sits this far *outboard of its own attachment point*,
        not of the payload centre — which is what makes the resulting trim
        moment-free (module docstring, "Static trim"). Must be strictly
        positive and strictly below every cable length.
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
        self._formation_radius = radius
        self._sync_default_controllers()

    @property
    def payload_mass(self) -> float:
        """Payload mass ``m_L``. Must be strictly positive.

        ``m_L > 0`` is half the condition under which the ``6 × 6`` system is
        positive definite — the other half, ``J_L ≻ 0``, is enforced by
        :attr:`payload_inertia`. At ``m_L < 0`` the mass matrix is indefinite
        and ``accelerations()`` still returns a number.
        """
        return self._payload_mass

    @payload_mass.setter
    def payload_mass(self, value: float):
        mass = float(value)
        self._check_positive(mass, "payload_mass")
        if mass < _MIN_PHYSICAL_MASS:
            _logger.warning(
                "payload_mass=%g is below %g kg: the coupled system stays "
                "positive definite but its conditioning is dominated by the "
                "cable terms, and the payload is effectively massless.",
                mass,
                _MIN_PHYSICAL_MASS,
            )
        self._payload_mass = mass
        self._sync_default_controllers()

    @property
    def payload_inertia(self) -> np.ndarray:
        """Payload inertia tensor ``J_L`` about its centre of mass, ``(3, 3)``.

        Accepts a ``(3,)`` diagonal or a full ``(3, 3)`` tensor; must be
        symmetric positive definite. ``J_L ≻ 0`` is what makes the ``M_22``
        block of the coupled system invertible, so a singular tensor is
        rejected rather than left to surface as a ``nan`` payload rate.
        """
        return self._payload_inertia

    @payload_inertia.setter
    def payload_inertia(self, value):
        arr = np.asarray(value, dtype=float)
        if arr.shape == (3,):
            arr = np.diag(arr)
        elif arr.shape != (3, 3):
            raise ValueError(f"payload_inertia must be (3,) or (3, 3); got {arr.shape}")
        self._check_inertia_tensor(arr, "payload_inertia")
        self._payload_inertia = arr
        self._payload_inertia_inv = np.linalg.inv(arr)
        self._sync_default_controllers()

    @property
    def payload_inertia_inv(self) -> np.ndarray:
        return self._payload_inertia_inv

    @property
    def attachment_points(self) -> np.ndarray:
        """Cable attachment points ``r_i`` in the **payload body frame**, ``(nQ, 3)``.

        The parameter that distinguishes M4 from M3: at ``r_i = 0`` for every
        ``i`` the payload rotation decouples from the cables and the
        translation reduces exactly to
        :class:`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase`.
        """
        return self._attachment_points

    @attachment_points.setter
    def attachment_points(self, value):
        arr = np.asarray(value, dtype=float)
        if arr.shape != (self._num_quadrotors, 3):
            raise ValueError(
                f"attachment_points must be ({self._num_quadrotors}, 3); got {arr.shape}"
            )
        self._attachment_points = arr.copy()
        self._sync_default_controllers()

    @property
    def grasp_map(self) -> np.ndarray:
        """``(6, 3 nQ)`` grasp matrix of the current attachment points.

        See :func:`udaan.utils.grasp.grasp_map` for what it maps and why its
        rank matters.
        """
        return grasp_map(self._attachment_points)

    # ─── Dynamics ──────────────────────────────────────────────────

    def _payload_blocks(self, q, thrusts):
        """Assemble the 6×6 coupled system of the module docstring.

        Args:
            q: list of ``(3,)`` cable directions.
            thrusts: ``(nQ,)`` thrust magnitudes.

        Returns:
            ``(mass_matrix, rhs, u_perp, A, centripetal)`` where ``A[i]`` is
            ``R_L hat(r_i)`` and ``centripetal[i]`` is ``c_i``.
        """
        nQ = self._num_quadrotors
        R_L = np.asarray(self.state.payload_orientation)
        Omega_L = np.asarray(self.state.payload_angular_velocity)
        Omega_hat = hat(Omega_L)

        M11 = self._payload_mass * np.eye(3)
        M12 = np.zeros((3, 3))
        M22 = self._payload_inertia.copy()
        h1 = np.zeros(3)
        h2 = -np.cross(Omega_L, self._payload_inertia @ Omega_L)

        u_perp: list[np.ndarray] = []
        A: list[np.ndarray] = []
        centripetal: list[np.ndarray] = []
        for i in range(nQ):
            A_i = R_L @ hat(self._attachment_points[i])
            c_i = R_L @ (Omega_hat @ (Omega_hat @ self._attachment_points[i]))
            A.append(A_i)
            centripetal.append(c_i)

            m_i = self._masses[i]
            qqT = np.outer(q[i], q[i])
            u_i = thrusts[i] * (self.state.orientations[i] @ self._e3)
            u_perp.append(u_i - qqT @ u_i)

            dq_i = self.state.dq(i)
            # b_i = m_i l_i ‖q̇_i‖² q_i − P_i u_i  (the part of 𝐓_i that does
            # not depend on the payload accelerations).
            b_i = m_i * self._cable_lengths[i] * np.dot(dq_i, dq_i) * q[i] - qqT @ u_i
            g_i = m_i * (qqT @ c_i) + b_i

            m_qqT = m_i * qqT
            M11 += m_qqT
            M12 -= m_qqT @ A_i
            M22 += A_i.T @ m_qqT @ A_i
            h1 -= g_i
            h2 += A_i.T @ g_i

        mass_matrix = np.block([[M11, M12], [M12.T, M22]])
        return mass_matrix, np.concatenate([h1, h2]), u_perp, A, centripetal

    def mass_matrix(self) -> np.ndarray:
        """The ``6 × 6`` coupled payload mass matrix at the current state.

        Positive definite for every cable configuration and every attachment
        geometry — see "Invertibility" in the module docstring. Exposed so
        that claim can be tested directly on degenerate geometries rather than
        inferred from a simulation that happens not to blow up.
        """
        q = [np.asarray(self.state.cable_attitudes[i]) for i in range(self._num_quadrotors)]
        return np.asarray(self._payload_blocks(q, np.zeros(self._num_quadrotors))[0])

    def accelerations(self, thrusts=None, torques=None) -> dict:
        """Every acceleration of the system at the current state.

        Solves the ``6 × 6`` system :eq:`m4-mass-matrix` once and
        back-substitutes :eq:`m4-tension` and :eq:`m4-cable`.  This is the whole
        right-hand side of the ODE; :meth:`_zoh` is nothing but an integrator
        wrapped around it, and the tests read residuals off it rather than
        re-deriving the algebra.

        Args:
            thrusts: ``(nQ,)`` thrust magnitudes ``f_i``, default zero.
            torques: ``(nQ, 3)`` body moments ``M_i``, default zero.

        Returns:
            A dict with keys ``payload_accel`` (``a_L``),
            ``payload_ang_accel`` (``Ω̇_L``), ``cable_ang_accels``
            (``(nQ, 3)`` of ``ω̇_i``), ``quad_ang_accels`` (``(nQ, 3)`` of
            ``Ω̇_i``), ``quad_accels`` (``(nQ, 3)`` of ``a_{Q_i}``),
            ``tensions`` (``(nQ, 3)`` of ``𝐓_i``, the force on the *quadrotor*)
            and ``attachment_accels`` (``(nQ, 3)`` of ``ẍ_{P_i}``).
        """
        nQ = self._num_quadrotors
        thrusts = np.zeros(nQ) if thrusts is None else np.asarray(thrusts, dtype=float)
        torques = np.zeros((nQ, 3)) if torques is None else np.asarray(torques, dtype=float)
        q = [np.asarray(self.state.cable_attitudes[i]) for i in range(nQ)]

        mass_matrix, rhs, u_perp, A, centripetal = self._payload_blocks(q, thrusts)
        solution = np.linalg.solve(mass_matrix, rhs)
        # α = a_L + g e3 is the natural unknown; recover a_L from it.
        accel_plus_g = solution[:3]
        payload_ang_accel = solution[3:]

        cable_ang_accels = np.zeros((nQ, 3))
        quad_ang_accels = np.zeros((nQ, 3))
        quad_accels = np.zeros((nQ, 3))
        tensions = np.zeros((nQ, 3))
        attachment_accels = np.zeros((nQ, 3))
        for i in range(nQ):
            m_i = self._masses[i]
            l_i = self._cable_lengths[i]
            q_hat = hat(q[i])
            qqT = np.outer(q[i], q[i])
            dq_i = self.state.dq(i)

            # Attachment acceleration + g e3: the payload rotation reaches every
            # cable only through this term.
            attach = accel_plus_g - A[i] @ payload_ang_accel + centripetal[i]
            attachment_accels[i] = attach - self._ge3

            b_i = m_i * l_i * np.dot(dq_i, dq_i) * q[i] - qqT @ (
                thrusts[i] * (self.state.orientations[i] @ self._e3)
            )
            tensions[i] = m_i * (qqT @ attach) + b_i

            cable_ang_accels[i] = (q_hat @ attach) / l_i - (q_hat @ u_perp[i]) / (m_i * l_i)
            quad_accels[i] = (
                attachment_accels[i]
                - l_i * np.cross(cable_ang_accels[i], q[i])
                + l_i * np.dot(dq_i, dq_i) * q[i]
            )

            Omega = np.asarray(self.state.angular_velocities[i])
            quad_ang_accels[i] = self._inertia_invs[i] @ (
                torques[i] - np.cross(Omega, self._inertias[i] @ Omega)
            )

        self._check_cable_tensions(tensions, q)

        return {
            "payload_accel": accel_plus_g - self._ge3,
            "payload_ang_accel": payload_ang_accel,
            "cable_ang_accels": cable_ang_accels,
            "quad_ang_accels": quad_ang_accels,
            "quad_accels": quad_accels,
            "tensions": tensions,
            "attachment_accels": attachment_accels,
        }

    def _check_cable_tensions(self, tensions: np.ndarray, q: list[np.ndarray]):
        """Count evaluations that violate the taut-cable precondition ``t_i ≥ 0``.

        Everything above is derived for a taut, massless cable, which can only
        *pull*: ``𝐓_i = t_i q_i`` with ``t_i ≥ 0`` (module docstring, Step 1).
        Nothing in the algebra enforces that, and nothing has to — the solve is
        exact either way — but outside ``t_i ≥ 0`` the answer is the answer for
        a rigid *strut*, not a cable, and it is silently the wrong model. It is
        not a rare corner: an ordinary closed-loop 4 m lateral step at
        ``nQ = 4`` puts a cable in compression on ~7% of its steps, peaking at
        -3.8 N.

        This model has no slack mode to switch into (a slack cable makes the
        configuration manifold itself change dimension), so the violation is
        reported rather than handled: the first one warns, the total lives in
        :attr:`compressive_cable_steps`, and the MuJoCo backend — whose tendons
        genuinely cannot push — is where a trajectory that trips it should be
        checked.
        """
        signed = np.array([float(np.dot(tensions[i], q[i])) for i in range(self._num_quadrotors)])
        if not np.any(signed < -_TENSION_TOL):
            return
        self._compressive_cable_steps += 1
        if self._compressive_cable_steps == 1:
            _logger.warning(
                "cable tension went compressive at t=%.4f: t=%s N. A taut massless "
                "cable can only pull, so the dynamics are outside the model's "
                "precondition on this step. Further occurrences are logged at "
                "debug level; see the compressive_cable_steps counter.",
                self.t,
                np.array2string(signed, precision=4),
            )
        else:
            _logger.debug("compressive cable tension at t=%.4f: %s", self.t, signed)

    @property
    def compressive_cable_steps(self) -> int:
        """Evaluations of :meth:`accelerations` with a compressive cable.

        Reset by :meth:`reset`. A non-zero count means the run left the taut
        massless-cable regime the model is derived for — see
        :meth:`_check_cable_tensions`.
        """
        return self._compressive_cable_steps

    def cable_tensions(self, thrusts=None) -> np.ndarray:
        """Cable force vectors ``𝐓_i`` at the current state, shape ``(nQ, 3)``.

        ``𝐓_i`` is the force the cable applies **to quadrotor** ``i`` (equation
        :eq:`m4-tension`); by Newton's third law the payload feels ``-𝐓_i`` at
        ``x_L + R_L r_i``. Feeding these to a payload-only SE(3) integrator is
        the oracle check of the module docstring.

        Args:
            thrusts: ``(nQ,)`` thrust magnitudes, default zero.
        """
        return np.asarray(self.accelerations(thrusts)["tensions"])

    def _zoh(self, thrusts: np.ndarray, torques: np.ndarray, h: float):
        """Zero-order hold Euler integration for one substep.

        The centripetal coefficient inside ``b_i`` is ‖q̇_i‖², not ‖ω_i‖²: the
        exact identity is ``q_i · q̈_i = -‖q̇_i‖²``, and the two agree only while
        ``ω_i`` is exactly tangent, which ZOH-Euler does not guarantee. Same
        reasoning as M3 and M2.

        Args:
            thrusts: ``(nQ,)`` thrust magnitudes ``f_i``.
            torques: ``(nQ, 3)`` body moments ``M_i``.
            h: substep size.
        """
        nQ = self._num_quadrotors
        omega = [np.asarray(self.state.cable_angular_velocities[i]).copy() for i in range(nQ)]
        Omega_L = np.asarray(self.state.payload_angular_velocity).copy()
        Omegas = [np.asarray(self.state.angular_velocities[i]).copy() for i in range(nQ)]

        acc = self.accelerations(thrusts, torques)
        payload_accel = acc["payload_accel"]

        # payload position dynamics
        self.state.payload_position = (
            self.state.payload_position
            + self.state.payload_velocity * h
            + 0.5 * payload_accel * h**2
        )
        self.state.payload_velocity = self.state.payload_velocity + payload_accel * h

        # payload attitude dynamics
        self.state.payload_orientation = self.state.payload_orientation.step(Omega_L * h)
        self.state.payload_angular_velocity += TSO3(acc["payload_ang_accel"] * h)

        for i in range(nQ):
            # cable attitude dynamics
            self.state.cable_attitudes[i] = self.state.cable_attitudes[i].step(omega[i] * h)
            self.state.cable_angular_velocities[i] += TS2(acc["cable_ang_accels"][i] * h)

            # quadrotor attitude dynamics
            self.state.orientations[i] = self.state.orientations[i].step(Omegas[i] * h)
            self.state.angular_velocities[i] += TSO3(acc["quad_ang_accels"][i] * h)

        # quadrotor positions & velocities derived from the payload pose + cables
        self._sync_quad_states()

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
        """One outer step: repackage input, then run inner substeps."""
        wrench = self._repackage_input(u, desired_att=desired_att)
        thrusts = wrench[:, 0]
        torques = wrench[:, 1:]
        for _ in range(self._inner_loop_steps):
            self._zoh(thrusts, torques, self.h)
            self.t += self.h

    def reset(self, **kwargs):
        """Reset time and state, then apply any state overrides.

        Recognised kwargs: ``payload_position``, ``payload_velocity``,
        ``payload_orientation``, ``payload_angular_velocity``, ``orientations``,
        ``angular_velocities``, ``cable_attitudes``,
        ``cable_angular_velocities``.  Per-quadrotor entries must be sequences
        of length ``num_quadrotors``.  When ``cable_attitudes`` is omitted the
        cables are placed in the default ring formation, one quadrotor radially
        outboard of each attachment point.
        """
        self.t = 0.0
        self._compressive_cable_steps = 0
        self.state.reset()
        self.state.cable_attitudes = formation_cable_attitudes(
            self._num_quadrotors, self._cable_lengths, self._formation_radius
        )

        for key in ("payload_position", "payload_velocity"):
            if key in kwargs:
                # Copy: the integrator rebinds these, but a caller-supplied
                # array must never be aliased into the state.
                setattr(self.state, key, np.array(kwargs[key], dtype=float))

        for key, pose_wrap in (
            ("payload_orientation", SO3),
            ("payload_angular_velocity", TSO3),
        ):
            if key in kwargs:
                setattr(self.state, key, pose_wrap(np.asarray(kwargs[key])))

        per_quad: tuple[tuple[str, type], ...] = (
            ("orientations", SO3),
            ("angular_velocities", TSO3),
            ("cable_attitudes", S2),
            ("cable_angular_velocities", TS2),
        )
        for key, wrap in per_quad:
            if key in kwargs:
                values = kwargs[key]
                self._check_length(values, key)
                # Rewrap every entry: the integrator mutates the tangent
                # wrappers in place, so a caller's object must not be aliased.
                setattr(self.state, key, [wrap(np.asarray(v)) for v in values])

        self._sync_quad_states()
        self._formation_targets = [p.copy() for p in self.state.positions]

    def attachment_position(self, i: int) -> np.ndarray:
        """World-frame position ``x_L + R_L r_i`` of attachment ``i``."""
        return np.asarray(
            self.state.payload_position
            + self.state.payload_orientation @ (self._attachment_points[i])
        )

    def attachment_velocity(self, i: int) -> np.ndarray:
        """World-frame velocity ``v_L + R_L (Ω_L × r_i)`` of attachment ``i``."""
        return np.asarray(
            self.state.payload_velocity
            + self.state.payload_orientation
            @ np.cross(np.asarray(self.state.payload_angular_velocity), self._attachment_points[i])
        )

    def _sync_quad_states(self):
        """Recompute quadrotor positions/velocities from the payload pose + cables."""
        for i in range(self._num_quadrotors):
            l_i = self._cable_lengths[i]
            self.state.positions[i] = self.attachment_position(i) - l_i * np.asarray(
                self.state.cable_attitudes[i]
            )
            self.state.velocities[i] = self.attachment_velocity(i) - l_i * self.state.dq(i)

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

        Mirrors :meth:`udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase.simulate`:
        the payload controller's ``setpoint`` is sniffed for a bound method
        whose owner exposes ``get_full(t) -> (pos, vel, acc, jerk, snap)``. A
        trajectory that grows higher derivatives therefore unlocks the
        feedforward on its own — no controller change required.
        """
        setpoint_fn = getattr(self._payload_controller, "setpoint", None)
        owner = getattr(setpoint_fn, "__self__", None)
        return getattr(owner, "get_full", None)

    @property
    def feedforward_drops(self) -> int:
        """How many steps have dropped the flatness feedforward.

        Reset by :meth:`simulate`. A non-zero count means the reference left
        the flat map's taut-cable certificate — or the range of the grasp — on
        those steps and the run was feedback-only there. The first drop is
        logged as a warning.
        """
        return self._feedforward_drops

    def _desired_attitudes(self, flat_map, flat_fn, t):
        """Per-quadrotor ``(R_i, Ω_i, Ω̇_i)`` recovered from the flat output.

        The payload's *translation* comes from the bound ``get_full``; its
        *attitude* comes from the payload controller's
        :attr:`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController.attitude_setpoint`,
        which is where an M4 attitude manoeuvre is commanded. The two are
        independent references for a rigid-body load in a way they cannot be
        for M3's point mass, so both have to be read.

        Returns ``None`` when the flat map refuses the sample — a cable would
        go slack, a quadrotor is in free fall, or the requested wrench is
        outside ``range Φ`` (which is every non-level attitude command at
        ``nQ ≤ 2``). That is a legitimate outcome, and the right response is to
        drop the feedforward for that step rather than to abort: the feedback
        cascade is still stabilising. It is *not* silent — the drop is counted
        in :attr:`feedforward_drops` and the first one warns.
        """
        pos, vel, acc, jerk, snap = flat_fn(t)
        # MultiQuadRigidbodyFlats needs x_L through its sixth derivative;
        # trajectories stop at snap, so the last two rungs are taken as zero.
        # They enter only the moment feedforward Ω̇_i, which is exactly the
        # approximation the M1 and M3 paths make by stopping their jets at snap.
        x_L = Jet(np.array([pos, vel, acc, jerk, snap, np.zeros(3), np.zeros(3)], dtype=float))
        attitude_fn = getattr(self._payload_controller, "attitude_setpoint", None)
        if attitude_fn is None:
            R_d, Omega_d, dOmega_d = np.eye(3), np.zeros(3), np.zeros(3)
        else:
            R_d, Omega_d, dOmega_d = (np.asarray(v, dtype=float) for v in attitude_fn(t))
        # Ω_L through its fifth derivative; the setpoint protocol stops at Ω̇_d,
        # so the rest are zero on the same grounds as x_L's tail.
        omega = np.zeros((6, 3))
        omega[0] = Omega_d
        omega[1] = dOmega_d
        flats = MultiQuadRigidbodyFlats(
            x_L=x_L,
            R_L=R_d,
            Omega_L=Jet(omega),
            psi=[Jet(np.zeros(3)) for _ in range(self._num_quadrotors)],
        )
        try:
            ref, _ = flat_map.recover(flats)
        except ValueError as err:
            # WrenchAllocationError is a ValueError, so an out-of-range wrench
            # lands here too — the nQ <= 2 attitude command, degraded rather
            # than crashed.
            self._feedforward_drops += 1
            if self._feedforward_drops == 1:
                _logger.warning(
                    "flatness feedforward dropped at t=%.4f (%s); this step ran "
                    "feedback-only. Further drops are logged at debug level; see "
                    "the feedforward_drops counter for the total.",
                    t,
                    err,
                )
            else:
                _logger.debug("flatness feedforward unavailable at t=%.4f: %s", t, err)
            return None
        return [
            (ref.orientations[i], ref.angular_velocities[i], ref.angular_accelerations[i])
            for i in range(self._num_quadrotors)
        ]

    def simulate(self, tf, **kwargs):
        """Run closed-loop simulation to time tf.

        When the payload controller's translation setpoint is a trajectory's
        bound ``get_full``, the run picks up the M4 flatness feedforward:
        :class:`udaan.flatness.MultiQuadRigidbody` recovers each quadrotor's
        reference attitude ``R_i``, body rate ``Ω_i`` and body angular
        acceleration ``Ω̇_i``, and all three are fed to the attitude loops as
        ``desired_att``. Binding ``get_full`` is the whole switch.

        The attitude loop consumes only the rates:
        :meth:`udaan.control.quadrotor.GeometricAttitudeController.compute`
        discards ``R_i`` and rebuilds its own ``R_d`` from the commanded thrust
        direction. Passing ``R_i = I`` therefore gives a bit-identical run, and
        the improvement the feedforward buys is entirely ``Ω_i``/``Ω̇_i``. The
        behaviour is shared with M1 and M3; see that method's docstring.

        The flat map is built from the payload controller's **own**
        ``tension_policy``. ``Φ⁺ W`` is only half the allocation; the other
        half is the null-space bias, so a reference built at a different bias
        describes a different formation from the one the loop is driving
        towards. A controller that does not publish its policy therefore gets
        no feedforward at all — falling back on a default here would
        manufacture exactly that mismatch, silently.
        """
        self.reset(**kwargs)
        self._feedforward_drops = 0

        flat_fn = self._flatness_feedforward_fn()
        flat_map = None
        if flat_fn is not None:
            policy = getattr(self._payload_controller, "tension_policy", None)
            if policy is None:
                _logger.warning(
                    "%s exposes no tension_policy, so the closed loop's wrench "
                    "split is unknown; running feedback-only rather than feeding "
                    "forward a reference built at a different split.",
                    type(self._payload_controller).__name__,
                )
            else:
                flat_map = MultiQuadRigidbody.from_model(self, tension_policy=policy)

        start_t = time.time_ns()
        while self.t < tf:
            u = self.compute_default_input()
            desired_att = (
                None if flat_map is None else self._desired_attitudes(flat_map, flat_fn, self.t)
            )
            if self.verbose >= 2:
                self._log_state(u)
            self.step(u, desired_att=desired_att)

        elapsed = (time.time_ns() - start_t) * 1e-9
        _logger.debug("Simulated %.2fs in %.3fs wall time", self.t, elapsed)

    def _log_state(self, u):
        wrench = self._repackage_input(u)
        _logger.debug(
            "t=%.4f xL=%s vL=%s OmegaL=%s thrust=%s",
            self.t,
            np.array2string(self.state.payload_position, precision=4),
            np.array2string(self.state.payload_velocity, precision=4),
            np.array2string(np.asarray(self.state.payload_angular_velocity), precision=4),
            np.array2string(wrench[:, 0], precision=4),
        )
