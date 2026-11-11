"""QuadrotorCsPayloadBase — pure dynamics and state, no rendering.

Rigid body with cable-suspended payload dynamics with ZOH (zero-order hold)
Euler integration: a 5 ms outer (payload-position) loop of
``N = physics_rate_hz · dt`` substeps, by default 8 substeps of 0.625 ms
(1600 Hz). The attitude loop closes at 800 Hz by default, four times per outer
step, on every input path (see :class:`QuadrotorCsPayloadBase`).

Dynamics reference:
Koushil Sreenath, Taeyoung Lee, and Vijay Kumar. "Geometric control and
differential flatness of a quadrotor UAV with a cable-suspended load."
In 52nd IEEE Conference on Decision and Control, pp. 2269-2274. IEEE, 2013.
https://hybrid-robotics.berkeley.edu/publications/CDC2013_supplement.pdf
"""

import time
from dataclasses import dataclass, field

import numpy as np

from ... import control
from ...core.defaults import (
    DEFAULT_ARM_LENGTH,
    DEFAULT_ATTITUDE_RATE_HZ,
    DEFAULT_FORCE_CONSTANT,
    DEFAULT_M1_QUAD_INERTIA,
    DEFAULT_M1_QUAD_MASS,
    DEFAULT_M2_CABLE_LENGTH,
    DEFAULT_M2_PAYLOAD_MASS,
    DEFAULT_MAX_THRUST,
    DEFAULT_MAX_TORQUE,
    DEFAULT_MIN_THRUST,
    DEFAULT_MIN_TORQUE,
    DEFAULT_PHYSICS_RATE_HZ,
    DEFAULT_PROP_MAX_FORCE,
    DEFAULT_PROP_MIN_FORCE,
    DEFAULT_ROTOR_ANGLES,
    DEFAULT_SPIN_DIRECTIONS,
    DEFAULT_TORQUE_CONSTANT,
    GRAVITY,
)
from ...core.types import ForceType, InputType, Vec3
from ...core.validation import MIN_PHYSICAL_MASS, check_positive
from ...flatness import Jet, QuadrotorCsPayload, QuadrotorCsPayloadFlats, yaw_jet
from ...flatness.feasibility import format_intervals, infeasible_intervals
from ...manif import S2, SO3, TS2, TSO3, zoh_step
from ...utils.logging import get_logger
from .._kwargs import check_reset_kwargs
from .._rates import attitude_updates_per_step, physics_steps_per_step
from ..quadrotor.base import _ThrustCommand

_logger = get_logger(__name__)

#: The initial-condition overrides :meth:`QuadrotorCsPayloadBase.reset` acts on.
#: Anything else is rejected rather than silently discarded — see
#: :mod:`udaan.models._kwargs`.
_SUPPORTED_RESET_KWARGS = (
    "position",
    "velocity",
    "orientation",
    "angular_velocity",
    "payload_position",
    "payload_velocity",
    "cable_attitude",
    "cable_angular_velocity",
)


@dataclass
class QuadrotorCsPayloadState:
    """Quadrotor with cable-suspended payload state."""

    # Quadrotor position and velocity
    position: Vec3 = field(default_factory=lambda: np.zeros(3))
    velocity: Vec3 = field(default_factory=lambda: np.zeros(3))

    # Quadrotor orientation and angular velocity
    orientation: SO3 = field(default_factory=SO3)
    angular_velocity: TSO3 = field(default_factory=TSO3)

    # Payload position and velocity
    payload_position: Vec3 = field(default_factory=lambda: np.zeros(3))
    payload_velocity: Vec3 = field(default_factory=lambda: np.zeros(3))

    # Cable attitude and angular velocity
    cable_attitude: S2 = field(default_factory=lambda: S2(np.array([0.0, 0.0, -1.0])))
    cable_angular_velocity: TS2 = field(default_factory=TS2)

    def dq(self):
        return np.cross(np.asarray(self.cable_angular_velocity), np.asarray(self.cable_attitude))

    def reset(self):
        QuadrotorCsPayloadState.__init__(self)


class QuadrotorCsPayloadBase:
    """Base quadrotor-cable-suspended payload model — pure dynamics, no rendering.

    Supports multiple input types (acceleration, wrench, propeller forces)
    and force types (wrench, propeller forces). Controllers are pluggable
    via the payload_controller and attitude_controller properties.

    The machine is the M1 airframe
    (:data:`~udaan.core.defaults.DEFAULT_M1_QUAD_MASS`,
    :data:`~udaan.core.defaults.DEFAULT_M1_QUAD_INERTIA`) carrying a
    :data:`~udaan.core.defaults.DEFAULT_M2_PAYLOAD_MASS` payload on a
    :data:`~udaan.core.defaults.DEFAULT_M2_CABLE_LENGTH` cable: the bodies of
    ``quad_payload_mj.xml`` and ``quad_flxcbl_mj.xml``.

    Two loops, at two rates. The **payload** loop runs once per :meth:`step`,
    at ``1/dt``: :class:`~udaan.control.quadrotor_cspayload.QuadCSPayloadController`
    computes the thrust-force vector, whose direction is the desired ``b₃``.
    The **attitude** loop runs inside :meth:`step` at ``attitude_rate_hz``: it
    turns the (held) thrust vector and attitude reference into a thrust and
    torques. The physics integrates underneath both at ``physics_rate_hz``.

    Keyword Args:
        dt: the payload-loop period, in s (default 5 ms, 200 Hz): what one
            :meth:`step` advances. ``physics_rate_hz · dt`` must be a whole
            number of substeps and the attitude updates per step must divide
            it, else ``ValueError``.
        attitude_rate_hz: rate at which the attitude loop closes, in Hz
            (default :data:`~udaan.core.defaults.DEFAULT_ATTITUDE_RATE_HZ`,
            800). ``None`` is once per outer step, ``1/dt``. It is the same on
            every input path. It must be ``k/dt`` for a whole ``k`` that
            divides the ``N`` physics substeps per step; anything else raises
            ``ValueError``. The payload loop and the feedforward reference stay
            at ``1/dt`` and are held across the updates.
        physics_rate_hz: rate at which the physics integrates, in Hz (default
            :data:`~udaan.core.defaults.DEFAULT_PHYSICS_RATE_HZ`, 1600). Each
            ``step()`` takes ``N = physics_rate_hz · dt`` substeps of
            ``h = dt/N``: ZOH substeps here, MuJoCo steps on
            :class:`~udaan.models.quadrotor_cspayload.QuadrotorCsPayloadMujoco`,
            which writes ``h`` into the MuJoCo model's timestep. It must lie in
            :data:`~udaan.core.defaults.PHYSICS_RATE_RANGE_HZ` (500–2000 Hz)
            and make ``N`` a whole number, else ``ValueError``.
        input: ``"acceleration"`` (default), ``"wrench"`` or ``"prop_forces"``:
            what :meth:`step` takes. :meth:`simulate` flies the payload
            controller on every one of them.
        force: ``"wrench"`` (default) or ``"prop_forces"``: whether the
            wrench is round-tripped through clipped rotor forces.
    """

    INPUT_TYPE = InputType
    FORCE_TYPE = ForceType

    def __init__(self, **kwargs):
        self.state = QuadrotorCsPayloadState()
        self.t = 0.0
        # Physics substeps taken since reset(); the clock is derived from it.
        self._substeps_taken = 0
        self.verbose = kwargs.get("verbose", 0)

        # System parameters
        self._mass = DEFAULT_M1_QUAD_MASS
        self._inertia = DEFAULT_M1_QUAD_INERTIA.copy()
        self._inertia_inv = np.linalg.inv(self._inertia)

        self._payload_mass = DEFAULT_M2_PAYLOAD_MASS
        self._cable_length = DEFAULT_M2_CABLE_LENGTH

        # Physical constants
        self._g = GRAVITY
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._e3 = np.array([0.0, 0.0, 1.0])

        # Actuator limits
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
        self._last_wrench = np.zeros(4)

        # Input/force type
        self._input_type = InputType.ACCELERATION
        self._force_type = ForceType.WRENCH
        if "input" in kwargs:
            if kwargs["input"] == "prop_forces":
                self._input_type = InputType.PROP_FORCES
            elif kwargs["input"] == "wrench":
                self._input_type = InputType.WRENCH
        if "force" in kwargs:
            if kwargs["force"] == "prop_forces":
                self._force_type = ForceType.PROP_FORCES

        # Allocation matrix
        self._compute_allocation_matrix()

        # Feedforward yaw reference; None is ψ ≡ 0.
        self._yaw_setpoint = None
        # Steps of the last simulate() that flew without the airframe feed-forward.
        self._feedforward_drops = 0

        # Default controllers
        self._init_default_controllers()

    # ─── Controllers ───────────────────────────────────────────────

    def _init_default_controllers(self):
        """Build the default controllers for this model's ``m_Q``, ``J``, ``m_L`` and ``ℓ``.

        The payload controller carries ``m_L`` plus :attr:`cable_mass` at the
        cable's end, so its hover feed-forward is the machine's whole weight.
        The payload controller's gains and this model's attitude gains are the
        time-scale design of
        :func:`~udaan.control.quadrotor_cspayload.cable_cascade_gains` for
        this machine: the attitude loop at
        :data:`~udaan.core.defaults.CABLE_CASCADE_ATTITUDE_BANDWIDTH` on roll
        and pitch, so that it is fast against the cable loop it serves.
        """
        self._pos_controller = control.quadrotor.PositionPDController(mass=self._mass)
        self._payload_controller = control.quadrotor_cspayload.QuadCSPayloadController(
            mass_q=self._mass,
            mass_l=self._payload_mass + self.cable_mass,
            cable_length=self._cable_length,
            inertia=self._inertia,
        )
        design = self._payload_controller.design_gains
        self._att_controller = control.quadrotor.GeometricAttitudeController(
            inertia=self._inertia, kp=design.attitude_kp, kd=design.attitude_kd
        )

    @property
    def position_controller(self):
        return self._pos_controller

    @position_controller.setter
    def position_controller(self, controller):
        self._pos_controller = controller

    @property
    def attitude_controller(self):
        return self._att_controller

    @attitude_controller.setter
    def attitude_controller(self, controller):
        self._att_controller = controller

    @property
    def payload_controller(self):
        return self._payload_controller

    @payload_controller.setter
    def payload_controller(self, controller):
        self._payload_controller = controller

    @property
    def yaw_setpoint(self):
        r"""Yaw reference for the flatness feedforward: ``ψ(t) -> (ψ, ψ̇, ψ̈)``.

        Yaw is the second flat output, ``(x_L, ψ)``; the thrust command fixes
        ``b₃``, and ``ψ`` is the heading the reference attitude adds on top
        through :meth:`~udaan.control.quadrotor.GeometricAttitudeController.compute`.
        ``None`` (the default) means ψ ≡ 0::

            mdl.payload_controller.setpoint = traj.get_full
            mdl.yaw_setpoint = lambda t: (0.5 * t, 0.5, 0.0)

        It is read only on the feedforward path — that is, only when the bound
        trajectory exposes ``get_jet`` (see :meth:`simulate`).
        """
        return self._yaw_setpoint

    @yaw_setpoint.setter
    def yaw_setpoint(self, setpoint):
        if setpoint is not None and not callable(setpoint):
            raise TypeError("yaw_setpoint must be callable psi(t) -> (psi, dpsi, ddpsi), or None")
        self._yaw_setpoint = setpoint

    @property
    def attitude_rate_hz(self) -> float:
        """The rate the attitude loop closes at, in Hz (read-only; set it in the constructor)."""
        return float(self._attitude_updates / self.dt)

    @property
    def physics_rate_hz(self) -> float:
        """The rate the physics integrates at, in Hz (read-only; set it in the constructor)."""
        return float(self._inner_loop_steps / self.dt)

    @property
    def physics_steps(self) -> int:
        """``N``: physics substeps per ``step()``, each of ``dt / N``."""
        return self._inner_loop_steps

    # ─── Properties ────────────────────────────────────────────────

    @property
    def mass(self):
        """Quadrotor mass ``m_Q``. Must be strictly positive.

        It divides the cable rate ``ω̇ = -q̂ f R e₃ / (m_Q l)`` and, with
        ``m_L``, forms the ``m_Q + m_L`` denominator of the payload
        acceleration.
        """
        return self._mass

    @mass.setter
    def mass(self, value):
        check_positive(value, "mass", warn_below=MIN_PHYSICAL_MASS)
        self._mass = float(value)

    @property
    def inertia(self):
        return self._inertia

    @inertia.setter
    def inertia(self, value):
        value = np.asarray(value, dtype=float)
        if value.ndim == 1:
            value = np.diag(value)
        if value.shape != (3, 3):
            raise ValueError(f"inertia must be (3,) or (3, 3); got {value.shape}")
        if not np.allclose(value, value.T):
            raise ValueError(f"inertia must be symmetric; got {value.tolist()}")
        if not np.all(np.linalg.eigvalsh(value) > 0.0):
            raise ValueError(f"inertia must be positive definite; got {value.tolist()}")
        self._inertia = value
        self._inertia_inv = np.linalg.inv(value)

    @property
    def inertia_inv(self):
        return self._inertia_inv

    @property
    def payload_mass(self) -> float:
        """Payload mass ``m_L``. Must be strictly positive.

        At ``m_L ≤ 0`` the ``m_Q + m_L`` denominator of the payload
        acceleration collapses — exactly zero at ``m_L = -m_Q``, and sign
        flipped below it — so the integrator either divides by zero or
        integrates dynamics that describe nothing.
        """
        return self._payload_mass

    @payload_mass.setter
    def payload_mass(self, value: float):
        check_positive(value, "payload_mass", warn_below=MIN_PHYSICAL_MASS)
        self._payload_mass = float(value)

    @property
    def cable_mass(self) -> float:
        """Mass of the cable itself, in kg: ``0.0``, the model's cable is massless.

        The default payload controller adds it to :attr:`payload_mass`
        (:meth:`~udaan.control.quadrotor_cspayload.QuadCSPayloadController.sync_from_model`
        does the same).
        """
        return 0.0

    @property
    def cable_length(self) -> float:
        """Cable length ``l``. Must be strictly positive.

        A zero length divides by zero in ``ω̇`` and leaves the quadrotor and
        payload at the same point, where the cable direction is undefined.
        """
        return self._cable_length

    @cable_length.setter
    def cable_length(self, value: float):
        check_positive(value, "cable_length")
        self._cable_length = float(value)

    # ─── Dynamics ──────────────────────────────────────────────────

    def _zoh(self, thrust, torque, h):
        """Zero-order hold Euler integration for one substep."""
        dq = self.state.dq()
        q = np.asarray(self.state.cable_attitude)

        mQ = self._mass
        mL = self._payload_mass
        l = self._cable_length

        fRe3 = thrust * self.state.orientation @ self._e3
        payload_accel = -self._ge3 + ((np.dot(q, fRe3) - mQ * l * np.dot(dq, dq)) / (mQ + mL)) * q

        # payload position dynamics
        self.state.payload_position += self.state.payload_velocity * h + 0.5 * payload_accel * h**2
        self.state.payload_velocity += payload_accel * h

        # cable attitude dynamics
        domega = -np.cross(q, fRe3) / (mQ * l)
        self.state.cable_attitude, self.state.cable_angular_velocity = zoh_step(
            self.state.cable_attitude, self.state.cable_angular_velocity, domega, h
        )

        # quadrotor attitude dynamics
        self.state.orientation = self.state.orientation.step(self.state.angular_velocity * h)
        dOmega = self._inertia_inv @ (
            torque
            - np.cross(
                self.state.angular_velocity,
                self._inertia @ self.state.angular_velocity,
            )
        )
        self.state.angular_velocity += TSO3(dOmega * h)

        # quadrotor position & velocity derived from payload + cable
        self.state.position = self.state.payload_position - l * np.asarray(
            self.state.cable_attitude
        )
        self.state.velocity = self.state.payload_velocity - l * self.state.dq()

    # ─── Allocation ────────────────────────────────────────────────

    def _compute_allocation_matrix(self):
        """The M1 rotor layout's allocation matrix, which both M2 MJCFs' rotor sites encode."""
        self._alloc = control.quadrotor.quadrotor_allocation_matrix(
            arm_length=DEFAULT_ARM_LENGTH,
            rotor_angles=DEFAULT_ROTOR_ANGLES,
            spin_directions=DEFAULT_SPIN_DIRECTIONS,
            moment_ratio=DEFAULT_TORQUE_CONSTANT / DEFAULT_FORCE_CONSTANT,
        )
        self._alloc_inv = np.linalg.pinv(self._alloc)

    @property
    def allocation_matrix(self):
        """The ``(4, 4)`` matrix taking rotor forces to ``[f, Mx, My, Mz]`` (a copy)."""
        return self._alloc.copy()

    def _wrench_to_propforces(self, wrench):
        return self._alloc_inv @ wrench

    def _propforces_to_wrench(self, prop_forces):
        return self._alloc @ prop_forces

    # ─── Input repackaging ─────────────────────────────────────────

    def _attitude_input(self, thrust_force, desired_att=None):
        """Close the attitude loop on ``thrust_force``, in this input path's representation.

        A wrench ``[f, Mx, My, Mz]`` on the WRENCH path, rotor forces on the
        PROP_FORCES path.
        """
        f, M = self._att_controller.compute(
            self.t,
            (self.state.orientation, self.state.angular_velocity),
            thrust_force,
            desired_att=desired_att,
        )
        u = np.array([f, *M])
        if self._input_type == InputType.PROP_FORCES:
            u = self._wrench_to_propforces(u)
        return u

    def _repackage_input(self, u, desired_att=None):
        """Convert user input to wrench [thrust, Mx, My, Mz]."""
        if isinstance(u, _ThrustCommand):
            # simulate()'s WRENCH / PROP_FORCES input: close the attitude loop
            # now, then carry on as that path's wrench or rotor forces.
            u = self._attitude_input(u.force, desired_att=desired_att)
        if self._input_type == InputType.ACCELERATION:
            # u is the desired thrust-force vector (3D) → attitude ctrl → wrench
            f, M = self._att_controller.compute(
                self.t,
                (self.state.orientation, self.state.angular_velocity),
                u,
                desired_att=desired_att,
            )
            wrench = np.array([f, *M])
        elif self._input_type == InputType.PROP_FORCES:
            wrench = self._propforces_to_wrench(u)
        else:
            wrench = np.array(u, dtype=float)

        # Clip to actuator limits
        wrench[0] = np.clip(wrench[0], self._min_thrust, self._max_thrust)
        wrench[1:] = np.clip(wrench[1:], self._min_torque, self._max_torque)

        # Convert to prop forces if needed
        if self._force_type == ForceType.PROP_FORCES:
            props = self._wrench_to_propforces(wrench)
            props = np.clip(props, self._prop_min_force, self._prop_max_force)
            wrench = self._propforces_to_wrench(props)

        self._last_wrench = wrench
        return wrench

    # ─── Step / Reset / Simulate ───────────────────────────────────

    def step(self, u, desired_att=None):
        """One outer step of ``dt``.

        The input is repackaged — which, on the ACCELERATION path and for
        :meth:`simulate`'s WRENCH and PROP_FORCES input, closes the attitude
        loop — :attr:`attitude_rate_hz` ``· dt`` times, each wrench held over
        an equal share of the ``N`` substeps. ``u`` and ``desired_att`` are
        held for the whole step.
        """
        hold = self._inner_loop_steps // self._attitude_updates
        for _ in range(self._attitude_updates):
            wrench = self._repackage_input(u, desired_att=desired_att)
            thrust = wrench[0]
            torque = wrench[1:]
            for _ in range(hold):
                self._zoh(thrust, torque, self.h)
            self._advance_clock(hold)

    def _advance_clock(self, substeps: int) -> None:
        """Advance :attr:`t` by ``substeps`` physics substeps, without accumulating round-off.

        The clock counts whole substeps since :meth:`reset` and divides by the
        physics rate, so ``t`` after ``k`` steps is the closest float to
        ``k · dt`` at every valid rate.
        """
        self._substeps_taken += substeps
        self.t = self._substeps_taken / self._physics_rate_hz

    def reset(self, **kwargs):
        """Reset time, state and any stateful controller, then apply overrides.

        Recognised kwargs: ``position``, ``velocity``, ``orientation``,
        ``angular_velocity``, ``payload_position``, ``payload_velocity``,
        ``cable_attitude``, ``cable_angular_velocity``. The quadrotor and the
        payload are tied by the cable, so giving one position derives the other.
        The velocities follow the same rule as the MuJoCo backends: the
        airframe's velocity is ``velocity`` (zero if not given); the payload's
        is ``payload_velocity`` when given, and the cable rate is then
        ``ω = q × (v_L − v_Q)/ℓ``; otherwise the payload's velocity is
        ``v_Q + ℓ (ω × q)`` from ``cable_angular_velocity`` (``q̇ = ω × q``).

        Raises:
            ValueError: for any other keyword, and for ``payload_velocity``
                together with ``cable_angular_velocity``, which over-determine
                the payload's velocity. An unrecognised keyword is an error
                rather than a no-op so that a stale name cannot silently run a
                different initial condition.
        """
        check_reset_kwargs(self, kwargs, _SUPPORTED_RESET_KWARGS)
        self._check_velocity_kwargs(kwargs)
        self.t = 0.0
        self._substeps_taken = 0
        for controller in (self._payload_controller, self._att_controller):
            reset = getattr(controller, "reset", None)
            if callable(reset):
                reset()
        self.state.reset()
        for key in _SUPPORTED_RESET_KWARGS:
            if key in kwargs:
                v = kwargs[key]
                # Copy ndarray inputs: _zoh() does in-place += on payload_position
                # and payload_velocity, which would otherwise silently mutate
                # caller arrays across repeated sims (e.g. shared EVAL_STARTS).
                if isinstance(v, np.ndarray):
                    v = v.copy()
                setattr(self.state, key, v)

        # Keep quadrotor and payload positions consistent via cable geometry.
        # If only one is given, derive the other.
        if "position" in kwargs and "payload_position" not in kwargs:
            self.state.payload_position = kwargs["position"] + self._cable_length * np.asarray(
                self.state.cable_attitude
            )
        elif "payload_position" in kwargs and "position" not in kwargs:
            self.state.position = kwargs["payload_position"] - self._cable_length * np.asarray(
                self.state.cable_attitude
            )

        # Velocities: the airframe's is given, the payload's and the cable
        # rate's are tied by v_L = v_Q + ℓ (ω × q).
        q = np.asarray(self.state.cable_attitude)
        v_Q = np.asarray(self.state.velocity, dtype=float)
        if "payload_velocity" in kwargs:
            v_L = np.asarray(self.state.payload_velocity, dtype=float)
            self.state.cable_angular_velocity = TS2(np.cross(q, v_L - v_Q) / self._cable_length)
        else:
            omega = np.asarray(self.state.cable_angular_velocity)
            self.state.payload_velocity = v_Q + self._cable_length * np.cross(omega, q)

    @staticmethod
    def _check_velocity_kwargs(kwargs):
        """Raise ``ValueError`` for ``payload_velocity`` with ``cable_angular_velocity``."""
        if "payload_velocity" in kwargs and "cable_angular_velocity" in kwargs:
            raise ValueError(
                "the payload's velocity follows from the airframe's and the cable rate: pass "
                "payload_velocity or cable_angular_velocity, not both"
            )

    @property
    def feedforward_drops(self) -> int:
        """Steps of the last :meth:`simulate` that flew without the airframe feed-forward.

        A step drops it where the flat map refuses the reference sample (the
        payload in free fall, a net thrust of zero) or recovers an airframe
        that is not upright (:attr:`~udaan.flatness.QuadrotorCsPayloadRefState.upright`).
        Reset by :meth:`simulate`.
        """
        return self._feedforward_drops

    def _reference_attitude(self, jet_fn, flat_map):
        """``(R, Ω, Ω̇)`` the flat map recovers from the payload jet at :attr:`t`.

        ``None`` where the map is singular (the payload in free fall, so the
        cable goes slack and ``q`` is undefined, or a net thrust of zero) and
        where the recovered airframe is not upright.
        """
        flats = QuadrotorCsPayloadFlats(
            x_L=Jet(np.asarray(jet_fn(self.t, 6), dtype=float)),  # order-6, dim-3
            psi=yaw_jet(self._yaw_setpoint, self.t),
        )
        try:
            ref, _ = flat_map.recover(flats)
        except ValueError:
            return None
        if not ref.upright:
            return None
        return (ref.orientation, ref.angular_velocity, ref.angular_acceleration)

    def simulate(self, tf, **kwargs):
        """Run the closed loop to time ``tf`` on the payload controller's setpoint.

        Every input path flies the same law: the payload controller's
        thrust-force vector once per :meth:`step`, and the attitude loop
        closed on it inside :meth:`step` at :attr:`attitude_rate_hz`.

        Attitude feedforward: when the payload controller's setpoint is a bound
        method of a trajectory that also defines ``get_jet(t, order)``, the
        payload position jet through order 6 goes through
        :class:`~udaan.flatness.QuadrotorCsPayload`, built from this model's
        parameters, with :attr:`yaw_setpoint` as ψ. The recovered
        ``(R, Ω, Ω̇)`` is passed to :meth:`step` as ``desired_att``: the thrust
        command keeps ``b₃``, the reference supplies the heading and the rate
        feedforward. A trajectory without ``get_jet`` flies without it.

        Where the reference needs an airframe that is not upright, or the map
        refuses the sample, the step flies on feedback alone: it is counted in
        :attr:`feedforward_drops`, and one warning at the end of the run gives
        the intervals. :meth:`~udaan.flatness.QuadrotorCsPayload.check_reference`
        finds them before flight.
        """
        self.reset(**kwargs)
        self._feedforward_drops = 0

        setpoint_fn = self._payload_controller.setpoint
        owner = getattr(setpoint_fn, "__self__", None)
        jet_fn = getattr(owner, "get_jet", None) if owner is not None else None
        flat_map = QuadrotorCsPayload.from_model(self) if jet_fn is not None else None
        sample_times: list[float] = []
        sample_upright: list[bool] = []

        start_t = time.time_ns()
        while self.t < tf:
            desired_att = None
            if jet_fn is not None and flat_map is not None:
                desired_att = self._reference_attitude(jet_fn, flat_map)
                sample_times.append(self.t)
                sample_upright.append(desired_att is not None)
                if desired_att is None:
                    self._feedforward_drops += 1

            thrust_force = self._payload_controller.compute(self.t, self.state)
            # Every path closes the attitude loop inside step(). ACCELERATION
            # takes the thrust-force vector as is; WRENCH and PROP_FORCES take
            # it wrapped, and step() hands the attitude loop's output on as a
            # wrench or, allocated, as rotor forces.
            if self._input_type == InputType.ACCELERATION:
                u = thrust_force
            else:
                u = _ThrustCommand(np.asarray(thrust_force, dtype=float))

            self.step(u, desired_att=desired_att)

            if self.verbose >= 2:
                self._log_state()

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

    def _log_state(self):
        """Log the state and the wrench last applied (never re-runs a controller)."""
        from ...manif import Rot2Eul

        rpy = np.degrees(Rot2Eul(np.asarray(self.state.orientation)))
        wrench = self._last_wrench
        _logger.debug(
            "t=%.4f pos=%s vel=%s rpy=%s Om=%s q=%s dq=%s f=%.4f M=%s",
            self.t,
            np.array2string(self.state.position, precision=4),
            np.array2string(self.state.velocity, precision=4),
            np.array2string(rpy, precision=2),
            np.array2string(np.asarray(self.state.angular_velocity), precision=4),
            np.array2string(np.asarray(self.state.cable_attitude), precision=4),
            np.array2string(self.state.dq(), precision=4),
            wrench[0],
            np.array2string(wrench[1:], precision=4),
        )
