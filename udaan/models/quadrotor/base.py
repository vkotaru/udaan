"""QuadrotorBase — pure dynamics and state, no rendering.

Rigid body quadrotor dynamics with ZOH (zero-order hold) Euler integration:
a 5 ms outer (position) loop of ``N = physics_rate_hz · dt`` substeps, by
default 8 substeps of 0.625 ms (1600 Hz). The attitude loop closes at 800 Hz by
default, four times per outer step, on every input path (see
:class:`QuadrotorBase`).
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
    PHYSICS_RATE_RANGE_HZ,
)
from ...core.types import ForceType, InputType, Vec3
from ...core.validation import check_positive
from ...flatness import Jet, Quadrotor, QuadrotorFlats, yaw_jet
from ...manif import SO3, TSO3
from ...utils.logging import get_logger
from .._kwargs import check_reset_kwargs

_logger = get_logger(__name__)

#: The initial-condition overrides :meth:`QuadrotorBase.reset` acts on. Anything
#: else is rejected rather than silently discarded — see :mod:`udaan.models._kwargs`.
_SUPPORTED_RESET_KWARGS = frozenset({"position", "velocity", "orientation", "angular_velocity"})

#: The rotor keywords :class:`QuadrotorBase` takes, and their defaults. On
#: :class:`~udaan.models.quadrotor.QuadrotorMujoco` the MJCF supplies every one
#: of them, and an explicitly passed value must agree with it.
ROTOR_PARAMETER_DEFAULTS = {
    "arm_length": DEFAULT_ARM_LENGTH,
    "rotor_angles": DEFAULT_ROTOR_ANGLES,
    "spin_directions": DEFAULT_SPIN_DIRECTIONS,
    "force_constant": DEFAULT_FORCE_CONSTANT,
    "torque_constant": DEFAULT_TORQUE_CONSTANT,
    "prop_min_force": DEFAULT_PROP_MIN_FORCE,
    "prop_max_force": DEFAULT_PROP_MAX_FORCE,
}


@dataclass
class QuadrotorState:
    """Quadrotor state: position, velocity, orientation, angular velocity."""

    position: Vec3 = field(default_factory=lambda: np.zeros(3))
    velocity: Vec3 = field(default_factory=lambda: np.zeros(3))
    orientation: SO3 = field(default_factory=SO3)
    angular_velocity: TSO3 = field(default_factory=TSO3)

    def reset(self):
        QuadrotorState.__init__(self)


@dataclass(frozen=True)
class _ThrustCommand:
    """A thrust-force vector whose attitude loop :meth:`QuadrotorBase.step` closes.

    :meth:`QuadrotorBase.simulate` hands this to ``step()`` on the WRENCH and
    PROP_FORCES paths, so that those paths close the attitude loop inside the
    step, at :attr:`QuadrotorBase.attitude_rate_hz`, exactly as the
    ACCELERATION path does — and then hand the result on in their own
    representation (a wrench, or rotor forces).
    """

    force: np.ndarray


class QuadrotorBase:
    """Base quadrotor model — pure dynamics, no rendering.

    Supports multiple input types (acceleration, wrench, propeller forces)
    and force types (wrench, propeller forces). Controllers are pluggable
    via position_controller and attitude_controller properties.

    Two loops, at two rates. The **position** loop runs once per :meth:`step`,
    at ``1/dt``: it computes the thrust vector, whose direction is the desired
    ``b₃`` and so the desired attitude. The **attitude** loop runs inside
    :meth:`step` at ``attitude_rate_hz``: it turns the (held) thrust vector and
    attitude reference into a thrust and torques. The physics integrates
    underneath both at ``physics_rate_hz``.

    Keyword Args:
        dt: the position-loop period, in s (default 5 ms, 200 Hz): what one
            :meth:`step` advances. It is validated with the two rates below:
            ``physics_rate_hz · dt`` must be a whole number of substeps and the
            attitude updates per step must divide it, else ``ValueError``.
        attitude_rate_hz: rate at which the attitude loop closes, in Hz
            (default :data:`~udaan.core.defaults.DEFAULT_ATTITUDE_RATE_HZ`,
            800). ``None`` is once per outer step, ``1/dt``. It is the same on
            every input path: inside :meth:`step` for ACCELERATION, and inside
            :meth:`simulate`'s call to :meth:`step` for WRENCH and PROP_FORCES.
            It must be ``k/dt`` for a whole ``k`` that divides the ``N``
            physics substeps per step (see ``physics_rate_hz``); anything else
            raises ``ValueError`` — so a ``physics_rate_hz`` whose ``N`` the
            default four updates do not divide (1000 Hz: ``N = 5``) needs an
            ``attitude_rate_hz`` too. The position loop and the feedforward
            reference stay at ``1/dt`` and are held across the updates.
        physics_rate_hz: rate at which the physics integrates, in Hz (default
            :data:`~udaan.core.defaults.DEFAULT_PHYSICS_RATE_HZ`, 1600). Each
            ``step()`` takes ``N = physics_rate_hz · dt`` substeps of
            ``h = dt/N``: ZOH substeps here, MuJoCo steps on
            :class:`~udaan.models.quadrotor.QuadrotorMujoco`, which writes ``h``
            into the MuJoCo model's timestep. It must lie in
            :data:`~udaan.core.defaults.PHYSICS_RATE_RANGE_HZ` (500–2000 Hz)
            and make ``N`` a whole number, else ``ValueError`` (at the default
            5 ms ``dt``: 600, 800, …, 2000 Hz; 500 Hz needs a ``dt`` that is a
            multiple of 2 ms). There is no rounding: ``h · N = dt`` exactly.

    Rotor keywords (they build the allocation matrix
    :attr:`allocation_matrix`, see
    :func:`~udaan.control.quadrotor.quadrotor_allocation_matrix`; defaults from
    :mod:`udaan.core.defaults`, the airframe of ``quadrotor_mj.xml``):

    - ``arm_length``: ``ℓ``, in m (0.2).
    - ``rotor_angles``: four ``θᵢ`` about ``b₃`` from ``b₁``, in rad
      (``π/4, 3π/4, 5π/4, 7π/4``, an X layout).
    - ``spin_directions``: four ``dᵢ = ±1``, the sign of each rotor's yaw
      moment (``−1, 1, −1, 1``).
    - ``force_constant``, ``torque_constant``: ``k_F`` and ``k_M``; only their
      ratio ``c = k_M/k_F`` enters the matrix.
    - ``prop_min_force``, ``prop_max_force``: per-rotor force limits, in N
      (0 and 10), applied under ``force="prop_forces"``.

    Each is validated here: a non-positive or non-finite length or constant, a
    spin direction other than ±1, limits with ``min >= max``, or a layout whose
    matrix cannot produce every wrench (rank < 4) raises ``ValueError``. On
    :class:`~udaan.models.quadrotor.QuadrotorMujoco` the MJCF is the source of
    every one of them, and a passed value that disagrees with it raises.
    """

    INPUT_TYPE = InputType
    FORCE_TYPE = ForceType

    def __init__(self, **kwargs):
        self.state = QuadrotorState()
        self.t = 0.0
        # Physics substeps taken since reset(); the clock is derived from it.
        self._substeps_taken = 0
        self.verbose = kwargs.get("verbose", 0)

        # System parameters
        self._mass = DEFAULT_M1_QUAD_MASS
        self._inertia = DEFAULT_M1_QUAD_INERTIA.copy()
        self._inertia_inv = np.linalg.inv(self._inertia)

        # Physical constants
        self._g = GRAVITY
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._e3 = np.array([0.0, 0.0, 1.0])

        # Actuator limits
        self._min_thrust = DEFAULT_MIN_THRUST
        self._max_thrust = DEFAULT_MAX_THRUST
        self._min_torque = DEFAULT_MIN_TORQUE.copy()
        self._max_torque = DEFAULT_MAX_TORQUE.copy()

        # Integration: a 5 ms outer loop of N substeps of h = dt/N
        # (N = physics_rate_hz · dt; 8 substeps of 0.625 ms at the default 1600 Hz).
        if "inner_loop_steps" in kwargs:
            raise TypeError(
                "inner_loop_steps is not a keyword; the substeps are set by physics_rate_hz: "
                "N = physics_rate_hz · dt substeps per step (800 Hz is 4 substeps at a 5 ms dt)"
            )
        self.dt = kwargs.get("dt", 1.0 / 200.0)
        self._physics_rate_hz = kwargs.get("physics_rate_hz", DEFAULT_PHYSICS_RATE_HZ)
        self._inner_loop_steps = self._physics_steps_per_step(self._physics_rate_hz)
        self.h = self.dt / self._inner_loop_steps
        self._attitude_rate_is_default = "attitude_rate_hz" not in kwargs
        self._attitude_rate_hz = kwargs.get("attitude_rate_hz", DEFAULT_ATTITUDE_RATE_HZ)
        self._attitude_updates = self._attitude_updates_per_step(self._inner_loop_steps)
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

        # Rotor layout, constants and limits → allocation matrix
        self._configure_rotors(
            **{key: kwargs.get(key, value) for key, value in ROTOR_PARAMETER_DEFAULTS.items()}
        )

        # Feedforward yaw reference; None is ψ ≡ 0.
        self._yaw_setpoint = None

        # Default controllers
        self._init_default_controllers()

    # ─── Controllers ───────────────────────────────────────────────

    def _init_default_controllers(self):
        self._att_controller = control.quadrotor.GeometricAttitudeController(inertia=self._inertia)
        self._pos_controller = control.quadrotor.PositionPDController(mass=self._mass)

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
    def yaw_setpoint(self):
        r"""Yaw reference for the flatness feedforward: ``ψ(t) -> (ψ, ψ̇, ψ̈)``.

        Yaw is the quadrotor's second flat output, and it is the *only* channel
        the reference attitude adds on top of the position loop — the thrust
        command already fixes ``b₃``, so ``ψ`` is what
        :meth:`~udaan.control.quadrotor.GeometricAttitudeController.compute`
        composes in. Without a yaw setpoint the feedforward runs at ψ ≡ 0 and
        the composition is close to a no-op.

        ``None`` (the default) means ψ ≡ 0, so binding nothing changes nothing.
        Set it beside the position trajectory::

            mdl.position_controller.setpoint = traj.get
            mdl.yaw_setpoint = lambda t: (0.5 * t, 0.5, 0.0)

        It is read only on the feedforward path — that is, only when the bound
        trajectory exposes ``get_full`` (see :meth:`simulate`).
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

    def _physics_steps_per_step(self, rate) -> int:
        """``N = rate · dt`` for a valid ``physics_rate_hz``.

        Raises:
            ValueError: unless the rate lies in
                :data:`~udaan.core.defaults.PHYSICS_RATE_RANGE_HZ` and ``rate · dt``
                is a whole number — so that ``N`` substeps of ``dt/N`` are exactly
                ``dt``, with nothing rounded.
        """
        lo, hi = PHYSICS_RATE_RANGE_HZ
        check_positive(self.dt, "dt")
        check_positive(rate, "physics_rate_hz")
        n = rate * self.dt
        if lo <= rate <= hi and abs(n - round(n)) <= 1e-9 * max(1.0, n):
            return int(round(n))
        whole = range(int(np.ceil(lo * self.dt - 1e-9)), int(np.floor(hi * self.dt + 1e-9)) + 1)
        valid = ", ".join(f"{k / self.dt:g}" for k in whole if k >= 1) or "none"
        raise ValueError(
            f"physics_rate_hz = {rate:g} Hz must lie in [{lo:g}, {hi:g}] Hz and be a whole "
            f"number of substeps per dt = {self.dt:g} s step; valid rates at this dt are "
            f"{valid} Hz. A rate r needs a dt that is a multiple of 1/r (500 Hz: of 2 ms)."
        )

    def _attitude_updates_per_step(self, substeps: int) -> int:
        """Attitude updates per outer step for ``self._attitude_rate_hz`` over ``substeps``.

        Raises:
            ValueError: unless the rate is ``k/dt`` for a whole ``k`` dividing
                ``substeps`` — so that every update is held for the same whole
                number of substeps and the clock never drifts.
        """
        rate = self._attitude_rate_hz
        if rate is None:
            return 1
        check_positive(rate, "attitude_rate_hz")
        k = rate * self.dt
        divisors = [d for d in range(1, substeps + 1) if substeps % d == 0]
        if abs(k - round(k)) > 1e-9 * max(1.0, k) or round(k) not in divisors:
            valid = ", ".join(f"{d / self.dt:g}" for d in divisors)
            raise ValueError(
                f"attitude_rate_hz = {rate:g} Hz is not a whole number of updates per "
                f"{self.dt:g} s step dividing its {substeps} substeps; valid rates are "
                f"{valid} Hz (or None for once per step)."
                + (
                    f" {rate:g} Hz is the default (DEFAULT_ATTITUDE_RATE_HZ): with this "
                    "physics_rate_hz or dt, pass attitude_rate_hz explicitly."
                    if self._attitude_rate_is_default
                    else ""
                )
            )
        return int(round(k))

    # ─── Properties ────────────────────────────────────────────────

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        self._mass = value

    @property
    def inertia(self):
        return self._inertia

    @inertia.setter
    def inertia(self, value):
        if value.ndim == 1:
            value = np.diag(value)
        self._inertia = value
        self._inertia_inv = np.linalg.inv(value)

    # ─── Dynamics ──────────────────────────────────────────────────

    def _zoh(self, thrust, torque, h):
        """Zero-order hold Euler integration for one substep."""
        accel = -self._ge3 + (thrust / self._mass) * self.state.orientation @ self._e3
        self.state.position += self.state.velocity * h + 0.5 * accel * h**2
        self.state.velocity += accel * h
        self.state.orientation = self.state.orientation.step(self.state.angular_velocity * h)
        dOmega = self._inertia_inv @ (
            torque
            - np.cross(self.state.angular_velocity, self._inertia @ self.state.angular_velocity)
        )
        self.state.angular_velocity += TSO3(dOmega * h)

    # ─── Allocation ────────────────────────────────────────────────

    def _configure_rotors(
        self,
        arm_length,
        rotor_angles,
        spin_directions,
        force_constant,
        torque_constant,
        prop_min_force,
        prop_max_force,
    ):
        """Validate and store the rotor parameters, then rebuild the allocation matrix."""
        check_positive(arm_length, "arm_length")
        check_positive(force_constant, "force_constant")
        check_positive(torque_constant, "torque_constant")
        angles = np.array(rotor_angles, dtype=float)
        if angles.shape != (4,) or not np.all(np.isfinite(angles)):
            raise ValueError(f"rotor_angles must be four finite angles; got {angles.tolist()}")
        spins = np.array(spin_directions, dtype=float)
        if spins.shape != (4,) or not np.all(np.abs(spins) == 1.0):
            raise ValueError(
                f"spin_directions must be four values of +1 or -1; got {spins.tolist()}"
            )
        limits = np.array([prop_min_force, prop_max_force], dtype=float)
        if not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
            raise ValueError(
                "prop_min_force must be finite and below prop_max_force; got "
                f"{prop_min_force} and {prop_max_force}"
            )
        self._arm_length = float(arm_length)
        self._rotor_angles = angles
        self._spin_directions = spins
        self._force_constant = float(force_constant)
        self._torque_constant = float(torque_constant)
        self._prop_min_force = float(prop_min_force)
        self._prop_max_force = float(prop_max_force)
        self._compute_allocation_matrix()
        if np.linalg.matrix_rank(self._alloc) < 4:
            raise ValueError(
                f"rotor_angles {angles.tolist()} with spin_directions {spins.tolist()} give "
                "an allocation matrix of rank < 4: the rotors cannot produce every wrench"
            )

    def _compute_allocation_matrix(self):
        self._alloc = control.quadrotor.quadrotor_allocation_matrix(
            arm_length=self._arm_length,
            rotor_angles=self._rotor_angles,
            spin_directions=self._spin_directions,
            moment_ratio=self._torque_constant / self._force_constant,
        )
        self._alloc_inv = np.linalg.pinv(self._alloc)

    @property
    def allocation_matrix(self):
        """The ``(4, 4)`` matrix taking rotor forces to ``[f, Mx, My, Mz]`` (a copy)."""
        return self._alloc.copy()

    @property
    def rotor_parameters(self):
        """The rotor keywords this model was built with (MJCF-read on MuJoCo), as a dict."""
        return {
            "arm_length": self._arm_length,
            "rotor_angles": self._rotor_angles.copy(),
            "spin_directions": self._spin_directions.copy(),
            "force_constant": self._force_constant,
            "torque_constant": self._torque_constant,
            "prop_min_force": self._prop_min_force,
            "prop_max_force": self._prop_max_force,
        }

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
            # u is desired thrust force vector (3D) → attitude ctrl → wrench
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
            wrench = u

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

        The input is repackaged — which, on the ACCELERATION path, closes the
        attitude loop — :attr:`attitude_rate_hz` ``· dt`` times, each wrench
        held over an equal share of the substeps. ``u`` and ``desired_att``
        are held for the whole step.
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
        ``k · dt`` at every valid rate, rather than a running sum of ``h``, which
        drifts when ``h`` is not a binary fraction.
        """
        self._substeps_taken += substeps
        self.t = self._substeps_taken / self._physics_rate_hz

    def reset(self, **kwargs):
        """Reset time, state and any stateful controller, then apply overrides.

        A controller exposing ``reset()`` (the L1 controllers do) is reset too.

        Recognised kwargs: ``position``, ``velocity``, ``orientation``,
        ``angular_velocity``.

        Raises:
            ValueError: for any other keyword. An unrecognised keyword is an
                error rather than a no-op so that a stale name cannot silently
                run a different initial condition.
        """
        check_reset_kwargs(self, kwargs, _SUPPORTED_RESET_KWARGS)
        self.t = 0.0
        self._substeps_taken = 0
        # Controllers with internal state (the L1 predictors and estimates)
        # start each run fresh, so a second `simulate()` does not begin from the
        # first run's end state.
        for controller in (self._pos_controller, self._att_controller):
            reset = getattr(controller, "reset", None)
            if callable(reset):
                reset()
        self.state.reset()
        for key in _SUPPORTED_RESET_KWARGS:
            if key in kwargs:
                setattr(self.state, key, kwargs[key])

    def simulate(self, tf, **kwargs):
        """Run closed-loop simulation to time tf."""
        self.reset(**kwargs)

        # Check if trajectory provides higher derivatives for feedforward
        setpoint_fn = self._pos_controller.setpoint
        has_flat = hasattr(setpoint_fn, "__self__") and hasattr(setpoint_fn.__self__, "get_full")
        flat_fn = setpoint_fn.__self__.get_full if has_flat else None
        flat_map = Quadrotor.from_model(self) if flat_fn is not None else None
        # A trajectory that knows its own attitude over a segment (the flip's
        # fixed-axis rotation) supplies it there and returns None elsewhere.
        attitude_fn = (
            getattr(setpoint_fn.__self__, "reference_attitude", None) if has_flat else None
        )

        start_t = time.time_ns()
        while self.t < tf:
            # Differential-flatness attitude feedforward from the flat output,
            # shared by all three input paths so that they fly the same law.
            # `flat_map` is non-None exactly when `flat_fn` is; both are tested
            # so the checker can see it too.
            desired_att = None
            own = attitude_fn(self.t) if attitude_fn is not None else None
            if own is not None:
                desired_att = (SO3(own[0]), TSO3(own[1]), np.asarray(own[2]))
            elif flat_map is not None and flat_fn is not None:
                pos, vel, acc, jerk, snap = flat_fn(self.t)
                flats = QuadrotorFlats(
                    x=Jet(np.array([pos, vel, acc, jerk, snap])),  # order-4, dim-3
                    psi=yaw_jet(self._yaw_setpoint, self.t),
                )
                ref, _ = flat_map.recover(flats)
                desired_att = (ref.orientation, ref.angular_velocity, ref.angular_acceleration)

            # The orientation rides along as an optional third element: the PD
            # controller ignores it, and PositionL1Controller's predictor needs
            # it for the thrust the attitude actually delivers.
            thrust_force = self._pos_controller.compute(
                self.t, (self.state.position, self.state.velocity, self.state.orientation)
            )
            # Every path closes the attitude loop inside step(), at
            # attitude_rate_hz on every backend (800 Hz by default).
            # ACCELERATION takes the thrust-force vector as is; WRENCH and
            # PROP_FORCES take it wrapped, and step() hands the attitude loop's
            # output on as a wrench or, allocated, as rotor forces. All three
            # therefore fly this model's position loop, `yaw_setpoint` and
            # feedforward.
            if self._input_type == InputType.ACCELERATION:
                u = thrust_force
            else:
                u = _ThrustCommand(np.asarray(thrust_force, dtype=float))

            self.step(u, desired_att=desired_att)

            if self.verbose >= 2:
                self._log_state()

        elapsed = (time.time_ns() - start_t) * 1e-9
        _logger.debug("Simulated %.2fs in %.3fs wall time", self.t, elapsed)

    def _log_state(self):
        """Log the state and the wrench last applied (never re-runs a controller)."""
        from ...manif import Rot2Eul

        rpy = np.degrees(Rot2Eul(np.asarray(self.state.orientation)))
        wrench = self._last_wrench
        _logger.debug(
            "t=%.4f pos=%s vel=%s rpy=%s Om=%s f=%.4f M=%s",
            self.t,
            np.array2string(self.state.position, precision=4),
            np.array2string(self.state.velocity, precision=4),
            np.array2string(rpy, precision=2),
            np.array2string(np.asarray(self.state.angular_velocity), precision=4),
            wrench[0],
            np.array2string(wrench[1:], precision=4),
        )
