"""QuadrotorBase — pure dynamics and state, no rendering.

Rigid body quadrotor dynamics with ZOH (zero-order hold) Euler integration.
Matches floating_models integration structure: 5ms outer loop, 4 inner
substeps at 1.25ms for numerical stability.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from ... import control
from ...core.defaults import (
    DEFAULT_ARM_LENGTH,
    DEFAULT_FORCE_CONSTANT,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    DEFAULT_TORQUE_CONSTANT,
    GRAVITY,
)
from ...core.types import ForceType, InputType, Vec3
from ...manif import SO3, TSO3
from ...utils.logging import get_logger

_logger = get_logger(__name__)


@dataclass
class QuadrotorState:
    """Quadrotor state: position, velocity, orientation, angular velocity."""

    position: Vec3 = field(default_factory=lambda: np.zeros(3))
    velocity: Vec3 = field(default_factory=lambda: np.zeros(3))
    orientation: SO3 = field(default_factory=SO3)
    angular_velocity: TSO3 = field(default_factory=TSO3)

    def reset(self):
        self.__init__()


class QuadrotorBase:
    """Base quadrotor model — pure dynamics, no rendering.

    Supports multiple input types (acceleration, wrench, propeller forces)
    and force types (wrench, propeller forces). Controllers are pluggable
    via position_controller and attitude_controller properties.
    """

    INPUT_TYPE = InputType
    FORCE_TYPE = ForceType

    def __init__(self, **kwargs):
        self.state = QuadrotorState()
        self.t = 0.0
        self.verbose = kwargs.get("verbose", 0)

        # System parameters
        self._mass = DEFAULT_QUAD_MASS
        self._inertia = DEFAULT_QUAD_INERTIA.copy()
        self._inertia_inv = np.linalg.inv(self._inertia)

        # Physical constants
        self._g = GRAVITY
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._e3 = np.array([0.0, 0.0, 1.0])

        # Actuator limits
        self._min_thrust = 0.0
        self._max_thrust = 20.0
        self._min_torque = np.array([-5.0, -5.0, -2.0])
        self._max_torque = np.array([5.0, 5.0, 2.0])
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
        if "force" in kwargs:
            if kwargs["force"] == "prop_forces":
                self._force_type = ForceType.PROP_FORCES

        # Allocation matrix
        self._compute_allocation_matrix()

        # Default controllers
        self._init_default_controllers()

    # ─── Controllers ───────────────────────────────────────────────

    def _init_default_controllers(self):
        self._att_controller = control.quadrotor.GeometricAttitudeController(inertia=self._inertia)
        self._pos_controller = control.quadrotor.PositionPDController(mass=self._mass)
        self._prop_controller = control.quadrotor.DirectPropellerForceController(
            mass=self._mass, inertia=self._inertia
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

    def _compute_allocation_matrix(self):
        kf = DEFAULT_FORCE_CONSTANT
        km = DEFAULT_TORQUE_CONSTANT
        c = km / kf
        l = DEFAULT_ARM_LENGTH
        ang = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
        d = [-1.0, 1.0, -1.0, 1.0]

        self._alloc = np.zeros((4, 4))
        for i in range(4):
            self._alloc[0, i] = 1.0
            self._alloc[1, i] = l * np.sin(ang[i])
            self._alloc[2, i] = -l * np.cos(ang[i])
            self._alloc[3, i] = c * d[i]
        self._alloc_inv = np.linalg.pinv(self._alloc)

    def _wrench_to_propforces(self, wrench):
        return self._alloc_inv @ wrench

    def _propforces_to_wrench(self, prop_forces):
        return self._alloc @ prop_forces

    # ─── Input repackaging ─────────────────────────────────────────

    def _repackage_input(self, u, desired_att=None):
        """Convert user input to wrench [thrust, Mx, My, Mz]."""
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

        return wrench

    # ─── Step / Reset / Simulate ───────────────────────────────────

    def step(self, u, desired_att=None):
        """One outer step: repackage input, then run inner substeps."""
        wrench = self._repackage_input(u, desired_att=desired_att)
        thrust = wrench[0]
        torque = wrench[1:]
        for _ in range(self._inner_loop_steps):
            self._zoh(thrust, torque, self.h)
            self.t += self.h

    def reset(self, **kwargs):
        self.t = 0.0
        self.state.reset()
        for key in ["position", "velocity", "orientation", "angular_velocity"]:
            if key in kwargs:
                setattr(self.state, key, kwargs[key])

    def simulate(self, tf, **kwargs):
        """Run closed-loop simulation to time tf."""
        self.reset(**kwargs)

        # Check if trajectory provides higher derivatives for feedforward
        setpoint_fn = self._pos_controller.setpoint
        has_flat = hasattr(setpoint_fn, "__self__") and hasattr(setpoint_fn.__self__, "get_full")
        flat_fn = setpoint_fn.__self__.get_full if has_flat else None

        start_t = time.time_ns()
        while self.t < tf:
            desired_att = None
            if self._input_type == InputType.PROP_FORCES:
                u = self._prop_controller.compute(
                    self.t,
                    (
                        self.state.position,
                        self.state.velocity,
                        self.state.orientation,
                        self.state.angular_velocity,
                    ),
                )
            elif self._input_type == InputType.ACCELERATION:
                u = self._pos_controller.compute(self.t, (self.state.position, self.state.velocity))
                # Compute feedforward from flat output if available
                if flat_fn is not None:
                    from ...utils.flat2state_utils import flat2state

                    _p, _v, acc, jerk, snap = flat_fn(self.t)
                    Rd, Omd, dOmd, _f = flat2state(acc, jerk, snap, self._mass, self._inertia)
                    desired_att = (Rd, Omd, dOmd)
            else:
                thrust_force = self._pos_controller.compute(
                    self.t, (self.state.position, self.state.velocity)
                )
                f, M = self._att_controller.compute(
                    self.t,
                    (self.state.orientation, self.state.angular_velocity),
                    thrust_force,
                )
                u = np.array([f, *M])

            if self.verbose >= 2:
                self._log_state(u)

            self.step(u, desired_att=desired_att)

        elapsed = (time.time_ns() - start_t) * 1e-9
        _logger.debug("Simulated %.2fs in %.3fs wall time", self.t, elapsed)

    def _log_state(self, u):
        from ...manif import Rot2Eul

        rpy = np.degrees(Rot2Eul(np.asarray(self.state.orientation)))
        wrench = self._repackage_input(u)
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
