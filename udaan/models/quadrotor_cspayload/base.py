"""QuadrotorCsPayloadBase — pure dynamics and state, no rendering.

Rigid body with cable-suspended payload dynamics with ZOH (zero-order hold)
Euler integration.

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
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORCE_CONSTANT,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    DEFAULT_TORQUE_CONSTANT,
    GRAVITY,
)
from ...core.types import ForceType, InputType, Vec3
from ...manif import S2, SO3, TS2, TSO3
from ...utils.logging import get_logger

_logger = get_logger(__name__)


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
        self.__init__()


class QuadrotorCsPayloadBase:
    """Base quadrotor-cable-suspended payload model — pure dynamics, no rendering.

    Supports multiple input types (acceleration, wrench, propeller forces)
    and force types (wrench, propeller forces). Controllers are pluggable
    via position_controller and attitude_controller properties.
    """

    INPUT_TYPE = InputType
    FORCE_TYPE = ForceType

    def __init__(self, **kwargs):
        self.state = QuadrotorCsPayloadState()
        self.t = 0.0
        self.verbose = kwargs.get("verbose", 0)

        # System parameters
        self._mass = DEFAULT_QUAD_MASS
        self._inertia = DEFAULT_QUAD_INERTIA.copy()
        self._inertia_inv = np.linalg.inv(self._inertia)

        self._payload_mass = DEFAULT_PAYLOAD_MASS
        self._cable_length = DEFAULT_CABLE_LENGTH

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
        self._payload_controller = control.quadrotor_cspayload.QuadCSPayloadController()

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

    @property
    def inertia_inv(self):
        return self._inertia_inv

    @property
    def payload_mass(self) -> float:
        return self._payload_mass

    @payload_mass.setter
    def payload_mass(self, value: float):
        self._payload_mass = value

    @property
    def cable_length(self) -> float:
        return self._cable_length

    @cable_length.setter
    def cable_length(self, value: float):
        self._cable_length = value

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
        self.state.cable_attitude = self.state.cable_attitude.step(
            np.asarray(self.state.cable_angular_velocity) * h
        )
        domega = -np.cross(q, fRe3) / (mQ * l)
        self.state.cable_angular_velocity += TS2(domega * h)

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
        for key in [
            "position",
            "velocity",
            "orientation",
            "angular_velocity",
            "payload_position",
            "payload_velocity",
            "cable_attitude",
            "cable_angular_velocity",
        ]:
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

    def simulate(self, tf, **kwargs):
        """Run closed-loop simulation to time tf."""
        self.reset(**kwargs)

        start_t = time.time_ns()
        while self.t < tf:
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
            else:
                # Payload controller returns thrust force vector (3D)
                u = self._payload_controller.compute(self.t, self.state)

            if self.verbose >= 2:
                self._log_state(u)

            self.step(u)

        elapsed = (time.time_ns() - start_t) * 1e-9
        _logger.debug("Simulated %.2fs in %.3fs wall time", self.t, elapsed)

    def _log_state(self, u):
        from ...manif import Rot2Eul

        rpy = np.degrees(Rot2Eul(np.asarray(self.state.orientation)))
        wrench = self._repackage_input(u)
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
