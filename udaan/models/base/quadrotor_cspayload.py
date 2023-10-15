from scipy.linalg import block_diag as blkdiag
import numpy as np
import scipy as sp
import time
import enum
from scipy.linalg import expm

from . import BaseModel
from ... import control
from ... import utils


class QuadrotorCSPayload(BaseModel):
    """Quadrotor with cable suspended paylaod model."""

    class INPUT_TYPE(enum.Enum):
        CMD_WRENCH = 0  # thrust [N] (scalar), torque [Nm] (3x1) : (4x1)
        CMD_PROP_FORCES = 1  # propeller forces [N] (4x1)
        CMD_ACCEL = 2  # acceleration [m/s^2] (3x1)

    class State(object):

        def __init__(self, l=1):
            self.cable_length = l
            # Quadrotor state
            self.position = np.array([0.0, 0.0, 1.0])
            self.velocity = np.zeros(3)
            self.orientation = np.eye(3)
            self.angular_velocity = np.zeros(3)
            # Payload state
            self.payload_position = np.zeros(3)
            self.payload_velocity = np.zeros(3)
            self.cable_attitude = np.array([0.0, 0.0, -1.0])
            self.cable_ang_velocity = np.zeros(3)
            return

        def dq(self):
            return np.cross(self.cable_ang_velocity, self.cable_attitude)

        def reset(self):
            # Quadrotor state
            self.position = np.array([0.0, 0.0, 1.0])
            self.velocity = np.zeros(3)
            self.orientation = np.eye(3)
            self.angular_velocity = np.zeros(3)
            # Payload state
            self.payload_position = np.zeros(3)
            self.payload_velocity = np.zeros(3)
            self.cable_attitude = np.array([0.0, 0.0, -1.0])
            self.cable_ang_velocity = np.zeros(3)
            return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = QuadrotorCSPayload.State()

        # system parameters
        self.mQ = 0.9  # kg
        self.JQ = np.array([[0.0023, 0.0, 0.0], [0.0, 0.0023, 0.0],
                            [0.0, 0.0, 0.004]])  # kg m^2
        self.JQinv = np.linalg.inv(self.JQ)
        self.mL = 0.2  # kg
        self.l = 1.0  # m

        self._min_thrust = 0.0
        self._max_thrust = 20.0
        self._min_torque = np.array([-5.0, -5.0, -2.0])
        self._max_torque = np.array([5.0, 5.0, 2.0])
        self._prop_min_force = 0.0
        self._prop_max_force = 10.0
        self._wrench_min = np.concatenate(
            [np.array([self._min_thrust]), self._min_torque])
        self._wrench_max = np.concatenate(
            [np.array([self._max_thrust]), self._max_torque])

        self._input_type = QuadrotorCSPayload.INPUT_TYPE.CMD_ACCEL
        self._n_action = 4
        self._step_freq = 500.0
        self._step_iter = max(1,
                              int(1.0 / self._step_freq / self.sim_timestep))

        self._parse_args(**kwargs)
        if "sim_timestep" in kwargs.keys():
            self.sim_timestep = kwargs["sim_timestep"]
        if "input" in kwargs.keys():
            if kwargs["input"] == "prop_forces":
                self._input_type = QuadrotorCSPayload.INPUT_TYPE.CMD_PROP_FORCES
                self._n_action = 4
                self._step_freq = 500.0
                self._step_iter = max(
                    1, int(1.0 / self._step_freq / self.sim_timestep))
            elif kwargs["input"] == "accel":
                self._input_type = QuadrotorCSPayload.INPUT_TYPE.CMD_ACCEL
                self._n_action = 3
                self._step_freq = 100.0
                self._step_iter = max(
                    1, int(1.0 / self._step_freq / self.sim_timestep))
            else:
                self._input_type = QuadrotorCSPayload.INPUT_TYPE.CMD_WRENCH
                self._n_action = 4
                self._step_freq = 500.0
                self._step_iter = max(
                    1, int(1.0 / self._step_freq / self.sim_timestep))

        if self.render:
            from ...utils import vfx

            self._vfx = vfx.QuadrotorCSPayloadVFX(l=self.l)

        self._init_default_controllers()
        return

    def _init_default_controllers(self):
        self._att_controller = control.QuadAttGeoPD(
            inertia=self.qrotor_inertia)
        self._payload_controller = control.QuadCSPayloadController()
        # self._prop_controller = control.QuadPropForceController(mass=self.mass, inertia=self.inertia)
        return

    @property
    def qrotor_mass(self) -> float:
        return self.mQ

    @qrotor_mass.setter
    def qrotor_mass(self, value: float):
        self.mQ = value

    @property
    def qrotor_inertia(self) -> np.ndarray:
        return self.JQ

    @qrotor_inertia.setter
    def qrotor_inertia(self, value: np.ndarray):
        if value.ndim == 1:
            value = np.diag(value)
        self.JQ = value
        self.JQinv = np.linalg.inv(self.JQ)

    @property
    def payload_mass(self) -> float:
        return self._payload_mass

    @payload_mass.setter
    def payload_mass(self, value: float):
        self._payload_mass = value

    @property
    def cable_length(self) -> float:
        return self.l

    @cable_length.setter
    def cable_length(self, value: float):
        self.l = value

    def _zoh(self, thrust: float, torque: np.ndarray):
        """Zero-order hold on the system equations of motion
        for the quadrotor with cable-suspended payload

        Dynamics are defined here
        Koushil Sreenath, Taeyoung Lee, and Vijay Kumar. "Geometric control and
        differential flatness of a quadrotor UAV with a cable-suspended load."
        In 52nd IEEE Conference on Decision and Control, pp. 2269-2274. IEEE, 2013.
        https://hybrid-robotics.berkeley.edu/publications/CDC2013_supplement.pdf
        """
        dq = self.state.dq()
        q = self.state.cable_attitude
        h = self.sim_timestep

        fRe3 = thrust * self.state.orientation @ self._e3
        payload_accel = (-self._ge3 + (
            (np.dot(q, fRe3) - self.mQ * self.l * np.dot(dq, dq)) /
            (self.mQ + self.mL)) * q)

        # payload position dynamics
        self.state.payload_position += (self.state.payload_velocity * h +
                                        0.5 * payload_accel * h**2)
        self.state.payload_velocity += payload_accel * h
        # cable attitude dynamics
        self.state.cable_attitude = (
            expm(utils.hat(h * self.state.cable_ang_velocity)) @ q)
        domega = -np.cross(q, fRe3) / (self.mQ * self.l)
        self.state.cable_ang_velocity += h * domega
        # quadrotor attitude dynamics
        self.state.orientation = self.state.orientation @ expm(
            utils.hat(self.state.angular_velocity * h))
        ang_vel_dot = self.JQinv @ (
            torque - np.cross(self.state.angular_velocity,
                              self.JQ @ self.state.angular_velocity))
        self.state.angular_velocity += ang_vel_dot * h
        # Update quadrotor position & velocity
        self.state.position = (self.state.payload_position -
                               self.l * self.state.cable_attitude)
        self.state.velocity = self.state.payload_velocity - self.l * self.state.dq(
        )
        return

    def _parse_input(self, input):
        if self._input_type == QuadrotorCSPayload.INPUT_TYPE.CMD_ACCEL:
            thrust, torque = self._att_controller.compute(
                self.t, (self.state.orientation, self.state.angular_velocity),
                input)
        elif self._input_type == QuadrotorCSPayload.INPUT_TYPE.CMD_PROP_FORCES:
            utils.printc_warn("TODO: Incorrect implementation verify")
            wrench = self._propforces_to_wrench(input)
            thrust, torque = wrench[0], wrench[1:]
        else:
            thrust, torque = input[0], input[1:]

        thrust = np.clip(thrust, self._min_thrust, self._max_thrust)
        torque = np.clip(torque, self._min_torque, self._max_torque)
        return thrust, torque

    def step(self, input: np.ndarray):
        """Zero-order hold on the system equations of motion

        :param input: input to the quadrotor (with cable-suspended payload)
        :type input: np.ndarray

        :return: None
        """
        for _ in range(self._step_iter):
            # integrate dynamics
            thrust, torque = self._parse_input(input)
            # physical input limits
            thrust = np.clip(thrust, self._min_thrust, self._max_thrust)
            torque = np.clip(torque, self._min_torque, self._max_torque)
            # dynamics zero-order hold integration (Euler integration)
            self._zoh(thrust, torque)
            # update time
            self.t += self.sim_timestep
        if self.render:
            self.update_render()
        return

    def update_render(self):
        if self.render:
            self._vfx.update(
                self.state.payload_position,
                self.state.cable_attitude,
                self.state.orientation,
            )
        return

    def reset(self, **kwargs):
        """reset state and time"""
        self.t = 0.0
        self.state.reset()

        k = [
            "position",
            "velocity",
            "orientation",
            "angular_velocity",
            "payload_position",
            "payload_velocity",
            "cable_attitude",
            "cable_ang_velocity",
        ]
        for key in k:
            if key in kwargs:
                setattr(self.state, key, kwargs[key])

        #
        # Warning: the code is only setting payload position/quadrotor position
        #

        # if only position is given, set payload position accordingly
        if "position" in kwargs:
            self.state.payload_position = (kwargs["position"] +
                                           self.l * self.state.cable_attitude)
        # if only payload position is given, set position accordingly
        if "payload_position" in kwargs:
            self.state.position = (kwargs["payload_position"] -
                                   self.l * self.state.cable_attitude)

        return

    def simulate(self, tf, **kwargs):
        self.reset(**kwargs)
        self.update_render()

        start_t = time.time_ns()
        while self.t < tf:
            if self._input_type == QuadrotorCSPayload.INPUT_TYPE.CMD_PROP_FORCES:
                raise NotImplementedError("TODO: Implement")
                u = self._prop_controller.compute(self.t, self.state)
            else:
                u = self._payload_controller.compute(self.t, self.state)
            self.step(u)

        end_t = time.time_ns()
        time_taken = (end_t - start_t) * 1e-9
        print("Took (%.4f)s for simulating (%.4f)s" % (time_taken, self.t))
        return
