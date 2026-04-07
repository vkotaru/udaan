import numpy as np
import scipy.linalg

from ..base import BaseModel
from ... import manif


class PointmassSuspendedPayload(BaseModel):
    """Single pointmass quadrotor with cable-suspended payload.

    State: payload_position (3), payload_velocity (3),
           cable_attitude (3, S2), cable_ang_velocity (3)
    -> n_state=12, n_action=3
    """

    class State(object):
        def __init__(self):
            self.payload_position = np.zeros(3)
            self.payload_velocity = np.zeros(3)
            self.cable_attitude = np.array([0.0, 0.0, -1.0])
            self.cable_ang_velocity = np.zeros(3)
            return

        def reset(self):
            self.payload_position = np.zeros(3)
            self.payload_velocity = np.zeros(3)
            self.cable_attitude = np.array([0.0, 0.0, -1.0])
            self.cable_ang_velocity = np.zeros(3)
            return

        @property
        def cable_dq(self):
            return np.cross(self.cable_ang_velocity, self.cable_attitude)

        @property
        def quadrotor_position(self):
            return None  # requires cable_length; set via parent model

        @property
        def quadrotor_velocity(self):
            return None  # requires cable_length; set via parent model

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_state = 12
        self._n_action = 3
        self.mQ = 1.0
        self.mL = 0.15
        self.cable_length = 1.0
        self.state = PointmassSuspendedPayload.State()
        self._parse_args(**kwargs)
        return

    @property
    def quadrotor_position(self):
        return (
            self.state.payload_position - self.cable_length * self.state.cable_attitude
        )

    @property
    def quadrotor_velocity(self):
        return self.state.payload_velocity - self.cable_length * np.cross(
            self.state.cable_ang_velocity, self.state.cable_attitude
        )

    def _zoh(self, force):
        """Zero-order hold Euler integration for payload-quadrotor system."""
        dt = self.sim_timestep
        q = self.state.cable_attitude
        om = self.state.cable_ang_velocity
        dq = np.cross(om, q)

        net_accel = (
            -self._ge3
            + (
                (np.dot(q, force) - self.mQ * self.cable_length * np.dot(dq, dq))
                / (self.mQ + self.mL)
            )
            * q
        )

        self.state.payload_position += (
            dt * self.state.payload_velocity + 0.5 * net_accel * dt * dt
        )
        self.state.payload_velocity += dt * net_accel

        # S2 update for cable attitude via SO(3) exponential map
        self.state.cable_attitude = scipy.linalg.expm(manif.hat(om * dt)) @ q
        # angular velocity update
        self.state.cable_ang_velocity = om + (
            dt / (self.mQ * self.cable_length)
        ) * np.cross(-q, force)

    def step(self, action):
        self._zoh(action)
        self.t += self.sim_timestep

    def reset(self, **kwargs):
        self.t = 0.0
        self.state.reset()
        for key in [
            "payload_position",
            "payload_velocity",
            "cable_attitude",
            "cable_ang_velocity",
        ]:
            if key in kwargs:
                setattr(self.state, key, kwargs[key])
        return

    def get_rand_init_state(self, rand=False):
        if rand:
            init_pos = -5.0 + 10 * np.random.rand(3)
        else:
            init_pos = np.zeros(3)
        return {
            "payload_position": init_pos,
            "payload_velocity": np.zeros(3),
            "cable_attitude": np.array([0.0, 0.0, -1.0]),
            "cable_ang_velocity": np.zeros(3),
        }

    @staticmethod
    def err_q(qd, q):
        return np.cross(qd, q)

    @staticmethod
    def err_om(omd, om, q):
        return om + manif.hat(q) @ manif.hat(q) @ omd
