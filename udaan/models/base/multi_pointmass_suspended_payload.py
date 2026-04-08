import numpy as np
import scipy.linalg

from ... import manif
from ..base import BaseModel


class MultiPointmassSuspendedPayload(BaseModel):
    """Multiple pointmass quadrotors carrying a single point-mass payload.

    State: payload_position (3), payload_velocity (3),
           cable_attitudes (3, nQ), cable_ang_velocities (3, nQ)
    -> n_state = 6 + 6*nQ, n_action = 3*nQ
    """

    class State:
        def __init__(self, nQ):
            self._nQ = nQ
            self.payload_position = np.zeros(3)
            self.payload_velocity = np.zeros(3)
            self.cable_attitudes = np.tile(np.array([0.0, 0.0, -1.0]), (nQ, 1)).T  # (3, nQ)
            self.cable_ang_velocities = np.zeros((3, nQ))
            return

        def reset(self):
            self.payload_position = np.zeros(3)
            self.payload_velocity = np.zeros(3)
            self.cable_attitudes = np.tile(np.array([0.0, 0.0, -1.0]), (self._nQ, 1)).T
            self.cable_ang_velocities = np.zeros((3, self._nQ))
            return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nQ = kwargs.get("nQ", 2)
        self._n_state = 6 + 6 * self.nQ
        self._n_action = 3 * self.nQ
        self.mQ = kwargs.get("mQ", np.ones(self.nQ))
        self.mL = kwargs.get("mL", 0.5)
        self.cable_lengths = kwargs.get("cable_lengths", np.ones(self.nQ))
        self._inner_dt = 1.0 / 1000.0
        self._inner_loop_steps = max(1, int(self.sim_timestep / self._inner_dt))
        self.state = MultiPointmassSuspendedPayload.State(self.nQ)
        return

    def _zoh(self, force):
        """Zero-order hold Euler integration for multi-quad payload system.

        Args:
            force: (3, nQ) force matrix, one column per quadrotor.
        """
        h = self._inner_dt
        q = self.state.cable_attitudes
        om = self.state.cable_ang_velocities

        u_para = []
        u_perp = []
        qiqiT = []
        accel_rhs = np.zeros(3)
        Mq = np.eye(3) * self.mL

        for i in range(self.nQ):
            qi = q[:, i]
            qqT = np.outer(qi, qi)
            qiqiT.append(qqT)
            Mq += self.mQ[i] * qqT
            ui = force[:, i]
            u_para.append(qqT @ ui)
            u_perp.append((np.eye(3) - qqT) @ ui)
            accel_rhs += (
                u_para[i] - self.mQ[i] * self.cable_lengths[i] * np.dot(om[:, i], om[:, i]) * qi
            )

        net_accel = -self._ge3 + np.linalg.inv(Mq) @ accel_rhs
        self.state.payload_position += h * self.state.payload_velocity + 0.5 * net_accel * h * h
        self.state.payload_velocity += h * net_accel

        q_next = []
        om_next = []
        for i in range(self.nQ):
            qi = q[:, i]
            domi = (1 / self.cable_lengths[i]) * manif.hat(qi) @ (net_accel + self._ge3) - (
                1 / (self.cable_lengths[i] * self.mQ[i])
            ) * manif.hat(qi) @ u_perp[i]
            q_next.append(scipy.linalg.expm(manif.hat(om[:, i] * h)) @ qi)
            om_next.append(om[:, i] + domi * h)

        self.state.cable_attitudes = np.array(q_next).T
        self.state.cable_ang_velocities = np.array(om_next).T

    def step(self, action):
        """Step the dynamics forward by sim_timestep.

        Args:
            action: (3, nQ) force matrix or (3*nQ,) flat vector.
        """
        force = action.reshape(3, self.nQ) if action.ndim == 1 else action

        for _ in range(self._inner_loop_steps):
            self._zoh(force)
        self.t += self.sim_timestep

    def reset(self, **kwargs):
        self.t = 0.0
        self.state.reset()
        for key in [
            "payload_position",
            "payload_velocity",
            "cable_attitudes",
            "cable_ang_velocities",
        ]:
            if key in kwargs:
                setattr(self.state, key, kwargs[key])
        return

    def get_rand_init_state(self, rand=False):
        rng = np.random.default_rng()
        init_pos = -5.0 + 10 * rng.random(3) if rand else np.zeros(3)
        return {
            "payload_position": init_pos,
            "payload_velocity": np.zeros(3),
            "cable_attitudes": np.tile(np.array([0.0, 0.0, -1.0]), (self.nQ, 1)).T,
            "cable_ang_velocities": np.zeros((3, self.nQ)),
        }
