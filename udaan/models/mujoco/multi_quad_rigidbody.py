import numpy as np
import copy

from ..mujoco import MujocoModel
from ..base import BaseModel
from ... import manif


class MultiQuadRigidbody(BaseModel):
    """Multi-quadrotor rigid-body payload MuJoCo model.

    4 quadrotors carrying a rigid-body payload connected via cables.
    Outer loop: position-level thrust vectors (3 per quad).
    Inner loop: geometric attitude controller converts to wrench.
    """

    class State(object):
        class Odometry(object):
            def __init__(self, **kwargs):
                self.rotation = np.eye(3)
                self.angvel = np.zeros(3)
                self.position = np.zeros(3)
                self.velocity = np.zeros(3)
                self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
                for key, value in kwargs.items():
                    setattr(self, key, value)
                return

            def reset(self):
                self.rotation = np.eye(3)
                self.angvel = np.zeros(3)
                self.position = np.zeros(3)
                self.velocity = np.zeros(3)
                self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
                return

        class CableState(object):
            def __init__(self, **kwargs):
                self.q = np.array([0.0, 0.0, -1.0])
                self.omega = np.zeros(3)
                self.dq = np.zeros(3)
                self.length = 1.0
                for key, value in kwargs.items():
                    setattr(self, key, value)
                return

            def reset(self):
                self.q = np.array([0.0, 0.0, -1.0])
                self.omega = np.zeros(3)
                self.dq = np.zeros(3)
                self.length = 1.0
                return

        def __init__(self, nQ: int, **kwargs):
            self.quads = [self.Odometry() for _ in range(nQ)]
            self.cables = [self.CableState() for _ in range(nQ)]
            self.load = self.Odometry()
            return

        def reset(self):
            for quad in self.quads:
                quad.reset()
            for cable in self.cables:
                cable.reset()
            self.load.reset()
            return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nQ = kwargs.get("nQ", 4)
        self._n_action = 3 * self.nQ

        self._mjMdl = MujocoModel(
            model_path="multi_quad_rigidbody.xml", render=self.render
        )

        self._mjDt = 1.0 / 500.0
        self._step_iter = int(self.sim_timestep / self._mjDt)
        self._nFrames = 1

        self._attitude_zoh = kwargs.get("attitude_zoh", True)
        if self._attitude_zoh:
            self._step_iter, self._nFrames = self._nFrames, self._step_iter

        self.mQ = kwargs.get("mQ", [0.75] * self.nQ)
        self.mL = kwargs.get("mL", 1.0)
        self._inertia_matrix = np.array(
            [[0.0023, 0.0, 0.0], [0.0, 0.0023, 0.0], [0.0, 0.0, 0.004]]
        )

        self._grasp_map = np.array(
            [[0.5, -0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [0.05, 0.05, 0.05, 0.05]]
        )

        self._feasible_max_input = np.array([30.0, 3.0, 3.0, 3.0] * self.nQ)
        self._feasible_min_input = np.array([0.0, -3.0, -3.0, -3.0] * self.nQ)
        self.state = MultiQuadRigidbody.State(self.nQ)
        self._query_latest_state()
        self._init_state = copy.deepcopy(self.state)
        return

    def step(self, action):
        """Step with position-level thrust vectors (3*nQ).

        Args:
            action: (3*nQ,) thrust force vector per quadrotor.
        """
        u = np.zeros(4 * self.nQ)
        for _ in range(self._step_iter):
            for i in range(self.nQ):
                u[4 * i : 4 * (i + 1)] = self.compute_attitude_control(
                    i, action[3 * i : 3 * (i + 1)]
                )
            u = np.clip(u, self._feasible_min_input, self._feasible_max_input)
            self._mjMdl.data.ctrl[:] = u
            self._mjMdl._step_mujoco_simulation(self._nFrames)
            self._query_latest_state()
        self.t = self._mjMdl.data.time
        return

    def reset(self, **kwargs):
        self.t = 0.0
        self._mjMdl.reset()
        if "xL" in kwargs:
            for i in range(self.nQ + 1):
                self._mjMdl.data.qpos[i * 7 : i * 7 + 3] += kwargs["xL"]
        self._query_latest_state()
        return

    def _query_latest_state(self):
        for i in range(self.nQ):
            self.state.quads[i].position = self._mjMdl.data.qpos[
                7 * i : 7 * i + 3
            ].copy()
            _quat = self._mjMdl.data.qpos[7 * i + 3 : 7 * i + 7].copy()
            self.state.quads[i].quaternion = _quat
            self.state.quads[i].rotation = self._mjMdl._quat2rot(_quat)
            self.state.quads[i].velocity = self._mjMdl.data.qvel[
                6 * i : 6 * i + 3
            ].copy()
            self.state.quads[i].angvel = self._mjMdl.data.qvel[
                6 * i + 3 : 6 * i + 6
            ].copy()

        nQ = self.nQ
        self.state.load.position = self._mjMdl.data.qpos[7 * nQ : 7 * nQ + 3].copy()
        self.state.load.velocity = self._mjMdl.data.qvel[6 * nQ : 6 * nQ + 3].copy()
        self.state.load.quaternion = self._mjMdl.data.qpos[
            7 * nQ + 3 : 7 * nQ + 7
        ].copy()
        self.state.load.rotation = self._mjMdl._quat2rot(self.state.load.quaternion)
        self.state.load.angvel = self._mjMdl.data.qvel[6 * nQ + 3 : 6 * nQ + 6].copy()

        for i in range(self.nQ):
            poc_wf = (
                self.state.load.position
                + self.state.load.rotation @ self._grasp_map[:, i]
            )
            dpoc_wf = (
                self.state.load.velocity
                + self.state.load.rotation
                @ manif.hat(self.state.load.angvel)
                @ self._grasp_map[:, i]
            )

            p = poc_wf - self.state.quads[i].position
            self.state.cables[i].length = np.linalg.norm(p)
            self.state.cables[i].q = p / self.state.cables[i].length
            self.state.cables[i].dq = dpoc_wf - self.state.quads[i].velocity
            self.state.cables[i].omega = np.cross(
                self.state.cables[i].q, self.state.cables[i].dq
            )
        return

    def compute_attitude_control(self, i, thrust_force):
        """Geometric attitude controller for quadrotor i.

        Args:
            i: quadrotor index
            thrust_force: (3,) desired thrust vector in world frame

        Returns:
            (4,) [thrust_scalar, Mx, My, Mz] wrench
        """
        norm_thrust = np.linalg.norm(thrust_force)
        if norm_thrust < 1e-6:
            return np.zeros(4)

        b1d = np.array([1.0, 0.0, 0.0])
        b3c = thrust_force / norm_thrust
        b3_b1d = np.cross(b3c, b1d)
        norm_b3_b1d = np.linalg.norm(b3_b1d)
        b1c = (-1 / norm_b3_b1d) * np.cross(b3c, b3_b1d)
        b2c = np.cross(b3c, b1c)
        Rd = np.hstack(
            [
                np.expand_dims(b1c, axis=1),
                np.expand_dims(b2c, axis=1),
                np.expand_dims(b3c, axis=1),
            ]
        )

        R = self.state.quads[i].rotation
        Omega = self.state.quads[i].angvel
        Omegad = np.zeros(3)
        dOmegad = np.zeros(3)

        # vee-map rotation error
        tmp = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])
        eOmega = Omega - R.T @ Rd @ Omegad

        kR = np.array([2.4, 2.4, 1.35])
        kOm = np.array([0.35, 0.35, 0.225])

        M = -kR * eR - kOm * eOmega + np.cross(Omega, self._inertia_matrix @ Omega)
        M += (
            -1
            * self._inertia_matrix
            @ (manif.hat(Omega) @ R.T @ Rd @ Omegad - R.T @ Rd @ dOmegad)
        )

        f = thrust_force.dot(R[:, 2])
        return np.hstack([f, M])
