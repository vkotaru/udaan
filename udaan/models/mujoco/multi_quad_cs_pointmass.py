import copy
import os
import time

import numpy as np

from ... import _FOLDER_PATH, manif
from ...core.defaults import (
    DEFAULT_ATT_KD,
    DEFAULT_ATT_KP,
    DEFAULT_POS_KD,
    DEFAULT_POS_KP,
    DEFAULT_QUAD_INERTIA,
)
from ...utils.logging import get_logger
from ..base import BaseModel
from ..mujoco import MujocoModel

_logger = get_logger(__name__)

_MJCF_DIR = os.path.join(_FOLDER_PATH, "udaan", "models", "assets", "mjcf")


class MultiQuadrotorCSPointmass(BaseModel):
    """Multi-quadrotor with cable-suspended pointmass payload (MuJoCo)."""

    class State:
        class Quadrotor:
            def __init__(self, **kwargs):
                self.rotation = np.eye(3)
                self.angvel = np.zeros(3)
                self.position = np.zeros(3)
                self.velocity = np.zeros(3)
                self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def reset(self):
                self.rotation = np.eye(3)
                self.angvel = np.zeros(3)
                self.position = np.zeros(3)
                self.velocity = np.zeros(3)
                self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        class Cable:
            def __init__(self, **kwargs):
                self.q = np.array([0.0, 0.0, -1.0])
                self.omega = np.zeros(3)
                self.dq = np.zeros(3)
                self.length = 1.0
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def reset(self):
                self.q = np.array([0.0, 0.0, -1.0])
                self.omega = np.zeros(3)
                self.dq = np.zeros(3)
                self.length = 1.0

        def __init__(self, nQ: int, **kwargs):
            self.num_quadrotors = nQ
            self.quads = [self.Quadrotor() for _ in range(nQ)]
            self.cables = [self.Cable() for _ in range(nQ)]
            self.load_position = np.zeros(3)
            self.load_velocity = np.zeros(3)

        def reset(self):
            for quad in self.quads:
                quad.reset()
            self.load_position = np.zeros(3)
            self.load_velocity = np.zeros(3)

    @staticmethod
    def _ensure_xml(nQ):
        """Return MJCF filename, generating it at runtime if needed."""
        from ...utils.assets import xml_model_generator

        default_xml = "multi_quad_pointmass.xml"
        default_path = os.path.join(_MJCF_DIR, default_xml)

        # Check if default XML matches requested nQ
        if os.path.exists(default_path):
            import mujoco

            m = mujoco.MjModel.from_xml_path(default_path)
            existing_nQ = m.nq // 7 - 1
            if existing_nQ == nQ:
                return default_xml

        # Generate XML for requested nQ
        generated_name = f"multi_quad_pointmass_{nQ}q.xml"
        generated_path = os.path.join(_MJCF_DIR, generated_name)
        if not os.path.exists(generated_path):
            _logger.info("Generating MJCF for %d quadrotors: %s", nQ, generated_name)
            xml_model_generator.multi_quad_pointmass(nQ=nQ, filename=generated_path, verbose=False)
        return generated_name

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nQ = kwargs.get("num_quadrotors", 3)
        self.render = kwargs.get("render", False)

        # mujoco model — generate XML at runtime if needed
        self._mj_render = self.render
        xml_path = self._ensure_xml(self.nQ)
        self._mjMdl = MujocoModel(model_path=xml_path, render=self._mj_render)

        # Verify nQ matches the loaded model
        n_free_bodies = self._mjMdl.model.nq // 7
        self.nQ = n_free_bodies - 1
        self.state = MultiQuadrotorCSPointmass.State(nQ=self.nQ)

        self._attitude_zoh = kwargs.get("attitude_zoh", False)

        self._mjDt = 1.0 / 500.0
        self._step_iter = int(self.sim_timestep / self._mjDt)
        self._nFrames = 1
        if self._attitude_zoh:
            self._step_iter, self._nFrames = self._nFrames, self._step_iter

        self._quad_masses = np.array([0.99] * self.nQ)
        self._payload_mass = 0.15
        self._inertia_matrix = DEFAULT_QUAD_INERTIA.copy()
        self._min_thrust = 0.0
        self._max_thrust = 20.0
        self._min_torque = np.array([-5.0, -5.0, -2.0])
        self._max_torque = np.array([5.0, 5.0, 2.0])
        self._prop_min_force = 0.0
        self._prop_max_force = 10.0
        self._wrench_min = np.concatenate([np.array([self._min_thrust]), self._min_torque])
        self._wrench_max = np.concatenate([np.array([self._max_thrust]), self._max_torque])
        self._feasible_min_input = np.tile(self._wrench_min, self.nQ)
        self._feasible_max_input = np.tile(self._wrench_max, self.nQ)
        _logger.info("MuJoCo model loaded")

        self._init_state = None

    def step(self, u):
        for _ in range(self._step_iter):
            u_clamped = np.clip(u, self._feasible_min_input, self._feasible_max_input)
            self._mjMdl.data.ctrl[:] = u_clamped
            self._mjMdl._step_mujoco_simulation(self._nFrames)
            self._query_latest_state()
        self.t = self._mjMdl.data.time

    def reset(self, **kwargs):
        """reset state and time"""
        self.t = 0.0
        self._mjMdl.reset()
        if "xL" in kwargs:
            for i in range(self.nQ + 1):
                self._mjMdl.data.qpos[i * 7 : i * 7 + 3] += kwargs["xL"]

        self._query_latest_state()
        self._init_state = copy.deepcopy(self.state)

    def _query_latest_state(self):
        for i in range(self.nQ):
            self.state.quads[i].position = self._mjMdl.data.qpos[7 * i : 7 * i + 3]
            _quat = self._mjMdl.data.qpos[7 * i + 3 : 7 * i + 7]
            self.state.quads[i].quaternion = _quat
            self.state.quads[i].rotation = self._mjMdl._quat2rot(_quat)
            self.state.quads[i].velocity = self._mjMdl.data.qvel[6 * i : 6 * i + 3]
            self.state.quads[i].angvel = self._mjMdl.data.qvel[6 * i + 3 : 6 * i + 6]

        self.state.load_position = self._mjMdl.data.qpos[7 * self.nQ : 7 * self.nQ + 3]
        self.state.load_velocity = self._mjMdl.data.qvel[6 * self.nQ : 6 * self.nQ + 3]

        for i in range(self.nQ):
            p = self.state.load_position - self.state.quads[i].position
            self.state.cables[i].length = np.linalg.norm(p)
            self.state.cables[i].q = p / self.state.cables[i].length
            self.state.cables[i].dq = self.state.load_velocity - self.state.quads[i].velocity
            self.state.cables[i].omega = np.cross(self.state.cables[i].q, self.state.cables[i].dq)

    def quad_position_control(self):
        """quadrotor position control"""
        thrust_vec = np.zeros(3 * self.nQ)
        for i in range(self.nQ):
            kp = DEFAULT_POS_KP.copy()
            kd = DEFAULT_POS_KD.copy()

            ex = self.state.quads[i].position - self._init_state.quads[i].position
            ev = self.state.quads[i].velocity - np.zeros(3)
            Fpd = -kp * ex - kd * ev
            Fff = (self._quad_masses[i] + self._payload_mass) * (self._g * self._e3)
            thrust_vec[3 * i : 3 * i + 3] = Fpd + Fff

        return thrust_vec

    def compute_attitude_control(self, i, thrust_force):
        norm_thrust = np.linalg.norm(thrust_force)
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
        Omegad = np.zeros(3)  # TODO add differential flatness
        dOmegad = np.zeros(3)  # TODO add differential flatness

        # attitude control
        tmp = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])  # vee-map
        eOmega = Omega - R.T @ Rd @ Omegad

        kR = DEFAULT_ATT_KP.copy()
        kOm = DEFAULT_ATT_KD.copy()

        M = -kR * eR - kOm * eOmega + np.cross(Omega, self._inertia_matrix @ Omega)
        M += -1 * self._inertia_matrix @ (manif.hat(Omega) @ R.T @ Rd @ Omegad - R.T @ Rd @ dOmegad)
        f = thrust_force.dot(R[:, 2])
        return np.hstack([f, M])

    def simulate(self, tf, **kwargs):
        self.reset(**kwargs)

        start_t = time.time_ns()
        while self.t < tf:
            thrust_vecs = self.quad_position_control()
            # Convert 3D thrust vectors to wrench (thrust + torque) per quad
            u = np.zeros(4 * self.nQ)
            for i in range(self.nQ):
                fi = thrust_vecs[3 * i : 3 * i + 3]
                wrench = self.compute_attitude_control(i, fi)
                u[4 * i : 4 * i + 4] = wrench
            self.step(u)
        end_t = time.time_ns()
        _logger.debug("Took (%.4f)s for simulating (%.4f)s", float(end_t - start_t) * 1e-9, self.t)

        if self._mj_render and self._mjMdl._viewer is not None:
            self._mjMdl.wait_for_close()
