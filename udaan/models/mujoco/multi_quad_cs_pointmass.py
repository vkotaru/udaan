import numpy as np
from scipy.linalg import expm
import copy
import time
import numpy.matlib

from ..mujoco import MujocoModel, mujoco
from ..base import BaseModel
from ... import utils


class MultiQuadrotorCSPointmass(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    class State(object):
        class Quadrotor(object):
            def __init__(self, **kwargs):
                self.rotation = np.eye(3)
                self.angvel = np.zeros(3)
                self.position = np.zeros(3)
                self.velocity = np.zeros(3)
                self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                for key, value in kwargs.items():
                    setattr(self, key, value)
                return

            def reset(self):
                self.rotation = np.eye(3)
                self.angvel = np.zeros(3)
                self.position = np.zeros(3)
                self.velocity = np.zeros(3)
                self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                return

        class Cable(object):
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
            self.num_quadrotors = nQ
            self.quads = [self.Quadrotor() for _ in range(nQ)]
            self.cables = [self.Cable() for _ in range(nQ)]
            self.load_position = np.zeros(3)
            self.load_velocity = np.zeros(3)
            return

        def reset(self):
            for quad in self.quads:
                quad.reset()
            self.load_position = np.zeros(3)
            self.load_velocity = np.zeros(3)
            return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "num_quadrotors" in kwargs:
            self.nQ = kwargs["num_quadrotors"]
        else:
            self.nQ = 3

        self.state = MultiQuadrotorCSPointmass.State(nQ=self.nQ)

        # mujoco model param handling
        self._mjMdl = None
        self._mj_quad_index = None
        self._mj_payload_index = None
        self._mj_cable_index = None
        self._mj_render = self.render

        # TODO Create the xml file as required at runtime. 
        self._mjMdl = MujocoModel(
            model_path="multi_quad_pointmass.xml", render=self._mj_render
        )

        self._attitude_zoh = False
        if "attitude_zoh" in kwargs:
            self._attitude_zoh = kwargs["attitude_zoh"]

        self._mjDt = 1.0 / 500.0
        self._step_iter = int(self.sim_timestep / self._mjDt)
        self._nFrames = 1
        if self._attitude_zoh:
            self._step_iter, self._nFrames = self._nFrames, self._step_iter

        self.mQ = np.array([0.99]*self.nQ)
        self.mL = 0.15
        self._inertia_matrix = np.array(
            [[0.0023, 0.0, 0.0], [0.0, 0.0023, 0.0], [0.0, 0.0, 0.004]]
        )  # TODO using same inertia matrix for all quadrotors
        self._min_thrust = 0.0
        self._max_thrust = 20.0
        self._min_torque = np.array([-5.0, -5.0, -2.0])
        self._max_torque = np.array([5.0, 5.0, 2.0])
        self._prop_min_force = 0.0
        self._prop_max_force = 10.0
        self._wrench_min = np.concatenate(
            [np.array([self._min_thrust]), self._min_torque]
        )
        self._wrench_max = np.concatenate(
            [np.array([self._max_thrust]), self._max_torque]
        )
        self._feasible_min_input = np.matlib.repmat(self._wrench_min, self.nQ, 1).flatten()
        self._feasible_max_input = np.matlib.repmat(self._wrench_max, self.nQ, 1).flatten()
        print("Mujoco model loaded")
        return


    def step(self, u):
        for _ in range(self._step_iter):
          u_clamped = np.clip(u, self._feasible_min_input, self._feasible_max_input)
          # set control
          self._mjMdl.data.ctrl[self._ctrl_index : self._ctrl_index + 4] = u_clamped
          self._mjMdl._step_mujoco_simulation(self._nFrames)
          self._query_latest_state()
        return

    def reset(self, **kwargs):
        """reset state and time"""
        self.t = 0.0
        self._reset_to_default_state()
        if "xL" in kwargs:
            for i in range(self.nQ + 1):
                self.mjMdl.data.qpos[i * 7 : i * 7 + 3] += kwargs["xL"]

        self._query_latest_state()
        return

    def _reset_to_default_state(self):
        self.mjMdl.reset()
        return

    def _query_latest_state(self):
        for i in range(self.nQ):
            self._state.quads[i].position = self.mjMdl.data.qpos[7 * i : 7 * i + 3]
            _quat = self.mjMdl.data.qpos[7 * i + 3 : 7 * i + 7]
            self._state.quads[i].quaternion = _quat
            self._state.quads[i].rotation = self.mjMdl._quat2rot(_quat)
            self._state.quads[i].velocity = self.mjMdl.data.qvel[6 * i : 6 * i + 3]
            self._state.quads[i].angvel = self.mjMdl.data.qvel[6 * i + 3 : 6 * i + 6]

        self._state.load_position = self.mjMdl.data.qpos[7 * self.nQ : 7 * self.nQ + 3]
        self._state.load_velocity = self.mjMdl.data.qvel[6 * self.nQ : 6 * self.nQ + 3]

        for i in range(self.nQ):
            p = self._state.load_position - self._state.quads[i].position
            self._state.cables[i].length = np.linalg.norm(p)
            self._state.cables[i].q = p / self._state.cables[i].length
            self._state.cables[i].dq = (
                self._state.load_velocity - self._state.quads[i].velocity
            )
            self._state.cables[i].omega = np.cross(
                self._state.cables[i].q, self._state.cables[i].dq
            )
        return

    def quad_position_control(self):
        """quadrotor position control"""
        thrust_vec = np.zeros(3 * self.nQ)
        for i in range(self.nQ):
            kp = np.array([4.1, 4.1, 8.1])
            kd = 1.5 * np.array([2.0, 2.0, 6.0])

            ex = self._state.quads[i].position - self._init_state.quads[i].position
            ev = self._state.quads[i].velocity - np.zeros(3)
            Fpd = -kp * ex - kd * ev
            Fff = (self.mQ[i] + self.mL) * (self.g * self.e3)
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
        R = self._state.quads[i].rotation
        Omega = self._state.quads[i].angvel
        Omegad = np.zeros(3)  # TODO add differential flatness
        dOmegad = np.zeros(3)  # TODO add differential flatness

        # attitude control
        tmp = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])  # vee-map
        eOmega = Omega - R.T @ Rd @ Omegad

        kR = np.array([2.4, 2.4, 1.35])
        kOm = np.array([0.35, 0.35, 0.225])

        M = -kR * eR - kOm * eOmega + np.cross(Omega, self._inertia_matrix @ Omega)
        M += (
            -1
            * self._inertia_matrix
            @ (manif.hat(Omega) @ R.T @ Rd @ Omegad - R.T @ Rd @ dOmegad)
        )
        # ignoring this for since Omegad is zero
        f = thrust_force.dot(R[:, 2])
        # print(f,M)
        return np.hstack([f, M])

    def simulate(self, tf, **kwargs):
        self.reset(**kwargs)

        start_t = time.time_ns()
        while self.t < tf:
            thrust_vecs = self.quad_position_control()
            self.step(thrust_vecs)
        end_t = time.time_ns()
        print(
            "Took (%.4f)s for simulating (%.4f)s"
            % (float(end_t - start_t) * 1e-9, self.t)
        )
        pass
