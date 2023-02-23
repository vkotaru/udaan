import numpy as np
import os
from ..mujoco import MujocoModel
from .. import base
import time
import copy
from ... import utils
import enum
from scipy.spatial.transform import Rotation as sp_rot

class Quadrotor(base.Quadrotor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # mujoco model param handling
        self._mjMdl = None
        self._mj_quad_index = None
        self._mj_payload_index = None
        self._mj_cable_index = None

        if "attitude_zoh" in kwargs:
            self._attitude_zoh = kwargs["attitude_zoh"]

        # loading mujoco model
        self._mjMdl = MujocoModel("quadrotor_mj.xml", render=self._render)
        self._mj_quad_body_index = 1

        # read inertial parameters from mujoco model
        self.mass = copy.deepcopy(self._mjMdl.model.body_mass[self._mj_quad_body_index])
        self.inertia = copy.deepcopy(self._mjMdl.model.body_inertia[self._mj_quad_body_index])

        
        if self._verbose:
            utils.printc_ok("Mujoco model loaded")
        return

    @base.Quadrotor.mass.setter
    def mass(self, m):
        super(Quadrotor, self.__class__).mass.fset(self, m)
        self._mjMdl.model.body_mass[self._mj_quad_body_index] = m
        return
    
    @base.Quadrotor.inertia.setter
    def inertia(self, I):
        super(Quadrotor, self.__class__).inertia.fset(self, I)
        if (I.ndim==2):
            I = np.diag(I)
        self._mjMdl.model.body_inertia[self._mj_quad_body_index] = I
        return
    

    def reset(self, **kwargs):
        self.t = 0.
        self.super().reset(**kwargs)
        self._mjMdl.reset()

        self._mjMdl.data.qpos[:3] = self.state.position
        self._mjMdl.data.qvel[:3] = self.state.velocity
        qQ = sp_rot.from_matrix(self.state.orientation).as_quat()
        self._mjMdl.data.qpos[3:7] = np.array([qQ[3], qQ[0], qQ[1], qQ[2]])
        self._mjMdl.data.qvel[3:6] = self.state.angular_velocity

        self._query_latest_state()
        return
    
    def step(self, u):
        for _ in range(self._step_iter):
            if self._INPUT_TYPE == INPUT_TYPE.FORCES:
                u = self.wrench_to_propforces(u)
            self._mjMdl._step_mujoco_simulation(u, self._inner_loop_steps)
            self._query_latest_state()
            # add tracking marker
            if self._mjMdl.render:
                self.add_reference_marker(self.xQd)

        return
      
    def _query_latest_state(self):
        self.t = copy.deepcopy(self._mjMdl.data.time)
        pos = copy.deepcopy(self._mjMdl.data.qpos[:3])
        q = copy.deepcopy(self._mjMdl.data.qpos[3:7])
        vel = copy.deepcopy(self._mjMdl.data.qvel[:3])
        ang_vel = copy.deepcopy(self._mjMdl.data.qvel[3:6])

        self.state.position = np.array(pos)
        self.state.velocity = np.array(vel)
        self.state.orientation = self._mjMdl._quat2rot(q)
        self.state.angular_velocity = np.array(ang_vel)
        return
    
    def add_reference_marker(self, x):
        self._mjMdl.add_marker_at(x, label="xQd")
        return
