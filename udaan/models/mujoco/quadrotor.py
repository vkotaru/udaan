import numpy as np
import time
import copy
import enum
from scipy.spatial.transform import Rotation as sp_rot

from ..mujoco import MujocoModel
from .. import base
from ... import utils


class Quadrotor(base.Quadrotor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # mujoco model param handling
        self._mjMdl = None
        self._mj_quad_index = None
        self._mj_payload_index = None
        self._mj_cable_index = None

        self._attitude_zoh = False
        if "attitude_zoh" in kwargs:
            self._attitude_zoh = kwargs["attitude_zoh"]

        self._mjDt = 1. / 500.
        self._step_iter = int(self.sim_timestep / self._mjDt)
        self._nFrames = 1
        if self._attitude_zoh:
            self._step_iter, self._nFrames = self._nFrames, self._step_iter

        # loading mujoco model
        self._mjMdl = MujocoModel("quadrotor_mj.xml", render=self.render)
        self._mj_quad_body_index = 1
        if self._force_type == base.Quadrotor.FORCE_TYPE.PROP_FORCES:
            self._ctrl_index = 0
        else:
            self._ctrl_index = 4

        # read inertial parameters from mujoco model
        self.mass = copy.deepcopy(
            self._mjMdl.model.body_mass[self._mj_quad_body_index])
        self.inertia = copy.deepcopy(
            self._mjMdl.model.body_inertia[self._mj_quad_body_index])

        # reinitialize controllers after loading mujoco model & params
        self._init_default_controllers()

        if self.verbose:
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
        if (I.ndim == 2):
            I = np.diag(I)
        self._mjMdl.model.body_inertia[self._mj_quad_body_index] = I
        return
      
    def _parse_input(self):
        """Parse input type based on input and force type"""
        if self._input_type == Quadrotor.INPUT_TYPE.CMD_ACCEL:
            if self._force_type == Quadrotor.FORCE_TYPE.PROP_FORCES:
              self._repack_input = lambda u : self._in_accel_out_propforces(u)
            else:
              self._repack_input = lambda u : self._in_accel_out_wrench(u)
        elif self._input_type == Quadrotor.INPUT_TYPE.CMD_PROP_FORCES:
            if self._force_type == Quadrotor.FORCE_TYPE.PROP_FORCES:
              self._repack_input = lambda u : self._in_propforces_out_propforces(u)
            else:
              self._repack_input = lambda u : self._in_propforces_out_wrench(u)
        else:            
          if self._force_type == Quadrotor.FORCE_TYPE.PROP_FORCES:
            self._repack_input = lambda u : self._in_wrench_out_propforces(u)
          else:
            self._repack_input = lambda u: self._in_wrench_out_wrench(u)
        return  

    def reset(self, **kwargs):
        self.t = 0.
        super().reset(**kwargs)
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
            u_clamped = self._repack_input(u)
            # set control
            self._mjMdl.data.ctrl[self._ctrl_index:self._ctrl_index +
                                  4] = u_clamped
            # mujoco simulation
            self._mjMdl._step_mujoco_simulation(self._nFrames)
            # update state
            self._query_latest_state()
            # add tracking marker
            if self._mjMdl.render:
                # self.add_reference_marker(self.xQd)
                if self._force_type == base.Quadrotor.FORCE_TYPE.PROP_FORCES:
                    for i in range(4):
                        self._mjMdl.add_arrow_at(
                            self._mjMdl.data.site_xpos[i],
                            self._mjMdl.data.site_xmat[i],
                            s=[0.005, 0.005, 0.25 * float(u_clamped[i])],
                            label="f%d" % i,
                            color=[1., 1., 0., 1.])
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
