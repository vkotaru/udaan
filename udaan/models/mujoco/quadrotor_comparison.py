import numpy as np
import time
import copy
import enum
from scipy.spatial.transform import Rotation as sp_rot

from . import MujocoModel
from .. import base
from ... import utils


class QuadrotorComparison(base.BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.plant = base.Quadrotor()
        self.reference = base.Quadrotor()

        # mujoco model param handling
        self._mjMdl = None
        self._mj_quad_index = None
        self._mj_payload_index = None
        self._mj_cable_index = None

        self._attitude_zoh = False
        if "attitude_zoh" in kwargs:
            self._attitude_zoh = kwargs["attitude_zoh"]
        if "render" in kwargs:
            self.render = kwargs["render"]

        self._mjDt = 1.0 / 500.0
        self._step_iter = int(self.sim_timestep / self._mjDt)
        self._nFrames = 1
        if self._attitude_zoh:
            self._step_iter, self._nFrames = self._nFrames, self._step_iter

        # loading mujoco model
        self._mjMdl = MujocoModel("quadrotor_comparison.xml",
                                  render=self.render)
        self._mj_plant_body_idx = 1
        self._mj_reference_body_idx = 2
        self._mj_plant_ctrl_idx = 0
        self._mj_reference_ctrl_idx = 4

        # read inertial parameters from mujoco model
        self.plant.mass = copy.deepcopy(
            self._mjMdl.model.body_mass[self._mj_plant_body_idx])
        self.plant.inertia = copy.deepcopy(
            self._mjMdl.model.body_inertia[self._mj_plant_body_idx])
        self.reference.mass = copy.deepcopy(
            self._mjMdl.model.body_mass[self._mj_reference_body_idx])
        self.reference.inertia = copy.deepcopy(
            self._mjMdl.model.body_inertia[self._mj_reference_body_idx])

        # reinitialize controllers after loading mujoco model & params
        self.plant._init_default_controllers()
        self.reference._init_default_controllers()

        if self.verbose:
            utils.printc_ok("Mujoco model loaded")
        return

    def set_mass(self, m, plant=True):
        if plant:
            self.plant.mass = m
            self._mjMdl.model.body_mass[self._mj_plant_body_idx] = m
        else:
            self.reference.mass = m
            self._mjMdl.model.body_mass[self._mj_reference_body_idx] = m
        return

    def set_inertia(self, I, plant=True):
        if I.ndim == 2:
            I = np.diag(I)
        if plant:
            self.plant.inertia = I
            self._mjMdl.model.body_inertia[self._mj_plant_body_idx] = I
        else:
            self.reference.mass = I
            self._mjMdl.model.body_mass[self._mj_reference_body_idx] = I
        return

    def reset(self, **kwargs):
        self.t = 0.0
        self.plant.state.reset()
        self.reference.state.reset()
        k = ["position", "velocity", "orientation", "angular_velocity"]
        for key in k:
            if key in kwargs:
                setattr(self.plant.state, key, kwargs[key])
                setattr(self.reference.state, key, kwargs[key])
        self._mjMdl.reset()

        # Set plant position
        self._mjMdl.data.qpos[:3] = self.plant.state.position
        self._mjMdl.data.qvel[:3] = self.plant.state.velocity
        qQ = sp_rot.from_matrix(self.plant.state.orientation).as_quat()
        self._mjMdl.data.qpos[3:7] = np.array([qQ[3], qQ[0], qQ[1], qQ[2]])
        self._mjMdl.data.qvel[3:6] = self.plant.state.angular_velocity
        
        # Set reference position
        self._mjMdl.data.qpos[7:10] = self.reference.state.position
        self._mjMdl.data.qvel[6:9] = self.reference.state.velocity
        qQ = sp_rot.from_matrix(self.reference.state.orientation).as_quat()
        self._mjMdl.data.qpos[10:14] = np.array([qQ[3], qQ[0], qQ[1], qQ[2]])
        self._mjMdl.data.qvel[9:12] = self.reference.state.angular_velocity
        
        self._query_latest_state()
        return

    def _query_latest_state(self):
        self.t = copy.deepcopy(self._mjMdl.data.time)
        pos = copy.deepcopy(self._mjMdl.data.qpos[:3])
        q = copy.deepcopy(self._mjMdl.data.qpos[3:7])
        vel = copy.deepcopy(self._mjMdl.data.qvel[:3])
        ang_vel = copy.deepcopy(self._mjMdl.data.qvel[3:6])

        self.plant.state.position = np.array(pos)
        self.plant.state.velocity = np.array(vel)
        self.plant.state.orientation = self._mjMdl._quat2rot(q)
        self.plant.state.angular_velocity = np.array(ang_vel)
        
        pos2 = copy.deepcopy(self._mjMdl.data.qpos[7:10])
        q2 = copy.deepcopy(self._mjMdl.data.qpos[10:14])
        vel2 = copy.deepcopy(self._mjMdl.data.qvel[6:9])
        ang_vel2 = copy.deepcopy(self._mjMdl.data.qvel[9:12])
        
        self.reference.state.position = np.array(pos2)
        self.reference.state.velocity = np.array(vel2)
        self.reference.state.orientation = self._mjMdl._quat2rot(q2)
        self.reference.state.angular_velocity = np.array(ang_vel2)
        return

    def step(self, u):
        for _ in range(self._step_iter):
            u_plant = u[0:4]
            u_reference = u[4:8]
            # set control
            self._mjMdl.data.ctrl[self._mj_plant_ctrl_idx:self._mj_plant_ctrl_idx +
                                  4] = u_plant
            self._mjMdl.data.ctrl[self._mj_reference_ctrl_idx:self._mj_reference_ctrl_idx +
                                  4] = u_reference
            # mujoco simulation
            self._mjMdl._step_mujoco_simulation(self._nFrames)
            # update state
            self._query_latest_state()
        return

    def add_reference_marker(self, x, label="xQd"):
        self._mjMdl.add_marker_at(x, size=[0.01, 0.01, 0.01], label=label)
        return
