"""QuadrotorMujoco — MuJoCo physics backend with GLFW viewer."""

import copy

import numpy as np
from scipy.spatial.transform import Rotation as sp_rot

from ...manif import SO3, TSO3
from ...utils.logging import get_logger
from ..mujoco import MujocoModel
from .base import QuadrotorBase

_logger = get_logger(__name__)


class QuadrotorMujoco(QuadrotorBase):
    """Quadrotor backed by MuJoCo physics engine.

    Overrides step/reset to use MuJoCo instead of Euler ZOH.
    Reads mass/inertia from the MJCF XML model.
    """

    def __init__(self, **kwargs):
        self.render = kwargs.get("render", False)
        super().__init__(**kwargs)

        self._mjMdl = None
        self._mj_quad_index = 1

        self._attitude_zoh = kwargs.get("attitude_zoh", False)

        self._mjDt = 1.0 / 500.0
        self._step_iter = max(1, int(self.dt / self._mjDt))
        self._nFrames = 1
        if self._attitude_zoh:
            self._step_iter, self._nFrames = self._nFrames, self._step_iter

        # Load MuJoCo model
        self._mjMdl = MujocoModel("quadrotor_mj.xml", render=self.render)
        if self._force_type == QuadrotorBase.FORCE_TYPE.PROP_FORCES:
            self._ctrl_index = 0
        else:
            self._ctrl_index = 4

        # Read params from MuJoCo model
        self.mass = copy.deepcopy(self._mjMdl.model.body_mass[self._mj_quad_index])
        self.inertia = copy.deepcopy(self._mjMdl.model.body_inertia[self._mj_quad_index])

        # Reinit controllers with MuJoCo params
        self._init_default_controllers()

        if self.verbose:
            _logger.success("MuJoCo quadrotor loaded")

    @QuadrotorBase.mass.setter
    def mass(self, m):
        QuadrotorBase.mass.fset(self, m)
        if self._mjMdl is not None and self._mj_quad_index is not None:
            self._mjMdl.model.body_mass[self._mj_quad_index] = m
        else:
            _logger.warning("MuJoCo model not loaded, cannot set mass")

    @QuadrotorBase.inertia.setter
    def inertia(self, I):
        QuadrotorBase.inertia.fset(self, I)
        if self._mjMdl is not None and self._mj_quad_index is not None:
            if I.ndim == 2:
                I = np.diag(I)
            self._mjMdl.model.body_inertia[self._mj_quad_index] = I

    def reset(self, **kwargs):
        self.t = 0.0
        super().reset(**kwargs)
        self._mjMdl.reset()

        # Write state to MuJoCo
        self._mjMdl.data.qpos[:3] = self.state.position
        self._mjMdl.data.qvel[:3] = self.state.velocity
        qQ = sp_rot.from_matrix(np.asarray(self.state.orientation)).as_quat()
        self._mjMdl.data.qpos[3:7] = np.array([qQ[3], qQ[0], qQ[1], qQ[2]])
        self._mjMdl.data.qvel[3:6] = np.asarray(self.state.angular_velocity)

        self._query_latest_state()

        # Set visual markers
        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl._viewer.set_start(self.state.position.copy())
            target = self._pos_controller.setpoint(0.0)[0]
            self._mjMdl._viewer.set_target(target)

    def step(self, u):
        """Step MuJoCo simulation."""
        for _ in range(self._step_iter):
            wrench = self._repackage_input(u)
            # Convert to prop forces if MuJoCo expects individual motor commands
            if self._force_type == QuadrotorBase.FORCE_TYPE.PROP_FORCES:
                ctrl = self._wrench_to_propforces(wrench)
            else:
                ctrl = wrench
            self._mjMdl.data.ctrl[self._ctrl_index : self._ctrl_index + 4] = ctrl
            self._mjMdl._step_mujoco_simulation(self._nFrames)
            self._query_latest_state()
            self.t = self._mjMdl.data.time

        # Trail point every outer step
        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl._viewer.add_trail_point(self.state.position)

    def _query_latest_state(self):
        """Sync state from MuJoCo data."""
        self.t = self._mjMdl.data.time
        self.state.position = copy.deepcopy(self._mjMdl.data.qpos[:3])
        q = copy.deepcopy(self._mjMdl.data.qpos[3:7])
        self.state.velocity = copy.deepcopy(self._mjMdl.data.qvel[:3])
        ang_vel = copy.deepcopy(self._mjMdl.data.qvel[3:6])
        self.state.orientation = SO3(self._mjMdl._quat2rot(q))
        self.state.angular_velocity = TSO3(np.array(ang_vel))

    def add_reference_marker(self, x):
        self._mjMdl.add_marker_at(x, label="xQd")
