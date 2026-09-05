import copy

import numpy as np
from scipy.spatial.transform import Rotation as sp_rot

from ...manif import SO3, TSO3
from ...utils.logging import get_logger
from .._kwargs import check_reset_kwargs
from ..backend import MujocoModel
from ..quadrotor import QuadrotorBase
from . import _base as base

_logger = get_logger(__name__)


#: What :meth:`QuadrotorComparison.reset` acts on.  Both the plant and the
#: reference start from the same state, so there is one set of keywords.
_SUPPORTED_RESET_KWARGS = frozenset({"position", "velocity", "orientation", "angular_velocity"})


class QuadrotorComparison(base.BaseModel):
    # Always a live handle once ``__init__`` returns — it is loaded below and
    # never cleared, so every method may dereference it unconditionally.
    _mjMdl: MujocoModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Actual plant model.
        self.plant = QuadrotorBase()
        self.reference = QuadrotorBase()

        # mujoco model param handling
        self._mj_quad_index = None
        self._mj_payload_index = None
        self._mj_cable_index = None

        self._attitude_zoh = kwargs["attitude_zoh"] if "attitude_zoh" in kwargs else False
        self.render = kwargs["render"] if "render" in kwargs else False
        filename = kwargs["filename"] if "filename" in kwargs else "quadrotor_comparison.xml"

        # loading mujoco model
        self._mjMdl = MujocoModel(filename, render=self.render)

        # Read the integrator step from the loaded asset rather than assuming
        # it. `quadrotor_comparison.xml` is hand-written and carries no
        # `<option>`, so this is MuJoCo's 2 ms default and divides
        # `sim_timestep` exactly — but a hardcoded 1/500 would silently stop
        # being true the moment the asset gained one, and `step()` would then
        # advance a different amount of time from the one it reports.
        self._mjDt = float(self._mjMdl.model.opt.timestep)
        self._step_iter = max(1, int(round(self.sim_timestep / self._mjDt)))
        self._nFrames = 1
        achieved = self._step_iter * self._mjDt
        if abs(achieved - self.sim_timestep) > 1e-12:
            _logger.warning(
                "MuJoCo timestep %.6g s does not divide sim_timestep = %.6g s; each "
                "step() advances %.6g s instead.",
                self._mjDt,
                self.sim_timestep,
                achieved,
            )
        if self._attitude_zoh:
            self._step_iter, self._nFrames = self._nFrames, self._step_iter

        self._plant_body_index = 1
        self._reference_body_index = 2
        self._plant_ctrl_index = 0
        self._reference_ctrl_index = 4

        # read inertial parameters from mujoco model
        self.plant.mass = copy.deepcopy(self._mjMdl.model.body_mass[self._plant_body_index])
        self.plant.inertia = copy.deepcopy(self._mjMdl.model.body_inertia[self._plant_body_index])
        self.reference.mass = copy.deepcopy(self._mjMdl.model.body_mass[self._reference_body_index])
        self.reference.inertia = copy.deepcopy(
            self._mjMdl.model.body_inertia[self._reference_body_index]
        )

        # reinitialize controllers after loading mujoco model & params
        self.plant._init_default_controllers()
        self.reference._init_default_controllers()

        if self.verbose:
            _logger.info("Mujoco model loaded")
        return

    def set_mass(self, m, plant=True):
        if plant:
            self.plant.mass = m
            self._mjMdl.model.body_mass[self._plant_body_index] = m
        else:
            self.reference.mass = m
            self._mjMdl.model.body_mass[self._reference_body_index] = m
        return

    def set_inertia(self, I, plant=True):
        if I.ndim == 2:
            I = np.diag(I)
        if plant:
            self.plant.inertia = I
            self._mjMdl.model.body_inertia[self._plant_body_index] = I
        else:
            self.reference.inertia = I
            self._mjMdl.model.body_inertia[self._reference_body_index] = I
        return

    def reset(self, **kwargs):
        """Reset plant and reference to a common state.

        Recognised kwargs: ``position``, ``velocity``, ``orientation``,
        ``angular_velocity``.

        Raises:
            ValueError: for any other keyword.
        """
        check_reset_kwargs(self, kwargs, _SUPPORTED_RESET_KWARGS)
        self.t = 0.0
        self.plant.state.reset()
        self.reference.state.reset()
        for key in _SUPPORTED_RESET_KWARGS:
            if key in kwargs:
                setattr(self.plant.state, key, kwargs[key])
                setattr(self.reference.state, key, kwargs[key])
        self._mjMdl.reset()

        # Set plant position
        self._mjMdl.data.qpos[:3] = self.plant.state.position
        self._mjMdl.data.qvel[:3] = self.plant.state.velocity
        qQ = sp_rot.from_matrix(np.asarray(self.plant.state.orientation)).as_quat()
        self._mjMdl.data.qpos[3:7] = np.array([qQ[3], qQ[0], qQ[1], qQ[2]])
        self._mjMdl.data.qvel[3:6] = np.asarray(self.plant.state.angular_velocity)

        # Set reference position
        self._mjMdl.data.qpos[7:10] = self.reference.state.position
        self._mjMdl.data.qvel[6:9] = self.reference.state.velocity
        qQ = sp_rot.from_matrix(np.asarray(self.reference.state.orientation)).as_quat()
        self._mjMdl.data.qpos[10:14] = np.array([qQ[3], qQ[0], qQ[1], qQ[2]])
        self._mjMdl.data.qvel[9:12] = np.asarray(self.reference.state.angular_velocity)

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
        self.plant.state.orientation = SO3(self._mjMdl._quat2rot(q))
        self.plant.state.angular_velocity = TSO3(ang_vel)

        pos2 = copy.deepcopy(self._mjMdl.data.qpos[7:10])
        q2 = copy.deepcopy(self._mjMdl.data.qpos[10:14])
        vel2 = copy.deepcopy(self._mjMdl.data.qvel[6:9])
        ang_vel2 = copy.deepcopy(self._mjMdl.data.qvel[9:12])

        self.reference.state.position = np.array(pos2)
        self.reference.state.velocity = np.array(vel2)
        self.reference.state.orientation = SO3(self._mjMdl._quat2rot(q2))
        self.reference.state.angular_velocity = TSO3(ang_vel2)
        return

    def step(self, u):
        for _ in range(self._step_iter):
            u_plant = u[0:4]
            u_reference = u[4:8]
            # set control
            self._mjMdl.data.ctrl[self._plant_ctrl_index : self._plant_ctrl_index + 4] = u_plant
            self._mjMdl.data.ctrl[self._reference_ctrl_index : self._reference_ctrl_index + 4] = (
                u_reference
            )
            # mujoco simulation
            self._mjMdl._step_mujoco_simulation(self._nFrames)
            # update state
            self._query_latest_state()
        return

    def add_reference_marker(self, x, label="xQd"):
        self._mjMdl.add_marker_at(x, size=[0.01, 0.01, 0.01], label=label)
        return
