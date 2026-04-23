"""QuadrotorCsPayloadMujoco — MuJoCo physics backend for quadrotor with cable-suspended payload."""

import copy
import enum

import numpy as np

from ...manif import S2, SO3, TS2, TSO3
from ...utils.logging import get_logger
from ..mujoco import MujocoModel
from .base import QuadrotorCsPayloadBase

_logger = get_logger(__name__)


class QuadrotorCsPayloadMujoco(QuadrotorCsPayloadBase):
    """Quadrotor with cable-suspended payload backed by MuJoCo physics engine.

    Overrides step/reset to use MuJoCo instead of Euler ZOH.
    Supports three cable modeling approaches:
        - TENDON: spatial tendon constraint (fast, less realistic)
        - LINKS: rigid cylinder links with ball joints (slower, most stable)
        - CABLE: MuJoCo 2.3+ composite cable (experimental)
    """

    class MODEL_TYPE(enum.Enum):
        LINKS = 0
        TENDON = 1
        CABLE = 2

    def __init__(self, **kwargs):
        self.render = kwargs.get("render", False)
        super().__init__(**kwargs)

        self._mjMdl = None
        self._mj_quad_index = None
        self._mj_payload_index = None
        self._mj_cable_index = None

        self._model_type = self.MODEL_TYPE.LINKS

        self._attitude_zoh = kwargs.get("attitude_zoh", False)
        if "cable_model" in kwargs:
            cable_model = kwargs["cable_model"]
            if cable_model == "tendon":
                self._model_type = self.MODEL_TYPE.TENDON
            elif cable_model == "cable":
                self._model_type = self.MODEL_TYPE.CABLE
                _logger.warning(
                    "cable model 'cable' is not officially supported; "
                    "included for experimentation only"
                )
            elif cable_model == "links":
                self._model_type = self.MODEL_TYPE.LINKS
            else:
                raise ValueError(f"Invalid cable_model type: {cable_model!r}")

        # Initialize the appropriate MuJoCo model
        if self._model_type == self.MODEL_TYPE.TENDON:
            self._init_tendon_model()
        elif self._model_type == self.MODEL_TYPE.LINKS:
            self._init_nlink_model()
        elif self._model_type == self.MODEL_TYPE.CABLE:
            self._init_composite_model()

        if self._force_type == self.FORCE_TYPE.PROP_FORCES:
            self._ctrl_index = 0
        else:
            self._ctrl_index = 4

        self._mjDt = 1.0 / 500.0
        self._step_iter = max(1, int(self.dt / self._mjDt))
        self._nFrames = 1
        if self._attitude_zoh:
            self._step_iter, self._nFrames = self._nFrames, self._step_iter

        # Read params from MuJoCo model
        if self._mj_quad_index is not None:
            self.mass = copy.deepcopy(self._mjMdl.model.body_mass[self._mj_quad_index])
            self.inertia = copy.deepcopy(self._mjMdl.model.body_inertia[self._mj_quad_index])
        if self._mj_payload_index is not None:
            self.payload_mass = float(
                copy.deepcopy(self._mjMdl.model.body_mass[self._mj_payload_index])
            )

        # Reinit controllers with MuJoCo params
        self._init_default_controllers()

        if self.verbose:
            _logger.info("MuJoCo quadrotor-cspayload loaded (%s)", self._model_type.name)

    # ─── Model initialization ──────────────────────────────────────

    def _init_tendon_model(self):
        self._mjMdl = MujocoModel(model_path="quad_payload_mj.xml", render=self.render)
        self._mj_quad_pos_index = 0
        self._mj_quad_quat_index = 3
        self._mj_quad_vel_index = 0
        self._mj_quad_omega_index = 3

        self._mj_payload_pos_index = 7
        self._mj_payload_quat_index = 10
        self._mj_payload_vel_index = 6
        self._mj_payload_omega_index = 9

        self._mj_cable_index = 0
        self._mj_quad_index = 1
        self._mj_payload_index = 2

    def _init_nlink_model(self):
        self._mjMdl = MujocoModel(model_path="quad_flxcbl_mj.xml", render=self.render)
        self._mj_quad_pos_index = 0
        self._mj_quad_quat_index = 3
        self._mj_quad_vel_index = 0
        self._mj_quad_omega_index = 3

        self._mj_quad_index = 1
        self._mj_payload_index = 27
        self._mj_data__payload_pos_prev = np.zeros(3)
        self._mj_data__payload_vel_initialized = False

    def _init_composite_model(self):
        self._mjMdl = MujocoModel(model_path="quad_cable_mj.xml", render=self.render)
        self._mj_quad_pos_index = 0
        self._mj_quad_quat_index = 3
        self._mj_quad_vel_index = 0
        self._mj_quad_omega_index = 3

        self._mj_payload_pos_index = 167
        self._mj_payload_quat_index = 170
        self._mj_payload_vel_index = 126
        self._mj_payload_omega_index = 129

    # ─── Property setters (sync to MuJoCo) ─────────────────────────

    @QuadrotorCsPayloadBase.mass.setter
    def mass(self, value):
        QuadrotorCsPayloadBase.mass.fset(self, value)
        if self._mjMdl is not None and self._mj_quad_index is not None:
            self._mjMdl.model.body_mass[self._mj_quad_index] = value

    @QuadrotorCsPayloadBase.inertia.setter
    def inertia(self, value):
        QuadrotorCsPayloadBase.inertia.fset(self, value)
        if self._mjMdl is not None and self._mj_quad_index is not None:
            if value.ndim == 2:
                value = np.diag(value)
            self._mjMdl.model.body_inertia[self._mj_quad_index] = value

    @QuadrotorCsPayloadBase.payload_mass.setter
    def payload_mass(self, value):
        QuadrotorCsPayloadBase.payload_mass.fset(self, value)
        if self._mjMdl is not None and self._mj_payload_index is not None:
            self._mjMdl.model.body_mass[self._mj_payload_index] = value

    @QuadrotorCsPayloadBase.cable_length.setter
    def cable_length(self, value):
        QuadrotorCsPayloadBase.cable_length.fset(self, value)
        if self._mjMdl is not None and self._mj_cable_index is not None:
            from ..mujoco import mujoco

            self._mjMdl.model.tendon_length0[self._mj_cable_index] = value
            self._mjMdl.model.tendon_limited[self._mj_cable_index] = True
            self._mjMdl.model.tendon_range[self._mj_cable_index][1] = value
            self._mjMdl.model.tendon_lengthspring[self._mj_cable_index] = value
            mujoco.mj_tendon(self._mjMdl.model, self._mjMdl.data)

    # ─── Reset ─────────────────────────────────────────────────────

    def reset(self, **kwargs):
        self.t = 0.0
        self._mjMdl.reset()
        super().reset(**kwargs)

        if self._model_type == self.MODEL_TYPE.TENDON:
            self._reset_tendon_model()
        elif self._model_type == self.MODEL_TYPE.LINKS:
            self._reset_nlink_model()
        elif self._model_type == self.MODEL_TYPE.CABLE:
            self._reset_composite_model()

        self._query_latest_state()

    def _reset_tendon_model(self):
        from scipy.spatial.transform import Rotation as sp_rot

        _qQ = sp_rot.from_matrix(np.asarray(self.state.orientation)).as_quat()
        quat_Q = np.array([_qQ[3], _qQ[0], _qQ[1], _qQ[2]])

        i = self._mj_quad_pos_index
        self._mjMdl.data.qpos[i : i + 3] = self.state.position
        i = self._mj_quad_quat_index
        self._mjMdl.data.qpos[i : i + 4] = quat_Q
        i = self._mj_quad_vel_index
        self._mjMdl.data.qvel[i : i + 3] = self.state.velocity
        i = self._mj_quad_omega_index
        self._mjMdl.data.qvel[i : i + 3] = np.asarray(self.state.angular_velocity)

        i = self._mj_payload_pos_index
        self._mjMdl.data.qpos[i : i + 3] = self.state.payload_position
        i = self._mj_payload_quat_index
        self._mjMdl.data.qpos[i : i + 4] = np.array([1.0, 0.0, 0.0, 0.0])
        i = self._mj_payload_vel_index
        self._mjMdl.data.qvel[i : i + 3] = self.state.payload_velocity
        i = self._mj_payload_omega_index
        self._mjMdl.data.qvel[i : i + 3] = np.zeros(3)

    def _reset_nlink_model(self):
        i = self._mj_quad_pos_index
        self._mjMdl.data.qpos[i : i + 3] = self.state.position
        self._mj_data__payload_pos_prev = np.zeros(3)
        self._mj_data__payload_vel_initialized = False

    def _reset_composite_model(self):
        # Composite cable reset is not fully implemented;
        _logger.warning("Composite cable reset is not supported")
        pass

    # ─── Step ──────────────────────────────────────────────────────

    def step(self, u, desired_att=None):
        """Step MuJoCo simulation."""
        for _ in range(self._step_iter):
            wrench = self._repackage_input(u, desired_att=desired_att)
            if self._force_type == self.FORCE_TYPE.PROP_FORCES:
                ctrl = self._wrench_to_propforces(wrench)
            else:
                ctrl = wrench
            self._mjMdl.data.ctrl[self._ctrl_index : self._ctrl_index + 4] = ctrl
            self._mjMdl._step_mujoco_simulation(self._nFrames)
            self._query_latest_state()
            self.t = self._mjMdl.data.time

        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl._viewer.add_trail_point(self.state.position)
            self._mjMdl._viewer.add_trail_point(
                self.state.payload_position, key=1, rgba=[0.2, 0.2, 1.0, 0.6]
            )

    # ─── State query ───────────────────────────────────────────────

    def _query_latest_state(self):
        """Sync state from MuJoCo data."""
        if self._model_type == self.MODEL_TYPE.TENDON:
            self._query_state_tendon_model()
        elif self._model_type == self.MODEL_TYPE.CABLE:
            self._query_state_composite_model()
        elif self._model_type == self.MODEL_TYPE.LINKS:
            self._query_state_nlink_model()

        self.t = self._mjMdl.data.time

        # Quadrotor state
        q = copy.deepcopy(self._mj_data__quad_quat)
        self.state.position = copy.deepcopy(self._mj_data__quad_pos)
        self.state.velocity = copy.deepcopy(self._mj_data__quad_vel)
        self.state.orientation = SO3(self._mjMdl._quat2rot(q))
        self.state.angular_velocity = TSO3(np.array(copy.deepcopy(self._mj_data__quad_angvel)))

        # Payload state
        self.state.payload_position = copy.deepcopy(self._mj_data__payload_pos)
        self.state.payload_velocity = copy.deepcopy(self._mj_data__payload_vel)

        # Derive cable attitude and angular velocity from positions
        p = self.state.payload_position - self.state.position
        norm_p = np.linalg.norm(p)
        if norm_p > 1e-8:
            self.state.cable_attitude = S2(p / norm_p)
        dq = self.state.payload_velocity - self.state.velocity
        self.state.cable_angular_velocity = TS2(np.cross(np.asarray(self.state.cable_attitude), dq))

    def _query_state_tendon_model(self):
        i = self._mj_quad_pos_index
        self._mj_data__quad_pos = self._mjMdl.data.qpos[i : i + 3]
        i = self._mj_quad_quat_index
        self._mj_data__quad_quat = self._mjMdl.data.qpos[i : i + 4]
        i = self._mj_quad_vel_index
        self._mj_data__quad_vel = self._mjMdl.data.qvel[i : i + 3]
        i = self._mj_quad_omega_index
        self._mj_data__quad_angvel = self._mjMdl.data.qvel[i : i + 3]

        i = self._mj_payload_pos_index
        self._mj_data__payload_pos = self._mjMdl.data.qpos[i : i + 3]
        i = self._mj_payload_vel_index
        self._mj_data__payload_vel = self._mjMdl.data.qvel[i : i + 3]

    def _query_state_composite_model(self):
        i = self._mj_quad_pos_index
        self._mj_data__quad_pos = self._mjMdl.data.qpos[i : i + 3]
        i = self._mj_quad_quat_index
        self._mj_data__quad_quat = self._mjMdl.data.qpos[i : i + 4]
        i = self._mj_quad_vel_index
        self._mj_data__quad_vel = self._mjMdl.data.qvel[i : i + 3]
        i = self._mj_quad_omega_index
        self._mj_data__quad_angvel = self._mjMdl.data.qvel[i : i + 3]

        i = self._mj_payload_pos_index
        self._mj_data__payload_pos = self._mjMdl.data.qpos[i : i + 3]
        i = self._mj_payload_vel_index
        self._mj_data__payload_vel = self._mjMdl.data.qvel[i : i + 3]

    def _query_state_nlink_model(self):
        i = self._mj_quad_pos_index
        self._mj_data__quad_pos = self._mjMdl.data.qpos[i : i + 3]
        i = self._mj_quad_quat_index
        self._mj_data__quad_quat = self._mjMdl.data.qpos[i : i + 4]
        i = self._mj_quad_vel_index
        self._mj_data__quad_vel = self._mjMdl.data.qvel[i : i + 3]
        i = self._mj_quad_omega_index
        self._mj_data__quad_angvel = self._mjMdl.data.qvel[i : i + 3]

        self._mj_data__payload_pos = self._mjMdl.data.xpos[self._mj_payload_index]
        self._mj_data__payload_vel = (
            self._mj_data__payload_pos - self._mj_data__payload_pos_prev
        ) / self._mjMdl.model.opt.timestep
        if not self._mj_data__payload_vel_initialized:
            self._mj_data__payload_vel = np.zeros(3)
            self._mj_data__payload_vel_initialized = True
        self._mj_data__payload_pos_prev = copy.deepcopy(self._mj_data__payload_pos)
