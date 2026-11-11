"""QuadrotorCsPayloadMujoco — MuJoCo physics backend for quadrotor with cable-suspended payload."""

import copy
import enum

import mujoco
import numpy as np

from ...manif import S2, SO3, TS2, TSO3
from ...utils.logging import get_logger
from .._kwargs import check_reset_kwargs
from ..mujoco import MujocoModel
from .base import QuadrotorCsPayloadBase

_logger = get_logger(__name__)

#: The cable models :class:`QuadrotorCsPayloadMujoco` offers, and their MJCF.
CABLE_MODELS = {"tendon": "quad_payload_mj.xml", "links": "quad_flxcbl_mj.xml"}

#: The reset keywords the link chain honours. :meth:`QuadrotorCsPayloadMujoco.reset`
#: places the airframe and turns the chain, straight, about its first ball joint:
#: the payload's velocity then follows from the airframe's and the cable's, so
#: ``payload_velocity`` is not one of them.
_LINKS_RESET_KWARGS = (
    "position",
    "velocity",
    "orientation",
    "angular_velocity",
    "payload_position",
    "cable_attitude",
    "cable_angular_velocity",
)

#: The tendon's limit time constant, in MuJoCo timesteps: ``solreflimit`` is
#: written as ``(TENDON_LIMIT_TIMESTEPS · h, 1)`` from the timestep ``h = dt/N``,
#: the stiffest limit MuJoCo integrates stably, so the cable holds its length.
TENDON_LIMIT_TIMESTEPS = 2.0

#: The MuJoCo warnings that mean the integration diverged: MuJoCo answers each
#: by resetting the state, its clock included.
_DIVERGENCE_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQPOS,
)


class QuadrotorCsPayloadMujoco(QuadrotorCsPayloadBase):
    """Quadrotor with cable-suspended payload backed by MuJoCo physics engine.

    Overrides step/reset to use MuJoCo instead of Euler ZOH.
    Two cable models, selected with ``cable_model=``:

    - ``"tendon"`` (default): a massless spatial tendon with an upper length
      limit ``ℓ`` between the airframe and a free payload body; it goes slack
      under compression and carries no force then. Its limit is stiffened to a
      time constant of two MuJoCo timesteps.
    - ``"links"``: a 10-link chain of ball joints carrying 10 g of cable mass,
      with the payload welded to its end. :attr:`cable_mass` reports the
      chain's mass, and the default payload controller carries it as part of
      the payload's.

    Note:
        On the tendon and the link chain the MJCF is the source of the machine:
        the airframe's mass ``m_Q`` and full inertia tensor ``J`` (MuJoCo's
        principal moments rotated back by ``body_iquat``), the payload mass
        ``m_L`` and the cable length ``ℓ`` (the tendon's length limit; the
        chain's airframe-to-payload distance as built) are read from it, and
        the default controllers are rebuilt with them. Both MJCFs describe the
        M1 airframe with a 0.15 kg payload on a 1.000 m cable, the analytical
        model's defaults.

        MuJoCo's timestep is not the MJCF's: it is written as ``dt / N`` from
        ``physics_rate_hz`` (1600 Hz, eight steps of 0.625 ms, by default), so
        a ``step()`` advances exactly ``dt``. The clock counts substeps and is
        written into ``data.time``. The attitude loop closes at
        ``attitude_rate_hz`` (800 Hz by default) on every input path.
        ``attitude_zoh`` is not a keyword (it raises ``TypeError``).

        ``cable_model="links"`` reports its chain as this class's single
        point-mass cable: one direction from the airframe to the payload, and
        the payload's velocity as MuJoCo computes it for the payload body. Its
        ``reset()`` places the airframe and turns the straight chain about its
        first ball joint to ``cable_attitude`` at ``cable_angular_velocity``;
        ``payload_velocity`` follows from those and is refused.

        :meth:`step` raises ``RuntimeError`` when MuJoCo reports a diverged
        integration (a bad acceleration, velocity or position, after which it
        resets the state) or its time runs backwards.
    """

    class MODEL_TYPE(enum.Enum):
        LINKS = 0
        TENDON = 1

    # Always a live handle once ``__init__`` returns: exactly one of the two
    # ``_init_*_model`` branches below runs, and neither can leave it unset.
    _mjMdl: MujocoModel

    def __init__(self, **kwargs):
        if "attitude_zoh" in kwargs:
            raise TypeError(
                "attitude_zoh is not a keyword: the attitude loop runs at attitude_rate_hz "
                "on every input path (None for once per step)"
            )
        self.render = kwargs.get("render", False)
        super().__init__(**kwargs)

        self._mj_quad_index = None
        self._mj_payload_index = None
        self._mj_cable_index = None

        cable_model = kwargs.get("cable_model", "tendon")
        if cable_model not in CABLE_MODELS:
            raise ValueError(
                f"Invalid cable_model {cable_model!r}; valid choices: {', '.join(CABLE_MODELS)}"
            )
        if cable_model == "tendon":
            self._model_type = self.MODEL_TYPE.TENDON
            self._init_tendon_model()
        else:
            self._model_type = self.MODEL_TYPE.LINKS
            self._init_nlink_model()

        if self._force_type == self.FORCE_TYPE.PROP_FORCES:
            self._ctrl_index = 0
        else:
            self._ctrl_index = 4

        # The physics rate, not the MJCF, sets MuJoCo's timestep: N steps of
        # dt/N per step(), the base model's own N, so a step advances exactly dt.
        self._step_iter = self._inner_loop_steps
        self._mjDt = self.dt / self._step_iter
        self._mjMdl.model.opt.timestep = self._mjDt
        if self._model_type == self.MODEL_TYPE.TENDON:
            self._mjMdl.model.tendon_solref_lim[self._mj_cable_index] = [
                TENDON_LIMIT_TIMESTEPS * self._mjDt,
                1.0,
            ]

        self._read_mjcf_parameters()

        # Rebuild the controllers for the machine the MJCF describes
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
        m = self._mjMdl.model
        self._mj_quad_pos_index = 0
        self._mj_quad_quat_index = 3
        self._mj_quad_vel_index = 0
        self._mj_quad_omega_index = 3

        self._mj_quad_index = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        self._mj_payload_index = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "payload")
        # The chain's links: every body below the airframe but the payload.
        self._mj_cable_bodies = [
            b
            for b in range(m.nbody)
            if b not in (self._mj_quad_index, self._mj_payload_index)
            and self._is_below(m, b, self._mj_quad_index)
        ]
        joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cable_joint_0")
        self._mj_cable_joint_qpos = int(m.jnt_qposadr[joint])
        self._mj_cable_joint_qvel = int(m.jnt_dofadr[joint])
        self._mj_payload_velocity = np.zeros(6)

    @staticmethod
    def _is_below(m, body, ancestor):
        """Whether ``body`` is in the subtree of ``ancestor`` (and is not ``ancestor``)."""
        while body > 0:
            body = int(m.body_parentid[body])
            if body == ancestor:
                return True
        return False

    def _mjcf_inertia(self, body):
        """Body ``body``'s inertia tensor in its own frame, ``R diag(I) Rᵀ``.

        MuJoCo stores principal moments ``I`` in the frame ``body_iquat``; with
        an explicit diagonal ``<inertial>`` the rotation is the identity.
        """
        m = self._mjMdl.model
        moments = np.array(m.body_inertia[body], dtype=float)
        if np.array_equal(m.body_iquat[body], [1.0, 0.0, 0.0, 0.0]):
            return np.diag(moments)
        flat = np.zeros(9)
        mujoco.mju_quat2Mat(flat, m.body_iquat[body])
        R = flat.reshape(3, 3)
        return R @ np.diag(moments) @ R.T

    def _mjcf_chain_length(self):
        """The link chain's airframe-to-payload distance in the MJCF's own configuration."""
        m = self._mjMdl.model
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        return float(np.linalg.norm(d.xpos[self._mj_payload_index] - d.xpos[self._mj_quad_index]))

    def _read_mjcf_parameters(self):
        """Take ``m_Q``, ``J``, ``m_L`` and ``ℓ`` from the tendon's or the chain's MJCF."""
        m = self._mjMdl.model
        self.mass = float(m.body_mass[self._mj_quad_index])
        self.inertia = self._mjcf_inertia(self._mj_quad_index)
        self.payload_mass = float(m.body_mass[self._mj_payload_index])
        if self._model_type == self.MODEL_TYPE.TENDON:
            self.cable_length = float(m.tendon_range[self._mj_cable_index][1])
        else:
            self._chain_length = self._mjcf_chain_length()
            self.cable_length = self._chain_length

    @property
    def cable_mass(self) -> float:
        """Mass of the cable itself, in kg.

        ``0.0`` on the tendon, which is massless. On the link chain, the links'
        masses in the MJCF (10 g), which the default payload controller carries
        as part of the payload's mass.
        """
        bodies = getattr(self, "_mj_cable_bodies", None)
        if not bodies:
            return 0.0
        return float(self._mjMdl.model.body_mass[bodies].sum())

    # ─── Property setters (sync to MuJoCo) ─────────────────────────

    # mypy cannot model a subclass re-opening a base class property to add a
    # setter (python/mypy#5936); the pattern itself is correct at runtime.
    @QuadrotorCsPayloadBase.mass.setter  # type: ignore[attr-defined]
    def mass(self, value):
        QuadrotorCsPayloadBase.mass.fset(self, value)  # type: ignore[attr-defined]
        if self._mjMdl is not None and self._mj_quad_index is not None:
            self._mjMdl.model.body_mass[self._mj_quad_index] = value

    @QuadrotorCsPayloadBase.inertia.setter  # type: ignore[attr-defined]  # see mass, above
    def inertia(self, value):
        QuadrotorCsPayloadBase.inertia.fset(self, value)  # type: ignore[attr-defined]
        if self._mjMdl is not None and self._mj_quad_index is not None:
            # MuJoCo holds principal moments and the frame they are taken in.
            J = self._inertia
            if np.count_nonzero(J - np.diag(np.diag(J))) == 0:
                moments, quat = np.diag(J).copy(), np.array([1.0, 0.0, 0.0, 0.0])
            else:
                moments, axes = np.linalg.eigh(J)
                if np.linalg.det(axes) < 0.0:
                    axes[:, 0] = -axes[:, 0]
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, np.ascontiguousarray(axes).ravel())
            self._mjMdl.model.body_inertia[self._mj_quad_index] = moments
            self._mjMdl.model.body_iquat[self._mj_quad_index] = quat

    @QuadrotorCsPayloadBase.payload_mass.setter  # type: ignore[attr-defined]  # see mass, above
    def payload_mass(self, value):
        QuadrotorCsPayloadBase.payload_mass.fset(self, value)  # type: ignore[attr-defined]
        if self._mjMdl is not None and self._mj_payload_index is not None:
            self._mjMdl.model.body_mass[self._mj_payload_index] = value

    @QuadrotorCsPayloadBase.cable_length.setter  # type: ignore[attr-defined]  # see mass, above
    def cable_length(self, value):
        if self._model_type == self.MODEL_TYPE.LINKS and hasattr(self, "_chain_length"):
            if abs(float(value) - self._chain_length) > 1e-9:
                raise ValueError(
                    f"cable_length = {value} disagrees with the link chain, which is "
                    f"{self._chain_length:.6f} m long; the chain's length is set by its MJCF"
                )
        QuadrotorCsPayloadBase.cable_length.fset(self, value)  # type: ignore[attr-defined]
        if self._mjMdl is not None and self._mj_cable_index is not None:
            self._mjMdl.model.tendon_length0[self._mj_cable_index] = value
            self._mjMdl.model.tendon_limited[self._mj_cable_index] = True
            self._mjMdl.model.tendon_range[self._mj_cable_index][1] = value
            self._mjMdl.model.tendon_lengthspring[self._mj_cable_index] = value
            mujoco.mj_tendon(self._mjMdl.model, self._mjMdl.data)

    # ─── Reset ─────────────────────────────────────────────────────

    def reset(self, **kwargs):
        """Reset as :meth:`QuadrotorCsPayloadBase.reset`, then write the state into MuJoCo.

        The velocities are the base model's: the payload's is
        ``payload_velocity`` when given, else ``v_Q + ℓ (ω × q)``; the tendon
        writes both bodies' velocities from them, so the cable rate reads back.

        Raises:
            ValueError: for both ``payload_velocity`` and
                ``cable_angular_velocity``; on the link chain, for
                ``payload_velocity`` or any keyword it does not honour, or for
                both ``position`` and ``payload_position``.
        """
        self._check_velocity_kwargs(kwargs)
        if self._model_type == self.MODEL_TYPE.LINKS:
            check_reset_kwargs(
                self,
                kwargs,
                _LINKS_RESET_KWARGS,
                detail=(
                    "the link chain starts straight, so the payload's velocity follows from "
                    "the airframe's and the cable's"
                ),
            )
            if "position" in kwargs and "payload_position" in kwargs:
                raise ValueError(
                    "the link chain places the payload from the airframe: pass position or "
                    "payload_position, not both"
                )
        self.t = 0.0
        self._mjMdl.reset()
        super().reset(**kwargs)

        if self._model_type == self.MODEL_TYPE.TENDON:
            self._reset_tendon_model()
        else:
            self._reset_nlink_model()

        self._mjMdl.data.time = self.t
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
        """Place the airframe, then turn the straight chain about its first ball joint.

        The chain hangs along the airframe's ``−b₃`` in the MJCF, so the joint's
        rotation is the minimal one taking ``−e₃`` to ``Rᵀ q``, and its rate (in
        the first link's frame) is the cable's angular velocity relative to the
        airframe's. ``mj_forward`` then computes the payload's position and
        velocity from the joints.
        """
        from scipy.spatial.transform import Rotation as sp_rot

        d = self._mjMdl.data
        R = np.asarray(self.state.orientation)
        _qQ = sp_rot.from_matrix(R).as_quat()
        i = self._mj_quad_pos_index
        d.qpos[i : i + 3] = self.state.position
        i = self._mj_quad_quat_index
        d.qpos[i : i + 4] = np.array([_qQ[3], _qQ[0], _qQ[1], _qQ[2]])
        i = self._mj_quad_vel_index
        d.qvel[i : i + 3] = self.state.velocity
        i = self._mj_quad_omega_index
        d.qvel[i : i + 3] = np.asarray(self.state.angular_velocity)

        down = np.array([0.0, 0.0, -1.0])
        target = R.T @ np.asarray(self.state.cable_attitude)
        axis = np.cross(down, target)
        sin, cos = float(np.linalg.norm(axis)), float(down @ target)
        if sin > 1e-12:
            Rj = sp_rot.from_rotvec(axis / sin * np.arctan2(sin, cos)).as_matrix()
        elif cos > 0.0:
            Rj = np.eye(3)
        else:
            Rj = sp_rot.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
        qj = sp_rot.from_matrix(Rj).as_quat()
        a = self._mj_cable_joint_qpos
        d.qpos[a : a + 4] = np.array([qj[3], qj[0], qj[1], qj[2]])
        omega = np.asarray(self.state.cable_angular_velocity)
        relative = omega - R @ np.asarray(self.state.angular_velocity)
        v = self._mj_cable_joint_qvel
        d.qvel[v : v + 3] = (R @ Rj).T @ relative
        mujoco.mj_forward(self._mjMdl.model, d)

    # ─── Step ──────────────────────────────────────────────────────

    def step(self, u, desired_att=None):
        """One outer step of ``dt``: :attr:`attitude_rate_hz` ``· dt`` attitude
        updates, each held over an equal share of the MuJoCo steps.

        Raises:
            RuntimeError: if MuJoCo reports a diverged integration or its time
                runs backwards; MuJoCo has then reset the state, and a loop on
                :attr:`t` would never reach its end.
        """
        hold = self._step_iter // self._attitude_updates
        for _ in range(self._attitude_updates):
            wrench = self._repackage_input(u, desired_att=desired_att)
            if self._force_type == self.FORCE_TYPE.PROP_FORCES:
                ctrl = self._wrench_to_propforces(wrench)
            else:
                ctrl = wrench
            self._mjMdl.data.ctrl[self._ctrl_index : self._ctrl_index + 4] = ctrl
            time_before = float(self._mjMdl.data.time)
            warnings_before = self._divergence_warnings()
            self._mjMdl._step_mujoco_simulation(hold)
            self._check_divergence(time_before, warnings_before)
            # The clock counts substeps (see QuadrotorCsPayloadBase._advance_clock)
            # rather than reading MuJoCo's, which accumulates `time += timestep`;
            # MuJoCo's is overwritten so the two agree.
            self._advance_clock(hold)
            self._mjMdl.data.time = self.t
            self._query_latest_state()

        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl._viewer.add_trail_point(self.state.position)
            self._mjMdl._viewer.add_trail_point(
                self.state.payload_position, key=1, rgba=[0.2, 0.2, 1.0, 0.6]
            )

    def _divergence_warnings(self) -> int:
        """How many divergence warnings MuJoCo has counted since its last reset."""
        warning = self._mjMdl.data.warning
        return sum(int(warning[w].number) for w in _DIVERGENCE_WARNINGS)

    def _check_divergence(self, time_before: float, warnings_before: int) -> None:
        """Raise if the MuJoCo steps just taken diverged."""
        data = self._mjMdl.data
        if self._divergence_warnings() > warnings_before or data.time < time_before:
            raise RuntimeError(
                f"MuJoCo integration diverged at t = {self.t:.4f} s "
                f"(cable_model {self._model_type.name.lower()!r}): MuJoCo reported a bad "
                "acceleration, velocity or position and reset the state "
                f"(data.time {time_before:.4f} s -> {float(data.time):.4f} s)"
            )

    # ─── State query ───────────────────────────────────────────────

    def _query_latest_state(self):
        """Sync state from MuJoCo data (the clock is the model's own, see :meth:`step`)."""
        if self._model_type == self.MODEL_TYPE.TENDON:
            self._query_state_tendon_model()
        else:
            self._query_state_nlink_model()

        # Quadrotor state
        q = copy.deepcopy(self._mj_data__quad_quat)
        self.state.position = copy.deepcopy(self._mj_data__quad_pos)
        self.state.velocity = copy.deepcopy(self._mj_data__quad_vel)
        self.state.orientation = SO3(self._mjMdl._quat2rot(q))
        self.state.angular_velocity = TSO3(np.array(copy.deepcopy(self._mj_data__quad_angvel)))

        # Payload state
        self.state.payload_position = copy.deepcopy(self._mj_data__payload_pos)
        self.state.payload_velocity = copy.deepcopy(self._mj_data__payload_vel)

        # Cable attitude and angular velocity from the two bodies:
        # q = (x_L - x_Q)/‖·‖ and ω = q × q̇ with q̇ = (v_L - v_Q)/ℓ.
        p = self.state.payload_position - self.state.position
        norm_p = np.linalg.norm(p)
        if norm_p > 1e-8:
            self.state.cable_attitude = S2(p / norm_p)
        dq = (self.state.payload_velocity - self.state.velocity) / self._cable_length
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

    def _query_state_nlink_model(self):
        # The payload sits at the end of the chain's ball joints, so its position
        # and velocity are the body's world-frame kinematics. mj_step leaves
        # xpos and cvel at the configuration it started from, so they are
        # recomputed for the current qpos/qvel first.
        m, d = self._mjMdl.model, self._mjMdl.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)

        i = self._mj_quad_pos_index
        self._mj_data__quad_pos = d.qpos[i : i + 3]
        i = self._mj_quad_quat_index
        self._mj_data__quad_quat = d.qpos[i : i + 4]
        i = self._mj_quad_vel_index
        self._mj_data__quad_vel = d.qvel[i : i + 3]
        i = self._mj_quad_omega_index
        self._mj_data__quad_angvel = d.qvel[i : i + 3]

        self._mj_data__payload_pos = d.xpos[self._mj_payload_index]
        # [angular; linear] at the body's origin, world frame (flg_local = 0).
        mujoco.mj_objectVelocity(
            m, d, mujoco.mjtObj.mjOBJ_BODY, self._mj_payload_index, self._mj_payload_velocity, 0
        )
        self._mj_data__payload_vel = self._mj_payload_velocity[3:6]
