"""QuadrotorMujoco — MuJoCo physics backend with GLFW viewer."""

import copy

import numpy as np
from scipy.spatial.transform import Rotation as sp_rot

from ...manif import SO3, TSO3
from ...utils.logging import get_logger
from ..mujoco import MujocoModel
from .base import ROTOR_PARAMETER_DEFAULTS, QuadrotorBase

_logger = get_logger(__name__)

#: Relative and absolute tolerance within which a rotor parameter agrees with the
#: MJCF's. ``quadrotor_mj.xml`` writes its rotor sites and yaw gears to ten
#: digits (``ℓ/√2 = 0.1414213562``, ``c = 0.0249945776``), 3e-10 from the
#: defaults they encode; a real disagreement is many orders larger.
_MJCF_RTOL = 1e-8
_MJCF_ATOL = 1e-9


class QuadrotorMujoco(QuadrotorBase):
    """Quadrotor backed by MuJoCo physics engine.

    Overrides step/reset to use MuJoCo instead of Euler ZOH.
    Reads mass/inertia from the MJCF XML model.

    Note:
        The mass and inertia read from ``quadrotor_mj.xml`` (0.75 kg,
        ``diag(0.0053, 0.0049, 0.0098)`` kg·m²) are the same airframe
        :class:`~udaan.models.quadrotor.QuadrotorBase` flies, through
        :data:`~udaan.core.defaults.DEFAULT_M1_QUAD_MASS` and
        :data:`~udaan.core.defaults.DEFAULT_M1_QUAD_INERTIA`. So are the rotor
        geometry (arm length, ``k_M/k_F``) and the actuator limits.
        ``tests/test_quadrotor_mujoco.py`` checks every one of them against the
        MJCF.

        The rotor parameters (``arm_length``, ``rotor_angles``,
        ``spin_directions``, ``k_M/k_F`` and the per-rotor force limits) are
        read from the MJCF too: the four rotor sites give ``ℓ`` and ``θᵢ``, the
        sign and size of each rotor actuator's yaw gear give ``dᵢ`` and
        ``c = k_M/k_F``, and its ``ctrlrange`` gives the limits. A value that
        agrees with the MJCF to its written precision keeps its exact passed
        (or default) form; a default that disagrees is replaced by the MJCF's,
        and an explicitly passed one that disagrees raises ``ValueError`` —
        the model would otherwise fly one machine while MuJoCo flies another.
        Of ``force_constant`` and ``torque_constant`` the MJCF fixes only the
        ratio, so passing one of them derives the other from it.

        MuJoCo's timestep is not the MJCF's: it is written as ``dt / N`` from
        ``physics_rate_hz`` (see :class:`~udaan.models.quadrotor.QuadrotorBase`;
        1600 Hz, eight steps of 0.625 ms, by default), so a ``step()`` advances
        exactly ``dt`` at every valid rate.

        The attitude loop closes at ``attitude_rate_hz`` (see
        :class:`~udaan.models.quadrotor.QuadrotorBase`), 800 Hz by default, on
        every input path; its updates must divide the ``N`` MuJoCo steps per
        ``dt`` (eight by default, so 200, 400, 800 or 1600 Hz), and ``None`` is
        once per ``dt``. ``attitude_zoh`` is not a keyword (it raises
        ``TypeError``); ``attitude_rate_hz=None`` gives once per step.
    """

    # Always a live handle once ``__init__`` returns — it is loaded below and
    # never cleared, so every method may dereference it unconditionally.
    _mjMdl: MujocoModel

    def __init__(self, **kwargs):
        if "attitude_zoh" in kwargs:
            raise TypeError(
                "attitude_zoh is not a keyword: the attitude loop runs at attitude_rate_hz "
                "on every input path (None for once per step)"
            )
        self.render = kwargs.get("render", False)
        # QuadrotorBase validates attitude_rate_hz (and its default) against N,
        # which is also the number of MuJoCo steps per dt.
        super().__init__(**kwargs)

        self._mj_quad_index = 1

        # Load MuJoCo model
        self._mjMdl = MujocoModel("quadrotor_mj.xml", render=self.render)

        # The physics rate, not the MJCF, sets MuJoCo's timestep: N substeps of
        # dt/N per step(), the base model's own N, so a step advances exactly dt
        # and nothing is rounded.
        self._step_iter = self._inner_loop_steps
        self._mjDt = self.dt / self._step_iter
        self._mjMdl.model.opt.timestep = self._mjDt
        self._attitude_updates = self._attitude_updates_per_step(self._step_iter)
        if self._force_type == QuadrotorBase.FORCE_TYPE.PROP_FORCES:
            self._ctrl_index = 0
        else:
            self._ctrl_index = 4

        # Read params from MuJoCo model
        self.mass = copy.deepcopy(self._mjMdl.model.body_mass[self._mj_quad_index])
        self.inertia = copy.deepcopy(self._mjMdl.model.body_inertia[self._mj_quad_index])

        self._adopt_mjcf_rotors(explicit={k for k in ROTOR_PARAMETER_DEFAULTS if k in kwargs})

        # Reinit controllers with MuJoCo params
        self._init_default_controllers()

        if self.verbose:
            _logger.info("MuJoCo quadrotor loaded")

    def _read_mjcf_rotors(self) -> dict:
        """The rotor parameters ``quadrotor_mj.xml`` describes, from its first four actuators.

        Actuator ``i`` drives rotor site ``i``: its position gives ``ℓ`` and
        ``θᵢ``, its gear's yaw entry gives ``dᵢ`` (sign) and ``c = k_M/k_F``
        (size), and its ``ctrlrange`` the per-rotor force limits.

        Raises:
            ValueError: if the MJCF describes a machine this model cannot — a
                rotor not thrusting purely along ``b₃``, unequal arms, unequal
                yaw ratios or unequal force limits.
        """
        m = self._mjMdl.model
        pos = np.array([m.site_pos[m.actuator_trnid[i, 0]] for i in range(4)])
        gear = np.array(m.actuator_gear[:4])
        ranges = np.array(m.actuator_ctrlrange[:4])
        if not np.allclose(gear[:, :5], [0.0, 0.0, 1.0, 0.0, 0.0], rtol=0.0, atol=_MJCF_ATOL):
            raise ValueError(
                f"rotor actuators must thrust along b3 only; MJCF gears {gear.tolist()}"
            )
        arms = np.hypot(pos[:, 0], pos[:, 1])
        ratios = np.abs(gear[:, 5])
        for name, values in (("arm lengths", arms), ("yaw ratios", ratios), ("ctrlranges", ranges)):
            if not np.allclose(values, values[0], rtol=_MJCF_RTOL, atol=_MJCF_ATOL):
                raise ValueError(f"the MJCF's four rotors have unequal {name}: {values.tolist()}")
        return {
            "arm_length": float(arms.mean()),
            "rotor_angles": np.arctan2(pos[:, 1], pos[:, 0]),
            "spin_directions": np.sign(gear[:, 5]),
            "moment_ratio": float(ratios.mean()),
            "prop_min_force": float(ranges[0, 0]),
            "prop_max_force": float(ranges[0, 1]),
        }

    def _adopt_mjcf_rotors(self, explicit):
        """Make the rotor parameters the MJCF's; refuse an ``explicit`` one that disagrees."""
        mjcf = self._read_mjcf_rotors()
        params = self.rotor_parameters
        source = "quadrotor_mj.xml"

        def refuse(key, value, mjcf_value):
            raise ValueError(
                f"{key} = {value} disagrees with {source}, which gives {mjcf_value}. On "
                "QuadrotorMujoco the MJCF is the source of the rotor parameters: edit the "
                "MJCF, or use QuadrotorBase for a different airframe."
            )

        for key in (
            "arm_length",
            "rotor_angles",
            "spin_directions",
            "prop_min_force",
            "prop_max_force",
        ):
            diff = np.asarray(params[key]) - mjcf[key]
            if key == "rotor_angles":
                diff = (diff + np.pi) % (2 * np.pi) - np.pi  # the same rotor, a turn apart
            scale = np.abs(mjcf[key])
            if np.all(np.abs(diff) <= _MJCF_ATOL + _MJCF_RTOL * scale):
                continue
            if key in explicit:
                refuse(key, params[key], mjcf[key])
            params[key] = mjcf[key]

        c = mjcf["moment_ratio"]
        kf, km = params["force_constant"], params["torque_constant"]
        if abs(km / kf - c) > _MJCF_ATOL + _MJCF_RTOL * c:
            if {"force_constant", "torque_constant"} <= explicit:
                refuse("torque_constant / force_constant", km / kf, c)
            if "torque_constant" in explicit:
                params["force_constant"] = km / c
            else:
                params["torque_constant"] = c * kf

        if any(not np.array_equal(params[k], v) for k, v in self.rotor_parameters.items()):
            _logger.info("rotor parameters taken from %s: %s", source, params)
            self._configure_rotors(**params)

    # mypy cannot model a subclass re-opening a base class property to add a
    # setter (python/mypy#5936); the pattern itself is correct at runtime.
    @QuadrotorBase.mass.setter  # type: ignore[attr-defined]
    def mass(self, m):
        QuadrotorBase.mass.fset(self, m)  # type: ignore[attr-defined]
        if self._mjMdl is not None and self._mj_quad_index is not None:
            self._mjMdl.model.body_mass[self._mj_quad_index] = m
        else:
            _logger.warning("MuJoCo model not loaded, cannot set mass")

    @QuadrotorBase.inertia.setter  # type: ignore[attr-defined]  # see mass, above
    def inertia(self, I):
        QuadrotorBase.inertia.fset(self, I)  # type: ignore[attr-defined]
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

    def step(self, u, desired_att=None):
        """One outer step of ``dt``: :attr:`attitude_rate_hz` ``· dt`` attitude
        updates, each held over an equal share of the MuJoCo steps."""
        hold = self._step_iter // self._attitude_updates
        for _ in range(self._attitude_updates):
            wrench = self._repackage_input(u, desired_att=desired_att)
            # Convert to prop forces if MuJoCo expects individual motor commands
            if self._force_type == QuadrotorBase.FORCE_TYPE.PROP_FORCES:
                ctrl = self._wrench_to_propforces(wrench)
            else:
                ctrl = wrench
            self._mjMdl.data.ctrl[self._ctrl_index : self._ctrl_index + 4] = ctrl
            self._mjMdl._step_mujoco_simulation(hold)
            # The clock counts substeps (see QuadrotorBase._advance_clock) rather
            # than reading MuJoCo's, which accumulates `time += timestep` and
            # drifts the same way; MuJoCo's is overwritten so the two agree.
            self._advance_clock(hold)
            self._mjMdl.data.time = self.t
            self._query_latest_state()

        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl._viewer.add_trail_point(self.state.position)
            self._mjMdl._viewer.set_target(self._pos_controller.setpoint(self.t)[0])

    def _query_latest_state(self):
        """Sync state from MuJoCo data (the clock is the model's own, see :meth:`step`)."""
        self.state.position = copy.deepcopy(self._mjMdl.data.qpos[:3])
        q = copy.deepcopy(self._mjMdl.data.qpos[3:7])
        self.state.velocity = copy.deepcopy(self._mjMdl.data.qvel[:3])
        ang_vel = copy.deepcopy(self._mjMdl.data.qvel[3:6])
        self.state.orientation = SO3(self._mjMdl._quat2rot(q))
        self.state.angular_velocity = TSO3(np.array(ang_vel))

    def add_reference_marker(self, x):
        self._mjMdl.add_marker_at(x, label="xQd")
