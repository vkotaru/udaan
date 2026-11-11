"""MultiQuadCsPayloadMujoco — MuJoCo physics backend for the nQ-quad cable-suspended payload."""

from __future__ import annotations

import copy
import hashlib
import json

import mujoco
import numpy as np

from ...core.defaults import (
    DEFAULT_MAX_THRUST,
    DEFAULT_MAX_TORQUE,
    DEFAULT_MIN_THRUST,
    DEFAULT_MIN_TORQUE,
    DEFAULT_TENDON_LIMIT_TIMESTEPS,
)
from ...manif import SO3, TSO3
from ...utils.assets.cache import ensure_generated
from ...utils.assets.mujoco_asset_creator import MJCF_GENERATOR_VERSION
from ...utils.formation import default_spawn_height
from ...utils.grasp import formation_cable_attitudes
from ...utils.logging import get_logger
from .._kwargs import check_reset_kwargs
from ..mujoco import MujocoModel
from .base import MultiQuadCsPayloadBase, cable_state_from_geometry

_logger = get_logger(__name__)


#: Body-name templates used by the MJCF generator.
_QUAD_BODY = "quad{}"
_PAYLOAD_BODY = "pyld"


def _count_quad_bodies(model) -> int:
    """Number of ``quad{i}`` bodies in a loaded MuJoCo model."""
    n = 0
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _QUAD_BODY.format(n)) >= 0:
        n += 1
    return n


#: The reset overrides this backend can apply. The analytical base documents
#: six; a placement of the taut formation and a rigid velocity of all of it are
#: the ones that stay consistent with the tendon constraints.
_SUPPORTED_RESET_KWARGS = frozenset({"payload_position", "payload_velocity"})

#: The MuJoCo warnings that mean the integration diverged: MuJoCo answers each
#: by resetting the state, its clock included.
_DIVERGENCE_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQPOS,
)

#: Absolute tolerance on the MJCF's ``ctrlrange`` against :mod:`udaan.core.defaults`.
_CTRLRANGE_ATOL = 1e-9


def _quat_to_matrix(quat) -> np.ndarray:
    """Rotation matrix of a MuJoCo ``(w, x, y, z)`` quaternion."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    return R.reshape(3, 3)


def body_frame_inertia(principal_moments, iquat) -> np.ndarray:
    """``R diag(I) Rᵀ``: a body's inertia tensor in its own frame.

    MuJoCo stores ``body_inertia`` as principal moments ``I`` in the principal
    frame ``body_iquat``; with ``inertiafromgeom`` that frame can be rotated
    from the body frame, so ``diag(I)`` alone is not the body-frame tensor.
    """
    R = _quat_to_matrix(iquat)
    return np.asarray(R @ np.diag(np.asarray(principal_moments, dtype=float)) @ R.T)


def principal_inertia(tensor) -> tuple[np.ndarray, np.ndarray]:
    """``(I, iquat)`` with ``R(iquat) diag(I) R(iquat)ᵀ = tensor``.

    The inverse of :func:`body_frame_inertia`: the eigen-decomposition of the
    symmetric tensor, its eigenvector matrix made a proper rotation.
    """
    moments, vectors = np.linalg.eigh(np.asarray(tensor, dtype=float))
    if np.linalg.det(vectors) < 0.0:
        vectors[:, 0] = -vectors[:, 0]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, vectors.reshape(-1))
    return moments, quat


class MultiQuadCsPayloadMujoco(MultiQuadCsPayloadBase):
    """``nQ`` quadrotors carrying one cable-suspended point-mass payload, in MuJoCo.

    Overrides ``step``/``reset``/``_query_latest_state`` to delegate to MuJoCo;
    the input pipeline, controllers and parameter bookkeeping are inherited.
    The cables are MuJoCo spatial tendons with an upper length limit, so they
    pull but never push.

    The MJCF is generated on demand from the construction parameters
    (``num_quadrotors``, ``cable_lengths``, ``payload_mass``, ``masses``,
    ``formation_radius`` and ``spawn_height``) and cached under a filename
    carrying a hash of them, so a changed parameter never reuses a stale asset.
    The generated formation starts **taut**: the payload height is derived from
    the cable length and the ring radius, not chosen independently.
    ``spawn_height`` defaults to
    :func:`udaan.utils.formation.default_spawn_height`.

    The MJCF is the source of the physical parameters: masses, the body-frame
    inertia ``R(iquat) diag(I) R(iquat)ᵀ``, cable lengths and the actuator
    envelope (each airframe's ``ctrlrange``, which the generator writes from
    :mod:`udaan.core.defaults`; a different one is adopted with a warning).

    MuJoCo's timestep is not the MJCF's: it is written as ``dt / N`` from
    ``physics_rate_hz`` (see :class:`MultiQuadCsPayloadBase`; 1600 Hz, eight
    steps of 0.625 ms, by default), so a ``step()`` advances exactly ``dt``.
    Each tendon's limit is written as ``solreflimit =
    (DEFAULT_TENDON_LIMIT_TIMESTEPS · dt/N, 1)``. The clock counts substeps and
    is written into ``data.time``; :meth:`step` raises ``RuntimeError`` when
    MuJoCo reports a diverged integration (it then resets the state) or its
    time runs backwards.

    Because one MJCF encodes one ring and one tendon length, ``masses`` and
    ``cable_lengths`` must be uniform at construction time; assign per-index
    values afterwards (the setters write straight into MuJoCo, and
    :meth:`reset` places the formation from the current lengths).

    ``inertias`` cannot be passed at construction at all. MuJoCo derives the
    body inertia from the generated geometry (``inertiafromgeom``) and
    :meth:`_read_params_from_mjcf` adopts it, so a constructor argument would
    be silently overwritten; it is rejected instead. Assign ``mdl.inertias``
    after construction — that setter writes the tensor's principal moments and
    frame into MuJoCo.
    """

    def __init__(self, **kwargs):
        self.render = kwargs.get("render", False)
        if "inertias" in kwargs:
            raise ValueError(
                "the MuJoCo backend derives inertias from the generated MJCF geometry, "
                "so an `inertias` constructor argument would be silently overwritten; "
                "construct the model and then assign `mdl.inertias`"
            )
        super().__init__(**kwargs)

        spawn_height = kwargs.get("spawn_height")
        if spawn_height is None:
            radius = 0.0 if self._num_quadrotors == 1 else self._formation_radius
            spawn_height = default_spawn_height(
                self._uniform(self._cable_lengths, "cable_lengths"), radius
            )
        self._spawn_height = float(spawn_height)
        xml_path = self._ensure_xml(
            self._num_quadrotors,
            self._uniform(self._cable_lengths, "cable_lengths"),
            self._payload_mass,
            self._uniform(self._masses, "masses"),
            self._formation_radius,
            self._spawn_height,
        )
        self._mjMdl = MujocoModel(model_path=xml_path, render=self.render)

        self._resolve_mj_indices()

        # The physics rate, not the MJCF, sets MuJoCo's timestep: N steps of
        # dt/N per step(), the base model's own N, so a step advances exactly dt.
        self._step_iter = self._inner_loop_steps
        self._mjDt = self.dt / self._step_iter
        model = self._mjMdl.model
        model.opt.timestep = self._mjDt
        for i in range(model.ntendon):
            model.tendon_solref_lim[i] = [DEFAULT_TENDON_LIMIT_TIMESTEPS * self._mjDt, 1.0]

        self._read_params_from_mjcf()

        # Reinit controllers with the MuJoCo parameters
        self._init_default_controllers()

        if self.verbose:
            _logger.info("MuJoCo multi-quad cspayload loaded (%d quadrotors)", self._num_quadrotors)

    # ─── Model initialization ──────────────────────────────────────

    @staticmethod
    def _uniform(values: np.ndarray, name: str) -> float:
        """Collapse a per-quadrotor array to the single value the MJCF encodes.

        The generator lays every quadrotor out on one ring with one tendon
        length, so a per-index array it cannot express is an error rather than
        a value to be silently averaged away.

        The comparison is **exact**: a tolerance would silently flatten a
        near-uniform array such as ``[1.0, 1.0000001]`` to its first entry.
        Either the MJCF encodes the value or it does not, so any difference at
        all raises.
        """
        arr = np.asarray(values, dtype=float)
        if not np.all(arr == arr.flat[0]):
            raise ValueError(
                f"the MuJoCo backend needs a uniform {name} at construction time, got {arr}; "
                f"build the model uniformly and assign per-index {name} afterwards"
            )
        return float(arr.flat[0])

    @staticmethod
    def _ensure_xml(
        num_quadrotors: int,
        cable_length: float,
        payload_mass: float,
        quad_mass: float,
        formation_radius: float,
        spawn_height: float,
    ) -> str:
        """Return the MJCF path for this configuration, generating it if needed.

        The filename carries a hash of every parameter that shapes the XML —
        *and* of :data:`MJCF_GENERATOR_VERSION`, so a release that changes the
        emitted structure cannot reuse a file an earlier one left in a warm
        asset directory.

        The file lives in the user asset cache
        (:func:`udaan.utils.assets.cache.generated_mjcf_dir`), not inside the
        installed package, and an absolute path is returned rather than a bare
        name so the caller loads exactly the file this wrote.
        """
        # A lone quadrotor hovers on the axis; the ring radius is unused, so it
        # must not enter the hash either.
        if num_quadrotors == 1:
            formation_radius = 0.0
        config = {
            "v": MJCF_GENERATOR_VERSION,
            "nQ": int(num_quadrotors),
            "L": float(cable_length),
            "mL": float(payload_mass),
            "mQ": float(quad_mass),
            "r": float(formation_radius),
            "z": float(spawn_height),
        }
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        generated_name = f"multi_quad_cspayload_{num_quadrotors}q_{config_hash}.xml"

        def build(path: str) -> None:
            from ...utils.assets import xml_model_generator

            _logger.info("Generating MJCF for %d quadrotors: %s", num_quadrotors, generated_name)
            xml_model_generator.multi_quad_cspayload(
                num_quadrotors=num_quadrotors,
                filename=path,
                cable_length=cable_length,
                payload_mass=payload_mass,
                quad_mass=quad_mass,
                formation_radius=formation_radius,
                spawn_height=spawn_height,
                verbose=False,
            )

        # Generated assets go to the user cache, not into the installed
        # package, and the write is atomic — see
        # :func:`udaan.utils.assets.cache.ensure_generated`.
        return ensure_generated(generated_name, build)

    def _resolve_mj_indices(self):
        """Look up body ids and per-body qpos/qvel addresses by name."""
        model = self._mjMdl.model
        nQ = self._num_quadrotors

        loaded_nQ = _count_quad_bodies(model)
        if loaded_nQ != nQ:
            raise RuntimeError(
                f"MJCF defines {loaded_nQ} quadrotors but the model was built for {nQ}"
            )
        if model.nu != 4 * nQ:
            raise RuntimeError(f"MJCF exposes {model.nu} actuators, expected {4 * nQ}")

        self._quad_body_id = np.zeros(nQ, dtype=int)
        self._quad_qposadr = np.zeros(nQ, dtype=int)
        self._quad_qveladr = np.zeros(nQ, dtype=int)
        for i in range(nQ):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _QUAD_BODY.format(i))
            self._quad_body_id[i] = body_id
            jnt_id = model.body_jntadr[body_id]
            self._quad_qposadr[i] = model.jnt_qposadr[jnt_id]
            self._quad_qveladr[i] = model.jnt_dofadr[jnt_id]

        payload_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _PAYLOAD_BODY)
        if payload_id < 0:
            raise RuntimeError(f"MJCF has no body named {_PAYLOAD_BODY!r}")
        self._payload_body_id = payload_id
        payload_jnt = model.body_jntadr[payload_id]
        self._payload_qposadr = model.jnt_qposadr[payload_jnt]
        self._payload_qveladr = model.jnt_dofadr[payload_jnt]

    def _read_params_from_mjcf(self):
        """Adopt masses, inertias, cable lengths and the actuator envelope from the MJCF."""
        model = self._mjMdl.model
        self.masses = np.array([model.body_mass[i] for i in self._quad_body_id])
        # body_inertia holds principal moments in the principal frame body_iquat.
        self.inertias = np.stack(
            [
                body_frame_inertia(model.body_inertia[i], model.body_iquat[i])
                for i in self._quad_body_id
            ]
        )
        self.payload_mass = float(model.body_mass[self._payload_body_id])
        if model.ntendon == self._num_quadrotors:
            # One spatial tendon per quadrotor, emitted in quadrotor order; the
            # upper limit of its range is the taut cable length.
            self.cable_lengths = np.array(model.tendon_range[:, 1], dtype=float)
        self._read_ctrlrange()

    def _read_ctrlrange(self):
        """Adopt the airframes' ``ctrlrange`` as the clip, warning if it is not core.defaults'.

        Each airframe has four motors, thrust then ``M_x``, ``M_y``, ``M_z``,
        in quadrotor order; every airframe must carry the same envelope, which
        the model's single clip then is.

        Raises:
            ValueError: if the airframes' envelopes differ.
        """
        ranges = np.array(self._mjMdl.model.actuator_ctrlrange, dtype=float).reshape(
            self._num_quadrotors, 4, 2
        )
        if not np.allclose(ranges, ranges[0], rtol=0.0, atol=_CTRLRANGE_ATOL):
            raise ValueError(
                "the MJCF's airframes carry different actuator ctrlranges; the model "
                f"clips every airframe to one envelope: {ranges.tolist()}"
            )
        mjcf = ranges[0]
        defaults = np.array(
            [
                [DEFAULT_MIN_THRUST, DEFAULT_MAX_THRUST],
                *np.stack([DEFAULT_MIN_TORQUE, DEFAULT_MAX_TORQUE], axis=1),
            ]
        )
        if not np.allclose(mjcf, defaults, rtol=0.0, atol=_CTRLRANGE_ATOL):
            _logger.warning(
                "the MJCF's actuator ctrlrange %s differs from core.defaults %s; the "
                "MJCF is the source, and both models clip to it",
                mjcf.tolist(),
                defaults.tolist(),
            )
        self._min_thrust, self._max_thrust = float(mjcf[0, 0]), float(mjcf[0, 1])
        self._min_torque = np.array(mjcf[1:, 0])
        self._max_torque = np.array(mjcf[1:, 1])

    # ─── Property setters (sync to MuJoCo) ─────────────────────────
    #
    # The `@Base.prop.setter` override pattern is standard at runtime but
    # mypy cannot model it, hence the attr-defined ignores below.

    @MultiQuadCsPayloadBase.masses.setter  # type: ignore[attr-defined]
    def masses(self, value):
        MultiQuadCsPayloadBase.masses.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            for i, body_id in enumerate(self._quad_body_id):
                self._mjMdl.model.body_mass[body_id] = self._masses[i]

    @MultiQuadCsPayloadBase.inertias.setter  # type: ignore[attr-defined]
    def inertias(self, value):
        MultiQuadCsPayloadBase.inertias.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            for i, body_id in enumerate(self._quad_body_id):
                moments, quat = principal_inertia(self._inertias[i])
                self._mjMdl.model.body_inertia[body_id] = moments
                self._mjMdl.model.body_iquat[body_id] = quat

    @MultiQuadCsPayloadBase.payload_mass.setter  # type: ignore[attr-defined]
    def payload_mass(self, value):
        MultiQuadCsPayloadBase.payload_mass.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            self._mjMdl.model.body_mass[self._payload_body_id] = self._payload_mass

    @MultiQuadCsPayloadBase.cable_lengths.setter  # type: ignore[attr-defined]
    def cable_lengths(self, value):
        MultiQuadCsPayloadBase.cable_lengths.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            model = self._mjMdl.model
            if model.ntendon == self._num_quadrotors:
                for i in range(self._num_quadrotors):
                    model.tendon_limited[i] = True
                    model.tendon_range[i, 1] = self._cable_lengths[i]
                    model.tendon_length0[i] = self._cable_lengths[i]
                mujoco.mj_tendon(model, self._mjMdl.data)

    # ─── Reset ─────────────────────────────────────────────────────

    def reset(self, **kwargs):
        """Reset to the taut ring formation at the current cable lengths.

        Unlike the analytical base, this does not call ``super().reset()``: the
        state is read back from MuJoCo. Each airframe is placed level at
        ``x_L − l_i q̄_i``, with ``q̄_i`` the ring's cable directions
        (:func:`~udaan.utils.grasp.formation_cable_attitudes`) at the current
        ``cable_lengths`` and ``formation_radius``, so every tendon starts at
        its limit. ``x_L`` is ``payload_position``, or the MJCF's payload
        position. ``payload_velocity`` is given to every body, a rigid velocity
        of the whole formation.

        Raises:
            ValueError: for any other keyword. Setting attitudes, rates or cable
                directions means re-solving the formation against the tendon
                constraints; an unsupported keyword is an error, not a no-op.
        """
        check_reset_kwargs(
            self,
            kwargs,
            _SUPPORTED_RESET_KWARGS,
            detail=(
                "the initial formation is the taut ring at the current cable lengths, and "
                "only its position and a rigid velocity of the whole formation can be set. "
                "Use MultiQuadCsPayloadBase for arbitrary initial conditions, or set the "
                "MuJoCo state directly through _mjMdl.data."
            ),
        )
        self.t = 0.0
        self._substeps_taken = 0
        self._compressive_cable_steps = 0
        self._mjMdl.reset()
        self.state.reset()

        data = self._mjMdl.data
        i = self._payload_qposadr
        if "payload_position" in kwargs:
            x_L = np.asarray(kwargs["payload_position"], dtype=float)
        else:
            x_L = np.array(data.qpos[i : i + 3])
        data.qpos[i : i + 3] = x_L
        directions = formation_cable_attitudes(
            self._num_quadrotors, self._cable_lengths, self._formation_radius
        )
        for k, adr in enumerate(self._quad_qposadr):
            data.qpos[adr : adr + 3] = x_L - self._cable_lengths[k] * np.asarray(directions[k])
            data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

        data.qvel[:] = 0.0
        if "payload_velocity" in kwargs:
            v = np.asarray(kwargs["payload_velocity"], dtype=float)
            for adr in (*self._quad_qveladr, self._payload_qveladr):
                data.qvel[adr : adr + 3] = v

        data.time = 0.0
        mujoco.mj_forward(self._mjMdl.model, data)
        self._query_latest_state()
        self._formation_targets = [p.copy() for p in self.state.positions]

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
            self._mjMdl.data.ctrl[:] = wrench.reshape(-1)
            time_before = float(self._mjMdl.data.time)
            warnings_before = self._divergence_warnings()
            self._mjMdl._step_mujoco_simulation(hold)
            self._check_divergence(time_before, warnings_before)
            # The clock counts substeps (see MultiQuadCsPayloadBase._advance_clock)
            # rather than reading MuJoCo's, which accumulates `time += timestep`;
            # MuJoCo's is overwritten so the two agree.
            self._advance_clock(hold)
            self._mjMdl.data.time = self.t
            self._query_latest_state()
            self.state.cable_tensions = self.cable_tensions()

        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl._viewer.add_trail_point(
                self.state.payload_position, key=0, rgba=[0.2, 0.2, 1.0, 0.6]
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
                f"MuJoCo integration diverged at t = {self.t:.4f} s: MuJoCo reported a bad "
                "acceleration, velocity or position and reset the state "
                f"(data.time {time_before:.4f} s -> {float(data.time):.4f} s)"
            )

    # ─── State query ───────────────────────────────────────────────

    def cable_tensions(self) -> np.ndarray:
        """``(nQ,)`` cable tensions of the last MuJoCo step, in N (zero: slack).

        Each cable's tendon-limit constraint force, the force the tendon
        transmits while it is at its length; a slack tendon has no active limit
        row and reads zero.
        """
        data = self._mjMdl.data
        tensions = np.zeros(self._num_quadrotors)
        rows = np.flatnonzero(
            data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
        )
        for r in rows:
            tensions[data.efc_id[r]] += abs(float(data.efc_force[r]))
        return tensions

    def _query_latest_state(self):
        """Sync state from MuJoCo data (the clock is the model's own, see :meth:`step`)."""
        data = self._mjMdl.data

        i = self._payload_qposadr
        self.state.payload_position = copy.deepcopy(data.qpos[i : i + 3])
        i = self._payload_qveladr
        self.state.payload_velocity = copy.deepcopy(data.qvel[i : i + 3])

        for k in range(self._num_quadrotors):
            i = self._quad_qposadr[k]
            self.state.positions[k] = copy.deepcopy(data.qpos[i : i + 3])
            self.state.orientations[k] = SO3(self._mjMdl._quat2rot(data.qpos[i + 3 : i + 7]))
            i = self._quad_qveladr[k]
            self.state.velocities[k] = copy.deepcopy(data.qvel[i : i + 3])
            self.state.angular_velocities[k] = TSO3(copy.deepcopy(data.qvel[i + 3 : i + 6]))

            # Cable direction and angular velocity from the measured endpoints.
            # The 1/length factor is essential — see cable_state_from_geometry.
            try:
                q, omega, _ = cable_state_from_geometry(
                    self.state.payload_position,
                    self.state.payload_velocity,
                    self.state.positions[k],
                    self.state.velocities[k],
                )
            except ValueError:
                continue
            self.state.cable_attitudes[k] = q
            self.state.cable_angular_velocities[k] = omega

    # ─── Simulate ──────────────────────────────────────────────────

    def simulate(self, tf, **kwargs):
        super().simulate(tf, **kwargs)
        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl.wait_for_close()
