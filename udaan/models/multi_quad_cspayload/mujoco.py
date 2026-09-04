"""MultiQuadCsPayloadMujoco — MuJoCo physics backend for the nQ-quad cable-suspended payload."""

from __future__ import annotations

import copy
import hashlib
import json

import mujoco
import numpy as np

from ...core.defaults import DEFAULT_SPAWN_HEIGHT
from ...manif import SO3, TSO3
from ...utils.assets.cache import ensure_generated
from ...utils.assets.mujoco_asset_creator import MJCF_GENERATOR_VERSION
from ...utils.logging import get_logger
from ..backend import MujocoModel
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

    Because one MJCF encodes one ring and one tendon length, ``masses`` and
    ``cable_lengths`` must be uniform at construction time; assign per-index
    values afterwards (the setters write straight into MuJoCo).

    ``inertias`` cannot be passed at construction at all. MuJoCo derives the
    body inertia from the generated geometry (``inertiafromgeom``) and
    :meth:`_read_params_from_mjcf` adopts it, so a constructor argument would
    be silently overwritten; it is rejected instead. Assign ``mdl.inertias``
    after construction — that setter *does* write into MuJoCo.
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

        self._spawn_height = float(kwargs.get("spawn_height", DEFAULT_SPAWN_HEIGHT))
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
        self._read_params_from_mjcf()

        self._mjDt = float(self._mjMdl.model.opt.timestep)
        self._step_iter = max(1, int(round(self.dt / self._mjDt)))

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

        The comparison is **exact**: under ``np.allclose`` a near-uniform array
        like ``[1.0, 1.0000001]`` was accepted and quietly flattened to its
        first entry, while ``[1.0, 1.00002]`` raised. There is no tolerance
        this backend could honour — either the MJCF encodes the value or it
        does not — so any difference at all is reported.
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
        generated_name = f"multi_quad_pointmass_{num_quadrotors}q_{config_hash}.xml"

        def build(path: str) -> None:
            from ...utils.assets import xml_model_generator

            _logger.info("Generating MJCF for %d quadrotors: %s", num_quadrotors, generated_name)
            xml_model_generator.multi_quad_pointmass(
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
        """Adopt masses, inertias and cable lengths from the loaded MJCF."""
        model = self._mjMdl.model
        self.masses = np.array([model.body_mass[i] for i in self._quad_body_id])
        # body_inertia holds principal-axis diagonals; the model wants tensors.
        self.inertias = np.stack([np.diag(model.body_inertia[i]) for i in self._quad_body_id])
        self.payload_mass = float(model.body_mass[self._payload_body_id])
        if model.ntendon == self._num_quadrotors:
            # One spatial tendon per quadrotor, emitted in quadrotor order; the
            # upper limit of its range is the taut cable length.
            self.cable_lengths = np.array(model.tendon_range[:, 1], dtype=float)

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
                self._mjMdl.model.body_inertia[body_id] = np.diag(self._inertias[i])

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
        """Reset to the MJCF's own formation, optionally translated.

        Unlike the analytical base, the initial geometry comes from the MJCF
        (the tendon constraints are only consistent with it), so this does not
        call ``super().reset()``. ``payload_position`` translates the whole
        formation — quadrotors and payload alike — so the payload lands there.
        """
        self.t = 0.0
        self._mjMdl.reset()
        self.state.reset()

        if "payload_position" in kwargs:
            target = np.asarray(kwargs["payload_position"], dtype=float)
            i = self._payload_qposadr
            offset = target - self._mjMdl.data.qpos[i : i + 3]
            for adr in (*self._quad_qposadr, self._payload_qposadr):
                self._mjMdl.data.qpos[adr : adr + 3] += offset

        mujoco.mj_forward(self._mjMdl.model, self._mjMdl.data)
        self._query_latest_state()
        self._formation_targets = [p.copy() for p in self.state.positions]

    # ─── Step ──────────────────────────────────────────────────────

    def step(self, u, desired_att=None):
        """Step MuJoCo simulation. ``u`` follows the base input pipeline."""
        for _ in range(self._step_iter):
            wrench = self._repackage_input(u, desired_att=desired_att)
            self._mjMdl.data.ctrl[:] = wrench.reshape(-1)
            self._mjMdl._step_mujoco_simulation(1)
            self._query_latest_state()

        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl._viewer.add_trail_point(
                self.state.payload_position, key=0, rgba=[0.2, 0.2, 1.0, 0.6]
            )

    # ─── State query ───────────────────────────────────────────────

    def _query_latest_state(self):
        """Sync state from MuJoCo data."""
        data = self._mjMdl.data
        self.t = data.time

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
