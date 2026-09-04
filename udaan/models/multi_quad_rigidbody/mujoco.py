"""MultiQuadRigidbodyMujoco — MuJoCo physics backend for the nQ-quad rigid-body payload."""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from ...manif import SO3, TSO3
from ...utils.logging import get_logger
from ..mujoco import MujocoModel
from .base import MultiQuadRigidbodyBase, cable_state_from_geometry

_logger = get_logger(__name__)

#: Body/site name templates in ``multi_quad_rigidbody.xml``.
_QUAD_BODY = "quad{}"
_QUAD_SITE = "quad{}_end1"
_PAYLOAD_BODY = "pyld"
_PAYLOAD_SITE = "pyld_poc{}"

#: The shipped MJCF is hand-written and fixed at four quadrotors. The generator
#: that would make this a parameter is a later step; until then the number is a
#: property of the asset, and asking for any other count is an error rather
#: than something to be silently rounded down to four.
MJCF_NUM_QUADROTORS = 4

#: The attachment points ``multi_quad_rigidbody.xml`` encodes, in the payload
#: body frame — the four top corners of the 1.0 x 1.0 x 0.1 m slab.
#:
#: These are declared here, not read off the file, precisely so that the two can
#: be *compared*: the retired ``models/mujoco/multi_quad_rigidbody.py`` carried a
#: hardcoded 3x4 ``_grasp_map`` whose third column was ``(0.5, -0.5)`` while the
#: matching site ``pyld_poc2`` sits at ``(-0.5, -0.5)``, leaving columns 3 and 4
#: duplicates and one whole corner of the grasp unrepresented. Nothing checked
#: it. :meth:`MultiQuadRigidbodyMujoco._read_params_from_mjcf` now does.
MJCF_ATTACHMENT_POINTS = np.array(
    [
        [0.5, 0.5, 0.05],
        [-0.5, 0.5, 0.05],
        [-0.5, -0.5, 0.05],
        [0.5, -0.5, 0.05],
    ]
)

#: Geometry read out of the MJCF must match the declared parameters to this
#: absolute tolerance, in metres. The file stores coordinates with full double
#: precision, so real agreement is at round-off; anything looser would be
#: hiding a genuine difference.
_GEOMETRY_TOL = 1e-9

#: The reset overrides this backend can actually apply to the MJCF formation.
#: See :meth:`MultiQuadRigidbodyMujoco.reset`.
_SUPPORTED_RESET_KWARGS = frozenset({"payload_position"})


class MultiQuadRigidbodyMujoco(MultiQuadRigidbodyBase):
    """``nQ`` quadrotors carrying one cable-suspended rigid-body payload, in MuJoCo.

    Overrides ``step``/``reset``/``_query_latest_state`` to delegate to MuJoCo;
    the input pipeline, controllers and parameter bookkeeping are inherited.
    The cables are MuJoCo spatial tendons with an upper length limit, so they
    pull but never push.

    The MJCF (``multi_quad_rigidbody.xml``) is a fixed, hand-written asset — the
    generator that would make its geometry a construction parameter is a later
    step.  Until then :data:`MJCF_NUM_QUADROTORS` and
    :data:`MJCF_ATTACHMENT_POINTS` are what this backend is built for, and every
    one of them is **checked** against the loaded file rather than adopted from
    it.  Passing a different ``num_quadrotors`` or a different set of
    ``attachment_points`` raises.

    Masses, inertias, the payload's mass and inertia and the cable lengths *are*
    read from the file, because MuJoCo is the authority on them: the body mass
    is the sum of its geoms' masses, and the retired model's hardcoded
    ``mQ = 0.75`` was the mass of the quadrotor's *centre box* alone — the arms
    and propellers bring the body to 0.99 kg, so every feedforward term was 24%
    light.
    """

    def __init__(self, **kwargs):
        self.render = kwargs.get("render", False)
        if "inertias" in kwargs:
            raise ValueError(
                "the MuJoCo backend derives inertias from the MJCF geometry, so an "
                "`inertias` constructor argument would be silently overwritten; "
                "construct the model and then assign `mdl.inertias`"
            )
        kwargs.setdefault("num_quadrotors", MJCF_NUM_QUADROTORS)
        if int(kwargs["num_quadrotors"]) != MJCF_NUM_QUADROTORS:
            raise ValueError(
                f"multi_quad_rigidbody.xml defines exactly {MJCF_NUM_QUADROTORS} "
                f"quadrotors, but num_quadrotors={kwargs['num_quadrotors']} was "
                "requested. The MJCF generator for this system is a later step; "
                "until it lands the asset's count is not a parameter."
            )
        kwargs.setdefault("attachment_points", MJCF_ATTACHMENT_POINTS)
        super().__init__(**kwargs)

        self._mjMdl = MujocoModel(model_path="multi_quad_rigidbody.xml", render=self.render)

        self._resolve_mj_indices()
        self._read_params_from_mjcf()

        self._mjDt = float(self._mjMdl.model.opt.timestep)
        self._step_iter = max(1, int(round(self.dt / self._mjDt)))

        # Reinit controllers with the MuJoCo parameters
        self._init_default_controllers()

        self._warn_if_spawn_is_slack()

        if self.verbose:
            _logger.info(
                "MuJoCo multi-quad rigid-body payload loaded (%d quadrotors)", self._num_quadrotors
            )

    # ─── Model initialization ──────────────────────────────────────

    def _resolve_mj_indices(self):
        """Look up body ids and per-body qpos/qvel addresses by name."""
        model = self._mjMdl.model
        nQ = self._num_quadrotors

        self._quad_body_id = np.zeros(nQ, dtype=int)
        self._quad_qposadr = np.zeros(nQ, dtype=int)
        self._quad_qveladr = np.zeros(nQ, dtype=int)
        for i in range(nQ):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _QUAD_BODY.format(i))
            if body_id < 0:
                raise RuntimeError(f"MJCF has no body named {_QUAD_BODY.format(i)!r}")
            self._quad_body_id[i] = body_id
            jnt_id = model.body_jntadr[body_id]
            self._quad_qposadr[i] = model.jnt_qposadr[jnt_id]
            self._quad_qveladr[i] = model.jnt_dofadr[jnt_id]

        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _QUAD_BODY.format(nQ)) >= 0:
            raise RuntimeError(f"MJCF defines more than the expected {nQ} quadrotors")
        if model.nu != 4 * nQ:
            raise RuntimeError(f"MJCF exposes {model.nu} actuators, expected {4 * nQ}")

        payload_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _PAYLOAD_BODY)
        if payload_id < 0:
            raise RuntimeError(f"MJCF has no body named {_PAYLOAD_BODY!r}")
        self._payload_body_id = payload_id
        payload_jnt = model.body_jntadr[payload_id]
        self._payload_qposadr = model.jnt_qposadr[payload_jnt]
        self._payload_qveladr = model.jnt_dofadr[payload_jnt]

        self._attachment_site_id = np.zeros(nQ, dtype=int)
        for i in range(nQ):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _PAYLOAD_SITE.format(i))
            if site_id < 0:
                raise RuntimeError(f"MJCF has no site named {_PAYLOAD_SITE.format(i)!r}")
            self._attachment_site_id[i] = site_id

    def _read_params_from_mjcf(self):
        """Check the geometry against the declared parameters, adopt the inertias.

        The split is deliberate. *Geometry* — how many quadrotors there are and
        where the cables attach — is declared by the constructor and verified
        here, because a mismatch means the analytical model and the physics are
        describing different machines. *Inertial properties* are read from the
        file, because MuJoCo computes them from the geoms and is the authority.
        """
        model = self._mjMdl.model
        nQ = self._num_quadrotors

        self._check_tendon_wiring()

        site_points = np.array([model.site_pos[i] for i in self._attachment_site_id])
        if not np.allclose(site_points, self._attachment_points, rtol=0.0, atol=_GEOMETRY_TOL):
            raise RuntimeError(
                "attachment_points disagree with the MJCF's pyld_poc sites:\n"
                f"  model: {self._attachment_points.tolist()}\n"
                f"  MJCF:  {site_points.tolist()}"
            )
        # The attachment points are expressed relative to the payload's centre
        # of mass; MuJoCo expresses site positions relative to the *body frame*.
        # They coincide only when the body's inertial origin is the frame
        # origin, which is true for the shipped centred slab.
        ipos = model.body_ipos[self._payload_body_id]
        if not np.allclose(ipos, 0.0, atol=_GEOMETRY_TOL):
            raise RuntimeError(
                "payload centre of mass is offset from its body frame by "
                f"{ipos.tolist()}; attachment points are defined about the centre "
                "of mass and would need shifting"
            )

        self.masses = np.array([model.body_mass[i] for i in self._quad_body_id])
        # body_inertia holds principal-axis diagonals; the model wants tensors.
        self.inertias = np.stack([np.diag(model.body_inertia[i]) for i in self._quad_body_id])
        self.payload_mass = float(model.body_mass[self._payload_body_id])

        iquat = model.body_iquat[self._payload_body_id]
        if not np.allclose(np.abs(iquat), np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-9):
            raise RuntimeError(
                "payload principal axes are rotated relative to its body frame "
                f"(iquat={iquat.tolist()}); the diagonal inertia read from the MJCF "
                "would not be the tensor in the frame the attachment points use"
            )
        self.payload_inertia = np.diag(model.body_inertia[self._payload_body_id])

        if model.ntendon == nQ:
            # One spatial tendon per quadrotor, emitted in quadrotor order; the
            # upper limit of its range is the taut cable length.
            self.cable_lengths = np.array(model.tendon_range[:, 1], dtype=float)

    def _check_tendon_wiring(self):
        """Verify tendon ``i`` really joins ``quad{i}`` to ``pyld_poc{i}``.

        The pairing is what makes ``attachment_points[i]`` belong to cable
        ``i``. Nothing in the MJCF enforces it, and the retired model assumed
        it while its own grasp matrix disagreed with the sites.
        """
        model = self._mjMdl.model
        nQ = self._num_quadrotors
        if model.ntendon != nQ:
            raise RuntimeError(f"MJCF has {model.ntendon} tendons, expected {nQ}")
        for i in range(nQ):
            start = int(model.tendon_adr[i])
            count = int(model.tendon_num[i])
            names = []
            for k in range(start, start + count):
                if int(model.wrap_type[k]) != int(mujoco.mjtWrap.mjWRAP_SITE):
                    continue
                names.append(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, int(model.wrap_objid[k]))
                )
            expected = [_QUAD_SITE.format(i), _PAYLOAD_SITE.format(i)]
            if names != expected:
                raise RuntimeError(f"tendon {i} joins {names}, expected {expected}")

    def _warn_if_spawn_is_slack(self):
        """Report a spawn pose whose cables are shorter than the tendon limit.

        The shipped asset spawns the quadrotors 0.61 m above attachment points
        0.5 m inboard of them, so every cable measures 0.789 m against a 1.0 m
        tendon limit: the formation starts **slack** and the payload drops
        ~0.2 m before the cables catch. The analytical model has no slack mode
        at all, so this is a real difference between the two backends, and it
        is a defect in the asset rather than in either model — the MJCF
        generator step is where the spawn geometry gets derived from the cable
        length instead of chosen independently.
        """
        mujoco.mj_forward(self._mjMdl.model, self._mjMdl.data)
        self._query_latest_state()
        measured = np.array(
            [
                np.linalg.norm(self.attachment_position(i) - self.state.positions[i])
                for i in range(self._num_quadrotors)
            ]
        )
        slack = self._cable_lengths - measured
        if np.any(slack > 1e-3):
            _logger.warning(
                "multi_quad_rigidbody.xml spawns with slack cables: measured %s "
                "against tendon limits %s. The payload will drop before the cables "
                "go taut; the analytical model has no slack mode. Fixed when the "
                "MJCF is generated from the cable length.",
                np.array2string(measured, precision=4),
                np.array2string(self._cable_lengths, precision=4),
            )

    # ─── Property setters (sync to MuJoCo) ─────────────────────────
    #
    # The `@Base.prop.setter` override pattern is standard at runtime but
    # mypy cannot model it, hence the attr-defined ignores below.

    @MultiQuadRigidbodyBase.masses.setter  # type: ignore[attr-defined]
    def masses(self, value):
        MultiQuadRigidbodyBase.masses.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            for i, body_id in enumerate(self._quad_body_id):
                self._mjMdl.model.body_mass[body_id] = self._masses[i]

    @MultiQuadRigidbodyBase.inertias.setter  # type: ignore[attr-defined]
    def inertias(self, value):
        MultiQuadRigidbodyBase.inertias.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            for i, body_id in enumerate(self._quad_body_id):
                self._mjMdl.model.body_inertia[body_id] = np.diag(self._inertias[i])

    @MultiQuadRigidbodyBase.payload_mass.setter  # type: ignore[attr-defined]
    def payload_mass(self, value):
        MultiQuadRigidbodyBase.payload_mass.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            self._mjMdl.model.body_mass[self._payload_body_id] = self._payload_mass

    @MultiQuadRigidbodyBase.payload_inertia.setter  # type: ignore[attr-defined]
    def payload_inertia(self, value):
        MultiQuadRigidbodyBase.payload_inertia.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            self._mjMdl.model.body_inertia[self._payload_body_id] = np.diag(self._payload_inertia)

    @MultiQuadRigidbodyBase.cable_lengths.setter  # type: ignore[attr-defined]
    def cable_lengths(self, value):
        MultiQuadRigidbodyBase.cable_lengths.fset(self, value)  # type: ignore[attr-defined]
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
        call ``super().reset()``. ``payload_position`` is the **absolute**
        position the payload should land at — the whole formation, quadrotors
        included, translates so that it does. That matches ``multi-quad``/M3;
        the retired model read the same argument as a relative *offset* under
        identical help text.

        A rigid translation is the *only* override this backend can honour, so
        it is the only one accepted. The base class documents seven more, and
        this method used to drop every one of them on the floor — including
        through ``simulate(tf, payload_orientation=...)``, which then ran a
        completely different initial condition from the one asked for, silently.
        Setting them for real means re-solving the whole formation against the
        tendon constraints (a cable is a *constraint* here, not a state), which
        is the MJCF-generator step's job. Until then an unsupported kwarg is an
        error, not a no-op.

        Raises:
            ValueError: for any keyword other than ``payload_position``.
        """
        unsupported = sorted(k for k in kwargs if k not in _SUPPORTED_RESET_KWARGS)
        if unsupported:
            raise ValueError(
                f"{type(self).__name__}.reset() cannot honour {unsupported}: the "
                "initial formation is fixed by multi_quad_rigidbody.xml and its "
                "tendon constraints, and only a rigid translation of the whole "
                f"formation ({', '.join(sorted(_SUPPORTED_RESET_KWARGS))}) can be "
                "applied to it. Use MultiQuadRigidbodyBase for arbitrary initial "
                "conditions, or set the MuJoCo state directly through _mjMdl.data."
            )

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
        self.state.payload_orientation = SO3(self._mjMdl._quat2rot(data.qpos[i + 3 : i + 7]))
        i = self._payload_qveladr
        self.state.payload_velocity = copy.deepcopy(data.qvel[i : i + 3])
        # A MuJoCo free joint stores its angular velocity in the *body* frame,
        # which is the frame Ω_L lives in here.
        self.state.payload_angular_velocity = TSO3(copy.deepcopy(data.qvel[i + 3 : i + 6]))

        for k in range(self._num_quadrotors):
            i = self._quad_qposadr[k]
            self.state.positions[k] = copy.deepcopy(data.qpos[i : i + 3])
            self.state.orientations[k] = SO3(self._mjMdl._quat2rot(data.qpos[i + 3 : i + 7]))
            i = self._quad_qveladr[k]
            self.state.velocities[k] = copy.deepcopy(data.qvel[i : i + 3])
            self.state.angular_velocities[k] = TSO3(copy.deepcopy(data.qvel[i + 3 : i + 6]))

            # Cable direction and angular velocity from the measured endpoints —
            # the *attachment point*, not the payload centre, which is the whole
            # difference from M3. The 1/length factor in ω is essential; the
            # retired model dropped it and scaled every cable rate by the cable
            # length. See cable_state_from_geometry.
            try:
                q, omega, _ = cable_state_from_geometry(
                    self.attachment_position(k),
                    self.attachment_velocity(k),
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
