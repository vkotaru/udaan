"""MultiQuadRigidbodyMujoco — MuJoCo physics backend for the nQ-quad rigid-body payload."""

from __future__ import annotations

import copy
import hashlib
import json

import mujoco
import numpy as np

from ...core.defaults import (
    DEFAULT_ATTACHMENT_HEIGHT,
    DEFAULT_ATTACHMENT_RADIUS,
    DEFAULT_NUM_QUADROTORS,
    DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS,
    DEFAULT_SPAWN_HEIGHT,
)
from ...manif import SO3, TSO3
from ...utils.assets.cache import ensure_generated
from ...utils.assets.mujoco_asset_creator import MJCF_GENERATOR_VERSION
from ...utils.grasp import hover_wrench_map, ring_attachment_points
from ...utils.logging import get_logger
from ..backend import MujocoModel
from .base import MultiQuadRigidbodyBase, cable_state_from_geometry

_logger = get_logger(__name__)


#: Body/site name templates emitted by
#: :func:`udaan.utils.assets.multi_quad_rigidbody`.
_QUAD_BODY = "quad{}"
_QUAD_SITE = "quad{}_end1"
_PAYLOAD_BODY = "pyld"
_PAYLOAD_SITE = "pyld_poc{}"

#: Geometry read out of the MJCF must match the declared parameters to this
#: absolute tolerance, in metres. The file stores coordinates with full double
#: precision, so real agreement is at round-off; anything looser would be
#: hiding a genuine difference.
_GEOMETRY_TOL = 1e-9

#: Spawn slack, in metres, above which the loaded formation is reported as not
#: taut. The generator derives the payload height from the cable length, so
#: this can only fire on an asset that did not come from it.
_SLACK_TOL = 1e-6

#: Static-hover residual, in newtons, above which the loaded grasp geometry is
#: reported as unable to hover under the controller's own trim. A file and a
#: controller that agree read ~1e-14 N; a quadrotor ring hand-edited from
#: 0.50 m to 0.80 m reads 20 N.
_HOVER_RESIDUAL_TOL = 1e-6

#: The reset overrides this backend can actually apply to the MJCF formation.
#: See :meth:`MultiQuadRigidbodyMujoco.reset`.
_SUPPORTED_RESET_KWARGS = frozenset({"payload_position"})


class MultiQuadRigidbodyMujoco(MultiQuadRigidbodyBase):
    """``nQ`` quadrotors carrying one cable-suspended rigid-body payload, in MuJoCo.

    Overrides ``step``/``reset``/``_query_latest_state`` to delegate to MuJoCo;
    the input pipeline, controllers and parameter bookkeeping are inherited.
    The cables are MuJoCo spatial tendons with an upper length limit, so they
    pull but never push.

    The MJCF is generated on demand from the construction parameters
    (``num_quadrotors``, ``cable_lengths``, ``payload_mass``, ``masses``,
    ``attachment_radius``, ``attachment_height``, ``payload_half_extents``,
    ``formation_radius`` and ``spawn_height``) and cached under a filename
    carrying a hash of them, so a changed parameter never reuses a stale asset.
    The generated formation starts **taut** and its attachment azimuths are the
    model's own — the two defects of the retired hand-written asset, both of
    which are re-checked here at load time rather than trusted.

    The asset also differs from a stock MuJoCo model in two ways that exist to
    make this backend and :class:`MultiQuadRigidbodyBase` the *same machine*
    rather than merely similar ones, and both belong on any list of what a
    base-vs-MuJoCo comparison is and is not tolerating:

    * **No free-joint damping.** ``MujocoAssetCreator.joint`` defaults to
      ``damping=0.1``; on the quadrotors' and the payload's free joints that is
      a viscous drag on all six degrees of freedom, and the analytical model
      has no such term. It was the largest single contributor to backend
      disagreement — two thirds of it.
    * **A 1.25 ms integrator timestep**, the analytical model's own inner
      substep. MuJoCo's default 2 ms does not divide the 5 ms outer ``dt``, so
      this backend took ``round(2.5) = 2`` substeps and advanced 4 ms per
      :meth:`step` while reporting 5 ms; the two backends then ran to different
      final times under the same ``tf``.

    What *is* left for the tolerance to absorb is integration: ZOH Euler on a
    constraint-eliminated reduction where the cable is exact and inextensible,
    against MuJoCo's solver with the cable as a soft tendon-limit constraint.

    Because one MJCF encodes one ring and one tendon length, ``masses`` and
    ``cable_lengths`` must be uniform at construction time; assign per-index
    values afterwards (the setters write straight into MuJoCo).

    ``attachment_points`` cannot be passed at construction: the generator lays
    the sites out on the ring the *model* uses, and an arbitrary set would
    disagree with the file. Pass ``attachment_radius``/``attachment_height``
    instead, which move the ring in both the MJCF and the model together.

    ``inertias`` cannot be passed either. MuJoCo derives the body inertia from
    the generated geometry (``inertiafromgeom``) and
    :meth:`_read_params_from_mjcf` adopts it, so a constructor argument would
    be silently overwritten; it is rejected instead. Assign ``mdl.inertias``
    after construction — that setter *does* write into MuJoCo.

    Masses, inertias, the payload's mass and inertia and the cable lengths are
    all read back from the file, because MuJoCo is the authority on them: a
    body's mass is the sum of its geoms', and the retired model's hardcoded
    ``mQ = 0.75`` was the mass of the quadrotor's *centre box* alone — the arms
    and propellers brought that airframe to 0.99 kg, so every feedforward term
    was 24% light.
    """

    def __init__(self, **kwargs):
        self.render = kwargs.get("render", False)
        if "inertias" in kwargs:
            raise ValueError(
                "the MuJoCo backend derives inertias from the generated MJCF geometry, "
                "so an `inertias` constructor argument would be silently overwritten; "
                "construct the model and then assign `mdl.inertias`"
            )
        if "attachment_points" in kwargs:
            raise ValueError(
                "the MuJoCo backend lays the pyld_poc sites out on the model's own "
                "ring, so an `attachment_points` constructor argument would disagree "
                "with the generated MJCF; pass attachment_radius/attachment_height "
                "instead, which move the ring in both"
            )

        self._payload_half_extents = np.asarray(
            kwargs.pop("payload_half_extents", DEFAULT_RIGID_PAYLOAD_HALF_EXTENTS), dtype=float
        )
        self._attachment_radius = float(kwargs.pop("attachment_radius", DEFAULT_ATTACHMENT_RADIUS))
        self._attachment_height = float(kwargs.pop("attachment_height", DEFAULT_ATTACHMENT_HEIGHT))
        self._spawn_height = float(kwargs.pop("spawn_height", DEFAULT_SPAWN_HEIGHT))

        # The sites the generator emits *are* the model's ring, so the two are
        # built from one call rather than declared twice and compared later.
        # (_read_params_from_mjcf still compares them: a hand-edited asset, or
        # a stale cache, would otherwise put the analytical grasp and the
        # physics on different geometries — the retired asset's failure mode.)
        nQ = int(kwargs.get("num_quadrotors", DEFAULT_NUM_QUADROTORS))
        kwargs["attachment_points"] = ring_attachment_points(
            nQ, self._attachment_radius, self._attachment_height
        )
        super().__init__(**kwargs)

        xml_path = self._ensure_xml(
            self._num_quadrotors,
            self._uniform(self._cable_lengths, "cable_lengths"),
            self._payload_mass,
            self._uniform(self._masses, "masses"),
            self._payload_half_extents,
            self._attachment_radius,
            self._attachment_height,
            self._formation_radius,
            self._spawn_height,
        )
        self._mjMdl = MujocoModel(model_path=xml_path, render=self.render)

        self._resolve_mj_indices()
        self._read_params_from_mjcf()

        self._mjDt = float(self._mjMdl.model.opt.timestep)
        self._step_iter = max(1, int(round(self.dt / self._mjDt)))
        # `step()` advances `_step_iter * _mjDt`, which equals the requested dt
        # only when the timestep divides it. The generated asset sets
        # DEFAULT_MJCF_TIMESTEP so that it does; a hand-edited or stale file
        # need not, and `dt` then names a duration this backend never takes —
        # which is how a `while self.t < tf` loop here ended 5 ms short of the
        # analytical model's over a 2 s run and the two were compared at
        # different simulation times. Report what is actually taken.
        achieved = self._step_iter * self._mjDt
        if abs(achieved - self.dt) > 1e-12:
            _logger.warning(
                "MuJoCo timestep %.6g s does not divide dt = %.6g s; each step() "
                "advances %.6g s. Reporting dt = %.6g so that self.t stays truthful.",
                self._mjDt,
                self.dt,
                achieved,
                achieved,
            )
            self.dt = achieved

        # Reinit controllers with the MuJoCo parameters
        self._init_default_controllers()

        self._check_spawn_is_taut()

        if self.verbose:
            _logger.info(
                "MuJoCo multi-quad rigid-body payload loaded (%d quadrotors)", self._num_quadrotors
            )

    # ─── MJCF generation ───────────────────────────────────────────

    @staticmethod
    def _uniform(values: np.ndarray, name: str) -> float:
        """Collapse a per-quadrotor array to the single value the MJCF encodes.

        The generator lays every quadrotor out on one ring with one tendon
        length, so a per-index array it cannot express is an error rather than
        a value to be silently averaged away.  The comparison is **exact**, for
        the reason :meth:`MultiQuadCsPayloadMujoco._uniform` documents: there is
        no tolerance this backend could honour — either the MJCF encodes the
        value or it does not.
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
        payload_half_extents: np.ndarray,
        attachment_radius: float,
        attachment_height: float,
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
        # A lone quadrotor hovers on the payload's own axis; neither ring
        # radius is used, so neither may enter the hash either.
        if num_quadrotors == 1:
            formation_radius = 0.0
            attachment_radius = 0.0
        config = {
            "v": MJCF_GENERATOR_VERSION,
            "nQ": int(num_quadrotors),
            "L": float(cable_length),
            "mL": float(payload_mass),
            "mQ": float(quad_mass),
            "e": [float(v) for v in np.asarray(payload_half_extents).reshape(-1)],
            "ra": float(attachment_radius),
            "ha": float(attachment_height),
            "r": float(formation_radius),
            "z": float(spawn_height),
        }
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        generated_name = f"multi_quad_rigidbody_{num_quadrotors}q_{config_hash}.xml"

        def build(path: str) -> None:
            from ...utils.assets import xml_model_generator

            _logger.info("Generating MJCF for %d quadrotors: %s", num_quadrotors, generated_name)
            xml_model_generator.multi_quad_rigidbody(
                num_quadrotors=num_quadrotors,
                filename=path,
                cable_length=cable_length,
                payload_mass=payload_mass,
                payload_half_extents=payload_half_extents,
                quad_mass=quad_mass,
                attachment_radius=attachment_radius,
                attachment_height=attachment_height,
                formation_radius=formation_radius,
                spawn_height=spawn_height,
                verbose=False,
            )

        # Generated assets go to the user cache, not into the installed
        # package, and the write is atomic — see
        # :func:`udaan.utils.assets.cache.ensure_generated`.
        return ensure_generated(generated_name, build)

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

    def _check_spawn_is_taut(self):
        """Refuse a spawn pose whose cables are shorter than the tendon limit.

        The retired hand-written asset spawned the quadrotors 0.61 m above
        attachment points 0.5 m inboard of them, so every cable measured
        0.7887 m against a 1.0 m tendon limit: 21% slack, no tension at ``t =
        0``, and — the tendons carrying no stiffness — a payload that dropped
        ~0.2 m before the limit constraint caught it. The analytical model has
        no slack mode at all, so the two backends were not simulating the same
        machine, and every base-vs-MuJoCo comparison would have been measuring
        that instead of an integration difference.

        :func:`udaan.utils.assets.rigid_payload_height` now derives the payload
        height from the cable length, so this cannot happen through the
        generator. It stays as a check rather than a comment because a stale
        cached asset or a hand-edited file would reintroduce it in silence.
        """
        mujoco.mj_forward(self._mjMdl.model, self._mjMdl.data)
        self._query_latest_state()
        measured = np.array(
            [
                np.linalg.norm(self.attachment_position(i) - self.state.positions[i])
                for i in range(self._num_quadrotors)
            ]
        )
        if not np.allclose(measured, self._cable_lengths, rtol=0.0, atol=_SLACK_TOL):
            raise RuntimeError(
                "the loaded MJCF does not spawn taut: quadrotor-to-attachment "
                f"distances {np.array2string(measured, precision=4)} against tendon "
                f"limits {np.array2string(self._cable_lengths, precision=4)}. The "
                "payload would drop before the cables caught, and the analytical "
                "model has no slack mode to match it. Delete the cached asset so it "
                "is regenerated from the cable length."
            )

        residual = self.mjcf_hover_residual
        if residual is not None and residual > _HOVER_RESIDUAL_TOL:
            _logger.warning(
                "the loaded MJCF does not hover under the payload cascade's own trim: "
                "applying its hover tensions along the file's measured cable "
                "directions leaves a %.4g N residual. Either the file's geometry "
                "admits no static hover, or it is not the geometry the controller "
                "allocates through — a stale cached asset or a hand-edited one. The "
                "formation will not settle. Delete the cached asset so it is "
                "regenerated from the declared parameters.",
                residual,
            )

    @property
    def mjcf_hover_residual(self) -> float | None:
        """Static-hover residual of the **loaded file**, in newtons.

        The payload cascade's own
        :attr:`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController.hover_residual`
        is not this number and cannot be: it is solved from the cable
        directions the *constructor's* ``formation_radius``/
        ``attachment_radius``/``attachment_height`` imply, and never reads the
        MJCF at all. It therefore stays at round-off for exactly the case it
        was meant to catch — a hand-edited asset whose quadrotor ring sits at
        0.80 m instead of 0.50 m loads clean, passes
        :meth:`_check_spawn_is_taut` (every cable is still exactly 1.000 m) and
        reports 3.9e-15 N while the controller's assumed cable directions are
        36° off the file's.

        This one takes both ingredients from the file after
        ``mj_forward``: the measured cable directions ``q_i`` and the measured
        attachment sites ``r_i``. It then asks whether the *controller's* hover
        tensions balance the payload on *that* geometry,

        .. math::

            \\bigl\\| G(q^{\\mathrm{file}}, r^{\\mathrm{file}})\\, t^{\\mathrm{ctrl}}
              - [-m_L g e_3;\\; 0] \\bigr\\| ,

        which is zero only when the file's grasp both admits a hover and is the
        one the controller allocates through. A file/controller *pairing*
        mismatch is what the retired asset really had — its corner attachments
        hover at 7e-15 N under the 45° directions its own quadrotor ring
        implies, and at 2.7208 N under the model's 0°/90° ring.

        ``None`` when the payload controller publishes no ``hover_tensions``,
        which a caller-substituted one need not.
        """
        tensions = getattr(self._payload_controller, "hover_tensions", None)
        if tensions is None:
            return None
        model = self._mjMdl.model
        q = np.array(
            [np.asarray(self.state.cable_attitudes[i]) for i in range(self._num_quadrotors)]
        )
        r = np.array([model.site_pos[i] for i in self._attachment_site_id])
        wrench_map = hover_wrench_map(q, r, self.state.payload_orientation)
        target = np.concatenate([-self._payload_mass * self._ge3, np.zeros(3)])
        return float(np.linalg.norm(wrench_map @ np.asarray(tensions) - target))

    # ─── Generated geometry ────────────────────────────────────────

    @property
    def attachment_radius(self) -> float:
        """Radius of the attachment ring in the payload body frame, in metres."""
        return self._attachment_radius

    @property
    def attachment_height(self) -> float:
        """Height of the attachment ring above the payload centre, in metres."""
        return self._attachment_height

    @property
    def payload_half_extents(self) -> np.ndarray:
        """``(a, b, c)`` half-extents of the slab the MJCF encodes, in metres."""
        return self._payload_half_extents

    @property
    def spawn_height(self) -> float:
        """Height of the quadrotor ring the MJCF spawns at, in metres."""
        return self._spawn_height

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
                "initial formation is fixed by the generated MJCF and its "
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
