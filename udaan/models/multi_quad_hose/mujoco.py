"""MultiQuadHoseMujoco — MuJoCo physics backend for the M6 flexible-hose system."""

from __future__ import annotations

import copy
import hashlib
import json

import mujoco
import numpy as np

from ...core.defaults import DEFAULT_HOSE_MASS, DEFAULT_SPAWN_HEIGHT
from ...manif import S2, SO3, TS2, TSO3
from ...utils.assets.cache import ensure_generated
from ...utils.assets.mujoco_asset_creator import MJCF_GENERATOR_VERSION
from ...utils.logging import get_logger
from ..backend import MujocoModel
from .base import MultiQuadHoseBase

_logger = get_logger(__name__)


#: Body/joint name templates emitted by
#: :func:`udaan.utils.assets.xml_model_generator.multi_quad_hose`.
_ROOT_BODY = "hose_vertex0"
_VERTEX_GEOM = "hose_vertex_geom{}"
_ROOT_JOINT = "hose_root_joint"
_LINK_BODY = "hose_link{}"
_LINK_JOINT = "hose_joint{}"
_QUAD_BODY = "quad{}"

#: Link lengths read out of the MJCF must match the declared ones to this
#: absolute tolerance, in metres — round-off, because they are read from the
#: **joint anchors**, which ``MujocoAssetCreator.joint`` writes with ``str()``
#: and therefore at full double precision. The cylinder geoms are not the place
#: to read them: ``MujocoAssetCreator.cylinder`` formats ``size`` with ``%g``,
#: six significant digits, which is six orders of magnitude past this
#: tolerance. See M5's backend for the full argument.
_GEOMETRY_TOL = 1e-12

#: The reset overrides this backend can apply.
_SUPPORTED_RESET_KWARGS = frozenset(
    {
        "root_position",
        "root_velocity",
        "link_attitudes",
        "link_angular_velocities",
        "orientations",
        "angular_velocities",
    }
)

_E1 = np.array([1.0, 0.0, 0.0])


class MultiQuadHoseMujoco(MultiQuadHoseBase):
    """``nQ`` quadrotors carrying one ``n``-link flexible hose, in MuJoCo.

    Overrides ``step``/``reset``/``_query_latest_state`` to delegate to MuJoCo;
    the input pipeline, controllers and parameter bookkeeping are inherited.

    The MJCF is generated on demand from the construction parameters
    (``num_links``, ``attachments``, ``hose_length``, ``hose_mass``, the
    airframe mass, ``sag_ratio`` and ``spawn_height``) and cached under a
    filename carrying a hash of them, so a changed parameter never reuses a
    stale asset.  There is no tendon in it: the hose is a kinematic tree of ball
    joints, which is why — like
    :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableMujoco` and
    unlike :class:`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyMujoco`
    — this backend can honour arbitrary initial link directions and rates.

    **The per-link state is measured, not reconstructed.**  Every link's
    direction comes from its body frame (``q_i = -R_i e_3``, exact) and its rate
    from ``mj_objectVelocity`` (exact); nothing is finite-differenced.

    **The two backends are very nearly the same machine** — closer than M5 can
    manage — and the reason is where the mass sits.  M5's MJCF spreads each
    link's mass along a cylinder and its backend lumps that onto the distal
    joint, a first-order approximation.  Here the hose's mass is already *at*
    the vertices — the cylinders are massless and each link body carries a
    sphere at its distal end — so MuJoCo's derived centre of mass lands exactly
    where the analytical model lumps it.

    They are not *identical*, and it is worth being exact about the gap,
    because it is what makes the measured agreement evidence rather than a
    tautology.  Each vertex sphere carries ``2/5 m r^2 = 1.6e-5 kg m^2`` of its
    own rotational inertia, which has no term at all in the analytical model
    (0.06% of ``m l^2`` at the default geometry).  That single omission is the
    *whole* residual: root-acceleration disagreement at ``n = 3`` on the
    defaults is 4.05e-4 m/s^2, and shrinking the sphere radius drives it as
    exactly ``r^2`` — 4.05e-4, 4.05e-6, 4.05e-8 for ``r`` = 0.02, 0.002,
    0.0002.  A backend that agreed because it was reading the analytical
    model's own numbers would not scale with a geometric parameter the
    analytical model never sees.

    The other difference is structural rather than numerical: MuJoCo carries
    the gauge degrees of freedom the analytical model
    factors out — the spin of each link about its own axis and the rotation of
    the root vertex.  Neither is driven: every force in this model acts through
    a body's own centre of mass, and every ball-joint anchor sits on its
    parent's own symmetry axis, so no torque reaches them.  The root's rotation
    is therefore pinned outright.  A link's body frame does show a non-zero
    ``omega_i . q_i`` further down the chain, but that is *inherited* — the
    parent's tilt rate has a component along ``q_i`` whenever two links are not
    parallel — and it is exactly the direction ``q_i = -R_i e_3`` is blind to,
    so :meth:`_query_latest_state` projects it out and the integrated state
    never sees it.

    Contacts are disabled in the generated file, deliberately; see
    :func:`udaan.utils.assets.xml_model_generator.multi_quad_hose`.

    ``masses``, ``inertias`` and ``vertex_masses`` are read back from the file
    and their setters write into it, because MuJoCo is the authority on them.
    ``link_lengths`` is read back and *checked*, but assigning it afterwards
    changes only the analytical parameters — a length change re-shapes the whole
    body tree, which is the generator's job.
    """

    def __init__(self, **kwargs):
        self.render = kwargs.get("render", False)
        self._spawn_height = float(kwargs.pop("spawn_height", DEFAULT_SPAWN_HEIGHT))
        if "vertex_masses" in kwargs or "link_lengths" in kwargs:
            raise ValueError(
                "the MuJoCo backend derives vertex masses and link lengths from the "
                "generated MJCF, so `vertex_masses`/`link_lengths` constructor arguments "
                "would be silently overwritten; pass hose_length/hose_mass instead, or "
                "assign `mdl.vertex_masses` after construction"
            )
        if "inertias" in kwargs:
            raise ValueError(
                "the MuJoCo backend derives inertias from the generated MJCF geometry, so "
                "an `inertias` constructor argument would be silently overwritten; "
                "construct the model and then assign `mdl.inertias`"
            )
        self._hose_mass = float(kwargs.get("hose_mass", DEFAULT_HOSE_MASS))

        super().__init__(**kwargs)

        xml_path = self._ensure_xml(
            self._num_links,
            self._attachments,
            self.hose_length,
            self._hose_mass,
            self._uniform(self._masses, "masses"),
            self._sag_ratio,
            self._spawn_height,
        )
        self._mjMdl = MujocoModel(model_path=xml_path, render=self.render)

        self._resolve_mj_indices()
        self._read_params_from_mjcf()

        self._mjDt = float(self._mjMdl.model.opt.timestep)
        self._step_iter = max(1, int(round(self.dt / self._mjDt)))
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

        mujoco.mj_forward(self._mjMdl.model, self._mjMdl.data)
        self._query_latest_state()
        self._station_targets = [p.copy() for p in self.state.positions]

        if self.verbose:
            _logger.info(
                "MuJoCo multi-quad hose loaded (%d links, %d quadrotors at %s)",
                self._num_links,
                self._num_quadrotors,
                self._attachments,
            )

    # ─── MJCF generation ───────────────────────────────────────────

    @staticmethod
    def _uniform(values: np.ndarray, name: str) -> float:
        """Collapse a per-index array to the single value the MJCF encodes.

        The generator gives every airframe the same geometry, so a per-index
        array it cannot express is an error rather than a value to be silently
        averaged away.  The comparison is **exact**: either the MJCF encodes the
        value or it does not, so there is no tolerance this backend could
        honour.
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
        num_links: int,
        attachments: tuple[int, ...],
        hose_length: float,
        hose_mass: float,
        quad_mass: float,
        sag_ratio: float,
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
        config = {
            "v": MJCF_GENERATOR_VERSION,
            "n": int(num_links),
            "a": [int(i) for i in attachments],
            "L": float(hose_length),
            "mh": float(hose_mass),
            "mQ": float(quad_mass),
            "s": float(sag_ratio),
            "z": float(spawn_height),
        }
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        generated_name = f"multi_quad_hose_{num_links}l_{len(attachments)}q_{config_hash}.xml"

        def build(path: str) -> None:
            from ...utils.assets import xml_model_generator

            _logger.info(
                "Generating MJCF for a %d-link hose with %d quadrotors: %s",
                num_links,
                len(attachments),
                generated_name,
            )
            xml_model_generator.multi_quad_hose(
                num_links=num_links,
                attachments=attachments,
                filename=path,
                hose_length=hose_length,
                hose_mass=hose_mass,
                quad_mass=quad_mass,
                sag_ratio=sag_ratio,
                spawn_height=spawn_height,
                verbose=False,
            )

        # Generated assets go to the user cache, not into the installed
        # package, and the write is atomic — see
        # :func:`udaan.utils.assets.cache.ensure_generated`.
        return ensure_generated(generated_name, build)

    # ─── Model initialization ──────────────────────────────────────

    def _resolve_mj_indices(self):
        """Look up body/joint ids by name and check the tree is the declared one."""
        model = self._mjMdl.model
        n = self._num_links
        nQ = self._num_quadrotors

        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _ROOT_BODY)
        if root_id < 0:
            raise RuntimeError(f"MJCF has no body named {_ROOT_BODY!r}")
        self._root_body_id = root_id
        root_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _ROOT_JOINT)
        if root_jnt < 0:
            raise RuntimeError(f"MJCF has no joint named {_ROOT_JOINT!r}")
        self._root_qposadr = int(model.jnt_qposadr[root_jnt])
        self._root_qveladr = int(model.jnt_dofadr[root_jnt])

        self._link_body_id = np.zeros(n, dtype=int)
        self._link_jnt_id = np.zeros(n, dtype=int)
        self._link_qposadr = np.zeros(n, dtype=int)
        self._link_qveladr = np.zeros(n, dtype=int)
        for i in range(n):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _LINK_BODY.format(i))
            if body_id < 0:
                raise RuntimeError(f"MJCF has no body named {_LINK_BODY.format(i)!r}")
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _LINK_JOINT.format(i))
            if jnt_id < 0:
                raise RuntimeError(f"MJCF has no joint named {_LINK_JOINT.format(i)!r}")
            if int(model.jnt_type[jnt_id]) != int(mujoco.mjtJoint.mjJNT_BALL):
                raise RuntimeError(f"joint {_LINK_JOINT.format(i)!r} is not a ball joint")
            self._link_body_id[i] = body_id
            self._link_jnt_id[i] = jnt_id
            self._link_qposadr[i] = model.jnt_qposadr[jnt_id]
            self._link_qveladr[i] = model.jnt_dofadr[jnt_id]

        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _LINK_BODY.format(n)) >= 0:
            raise RuntimeError(f"MJCF defines more than the expected {n} hose links")

        self._quad_body_id = np.zeros(nQ, dtype=int)
        self._quad_qposadr = np.zeros(nQ, dtype=int)
        self._quad_qveladr = np.zeros(nQ, dtype=int)
        for k in range(nQ):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _QUAD_BODY.format(k))
            if body_id < 0:
                raise RuntimeError(f"MJCF has no body named {_QUAD_BODY.format(k)!r}")
            jnt_id = int(model.body_jntadr[body_id])
            if int(model.jnt_type[jnt_id]) != int(mujoco.mjtJoint.mjJNT_BALL):
                raise RuntimeError(f"quadrotor {k}'s root joint is not a ball joint")
            self._quad_body_id[k] = body_id
            self._quad_qposadr[k] = model.jnt_qposadr[jnt_id]
            self._quad_qveladr[k] = model.jnt_dofadr[jnt_id]

        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _QUAD_BODY.format(nQ)) >= 0:
            raise RuntimeError(f"MJCF defines more than the expected {nQ} quadrotors")

        # Each vehicle is a child of the body owning its vertex, so the
        # attachment pattern is recoverable from the tree and can be checked
        # rather than assumed.
        expected = [
            self._root_body_id if v == 0 else int(self._link_body_id[v - 1])
            for v in self._attachments
        ]
        actual = [int(model.body_parentid[i]) for i in self._quad_body_id]
        if actual != expected:
            raise RuntimeError(
                f"the MJCF bolts the quadrotors to bodies {actual}, but attachments "
                f"{self._attachments} put them on {expected}"
            )

        if model.nu != 4 * nQ:
            raise RuntimeError(f"MJCF exposes {model.nu} actuators, expected {4 * nQ}")

        # Radius of each vertex's mass sphere, cached so the `vertex_masses`
        # setter can rebuild `body_inertia` rather than leave it stale — see
        # the setter. Read from the compiled model, not from the generator
        # default, so a file written with a different `vertex_radius` is
        # still handled correctly.
        self._vertex_radius = np.zeros(n + 1)
        for v in range(n + 1):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _VERTEX_GEOM.format(v))
            if geom_id < 0:
                raise RuntimeError(f"MJCF has no geom named {_VERTEX_GEOM.format(v)!r}")
            self._vertex_radius[v] = float(model.geom_size[geom_id, 0])

        # A body's world orientation is ``R_parent · body_quat · joint_quat``,
        # so writing a desired attitude into a ball joint means dividing the
        # body's own spawn rotation back out. M5's chain gets away without this
        # because every link body there is emitted with an identity ``quat``
        # and the whole cable hangs straight down; M6's spawn folds, so each
        # link body carries a real rotation and ``body_quat`` is not identity.
        self._link_body_quat = np.stack(
            [self._quat2rot(model.body_quat[int(i)]) for i in self._link_body_id]
        )
        self._quad_body_quat = np.stack(
            [self._quat2rot(model.body_quat[int(i)]) for i in self._quad_body_id]
        )

    def _read_params_from_mjcf(self):
        """Adopt the file's masses, inertias and link lengths.

        The link lengths come from the ball-joint anchors and are *checked*
        against the declared ones — a mismatch means the analytical model and
        the physics describe different hoses.  Every other mass is adopted:
        each link body's mass is the sphere at its distal end, i.e. exactly the
        analytical model's vertex mass, with no lumping approximation to make.
        """
        model = self._mjMdl.model
        n = self._num_links

        lengths = np.zeros(n)
        vertex_masses = np.zeros(n + 1)
        vertex_masses[0] = float(model.body_mass[self._root_body_id])
        for i in range(n):
            body_id = int(self._link_body_id[i])
            # The ball joint sits half a link above the body's own origin, so
            # the link's length is twice that offset. Read from the joint, not
            # the geom — see _GEOMETRY_TOL.
            lengths[i] = 2.0 * abs(float(model.jnt_pos[int(self._link_jnt_id[i]), 2]))
            vertex_masses[i + 1] = float(model.body_mass[body_id])

        if not np.allclose(lengths, self._link_lengths, rtol=0.0, atol=_GEOMETRY_TOL):
            raise RuntimeError(
                "link_lengths disagree with the MJCF's joint anchors:\n"
                f"  model: {self._link_lengths.tolist()}\n"
                f"  MJCF:  {lengths.tolist()}"
            )
        self.link_lengths = lengths

        # Order matters: the masses guard checks the *total* at every vertex, so
        # the airframe masses have to be in place before the hose's are relaxed.
        self.masses = np.array([model.body_mass[i] for i in self._quad_body_id])
        self.vertex_masses = vertex_masses
        # body_inertia holds principal-axis diagonals; the model wants tensors.
        self.inertias = np.stack([np.diag(model.body_inertia[i]) for i in self._quad_body_id])

    # ─── Generated geometry ────────────────────────────────────────

    @property
    def spawn_height(self) -> float:
        """Height the MJCF spawns vertex 0 at, in metres."""
        return self._spawn_height

    # ─── Property setters (sync to MuJoCo) ─────────────────────────
    #
    # The `@Base.prop.setter` override pattern is standard at runtime but mypy
    # cannot model it, hence the attr-defined ignores below.

    @MultiQuadHoseBase.masses.setter  # type: ignore[attr-defined]
    def masses(self, value):
        MultiQuadHoseBase.masses.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            for k, body_id in enumerate(self._quad_body_id):
                self._mjMdl.model.body_mass[body_id] = self._masses[k]

    @MultiQuadHoseBase.inertias.setter  # type: ignore[attr-defined]
    def inertias(self, value):
        MultiQuadHoseBase.inertias.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            for k, body_id in enumerate(self._quad_body_id):
                self._mjMdl.model.body_inertia[body_id] = np.diag(self._inertias[k])

    @MultiQuadHoseBase.vertex_masses.setter  # type: ignore[attr-defined]
    def vertex_masses(self, value):
        """Push new hose vertex masses into MuJoCo, inertia included.

        Each vertex is one sphere of radius ``r``, so its ``body_inertia`` is
        ``2/5 m r^2`` on every axis and is *not* independent of the mass.
        MuJoCo derives it once at compile time and never again, so writing
        ``body_mass`` alone would leave every vertex spinning with the inertia
        of the mass it used to have — the same class of staleness M4's
        controller cache had.  The radii are read off the compiled model in
        :meth:`_resolve_mj_indices`.
        """
        MultiQuadHoseBase.vertex_masses.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            model = self._mjMdl.model
            radii = self._vertex_radius
            for v, body_id in enumerate(
                [self._root_body_id, *(int(b) for b in self._link_body_id)]
            ):
                mass = float(self._vertex_masses[v])
                model.body_mass[body_id] = mass
                model.body_inertia[body_id] = 0.4 * mass * radii[v] ** 2

    # `link_lengths` deliberately keeps the base setter: pushing a length back
    # into the file would mean re-deriving the whole body tree, which is the
    # generator's job — regenerate with a different hose_length instead.

    # ─── Reset ─────────────────────────────────────────────────────

    def reset(self, **kwargs):
        """Reset MuJoCo to a requested configuration.

        Every kwarg the analytical base accepts is honoured.  ``root_position``
        and ``root_velocity`` write the free joint at the top of the tree
        directly; the link directions and rates are walked down the chain, and
        each vehicle's attitude is written into its own ball joint relative to
        the body it is bolted to.

        Off-manifold per-link input is repaired exactly as the base
        :meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.reset` repairs it
        — ``q_i`` normalised, ``omega_i`` projected onto the tangent plane — so
        the two backends land on the same state from the same kwargs.  MuJoCo
        has no choice about it (a ball joint cannot represent a stretched link,
        and a spin about the link axis is unobservable in ``q``), which is why
        the base was made to match this rather than the other way round.

        Raises:
            ValueError: for any unsupported keyword.
        """
        unsupported = sorted(k for k in kwargs if k not in _SUPPORTED_RESET_KWARGS)
        if unsupported:
            raise ValueError(
                f"{type(self).__name__}.reset() cannot honour {unsupported}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_RESET_KWARGS))}."
            )

        self.t = 0.0
        self._mjMdl.reset()
        self.state.reset()
        data = self._mjMdl.data
        model = self._mjMdl.model

        n = self._num_links
        directions = self._check_per_index(
            kwargs.get("link_attitudes", [np.asarray(q) for q in self.spawn_link_attitudes()]),
            "link_attitudes",
            n,
        )
        rates = self._check_per_index(
            kwargs.get("link_angular_velocities", [np.zeros(3)] * n),
            "link_angular_velocities",
            n,
        )

        # Vertex 0 is a point mass: the analytical model gives it no attitude,
        # and no force in this system produces a torque about any body's own
        # centre of mass, so the root's own rotational degrees of freedom are
        # inert. Pin them at identity and zero rather than leave them to drift.
        adr = self._root_qposadr
        if "root_position" in kwargs:
            data.qpos[adr : adr + 3] = np.asarray(kwargs["root_position"], dtype=float)
        data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        adr = self._root_qveladr
        if "root_velocity" in kwargs:
            data.qvel[adr : adr + 3] = np.asarray(kwargs["root_velocity"], dtype=float)
        data.qvel[adr + 3 : adr + 6] = 0.0

        # Walk the chain: a ball joint's qpos is the child's rotation relative to
        # its parent, and its qvel is the *relative* angular velocity expressed
        # in the child's frame. Frames start world-aligned at the root.
        parent_frames = [np.eye(3)]
        parent_omegas = [np.zeros(3)]
        for k in range(n):
            q_k = np.asarray(directions[k], dtype=float)
            q_k = q_k / np.linalg.norm(q_k)
            # The link's body z-axis points from its distal end back to its
            # proximal joint, so its third column is -q_k.
            frame = np.asarray(SO3.from_two_vectors(-q_k, _E1))
            adr = int(self._link_qposadr[k])
            data.qpos[adr : adr + 4] = self._rot2quat(
                self._link_body_quat[k].T @ parent_frames[k].T @ frame
            )

            omega_k = np.asarray(rates[k], dtype=float)
            # Only the component tangent to q_k moves the link; a spin about
            # q_k is the gauge freedom the analytical model factors out.
            omega_k = omega_k - float(np.dot(omega_k, q_k)) * q_k
            adr = int(self._link_qveladr[k])
            data.qvel[adr : adr + 3] = frame.T @ (omega_k - parent_omegas[k])

            parent_frames.append(frame)
            parent_omegas.append(omega_k)

        orientations = kwargs.get("orientations", [np.eye(3)] * self._num_quadrotors)
        orientations = self._check_per_index(orientations, "orientations", self._num_quadrotors)
        body_rates = kwargs.get("angular_velocities", [np.zeros(3)] * self._num_quadrotors)
        body_rates = self._check_per_index(body_rates, "angular_velocities", self._num_quadrotors)
        for k, vertex in enumerate(self._attachments):
            R_k = np.asarray(orientations[k], dtype=float)
            adr = int(self._quad_qposadr[k])
            data.qpos[adr : adr + 4] = self._rot2quat(
                self._quad_body_quat[k].T @ parent_frames[vertex].T @ R_k
            )
            # Omega is the *body*-frame rate (the model's convention and
            # MuJoCo's); the chain works in world coordinates, so rotate it out
            # before differencing against the parent.
            omega_world = R_k @ np.asarray(body_rates[k], dtype=float)
            adr = int(self._quad_qveladr[k])
            data.qvel[adr : adr + 3] = R_k.T @ (omega_world - parent_omegas[vertex])

        mujoco.mj_forward(model, data)
        self._query_latest_state()
        self._station_targets = [p.copy() for p in self.state.positions]

    @staticmethod
    def _rot2quat(rot: np.ndarray) -> np.ndarray:
        """3x3 rotation -> MuJoCo's ``(w, x, y, z)`` quaternion."""
        from scipy.spatial.transform import Rotation as sp_rot

        xyzw = sp_rot.from_matrix(np.asarray(rot, dtype=float)).as_quat()
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    @staticmethod
    def _quat2rot(quat) -> np.ndarray:
        """MuJoCo's ``(w, x, y, z)`` quaternion -> 3x3 rotation."""
        from scipy.spatial.transform import Rotation as sp_rot

        w, x, y, z = (float(v) for v in quat)
        return np.asarray(sp_rot.from_quat([x, y, z, w]).as_matrix())

    # ─── Step ──────────────────────────────────────────────────────

    def step(self, u, desired_att=None):
        """Step MuJoCo simulation. ``u`` follows the base input pipeline."""
        for _ in range(self._step_iter):
            wrench = self._repackage_input(u, desired_att=desired_att)
            self._mjMdl.data.ctrl[:] = wrench.reshape(-1)
            self._mjMdl._step_mujoco_simulation(1)
            self._query_latest_state()

        if self.render and self._mjMdl._viewer is not None:
            for k in range(self._num_quadrotors):
                self._mjMdl._viewer.add_trail_point(self.state.positions[k], key=k)

    # ─── State query ───────────────────────────────────────────────

    def _body_world_velocity(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        """``(angular, linear)`` world-frame velocity of a body's own frame."""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(
            self._mjMdl.model,
            self._mjMdl.data,
            int(mujoco.mjtObj.mjOBJ_BODY),
            int(body_id),
            res,
            0,  # flg_local = 0 -> world-aligned frame
        )
        return res[0:3].copy(), res[3:6].copy()

    def _query_latest_state(self):
        """Sync state from MuJoCo data — every link read, nothing differenced."""
        data = self._mjMdl.data
        self.t = data.time

        adr = self._root_qposadr
        self.state.root_position = copy.deepcopy(data.qpos[adr : adr + 3])
        adr = self._root_qveladr
        self.state.root_velocity = copy.deepcopy(data.qvel[adr : adr + 3])

        for k in range(self._num_links):
            body_id = int(self._link_body_id[k])
            # q_k is exact, not differenced: the link's body z-axis points from
            # its distal end back to its proximal joint.
            frame = np.asarray(data.xmat[body_id]).reshape(3, 3)
            q_k = -frame[:, 2]
            omega_world, _ = self._body_world_velocity(body_id)
            # Project out the spin about q_k: it is the S² gauge freedom and
            # contributes nothing to q̇_k.
            omega_k = omega_world - float(np.dot(omega_world, q_k)) * q_k
            self.state.link_attitudes[k] = S2(q_k)
            self.state.link_angular_velocities[k] = TS2(omega_k)

        for k in range(self._num_quadrotors):
            body_id = int(self._quad_body_id[k])
            R_k = np.asarray(data.xmat[body_id]).reshape(3, 3)
            self.state.orientations[k] = SO3(R_k.copy())
            omega_world, _ = self._body_world_velocity(body_id)
            # Omega lives in the body frame; the query above is world-aligned.
            # Rotating rather than asking for the local frame is deliberate:
            # mj_objectVelocity's local frame is the body's *inertial* frame,
            # which need not share axes with the body frame the model uses.
            self.state.angular_velocities[k] = TSO3(R_k.T @ omega_world)
            self.state.positions[k] = copy.deepcopy(np.asarray(data.xpos[body_id]))
            _, linear = self._body_world_velocity(body_id)
            self.state.velocities[k] = linear

    # ─── Simulate ──────────────────────────────────────────────────

    def simulate(self, tf, **kwargs):
        super().simulate(tf, **kwargs)
        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl.wait_for_close()
