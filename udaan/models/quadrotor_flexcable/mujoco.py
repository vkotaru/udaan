"""QuadrotorFlexCableMujoco — MuJoCo physics backend for the M5 flexible-cable system."""

from __future__ import annotations

import copy
import hashlib
import json

import mujoco
import numpy as np

from ...core.defaults import (
    DEFAULT_CABLE_MASS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_SPAWN_HEIGHT,
)
from ...manif import S2, SO3, TS2, TSO3
from ...utils.assets.cache import ensure_generated
from ...utils.assets.mujoco_asset_creator import MJCF_GENERATOR_VERSION
from ...utils.logging import get_logger
from ..backend import MujocoModel
from .base import QuadrotorFlexCableBase

_logger = get_logger(__name__)


#: Body/joint name templates emitted by :func:`udaan.utils.assets.quad_flexcable`.
_QUAD_BODY = "quad"
_LINK_BODY = "cable_link{}"
_LINK_JOINT = "cable_joint{}"
_PAYLOAD_BODY = "cable_payload"

#: Link lengths read out of the MJCF must match the declared ones to this
#: absolute tolerance, in metres — round-off, because they are read from the
#: **joint anchors**, which ``MujocoAssetCreator.joint`` writes with ``str()``
#: and therefore at full double precision.
#:
#: The cylinder geoms are *not* the place to read them:
#: ``MujocoAssetCreator.cylinder`` formats ``size`` with ``%g``, six
#: significant digits, so a three-link cable's 1/3 m links come back as
#: 0.333334 — 2e-6 m per link, six orders of magnitude past this tolerance.
#: What that would break is *this check*, not the simulation: body and joint
#: ``pos`` are both written with ``str()``, so the chain is placed and spawned
#: at full precision and the payload lands exactly where it should.  Reading
#: lengths off the geoms would therefore raise a spurious ``RuntimeError`` on a
#: perfectly good model (and, if the tolerance were loosened to swallow it,
#: would quietly feed the analytical model link lengths 2e-6 m off the ones
#: MuJoCo is actually integrating).
_GEOMETRY_TOL = 1e-12

#: The reset overrides this backend can apply. ``payload_velocity`` is
#: deliberately absent — see :meth:`QuadrotorFlexCableMujoco.reset`.
_SUPPORTED_RESET_KWARGS = frozenset(
    {
        "position",
        "payload_position",
        "velocity",
        "orientation",
        "angular_velocity",
        "link_attitudes",
        "link_angular_velocities",
    }
)


class QuadrotorFlexCableMujoco(QuadrotorFlexCableBase):
    """One quadrotor, an ``n``-link cable and a payload, in MuJoCo.

    Overrides ``step``/``reset``/``_query_latest_state`` to delegate to MuJoCo;
    the input pipeline, controllers and parameter bookkeeping are inherited.

    The MJCF is generated on demand from the construction parameters
    (``num_links``, ``cable_length``, ``cable_mass``, ``payload_mass``,
    ``mass``, ``spawn_height``) and cached under a filename carrying a hash of
    them, so a changed parameter never reuses a stale asset.  There is no
    tendon in it: the cable is a kinematic tree of ball joints, which is why —
    unlike :class:`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyMujoco`
    — this backend can honour arbitrary initial link directions and rates.

    **The per-link state is measured, not reconstructed.**  This is the whole
    point of promoting M5 out of
    :class:`~udaan.models.quadrotor_cspayload.QuadrotorCsPayloadMujoco`: that
    class loads the very same kind of N-link chain under
    ``cable_model="links"`` and then collapses it to a single point-mass cable
    — one direction from the airframe to the payload, and a payload velocity
    obtained by *finite-differencing consecutive positions*, which is first
    order in the timestep and pure noise at spawn.  Here every link's direction
    comes from its body frame (``q_i = -R_i e_3``, exact) and every link's rate
    from ``mj_objectVelocity`` (exact), so the state that comes back is the
    state the analytical model integrates.

    Where the two backends genuinely differ, and it is not integration:

    * **Mass distribution.**  MuJoCo's links are cylinders with their mass
      spread along their length and a rotational inertia of their own; the
      analytical model lumps a point mass at each joint.
      :meth:`_read_params_from_mjcf` reads each cylinder's mass onto that
      link's *distal* end, which keeps the total mass exact and the map
      between MJCF bodies and model parameters one-to-one, but is only first
      order in ``l_i`` (a consistent half-lumping would be second order).  At
      the default 0.1 kg cable over three links that shifts ~17 g per joint
      against a 0.2 kg payload; it is a *physics*
      difference between the backends, not an integration one, and it is why a
      base-vs-MuJoCo comparison here is a qualitative agreement rather than the
      near-exact one M4's tendons allow.

      The half-lumping was measured and **deliberately refused**.  Splitting
      each cylinder's mass ``mu_i/2`` onto both of its ends, with link 0's
      proximal half landing on the airframe, cuts the ``n = 1`` peak airframe
      gap over the 1 s free swing by 2.6x — 1.69e-2 m to 6.47e-3 m.  It is not
      taken because distal lumping is what makes the ``n = 1`` case reduce to
      M2 *exactly*: half-lumping moves mass onto the airframe, which M2 has no
      term for, and the reduction — the strongest validation this model has
      (``TestDynamics::test_reduction_to_m2_at_one_link``) — would
      become approximate.  A 2.6x tightening of one backend-agreement number is
      not worth downgrading the one check that compares against independently
      verified code.

    ``mass`` and ``inertia`` are read back from the file and their setters
    write into it, because MuJoCo is the authority on them: a body's mass is
    the sum of its geoms', and the retired ``mQ = 0.75`` literal was the mass
    of the airframe's centre box alone.  ``link_masses`` and ``link_lengths``
    are read back too, but assigning them afterwards changes only the
    *analytical* parameters — the cable's mass and geometry live in cylinder
    geoms with no one-to-one map to a lumped joint mass, so regenerate the
    model with a different ``cable_mass``/``cable_length`` instead.
    """

    def __init__(self, **kwargs):
        self.render = kwargs.get("render", False)
        self._spawn_height = float(kwargs.pop("spawn_height", DEFAULT_SPAWN_HEIGHT))
        if "link_masses" in kwargs or "link_lengths" in kwargs:
            raise ValueError(
                "the MuJoCo backend derives link masses and lengths from the generated "
                "MJCF, so `link_masses`/`link_lengths` constructor arguments would be "
                "silently overwritten; pass cable_length/cable_mass/payload_mass "
                "instead, or assign `mdl.link_masses` after construction"
            )
        self._cable_mass = float(kwargs.get("cable_mass", DEFAULT_CABLE_MASS))
        self._payload_mass = float(kwargs.get("payload_mass", DEFAULT_PAYLOAD_MASS))

        super().__init__(**kwargs)

        xml_path = self._ensure_xml(
            self._num_links,
            self.cable_length,
            self._cable_mass,
            self._payload_mass,
            self._mass,
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

        if self.verbose:
            _logger.info(
                "MuJoCo quadrotor flexible-cable payload loaded (%d links)", self._num_links
            )

    # ─── MJCF generation ───────────────────────────────────────────

    @staticmethod
    def _ensure_xml(
        num_links: int,
        cable_length: float,
        cable_mass: float,
        payload_mass: float,
        quad_mass: float,
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
            "L": float(cable_length),
            "mc": float(cable_mass),
            "mL": float(payload_mass),
            "mQ": float(quad_mass),
            "z": float(spawn_height),
        }
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        generated_name = f"quad_flexcable_{num_links}l_{config_hash}.xml"

        def build(path: str) -> None:
            from ...utils.assets import xml_model_generator

            _logger.info("Generating MJCF for a %d-link cable: %s", num_links, generated_name)
            xml_model_generator.quad_flexcable(
                num_links=num_links,
                filename=path,
                cable_length=cable_length,
                cable_mass=cable_mass,
                payload_mass=payload_mass,
                quad_mass=quad_mass,
                spawn_height=spawn_height,
                verbose=False,
            )

        # Generated assets go to the user cache, not into the installed
        # package, and the write is atomic — see
        # :func:`udaan.utils.assets.cache.ensure_generated`.
        return ensure_generated(generated_name, build)

    # ─── Model initialization ──────────────────────────────────────

    def _resolve_mj_indices(self):
        """Look up body/joint ids by name and check the chain is the declared one."""
        model = self._mjMdl.model
        n = self._num_links

        quad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _QUAD_BODY)
        if quad_id < 0:
            raise RuntimeError(f"MJCF has no body named {_QUAD_BODY!r}")
        self._quad_body_id = quad_id
        quad_jnt = model.body_jntadr[quad_id]
        self._quad_qposadr = int(model.jnt_qposadr[quad_jnt])
        self._quad_qveladr = int(model.jnt_dofadr[quad_jnt])

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
            raise RuntimeError(f"MJCF defines more than the expected {n} cable links")

        payload_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _PAYLOAD_BODY)
        if payload_id < 0:
            raise RuntimeError(f"MJCF has no body named {_PAYLOAD_BODY!r}")
        self._payload_body_id = payload_id

        if model.nu != 4:
            raise RuntimeError(f"MJCF exposes {model.nu} actuators, expected 4")

    def _read_params_from_mjcf(self):
        """Adopt the file's masses, inertia and link lengths.

        The link lengths come from the cylinder geoms' half-lengths and are
        *checked* against the declared ones — a mismatch means the analytical
        model and the physics are describing different cables.  Each cylinder's
        mass is then read onto that link's **distal** end, and the payload
        body's mass is added to the last one, so the total is exact and every
        model parameter maps to exactly one MJCF body.  See the class docstring
        for what that costs relative to a consistent half-lumping.
        """
        model = self._mjMdl.model
        n = self._num_links

        lengths = np.zeros(n)
        cyl_masses = np.zeros(n)
        for i in range(n):
            body_id = int(self._link_body_id[i])
            if int(model.body_geomnum[body_id]) != 1:
                raise RuntimeError(
                    f"link body {i} carries {int(model.body_geomnum[body_id])} geoms, expected 1"
                )
            # The ball joint sits half a link above the body's own origin, so
            # the link's length is twice that offset. Read from the joint, not
            # the geom — see _GEOMETRY_TOL.
            lengths[i] = 2.0 * abs(float(model.jnt_pos[int(self._link_jnt_id[i]), 2]))
            cyl_masses[i] = float(model.body_mass[body_id])

        if not np.allclose(lengths, self._link_lengths, rtol=0.0, atol=_GEOMETRY_TOL):
            raise RuntimeError(
                "link_lengths disagree with the MJCF's joint anchors:\n"
                f"  model: {self._link_lengths.tolist()}\n"
                f"  MJCF:  {lengths.tolist()}"
            )
        self.link_lengths = lengths

        lumped = cyl_masses.copy()
        lumped[-1] += float(model.body_mass[self._payload_body_id])
        self.link_masses = lumped

        self.mass = float(model.body_mass[self._quad_body_id])
        # body_inertia holds principal-axis diagonals; the model wants a tensor.
        self.inertia = np.diag(model.body_inertia[self._quad_body_id])

    # ─── Generated geometry ────────────────────────────────────────

    @property
    def spawn_height(self) -> float:
        """Height the MJCF spawns the airframe at, in metres."""
        return self._spawn_height

    # ─── Property setters (sync to MuJoCo) ─────────────────────────
    #
    # The `@Base.prop.setter` override pattern is standard at runtime but mypy
    # cannot model it, hence the attr-defined ignores below.

    @QuadrotorFlexCableBase.mass.setter  # type: ignore[attr-defined]
    def mass(self, value):
        QuadrotorFlexCableBase.mass.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            self._mjMdl.model.body_mass[self._quad_body_id] = self._mass

    @QuadrotorFlexCableBase.inertia.setter  # type: ignore[attr-defined]
    def inertia(self, value):
        QuadrotorFlexCableBase.inertia.fset(self, value)  # type: ignore[attr-defined]
        if getattr(self, "_mjMdl", None) is not None:
            self._mjMdl.model.body_inertia[self._quad_body_id] = np.diag(self._inertia)

    # `link_masses` and `link_lengths` deliberately keep the base setters: they
    # change the analytical parameters only. Pushing a lumped joint mass or a
    # link length back into the file would mean re-deriving the cylinder geoms
    # (and, for a length change, the whole body tree), which is the generator's
    # job — regenerate with a different cable_mass/cable_length instead.

    # ─── Reset ─────────────────────────────────────────────────────

    def reset(self, **kwargs):
        """Reset MuJoCo to a requested configuration.

        Every kwarg the analytical base accepts is honoured except
        ``payload_velocity``: the airframe's free joint is the root of the tree
        here, so the payload's velocity is a *derived* quantity and setting it
        would mean solving the chain's rates backwards — an inverse problem
        with an ``n``-dimensional family of solutions.  Pass ``velocity``
        (the airframe's) and ``link_angular_velocities`` instead.

        ``payload_position`` is honoured as a rigid translation: the whole
        model, airframe included, moves so that the payload lands there — which
        matches how M3/M4 read the same argument.

        Raises:
            ValueError: for any unsupported keyword.
        """
        unsupported = sorted(k for k in kwargs if k not in _SUPPORTED_RESET_KWARGS)
        if unsupported:
            raise ValueError(
                f"{type(self).__name__}.reset() cannot honour {unsupported}: the "
                "airframe's free joint is the root of the MuJoCo tree, so the payload "
                "velocity is derived from the chain rather than set. Supported: "
                f"{', '.join(sorted(_SUPPORTED_RESET_KWARGS))}."
            )

        self.t = 0.0
        self._mjMdl.reset()
        self.state.reset()
        data = self._mjMdl.data
        model = self._mjMdl.model

        # Airframe pose and twist.
        R_quad = SO3(np.asarray(kwargs["orientation"])) if "orientation" in kwargs else SO3()
        i = self._quad_qposadr
        if "position" in kwargs:
            data.qpos[i : i + 3] = np.asarray(kwargs["position"], dtype=float)
        data.qpos[i + 3 : i + 7] = self._rot2quat(np.asarray(R_quad))
        i = self._quad_qveladr
        if "velocity" in kwargs:
            data.qvel[i : i + 3] = np.asarray(kwargs["velocity"], dtype=float)
        if "angular_velocity" in kwargs:
            data.qvel[i + 3 : i + 6] = np.asarray(kwargs["angular_velocity"], dtype=float)

        # Link directions and rates, walked down the chain: a ball joint's qpos
        # is the child's rotation relative to its parent, and its qvel is the
        # *relative* angular velocity expressed in the child's frame.
        #
        # Off-manifold input is repaired exactly as the base ``reset()`` repairs
        # it — ``q_i`` normalised, ``omega_i`` projected onto the tangent plane
        # — so the two backends land on the same state from the same kwargs.
        # MuJoCo has no choice about it (a ball joint cannot represent a
        # stretched link, and a spin about the link axis is unobservable), which
        # is why the base was made to match this rather than the other way
        # round; see :meth:`QuadrotorFlexCableBase.reset`.
        directions = kwargs.get("link_attitudes", [np.array([0.0, 0.0, -1.0])] * self._num_links)
        rates = kwargs.get("link_angular_velocities", [np.zeros(3)] * self._num_links)
        directions = self._check_per_link(directions, "link_attitudes")
        rates = self._check_per_link(rates, "link_angular_velocities")

        R_parent = np.asarray(R_quad)
        # Ω is the airframe's *body*-frame rate (MuJoCo's free-joint convention
        # and the model's); the chain below works in world coordinates, so it
        # has to be rotated out. Skipping this rotation leaves every link's
        # recovered ω wrong by an O(Ω) amount that survives mj_forward.
        omega_parent = (
            R_parent @ np.asarray(kwargs["angular_velocity"], dtype=float)
            if "angular_velocity" in kwargs
            else np.zeros(3)
        )
        for k in range(self._num_links):
            q_k = np.asarray(directions[k], dtype=float)
            q_k = q_k / np.linalg.norm(q_k)
            # The link's body z-axis points from its distal end back to its
            # proximal joint, so its third column is -q_k.
            R_link = np.asarray(SO3.from_two_vectors(-q_k, np.array([1.0, 0.0, 0.0])))
            adr = int(self._link_qposadr[k])
            data.qpos[adr : adr + 4] = self._rot2quat(R_parent.T @ R_link)

            omega_k = np.asarray(rates[k], dtype=float)
            # Only the component tangent to q_k moves the link; a spin about
            # q_k is the gauge freedom the analytical model factors out.
            omega_k = omega_k - float(np.dot(omega_k, q_k)) * q_k
            adr = int(self._link_qveladr[k])
            data.qvel[adr : adr + 3] = R_link.T @ (omega_k - omega_parent)

            R_parent = R_link
            omega_parent = omega_k

        mujoco.mj_forward(model, data)

        if "payload_position" in kwargs:
            target = np.asarray(kwargs["payload_position"], dtype=float)
            offset = target - np.asarray(data.xpos[self._payload_body_id])
            i = self._quad_qposadr
            data.qpos[i : i + 3] += offset
            mujoco.mj_forward(model, data)

        self._query_latest_state()

    @staticmethod
    def _rot2quat(rot: np.ndarray) -> np.ndarray:
        """3x3 rotation -> MuJoCo's ``(w, x, y, z)`` quaternion."""
        from scipy.spatial.transform import Rotation as sp_rot

        xyzw = sp_rot.from_matrix(np.asarray(rot, dtype=float)).as_quat()
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    # ─── Step ──────────────────────────────────────────────────────

    def step(self, u, desired_att=None):
        """Step MuJoCo simulation. ``u`` follows the base input pipeline."""
        for _ in range(self._step_iter):
            wrench = self._repackage_input(u, desired_att=desired_att)
            self._mjMdl.data.ctrl[:] = wrench
            self._mjMdl._step_mujoco_simulation(1)
            self._query_latest_state()

        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl._viewer.add_trail_point(self.state.position)
            self._mjMdl._viewer.add_trail_point(
                self.state.payload_position, key=1, rgba=[0.2, 0.2, 1.0, 0.6]
            )

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

        i = self._quad_qposadr
        self.state.position = copy.deepcopy(data.qpos[i : i + 3])
        self.state.orientation = SO3(self._mjMdl._quat2rot(data.qpos[i + 3 : i + 7]))
        i = self._quad_qveladr
        self.state.velocity = copy.deepcopy(data.qvel[i : i + 3])
        # A MuJoCo free joint stores its angular velocity in the body frame,
        # which is the frame Ω lives in here.
        self.state.angular_velocity = TSO3(copy.deepcopy(data.qvel[i + 3 : i + 6]))

        for k in range(self._num_links):
            body_id = int(self._link_body_id[k])
            # q_k is exact, not differenced: the link's body z-axis points from
            # its distal end back to its proximal joint.
            R_link = np.asarray(data.xmat[body_id]).reshape(3, 3)
            q_k = -R_link[:, 2]
            omega_world, _ = self._body_world_velocity(body_id)
            # Project out the spin about q_k: it is the S² gauge freedom and
            # contributes nothing to q̇_k.
            omega_k = omega_world - float(np.dot(omega_world, q_k)) * q_k
            self.state.link_attitudes[k] = S2(q_k)
            self.state.link_angular_velocities[k] = TS2(omega_k)

        self.state.payload_position = copy.deepcopy(np.asarray(data.xpos[self._payload_body_id]))
        _, payload_vel = self._body_world_velocity(self._payload_body_id)
        self.state.payload_velocity = payload_vel

    # ─── Simulate ──────────────────────────────────────────────────

    def simulate(self, tf, **kwargs):
        super().simulate(tf, **kwargs)
        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl.wait_for_close()
