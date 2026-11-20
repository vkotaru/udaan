"""TetheredHoseMujoco — MuJoCo backend for quadrotors on a hose pinned to a ground anchor."""

from __future__ import annotations

import copy
import hashlib
import json

import mujoco
import numpy as np

from ...core.defaults import DEFAULT_TETHER_ANCHOR_HEIGHT
from ...manif import S2, SO3, TS2, TSO3
from ...utils.assets.cache import ensure_generated
from ...utils.assets.mujoco_asset_creator import MJCF_GENERATOR_VERSION
from ...utils.logging import get_logger
from .._kwargs import check_reset_kwargs
from ..multi_quad_hose.mujoco import (
    _LINK_BODY,
    _LINK_JOINT,
    _QUAD_BODY,
    _ROOT_BODY,
    _VERTEX_GEOM,
    MultiQuadHoseMujoco,
)
from .base import _SUPPORTED_RESET_KWARGS, TetheredHoseBase, TetheredQuadrotorBase

_logger = get_logger(__name__)

__all__ = ["TetheredHoseMujoco", "TetheredQuadrotorMujoco"]

_E1 = np.array([1.0, 0.0, 0.0])


class TetheredHoseMujoco(MultiQuadHoseMujoco, TetheredHoseBase):
    """Quadrotors on an ``n``-link hose pinned to a ground anchor, in MuJoCo (M8).

    M6's MuJoCo backend on the MJCF of
    :func:`~udaan.utils.assets.xml_model_generator.tethered_hose`: the root
    body ``hose_vertex0`` has no joint, so MuJoCo welds it to the world at the
    anchor.  That is the pin, exact by construction.  Stepping, the parameter
    setters, the divergence check and the measured (not differenced) link and
    airframe states are M6's; this class drops the root joint from the index
    look-up, the reset and the state query, and takes the analytical model's
    defaults, controller and :meth:`simulate`.

    The MJCF spawns the hose standing straight up; the constructor and
    :meth:`reset` put it in the reference at ``t = 0`` (by default the static
    trim at the controller's setpoint), as the analytical model does.

    The two backends differ by the vertex spheres' own rotational inertia,
    ``2/5 m r^2`` each, which the analytical model's point masses do not have
    (M6's residual).  Contacts are off, so a hose below the anchor's height
    passes through the floor.
    """

    def __init__(self, **kwargs):
        if "spawn_height" in kwargs:
            raise ValueError(
                f"{type(self).__name__} takes anchor_height, not spawn_height: vertex 0 is the "
                "anchor"
            )
        kwargs = self._apply_defaults(kwargs)
        kwargs["spawn_height"] = float(kwargs.get("anchor_height", DEFAULT_TETHER_ANCHOR_HEIGHT))
        super().__init__(**kwargs)
        self.reset()

    # ─── MJCF generation ───────────────────────────────────────────

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
        """The MJCF path for this configuration; ``spawn_height`` is the anchor height.

        ``sag_ratio`` does not shape a tethered hose's file and is not hashed.
        """
        config = {
            "v": MJCF_GENERATOR_VERSION,
            "n": int(num_links),
            "a": [int(i) for i in attachments],
            "L": float(hose_length),
            "mh": float(hose_mass),
            "mQ": float(quad_mass),
            "z": float(spawn_height),
        }
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        generated_name = f"tethered_hose_{num_links}l_{len(attachments)}q_{config_hash}.xml"

        def build(path: str) -> None:
            from ...utils.assets import xml_model_generator

            _logger.info("Generating MJCF for a tethered hose: %s", generated_name)
            xml_model_generator.tethered_hose(
                num_links=num_links,
                attachments=attachments,
                filename=path,
                hose_length=hose_length,
                hose_mass=hose_mass,
                quad_mass=quad_mass,
                anchor_height=spawn_height,
                verbose=False,
            )

        return ensure_generated(generated_name, build)

    def _resolve_mj_indices(self):
        """M6's look-up without the root joint; the root body must have none."""
        model = self._mjMdl.model
        n = self._num_links
        nQ = self._num_quadrotors

        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _ROOT_BODY)
        if root_id < 0:
            raise RuntimeError(f"MJCF has no body named {_ROOT_BODY!r}")
        if int(model.body_jntnum[root_id]) != 0 or int(model.body_parentid[root_id]) != 0:
            raise RuntimeError(
                f"{_ROOT_BODY!r} must be a jointless child of the world (the anchor); it has "
                f"{int(model.body_jntnum[root_id])} joint(s)"
            )
        self._root_body_id = root_id

        self._link_body_id = np.zeros(n, dtype=int)
        self._link_jnt_id = np.zeros(n, dtype=int)
        self._link_qposadr = np.zeros(n, dtype=int)
        self._link_qveladr = np.zeros(n, dtype=int)
        for i in range(n):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _LINK_BODY.format(i))
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _LINK_JOINT.format(i))
            if body_id < 0 or jnt_id < 0:
                raise RuntimeError(f"MJCF lacks link {i} ({_LINK_BODY.format(i)!r} and its joint)")
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

        expected = [int(self._link_body_id[v - 1]) for v in self._attachments]
        actual = [int(model.body_parentid[i]) for i in self._quad_body_id]
        if actual != expected:
            raise RuntimeError(
                f"the MJCF bolts the quadrotors to bodies {actual}, but attachments "
                f"{self._attachments} put them on {expected}"
            )
        if model.nu != 4 * nQ:
            raise RuntimeError(f"MJCF exposes {model.nu} actuators, expected {4 * nQ}")

        self._vertex_radius = np.zeros(n + 1)
        for v in range(n + 1):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _VERTEX_GEOM.format(v))
            if geom_id < 0:
                raise RuntimeError(f"MJCF has no geom named {_VERTEX_GEOM.format(v)!r}")
            self._vertex_radius[v] = float(model.geom_size[geom_id, 0])

        self._link_body_quat = np.stack(
            [self._quat2rot(model.body_quat[int(i)]) for i in self._link_body_id]
        )
        self._quad_body_quat = np.stack(
            [self._quat2rot(model.body_quat[int(i)]) for i in self._quad_body_id]
        )

    # ─── Reset ─────────────────────────────────────────────────────

    def reset(self, **kwargs):
        """Reset MuJoCo; by default to the reference at ``t = 0``.

        The analytical model's keywords and defaults
        (:meth:`TetheredHoseBase.reset`), written into the ball joints by M6's
        chain walk from the world-aligned anchor body.

        Raises:
            ValueError: for any other keyword, ``root_position`` and
                ``root_velocity`` included.
        """
        check_reset_kwargs(self, kwargs, _SUPPORTED_RESET_KWARGS)
        ref, _ = self._reset_reference()
        n = self._num_links
        nQ = self._num_quadrotors
        directions = self._check_per_index(
            kwargs.get("link_attitudes", ref.link_attitudes), "link_attitudes", n
        )
        rates = self._check_per_index(
            kwargs.get("link_angular_velocities", ref.link_angular_velocities),
            "link_angular_velocities",
            n,
        )
        orientations = self._check_per_index(
            kwargs.get("orientations", ref.orientations), "orientations", nQ
        )
        body_rates = self._check_per_index(
            kwargs.get("angular_velocities", ref.angular_velocities), "angular_velocities", nQ
        )

        self.t = 0.0
        self._substeps_taken = 0
        self._mjMdl.reset()
        self._mjMdl.data.time = self.t
        self.state.reset()
        data = self._mjMdl.data
        model = self._mjMdl.model

        # A ball joint's qpos is the child's rotation relative to its parent and
        # its qvel the relative angular velocity in the child's frame; frames
        # start world-aligned at the welded anchor body.
        parent_frames = [np.eye(3)]
        parent_omegas = [np.zeros(3)]
        for k in range(n):
            q_k = np.asarray(directions[k], dtype=float)
            q_k = q_k / np.linalg.norm(q_k)
            frame = np.asarray(SO3.from_two_vectors(-q_k, _E1))
            adr = int(self._link_qposadr[k])
            data.qpos[adr : adr + 4] = self._rot2quat(
                self._link_body_quat[k].T @ parent_frames[k].T @ frame
            )
            omega_k = np.asarray(rates[k], dtype=float)
            omega_k = omega_k - float(np.dot(omega_k, q_k)) * q_k
            adr = int(self._link_qveladr[k])
            data.qvel[adr : adr + 3] = frame.T @ (omega_k - parent_omegas[k])
            parent_frames.append(frame)
            parent_omegas.append(omega_k)

        for k, vertex in enumerate(self._attachments):
            R_k = np.asarray(orientations[k], dtype=float)
            adr = int(self._quad_qposadr[k])
            data.qpos[adr : adr + 4] = self._rot2quat(
                self._quad_body_quat[k].T @ parent_frames[vertex].T @ R_k
            )
            omega_world = R_k @ np.asarray(body_rates[k], dtype=float)
            adr = int(self._quad_qveladr[k])
            data.qvel[adr : adr + 3] = R_k.T @ (omega_world - parent_omegas[vertex])

        mujoco.mj_forward(model, data)
        self._query_latest_state()
        self._station_targets = [p.copy() for p in self.state.positions]

    # ─── State query ───────────────────────────────────────────────

    def _query_latest_state(self):
        """M6's measured state query; the root is the anchor, at rest."""
        model, data = self._mjMdl.model, self._mjMdl.data
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)

        self.state.root_position = self._anchor.copy()
        self.state.root_velocity = np.zeros(3)
        for k in range(self._num_links):
            body_id = int(self._link_body_id[k])
            frame = np.asarray(data.xmat[body_id]).reshape(3, 3)
            q_k = -frame[:, 2]
            omega_world, _ = self._body_world_velocity(body_id)
            omega_k = omega_world - float(np.dot(omega_world, q_k)) * q_k
            self.state.link_attitudes[k] = S2(q_k)
            self.state.link_angular_velocities[k] = TS2(omega_k)

        for k in range(self._num_quadrotors):
            body_id = int(self._quad_body_id[k])
            R_k = np.asarray(data.xmat[body_id]).reshape(3, 3)
            self.state.orientations[k] = SO3(R_k.copy())
            omega_world, linear = self._body_world_velocity(body_id)
            self.state.angular_velocities[k] = TSO3(R_k.T @ omega_world)
            self.state.positions[k] = copy.deepcopy(np.asarray(data.xpos[body_id]))
            self.state.velocities[k] = linear


class TetheredQuadrotorMujoco(TetheredHoseMujoco, TetheredQuadrotorBase):
    """One quadrotor on an ``n``-link tether pinned to an anchor, in MuJoCo (M7)."""
