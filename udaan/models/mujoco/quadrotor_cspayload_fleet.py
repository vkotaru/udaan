"""N-quadrotor-cable-payload fleet for comparing controllers/gains side-by-side.

Each agent is an independent quadrotor with an N-link cable-suspended point-mass
payload. All agents share one MuJoCo scene but have no inter-agent contacts.
State, controllers, and labels are per-agent; one unified `ctrl` buffer is
written once per step.
"""

import copy
import hashlib
import json
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sp_rot

from ... import _FOLDER_PATH
from ...manif import S2, SO3, TS2, TSO3
from ...utils.logging import get_logger
from .. import base
from ..quadrotor_cspayload import QuadrotorCsPayloadBase
from . import MujocoModel

_logger = get_logger(__name__)

_MJCF_DIR = os.path.join(_FOLDER_PATH, "udaan", "models", "assets", "mjcf")

# X-spacing between adjacent agents, used both at XML-generation time
# (per-agent spawn offset) and at runtime (per-agent lane offsets).
_AGENT_SPACING = 2.0


def _agent_offsets(nQ):
    return np.linspace(-(nQ - 1) * _AGENT_SPACING / 2, (nQ - 1) * _AGENT_SPACING / 2, nQ)


# Distinct colors for up to 8 agents (same palette as QuadrotorFleet)
_FLEET_COLORS = [
    ([0.90, 0.20, 0.20], "red"),
    ([0.20, 0.50, 0.90], "blue"),
    ([0.20, 0.80, 0.40], "green"),
    ([0.90, 0.55, 0.10], "orange"),
    ([0.65, 0.30, 0.85], "purple"),
    ([0.10, 0.80, 0.80], "cyan"),
    ([0.85, 0.20, 0.60], "pink"),
    ([0.60, 0.60, 0.20], "olive"),
]


class QuadrotorCsPayloadFleet(base.BaseModel):
    """Multiple independent quadrotor-cable-payload agents in one MuJoCo scene.

    Each agent keeps its own `QuadrotorCsPayloadBase` for controller/state
    bookkeeping; the fleet owns a single MuJoCo model containing N quads +
    N cables + N payloads.

    Usage::

        fleet = QuadrotorCsPayloadFleet(num_agents=2, render=True)
        fleet[0]._payload_controller.setpoint = lambda t: (target, 0, 0)
        fleet[1]._payload_controller.setpoint = lambda t: (target, 0, 0)
        fleet.simulate(tf=8.0, payload_positions=[[1,1,0.5], [-1,-1,0.5]])
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nQ = kwargs.get("num_agents", 2)
        self.render = kwargs.get("render", False)
        self._labels = kwargs.get("labels", [f"agent{i}" for i in range(self.nQ)])
        self._cable_links = kwargs.get("cable_links", 10)
        self._cable_length = kwargs.get("cable_length", 1.0)
        self._payload_mass = kwargs.get("payload_mass", 0.15)

        # Per-agent pure-Python bookkeeping (state + controllers)
        self.agents = [QuadrotorCsPayloadBase() for _ in range(self.nQ)]

        xml_path = self._generate_xml(
            self.nQ, self._cable_links, self._cable_length, self._payload_mass
        )
        self._mjMdl = MujocoModel(model_path=xml_path, render=self.render)

        self._mjDt = 1.0 / 500.0
        self._step_iter = int(self.sim_timestep / self._mjDt)
        self._nFrames = 1

        self._offsets = _agent_offsets(self.nQ)

        self._resolve_mj_indices()

    def __getitem__(self, i):
        return self.agents[i]

    # ─── XML generation ───────────────────────────────────────────────

    @staticmethod
    def _generate_xml(nQ, cable_links, cable_length, payload_mass):
        from ...utils.assets.mujoco_asset_creator import MujocoAssetCreator

        config_key = json.dumps(
            {"nQ": nQ, "N": cable_links, "L": cable_length, "m": payload_mass},
            sort_keys=True,
        )
        config_hash = hashlib.md5(config_key.encode()).hexdigest()[:8]
        filename = f"cspayload_fleet_{nQ}q_{config_hash}.xml"
        filepath = os.path.join(_MJCF_DIR, filename)
        if os.path.exists(filepath):
            return filename

        _logger.info("Generating cspayload fleet MJCF for %d agents", nQ)
        writer = MujocoAssetCreator(f"CsPayloadFleet{nQ}")

        offsets = _agent_offsets(nQ)

        for i in range(nQ):
            color, _ = _FLEET_COLORS[i % len(_FLEET_COLORS)]
            agent_name = f"agent{i}"
            chassis, _ = writer.create_quadrotor0(
                writer.worldbody,
                f"{agent_name}_quad",
                np.array([offsets[i], 0.0, 2.0]),
                rgb=color,
            )
            writer.create_flexible_cable_payload(
                chassis,
                f"{agent_name}_cbl",
                pos=np.array([0.0, 0.0, 0.0]),
                N=cable_links,
                length=cable_length,
                mass=payload_mass,
                payload_rgb=tuple(color),
            )

        # Exclude inter-agent contacts across every body pair (quad, all cable
        # links, payload). Intra-agent contacts are untouched.
        def agent_bodies(i):
            return (
                [f"agent{i}_quad"]
                + [f"agent{i}_cbl_link{k}" for k in range(cable_links)]
                + [f"agent{i}_cbl_payload"]
            )

        for i in range(nQ):
            bodies_i = agent_bodies(i)
            for j in range(i + 1, nQ):
                bodies_j = agent_bodies(j)
                for bi in bodies_i:
                    for bj in bodies_j:
                        writer.exclude_contact(bi, bj)

        writer.save_to(filepath, verbose=False)
        return filename

    # ─── Index / address resolution after MuJoCo load ─────────────────

    def _resolve_mj_indices(self):
        """Look up body ids and per-agent qpos/qvel adrs via MuJoCo's name tables."""
        m = self._mjMdl.model
        self._quad_body_id = np.zeros(self.nQ, dtype=int)
        self._payload_body_id = np.zeros(self.nQ, dtype=int)
        self._quad_qposadr = np.zeros(self.nQ, dtype=int)
        self._quad_qveladr = np.zeros(self.nQ, dtype=int)

        for i in range(self.nQ):
            qb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"agent{i}_quad")
            pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"agent{i}_cbl_payload")
            if qb < 0 or pb < 0:
                raise RuntimeError(
                    f"agent{i}: quad_body={qb}, payload_body={pb} — name lookup failed"
                )
            self._quad_body_id[i] = qb
            self._payload_body_id[i] = pb
            # The quad has a single free joint; jnt_qposadr/jnt_dofadr give its start
            jnt_id = m.body_jntadr[qb]
            self._quad_qposadr[i] = m.jnt_qposadr[jnt_id]
            self._quad_qveladr[i] = m.jnt_dofadr[jnt_id]

        # Per-agent payload velocity is derived via finite difference on world xpos
        self._payload_pos_prev = np.zeros((self.nQ, 3))
        self._payload_vel_initialized = False

    # ─── Reset / state sync ───────────────────────────────────────────

    def reset(self, **kwargs):
        self.t = 0.0
        self._mjMdl.reset()

        payload_positions = kwargs.get("payload_positions")
        if payload_positions is None:
            payload_positions = [np.array([self._offsets[i], 0.0, 0.5]) for i in range(self.nQ)]
        else:
            payload_positions = [np.asarray(p, dtype=float) for p in payload_positions]

        # Place each quad directly above its intended payload position with cable down
        for i in range(self.nQ):
            agent = self.agents[i]
            agent.state.reset()
            agent.state.payload_position = payload_positions[i].copy()
            agent.state.position = payload_positions[i] + np.array([0.0, 0.0, self._cable_length])
            agent.t = 0.0

            qi = self._quad_qposadr[i]
            vi = self._quad_qveladr[i]
            self._mjMdl.data.qpos[qi : qi + 3] = agent.state.position
            quat = sp_rot.from_matrix(np.asarray(agent.state.orientation)).as_quat()
            self._mjMdl.data.qpos[qi + 3 : qi + 7] = [quat[3], quat[0], quat[1], quat[2]]
            self._mjMdl.data.qvel[vi : vi + 3] = 0.0
            self._mjMdl.data.qvel[vi + 3 : vi + 6] = 0.0

        self._payload_vel_initialized = False
        # Propagate qpos into xpos before reading payload body positions
        mujoco.mj_forward(self._mjMdl.model, self._mjMdl.data)
        self._query_latest_state()

    def _query_latest_state(self):
        self.t = self._mjMdl.data.time
        dt = max(self._mjMdl.model.opt.timestep, 1e-6)
        for i in range(self.nQ):
            agent = self.agents[i]
            qi = self._quad_qposadr[i]
            vi = self._quad_qveladr[i]
            agent.state.position = copy.deepcopy(self._mjMdl.data.qpos[qi : qi + 3])
            q = copy.deepcopy(self._mjMdl.data.qpos[qi + 3 : qi + 7])
            agent.state.orientation = SO3(self._mjMdl._quat2rot(q))
            agent.state.velocity = copy.deepcopy(self._mjMdl.data.qvel[vi : vi + 3])
            agent.state.angular_velocity = TSO3(
                copy.deepcopy(self._mjMdl.data.qvel[vi + 3 : vi + 6])
            )

            # Payload: derive from world xpos (same technique as single-agent nlink model)
            pL = copy.deepcopy(self._mjMdl.data.xpos[self._payload_body_id[i]])
            agent.state.payload_position = pL
            if self._payload_vel_initialized:
                agent.state.payload_velocity = (pL - self._payload_pos_prev[i]) / dt
            else:
                agent.state.payload_velocity = np.zeros(3)
            self._payload_pos_prev[i] = pL

            # Cable unit vector: from quadrotor to payload, normalized
            d = pL - agent.state.position
            nd = float(np.linalg.norm(d))
            if nd > 1e-6:
                q_hat = d / nd
                agent.state.cable_attitude = S2(q_hat)
                # Cable angular velocity from velocities (best-effort)
                rel_v = agent.state.payload_velocity - agent.state.velocity
                omega = np.cross(q_hat, rel_v) / max(nd, 1e-6)
                agent.state.cable_angular_velocity = TS2(omega)

        self._payload_vel_initialized = True

    # ─── Step ─────────────────────────────────────────────────────────

    def step(self, u):
        """u is (4 * nQ,): stacked [thrust, Mx, My, Mz] per agent."""
        for _ in range(self._step_iter):
            self._mjMdl.data.ctrl[:] = u
            self._mjMdl._step_mujoco_simulation(self._nFrames)
            self._query_latest_state()

    def simulate(self, tf, **kwargs):
        """Run closed-loop sim. Each agent uses its own _payload_controller."""
        self.reset(**kwargs)
        self._update_legend()

        while self.t < tf:
            u = np.zeros(4 * self.nQ)
            for i in range(self.nQ):
                agent = self.agents[i]
                agent.t = self.t
                # Payload controller outputs a desired acceleration (3-vec) u_i
                u_i = agent._payload_controller.compute(self.t, agent.state)
                # Repackage into wrench [f, Mx, My, Mz] via the same path as single-agent
                wrench = agent._repackage_input(u_i)
                u[4 * i : 4 * i + 4] = wrench
            self.step(u)

            if self.render and self._mjMdl._viewer is not None and int(self.t * 200) % 10 == 0:
                for i in range(self.nQ):
                    rgb, _ = _FLEET_COLORS[i % len(_FLEET_COLORS)]
                    self._mjMdl._viewer.add_trail_point(
                        self.agents[i].state.payload_position, key=i, rgba=[*rgb, 0.6]
                    )
                    target = self.agents[i]._payload_controller.setpoint(self.t)[0]
                    self._mjMdl._viewer.set_target(target, key=i)

        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl.wait_for_close()

    def _update_legend(self):
        if not (self.render and self._mjMdl._viewer is not None):
            return
        lines = []
        for i in range(self.nQ):
            _, color_name = _FLEET_COLORS[i % len(_FLEET_COLORS)]
            lines.append(f"[{color_name}] {self._labels[i]}")
        self._mjMdl.set_overlay("\n".join(lines))
