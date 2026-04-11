"""N-quadrotor fleet for comparing controllers side-by-side."""

import copy
import hashlib
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as sp_rot

from ... import _FOLDER_PATH
from ...manif import SO3, TSO3
from ...utils.logging import get_logger
from .. import base
from ..quadrotor import QuadrotorBase
from . import MujocoModel

_logger = get_logger(__name__)

_MJCF_DIR = os.path.join(_FOLDER_PATH, "udaan", "models", "assets", "mjcf")

# Distinct colors for up to 8 quadrotors (ADHD-friendly, high contrast)
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


class QuadrotorFleet(base.BaseModel):
    """Multiple independent quadrotors in a single MuJoCo scene.

    Each quadrotor has its own controllers and can optionally carry
    an added mass at an offset (for robustness testing).

    Usage::

        fleet = QuadrotorFleet(num_quadrotors=3, render=True)
        fleet[0].position_controller = MyController(...)

        # Add disturbance to quad 1: 0.25kg at [0.2, 0.2, -0.05]
        fleet = QuadrotorFleet(
            num_quadrotors=3,
            render=True,
            disturbances={
                1: {"mass": 0.25, "offset": [0.2, 0.2, -0.05]},
            },
        )
        fleet.simulate(tf=10, position=np.array([0, 0, 1]))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nQ = kwargs.get("num_quadrotors", 2)
        self.render = kwargs.get("render", False)
        self._labels = kwargs.get("labels", [f"quad{i}" for i in range(self.nQ)])
        self._disturbances = kwargs.get("disturbances", {})

        # Create N base quadrotor models (for state + controllers)
        self.quadrotors = [QuadrotorBase() for _ in range(self.nQ)]

        # Generate and load MuJoCo model
        xml_path = self._generate_xml(self.nQ, self._disturbances)
        self._mjMdl = MujocoModel(model_path=xml_path, render=self.render)

        self._mjDt = 1.0 / 500.0
        self._step_iter = int(self.sim_timestep / self._mjDt)
        self._nFrames = 1

        # Compute spawn offsets (same as XML generation)
        self._spacing = 1.5
        self._offsets = np.linspace(
            -(self.nQ - 1) * self._spacing / 2,
            (self.nQ - 1) * self._spacing / 2,
            self.nQ,
        )

        # Read inertial params from MuJoCo and reinit controllers
        for i in range(self.nQ):
            body_idx = i + 1  # body 0 is world
            self.quadrotors[i].mass = copy.deepcopy(self._mjMdl.model.body_mass[body_idx])
            self.quadrotors[i].inertia = copy.deepcopy(self._mjMdl.model.body_inertia[body_idx])
            self.quadrotors[i]._init_default_controllers()
            dist_str = ""
            if i in self._disturbances:
                d = self._disturbances[i]
                dist_str = f" [disturbance: {d['mass']}kg at {d.get('offset', [0, 0, 0])}]"
            _logger.info("  quad%d: mass=%.3f%s", i, self.quadrotors[i].mass, dist_str)

        _logger.info("Fleet loaded: %d quadrotors", self.nQ)

    def __getitem__(self, idx):
        return self.quadrotors[idx]

    def _update_legend(self):
        """Build and set overlay legend showing color → controller for each quad."""
        lines = []
        for i in range(self.nQ):
            _, color_name = _FLEET_COLORS[i % len(_FLEET_COLORS)]
            ctrl_name = type(self.quadrotors[i].position_controller).__name__
            dist = ""
            if i in self._disturbances:
                d = self._disturbances[i]
                dist = f" +{d['mass']}kg"
            lines.append(f"[{color_name}] {self._labels[i]}: {ctrl_name}{dist}")
        self._mjMdl.set_overlay("\n".join(lines))

    @staticmethod
    def _generate_xml(nQ, disturbances):
        """Generate MJCF with N independent quadrotors, no contact between them."""
        from ...utils.assets.mujoco_asset_creator import MujocoAssetCreator

        # Deterministic filename based on config
        config_key = json.dumps({"nQ": nQ, "d": disturbances}, sort_keys=True)
        config_hash = hashlib.md5(config_key.encode()).hexdigest()[:8]
        filename = f"fleet_{nQ}q_{config_hash}.xml"
        filepath = os.path.join(_MJCF_DIR, filename)
        if os.path.exists(filepath):
            return filename

        _logger.info("Generating fleet MJCF for %d quadrotors", nQ)
        writer = MujocoAssetCreator(f"Fleet{nQ}")

        spacing = 1.5
        offsets = np.linspace(-(nQ - 1) * spacing / 2, (nQ - 1) * spacing / 2, nQ)

        for i in range(nQ):
            color, _ = _FLEET_COLORS[i % len(_FLEET_COLORS)]
            quad_kwargs = {"rgb": color}

            # Add disturbance if specified for this quad
            if i in disturbances:
                d = disturbances[i]
                quad_kwargs["unmodeled_mass"] = d.get("mass", 0.0)
                quad_kwargs["unmodeled_mass_loc"] = np.array(d.get("offset", [0, 0, 0]))

            writer.create_quadrotor0(
                writer.worldbody,
                f"quad{i}",
                np.array([offsets[i], 0.0, 1.0]),
                **quad_kwargs,
            )

        # Exclude contact between all pairs
        for i in range(nQ):
            for j in range(i + 1, nQ):
                writer.exclude_contact(f"quad{i}", f"quad{j}")

        writer.save_to(filepath, verbose=False)
        return filename

    def reset(self, **kwargs):
        self.t = 0.0
        self._mjMdl.reset()

        # Reset all quadrotor states with spacing offsets
        base_pos = kwargs.get("position", np.array([0.0, 0.0, 0.0]))
        for i, quad in enumerate(self.quadrotors):
            quad.state.reset()
            for key in ["velocity", "orientation", "angular_velocity"]:
                if key in kwargs:
                    setattr(quad.state, key, copy.deepcopy(kwargs[key]))
            # Apply x-offset so quads don't stack
            quad.state.position = base_pos + np.array([self._offsets[i], 0.0, 0.0])

        # Write initial states to MuJoCo qpos/qvel
        for i in range(self.nQ):
            qi = 7 * i  # qpos offset
            vi = 6 * i  # qvel offset
            self._mjMdl.data.qpos[qi : qi + 3] = self.quadrotors[i].state.position
            quat = sp_rot.from_matrix(np.asarray(self.quadrotors[i].state.orientation)).as_quat()
            self._mjMdl.data.qpos[qi + 3 : qi + 7] = [quat[3], quat[0], quat[1], quat[2]]
            self._mjMdl.data.qvel[vi : vi + 3] = self.quadrotors[i].state.velocity
            self._mjMdl.data.qvel[vi + 3 : vi + 6] = self.quadrotors[i].state.angular_velocity

        self._query_latest_state()

        # Set start markers
        if self.render and self._mjMdl._viewer is not None:
            for i in range(self.nQ):
                rgb, _ = _FLEET_COLORS[i % len(_FLEET_COLORS)]
                self._mjMdl._viewer.set_start(self.quadrotors[i].state.position.copy(), key=i)
                target = self.quadrotors[i].position_controller.setpoint(0.0)[0]
                self._mjMdl._viewer.set_target(target, key=i)

    def _query_latest_state(self):
        self.t = self._mjMdl.data.time
        for i in range(self.nQ):
            qi = 7 * i
            vi = 6 * i
            self.quadrotors[i].state.position = copy.deepcopy(self._mjMdl.data.qpos[qi : qi + 3])
            q = copy.deepcopy(self._mjMdl.data.qpos[qi + 3 : qi + 7])
            self.quadrotors[i].state.orientation = SO3(self._mjMdl._quat2rot(q))
            self.quadrotors[i].state.velocity = copy.deepcopy(self._mjMdl.data.qvel[vi : vi + 3])
            self.quadrotors[i].state.angular_velocity = TSO3(
                copy.deepcopy(self._mjMdl.data.qvel[vi + 3 : vi + 6])
            )

    def step(self, u):
        """Step all quadrotors. u is (4*nQ,) wrench vector."""
        for _ in range(self._step_iter):
            self._mjMdl.data.ctrl[:] = u
            self._mjMdl._step_mujoco_simulation(self._nFrames)
            self._query_latest_state()

    def simulate(self, tf, **kwargs):
        """Run simulation with each quadrotor using its own controllers."""
        self.reset(**kwargs)
        self._update_legend()
        log_interval = kwargs.get("log_interval", 1.0)
        next_log = log_interval

        while self.t < tf:
            u = np.zeros(4 * self.nQ)
            for i in range(self.nQ):
                s = self.quadrotors[i].state
                # Position control -> thrust force vector
                F = self.quadrotors[i].position_controller.compute(self.t, (s.position, s.velocity))
                # Attitude control -> scalar thrust + torque
                f, M = self.quadrotors[i].attitude_controller.compute(
                    self.t, (s.orientation, s.angular_velocity), F
                )
                u[4 * i : 4 * i + 4] = [f, *M]
            self.step(u)

            # Trail points and dynamic targets (every 10th step to avoid slowdown)
            if self.render and self._mjMdl._viewer is not None and int(self.t * 200) % 10 == 0:
                for i in range(self.nQ):
                    rgb, _ = _FLEET_COLORS[i % len(_FLEET_COLORS)]
                    self._mjMdl._viewer.add_trail_point(
                        self.quadrotors[i].state.position, key=i, rgba=[*rgb, 0.6]
                    )
                    target = self.quadrotors[i].position_controller.setpoint(self.t)[0]
                    self._mjMdl._viewer.set_target(target, key=i)

            if self.t >= next_log:
                self._log_state()
                next_log += log_interval

        self._log_state()

        if self.render and self._mjMdl._viewer is not None:
            self._mjMdl.wait_for_close()

    def _log_state(self):
        """Log each quadrotor's state at DEBUG level."""
        lines = [f"t={self.t:.2f}s"]
        for i in range(self.nQ):
            s = self.quadrotors[i].state
            ctrl = self.quadrotors[i].attitude_controller
            pos_str = np.array2string(s.position, precision=3, suppress_small=True)
            line = f"  quad{i}: pos={pos_str}"
            if hasattr(ctrl, "sigma_hat"):
                sigma_str = np.array2string(ctrl.sigma_hat, precision=3, suppress_small=True)
                line += f" σ̂={sigma_str}"
            lines.append(line)
        _logger.debug("\n".join(lines))
