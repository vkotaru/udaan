#!/usr/bin/env python3
"""Record the README gallery GIFs headless.

Each system runs with ``render=False``. During ``simulate()`` the MuJoCo
configuration (``qpos``) is logged at the GIF frame rate; afterwards the log is
replayed kinematically into an offscreen ``mujoco.Renderer`` with a fixed
camera framed on every body's path, and the frames are written as a GIF.

Usage::

    MUJOCO_GL=glfw python scripts/record_gallery.py            # every GIF
    MUJOCO_GL=glfw python scripts/record_gallery.py m1 m7      # a selection
    python scripts/record_gallery.py --list

``MUJOCO_GL`` selects the offscreen OpenGL backend (``egl``, ``osmesa`` or
``glfw``, which opens a hidden window); use whichever the machine provides.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

MEDIA_DIR = Path(__file__).resolve().parents[1] / ".media"
WIDTH, HEIGHT, FPS = 400, 300, 12
TRAIL_RGBA = (0.95, 0.45, 0.05, 0.9)


@dataclass
class Shot:
    """One gallery entry: how to build and fly a system, and what to film."""

    filename: str
    build: Callable[[], tuple[object, float, dict]]
    trail_bodies: tuple[str, ...] = ()
    azimuth: float = 135.0
    elevation: float = -22.0
    zoom: float = 1.0


# ── builders: each returns (model, tf, simulate kwargs) ───────────────────


def _m1():
    from udaan.models.quadrotor import QuadrotorMujoco
    from udaan.utils.trajectory import CircularTraj

    mdl = QuadrotorMujoco(render=False)
    circle = CircularTraj(
        center=np.array([0.0, 0.0, 1.5]), radius=1.0, speed=1.0, tf=12.0, ramp_time=2.0
    )
    mdl.position_controller.setpoint = circle.get
    return mdl, 12.0, {"position": circle.get(0.0)[0]}


def _m2(cable_model: str):
    from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadMujoco
    from udaan.utils.trajectory import CircularTraj, HelixTrajectory

    mdl = QuadrotorCsPayloadMujoco(render=False, cable_model=cable_model)
    if cable_model == "tendon":
        traj = CircularTraj(
            center=np.array([0.0, 0.0, 1.5]), radius=1.0, speed=1.0, tf=12.0, ramp_time=2.0
        )
        tf = 12.0
    else:
        traj = HelixTrajectory(start=np.array([0.0, 0.0, 1.5]))
        tf = traj.duration
    mdl.payload_controller.setpoint = traj.get_full
    return mdl, tf, {"payload_position": traj.get(0.0)[0]}


def _m3():
    from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadMujoco
    from udaan.utils.trajectory import CircularTraj

    mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3)
    traj = CircularTraj(
        center=np.array([0.0, 0.0, 1.0]), radius=1.0, speed=0.8, tf=12.0, ramp_time=3.0
    )
    mdl.payload_controller.setpoint = traj.get_full
    return mdl, 12.0, {"payload_position": traj.get(0.0)[0]}


def _m4():
    from udaan.models.multi_quad_rigidbody import MultiQuadRigidbodyMujoco
    from udaan.utils.trajectory import CircularTraj

    mdl = MultiQuadRigidbodyMujoco(render=False, num_quadrotors=4)
    centre = np.asarray(mdl.state.payload_position, dtype=float).copy()
    traj = CircularTraj(center=centre, radius=0.8, speed=0.6, tf=12.0, ramp_time=3.0)
    mdl.payload_controller.setpoint = traj.get_full
    return mdl, 12.0, {"payload_position": traj.get(0.0)[0]}


def _m5():
    from udaan.models.quadrotor_flexcable import QuadrotorFlexCableMujoco
    from udaan.utils.trajectory import CrazyTrajectory

    mdl = QuadrotorFlexCableMujoco(render=False)
    traj = CrazyTrajectory(
        center=np.array([0.0, 0.0, 1.0]), ax=0.4, ay=0.35, az=0.25, f1=0.18, f2=0.13, f3=0.09
    )
    mdl.payload_controller.setpoint = traj.get
    return mdl, 12.0, {"payload_position": traj.get(0.0)[0]}


def _m6():
    from udaan.models.multi_quad_hose import MultiQuadHoseMujoco
    from udaan.utils.trajectory import CrazyTrajectory

    mdl = MultiQuadHoseMujoco(render=False, num_links=4, input="wrench")
    x0 = np.asarray(mdl.state.root_position, dtype=float).copy()
    traj = CrazyTrajectory(center=x0, ax=0.3, ay=0.24, az=0.15, f1=0.2, f2=0.16, f3=0.12)
    mdl.hose_controller.setpoint = traj.get
    return mdl, 12.0, {"root_position": traj.get(0.0)[0]}


def _tethered(system: str):
    from udaan.control.tethered_hose import default_top_position
    from udaan.core.defaults import DEFAULT_TETHER_RAMP_TIME
    from udaan.flatness.tethered_hose import TetherCircle, circle_rate_bound
    from udaan.models.tethered_hose import TetheredHoseMujoco, TetheredQuadrotorMujoco

    model_cls = TetheredQuadrotorMujoco if system == "m7" else TetheredHoseMujoco
    mdl = model_cls(render=False)
    fm = mdl.hose_controller.flat_map
    top = default_top_position(mdl.hose_length, mdl.anchor_height)
    ref = TetherCircle(fm, top, rate=circle_rate_bound(fm, top), ramp_time=DEFAULT_TETHER_RAMP_TIME)
    mdl.hose_controller.setpoint = ref.get
    return mdl, 14.0, {}


def _fleet():
    from udaan.cli.demos import DEMOS
    from udaan.models.mujoco import QuadrotorFleet

    entry = DEMOS["l1-comparison"]
    params = entry["config"]()
    mdl = QuadrotorFleet(
        num_quadrotors=params["num_quadrotors"], render=False, disturbances=params["disturbances"]
    )
    entry["configure"](mdl)
    return mdl, 8.0, {"position": np.array([0.0, 0.0, 1.0])}


def _cspayload_fleet():
    from udaan.cli.demos import CSPAYLOAD_DEMOS
    from udaan.models.mujoco import QuadrotorCsPayloadFleet

    entry = CSPAYLOAD_DEMOS["gain-sweep"]
    params = entry["config"]()
    mdl = QuadrotorCsPayloadFleet(num_agents=params["num_agents"], render=False)
    starts = entry["configure"](mdl, same_start=False)
    return mdl, 8.0, {"payload_positions": starts}


SHOTS: dict[str, Shot] = {
    "m1": Shot("quadrotor.gif", _m1, ("quadrotor",), zoom=0.7),
    "m2-tendon": Shot("quad_payload_tendon.gif", lambda: _m2("tendon"), ("payload",), zoom=0.8),
    "m2-links": Shot("quad_payload_links.gif", lambda: _m2("links"), ("payload",), zoom=0.8),
    "m3": Shot("multi_quad_pointmass.gif", _m3, ("pyld",)),
    "m4": Shot("multi_quad_rigid.gif", _m4, ("pyld",)),
    "m5": Shot("quad_flexcable.gif", _m5, ("cable_payload",)),
    "m6": Shot("multi_quad_hose.gif", _m6, ("hose_vertex0",)),
    "m7": Shot("tethered_quad.gif", lambda: _tethered("m7"), ("quad0",), zoom=0.9),
    "m8": Shot("tethered_hose.gif", lambda: _tethered("m8"), ("quad2",), zoom=0.9),
    "fleet": Shot(
        "fleet_l1.gif",
        _fleet,
        ("quad0", "quad1", "quad2"),
        azimuth=90.0,
        elevation=-20.0,
        zoom=0.85,
    ),
    "cspayload-fleet": Shot(
        "cspayload_fleet.gif",
        _cspayload_fleet,
        tuple(f"agent{i}_cbl_payload" for i in range(4)),
        azimuth=90.0,
        elevation=-15.0,
        zoom=0.9,
    ),
}


# ── capture and render ────────────────────────────────────────────────────


def _log_qpos(mdl, tf: float, sim_kwargs: dict) -> list[np.ndarray]:
    """Fly ``mdl`` to ``tf`` and return its ``qpos`` sampled at ``FPS`` of simulated time."""
    mj = mdl._mjMdl
    frames: list[np.ndarray] = []
    next_t = [0.0]
    step = mj._step_mujoco_simulation

    def logged_step(n_frames=1):
        step(n_frames)
        if mj.data.time + 1e-9 >= next_t[0]:
            frames.append(mj.data.qpos.copy())
            next_t[0] += 1.0 / FPS

    mj._step_mujoco_simulation = logged_step
    mdl.simulate(tf=tf, **sim_kwargs)
    if not frames:
        raise RuntimeError(
            f"{type(mdl).__name__}.simulate() never called _step_mujoco_simulation; "
            "nothing was logged"
        )
    return frames


def _body_paths(model, data, frames) -> np.ndarray:
    """Every body's world position at every frame, ``(frames, nbody - 1, 3)``."""
    out = []
    for q in frames:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        out.append(data.xpos[1:].copy())
    return np.asarray(out)


def _camera(paths: np.ndarray, shot: Shot) -> mujoco.MjvCamera:
    pts = paths.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (lo + hi) / 2.0
    cam.distance = shot.zoom * max(2.5, 1.3 * float(np.max(hi - lo)) + 1.0)
    cam.azimuth = shot.azimuth
    cam.elevation = shot.elevation
    return cam


def _add_trail(scene, points: np.ndarray, width: float) -> None:
    for a, b in zip(points[:-1], points[1:], strict=True):
        if scene.ngeom >= scene.maxgeom or np.allclose(a, b):
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.asarray(TRAIL_RGBA, dtype=np.float32),
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.ngeom += 1


def record(name: str) -> Path:
    shot = SHOTS[name]
    mdl, tf, sim_kwargs = shot.build()
    frames = _log_qpos(mdl, tf, sim_kwargs)
    model, data = mdl._mjMdl.model, mdl._mjMdl.data
    paths = _body_paths(model, data, frames)
    cam = _camera(paths, shot)
    trail_ids = []
    for body in shot.trail_bodies:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id < 1:
            raise ValueError(f"{name}: no body named {body!r} in the model")
        trail_ids.append(body_id - 1)

    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    images = []
    for k, q in enumerate(frames):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        for i in trail_ids:
            _add_trail(renderer.scene, paths[: k + 1, i], 0.006 * cam.distance / 3.0)
        img = Image.fromarray(renderer.render())
        images.append(img.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=0))
    renderer.close()

    MEDIA_DIR.mkdir(exist_ok=True)
    out = MEDIA_DIR / shot.filename
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"{name}: {out.name}  {len(images)} frames  {out.stat().st_size / 1e6:.2f} MB")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("shots", nargs="*", help="shots to record (default: all)")
    parser.add_argument("--list", action="store_true", help="list the shots and exit")
    args = parser.parse_args(argv)
    if args.list:
        for name, shot in SHOTS.items():
            print(f"{name:16s} {shot.filename}")
        return 0
    unknown = [s for s in args.shots if s not in SHOTS]
    if unknown:
        parser.error(f"unknown shot(s): {', '.join(unknown)}; see --list")
    for name in args.shots or SHOTS:
        record(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
