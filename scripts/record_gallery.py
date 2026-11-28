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
OBSTACLE_RGBA = (0.18, 0.19, 0.21, 1.0)
OBSTACLE_HALF_DEPTH = 0.4  # m, the box's half-size across the flight plane


@dataclass
class Shot:
    """One gallery entry: how to build and fly a system, and what to film."""

    filename: str
    build: Callable[[], tuple[object, float, dict]]
    trail_bodies: tuple[str, ...] = ()
    azimuth: float = 135.0
    elevation: float = -22.0
    zoom: float = 1.0
    obstacle: tuple[float, float, float] | None = None
    """A box on the ground, ``(x_min, x_max, height)`` in m, drawn in the video only.

    The models have contacts off, so the flight is planned to pass over it;
    :func:`record` checks every frame keeps the cable and vehicles above it.
    """


class _StraightMove:
    """Rest-to-rest straight line from ``start`` to ``end``, ``duration`` s from ``start_time``.

    The progress is one smooth step continuous through ``smoothness``, and
    ``get_jet`` gives every derivative the hose flat map reads.
    """

    def __init__(self, start, end, duration: float, start_time: float, smoothness: int):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.duration = float(duration)
        self.start_time = float(start_time)
        self.smoothness = int(smoothness)

    def get_jet(self, t: float, order: int) -> np.ndarray:
        from udaan.utils.trajectory.ramp import step_jet

        s = step_jet(t - self.start_time, self.duration, order, self.smoothness)
        out = np.outer(s, self.end - self.start)
        out[0] += self.start
        return out

    def get(self, t: float):
        pos, vel, acc = self.get_jet(t, 2)
        return pos, vel, acc


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
    from udaan.flatness.chain import chain_jet_order
    from udaan.models.multi_quad_hose import MultiQuadHoseMujoco

    n = 8
    mdl = MultiQuadHoseMujoco(render=False, num_links=n, input="wrench")
    move = _StraightMove(
        [-1.0, 0.0, 1.6], [2.0, 0.0, 1.6], 8.0, start_time=1.5, smoothness=chain_jet_order(n)
    )
    mdl.hose_controller.setpoint = move.get
    return mdl, 12.5, {"root_position": move.start}


def _tethered(system: str, num_links: int, num_quadrotors: int, duration: float, start, end):
    """Fly the top vehicle from ``start`` to ``end`` (fractions of the hose length)."""
    from udaan.flatness.tethered_hose import TetherBlend
    from udaan.models.tethered_hose import TetheredHoseMujoco, TetheredQuadrotorMujoco

    if system == "m7":
        mdl = TetheredQuadrotorMujoco(render=False, num_links=num_links)
    else:
        mdl = TetheredHoseMujoco(render=False, num_links=num_links, num_quadrotors=num_quadrotors)
    L = mdl.hose_length
    ref = TetherBlend(
        mdl.hose_controller.flat_map,
        np.array([start[0] * L, 0.0, start[1] * L]),
        np.array([end[0] * L, 0.0, end[1] * L]),
        duration,
        start_time=1.0,
    )
    mdl.hose_controller.setpoint = ref.get
    return mdl, duration + 3.0, {}


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
    "m6": Shot(
        "multi_quad_hose.gif",
        _m6,
        ("hose_vertex0",),
        azimuth=90.0,
        elevation=-12.0,
        zoom=0.8,
        obstacle=(0.85, 1.55, 0.95),
    ),
    "m7": Shot(
        "tethered_quad.gif",
        lambda: _tethered("m7", 8, 1, 8.0, (0.15, 0.94), (0.8, 0.58)),
        ("quad0",),
        azimuth=90.0,
        elevation=-12.0,
        zoom=0.85,
        obstacle=(0.95, 1.35, 0.45),
    ),
    "m8": Shot(
        "tethered_hose.gif",
        lambda: _tethered("m8", 9, 3, 16.0, (0.15, 0.97), (0.78, 0.6)),
        ("quad2",),
        azimuth=90.0,
        elevation=-12.0,
        zoom=0.85,
        obstacle=(1.25, 1.75, 0.6),
    ),
    "m8-pair": Shot(
        "tethered_hose_pair.gif",
        lambda: _tethered("m8", 8, 2, 10.0, (0.15, 0.97), (0.78, 0.6)),
        ("quad1",),
        azimuth=90.0,
        elevation=-12.0,
        zoom=0.85,
        obstacle=(1.2, 1.7, 0.75),
    ),
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


def _obstacle_corners(shot: Shot) -> np.ndarray:
    x0, x1, h = shot.obstacle
    d = OBSTACLE_HALF_DEPTH
    return np.array([[x, y, z] for x in (x0, x1) for y in (-d, d) for z in (0.0, h)])


def _check_clearance(name: str, shot: Shot, paths: np.ndarray, links: np.ndarray) -> float:
    """Smallest height of the cable and the bodies above the box top, over every frame.

    ``links`` holds pairs of body indices (into ``paths``) joined by a straight
    link; each is sampled along its length. Raises if anything is inside the box.
    """
    x0, x1, h = shot.obstacle
    u = np.linspace(0.0, 1.0, 11)[None, :, None]
    pts = [paths]
    if len(links):
        a, b = paths[:, links[:, 0]], paths[:, links[:, 1]]
        pts.append((a[:, :, None] * (1 - u) + b[:, :, None] * u).reshape(len(paths), -1, 3))
    p = np.concatenate([q.reshape(len(paths), -1, 3) for q in pts], axis=1).reshape(-1, 3)
    over = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (np.abs(p[:, 1]) <= OBSTACLE_HALF_DEPTH)
    clearance = float(np.min(p[over, 2]) - h) if over.any() else float("inf")
    if clearance <= 0.0:
        raise RuntimeError(f"{name}: the flight enters the obstacle ({clearance:.3f} m)")
    return clearance


def _add_box(scene, shot: Shot) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    x0, x1, h = shot.obstacle
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.array([(x1 - x0) / 2.0, OBSTACLE_HALF_DEPTH, h / 2.0]),
        np.array([(x0 + x1) / 2.0, 0.0, h / 2.0]),
        np.eye(3).ravel(),
        np.asarray(OBSTACLE_RGBA, dtype=np.float32),
    )
    scene.ngeom += 1


def _camera(paths: np.ndarray, shot: Shot) -> mujoco.MjvCamera:
    pts = paths.reshape(-1, 3)
    if shot.obstacle is not None:
        pts = np.vstack([pts, _obstacle_corners(shot)])
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
    if shot.obstacle is not None:
        parents = model.body_parentid[1:] - 1
        links = np.array([(p, i) for i, p in enumerate(parents) if p >= 0], dtype=int)
        clearance = _check_clearance(name, shot, paths, links)
        print(f"{name}: clears the obstacle by {100 * clearance:.1f} cm")
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
        if shot.obstacle is not None:
            _add_box(renderer.scene, shot)
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
