"""Scenario builders: construct a model and bind its reference, without running it.

Each ``udaan run`` command and the MCP server (:mod:`udaan.mcp`) build their
simulation here, so the two run the same model, the same trajectory and the
same start for the same arguments. A builder returns a :class:`Scenario`; the
caller decides how to fly it (``scenario.run()``), what to print and whether to
record. Nothing here imports :mod:`typer` or touches a viewer.

A builder raises :class:`ScenarioError` for an argument it refuses — an unknown
backend, trajectory or controller, or a parameter the model rejects — with the
message the CLI prints.

The ``vfx`` (VPython) backends are not offered: they open a browser-served
scene at construction and block there, ``render=False`` included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Backends a scenario offers.
BACKENDS = ("base", "mujoco")
#: ``quadrotor`` trajectories.
QUADROTOR_TRAJECTORIES = (
    "hover",
    "step",
    "mission",
    "flip",
    "spiral",
    "circle",
    "lissajous",
    "figure8",
)
#: ``quad-payload`` trajectories.
QUAD_PAYLOAD_TRAJECTORIES = ("hover", "circle", "spiral")
#: ``quad-payload`` cable models: the cable models of ``QuadrotorCsPayloadMujoco``.
CABLE_MODELS = ("tendon", "links")
#: ``quad-flexcable`` and ``hose`` trajectories.
CHAIN_TRAJECTORIES = ("hover", "lissajous")
#: ``hose`` control loops.
HOSE_CONTROLLERS = ("lqr", "station")
#: ``tethered-quad`` and ``tethered-hose`` trajectories.
TETHER_TRAJECTORIES = ("hold", "circle", "climb")


class ScenarioError(ValueError):
    """An argument a scenario builder refuses; the message names the remedy."""


@dataclass
class Scenario:
    """A constructed model with its reference bound, ready to fly."""

    #: The model instance (any backend).
    model: Any
    #: Simulated duration, in s. Trajectories with their own length (flip,
    #: spiral, mission, figure-8, helix) set it themselves.
    tf: float
    #: Keyword arguments for ``model.simulate`` (the start state).
    initial: dict[str, Any]
    #: Short description, e.g. ``"base, hover"``.
    label: str
    #: ``() -> (k, 3)`` positions of the tracked points now: the airframe, the
    #: payload, the hose root or each vehicle.
    tracked: Callable[[], np.ndarray]
    #: ``t -> (k, 3)`` reference positions of the same points, or ``None`` when
    #: the scenario has nothing to score against.
    reference: Callable[[float], np.ndarray] | None
    #: Lines worth reporting before the run, e.g. the circle rate a tethered
    #: circle chose.
    notes: list[str] = field(default_factory=list)

    def run(self) -> None:
        """Fly the model to :attr:`tf` from :attr:`initial`."""
        self.model.simulate(tf=self.tf, **self.initial)

    def tracking_error(self) -> float:
        """The largest tracked-point distance from its reference now, in m (``nan`` without one)."""
        if self.reference is None:
            return float("nan")
        err = np.asarray(self.tracked()) - np.asarray(self.reference(self.model.t))
        return float(np.max(np.linalg.norm(err, axis=-1)))


# ─── Helpers ─────────────────────────────────────────────────────────


def _check_choice(kind: str, value: str, allowed: Sequence[str]) -> None:
    """Refuse a value outside ``allowed`` rather than fall through to a default branch."""
    if value not in allowed:
        raise ScenarioError(f"Unknown {kind} '{value}'. Expected one of: {', '.join(allowed)}.")


def _vec(value: Sequence[float] | np.ndarray | None, default: Sequence[float]) -> np.ndarray:
    """``value`` as a float 3-vector, or ``default`` when it is ``None``."""
    v = np.array(default if value is None else value, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ScenarioError(f"Expected a finite 'x,y,z' position, got {value!r}.")
    return v


def _rows(*points: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.array(points, dtype=float))


def _drop_none(opts: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in opts.items() if v is not None}


def _construct(cls: type, **kwargs: Any) -> Any:
    """``cls(**kwargs)``, with the model's own parameter refusals as :class:`ScenarioError`.

    ValueError covers the geometry a generator refuses to build and the
    parameter guards; RuntimeError covers the load-time checks that exist for a
    stale cached or hand-edited asset. Both are conditions with a stated
    remedy, not tracebacks.
    """
    try:
        return cls(**kwargs)
    except (ValueError, RuntimeError) as err:
        raise ScenarioError(f"Error: {err}") from err


# ─── M1: quadrotor ───────────────────────────────────────────────────


def quadrotor(
    *,
    backend: str = "mujoco",
    trajectory: str = "hover",
    duration: float = 10.0,
    position: Sequence[float] | None = None,
    render: bool = False,
    verbose: int = 0,
) -> Scenario:
    """M1: one quadrotor under geometric SE(3) control.

    Hover and the mission take off from the ground, at ``position`` (default
    ``(1, 1, 0)``). The step, circle, spiral, flip and figure-8 are flown *at*
    ``position`` (the circle's and figure-8's centre, the others' start), so
    their default sits 2 m up, ``(1, 1, 2)``. Every trajectory starts the
    vehicle on its own reference.
    """
    from udaan.models.quadrotor import QuadrotorBase

    _check_choice("model", backend, BACKENDS)
    _check_choice("trajectory", trajectory, QUADROTOR_TRAJECTORIES)

    on_ground = trajectory in ("hover", "lissajous", "mission")
    x0 = _vec(position, [1.0, 1.0, 0.0] if on_ground else [1.0, 1.0, 2.0])
    time = duration

    mdl: QuadrotorBase
    if backend == "mujoco":
        from udaan.models.quadrotor import QuadrotorMujoco

        mdl = QuadrotorMujoco(render=render, verbose=verbose)
    else:
        mdl = QuadrotorBase(verbose=verbose)

    # Each branch binds its own name: the classes share only `Trajectory`, and
    # `duration` lives on some of them, so a single `traj` would be typed as
    # the base and lose it.
    if trajectory == "flip":
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory(start=x0)
        mdl._pos_controller.setpoint = flip.get
        time = flip.duration
        x0 = flip.get(0.0)[0]
    elif trajectory == "spiral":
        from udaan.utils.trajectory import SpiralTrajectory

        spiral = SpiralTrajectory(start=x0)
        mdl._pos_controller.setpoint = spiral.get
        time = spiral.duration
        x0 = spiral.get(0.0)[0]
    elif trajectory == "circle":
        from udaan.utils.trajectory import CircularTraj

        circle = CircularTraj(center=x0, radius=1.0, speed=1.0, tf=time)
        mdl._pos_controller.setpoint = circle.get
        x0 = circle.get(0.0)[0]
    elif trajectory == "lissajous":
        from udaan.utils.trajectory import CrazyTrajectory

        # Lift the trajectory center so the z-oscillation (±1.5 m) stays above ground.
        center = x0 + np.array([0.0, 0.0, 2.0])
        lissajous = CrazyTrajectory(tf=time, center=center)
        mdl._pos_controller.setpoint = lissajous.get
        x0 = lissajous.get(0.0)[0]
    elif trajectory == "step":
        from udaan.utils.trajectory import StepSetpoint

        # Hold x0, then a true step of (1, 1, 0.5) m at t = 2 s (no feedforward).
        step = StepSetpoint(x0, x0 + np.array([1.0, 1.0, 0.5]), step_time=2.0)
        mdl._pos_controller.setpoint = step.get
    elif trajectory == "mission":
        from udaan.utils.trajectory import WaypointTrajectory

        # Take off from x0, five waypoints (stopping at each), land back on x0.
        offsets = [
            (0, 0, 0),
            (0, 0, 1.5),
            (2, 0, 1.5),
            (2, 2, 2.5),
            (0, 2, 1.5),
            (0, 0, 1.5),
            (0, 0, 0),
        ]
        mission = WaypointTrajectory([x0 + np.array(o, dtype=float) for o in offsets])
        mdl._pos_controller.setpoint = mission.get
        time = mission.duration
    elif trajectory == "figure8":
        from udaan.utils.trajectory import Figure8Trajectory

        # Two figure-8s about x0 with the heading swinging ±90°.
        figure8 = Figure8Trajectory(center=x0)
        mdl._pos_controller.setpoint = figure8.get
        mdl.yaw_setpoint = figure8.yaw
        time = figure8.duration
        x0 = figure8.get(0.0)[0]
    # Hover keeps the position controller's own setpoint and starts from x0.

    return Scenario(
        model=mdl,
        tf=time,
        initial={"position": x0},
        label=f"{backend}, {trajectory}",
        tracked=lambda: _rows(mdl.state.position),
        reference=lambda t: _rows(mdl.position_controller.setpoint(t)[0]),
    )


# ─── M2: quadrotor with a cable-suspended payload ───────────────────


def quad_payload(
    *,
    backend: str = "mujoco",
    cable_model: str = "tendon",
    trajectory: str = "hover",
    duration: float = 10.0,
    cable_length: float | None = None,
    position: Sequence[float] | None = None,
    render: bool = False,
) -> Scenario:
    """M2: one quadrotor carrying a point-mass payload on a taut cable.

    Hover starts the payload at ``position`` (default ``(-1, 2, 0.5)``) and
    flies it to the payload controller's own setpoint. The circle (1 m radius,
    1 m/s after a 2 s ramp from rest) and the helix (``spiral``) are flown *at*
    ``position`` (default ``(0, 0, 1.5)``). ``cable_model`` picks the MuJoCo
    cable discretisation; the analytical backend has one taut cable.
    """
    from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadBase

    _check_choice("model", backend, BACKENDS)
    _check_choice("cable model", cable_model, CABLE_MODELS)
    _check_choice("trajectory", trajectory, QUAD_PAYLOAD_TRAJECTORIES)

    x0 = _vec(position, [-1.0, 2.0, 0.5] if trajectory == "hover" else [0.0, 0.0, 1.5])
    time = duration

    mdl: QuadrotorCsPayloadBase
    if backend == "mujoco":
        from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadMujoco

        mdl = QuadrotorCsPayloadMujoco(render=render, cable_model=cable_model)
    else:
        mdl = QuadrotorCsPayloadBase()

    # The model refuses a non-positive or non-finite length, and the link chain
    # any length but its MJCF's.
    if cable_length is not None:
        try:
            mdl.cable_length = cable_length
        except ValueError as err:
            raise ScenarioError(f"--cable-length: {err}.") from err
        # The default controllers' gains and feed-forward are designed for ℓ.
        mdl._init_default_controllers()

    # Each trajectory binds `get_full`, whose owner also has `get_jet`, so the
    # cable and airframe feed-forwards both engage. The payload starts at rest
    # on its reference's initial position.
    if trajectory == "circle":
        from udaan.utils.trajectory import CircularTraj

        circle = CircularTraj(center=x0, radius=1.0, speed=1.0, tf=time, ramp_time=2.0)
        mdl.payload_controller.setpoint = circle.get_full
        x0 = circle.get(0.0)[0]
    elif trajectory == "spiral":
        from udaan.utils.trajectory import HelixTrajectory

        helix = HelixTrajectory(start=x0)
        mdl.payload_controller.setpoint = helix.get_full
        time = helix.duration
        x0 = helix.get(0.0)[0]

    # `cable_model` is a MuJoCo cable discretisation; naming one for the
    # analytical backend would misreport the run.
    detail = f"{backend}, {cable_model}" if backend == "mujoco" else backend
    return Scenario(
        model=mdl,
        tf=time,
        initial={"payload_position": x0},
        label=f"{detail}, {trajectory}, cable {mdl.cable_length:g} m",
        tracked=lambda: _rows(mdl.state.payload_position),
        reference=lambda t: _rows(mdl.payload_controller.setpoint(t)[0]),
    )


# ─── M3: nQ quadrotors, one point-mass payload ───────────────────────


def multi_quad_payload(
    *,
    backend: str = "mujoco",
    duration: float = 10.0,
    num_quadrotors: int = 3,
    cable_length: float | None = None,
    payload_mass: float | None = None,
    formation_radius: float | None = None,
    position: Sequence[float] | None = None,
    render: bool = False,
) -> Scenario:
    """M3: ``nQ`` quadrotors carrying one point-mass payload, payload at ``position``.

    The payload starts at ``position`` (default ``(-1, 2, 0.5)``) and is flown
    to the payload controller's own setpoint.
    """
    _check_choice("model", backend, BACKENDS)
    x0 = _vec(position, [-1.0, 2.0, 0.5])
    opts = _drop_none(
        {
            "cable_lengths": cable_length,
            "payload_mass": payload_mass,
            "formation_radius": formation_radius,
        }
    )
    if backend == "mujoco":
        from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadMujoco

        mdl = _construct(
            MultiQuadCsPayloadMujoco, render=render, num_quadrotors=num_quadrotors, **opts
        )
    else:
        from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadBase

        mdl = _construct(MultiQuadCsPayloadBase, num_quadrotors=num_quadrotors, **opts)
    return Scenario(
        model=mdl,
        tf=duration,
        initial={"payload_position": x0},
        label=f"{backend}, {mdl.num_quadrotors} quads",
        tracked=lambda: _rows(mdl.state.payload_position),
        reference=lambda t: _rows(mdl.payload_controller.setpoint(t)[0]),
    )


# ─── M4: nQ quadrotors, one rigid-body payload ───────────────────────


def multi_quad_rigidbody(
    *,
    backend: str = "mujoco",
    duration: float = 10.0,
    num_quadrotors: int = 4,
    cable_length: float | None = None,
    payload_mass: float | None = None,
    attachment_radius: float | None = None,
    formation_radius: float | None = None,
    spawn_height: float | None = None,
    position: Sequence[float] | None = None,
    render: bool = False,
) -> Scenario:
    """M4: ``nQ`` quadrotors carrying one rigid-body payload (geometric cascade).

    On MuJoCo the MJCF is generated from these knobs and cached, so the
    formation starts taut whatever they are set to, and the payload defaults
    to the MJCF's own spawn. On the analytical backend it defaults to
    ``(0, 0, 0.5)``; ``spawn_height`` shapes only the generated MJCF.
    """
    _check_choice("model", backend, BACKENDS)
    opts = _drop_none(
        {
            "cable_lengths": cable_length,
            "payload_mass": payload_mass,
            "attachment_radius": attachment_radius,
            "formation_radius": formation_radius,
        }
    )
    if backend == "mujoco":
        from udaan.models.multi_quad_rigidbody import MultiQuadRigidbodyMujoco

        if spawn_height is not None:
            opts["spawn_height"] = spawn_height
        mdl = _construct(
            MultiQuadRigidbodyMujoco, render=render, num_quadrotors=num_quadrotors, **opts
        )
        x0 = _vec(position, mdl.state.payload_position.copy())
    else:
        from udaan.models.multi_quad_rigidbody import MultiQuadRigidbodyBase

        mdl = _construct(MultiQuadRigidbodyBase, num_quadrotors=num_quadrotors, **opts)
        x0 = _vec(position, [0.0, 0.0, 0.5])
    return Scenario(
        model=mdl,
        tf=duration,
        initial={"payload_position": x0},
        label=f"{backend}, {mdl.num_quadrotors} quads",
        tracked=lambda: _rows(mdl.state.payload_position),
        reference=lambda t: _rows(mdl.payload_controller.setpoint(t)[0]),
    )


# ─── M5: quadrotor, payload on an n-link flexible cable ─────────────


def quad_flexcable(
    *,
    backend: str = "mujoco",
    trajectory: str = "hover",
    duration: float = 10.0,
    num_links: int | None = None,
    cable_length: float | None = None,
    cable_mass: float | None = None,
    payload_mass: float | None = None,
    position: Sequence[float] | None = None,
    render: bool = False,
    verbose: int = 0,
) -> Scenario:
    """M5: one quadrotor carrying a payload on a flexible ``n``-link cable (payload LQR).

    'lissajous' exposes ``get_jet``, so it also switches on the flatness
    feedforward at the full ``2n + 4`` derivative depth the chain needs.
    'hover' is a step response: the payload spawns at ``position`` (default
    ``(0, 0, 1)``) and is commanded ``(0.5, 0.3, 0.2)`` m away from it.
    """
    from udaan.models.quadrotor_flexcable import QuadrotorFlexCableBase

    _check_choice("model", backend, BACKENDS)
    _check_choice("trajectory", trajectory, CHAIN_TRAJECTORIES)
    opts = _drop_none(
        {
            "num_links": num_links,
            "cable_length": cable_length,
            "cable_mass": cable_mass,
            "payload_mass": payload_mass,
        }
    )
    Model: type[QuadrotorFlexCableBase]
    if backend == "mujoco":
        from udaan.models.quadrotor_flexcable import QuadrotorFlexCableMujoco

        Model = QuadrotorFlexCableMujoco
        opts["render"] = render
    else:
        Model = QuadrotorFlexCableBase
    mdl = _construct(Model, verbose=verbose, **opts)

    x0 = _vec(position, [0.0, 0.0, 1.0])
    if trajectory == "lissajous":
        from udaan.utils.trajectory import CrazyTrajectory

        # Gentle enough that no link leaves the hanging branch, which is what
        # the flat map refuses; the chain then feeds forward on every sample.
        traj = CrazyTrajectory(center=x0, ax=0.4, ay=0.35, az=0.25, f1=0.18, f2=0.13, f3=0.09)
        mdl.payload_controller.setpoint = traj.get
        x0 = traj.get(0.0)[0]
    else:
        # A step, not a fixed point at the spawn: commanding the payload to
        # where reset() puts it is already the closed-loop equilibrium, so the
        # run would measure nothing.
        target = x0 + np.array([0.5, 0.3, 0.2])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))

    return Scenario(
        model=mdl,
        tf=duration,
        initial={"payload_position": x0},
        label=f"{backend}, {mdl.num_links} links, {trajectory}",
        tracked=lambda: _rows(mdl.state.payload_position),
        reference=lambda t: _rows(mdl.payload_controller.setpoint(t)[0]),
    )


# ─── M6: nQ quadrotors in series on one hose ─────────────────────────


def hose(
    *,
    backend: str = "mujoco",
    duration: float = 10.0,
    num_links: int = 4,
    attachments: Sequence[int] | None = None,
    hose_length: float | None = None,
    hose_mass: float | None = None,
    quad_mass: float | None = None,
    sag_ratio: float | None = None,
    spawn_height: float | None = None,
    controller: str = "lqr",
    trajectory: str = "hover",
    position: Sequence[float] | None = None,
    render: bool = False,
) -> Scenario:
    """M6: several quadrotors carrying one flexible hose.

    ``controller="lqr"`` (the default) drives the hose LQR, which regulates
    the whole hose against the static trim at the root setpoint and needs a
    vehicle at the last vertex; a pattern without one is refused rather than
    silently station-kept. ``controller="station"`` runs the per-vehicle
    station-keeping PDs, which accept any attachment pattern. 'lissajous'
    exposes ``get_jet`` and switches on the flatness feedforward; 'hover' is a
    step of ``(0.3, 0.2, 0.2)`` m from the spawn. The analytical backend's
    root defaults to ``(0, 0, 2)``; MuJoCo's to its MJCF spawn.
    """
    from udaan.models.multi_quad_hose import MultiQuadHoseBase

    _check_choice("model", backend, BACKENDS)
    _check_choice("controller", controller, HOSE_CONTROLLERS)
    _check_choice("trajectory", trajectory, CHAIN_TRAJECTORIES)

    opts = _drop_none(
        {
            "hose_length": hose_length,
            "hose_mass": hose_mass,
            "masses": quad_mass,
            "sag_ratio": sag_ratio,
            "spawn_height": spawn_height,
        }
    )
    if attachments is not None:
        opts["attachments"] = tuple(int(v) for v in attachments)
    if controller == "lqr":
        # The LQR emits a per-vehicle wrench; the station-keeping PDs emit
        # thrust vectors. The two pipelines take incompatible shapes, so the
        # choice has to be made at construction.
        opts["input"] = "wrench"

    Model: type[MultiQuadHoseBase]
    if backend == "mujoco":
        from udaan.models.multi_quad_hose import MultiQuadHoseMujoco

        Model = MultiQuadHoseMujoco
    else:
        Model = MultiQuadHoseBase
        opts.pop("spawn_height", None)
        render = False
    mdl = _construct(Model, render=render, num_links=num_links, **opts)

    default_root = np.asarray(mdl.state.root_position).copy()
    if backend != "mujoco":
        default_root = np.array([0.0, 0.0, 2.0])
    x0 = _vec(position, default_root)

    tracked: Callable[[], np.ndarray]
    reference: Callable[[float], np.ndarray]
    if controller == "lqr":
        hc = mdl.hose_controller
        if hc is None:
            raise ScenarioError(
                f"Error: attachments {mdl.attachments} put no vehicle at the last vertex "
                f"({mdl.num_links}), so the hose is not differentially flat in the "
                "coordinates the LQR's reference is built in. Add that vertex, or run "
                "--controller station."
            )
        if trajectory == "lissajous":
            from udaan.utils.trajectory import CrazyTrajectory

            # Gentle: the constant gain is solved at the static trim and the
            # linearisation moves ~2 % over a reference this size.
            traj = CrazyTrajectory(center=x0, ax=0.3, ay=0.24, az=0.15, f1=0.2, f2=0.16, f3=0.12)
            hc.setpoint = traj.get
            x0 = traj.get(0.0)[0]
        else:
            # A step, not a fixed point at the spawn, so the error measures the gain.
            target = x0 + np.array([0.3, 0.2, 0.2])
            hc.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        tracked = lambda: _rows(mdl.state.root_position)
        reference = lambda t: _rows(hc.setpoint(t)[0])
    else:
        tracked = lambda: np.array(mdl.state.positions, dtype=float)
        reference = lambda t: np.array(
            [c.setpoint(t)[0] for c in mdl.position_controllers], dtype=float
        )

    return Scenario(
        model=mdl,
        tf=duration,
        initial={"root_position": x0},
        label=(
            f"{backend}, {mdl.num_links} links, {mdl.num_quadrotors} quads at "
            f"{mdl.attachments}, {controller}"
        ),
        tracked=tracked,
        reference=reference,
    )


# ─── M7 / M8: quadrotors on a hose pinned to a ground anchor ────────


def _tethered(
    *,
    system: str,
    backend: str,
    duration: float,
    opts: dict[str, Any],
    standoff: float | None,
    height: float | None,
    trajectory: str,
    rate: float | None,
    ramp: float | None,
    render: bool,
) -> Scenario:
    """Shared body of :func:`tethered_quad` (M7) and :func:`tethered_hose` (M8)."""
    from udaan.flatness.tethered_hose import (
        TetherBlend,
        TetherCircle,
        TetherHold,
        TetherReference,
        circle_rate_bound,
    )
    from udaan.models.tethered_hose import TetheredHoseBase, TetheredQuadrotorBase

    _check_choice("model", backend, BACKENDS)
    _check_choice("trajectory", trajectory, TETHER_TRAJECTORIES)
    opts = _drop_none(opts)
    Model: type[TetheredHoseBase]
    if backend == "mujoco":
        from udaan.models.tethered_hose import TetheredHoseMujoco, TetheredQuadrotorMujoco

        Model = TetheredQuadrotorMujoco if system == "m7" else TetheredHoseMujoco
    else:
        Model = TetheredQuadrotorBase if system == "m7" else TetheredHoseBase
        render = False
    mdl = _construct(Model, render=render, **opts)

    from udaan.control.tethered_hose import default_top_position
    from udaan.core.defaults import DEFAULT_TETHER_RAMP_TIME

    L = mdl.hose_length
    top = default_top_position(L, mdl.anchor_height)
    if standoff is not None:
        top[0] = standoff * L
    if height is not None:
        top[2] = mdl.anchor_height + height * L
    ramp = DEFAULT_TETHER_RAMP_TIME if ramp is None else ramp
    fm = mdl.hose_controller.flat_map
    notes: list[str] = []
    ref: TetherReference
    try:
        if trajectory == "hold":
            ref = TetherHold(fm, top)
        elif trajectory == "circle":
            w = circle_rate_bound(fm, top) if rate is None else rate
            ref = TetherCircle(fm, top, rate=w, ramp_time=ramp)
            notes.append(f"circle about the anchor at {w:.3f} rad/s, {ramp:g} s ramp from rest")
        else:
            # to 0.1 L closer and 0.08 L higher, starting at 1 s
            ref = TetherBlend(fm, top, top + np.array([-0.1 * L, 0.0, 0.08 * L]), ramp, 1.0)
    except ValueError as err:
        raise ScenarioError(f"Error: {err}") from err
    mdl.hose_controller.setpoint = ref.get

    return Scenario(
        model=mdl,
        tf=duration,
        initial={},
        label=(
            f"{backend}, {mdl.num_links} links, {mdl.num_quadrotors} quads at "
            f"{mdl.attachments}, {trajectory}"
        ),
        tracked=lambda: _rows(mdl.state.positions[-1]),
        reference=lambda t: _rows(ref.get(t)[0]),
        notes=notes,
    )


def tethered_quad(
    *,
    backend: str = "mujoco",
    duration: float = 10.0,
    num_links: int | None = None,
    length: float | None = None,
    tether_mass: float | None = None,
    anchor_height: float | None = None,
    standoff: float | None = None,
    height: float | None = None,
    trajectory: str = "hold",
    rate: float | None = None,
    ramp: float | None = None,
    render: bool = False,
) -> Scenario:
    """M7: one quadrotor tethered to a ground anchor by an ``n``-link cable.

    The hose LQR holds the vehicle at the given standoff and height ('hold'),
    flies a circle about the anchor started from rest by a ramp ('circle'), or
    blends to a higher, closer station ('climb'), with the flat map's
    feed-forward. ``standoff`` and ``height`` are fractions of the length.
    """
    return _tethered(
        system="m7",
        backend=backend,
        duration=duration,
        opts={
            "num_links": num_links,
            "hose_length": length,
            "hose_mass": tether_mass,
            "anchor_height": anchor_height,
        },
        standoff=standoff,
        height=height,
        trajectory=trajectory,
        rate=rate,
        ramp=ramp,
        render=render,
    )


def tethered_hose(
    *,
    backend: str = "mujoco",
    duration: float = 10.0,
    num_quadrotors: int | None = None,
    num_links: int | None = None,
    length: float | None = None,
    hose_mass: float | None = None,
    anchor_height: float | None = None,
    standoff: float | None = None,
    height: float | None = None,
    trajectory: str = "hold",
    rate: float | None = None,
    ramp: float | None = None,
    render: bool = False,
) -> Scenario:
    """M8: several quadrotors in series on one hose pinned to a ground anchor.

    The vehicles sit evenly on the line from the anchor to the top vehicle;
    the hose LQR holds them there ('hold'), flies the formation round the
    anchor ('circle') or blends it to a higher, closer station ('climb').
    """
    return _tethered(
        system="m8",
        backend=backend,
        duration=duration,
        opts={
            "num_quadrotors": num_quadrotors,
            "num_links": num_links,
            "hose_length": length,
            "hose_mass": hose_mass,
            "anchor_height": anchor_height,
        },
        standoff=standoff,
        height=height,
        trajectory=trajectory,
        rate=rate,
        ramp=ramp,
        render=render,
    )


# ─── MuJoCo-only multi-vehicle models ────────────────────────────────


def multi_quad(
    *,
    duration: float = 10.0,
    num_quadrotors: int = 3,
    position: Sequence[float] | None = None,
    render: bool = False,
) -> Scenario:
    """``MultiQuadrotorCSPointmass``: quadrotors station-keeping a point-mass payload.

    Each quadrotor's PD holds its own spawn position, offset by ``position``
    (default ``(-1, 2, 0.5)``); the tracked points are the quadrotors.
    """
    import udaan as U

    x0 = _vec(position, [-1.0, 2.0, 0.5])
    mdl: Any = _construct(
        U.models.mujoco.MultiQuadrotorCSPointmass, render=render, num_quadrotors=num_quadrotors
    )
    return Scenario(
        model=mdl,
        tf=duration,
        initial={"xL": x0},
        label=f"{mdl.nQ} quads",
        tracked=lambda: np.array([q.position for q in mdl.state.quads], dtype=float),
        reference=lambda t: np.array([q.position for q in mdl._init_state.quads], dtype=float),
    )


def multi_quad_rigid(
    *,
    duration: float = 10.0,
    position: Sequence[float] | None = None,
    render: bool = False,
) -> Scenario:
    """``MultiQuadRigidbody``: four quadrotors station-keeping a rigid-body payload.

    The formation is translated by ``position`` (default ``(0, 0, 0.5)``) and
    each quadrotor's PD holds the position it had at construction; the
    tracked points are the quadrotors.
    """
    import udaan as U

    x0 = _vec(position, [0.0, 0.0, 0.5])
    mdl = _construct(U.models.mujoco.MultiQuadRigidbody, render=render)
    return Scenario(
        model=mdl,
        tf=duration,
        initial={"xL": x0},
        label=f"{mdl.nQ} quads",
        tracked=lambda: np.array([q.position for q in mdl.state.quads], dtype=float),
        reference=lambda t: np.array([q.position for q in mdl._init_state.quads], dtype=float),
    )


def fleet(
    *,
    duration: float = 10.0,
    num_quadrotors: int = 3,
    demo: str | None = None,
    position: Sequence[float] | None = None,
    render: bool = False,
) -> Scenario:
    """``QuadrotorFleet``: independent quadrotors side by side, each with its own controllers.

    Without ``demo`` every quadrotor flies to ``(0, 0, 2)``. A demo
    (``l1-comparison``, ``gain-sweep``) fixes the count, the controllers and
    any disturbances. ``position`` (default ``(0, 0, 1)``) is the first
    quadrotor's start; the rest are offset along x.
    """
    import udaan as U

    from .demos import DEMOS

    x0 = _vec(position, [0.0, 0.0, 1.0])
    if demo is not None:
        _check_choice("demo", demo, tuple(DEMOS))
        entry: dict[str, Any] = DEMOS[demo]
        params = entry["config"]()
        f = U.models.mujoco.QuadrotorFleet(
            num_quadrotors=params["num_quadrotors"],
            render=render,
            disturbances=params["disturbances"],
        )
        entry["configure"](f)
        label = f"demo '{demo}', {f.nQ} quads"
    else:
        f = U.models.mujoco.QuadrotorFleet(num_quadrotors=num_quadrotors, render=render)
        target = np.array([0.0, 0.0, 2.0])
        for q in f.quadrotors:
            q.position_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        label = f"{f.nQ} quads"
    return Scenario(
        model=f,
        tf=duration,
        initial={"position": x0},
        label=label,
        tracked=lambda: np.array([q.state.position for q in f.quadrotors], dtype=float),
        reference=lambda t: np.array(
            [q.position_controller.setpoint(t)[0] for q in f.quadrotors], dtype=float
        ),
    )


def cspayload_fleet(
    *,
    duration: float = 8.0,
    num_agents: int = 2,
    demo: str | None = None,
    cable_links: int = 10,
    same_start: bool = False,
    render: bool = False,
) -> Scenario:
    """``QuadrotorCsPayloadFleet``: quadrotor-payload agents side by side on link-chain cables.

    Without ``demo`` every payload starts at ``(±1, ±1, 0.5)`` and flies to
    ``(0, 0, 1.5)``. A demo (``same-gains``, ``gain-sweep``) fixes the agent
    count, gains and per-lane manoeuvre; ``same_start`` puts every agent on
    one absolute start (demo only).
    """
    import udaan as U

    from .demos import CSPAYLOAD_DEMOS

    if demo is not None:
        _check_choice("demo", demo, tuple(CSPAYLOAD_DEMOS))
        entry: dict[str, Any] = CSPAYLOAD_DEMOS[demo]
        nQ = entry["config"]()["num_agents"]
        f = U.models.mujoco.QuadrotorCsPayloadFleet(
            num_agents=nQ, render=render, cable_links=cable_links
        )
        starts = entry["configure"](f, same_start=same_start)
        label = demo
    else:
        nQ = num_agents
        target = np.array([0.0, 0.0, 1.5])
        starts = [np.array([(-1.0) ** i * 1.0, (-1.0) ** i * 1.0, 0.5]) for i in range(nQ)]
        f = U.models.mujoco.QuadrotorCsPayloadFleet(
            num_agents=nQ, render=render, cable_links=cable_links
        )
        for i in range(nQ):
            f[i]._payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        label = f"{nQ} agents"
    return Scenario(
        model=f,
        tf=duration,
        initial={"payload_positions": starts},
        label=label,
        tracked=lambda: np.array([a.state.payload_position for a in f.agents], dtype=float),
        reference=lambda t: np.array(
            [a._payload_controller.setpoint(t)[0] for a in f.agents], dtype=float
        ),
    )
