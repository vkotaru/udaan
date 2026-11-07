"""Simulation run commands."""

import typer

from . import _ctx

run_app = typer.Typer(help="Run a simulation.", context_settings=_ctx)

# Common option for recording
_record_option = typer.Option(None, "--record", "-r", help="Save recording to file (.gif or .mp4).")


#: The backends ``udaan run quadrotor --model`` offers.  ``vfx`` is deliberately
#: absent even though :class:`~udaan.models.quadrotor.QuadrotorVfx` exists and is
#: importable: driven from the CLI, it blocks inside VPython and does not return,
#: ``--no-render`` included, because the VPython backend opens its
#: browser-served scene at construction rather than at render time.  Advertising
#: a hanging option is worse than not offering it, so the command refuses
#: ``-m vfx`` by name.
_CLI_MODELS = ("base", "mujoco")


def _check_model(model: str, allowed: tuple[str, ...]) -> None:
    """Reject an unrecognised ``--model`` instead of silently picking a backend.

    The routing block it guards is a chain of ``if``/``elif`` with a final
    ``else``, so without this guard a typo lands on whichever backend sits in
    the ``else``, and the run reports a backend the caller never asked for.
    """
    if model not in allowed:
        typer.secho(
            f"Unknown model '{model}'. Expected one of: {', '.join(allowed)}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


def _hold_viewer(mdl):
    """Keep MuJoCo viewer open after simulation if it has one."""
    if hasattr(mdl, "_mjMdl") and mdl._mjMdl is not None:
        mdl._mjMdl.wait_for_close()


def _setup_recording(mdl, record_path):
    """Enable recording on a model's viewer."""
    if record_path is None:
        return
    mj = mdl._mjMdl if hasattr(mdl, "_mjMdl") else None
    if mj and mj._viewer:
        mj._viewer._record_path = record_path
        mj._viewer._frames = []


#: ``udaan run quadrotor --traj`` choices.
_QUADROTOR_TRAJECTORIES = (
    "hover",
    "step",
    "mission",
    "flip",
    "spiral",
    "circle",
    "lissajous",
    "figure8",
)


@run_app.command("quadrotor")
def quadrotor(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    model: str = typer.Option("mujoco", "--model", "-m", help="Quadrotor model: base or mujoco."),
    trajectory: str = typer.Option(
        "hover",
        "--trajectory",
        "--traj",
        help="Trajectory: hover, step, mission, flip, spiral, circle, lissajous, figure8.",
    ),
    trail: bool = typer.Option(True, "--trail/--no-trail", help="Show trajectory trail."),
    verbose: int = typer.Option(0, "--verbose", "-v", help="Verbosity level (0-2)."),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None,
        "--position",
        "-p",
        help="Initial position as 'x,y,z' (the circle's and figure-8's centre, the "
        "mission's take-off point; default 1,1,0 for hover/lissajous/mission, 1,1,2 "
        "otherwise).",
    ),
):
    """Simulate a quadrotor with geometric SE(3) control."""
    import logging

    import numpy as np

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    from udaan.models.quadrotor import QuadrotorBase

    from . import parse_vec

    _check_model(model, _CLI_MODELS)
    if trajectory not in _QUADROTOR_TRAJECTORIES:
        typer.secho(
            f"Unknown trajectory '{trajectory}'. "
            f"Expected one of: {', '.join(_QUADROTOR_TRAJECTORIES)}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    # Hover and the mission take off from the ground. The step, circle, spiral,
    # flip and figure-8 are flown *at* `x0` (the circle's and figure-8's centre,
    # the others' start), so their default sits 2 m up, clear of the floor. An
    # explicit `-p` is taken as given.
    on_ground = trajectory in ("hover", "lissajous", "mission")
    default_start = [1.0, 1.0, 0.0] if on_ground else [1.0, 1.0, 2.0]
    x0 = parse_vec(position, default=np.array(default_start))

    mdl: QuadrotorBase
    if model == "mujoco":
        from udaan.models.quadrotor import QuadrotorMujoco

        mdl = QuadrotorMujoco(render=render, verbose=verbose)
    else:
        mdl = QuadrotorBase(verbose=verbose)

    # Set trajectory.  Each branch binds its own name: the four classes share
    # only `Trajectory`, and `duration` lives on two of them, so a single
    # `traj` would be typed as the base and lose it. Each also starts the
    # vehicle on its own reference, so the run opens on the manoeuvre rather
    # than on a jump to it.
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
    else:
        # Hover: default setpoint is [0,0,1], start from x0
        pass

    mj = getattr(mdl, "_mjMdl", None)
    if mj is not None and mj._viewer is not None:
        mj._viewer.show_trails = trail
    _setup_recording(mdl, record)
    typer.echo(f"Running quadrotor ({model}, {trajectory}) for {time:.1f}s ...")
    mdl.simulate(tf=time, position=x0)
    _hold_viewer(mdl)


@run_app.command("quad-payload")
def quad_payload(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    model: str = typer.Option(
        "mujoco", "--model", "-m", help="Quad-payload model: mujoco or base."
    ),
    cable_model: str = typer.Option(
        "tendon", "--cable-model", "-c", help="Cable model: tendon, links, or cable."
    ),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial payload position as 'x,y,z'."
    ),
):
    """Simulate a quadrotor with cable-suspended payload."""
    import numpy as np

    from . import parse_vec

    x0 = parse_vec(position, default=np.array([-1.0, 2.0, 0.5]))
    if cable_model == "cable":
        typer.secho(
            "WARNING: cable model 'cable' is not officially supported; "
            "included for experimentation only.",
            fg=typer.colors.YELLOW,
            bold=True,
            err=True,
        )
    if model == "mujoco":
        from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadMujoco

        mdl = QuadrotorCsPayloadMujoco(render=render, cable_model=cable_model)
    else:
        from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadVfx

        mdl = QuadrotorCsPayloadVfx(render=render)
    _setup_recording(mdl, record)
    typer.echo(f"Running quad-payload ({model}, {cable_model}) for {time}s ...")
    mdl.simulate(tf=time, payload_position=x0)
    _hold_viewer(mdl)


@run_app.command("multi-quad")
def multi_quad(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    num_quads: int = typer.Option(3, "--num-quads", "-n", help="Number of quadrotors."),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial payload position as 'x,y,z'."
    ),
):
    """Simulate multiple quadrotors with cable-suspended pointmass payload."""
    import numpy as np

    import udaan as U

    from . import parse_vec

    x0 = parse_vec(position, default=np.array([-1.0, 2.0, 0.5]))
    mdl = U.models.mujoco.MultiQuadrotorCSPointmass(render=render, num_quadrotors=num_quads)
    _setup_recording(mdl, record)
    typer.echo(f"Running multi-quad ({num_quads} quads) for {time}s ...")
    mdl.simulate(tf=time, xL=x0)
    _hold_viewer(mdl)


@run_app.command("multi-quad-rigid")
def multi_quad_rigid(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial payload position as 'x,y,z'."
    ),
):
    """Simulate multiple quadrotors with cable-suspended rigid-body payload."""
    import numpy as np

    import udaan as U

    from . import parse_vec

    x0 = parse_vec(position, default=np.array([0.0, 0.0, 0.5]))
    mdl = U.models.mujoco.MultiQuadRigidbody(render=render)
    _setup_recording(mdl, record)
    typer.echo(f"Running multi-quad rigid ({mdl.nQ} quads) for {time}s ...")
    mdl.simulate(tf=time, xL=x0)
    _hold_viewer(mdl)


@run_app.command("cspayload-fleet")
def cspayload_fleet(
    time: float = typer.Option(8.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    num_agents: int = typer.Option(2, "--num-agents", "-n", help="Number of quad-payload agents."),
    demo: str | None = typer.Option(None, "--demo", "-d", help="Run a preset demo."),
    cable_links: int = typer.Option(10, "--cable-links", help="Cable discretization (links)."),
    trail: bool = typer.Option(True, "--trail/--no-trail", help="Show payload trails."),
    same_start: bool = typer.Option(
        False,
        "--same-start",
        help="All agents start at the same absolute position (demo only; overlaps visually).",
    ),
    record: str | None = _record_option,
):
    """Compare N quadrotor-cable-payload agents side-by-side.

    Available demos: same-gains, gain-sweep
    """
    import udaan as U

    from .demos import CSPAYLOAD_DEMOS

    if demo is not None:
        if demo not in CSPAYLOAD_DEMOS:
            typer.echo(
                f"Unknown demo: '{demo}'. Available: {', '.join(CSPAYLOAD_DEMOS.keys())}",
                err=True,
            )
            raise typer.Exit(1)
        entry = CSPAYLOAD_DEMOS[demo]
        params = entry["config"]()
        nQ = params["num_agents"]

        f = U.models.mujoco.QuadrotorCsPayloadFleet(
            num_agents=nQ, render=render, cable_links=cable_links
        )
        starts = entry["configure"](f, same_start=same_start)
    else:
        nQ = num_agents
        import numpy as np

        target = np.array([0.0, 0.0, 1.5])
        starts = [np.array([(-1.0) ** i * 1.0, (-1.0) ** i * 1.0, 0.5]) for i in range(nQ)]
        f = U.models.mujoco.QuadrotorCsPayloadFleet(
            num_agents=nQ, render=render, cable_links=cable_links
        )
        traj = lambda t: (target, np.zeros(3), np.zeros(3))
        for i in range(nQ):
            f[i]._payload_controller.setpoint = traj

    if f._mjMdl._viewer is not None:
        f._mjMdl._viewer.show_trails = trail
    if record and f._mjMdl._viewer:
        f._mjMdl._viewer._record_path = record
        f._mjMdl._viewer._frames = []

    label = demo or f"{nQ} agents"
    typer.echo(f"Running cspayload-fleet ({label}) for {time}s ...")
    f.simulate(tf=time, payload_positions=starts)


@run_app.command("fleet")
def fleet(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    num_quads: int = typer.Option(3, "--num-quads", "-n", help="Number of quadrotors."),
    demo: str | None = typer.Option(None, "--demo", "-d", help="Run a preset demo."),
    trail: bool = typer.Option(False, "--trail/--no-trail", help="Show trajectory trails."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log quad states during sim."),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial position as 'x,y,z'."
    ),
):
    """Compare N quadrotors with independent controllers side-by-side.

    Available demos: l1-comparison, gain-sweep
    """
    import logging

    import numpy as np

    import udaan as U

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    from . import parse_vec
    from .demos import DEMOS

    x0 = parse_vec(position, default=np.array([0.0, 0.0, 1.0]))

    if demo is not None:
        if demo not in DEMOS:
            typer.echo(f"Unknown demo: '{demo}'. Available: {', '.join(DEMOS.keys())}", err=True)
            raise typer.Exit(1)

        entry = DEMOS[demo]
        params = entry["config"]()
        nQ = params["num_quadrotors"]

        f = U.models.mujoco.QuadrotorFleet(
            num_quadrotors=nQ, render=render, disturbances=params["disturbances"]
        )
        entry["configure"](f)
        if f._mjMdl._viewer is not None:
            f._mjMdl._viewer.show_trails = trail
        if record and f._mjMdl._viewer:
            f._mjMdl._viewer._record_path = record
            f._mjMdl._viewer._frames = []
        typer.echo(f"Running demo '{demo}' ({nQ} quads) for {time}s ...")
        f.simulate(tf=time, position=x0)
    else:
        f = U.models.mujoco.QuadrotorFleet(num_quadrotors=num_quads, render=render)
        if f._mjMdl._viewer is not None:
            f._mjMdl._viewer.show_trails = trail

        traj = lambda t: (np.array([0.0, 0.0, 2.0]), np.zeros(3), np.zeros(3))
        for q in f.quadrotors:
            q.position_controller.setpoint = traj

        if record and f._mjMdl._viewer:
            f._mjMdl._viewer._record_path = record
            f._mjMdl._viewer._frames = []
        typer.echo(f"Running fleet ({num_quads} quads) for {time}s ...")
        f.simulate(tf=time, position=x0)
