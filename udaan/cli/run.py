"""Simulation run commands."""

import typer

from . import _ctx

run_app = typer.Typer(help="Run a simulation.", context_settings=_ctx)

# Common option for recording
_record_option = typer.Option(None, "--record", "-r", help="Save recording to file (.gif or .mp4).")


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


@run_app.command("quadrotor")
def quadrotor(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    model: str = typer.Option(
        "mujoco", "--model", "-m", help="Quadrotor model: base, vfx, or mujoco."
    ),
    trajectory: str = typer.Option(
        "hover", "--trajectory", "--traj",
        help="Trajectory: hover, flip, spiral, circle, lissajous.",
    ),
    trail: bool = typer.Option(True, "--trail/--no-trail", help="Show trajectory trail."),
    verbose: int = typer.Option(0, "--verbose", "-v", help="Verbosity level (0-2)."),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial position as 'x,y,z'."
    ),
):
    """Simulate a quadrotor with geometric SE(3) control."""
    import logging

    import numpy as np

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    from udaan.models.quadrotor import QuadrotorBase

    from . import parse_vec

    x0 = parse_vec(position, default=np.array([1.0, 1.0, 0.0]))

    if model == "mujoco":
        from udaan.models.quadrotor import QuadrotorMujoco

        mdl = QuadrotorMujoco(render=render, verbose=verbose)
    elif model == "vfx":
        from udaan.models.quadrotor import QuadrotorVfx

        mdl = QuadrotorVfx(render=render, verbose=verbose)
    else:
        mdl = QuadrotorBase(verbose=verbose)

    # Set trajectory
    if trajectory == "flip":
        from udaan.utils.trajectory import FlipTrajectory

        traj = FlipTrajectory(start=x0)
        mdl._pos_controller.setpoint = traj.get
        time = traj.duration
    elif trajectory == "spiral":
        from udaan.utils.trajectory import SpiralTrajectory

        traj = SpiralTrajectory(start=x0)
        mdl._pos_controller.setpoint = traj.get
        time = traj.duration
    elif trajectory == "circle":
        from udaan.utils.trajectory import CircularTraj

        traj = CircularTraj(center=x0, radius=1.0, speed=1.0, tf=time)
        mdl._pos_controller.setpoint = traj.get
    elif trajectory == "lissajous":
        from udaan.utils.trajectory import CrazyTrajectory

        traj = CrazyTrajectory(tf=time, center=x0)
        mdl._pos_controller.setpoint = traj.get
        x0 = traj.get(0.0)[0]
    else:
        # Hover: default setpoint is [0,0,1], start from x0
        pass

    if hasattr(mdl, "_mjMdl") and mdl._mjMdl._viewer is not None:
        mdl._mjMdl._viewer.show_trails = trail
    _setup_recording(mdl, record)
    typer.echo(f"Running quadrotor ({model}, {trajectory}) for {time:.1f}s ...")
    mdl.simulate(tf=time, position=x0)
    _hold_viewer(mdl)


@run_app.command("quad-payload")
def quad_payload(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    backend: str = typer.Option(
        "mujoco", "--backend", "-b", help="Physics backend: mujoco or base."
    ),
    model_type: str = typer.Option(
        "tendon", "--model-type", "-m", help="Cable model: tendon, links, or cable."
    ),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial payload position as 'x,y,z'."
    ),
):
    """Simulate a quadrotor with cable-suspended payload."""
    import numpy as np

    import udaan as U

    from . import parse_vec

    x0 = parse_vec(position, default=np.array([-1.0, 2.0, 0.5]))
    mdl_module = getattr(U.models, backend)
    if backend == "mujoco":
        mdl = mdl_module.QuadrotorCSPayload(render=render, model=model_type)
    else:
        mdl = mdl_module.QuadrotorCSPayload(render=render)
    _setup_recording(mdl, record)
    typer.echo(f"Running quad-payload ({backend}, {model_type}) for {time}s ...")
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
