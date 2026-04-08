"""Simulation run commands."""

import typer

from . import _ctx

run_app = typer.Typer(help="Run a simulation.", context_settings=_ctx)


def _hold_viewer(mdl):
    """Keep MuJoCo viewer open after simulation if it has one."""
    if hasattr(mdl, "_mjMdl") and mdl._mjMdl is not None:
        mdl._mjMdl.wait_for_close()


@run_app.command("quadrotor")
def quadrotor(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    backend: str = typer.Option(
        "mujoco", "--backend", "-b", help="Physics backend: mujoco or base."
    ),
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial position as 'x,y,z'."
    ),
):
    """Simulate a quadrotor with geometric SE(3) control."""
    import numpy as np

    import udaan as U

    from . import parse_vec

    x0 = parse_vec(position, default=np.array([1.0, 1.0, 0.0]))
    mdl_module = getattr(U.models, backend)
    mdl = mdl_module.Quadrotor(render=render)
    typer.echo(f"Running quadrotor ({backend}) for {time}s ...")
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
    typer.echo(f"Running quad-payload ({backend}, {model_type}) for {time}s ...")
    mdl.simulate(tf=time, payload_position=x0)
    _hold_viewer(mdl)


@run_app.command("multi-quad")
def multi_quad(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    num_quads: int = typer.Option(3, "--num-quads", "-n", help="Number of quadrotors."),
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
    typer.echo(f"Running multi-quad ({num_quads} quads) for {time}s ...")
    mdl.simulate(tf=time, xL=x0)
    _hold_viewer(mdl)
