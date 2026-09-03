"""MuJoCo XML generation commands."""

import typer

from . import _ctx

generate_app = typer.Typer(help="Generate MuJoCo XML model files.", context_settings=_ctx)


@generate_app.command("multi-quad")
def multi_quad(
    num_quads: int = typer.Option(3, "--num-quads", "-n", help="Number of quadrotors."),
    output: str = typer.Option("multi_quad_pointmass.xml", "--output", "-o", help="Output file."),
    cable_length: float | None = typer.Option(
        None, "--cable-length", "-l", help="Taut cable length in metres."
    ),
    payload_mass: float | None = typer.Option(None, "--payload-mass", help="Payload mass in kg."),
    quad_mass: float | None = typer.Option(
        None, "--quad-mass", help="Total mass of each quadrotor in kg."
    ),
    formation_radius: float | None = typer.Option(
        None, "--radius", "-r", help="Radius of the quadrotor ring in metres."
    ),
    spawn_height: float | None = typer.Option(
        None, "--height", "-z", help="Height of the quadrotor ring in metres."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate MJCF XML for multi-quadrotor pointmass payload.

    The payload height is derived so every cable starts taut.
    """
    import udaan as U

    kwargs = {
        "cable_length": cable_length,
        "payload_mass": payload_mass,
        "quad_mass": quad_mass,
        "formation_radius": formation_radius,
        "spawn_height": spawn_height,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        U.utils.xml_model_generator.multi_quad_pointmass(
            num_quadrotors=num_quads, filename=output, verbose=verbose, **kwargs
        )
    except ValueError as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(1) from err
    typer.echo(f"Written to {output}")


@generate_app.command("comparison")
def comparison(
    output: str = typer.Option("quadrotor_comparison.xml", "--output", "-o", help="Output file."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate MJCF XML for quadrotor comparison model."""
    import udaan as U

    U.utils.xml_model_generator.quadrotor_comparison(filename=output, verbose=verbose)
    typer.echo(f"Written to {output}")
