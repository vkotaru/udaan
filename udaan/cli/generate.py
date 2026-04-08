"""MuJoCo XML generation commands."""

import typer

from . import _ctx

generate_app = typer.Typer(help="Generate MuJoCo XML model files.", context_settings=_ctx)


@generate_app.command("multi-quad")
def multi_quad(
    num_quads: int = typer.Option(3, "--num-quads", "-n", help="Number of quadrotors."),
    output: str = typer.Option("multi_quad_pointmass.xml", "--output", "-o", help="Output file."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate MJCF XML for multi-quadrotor pointmass payload."""
    import udaan as U

    U.utils.xml_model_generator.multi_quad_pointmass(nQ=num_quads, filename=output, verbose=verbose)
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
