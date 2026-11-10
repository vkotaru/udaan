"""MuJoCo XML generation commands."""

from typing import Any

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

    kwargs: dict[str, Any] = {
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


@generate_app.command("multi-quad-rigid")
def multi_quad_rigid(
    num_quads: int = typer.Option(3, "--num-quads", "-n", help="Number of quadrotors."),
    output: str = typer.Option("multi_quad_rigidbody.xml", "--output", "-o", help="Output file."),
    cable_length: float | None = typer.Option(
        None, "--cable-length", "-l", help="Taut cable length in metres."
    ),
    payload_mass: float | None = typer.Option(None, "--payload-mass", help="Payload mass in kg."),
    payload_half_extents: str | None = typer.Option(
        None, "--payload-size", help="Payload slab half-extents as 'a,b,c' in metres."
    ),
    quad_mass: float | None = typer.Option(
        None, "--quad-mass", help="Total mass of each quadrotor in kg."
    ),
    attachment_radius: float | None = typer.Option(
        None, "--attachment-radius", "-a", help="Radius of the attachment ring on the payload."
    ),
    attachment_height: float | None = typer.Option(
        None, "--attachment-height", help="Height of the attachment ring above the payload centre."
    ),
    formation_radius: float | None = typer.Option(
        None,
        "--radius",
        "-r",
        help="Horizontal offset of each quadrotor from its own attachment point.",
    ),
    spawn_height: float | None = typer.Option(
        None, "--height", "-z", help="Height of the quadrotor ring in metres."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate MJCF XML for multi-quadrotor rigid-body payload.

    The payload height is derived so every cable starts taut, and the
    attachment azimuths are the ones the analytical model's ring uses.
    """
    import udaan as U

    from . import parse_vec

    kwargs: dict[str, Any] = {
        "cable_length": cable_length,
        "payload_mass": payload_mass,
        "quad_mass": quad_mass,
        "attachment_radius": attachment_radius,
        "attachment_height": attachment_height,
        "formation_radius": formation_radius,
        "spawn_height": spawn_height,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if payload_half_extents is not None:
        kwargs["payload_half_extents"] = parse_vec(payload_half_extents, default=None)
    try:
        U.utils.xml_model_generator.multi_quad_rigidbody(
            num_quadrotors=num_quads, filename=output, verbose=verbose, **kwargs
        )
    except ValueError as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(1) from err
    typer.echo(f"Written to {output}")


@generate_app.command("hose")
def hose(
    num_links: int = typer.Option(4, "--num-links", "-n", help="Links the hose is split into."),
    output: str = typer.Option("multi_quad_hose.xml", "--output", "-o", help="Output file."),
    attachments: str | None = typer.Option(
        None,
        "--attachments",
        "-a",
        help="Vertex indices carrying a quadrotor, e.g. '0,2,4'. Default: both ends.",
    ),
    hose_length: float | None = typer.Option(
        None, "--length", "-l", help="Total hose length in metres."
    ),
    hose_mass: float | None = typer.Option(None, "--hose-mass", help="Total hose mass in kg."),
    quad_mass: float | None = typer.Option(
        None, "--quad-mass", help="Total mass of each quadrotor in kg."
    ),
    sag_ratio: float | None = typer.Option(
        None, "--sag", help="Span of each hose segment, as a fraction of its own length."
    ),
    spawn_height: float | None = typer.Option(
        None, "--height", "-z", help="Height of vertex 0 in metres."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate MJCF XML for several quadrotors carrying one flexible hose.

    The spawn is taut: every link sits at exactly its own length, and a segment
    span the hose cannot reach is refused rather than clamped.
    """
    import udaan as U

    kwargs: dict[str, Any] = {
        "hose_length": hose_length,
        "hose_mass": hose_mass,
        "quad_mass": quad_mass,
        "sag_ratio": sag_ratio,
        "spawn_height": spawn_height,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if attachments is not None:
        try:
            kwargs["attachments"] = tuple(int(v) for v in attachments.split(","))
        except ValueError:
            typer.echo(f"Invalid attachments: '{attachments}'. Expected e.g. '0,2,4'.", err=True)
            raise typer.Exit(1) from None
    try:
        U.utils.xml_model_generator.multi_quad_hose(
            num_links=num_links, filename=output, verbose=verbose, **kwargs
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
