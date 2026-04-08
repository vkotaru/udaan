"""Udaan command-line interface."""

import typer

_ctx = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    name="udaan",
    help="Simulation and geometric control of aerial manipulation systems.",
    no_args_is_help=True,
    context_settings=_ctx,
)

# --- Subcommand groups ---
from .generate import generate_app  # noqa: E402
from .run import run_app  # noqa: E402

app.add_typer(run_app, name="run")
app.add_typer(generate_app, name="generate-xml")


# --- Top-level commands ---
@app.command("version")
def version():
    """Show udaan version."""
    from udaan import __version__

    typer.echo(f"udaan {__version__}")


def parse_vec(s: str | None, default):
    """Parse 'x,y,z' string to numpy array, or return default."""
    import numpy as np

    if s is None:
        return default
    try:
        return np.array([float(x) for x in s.split(",")])
    except ValueError:
        typer.echo(f"Invalid position format: '{s}'. Expected 'x,y,z'.", err=True)
        raise typer.Exit(1) from None
