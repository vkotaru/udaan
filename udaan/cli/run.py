"""Simulation run commands.

Each command parses its options, builds its model through
:mod:`udaan.cli.scenarios` (shared with the MCP server), sets up the viewer and
any recording, and flies it.
"""

from collections.abc import Callable
from typing import Any

import typer

from . import _ctx
from . import scenarios as sc

run_app = typer.Typer(help="Run a simulation.", context_settings=_ctx)

# Common option for recording
_record_option = typer.Option(None, "--record", "-r", help="Save recording to file (.gif or .mp4).")


def _build(builder: Callable[..., sc.Scenario], **kwargs: Any) -> sc.Scenario:
    """Call a scenario builder, turning its refusal into a red message and exit code 1."""
    try:
        return builder(**kwargs)
    except sc.ScenarioError as err:
        typer.secho(str(err), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from err


def _parse_ints(value: str | None, example: str) -> tuple[int, ...] | None:
    """``'0,2,4'`` as ``(0, 2, 4)``; exits 1 on anything else."""
    if value is None:
        return None
    try:
        return tuple(int(v) for v in value.split(","))
    except ValueError:
        typer.echo(f"Invalid attachments: '{value}'. Expected e.g. '{example}'.", err=True)
        raise typer.Exit(1) from None


def _hold_viewer(mdl):
    """Keep MuJoCo viewer open after simulation if it has one."""
    if hasattr(mdl, "_mjMdl") and mdl._mjMdl is not None:
        mdl._mjMdl.wait_for_close()


def _setup_viewer(mdl, record_path, trail=None):
    """Set trails and enable recording on a model's viewer, if it has one."""
    mj = getattr(mdl, "_mjMdl", None)
    viewer = mj._viewer if mj is not None else None
    if viewer is None:
        return
    if trail is not None:
        viewer.show_trails = trail
    if record_path is not None:
        viewer._record_path = record_path
        viewer._frames = []


def _position(position: str | None):
    """Parse ``--position``; ``None`` lets the scenario pick its own default."""
    from . import parse_vec

    return parse_vec(position, default=None)


@run_app.command("quadrotor")
def quadrotor(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    model: str = typer.Option("mujoco", "--model", "-m", help="Quadrotor model: base or mujoco."),
    trajectory: str = typer.Option(
        "hover",
        "--trajectory",
        "--traj",
        help="Trajectory: hover, step, mission, flip, spiral, circle (1 m radius, 1 m/s after a "
        "2 s ramp from rest), lissajous, figure8.",
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

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    scn = _build(
        sc.quadrotor,
        backend=model,
        trajectory=trajectory,
        duration=time,
        position=_position(position),
        render=render,
        verbose=verbose,
    )
    _setup_viewer(scn.model, record, trail=trail)
    typer.echo(f"Running quadrotor ({scn.label}) for {scn.tf:.1f}s ...")
    scn.run()
    _hold_viewer(scn.model)


@run_app.command("quad-payload")
def quad_payload(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    model: str = typer.Option(
        "mujoco", "--model", "-m", help="Quad-payload model: base or mujoco."
    ),
    cable_model: str = typer.Option(
        "tendon", "--cable-model", "-c", help="Cable model: tendon or links."
    ),
    trajectory: str = typer.Option(
        "hover",
        "--trajectory",
        "--traj",
        help="Payload trajectory: hover, circle (1 m radius, 1 m/s after a 2 s ramp from "
        "rest), spiral (horizontal helix: 0.8 m radius, 2 m/s, two loops climbing 1 m).",
    ),
    cable_length: float | None = typer.Option(
        None,
        "--cable-length",
        help="Cable length in metres (default: the model's own, 1 m). The link chain's "
        "length is fixed by its MJCF, so -c links accepts only that length.",
    ),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None,
        "--position",
        "-p",
        help="Initial payload position as 'x,y,z' (the circle's centre, the helix's start; "
        "default -1,2,0.5 for hover, 0,0,1.5 otherwise).",
    ),
):
    """Simulate a quadrotor with cable-suspended payload."""
    scn = _build(
        sc.quad_payload,
        backend=model,
        cable_model=cable_model,
        trajectory=trajectory,
        duration=time,
        cable_length=cable_length,
        position=_position(position),
        render=render,
    )
    _setup_viewer(scn.model, record)
    typer.echo(f"Running quad-payload ({scn.label}) for {scn.tf:.1f}s ...")
    scn.run()
    _hold_viewer(scn.model)


@run_app.command("quad-flexcable")
def quad_flexcable(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    links: int | None = typer.Option(
        None, "--links", "-n", help="Number of rigid links the cable is discretised into."
    ),
    model: str = typer.Option("mujoco", "--model", "-m", help="Flex-cable model: mujoco or base."),
    trajectory: str = typer.Option(
        "hover", "--trajectory", "--traj", help="Payload trajectory: hover or lissajous."
    ),
    cable_length: float | None = typer.Option(
        None, "--cable-length", "-l", help="Total taut cable length in metres."
    ),
    cable_mass: float | None = typer.Option(None, "--cable-mass", help="Cable mass in kg."),
    payload_mass: float | None = typer.Option(None, "--payload-mass", help="Payload mass in kg."),
    verbose: int = typer.Option(0, "--verbose", "-v", help="Verbosity level (0-2)."),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial payload position as 'x,y,z'."
    ),
):
    """Simulate a quadrotor carrying a payload on a flexible n-link cable.

    Drives the M5 payload LQR. The 'lissajous' trajectory exposes ``get_jet``,
    so it also switches on the flatness feedforward at the full ``2n + 4``
    derivative depth the chain needs; 'hover' is a step response — the payload
    spawns at ``--position`` and is commanded 0.62 m away from it — and needs
    no feedforward.
    """
    import logging

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    scn = _build(
        sc.quad_flexcable,
        backend=model,
        trajectory=trajectory,
        duration=time,
        num_links=links,
        cable_length=cable_length,
        cable_mass=cable_mass,
        payload_mass=payload_mass,
        position=_position(position),
        render=render,
        verbose=verbose,
    )
    _setup_viewer(scn.model, record)
    typer.echo(f"Running quad-flexcable ({scn.label}) for {scn.tf}s ...")
    scn.run()
    typer.echo(
        f"  payload error {scn.tracking_error():.4f} m"
        f"  (feedforward drops: {scn.model.feedforward_drops})"
    )
    _hold_viewer(scn.model)


@run_app.command("hose")
def hose(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    model: str = typer.Option("mujoco", "--model", "-m", help="Backend: 'mujoco' or 'base'."),
    render: bool = typer.Option(True, help="Enable visualization."),
    num_links: int = typer.Option(4, "--num-links", "-n", help="Links the hose is split into."),
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
    quad_mass: float | None = typer.Option(None, "--quad-mass", help="Mass of each airframe, kg."),
    sag_ratio: float | None = typer.Option(
        None, "--sag", help="Spawn span of each hose segment, as a fraction of its length."
    ),
    spawn_height: float | None = typer.Option(
        None, "--height", "-z", help="Height of vertex 0 at spawn, in metres."
    ),
    controller: str = typer.Option(
        "lqr",
        "--controller",
        "-c",
        help=(
            "Control loop: 'lqr' (M6, the default; needs a vehicle at the last vertex) "
            "or 'station' (per-vehicle PD, any attachment pattern)."
        ),
    ),
    trajectory: str = typer.Option(
        "hover", "--trajectory", "--traj", help="Root-vertex trajectory: 'hover' or 'lissajous'."
    ),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial position of vertex 0 as 'x,y,z'."
    ),
):
    """Simulate several quadrotors carrying one flexible hose (M6).

    ``--controller lqr`` (the default) drives the M6 hose LQR, which regulates
    the whole hose against the static trim at the root setpoint; it needs a
    vehicle at the last vertex, since that is what makes the hose flat in the
    coordinates its reference is built in. ``--controller station`` falls back
    to the per-vehicle station-keeping PDs, which have no such requirement.

    Because the LQR is the default, a pattern with no vehicle at the last
    vertex — ``-n 4 -a 0,2``, say — **exits 1** with a message naming
    ``--controller station`` rather than silently station-keeping. This
    command's headline is M6's controller, and a run that quietly swaps in a
    different control law is the more expensive surprise.

    The 'lissajous' trajectory exposes ``get_jet``, so it also switches on the
    flatness feedforward at the full ``2n + 4`` derivative depth the hose
    needs; 'hover' is a step response and needs none.

    The MJCF is generated from these knobs and cached. Raising ``--num-links``
    past 4 leaves the analytical backend's stability envelope for some
    attachment patterns and it will say so; the MuJoCo backend has no such
    limit.
    """
    scn = _build(
        sc.hose,
        backend=model,
        duration=time,
        num_links=num_links,
        attachments=_parse_ints(attachments, "0,2,4"),
        hose_length=hose_length,
        hose_mass=hose_mass,
        quad_mass=quad_mass,
        sag_ratio=sag_ratio,
        spawn_height=spawn_height,
        controller=controller,
        trajectory=trajectory,
        position=_position(position),
        render=render,
    )
    _setup_viewer(scn.model, record)
    typer.echo(f"Running hose ({scn.label}) for {scn.tf}s ...")
    scn.run()
    if controller == "lqr":
        typer.echo(
            f"  root error {scn.tracking_error():.4f} m"
            f"  (feedforward drops: {scn.model.feedforward_drops})"
        )
    _hold_viewer(scn.model)


def _fly_tethered(scn: sc.Scenario, command: str, record: str | None) -> None:
    """Shared tail of ``tethered-quad`` (M7) and ``tethered-hose`` (M8)."""
    for note in scn.notes:
        typer.echo(f"  {note}")
    _setup_viewer(scn.model, record)
    typer.echo(f"Running {command} ({scn.label}) for {scn.tf}s ...")
    try:
        scn.run()
    except RuntimeError as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(1) from err
    typer.echo(
        f"  top vehicle error {scn.tracking_error():.4f} m"
        f"  (lowest reference vertex {scn.model.reference_ground_clearance:.3f} m above the ground)"
    )
    _hold_viewer(scn.model)


@run_app.command("tethered-quad")
def tethered_quad(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    model: str = typer.Option("mujoco", "--model", "-m", help="Backend: 'mujoco' or 'base'."),
    render: bool = typer.Option(True, help="Enable visualization."),
    num_links: int | None = typer.Option(None, "--num-links", "-n", help="Tether links."),
    length: float | None = typer.Option(None, "--length", "-l", help="Tether length, m."),
    tether_mass: float | None = typer.Option(None, "--tether-mass", help="Tether mass, kg."),
    anchor_height: float | None = typer.Option(None, "--anchor-height", help="Anchor height, m."),
    standoff: float | None = typer.Option(
        None, "--standoff", help="Horizontal distance from the anchor, as a fraction of the length."
    ),
    height: float | None = typer.Option(
        None, "--height", help="Height above the anchor, as a fraction of the length."
    ),
    trajectory: str = typer.Option(
        "hold", "--trajectory", "--traj", help="'hold', 'circle' (about the anchor) or 'climb'."
    ),
    rate: float | None = typer.Option(
        None, "--rate", help="Circle rate, rad/s. Default: the largest the rule allows."
    ),
    ramp: float | None = typer.Option(
        None, "--ramp", help="Circle ramp / climb blend duration, s (default 6)."
    ),
    record: str | None = _record_option,
):
    """Simulate one quadrotor tethered to a ground anchor by an n-link cable (M7).

    The hose LQR holds the vehicle and the tether at the static trim with the
    vehicle at the given standoff and height ('hold'), flies a circle about the
    anchor started from rest by a 6 s ramp ('circle'), or blends to a higher,
    closer station over 6 s ('climb'), with the flat map's feed-forward.
    """
    scn = _build(
        sc.tethered_quad,
        backend=model,
        duration=time,
        num_links=num_links,
        length=length,
        tether_mass=tether_mass,
        anchor_height=anchor_height,
        standoff=standoff,
        height=height,
        trajectory=trajectory,
        rate=rate,
        ramp=ramp,
        render=render,
    )
    _fly_tethered(scn, "tethered-quad", record)


@run_app.command("tethered-hose")
def tethered_hose(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    model: str = typer.Option("mujoco", "--model", "-m", help="Backend: 'mujoco' or 'base'."),
    render: bool = typer.Option(True, help="Enable visualization."),
    num_quadrotors: int | None = typer.Option(None, "--quads", "-q", help="Vehicles on the hose."),
    num_links: int | None = typer.Option(
        None, "--num-links", "-n", help="Hose links (default two per vehicle)."
    ),
    length: float | None = typer.Option(None, "--length", "-l", help="Hose length, m."),
    hose_mass: float | None = typer.Option(None, "--hose-mass", help="Hose mass, kg."),
    anchor_height: float | None = typer.Option(None, "--anchor-height", help="Anchor height, m."),
    standoff: float | None = typer.Option(
        None, "--standoff", help="Top vehicle's distance from the anchor, fraction of the length."
    ),
    height: float | None = typer.Option(
        None, "--height", help="Top vehicle's height above the anchor, fraction of the length."
    ),
    trajectory: str = typer.Option(
        "hold", "--trajectory", "--traj", help="'hold', 'circle' (about the anchor) or 'climb'."
    ),
    rate: float | None = typer.Option(
        None, "--rate", help="Circle rate, rad/s. Default: the largest the rule allows."
    ),
    ramp: float | None = typer.Option(
        None, "--ramp", help="Circle ramp / climb blend duration, s (default 6)."
    ),
    record: str | None = _record_option,
):
    """Simulate several quadrotors in series on one hose pinned to a ground anchor (M8).

    The vehicles sit evenly on the line from the anchor to the top vehicle; the
    hose LQR holds them there ('hold'), flies the formation round the anchor
    ('circle') or blends it to a higher, closer station ('climb').
    """
    scn = _build(
        sc.tethered_hose,
        backend=model,
        duration=time,
        num_quadrotors=num_quadrotors,
        num_links=num_links,
        length=length,
        hose_mass=hose_mass,
        anchor_height=anchor_height,
        standoff=standoff,
        height=height,
        trajectory=trajectory,
        rate=rate,
        ramp=ramp,
        render=render,
    )
    _fly_tethered(scn, "tethered-hose", record)


@run_app.command("multi-quad-payload")
def multi_quad_payload(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    num_quads: int = typer.Option(3, "--num-quads", "-n", help="Number of quadrotors."),
    cable_length: float | None = typer.Option(
        None, "--cable-length", "-l", help="Taut cable length in metres."
    ),
    payload_mass: float | None = typer.Option(None, "--payload-mass", help="Payload mass in kg."),
    formation_radius: float | None = typer.Option(
        None, "--radius", help="Radius of the quadrotor ring in metres."
    ),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial payload position as 'x,y,z'."
    ),
):
    """Simulate nQ quadrotors carrying one cable-suspended point-mass payload (M3)."""
    scn = _build(
        sc.multi_quad_payload,
        backend="mujoco",
        duration=time,
        num_quadrotors=num_quads,
        cable_length=cable_length,
        payload_mass=payload_mass,
        formation_radius=formation_radius,
        position=_position(position),
        render=render,
    )
    _setup_viewer(scn.model, record)
    typer.echo(f"Running multi-quad-payload ({num_quads} quads) for {scn.tf}s ...")
    scn.run()
    _hold_viewer(scn.model)


run_app.command(
    "multi-quad",
    help="Same as multi-quad-payload: nQ quadrotors carrying one cable-suspended "
    "point-mass payload (M3).",
)(multi_quad_payload)


@run_app.command("multi-quad-rigidbody")
def multi_quad_rigidbody(
    time: float = typer.Option(10.0, "--time", "-t", help="Simulation duration in seconds."),
    render: bool = typer.Option(True, help="Enable visualization."),
    num_quads: int = typer.Option(4, "--num-quads", "-n", help="Number of quadrotors."),
    cable_length: float | None = typer.Option(
        None, "--cable-length", "-l", help="Taut cable length in metres."
    ),
    payload_mass: float | None = typer.Option(None, "--payload-mass", help="Payload mass in kg."),
    attachment_radius: float | None = typer.Option(
        None, "--attachment-radius", "-a", help="Radius of the attachment ring on the payload."
    ),
    formation_radius: float | None = typer.Option(
        None,
        "--radius",
        help="Horizontal offset of each quadrotor from its own attachment point "
        "(default: 0.5 m, grown so that neighbouring airframes clear each other).",
    ),
    spawn_height: float | None = typer.Option(
        None,
        "--height",
        "-z",
        help="Height of the quadrotor ring in metres (default: the payload spawns at 1 m).",
    ),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial payload position as 'x,y,z'."
    ),
):
    """Simulate nQ >= 1 quadrotors carrying a rigid-body payload (geometric cascade).

    Runs :class:`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyMujoco`.
    The MJCF is generated from these knobs and cached, so the formation starts
    taut whatever they are set to.
    """
    scn = _build(
        sc.multi_quad_rigidbody,
        backend="mujoco",
        duration=time,
        num_quadrotors=num_quads,
        cable_length=cable_length,
        payload_mass=payload_mass,
        attachment_radius=attachment_radius,
        formation_radius=formation_radius,
        spawn_height=spawn_height,
        position=_position(position),
        render=render,
    )
    _setup_viewer(scn.model, record)
    typer.echo(
        f"Running multi-quad rigid body ({scn.model.num_quadrotors} quads) for {scn.tf}s ..."
    )
    scn.run()
    _hold_viewer(scn.model)


run_app.command(
    "multi-quad-rigid",
    help="Same as multi-quad-rigidbody: nQ quadrotors carrying one rigid-body payload (M4).",
)(multi_quad_rigidbody)


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
    scn = _build(
        sc.cspayload_fleet,
        duration=time,
        num_agents=num_agents,
        demo=demo,
        cable_links=cable_links,
        same_start=same_start,
        render=render,
    )
    _setup_viewer(scn.model, record, trail=trail)
    typer.echo(f"Running cspayload-fleet ({scn.label}) for {scn.tf}s ...")
    scn.run()


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

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    scn = _build(
        sc.fleet,
        duration=time,
        num_quadrotors=num_quads,
        demo=demo,
        position=_position(position),
        render=render,
    )
    _setup_viewer(scn.model, record, trail=trail)
    typer.echo(f"Running fleet ({scn.label}) for {scn.tf}s ...")
    scn.run()
