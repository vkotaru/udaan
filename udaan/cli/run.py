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
        "hover",
        "--trajectory",
        "--traj",
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

        # Lift the trajectory center so the z-oscillation (±1.5 m) stays above ground.
        center = x0 + np.array([0.0, 0.0, 2.0])
        traj = CrazyTrajectory(tf=time, center=center)
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

    import numpy as np

    from . import parse_vec

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    opts = {
        "num_links": links,
        "cable_length": cable_length,
        "cable_mass": cable_mass,
        "payload_mass": payload_mass,
    }
    opts = {k: v for k, v in opts.items() if v is not None}

    if model == "mujoco":
        from udaan.models.quadrotor_flexcable import QuadrotorFlexCableMujoco

        mdl = QuadrotorFlexCableMujoco(render=render, verbose=verbose, **opts)
    else:
        from udaan.models.quadrotor_flexcable import QuadrotorFlexCableBase

        mdl = QuadrotorFlexCableBase(verbose=verbose, **opts)

    x0 = parse_vec(position, default=np.array([0.0, 0.0, 1.0]))
    if trajectory == "lissajous":
        from udaan.utils.trajectory import CrazyTrajectory

        # Gentle enough that no link leaves the hanging branch, which is what
        # the flat map refuses; the chain then feeds forward on every sample.
        traj = CrazyTrajectory(center=x0, ax=0.4, ay=0.35, az=0.25, f1=0.18, f2=0.13, f3=0.09)
        mdl.payload_controller.setpoint = traj.get
        x0 = traj.get(0.0)[0]
    else:
        # A step, not a fixed point at the spawn.  Commanding the payload to
        # exactly where reset() puts it — hanging straight down, at rest — is
        # already the closed-loop equilibrium, so the run measures nothing and
        # reports 0.0000 m however good or bad the gain is.  Offsetting the
        # target makes the demo a step response the printed error can score.
        target = x0 + np.array([0.5, 0.3, 0.2])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))

    _setup_recording(mdl, record)
    typer.echo(
        f"Running quad-flexcable ({model}, {mdl.num_links} links, {trajectory}) for {time}s ..."
    )
    mdl.simulate(tf=time, payload_position=x0)
    typer.echo(
        f"  payload error {np.linalg.norm(mdl.state.payload_position - mdl.payload_controller.setpoint(mdl.t)[0]):.4f} m"
        f"  (feedforward drops: {mdl.feedforward_drops})"
    )
    _hold_viewer(mdl)


@run_app.command("multi-quad")
def multi_quad(
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
    """Simulate multiple quadrotors with cable-suspended pointmass payload."""
    import numpy as np

    from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadMujoco

    from . import parse_vec

    x0 = parse_vec(position, default=np.array([-1.0, 2.0, 0.5]))
    opts = {
        "cable_lengths": cable_length,
        "payload_mass": payload_mass,
        "formation_radius": formation_radius,
    }
    opts = {k: v for k, v in opts.items() if v is not None}
    try:
        mdl = MultiQuadCsPayloadMujoco(render=render, num_quadrotors=num_quads, **opts)
    except ValueError as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(1) from err
    _setup_recording(mdl, record)
    typer.echo(f"Running multi-quad ({num_quads} quads) for {time}s ...")
    mdl.simulate(tf=time, payload_position=x0)
    _hold_viewer(mdl)


@run_app.command("multi-quad-rigid")
def multi_quad_rigid(
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
        help="Horizontal offset of each quadrotor from its own attachment point.",
    ),
    spawn_height: float | None = typer.Option(
        None, "--height", "-z", help="Height of the quadrotor ring in metres."
    ),
    record: str | None = _record_option,
    position: str | None = typer.Option(
        None, "--position", "-p", help="Initial payload position as 'x,y,z'."
    ),
):
    """Simulate multiple quadrotors with cable-suspended rigid-body payload.

    The MJCF is generated from these knobs and cached, so the formation starts
    taut whatever they are set to.
    """

    from udaan.models.multi_quad_rigidbody import MultiQuadRigidbodyMujoco

    from . import parse_vec

    opts = {
        "cable_lengths": cable_length,
        "payload_mass": payload_mass,
        "attachment_radius": attachment_radius,
        "formation_radius": formation_radius,
        "spawn_height": spawn_height,
    }
    opts = {k: v for k, v in opts.items() if v is not None}
    try:
        mdl = MultiQuadRigidbodyMujoco(render=render, num_quadrotors=num_quads, **opts)
    # ValueError covers the geometry the generator refuses to build; RuntimeError
    # covers the load-time checks (`_read_params_from_mjcf`, `_check_spawn_is_taut`)
    # that exist precisely for a stale cached or hand-edited asset — which is a
    # user-facing condition with a stated remedy, not a traceback.
    except (ValueError, RuntimeError) as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(1) from err
    # Absolute, exactly as `multi-quad` reads it. The retired model took the
    # same flag under the same help text as a relative offset. Defaulting to
    # the MJCF's own spawn keeps the formation taut at t=0.
    x0 = parse_vec(position, default=mdl.state.payload_position.copy())
    _setup_recording(mdl, record)
    typer.echo(f"Running multi-quad rigid ({mdl.num_quadrotors} quads) for {time}s ...")
    mdl.simulate(tf=time, payload_position=x0)
    _hold_viewer(mdl)


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
    vertex — ``-n 4 -a 0,2``, say — now **exits 1** with a message naming
    ``--controller station``, where before it silently station-kept. That is
    deliberate: this command's headline is M6's controller, and a run that
    quietly swaps in a different control law is the more expensive surprise.

    The 'lissajous' trajectory exposes ``get_jet``, so it also switches on the
    flatness feedforward at the full ``2n + 4`` derivative depth the hose
    needs; 'hover' is a step response and needs none.

    The MJCF is generated from these knobs and cached. Raising ``--num-links``
    past 4 leaves the analytical backend's stability envelope for some
    attachment patterns and it will say so; the MuJoCo backend has no such
    limit.
    """
    import numpy as np

    from . import parse_vec

    if controller not in ("lqr", "station"):
        typer.echo(f"Unknown controller '{controller}'. Expected 'lqr' or 'station'.", err=True)
        raise typer.Exit(1)
    # Unvalidated, a misspelt --traj silently fell through to the hover branch
    # and printed a plausible-looking tracking error for a run that was never
    # the one asked for.
    if trajectory not in ("hover", "lissajous"):
        typer.echo(f"Unknown trajectory '{trajectory}'. Expected 'hover' or 'lissajous'.", err=True)
        raise typer.Exit(1)

    opts = {
        "hose_length": hose_length,
        "hose_mass": hose_mass,
        "masses": quad_mass,
        "sag_ratio": sag_ratio,
        "spawn_height": spawn_height,
    }
    opts = {k: v for k, v in opts.items() if v is not None}
    if attachments is not None:
        try:
            opts["attachments"] = tuple(int(v) for v in attachments.split(","))
        except ValueError:
            typer.echo(f"Invalid attachments: '{attachments}'. Expected e.g. '0,2,4'.", err=True)
            raise typer.Exit(1) from None
    if controller == "lqr":
        # The LQR emits a per-vehicle wrench; the station-keeping PDs emit
        # thrust vectors. The two pipelines take incompatible shapes, so the
        # choice has to be made at construction.
        opts["input"] = "wrench"

    if model == "mujoco":
        from udaan.models.multi_quad_hose import MultiQuadHoseMujoco as Model
    else:
        from udaan.models.multi_quad_hose import MultiQuadHoseBase as Model

        opts.pop("spawn_height", None)
        render = False

    try:
        mdl = Model(render=render, num_links=num_links, **opts)
    # ValueError covers the spawn geometry the generator refuses to build and
    # the parameter guards; RuntimeError covers the load-time checks that exist
    # for a stale cached or hand-edited asset. Both are user-facing conditions
    # with a stated remedy, not tracebacks.
    except (ValueError, RuntimeError) as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(1) from err

    default_root = np.asarray(mdl.state.root_position).copy()
    if model != "mujoco" and position is None:
        default_root = np.array([0.0, 0.0, 2.0])
    x0 = parse_vec(position, default=default_root)

    if controller == "lqr":
        if mdl.hose_controller is None:
            typer.echo(
                f"Error: attachments {mdl.attachments} put no vehicle at the last vertex "
                f"({mdl.num_links}), so the hose is not differentially flat in the "
                "coordinates the LQR's reference is built in. Add that vertex, or run "
                "--controller station.",
                err=True,
            )
            raise typer.Exit(1)
        if trajectory == "lissajous":
            from udaan.utils.trajectory import CrazyTrajectory

            # Gentle: the constant gain is solved at the static trim and the
            # linearisation moves ~2 % over a reference this size. See the
            # drift table in udaan.control.multi_quad_hose.hose_controller.
            traj = CrazyTrajectory(center=x0, ax=0.3, ay=0.24, az=0.15, f1=0.2, f2=0.16, f3=0.12)
            mdl.hose_controller.setpoint = traj.get
            x0 = traj.get(0.0)[0]
        else:
            # A step, not a fixed point at the spawn: commanding the root to
            # where reset() puts it would score 0.0000 m however good or bad
            # the gain is.
            target = x0 + np.array([0.3, 0.2, 0.2])
            mdl.hose_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))

    _setup_recording(mdl, record)
    typer.echo(
        f"Running hose ({model}, {mdl.num_links} links, "
        f"{mdl.num_quadrotors} quads at {mdl.attachments}, {controller}) for {time}s ..."
    )
    mdl.simulate(tf=time, root_position=x0)
    if controller == "lqr":
        typer.echo(
            f"  root error "
            f"{np.linalg.norm(np.asarray(mdl.state.root_position) - mdl.hose_controller.setpoint(mdl.t)[0]):.4f} m"
            f"  (feedforward drops: {mdl.feedforward_drops})"
        )
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
