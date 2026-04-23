"""Quadrotor simulation plotting — Bokeh interactive time-series."""

import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure


def record_quadrotor_state(mdl, tf, target):
    """Simulate a QuadrotorBase and record full state history.

    Args:
        mdl: QuadrotorBase instance (already reset to initial state).
        tf: simulation duration in seconds.
        target: target position (3-vector), used for error computation.

    Returns:
        dict of numpy arrays keyed by signal name.
    """
    from ...manif import Rot2Eul

    h = {
        "t": [],
        "x": [],
        "y": [],
        "z": [],
        "vx": [],
        "vy": [],
        "vz": [],
        "roll": [],
        "pitch": [],
        "yaw": [],
        "wx": [],
        "wy": [],
        "wz": [],
        "f": [],
        "Mx": [],
        "My": [],
        "Mz": [],
        "pos_err": [],
    }

    while mdl.t < tf:
        rpy = np.degrees(Rot2Eul(np.asarray(mdl.state.orientation)))
        h["t"].append(mdl.t)
        h["x"].append(mdl.state.position[0])
        h["y"].append(mdl.state.position[1])
        h["z"].append(mdl.state.position[2])
        h["vx"].append(mdl.state.velocity[0])
        h["vy"].append(mdl.state.velocity[1])
        h["vz"].append(mdl.state.velocity[2])
        h["roll"].append(rpy[0])
        h["pitch"].append(rpy[1])
        h["yaw"].append(rpy[2])
        h["wx"].append(float(mdl.state.angular_velocity[0]))
        h["wy"].append(float(mdl.state.angular_velocity[1]))
        h["wz"].append(float(mdl.state.angular_velocity[2]))
        h["pos_err"].append(np.linalg.norm(mdl.state.position - target))

        u = mdl._pos_controller.compute(mdl.t, (mdl.state.position, mdl.state.velocity))
        wrench = mdl._repackage_input(u)
        h["f"].append(wrench[0])
        h["Mx"].append(wrench[1])
        h["My"].append(wrench[2])
        h["Mz"].append(wrench[3])

        mdl.step(u)

    return {k: np.array(v) for k, v in h.items()}


def plot_quadrotor_simulation(history, target=None):
    """Create Bokeh grid of time-series plots from recorded history.

    Args:
        history: dict from record_quadrotor_state().
        target: optional target position for reference lines.

    Returns:
        Bokeh gridplot layout. Call bokeh.io.show(layout) to display.
    """
    t = history["t"]
    plots = []

    def _fig(title, ylabel):
        p = figure(
            title=title,
            x_axis_label="time [s]",
            y_axis_label=ylabel,
            width=500,
            height=250,
            x_range=plots[0].x_range if plots else None,
        )
        return p

    # Position
    p = figure(title="Position", x_axis_label="time [s]", y_axis_label="m", width=500, height=250)
    p.line(t, history["x"], legend_label="x", color="red")
    p.line(t, history["y"], legend_label="y", color="green")
    p.line(t, history["z"], legend_label="z", color="blue")
    if target is not None:
        for i, c in enumerate(["red", "green", "blue"]):
            p.line([t[0], t[-1]], [target[i], target[i]], line_dash="dashed", color=c, alpha=0.4)
    p.legend.click_policy = "hide"
    plots.append(p)

    # Velocity
    p = _fig("Velocity", "m/s")
    p.line(t, history["vx"], legend_label="vx", color="red")
    p.line(t, history["vy"], legend_label="vy", color="green")
    p.line(t, history["vz"], legend_label="vz", color="blue")
    p.legend.click_policy = "hide"
    plots.append(p)

    # Attitude
    p = _fig("Attitude (RPY)", "deg")
    p.line(t, history["roll"], legend_label="roll", color="red")
    p.line(t, history["pitch"], legend_label="pitch", color="green")
    p.line(t, history["yaw"], legend_label="yaw", color="blue")
    p.legend.click_policy = "hide"
    plots.append(p)

    # Angular velocity
    p = _fig("Angular Velocity", "rad/s")
    p.line(t, history["wx"], legend_label="Ωx", color="red")
    p.line(t, history["wy"], legend_label="Ωy", color="green")
    p.line(t, history["wz"], legend_label="Ωz", color="blue")
    p.legend.click_policy = "hide"
    plots.append(p)

    # Thrust
    p = _fig("Thrust", "N")
    p.line(t, history["f"], color="black")
    plots.append(p)

    # Torque
    p = _fig("Torque", "Nm")
    p.line(t, history["Mx"], legend_label="Mx", color="red")
    p.line(t, history["My"], legend_label="My", color="green")
    p.line(t, history["Mz"], legend_label="Mz", color="blue")
    p.legend.click_policy = "hide"
    plots.append(p)

    # Position error
    p = _fig("Position Error", "m")
    p.line(t, history["pos_err"], color="black")
    plots.append(p)

    # XY trajectory
    p = figure(
        title="XY Trajectory",
        x_axis_label="x [m]",
        y_axis_label="y [m]",
        width=500,
        height=250,
        match_aspect=True,
    )
    p.line(history["x"], history["y"], color="blue")
    p.scatter([history["x"][0]], [history["y"][0]], color="green", size=8, legend_label="start")
    if target is not None:
        p.scatter([target[0]], [target[1]], color="red", size=8, legend_label="target")
    p.legend.click_policy = "hide"
    plots.append(p)

    return gridplot([plots[i : i + 2] for i in range(0, len(plots), 2)])
