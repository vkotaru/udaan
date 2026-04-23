"""Quadrotor with cable-suspended payload — Bokeh plotting and history capture."""

import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure


def record_quadrotor_cspayload_state(mdl, tf, target):
    """Simulate a QuadrotorCsPayload model and record full state history.

    Args:
        mdl: QuadrotorCsPayloadBase instance (already reset to initial state).
        tf: simulation duration in seconds.
        target: payload target position (3-vector), used for error computation.

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
        "px": [],
        "py": [],
        "pz": [],
        "pvx": [],
        "pvy": [],
        "pvz": [],
        "qx": [],
        "qy": [],
        "qz": [],
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
        "payload_err": [],
    }

    while mdl.t < tf:
        rpy = np.degrees(Rot2Eul(np.asarray(mdl.state.orientation)))
        q = np.asarray(mdl.state.cable_attitude)
        h["t"].append(mdl.t)
        h["x"].append(mdl.state.position[0])
        h["y"].append(mdl.state.position[1])
        h["z"].append(mdl.state.position[2])
        h["vx"].append(mdl.state.velocity[0])
        h["vy"].append(mdl.state.velocity[1])
        h["vz"].append(mdl.state.velocity[2])
        h["px"].append(mdl.state.payload_position[0])
        h["py"].append(mdl.state.payload_position[1])
        h["pz"].append(mdl.state.payload_position[2])
        h["pvx"].append(mdl.state.payload_velocity[0])
        h["pvy"].append(mdl.state.payload_velocity[1])
        h["pvz"].append(mdl.state.payload_velocity[2])
        h["qx"].append(q[0])
        h["qy"].append(q[1])
        h["qz"].append(q[2])
        h["roll"].append(rpy[0])
        h["pitch"].append(rpy[1])
        h["yaw"].append(rpy[2])
        h["wx"].append(float(mdl.state.angular_velocity[0]))
        h["wy"].append(float(mdl.state.angular_velocity[1]))
        h["wz"].append(float(mdl.state.angular_velocity[2]))
        h["payload_err"].append(np.linalg.norm(mdl.state.payload_position - target))

        u = mdl._payload_controller.compute(mdl.t, mdl.state)
        wrench = mdl._repackage_input(u)
        h["f"].append(wrench[0])
        h["Mx"].append(wrench[1])
        h["My"].append(wrench[2])
        h["Mz"].append(wrench[3])

        mdl.step(u)

    return {k: np.array(v) for k, v in h.items()}


def plot_quadrotor_cspayload_simulation(history, target=None):
    """Create a Bokeh grid of time-series plots for cspayload simulation history."""
    t = history["t"]
    plots = []

    def _fig(title, ylabel):
        return figure(
            title=title,
            x_axis_label="time [s]",
            y_axis_label=ylabel,
            width=500,
            height=250,
            x_range=plots[0].x_range if plots else None,
        )

    p = figure(
        title="Payload Position", x_axis_label="time [s]", y_axis_label="m", width=500, height=250
    )
    p.line(t, history["px"], legend_label="px", color="red")
    p.line(t, history["py"], legend_label="py", color="green")
    p.line(t, history["pz"], legend_label="pz", color="blue")
    if target is not None:
        for i, c in enumerate(["red", "green", "blue"]):
            p.line([t[0], t[-1]], [target[i], target[i]], line_dash="dashed", color=c, alpha=0.4)
    p.legend.click_policy = "hide"
    plots.append(p)

    p = _fig("Quadrotor Position", "m")
    p.line(t, history["x"], legend_label="x", color="red")
    p.line(t, history["y"], legend_label="y", color="green")
    p.line(t, history["z"], legend_label="z", color="blue")
    p.legend.click_policy = "hide"
    plots.append(p)

    p = _fig("Payload Velocity", "m/s")
    p.line(t, history["pvx"], legend_label="vx", color="red")
    p.line(t, history["pvy"], legend_label="vy", color="green")
    p.line(t, history["pvz"], legend_label="vz", color="blue")
    p.legend.click_policy = "hide"
    plots.append(p)

    p = _fig("Cable Attitude q", "-")
    p.line(t, history["qx"], legend_label="qx", color="red")
    p.line(t, history["qy"], legend_label="qy", color="green")
    p.line(t, history["qz"], legend_label="qz", color="blue")
    p.legend.click_policy = "hide"
    plots.append(p)

    p = _fig("Quadrotor Attitude (RPY)", "deg")
    p.line(t, history["roll"], legend_label="roll", color="red")
    p.line(t, history["pitch"], legend_label="pitch", color="green")
    p.line(t, history["yaw"], legend_label="yaw", color="blue")
    p.legend.click_policy = "hide"
    plots.append(p)

    p = _fig("Angular Velocity", "rad/s")
    p.line(t, history["wx"], legend_label="Ωx", color="red")
    p.line(t, history["wy"], legend_label="Ωy", color="green")
    p.line(t, history["wz"], legend_label="Ωz", color="blue")
    p.legend.click_policy = "hide"
    plots.append(p)

    p = _fig("Thrust", "N")
    p.line(t, history["f"], color="black")
    plots.append(p)

    p = _fig("Torque", "Nm")
    p.line(t, history["Mx"], legend_label="Mx", color="red")
    p.line(t, history["My"], legend_label="My", color="green")
    p.line(t, history["Mz"], legend_label="Mz", color="blue")
    p.legend.click_policy = "hide"
    plots.append(p)

    p = _fig("Payload Position Error", "m")
    p.line(t, history["payload_err"], color="black")
    plots.append(p)

    p = figure(
        title="Payload XY Trajectory",
        x_axis_label="x [m]",
        y_axis_label="y [m]",
        width=500,
        height=250,
        match_aspect=True,
    )
    p.line(history["px"], history["py"], color="blue", legend_label="payload")
    p.line(history["x"], history["y"], color="orange", legend_label="quadrotor")
    p.scatter([history["px"][0]], [history["py"][0]], color="green", size=8, legend_label="start")
    if target is not None:
        p.scatter([target[0]], [target[1]], color="red", size=8, legend_label="target")
    p.legend.click_policy = "hide"
    plots.append(p)

    return gridplot([plots[i : i + 2] for i in range(0, len(plots), 2)])
