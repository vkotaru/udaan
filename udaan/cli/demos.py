"""Fleet demo presets.

Each demo has:
- config(): returns {"num_quadrotors": N, "disturbances": {...}}
- configure(fleet): sets up controllers on the fleet
"""

import numpy as np


def _hold(target):
    """A constant setpoint at ``target``: ``t -> (target, 0, 0)``."""
    target = np.asarray(target, dtype=float)
    return lambda t: (target, np.zeros(3), np.zeros(3))


def lane_setpoints(fleet, target):
    """One constant setpoint per vehicle, ``target`` translated into its own x-lane.

    Each vehicle spawns at its lane offset ``fleet._offsets[i]`` along x, so a
    shared absolute target would fly every vehicle into the same point; this
    gives vehicle ``i`` the target ``target + (offset_i, 0, 0)``.
    """
    return [_hold(np.asarray(target, dtype=float) + [dx, 0.0, 0.0]) for dx in fleet._offsets]


def _l1_comparison_config():
    return {
        "num_quadrotors": 3,
        "disturbances": {
            1: {"mass": 0.25, "offset": [0.2, 0.2, -0.05]},
            2: {"mass": 0.25, "offset": [0.2, 0.2, -0.05]},
        },
    }


def _l1_comparison_configure(fleet):
    """Nominal PD vs full L1 (position + attitude) with disturbance."""
    from udaan.control.quadrotor import PositionL1Controller
    from udaan.control.quadrotor.geometric_l1_attitude import GeometricL1AttitudeController

    lanes = lane_setpoints(fleet, [0.0, 0.0, 1.0])

    # quad 0: PD + GeomAtt, NO disturbance — reference baseline
    fleet[0].position_controller.setpoint = lanes[0]
    fleet._labels[0] = "PD+Geom (clean)"

    # quad 1: PD + GeomAtt, disturbed — shows drift
    fleet[1].position_controller.setpoint = lanes[1]
    fleet._labels[1] = "PD+Geom (disturbed)"

    # quad 2: PD_L1 + GeomL1Att, disturbed — full L1
    m = fleet[2].mass
    fleet[2].position_controller = PositionL1Controller(mass=m, setpoint=lanes[2])
    fleet[2].attitude_controller = GeometricL1AttitudeController(mass=m, inertia=fleet[2].inertia)
    fleet._labels[2] = "L1+GeomL1 (disturbed)"


def _gain_sweep_config():
    return {
        "num_quadrotors": 4,
        "disturbances": {},
    }


def _gain_sweep_configure(fleet):
    """The default PD position gains times 0.25, 0.5, 1 and 2 (soft -> aggressive)."""
    from udaan.control.quadrotor import PositionPDController
    from udaan.core.defaults import DEFAULT_POS_KD, DEFAULT_POS_KP

    lanes = lane_setpoints(fleet, [0.0, 0.0, 2.0])
    gain_scales = [0.25, 0.5, 1.0, 2.0]
    labels = ["very soft", "soft", "nominal", "aggressive"]

    for i, (scale, label) in enumerate(zip(gain_scales, labels, strict=True)):
        fleet[i].position_controller = PositionPDController(
            mass=fleet[i].mass,
            setpoint=lanes[i],
            kp=scale * DEFAULT_POS_KP,
            kd=scale * DEFAULT_POS_KD,
        )
        fleet._labels[i] = f"PD ({label})"


DEMOS = {
    "l1-comparison": {
        "config": _l1_comparison_config,
        "configure": _l1_comparison_configure,
        "description": "PD (clean) vs PD (disturbed) vs L1 (disturbed)",
    },
    "gain-sweep": {
        "config": _gain_sweep_config,
        "configure": _gain_sweep_configure,
        "description": "PD with increasing gains (soft -> aggressive)",
    },
}


# ══════════════════════════════════════════════════════════════════════
# Quad-cable-payload fleet demos
# ══════════════════════════════════════════════════════════════════════
#
# Schema:
#   config():                         returns {"num_agents": N}
#   configure(fleet, same_start):     sets per-agent setpoint/gains/label,
#                                     returns list of absolute per-agent
#                                     payload start positions.
#
# Default: per-agent start and target are translated by `fleet._offsets[i]`
# so every agent runs the same maneuver in its own X-lane. With
# same_start=True, all agents share one absolute start/target (they overlap
# visually but inter-agent contacts are disabled in the fleet MJCF).


def _cspayload_same_gains_config():
    return {"num_agents": 2}


def _cspayload_same_gains_configure(fleet, same_start=False):
    """All agents share the default gains; identical maneuver per lane.

    If same_start=True, all agents start at the same absolute position
    (visually overlap; diverge only as dynamics differ).
    """
    start_delta = np.array([1.0, 1.0, 0.5])
    target_delta = np.array([0.0, 0.0, 1.5])
    starts = []
    for i in range(fleet.nQ):
        offset = (
            np.array([0.0, 0.0, 0.0]) if same_start else np.array([fleet._offsets[i], 0.0, 0.0])
        )
        target = target_delta + offset
        fleet[i]._payload_controller.setpoint = _hold(target)
        fleet._labels[i] = "default gains"
        starts.append(start_delta + offset)
    return starts


def _cspayload_gain_sweep_config():
    return {"num_agents": 4}


def _cspayload_gain_sweep_configure(fleet, same_start=False):
    """Cable kp/kd scale sweep: each agent's own cable gains times 0.5, 1, 1.5, 2.

    By default each agent runs in its own X-lane with identical maneuver.
    If same_start=True, all agents start at the same absolute position and
    fly to the same target — divergence is due to gain differences alone.
    """
    from udaan.control.quadrotor_cspayload import QuadCSPayloadController

    start_delta = np.array([1.0, 0.0, 0.5])
    target_delta = np.array([0.0, 0.0, 1.5])
    scales = [0.5, 1.0, 1.5, 2.0][: fleet.nQ]
    starts = []
    for i in range(fleet.nQ):
        offset = (
            np.array([0.0, 0.0, 0.0]) if same_start else np.array([fleet._offsets[i], 0.0, 0.0])
        )
        target = target_delta + offset
        # Fixed constructor gains survive a later `sync_from_model`.
        cable_kp, cable_kd = fleet[i]._payload_controller.cable_gains
        controller = QuadCSPayloadController(
            cable_kp=scales[i] * cable_kp, cable_kd=scales[i] * cable_kd, setpoint=_hold(target)
        )
        controller.sync_from_model(fleet[i])
        fleet[i]._payload_controller = controller
        fleet._labels[i] = f"cable × {scales[i]:.1f}"
        starts.append(start_delta + offset)
    return starts


CSPAYLOAD_DEMOS = {
    "same-gains": {
        "config": _cspayload_same_gains_config,
        "configure": _cspayload_same_gains_configure,
        "description": "Two agents, identical default gains, identical per-lane maneuver",
    },
    "gain-sweep": {
        "config": _cspayload_gain_sweep_config,
        "configure": _cspayload_gain_sweep_configure,
        "description": "Four agents, cable kp/kd × [0.5, 1.0, 1.5, 2.0], identical per-lane maneuver",
    },
}
