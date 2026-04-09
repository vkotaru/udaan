"""Fleet demo presets.

Each demo has:
- config(): returns {"num_quadrotors": N, "disturbances": {...}}
- configure(fleet): sets up controllers on the fleet
"""

import numpy as np


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

    traj = lambda t: (np.array([0.0, 0.0, 2.0]), np.zeros(3), np.zeros(3))

    # quad 0 (red): PD + GeomAtt, NO disturbance — reference baseline
    fleet[0].position_controller.setpoint = traj
    fleet._labels[0] = "PD+Geom (clean)"

    # quad 1 (blue): PD + GeomAtt, disturbed — shows drift
    fleet[1].position_controller.setpoint = traj
    fleet._labels[1] = "PD+Geom (disturbed)"

    # quad 2 (green): PD_L1 + GeomL1Att, disturbed — full L1
    m = fleet[2].mass
    fleet[2].position_controller = PositionL1Controller(mass=m, setpoint=traj)
    fleet[2].attitude_controller = GeometricL1AttitudeController(mass=m, inertia=fleet[2].inertia)
    fleet._labels[2] = "L1+GeomL1 (disturbed)"


def _gain_sweep_config():
    return {
        "num_quadrotors": 4,
        "disturbances": {},
    }


def _gain_sweep_configure(fleet):
    """PD with increasing gains (soft -> aggressive)."""
    from udaan.control.quadrotor import PositionPDController

    traj = lambda t: (np.array([0.0, 0.0, 2.0]), np.zeros(3), np.zeros(3))
    gain_scales = [0.25, 0.5, 1.0, 2.0]
    labels = ["very soft", "soft", "nominal", "aggressive"]

    for i, (scale, label) in enumerate(zip(gain_scales, labels, strict=True)):
        fleet[i].position_controller = PositionPDController(
            mass=fleet[i].mass,
            setpoint=traj,
            kp=scale * np.array([4.0, 4.0, 8.0]),
            kd=scale * np.array([3.0, 3.0, 6.0]),
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
