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
    """PD (clean) vs PD (disturbed) vs L1 (disturbed)."""
    from udaan.control.quadrotor import PositionL1Controller

    traj = lambda t: (np.array([0.0, 0.0, 2.0]), np.zeros(3), np.zeros(3))

    # quad 0 (red): PD, clean — reference baseline
    fleet[0].position_controller.setpoint = traj

    # quad 1 (blue): PD, disturbed — shows drift
    fleet[1].position_controller.setpoint = traj

    # quad 2 (green): L1 adaptive, same disturbance — adapts
    m = fleet[2].mass
    fleet[2].position_controller = PositionL1Controller(mass=m, setpoint=traj)


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

    for i, scale in enumerate(gain_scales):
        fleet[i].position_controller = PositionPDController(
            mass=fleet[i].mass,
            setpoint=traj,
            kp=scale * np.array([4.0, 4.0, 8.0]),
            kd=scale * np.array([3.0, 3.0, 6.0]),
        )


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
