#!/usr/bin/env python3
"""L1 adaptive position control on a quadrotor, next to a plain PD baseline.

Implements:
Geometric $L_1$ Adaptive Attitude Control for a Quadrotor Unmanned Aerial Vehicle.
pdf: https://doi.org/10.1115/1.4045558

Both runs carry an unmodeled 0.25 kg payload: the plant is that much heavier
than the mass their position controllers were handed, so the loop has to make
up the missing hover thrust out of feedback alone and settles with a
steady-state sag.  The example prints the sag each controller is left with.

`PositionL1Controller` is the PD baseline plus piecewise-constant L1
adaptation, so it estimates the missing weight and removes the sag.

Usage: python examples/quadrotor_geometric_l1.py
"""

import numpy as np

from udaan.models.quadrotor import QuadrotorBase

TARGET = np.array([1.0, 1.0, 2.0])
START = np.array([0.0, 0.0, 0.2])
UNMODELED_MASS = 0.25  # kg of payload the controller is never told about


def fly(controller_factory):
    """Simulate one quadrotor whose position controller under-estimates the mass."""
    mdl = QuadrotorBase()
    setpoint = lambda t: (TARGET, np.zeros(3), np.zeros(3))
    # Controllers are pluggable at runtime — this is the whole integration point.
    mdl.position_controller = controller_factory(mass=mdl.mass - UNMODELED_MASS, setpoint=setpoint)
    mdl.simulate(tf=25.0, position=START)
    return np.linalg.norm(mdl.state.position - TARGET)


def main():
    from udaan.control.quadrotor import PositionL1Controller, PositionPDController

    l1_err = fly(PositionL1Controller)
    pd_err = fly(PositionPDController)

    print(f"Unmodeled mass {UNMODELED_MASS} kg, hover target {TARGET}")
    print(f"  PD position error {pd_err:.4f} m")
    print(f"  L1 position error {l1_err:.4f} m")
    assert pd_err < 1.0, "the PD run diverged"
    assert l1_err < 0.01 * pd_err, "L1 should remove the unmodeled-mass sag"


if __name__ == "__main__":
    main()
