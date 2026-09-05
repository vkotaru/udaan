#!/usr/bin/env python3
"""M1 — a single quadrotor under geometric SE(3) control.

Flies the analytical model (no renderer, no MuJoCo) from an off-target start to
a commanded hover point, then reports how close it got.

Usage: python examples/quadrotor.py
"""

import numpy as np

from udaan.models.quadrotor import QuadrotorBase

TARGET = np.array([1.0, -0.5, 1.5])
START = np.array([0.0, 0.0, 0.2])


def main():
    mdl = QuadrotorBase()

    # Controllers are pluggable at runtime; the setpoint protocol is
    # get(t) -> (position, velocity, acceleration).
    mdl.position_controller.setpoint = lambda t: (TARGET, np.zeros(3), np.zeros(3))

    mdl.simulate(tf=8.0, position=START)

    err = np.linalg.norm(mdl.state.position - TARGET)
    print(f"M1 quadrotor: start {START}  ->  target {TARGET}")
    print(f"  final position {mdl.state.position}")
    print(f"  position error {err:.4f} m")
    print(f"  speed          {np.linalg.norm(mdl.state.velocity):.4f} m/s")
    assert err < 0.05, f"quadrotor did not converge: {err:.4f} m"


if __name__ == "__main__":
    main()
