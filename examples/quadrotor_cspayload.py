#!/usr/bin/env python3
"""M2 — one quadrotor carrying a point-mass payload on a taut cable.

The cascade regulates the *payload*: the payload controller picks a cable
tension, the cable loop turns it into a thrust direction, and the attitude loop
flies the airframe there.  Runs on the analytical model, headless.

Usage: python examples/quadrotor_cspayload.py
"""

import numpy as np

from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadBase

TARGET = np.array([0.5, -0.5, 1.5])
START = np.array([0.0, 0.0, 1.0])


def main():
    mdl = QuadrotorCsPayloadBase()
    mdl.payload_controller.setpoint = lambda t: (TARGET, np.zeros(3), np.zeros(3))

    mdl.simulate(tf=10.0, payload_position=START)

    state = mdl.state
    err = np.linalg.norm(state.payload_position - TARGET)
    # The cable hangs below the airframe, so q points roughly down at rest.
    cable = np.asarray(state.cable_attitude)
    print(f"M2 quad + cable-suspended payload: start {START}  ->  target {TARGET}")
    print(f"  final payload  {state.payload_position}")
    print(f"  payload error  {err:.4f} m")
    print(f"  cable attitude {cable}  (|q| = {np.linalg.norm(cable):.4f})")
    assert err < 0.1, f"payload did not converge: {err:.4f} m"


if __name__ == "__main__":
    main()
