#!/usr/bin/env python3
"""M5 — one quadrotor carrying a payload on a *flexible* n-link cable.

The cable is discretised into ``n`` rigid links joined by ideal spherical
joints, so it can bend and swing rather than staying straight.  The payload LQR
regulates the far end of that chain.  Analytical model, headless.

Usage: python examples/quadrotor_flexcable.py
"""

import numpy as np

from udaan.models.quadrotor_flexcable import QuadrotorFlexCableBase

NUM_LINKS = 3
STEP = np.array([0.4, 0.25, 0.15])
START = np.array([0.0, 0.0, 1.0])


def main():
    mdl = QuadrotorFlexCableBase(num_links=NUM_LINKS)

    # A step, not the spawn: commanding the payload to exactly where reset()
    # puts it is already the closed-loop equilibrium, so it would score 0.0000 m
    # however good or bad the gain is.
    target = START + STEP
    mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))

    mdl.simulate(tf=12.0, payload_position=START)

    state = mdl.state
    err = np.linalg.norm(state.payload_position - target)
    bend = [
        float(np.degrees(np.arccos(np.clip(-np.asarray(q)[2], -1.0, 1.0))))
        for q in state.link_attitudes
    ]
    print(f"M5 quad + {NUM_LINKS}-link flexible cable: start {START}  ->  target {target}")
    print(f"  final payload  {state.payload_position}")
    print(f"  payload error  {err:.4f} m")
    print(f"  link tilts off vertical: {', '.join(f'{b:.1f}deg' for b in bend)}")
    print(f"  feedforward drops: {mdl.feedforward_drops}")
    assert err < 0.15, f"payload did not converge: {err:.4f} m"


if __name__ == "__main__":
    main()
