#!/usr/bin/env python3
"""M3 — three quadrotors cooperatively carrying one point-mass payload.

Each vehicle hangs off its own taut cable.  The payload controller distributes
the required net force across the team, and every agent runs its own cable and
attitude loops underneath.  Analytical model, headless.

Usage: python examples/multi_quad_cspayload.py
"""

import numpy as np

from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadBase

NUM_QUADS = 3
TARGET = np.array([1.0, -0.5, 1.8])
START = np.array([0.0, 0.0, 1.0])


def main():
    mdl = MultiQuadCsPayloadBase(num_quadrotors=NUM_QUADS)
    mdl.payload_controller.setpoint = lambda t: (TARGET, np.zeros(3), np.zeros(3))

    mdl.simulate(tf=10.0, payload_position=START)

    state = mdl.state
    err = np.linalg.norm(state.payload_position - TARGET)
    print(f"M3 {NUM_QUADS} quads + point-mass payload: start {START}  ->  target {TARGET}")
    print(f"  final payload {state.payload_position}")
    print(f"  payload error {err:.4f} m")
    for i, pos in enumerate(state.positions):
        print(f"  quad {i} at {np.asarray(pos)}")
    assert err < 0.1, f"payload did not converge: {err:.4f} m"


if __name__ == "__main__":
    main()
