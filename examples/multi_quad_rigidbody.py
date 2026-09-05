#!/usr/bin/env python3
"""M4 — four quadrotors carrying one rigid-body payload.

Unlike M3's point mass the payload has an attitude, so the team can exert a
moment on it: the controller tracks a full SE(3) pose.  This example commands a
position step *and* a 0.4 rad yaw, then checks both halves converged.
Analytical model, headless.

Usage: python examples/multi_quad_rigidbody.py
"""

import numpy as np

from udaan.manif.utils import rodrigues_expm
from udaan.models.multi_quad_rigidbody import MultiQuadRigidbodyBase

NUM_QUADS = 4
TARGET = np.array([0.5, -0.3, 1.4])
START = np.array([0.0, 0.0, 1.0])
YAW = 0.4  # rad


def attitude_angle_deg(R, R_d):
    """Geodesic distance on SO(3), in degrees."""
    cos = (np.trace(np.asarray(R_d).T @ np.asarray(R)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def main():
    mdl = MultiQuadRigidbodyBase(num_quadrotors=NUM_QUADS)
    R_d = rodrigues_expm(YAW * np.array([0.0, 0.0, 1.0]))
    mdl.payload_controller.setpoint = lambda t: (TARGET, np.zeros(3), np.zeros(3))
    mdl.payload_controller.attitude_setpoint = lambda t: (R_d, np.zeros(3), np.zeros(3))

    mdl.simulate(tf=15.0, payload_position=START)

    state = mdl.state
    err = np.linalg.norm(state.payload_position - TARGET)
    ang = attitude_angle_deg(state.payload_orientation, R_d)
    print(f"M4 {NUM_QUADS} quads + rigid-body payload: start {START}  ->  target {TARGET}")
    print(f"  final payload  {state.payload_position}")
    print(f"  position error {err:.4f} m")
    print(f"  attitude error {ang:.3f} deg  (commanded yaw {np.degrees(YAW):.1f} deg)")
    assert err < 0.1, f"payload position did not converge: {err:.4f} m"
    assert ang < 2.0, f"payload attitude did not converge: {ang:.3f} deg"


if __name__ == "__main__":
    main()
