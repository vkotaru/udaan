#!/usr/bin/env python3
"""M7 — one quadrotor tethered to a ground anchor by a 4-link cable.

The hose LQR holds the vehicle and its tether while the formation circles the
anchor: the circle starts from rest, ramps up over 6 s, and is planned through
the anchor-link force (the flat output), so the flat map supplies both the
reference state and the feed-forward wrench.  Analytical model, headless.

Usage: python examples/tethered_quad.py
"""

import numpy as np

from udaan.flatness.tethered_hose import TetherCircle, circle_rate_bound
from udaan.models.tethered_hose import TetheredQuadrotorBase


def main():
    mdl = TetheredQuadrotorBase()
    flat_map = mdl.hose_controller.flat_map
    top = np.array([1.0, 0.0, 1.6])  # 0.5 L out, 0.8 L up
    rate = circle_rate_bound(flat_map, top)
    circle = TetherCircle(flat_map, top, rate=rate, ramp_time=6.0)
    mdl.hose_controller.setpoint = circle.get

    mdl.simulate(tf=10.0)

    target = circle.get(mdl.t)[0]
    err = np.linalg.norm(np.asarray(mdl.state.positions[-1]) - target)
    print(
        f"M7 on a {mdl.hose_length:.1f} m, {mdl.num_links}-link tether, circling at {rate:.2f} rad/s"
    )
    print(f"  vehicle error {err:.5f} m after {mdl.t:.1f} s")
    print(f"  lowest reference vertex {mdl.reference_ground_clearance:.3f} m above the ground")
    assert err < 0.01, f"the vehicle left its circle: {err:.4f} m"


if __name__ == "__main__":
    main()
