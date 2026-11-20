#!/usr/bin/env python3
"""M8 — three quadrotors in series on a 3 m hose pinned to a ground anchor.

The vehicles sit evenly on the line from the anchor to the top one.  The hose
LQR moves the whole formation to a higher, closer station by blending the flat
tensions over 6 s, with the flat map's feed-forward.  Analytical model,
headless.

Usage: python examples/tethered_hose.py
"""

import numpy as np

from udaan.flatness.tethered_hose import TetherBlend
from udaan.models.tethered_hose import TetheredHoseBase


def main():
    mdl = TetheredHoseBase()
    start = np.array([1.5, 0.0, 2.4])
    end = np.array([1.2, 0.0, 2.64])
    blend = TetherBlend(mdl.hose_controller.flat_map, start, end, duration=6.0, start_time=1.0)
    mdl.hose_controller.setpoint = blend.get

    mdl.simulate(tf=9.0)

    errors = [
        np.linalg.norm(p - q)
        for p, q in zip(mdl.state.positions, blend.recover(mdl.t)[0].positions, strict=True)
    ]
    print(
        f"M8: {mdl.num_quadrotors} quads at vertices {mdl.attachments} on {mdl.hose_length:.1f} m"
    )
    print(f"  top vehicle {np.asarray(mdl.state.positions[-1]).round(4)} (target {end})")
    print(f"  worst vehicle error {max(errors):.2e} m")
    assert max(errors) < 1e-3, "the formation did not reach its new station"


if __name__ == "__main__":
    main()
