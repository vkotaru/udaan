#!/usr/bin/env python3
"""M6 — several quadrotors carrying one flexible hose.

M5's chain with the airframe at its head replaced by an arbitrary attachment
pattern: vehicles are bolted to chosen vertices of a single ``n``-link hose, so
``n`` and the vehicle count are independent.  The hose LQR regulates the whole
chain against the static trim at the root setpoint, and emits a per-vehicle
wrench — hence ``input="wrench"``.  Analytical model, headless.

Usage: python examples/multi_quad_hose.py
"""

import numpy as np

from udaan.models.multi_quad_hose import MultiQuadHoseBase

NUM_LINKS = 4
STEP = np.array([0.3, 0.2, 0.2])
START = np.array([0.0, 0.0, 2.0])


def main():
    # The LQR needs a vehicle at the last vertex; the default attachment
    # pattern is both ends, which qualifies.
    mdl = MultiQuadHoseBase(num_links=NUM_LINKS, input="wrench")

    target = START + STEP
    mdl.hose_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))

    mdl.simulate(tf=10.0, root_position=START)

    state = mdl.state
    err = np.linalg.norm(np.asarray(state.root_position) - target)
    print(
        f"M6 {mdl.num_quadrotors} quads at vertices {mdl.attachments} "
        f"+ {NUM_LINKS}-link hose: start {START}  ->  target {target}"
    )
    print(f"  final root  {np.asarray(state.root_position)}")
    print(f"  root error  {err:.4f} m")
    print(f"  feedforward drops: {mdl.feedforward_drops}")
    assert err < 0.15, f"hose root did not converge: {err:.4f} m"


if __name__ == "__main__":
    main()
