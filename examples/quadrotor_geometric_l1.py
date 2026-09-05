#!/usr/bin/env python3
"""L1 adaptive position control on a quadrotor, next to a plain PD baseline.

Implements:
Geometric $L_1$ Adaptive Attitude Control for a Quadrotor Unmanned Aerial Vehicle.
pdf: https://doi.org/10.1115/1.4045558

Both runs carry an unmodeled 0.25 kg payload: the plant is that much heavier
than the mass their position controllers were handed, so the loop has to make
up the missing hover thrust out of feedback alone and settles with a
steady-state sag.  The example prints the sag each controller is left with.

`PositionL1Controller` is EXPERIMENTAL and does not currently work: its
adaptation is frozen (see the class docstring), so what this example really
compares is two sets of non-adaptive gains — LQR against PD.  The L1 run is
expected to come out *slightly worse* than the baseline.  The example exists to
keep the controller exercised end to end and to keep that number visible, not
to demonstrate a win.

Needs the optional `l1` extra (`pip install 'udaan[l1]'`), which provides the
`control` package the L1 controller solves its LQR with.

Usage: python examples/quadrotor_geometric_l1.py
"""

import sys

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

    try:
        l1_err = fly(PositionL1Controller)
    except ModuleNotFoundError as err:
        if err.name != "control":
            raise
        print(
            "SKIP: this example needs the optional 'l1' extra, which provides the "
            "'control' package.\n      Install it with:  pip install 'udaan[l1]'",
            file=sys.stderr,
        )
        # 2 = "skipped, optional extra missing"; scripts/run_examples.sh reports SKIP.
        sys.exit(2)

    pd_err = fly(PositionPDController)

    print(f"Unmodeled mass {UNMODELED_MASS} kg, hover target {TARGET}")
    print(f"  PD position error {pd_err:.4f} m")
    print(f"  L1 position error {l1_err:.4f} m  (EXPERIMENTAL: adaptation is frozen)")
    # Both runs are only checked for divergence. There is deliberately no
    # "L1 beats PD" assertion: the adaptation in PositionL1Controller is frozen
    # by a known defect (see its class docstring), so L1 here is just an LQR
    # with different gains and lands a little *worse* than the baseline —
    # 0.3096 m against 0.3028 m at the time of writing. When the adaptation is
    # made to work and the gains are retuned, replace this with a real margin,
    # e.g. `assert l1_err < 0.75 * pd_err`.
    assert pd_err < 1.0 and l1_err < 1.0, "a run diverged"


if __name__ == "__main__":
    main()
