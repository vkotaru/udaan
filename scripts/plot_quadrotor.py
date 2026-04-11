"""Quadrotor simulation debugger.

Usage:
    python scripts/plot_quadrotor.py
    python scripts/plot_quadrotor.py --start 1,1,0 --target 0,0,1 --tf 10
"""

import argparse
from pathlib import Path

import numpy as np
from bokeh.io import output_file, show

from udaan.models.quadrotor import QuadrotorBase
from udaan.utils.plotting import plot_quadrotor_simulation
from udaan.utils.plotting.quadrotor import record_quadrotor_state

ARTIFACTS_DIR = Path(__file__).parent / ".." / "artifacts"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadrotor debug plots")
    parser.add_argument("--start", type=str, default="1,0,0", help="Start x,y,z")
    parser.add_argument("--target", type=str, default="0,0,1", help="Target x,y,z")
    parser.add_argument("--tf", type=float, default=10.0, help="Sim time")
    args = parser.parse_args()

    start = np.array([float(x) for x in args.start.split(",")])
    target = np.array([float(x) for x in args.target.split(",")])

    print(f"Simulating: {start} → {target} for {args.tf}s")

    mdl = QuadrotorBase()
    mdl._pos_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
    mdl.reset(position=start)

    history = record_quadrotor_state(mdl, args.tf, target)

    print(f"Final pos: {np.round(mdl.state.position, 4)}")
    print(f"Final err: {np.linalg.norm(mdl.state.position - target):.4f}m")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "quadrotor_debug.html"
    output_file(out_path)
    show(plot_quadrotor_simulation(history, target))
    print(f"Saved to {out_path}")
