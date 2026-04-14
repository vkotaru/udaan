"""Quadrotor with cable-suspended payload simulation debugger.

Usage:
    python scripts/plot_quadrotor_cspayload.py
    python scripts/plot_quadrotor_cspayload.py --start 1,1,0 --target 0,0,1 --tf 10
    python scripts/plot_quadrotor_cspayload.py --backend mujoco --cable-model links
"""

import argparse
from pathlib import Path

import numpy as np
from bokeh.io import output_file, show

from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadBase
from udaan.utils.plotting import plot_quadrotor_cspayload_simulation
from udaan.utils.plotting.quadrotor_cspayload import record_quadrotor_cspayload_state

ARTIFACTS_DIR = Path(__file__).parent / ".." / "artifacts"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadrotor cspayload debug plots")
    parser.add_argument("--start", type=str, default="1,0,0", help="Payload start x,y,z")
    parser.add_argument("--target", type=str, default="0,0,1", help="Payload target x,y,z")
    parser.add_argument("--tf", type=float, default=10.0, help="Sim time")
    parser.add_argument("--backend", choices=["base", "mujoco"], default="base")
    parser.add_argument(
        "--cable-model",
        choices=["tendon", "links", "cable"],
        default="links",
        help="MuJoCo cable model (mujoco backend only)",
    )
    args = parser.parse_args()

    start = np.array([float(x) for x in args.start.split(",")])
    target = np.array([float(x) for x in args.target.split(",")])

    print(f"Simulating ({args.backend}): payload {start} → {target} for {args.tf}s")

    if args.backend == "mujoco":
        from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadMujoco

        mdl = QuadrotorCsPayloadMujoco(render=False, cable_model=args.cable_model)
    else:
        mdl = QuadrotorCsPayloadBase()

    mdl._payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
    mdl.reset(payload_position=start)

    history = record_quadrotor_cspayload_state(mdl, args.tf, target)

    print(f"Final payload: {np.round(mdl.state.payload_position, 4)}")
    print(f"Final err:     {np.linalg.norm(mdl.state.payload_position - target):.4f}m")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.backend if args.backend == "base" else f"{args.backend}_{args.cable_model}"
    out_path = ARTIFACTS_DIR / f"quadrotor_cspayload_debug_{suffix}.html"
    output_file(out_path)
    show(plot_quadrotor_cspayload_simulation(history, target))
    print(f"Saved to {out_path}")
