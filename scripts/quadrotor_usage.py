#!/usr/bin/env python3
import udaan as U
import numpy as np
import argparse


def usage(args):
    mdl = getattr(U.models, args.model).Quadrotor(render=args.render)
    mdl.simulate(tf=args.time, position=np.array([1.0, 1.0, 0.0]))  # initial position

    # TODO add controller option
    # TODO add trajectory option

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="mujoco",
        choices=["base", "mujoco"],
        help="Model to use.",
    )
    parser.add_argument("--time", "-t", type=float, default=10.0)
    parser.add_argument("--render", "-r", action="store_true")

    args = parser.parse_args()

    usage(args)
