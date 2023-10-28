#!/usr/bin/env python3
import udaan as U
import numpy as np
import argparse
import os


def usage(args):
    try:
        generator = getattr(U.utils.xml_model_generator, args.model)
    except AttributeError:
        print("Given model is not yet supported.")
        return
    generator(
        nQ=args.num_of_quads,
        filename=U.PATH + "/udaan/models/assets/mjcf/quadrotor_comparison.xml",
        verbose=args.verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="quadrotor",
        choices=[
            "quadrotor", "quadrotor_comparison", "quadrotor_payload",
            "multi_quad_pointmass", "multi_quad_rigidload"
        ],
        help="Model to use.",
        required=True,
    )
    parser.add_argument("--num_of_quads", "-nQ", type=int, default=2)
    parser.add_argument("--output_filename",
                        "-of",
                        type=str,
                        default='temp.xml')
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    usage(args)
