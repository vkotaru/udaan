#!/usr/bin/env python3
"""
Usage: ./quadrotor_geometric_l1.py

Implements:
Geometric $L_1$ Adaptive Attitude Control for a Quadrotor Unmanned Aerial Vehicle.
pdf: https://doi.org/10.1115/1.4045558
"""

import udaan as U
import numpy as np
import time


def compute_input(t, sys):
    state = sys.state
    F = sys.position_controller.compute(t, (state.position, state.velocity))
    f, M = sys.attitude_controller.compute(
        t, (state.orientation, state.angular_velocity), F)
    u = np.array([f, *M])
    return u


# Simulate
def run_simulation(mdl, tf):
    mdl.reset(position=np.array([0.0, 0.0, 0.2]))
    start_t = time.time_ns()
    while mdl.t < tf:
        t = mdl.t

        # plant control
        u_plant = compute_input(t, mdl.plant)

        # reference control
        u_reference = compute_input(t, mdl.reference)

        # integrate
        mdl.step(np.concatenate((u_plant, u_reference)))

        # add target marker
        mdl.add_reference_marker(mdl.plant.position_controller.pos_setpoint)

    end_t = time.time_ns()
    time_taken = (end_t - start_t) * 1e-9
    print("Took (%.4f)s for simulating (%.4f)s" % (time_taken, mdl.t))
    return


def main():
    # Generate mujoco .xml file with unmodeled weight.
    unmodeled_mass = 0.25
    unmodeled_mass_loc = np.array([0.2, 0.2, -0.05])
    U.utils.xml_model_generator.quadrotor_comparison(
        name="QuadrotorL1Comparison",
        filename=U.PATH +
        "/udaan/models/assets/mjcf/quadrotor_l1_comparison.xml",
        verbose=True,
        unmodeled_mass=unmodeled_mass,
        unmodeled_mass_loc=unmodeled_mass_loc,
    )

    # Create the Udaan model.
    mdl = U.models.mujoco.QuadrotorComparison(
        filename="quadrotor_l1_comparison.xml", render=True)

    # System parameters without the added mass
    mass = 0.99
    inertia = np.array([[0.01153467, 0., 0.], [0., 0.00652658, 0.],
                        [0., 0., 0.00532658]])

    # Reference trajectory for the quadrotors.
    ref_traj = lambda t: (np.array([1., 1., 2.]), np.zeros(3), np.zeros(3))

    # Plant controllers
    mdl.plant.position_controller = U.control.QuadrotorPositionL1Controller(mass=mass,
                                                        setpoint=ref_traj)
    mdl.plant.attitude_controller = U.control.QuadAttGeoPD(inertia=inertia)

    # Reference controllers
    mdl.reference.position_controller = U.control.QuadPosPD(mass=mass,
                                                            setpoint=ref_traj)
    mdl.reference.attitude_controller = U.control.QuadAttGeoPD(inertia=inertia)

    # simulate
    run_simulation(mdl, 100.0)
    return


if __name__ == "__main__":
    main()
