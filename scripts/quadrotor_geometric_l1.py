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
def simulate(mdl, tf):
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

    # Create the Udaan model.
    mdl = U.models.mujoco.QuadrotorComparison(render=True)

    # System parameters
    mass = mdl.plant.mass
    inertia = mdl.plant.inertia

    # unmdoeled dynamics parameters
    added_mass = 0.5
    r = np.array([0.25, 0.25, -0.25])
    added_inertia = -added_mass * U.manif.hat(r) @ U.manif.hat(r)
    actual_mass = mass + added_mass
    actual_inertia = inertia + added_inertia

    mdl.set_mass(actual_mass, plant=True)
    mdl.set_inertia(actual_inertia, plant=True)

    # Reference trajectory for the quadrotors.
    setpoint = lambda t: (np.array([1., 1., 2.]), np.zeros(3), np.zeros(3))

    # Plant controllers
    mdl.plant.position_controller = U.control.QuadPosPD(mass=mass,
                                                        setpoint=setpoint)
    mdl.plant.attitude_controller = U.control.QuadAttGeoPD(inertia=inertia)

    # Reference controllers
    mdl.reference.position_controller = U.control.QuadPosPD(mass=mass,
                                                            setpoint=setpoint)
    mdl.reference.attitude_controller = U.control.QuadAttGeoPD(inertia=inertia)

    # simulate
    simulate(mdl, 100.0)
    return


if __name__ == "__main__":
    main()
