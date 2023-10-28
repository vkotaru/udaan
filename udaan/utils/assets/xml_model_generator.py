#!/usr/bin/python
from .mujoco_asset_creator import *
import math


def multi_quad_pointmass(nQ=2,
                         filename="./assets/multi_quad_pointmass.xml",
                         verbose=True):
    cable_length = 1.0
    z = 2.0
    r = 0.5
    dzL = np.sqrt(cable_length**2 - r**2)
    zL = z - dzL
    zL = 1.34
    dth = 2 * np.pi / nQ
    TH = np.arange(0, 2 * np.pi, dth)

    mjcWriter = MujocoAssetCreator("MultiQuadPointMass")
    for i in range(nQ):
        q1 = mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "quad" + str(i),
            np.array([r * np.cos(TH[i]), r * np.sin(TH[i]), z]),
        )

    pyld = mjcWriter.body(mjcWriter.worldbody,
                          "pyld",
                          pos=np.array([0.0, 0.0, zL]))
    mjcWriter.sphere(pyld,
                     "pyld",
                     radius=0.05,
                     rgba=[0.0, 1, 1, 1.0],
                     mass=0.15)
    mjcWriter.site(pyld,
                   "end2",
                   pos=np.array([0.0, 0.0, 0.0]),
                   type="sphere",
                   size=[0.01])
    mjcWriter.joint(pyld, "pyld_joint", type="free")

    tendon = mjcWriter.tendon(mjcWriter.root)
    for i in range(nQ):
        mjcWriter.spatial(tendon,
                          "quad" + str(i) + "_end1",
                          "end2",
                          range=[0.0, cable_length])

    mjcWriter.save_to(filename, verbose=verbose)
    return


def quadrotor_comparison(**kwargs):
    model_name = kwargs[
        'model_name'] if 'model_name' in kwargs else "QuadrotorComparison"
    if 'filename' in kwargs:
        filename = kwargs['filename']
    else:
        raise ValueError("filename not provided")
    verbosity = kwargs['verbose'] if 'verbose' in kwargs else False

    unmodeled_dynamics = False
    if 'unmodeled_mass' in kwargs:
        unmodeled_dynamics = True
        unmodeled_mass = kwargs['unmodeled_mass']
    if 'unmodeled_mass_loc' in kwargs:
        unmodeled_mass_loc = kwargs['unmodeled_mass_loc']
    else:
        unmodeled_mass_loc = np.array([0.0, 0.0, 0.0])

    mjcWriter = MujocoAssetCreator(model_name)
    if unmodeled_dynamics:
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "plant",
            np.array([0., 0, 0.4]),
            rgb=[1.0, 0., 0.],
            unmodeled_mass=unmodeled_mass,
            unmodeled_mass_loc=unmodeled_mass_loc,
        )
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "reference",
            np.array([0., 0, 0.4]),
            rgb=[1.0, 0., 0.4],
            alpha=0.25,
            unmodeled_mass=unmodeled_mass,
            unmodeled_mass_loc=unmodeled_mass_loc,
        )

    else:
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "plant",
            np.array([0., 0, 0.4]),
            rgb=[1.0, 0., 0.],
        )
        mjcWriter.create_quadrotor0(
            mjcWriter.worldbody,
            "reference",
            np.array([0., 0, 0.4]),
            rgb=[1.0, 0., 0.4],
            alpha=0.25,
        )
    mjcWriter.exclude_contact("plant", "reference")
    mjcWriter.save_to(filename=filename, verbose=verbosity)
    return


# TODO(vkotaru): Cleanup the following functions

# def create_quadcopter(verbose=True):
#     # mjcWriter = MujocoAssetCreator("Quadcopter")
#     # mjcWriter.create_isaacgym_quadcopter(mjcWriter.worldbody, "quadcopter", np.array([0., 0., 1.]))
#     # mjcWriter.save_to("./assets/quadcopter.xml", verbose=True)

#     mjcWriter = MujocoAssetCreator("Quadrotor")
#     mjcWriter.create_quadrotor(mjcWriter.worldbody, "quadrotor",
#                                np.array([0.0, 0.0, 1.0]))
#     mjcWriter.save_to("./assets/quadrotor.xml", verbose=verbose)
#     return

# def quadrotor_payload(verbose=True):
#     cable_length = 1.0
#     mjcWriter = MujocoAssetCreator("QuadrotorPayload")
#     quad = mjcWriter.create_quadrotor0(mjcWriter.worldbody, "quadrotor",
#                                        np.array([0.0, 0.0, 2.0]))
#     sitef = mjcWriter.site(quad,
#                            "thrust",
#                            pos=np.array([0.0, 0.0, 0.0]),
#                            rgba=[0.0, 1, 1, 1.0])
#     sitex = mjcWriter.site(
#         quad,
#         "rateX",
#         pos=np.array([0.0, 0.0, 0.0]),
#         size=np.array([0.06, 0.035, 0.025]),
#         rgba=[0.0, 1, 1, 1.0],
#     )
#     sitey = mjcWriter.site(
#         quad,
#         "rateY",
#         pos=np.array([0.0, 0.0, 0.0]),
#         size=np.array([0.06, 0.035, 0.025]),
#         rgba=[0.0, 1, 1, 1.0],
#     )
#     sitez = mjcWriter.site(
#         quad,
#         "rateZ",
#         pos=np.array([0.0, 0.0, 0.0]),
#         size=np.array([0.06, 0.035, 0.025]),
#         rgba=[0.0, 1, 1, 1.0],
#     )
#     mjcWriter.create_flexible_cable_payload(
#         quad,
#         "cable",
#         np.array([0.0, 0.0, -0.5 * cable_length]),
#         N=1,
#         length=cable_length,
#     )

#     actuator = mjcWriter.actuator(mjcWriter.root)
#     motorf = mjcWriter.motor(
#         actuator,
#         site="thrust",
#         range=[0.0, 30.0],
#         gear=np.array([0, 0.0, 1.0, 0.0, 0.0, 0.0]),
#     )
#     motorMx = mjcWriter.motor(
#         actuator,
#         site="rateX",
#         range=[-3.0, 3.0],
#         gear=np.array([0, 0.0, 0.0, 1.0, 0.0, 0.0]),
#     )
#     motorMy = mjcWriter.motor(
#         actuator,
#         site="rateY",
#         range=[-3.0, 3.0],
#         gear=np.array([0, 0.0, 0.0, 0.0, 1.0, 0.0]),
#     )
#     motorMz = mjcWriter.motor(
#         actuator,
#         site="rateZ",
#         range=[-3.0, 3.0],
#         gear=np.array([0, 0.0, 0.0, 0.0, 0.0, 1.0]),
#     )
#     mjcWriter.save_to("./assets/quadrotor_payload.xml", verbose=verbose)
#     return
